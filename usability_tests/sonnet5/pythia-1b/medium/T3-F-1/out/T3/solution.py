"""T3: mixed-precision export with sharding for Pythia-1B.

Plain script on top of `torch` + `safetensors` (both in F-allowed.md). No
mergekit/transformers dtype-export path gives per-tensor dtype control plus a
custom shard budget, so a direct script is the straightforward route here.

- Cast the 64 large projection matrices (attention.query_key_value.weight,
  attention.dense.weight, mlp.dense_h_to_4h.weight, mlp.dense_4h_to_h.weight,
  one set per of the 16 layers) to bfloat16.
- Upcast every other tensor to float32.
- Drop the 48 non-parameter buffers (attention.bias, attention.masked_bias,
  attention.rotary_emb.inv_freq per layer).
- Shard the result greedily into <=256 MiB (tensor-data-only) shards, each
  oversized tensor alone in its own shard, and write a HF-style index.
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("inputs/base/model.safetensors")
OUT_DIR = Path("out/T3")
SHARD_BUDGET = 256 * 1024 * 1024  # 268,435,456 bytes, tensor data only

BF16_PATTERN = re.compile(
    r"^gpt_neox\.layers\.\d+\.(attention\.query_key_value\.weight"
    r"|attention\.dense\.weight"
    r"|mlp\.dense_h_to_4h\.weight"
    r"|mlp\.dense_4h_to_h\.weight)$"
)
BUFFER_PATTERN = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)


def main() -> None:
    with safe_open(SRC, framework="pt") as f:
        keys = list(f.keys())
        tensors: dict[str, torch.Tensor] = {}
        for key in keys:
            if BUFFER_PATTERN.match(key):
                continue
            t = f.get_tensor(key)
            if BF16_PATTERN.match(key):
                tensors[key] = t.to(torch.bfloat16).contiguous()
            else:
                tensors[key] = t.to(torch.float32).contiguous()

    # --- required checks: fail loudly before writing anything ---
    bf16_keys = [k for k, t in tensors.items() if t.dtype == torch.bfloat16]
    assert len(bf16_keys) == 64, f"expected 64 bf16 tensors, got {len(bf16_keys)}"
    assert (
        tensors["gpt_neox.layers.0.attention.query_key_value.weight"].dtype
        == torch.bfloat16
    ), "layer 0 query_key_value.weight must be bfloat16"
    assert (
        tensors["gpt_neox.embed_in.weight"].dtype == torch.float32
    ), "embed_in.weight must be float32"
    assert len(tensors) == 196, f"expected 196 tensors, got {len(tensors)}"
    for key in tensors:
        assert not BUFFER_PATTERN.match(key), f"buffer {key} leaked into output"
    assert set(tensors) == {
        k for k in keys if not BUFFER_PATTERN.match(k)
    }, "tensor name set changed"

    # --- greedy sharding, oversized tensors get their own shard ---
    def nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for key, t in tensors.items():
        size = nbytes(t)
        if size > SHARD_BUDGET:
            if current:
                shards.append(current)
                current, current_size = [], 0
            shards.append([key])
            continue
        if current and current_size + size > SHARD_BUDGET:
            shards.append(current)
            current, current_size = [], 0
        current.append(key)
        current_size += size
    if current:
        shards.append(current)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, shard_keys in enumerate(shards, start=1):
        shard_name = f"model-{i:05d}-of-{n:05d}.safetensors"
        shard_tensors = {k: tensors[k] for k in shard_keys}
        save_file(shard_tensors, OUT_DIR / shard_name)
        for k, t in shard_tensors.items():
            weight_map[k] = shard_name
            total_size += nbytes(t)

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    print(f"wrote {len(tensors)} tensors across {n} shards, total_size={total_size}")


if __name__ == "__main__":
    main()
