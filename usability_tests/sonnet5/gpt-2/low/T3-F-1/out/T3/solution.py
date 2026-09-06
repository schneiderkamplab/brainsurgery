"""
T3: Mixed-precision export with sharding (GPT-2, 124M).

Cast the 48 large projection matrices to bfloat16, keep everything else
(embeddings, layer norms, biases) as float32, drop the 12 non-parameter
causal-mask buffers, and write a sharded safetensors checkpoint with an
index file, shards capped at 64 MiB of tensor data each.

Uses only `torch` and `safetensors` (see F-allowed.md).
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # sandbox root: out/T3 -> out -> root
IN_PATH = ROOT / "inputs" / "base" / "model.safetensors"
OUT_DIR = ROOT / "out" / "T3"

SHARD_LIMIT_BYTES = 64 * 1024 * 1024  # 64 MiB, tensor data only

NUM_LAYERS = 12

# Exact set of 48 projection-matrix names to cast to bfloat16.
BF16_NAMES = set()
for i in range(NUM_LAYERS):
    BF16_NAMES.add(f"h.{i}.attn.c_attn.weight")
    BF16_NAMES.add(f"h.{i}.attn.c_proj.weight")
    BF16_NAMES.add(f"h.{i}.mlp.c_fc.weight")
    BF16_NAMES.add(f"h.{i}.mlp.c_proj.weight")

# Exact set of 12 non-parameter buffers to drop.
BUFFER_RE = re.compile(r"^h\.(\d+)\.attn\.bias$")
DROP_NAMES = {f"h.{i}.attn.bias" for i in range(NUM_LAYERS)}


def main():
    state_dict = load_file(str(IN_PATH))

    assert len(state_dict) == 160, f"expected 160 input tensors, got {len(state_dict)}"

    # Sanity: every name we intend to touch actually exists.
    missing_bf16 = BF16_NAMES - state_dict.keys()
    assert not missing_bf16, f"missing expected projection tensors: {missing_bf16}"
    missing_drop = DROP_NAMES - state_dict.keys()
    assert not missing_drop, f"missing expected buffer tensors: {missing_drop}"

    out = {}
    for name, tensor in state_dict.items():
        if name in DROP_NAMES:
            continue
        if name in BF16_NAMES:
            out[name] = tensor.to(torch.bfloat16).contiguous()
        else:
            # Everything else stays float32, values unchanged.
            assert tensor.dtype == torch.float32, f"unexpected input dtype for {name}: {tensor.dtype}"
            out[name] = tensor.contiguous()

    # --- Required checks: fail loudly before writing anything. ---
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    assert n_bf16 == 48, f"expected exactly 48 bfloat16 tensors, got {n_bf16}"
    assert out["h.0.attn.c_attn.weight"].dtype == torch.bfloat16, "h.0.attn.c_attn.weight must be bfloat16"
    assert out["wte.weight"].dtype == torch.float32, "wte.weight must be float32"
    assert len(out) == 148, f"expected exactly 148 output tensors, got {len(out)}"
    for name in DROP_NAMES:
        assert name not in out, f"buffer {name} should have been dropped"

    # --- Shard into <=64MiB (tensor data) groups, oversized tensors alone. ---
    # Preserve deterministic order (insertion order == input order minus drops).
    items = list(out.items())

    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards = []  # list of dict(name -> tensor)
    current = {}
    current_size = 0
    for name, tensor in items:
        size = tensor_nbytes(tensor)
        if size > SHARD_LIMIT_BYTES:
            # oversized tensor: gets its own shard
            if current:
                shards.append(current)
                current = {}
                current_size = 0
            shards.append({name: tensor})
            continue
        if current and current_size + size > SHARD_LIMIT_BYTES:
            shards.append(current)
            current = {}
            current_size = 0
        current[name] = tensor
        current_size += size
    if current:
        shards.append(current)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_shards = len(shards)
    width = max(5, len(str(n_shards)))
    weight_map = {}
    total_size = 0
    for idx, shard in enumerate(shards, start=1):
        shard_filename = f"model-{idx:0{width}d}-of-{n_shards:0{width}d}.safetensors"
        save_file(shard, str(OUT_DIR / shard_filename), metadata={"format": "pt"})
        for name, tensor in shard.items():
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(tensor)

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"Wrote {len(out)} tensors across {n_shards} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
