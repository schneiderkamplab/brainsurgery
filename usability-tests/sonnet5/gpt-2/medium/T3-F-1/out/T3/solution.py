"""
T3: mixed-precision export with sharding for GPT-2 (124M).

Plain script on top of `safetensors` + `torch` (both in F-allowed.md). No
merge/adapter tooling is a good fit here: this is a per-tensor dtype cast and
buffer drop, not a merge or a LoRA operation, so a direct script is the
smallest correct route.

- Cast the 48 projection weight matrices (attn.c_attn/c_proj.weight,
  mlp.c_fc/c_proj.weight, per layer) to bfloat16.
- Keep every other tensor float32, values unchanged.
- Drop the 12 `h.<i>.attn.bias` causal-mask buffers (not parameters).
- Shard the result so no shard holds more than 64 MiB of tensor data,
  writing `model.safetensors.index.json` with the weight map.
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_PATH = Path("inputs/base/model.safetensors")
OUT_DIR = Path("out/T3")
SHARD_LIMIT_BYTES = 64 * 1024 * 1024  # 64 MiB, tensor data only

NUM_LAYERS = 12
PROJECTION_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)
BF16_NAMES = {f"h.{i}.{suf}" for i in range(NUM_LAYERS) for suf in PROJECTION_SUFFIXES}
BUFFER_NAMES = {f"h.{i}.attn.bias" for i in range(NUM_LAYERS)}

assert len(BF16_NAMES) == 48
assert len(BUFFER_NAMES) == 12


def dtype_size(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(IN_PATH), framework="pt") as f:
        for name in f.keys():
            if name in BUFFER_NAMES:
                continue  # drop non-parameter buffers
            t = f.get_tensor(name).clone().contiguous()
            if name in BF16_NAMES:
                t = t.to(torch.bfloat16)
            else:
                assert t.dtype == torch.float32, f"unexpected dtype for {name}: {t.dtype}"
            tensors[name] = t

    # --- required checks: fail loudly before writing anything ---
    n_bf16 = sum(1 for t in tensors.values() if t.dtype == torch.bfloat16)
    assert n_bf16 == 48, f"expected exactly 48 bfloat16 tensors, got {n_bf16}"
    assert tensors["h.0.attn.c_attn.weight"].dtype == torch.bfloat16, (
        "h.0.attn.c_attn.weight must be bfloat16"
    )
    assert tensors["wte.weight"].dtype == torch.float32, "wte.weight must be float32"
    assert len(tensors) == 148, f"expected exactly 148 tensors, got {len(tensors)}"
    for name in BF16_NAMES:
        assert re.fullmatch(r"h\.\d+\.(attn|mlp)\.c_\w+\.weight", name)
        assert tensors[name].dtype == torch.bfloat16
    for name, t in tensors.items():
        if name not in BF16_NAMES:
            assert t.dtype == torch.float32, f"{name} should stay float32, got {t.dtype}"
    for name in BUFFER_NAMES:
        assert name not in tensors, f"buffer {name} should have been dropped"

    # --- shard: sequential greedy bin-packing, <=64 MiB tensor data per shard ---
    names_sorted = sorted(tensors.keys())
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name in names_sorted:
        size = dtype_size(tensors[name])
        if current and current_size + size > SHARD_LIMIT_BYTES:
            shards.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += size
        if current_size > SHARD_LIMIT_BYTES:
            # a single oversized tensor: keep it alone in its own shard
            shards.append(current)
            current = []
            current_size = 0
    if current:
        shards.append(current)

    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for idx, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {name: tensors[name] for name in shard_names}
        save_file(shard_tensors, str(OUT_DIR / shard_filename))
        for name in shard_names:
            weight_map[name] = shard_filename
            total_size += dtype_size(tensors[name])

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"wrote {len(tensors)} tensors across {n_shards} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
