"""
T3: Mixed-precision export with sharding (GPT-2 124M), condition F.

Plain script on top of `torch` + `safetensors` (both on the F-allowed list).
No merge/adapter tooling is needed here: this is a per-tensor dtype cast and
buffer drop, which mergekit/peft/torch-state-bridge do not model any better
than a direct safetensors read/write, so a script is the most direct route.

Steps:
  1. Load every tensor from inputs/base/model.safetensors.
  2. Cast exactly the 48 projection matrices (attn.c_attn/c_proj,
     mlp.c_fc/c_proj weights, 4 per layer x 12 layers) to bfloat16.
  3. Drop the 12 `h.<i>.attn.bias` causal-mask buffers (not parameters).
  4. Leave every other tensor as float32, values unchanged.
  5. Run the required checks; fail loudly (AssertionError) if any fails.
  6. Greedily bin-pack tensors into shards of at most 64 MiB of tensor data
     (a single oversized tensor, wte.weight, gets its own shard), and write
     model.safetensors.index.json alongside the shard files.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INPUT = Path("inputs/base/model.safetensors")
OUT_DIR = Path("out/T3")
SHARD_BUDGET_BYTES = 64 * 1024 * 1024  # 64 MiB, tensor data only

N_LAYERS = 12
PROJECTIONS = [
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
]
CAST_TO_BF16 = {f"h.{i}.{p}" for i in range(N_LAYERS) for p in PROJECTIONS}
DROP_BUFFERS = {f"h.{i}.attn.bias" for i in range(N_LAYERS)}

assert len(CAST_TO_BF16) == 48
assert len(DROP_BUFFERS) == 12


def dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def load_tensors() -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(INPUT), framework="pt") as f:
        for key in f.keys():
            if key in DROP_BUFFERS:
                continue
            tensor = f.get_tensor(key)
            if key in CAST_TO_BF16:
                tensor = tensor.to(torch.bfloat16)
            else:
                assert tensor.dtype == torch.float32, f"{key} unexpectedly not float32"
            tensors[key] = tensor.contiguous()
    return tensors


def run_checks(tensors: dict[str, torch.Tensor]) -> None:
    bf16_keys = [k for k, t in tensors.items() if t.dtype == torch.bfloat16]
    assert len(bf16_keys) == 48, f"expected 48 bfloat16 tensors, got {len(bf16_keys)}"
    assert set(bf16_keys) == CAST_TO_BF16, "bfloat16 tensor set does not match intended targets"
    assert tensors["h.0.attn.c_attn.weight"].dtype == torch.bfloat16
    assert tensors["wte.weight"].dtype == torch.float32
    assert len(tensors) == 148, f"expected 148 tensors, got {len(tensors)}"
    for name in DROP_BUFFERS:
        assert name not in tensors, f"buffer {name} should have been dropped"


def plan_shards(tensors: dict[str, torch.Tensor]) -> list[list[str]]:
    """Greedy bin-packing in original key order; an oversized tensor gets its
    own shard rather than being split or forced to overflow a shard."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for key, tensor in tensors.items():
        size = tensor.numel() * dtype_size(tensor.dtype)
        if size > SHARD_BUDGET_BYTES:
            if current:
                shards.append(current)
                current, current_size = [], 0
            shards.append([key])
            continue
        if current and current_size + size > SHARD_BUDGET_BYTES:
            shards.append(current)
            current, current_size = [], 0
        current.append(key)
        current_size += size
    if current:
        shards.append(current)
    return shards


def write_output(tensors: dict[str, torch.Tensor], shards: list[list[str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for idx, keys in enumerate(shards, start=1):
        shard_name = f"model-{idx:05d}-of-{n:05d}.safetensors"
        shard_tensors = {k: tensors[k] for k in keys}
        save_file(shard_tensors, str(OUT_DIR / shard_name))
        for k, t in shard_tensors.items():
            weight_map[k] = shard_name
            total_size += t.numel() * dtype_size(t.dtype)

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    tensors = load_tensors()
    run_checks(tensors)
    shards = plan_shards(tensors)
    for keys in shards:
        size = sum(tensors[k].numel() * dtype_size(tensors[k].dtype) for k in keys)
        assert size <= SHARD_BUDGET_BYTES or len(keys) == 1, "shard exceeds budget"
    write_output(tensors, shards)
    print(f"wrote {len(tensors)} tensors in {len(shards)} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
