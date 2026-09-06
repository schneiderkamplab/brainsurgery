"""
T3: Mixed-precision export with sharding (OLMo-1B-0724-hf)

Cast the 112 attention/mlp projection matrices to bfloat16, keep everything
else (embeddings, lm_head) in float32, and write out a re-sharded safetensors
checkpoint with an index file, respecting a 256 MiB per-shard tensor-data
budget (oversized tensors get their own shard).
"""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INPUT_DIR = "inputs/base"
OUTPUT_DIR = "out/T3"
SHARD_BUDGET_BYTES = 256 * 1024 * 1024

# Exactly the projection matrices we intend to downcast.
PROJ_PATTERN = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def load_index(input_dir: str) -> dict:
    with open(os.path.join(input_dir, "model.safetensors.index.json")) as f:
        return json.load(f)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    index = load_index(INPUT_DIR)
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))

    # Load every tensor into memory, casting the intended matrices.
    tensors: dict[str, torch.Tensor] = {}
    for shard_file in shard_files:
        path = os.path.join(INPUT_DIR, shard_file)
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if PROJ_PATTERN.match(name):
                    t = t.to(torch.bfloat16)
                else:
                    t = t.to(torch.float32)
                tensors[name] = t.contiguous()

    # --- Required checks: fail loudly before writing anything. ---
    bf16_names = [n for n, t in tensors.items() if t.dtype == torch.bfloat16]
    if len(bf16_names) != 112:
        raise AssertionError(f"expected 112 bfloat16 tensors, got {len(bf16_names)}")

    if tensors["model.layers.0.self_attn.q_proj.weight"].dtype != torch.bfloat16:
        raise AssertionError("model.layers.0.self_attn.q_proj.weight is not bfloat16")

    if tensors["model.embed_tokens.weight"].dtype != torch.float32:
        raise AssertionError("model.embed_tokens.weight is not float32")

    if len(tensors) != 114:
        raise AssertionError(f"expected 114 tensors total, got {len(tensors)}")

    for name in tensors:
        if not PROJ_PATTERN.match(name) and tensors[name].dtype != torch.float32:
            raise AssertionError(f"{name} should be float32 but is {tensors[name].dtype}")

    # --- Build shards: greedy bin-packing respecting the per-shard budget. ---
    # Deterministic order (stable across the two input shards) for reproducible output.
    names_in_order = sorted(tensors.keys(), key=lambda n: (weight_map[n], n))

    def tensor_bytes(name: str) -> int:
        t = tensors[name]
        return t.numel() * dtype_size(t.dtype)

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names_in_order:
        size = tensor_bytes(name)
        if size > SHARD_BUDGET_BYTES:
            # Oversized tensor gets its own shard.
            if current:
                shards.append(current)
                current = []
                current_bytes = 0
            shards.append([name])
            continue
        if current and current_bytes + size > SHARD_BUDGET_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
    if current:
        shards.append(current)

    n_shards = len(shards)
    weight_map_out: dict[str, str] = {}
    total_size = 0
    for i, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {name: tensors[name] for name in shard_names}
        save_file(shard_tensors, os.path.join(OUTPUT_DIR, shard_filename))
        for name in shard_names:
            weight_map_out[name] = shard_filename
            total_size += tensor_bytes(name)

    out_index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map_out,
    }
    with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(out_index, f, indent=2, sort_keys=True)

    print(f"Wrote {len(tensors)} tensors across {n_shards} shards to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
