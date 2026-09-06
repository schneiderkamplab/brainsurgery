"""
T3: Mixed-precision export with sharding (OLMo-1B-0724-hf).

Casts the 112 large projection matrices (q/k/v/o_proj, gate/up/down_proj,
per layer) to bfloat16, keeps everything else (embeddings, lm_head) in
float32, and writes a sharded safetensors checkpoint with an index file
under a 256 MiB per-shard tensor-data budget.
"""

import json
import re
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INPUT_DIR = "inputs/base"
OUTPUT_DIR = "out/T3"
SHARD_BUDGET_BYTES = 256 * 1024 * 1024  # 256 MiB, tensor data only

# Exactly the 112 projection matrices: 7 per layer, 16 layers.
BF16_PATTERN = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def main() -> None:
    index_path = os.path.join(INPUT_DIR, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]  # tensor name -> shard filename (input)
    names = list(weight_map.keys())

    if len(names) != 114:
        raise AssertionError(f"expected 114 input tensors, found {len(names)}")

    # Load every tensor from its input shard file.
    tensors: dict[str, torch.Tensor] = {}
    shard_files = sorted(set(weight_map.values()))
    for shard_file in shard_files:
        path = os.path.join(INPUT_DIR, shard_file)
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    if set(tensors.keys()) != set(names):
        raise AssertionError("tensor keys loaded do not match index weight_map keys")

    # Cast: bf16 for exactly the 112 projection matrices, float32 elsewhere.
    bf16_names = set()
    for name, tensor in tensors.items():
        if tensor.dtype != torch.float32:
            raise AssertionError(f"expected input tensor {name} to be float32, got {tensor.dtype}")
        if BF16_PATTERN.match(name):
            tensors[name] = tensor.to(torch.bfloat16)
            bf16_names.add(name)
        # else: leave as float32, unchanged

    # --- Required checks (fail loudly before writing) ---
    n_bf16 = sum(1 for t in tensors.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 112:
        raise AssertionError(f"expected exactly 112 bfloat16 tensors, found {n_bf16}")

    if tensors["model.layers.0.self_attn.q_proj.weight"].dtype != torch.bfloat16:
        raise AssertionError("model.layers.0.self_attn.q_proj.weight must be bfloat16")

    if tensors["model.embed_tokens.weight"].dtype != torch.float32:
        raise AssertionError("model.embed_tokens.weight must be float32")

    if len(tensors) != 114:
        raise AssertionError(f"expected exactly 114 output tensors, found {len(tensors)}")

    # Sanity: exactly the intended set is bf16, nothing else.
    expected_bf16 = {n for n in names if BF16_PATTERN.match(n)}
    if bf16_names != expected_bf16:
        raise AssertionError("bfloat16 tensor set does not match the intended projection matrices")

    # --- Shard packing: greedy bin-packing in index order, 256 MiB budget ---
    # A tensor larger than the budget on its own gets its own shard.
    groups: list[list[str]] = []
    current: list[str] = []
    current_size = 0

    def tensor_bytes(name: str) -> int:
        t = tensors[name]
        return t.numel() * dtype_size(t.dtype)

    for name in names:
        size = tensor_bytes(name)
        if size > SHARD_BUDGET_BYTES:
            if current:
                groups.append(current)
                current = []
                current_size = 0
            groups.append([name])
            continue
        if current and current_size + size > SHARD_BUDGET_BYTES:
            groups.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += size
    if current:
        groups.append(current)

    num_shards = len(groups)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    weight_map_out = {}
    total_size = 0
    for i, group in enumerate(groups, start=1):
        shard_name = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {name: tensors[name].contiguous() for name in group}
        save_file(shard_tensors, os.path.join(OUTPUT_DIR, shard_name))
        for name in group:
            weight_map_out[name] = shard_name
            total_size += tensor_bytes(name)

    index_out = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map_out,
    }
    with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index_out, f, indent=2, sort_keys=True)

    print(f"Wrote {len(tensors)} tensors across {num_shards} shard(s) to {OUTPUT_DIR}")
    print(f"bfloat16 tensors: {n_bf16}, float32 tensors: {len(tensors) - n_bf16}")


if __name__ == "__main__":
    main()
