"""
T3: Mixed-precision export with sharding (GPT-2 124M).

Cast the large projection matrices to bfloat16, keep everything else
(embeddings, norms, biases) in float32, drop the non-parameter causal-mask
buffers, and write a sharded safetensors checkpoint with an index file.
"""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INPUT_PATH = "inputs/base/model.safetensors"
OUT_DIR = "out/T3"
MAX_SHARD_BYTES = 64 * 1024 * 1024  # 64 MiB of tensor data per shard
NUM_LAYERS = 12

# The 4 projection matrices per layer that must become bfloat16.
PROJECTION_SUFFIXES = [
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
]
BF16_KEYS = {f"h.{i}.{suffix}" for i in range(NUM_LAYERS) for suffix in PROJECTION_SUFFIXES}

# The non-parameter causal-mask buffers to drop.
DROP_KEYS = {f"h.{i}.attn.bias" for i in range(NUM_LAYERS)}

assert len(BF16_KEYS) == 48
assert len(DROP_KEYS) == 12


def load_tensors(path):
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def build_output(tensors):
    output = {}
    for key, tensor in tensors.items():
        if key in DROP_KEYS:
            continue
        if key in BF16_KEYS:
            output[key] = tensor.to(torch.bfloat16).contiguous()
        else:
            # Must stay float32 with unchanged values.
            output[key] = tensor.to(torch.float32).contiguous()
    return output


def run_required_checks(input_tensors, output_tensors):
    missing_bf16 = BF16_KEYS - set(output_tensors)
    if missing_bf16:
        raise AssertionError(f"expected bf16 keys missing from output: {sorted(missing_bf16)}")

    bf16_count = sum(1 for t in output_tensors.values() if t.dtype == torch.bfloat16)
    if bf16_count != 48:
        raise AssertionError(f"expected exactly 48 bfloat16 tensors, got {bf16_count}")

    if output_tensors["h.0.attn.c_attn.weight"].dtype != torch.bfloat16:
        raise AssertionError("h.0.attn.c_attn.weight must be bfloat16")

    if output_tensors["wte.weight"].dtype != torch.float32:
        raise AssertionError("wte.weight must be float32")

    if len(output_tensors) != 148:
        raise AssertionError(f"expected exactly 148 output tensors, got {len(output_tensors)}")

    for key in DROP_KEYS:
        if key in output_tensors:
            raise AssertionError(f"buffer {key} should have been dropped")

    if len(input_tensors) != 160:
        raise AssertionError(f"expected 160 input tensors, got {len(input_tensors)}")


def tensor_nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def plan_shards(output_tensors):
    """Greedy bin-packing in insertion order; a single oversized tensor gets
    its own shard."""
    shards = []  # list of list[str] (keys)
    current = []
    current_size = 0

    for key, tensor in output_tensors.items():
        size = tensor_nbytes(tensor)
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current = []
                current_size = 0
            shards.append([key])
            continue
        if current and current_size + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = []
            current_size = 0
        current.append(key)
        current_size += size

    if current:
        shards.append(current)

    return shards


def write_shards(output_tensors, shards):
    os.makedirs(OUT_DIR, exist_ok=True)
    num_shards = len(shards)
    weight_map = {}
    total_size = 0

    for idx, keys in enumerate(shards, start=1):
        filename = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {key: output_tensors[key] for key in keys}
        save_file(shard_tensors, os.path.join(OUT_DIR, filename), metadata={"format": "pt"})
        for key, tensor in shard_tensors.items():
            weight_map[key] = filename
            total_size += tensor_nbytes(tensor)

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")

    return num_shards, weight_map


def verify_shard_budget(output_tensors, shards):
    for keys in shards:
        size = sum(tensor_nbytes(output_tensors[k]) for k in keys)
        if len(keys) > 1 and size > MAX_SHARD_BYTES:
            raise AssertionError(f"shard exceeds {MAX_SHARD_BYTES} bytes: {size} for keys {keys}")


def main():
    input_tensors = load_tensors(INPUT_PATH)
    output_tensors = build_output(input_tensors)
    run_required_checks(input_tensors, output_tensors)

    shards = plan_shards(output_tensors)
    verify_shard_budget(output_tensors, shards)

    num_shards, weight_map = write_shards(output_tensors, shards)

    if set(weight_map) != set(output_tensors):
        raise AssertionError("weight_map key set does not match output tensor key set")

    print(f"Wrote {len(output_tensors)} tensors across {num_shards} shard(s) to {OUT_DIR}/")


if __name__ == "__main__":
    main()
