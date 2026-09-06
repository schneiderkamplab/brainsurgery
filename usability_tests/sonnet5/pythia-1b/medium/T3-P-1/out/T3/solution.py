"""
T3: Mixed-precision export with sharding (Pythia-1B).

Reads inputs/base/model.safetensors, casts the 64 large projection matrices
to bfloat16, upcasts everything else to float32, drops the 48 non-parameter
buffers, and writes a sharded safetensors checkpoint (<=256 MiB of tensor
data per shard, except oversized single tensors which get their own shard)
plus a model.safetensors.index.json.
"""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INPUT_PATH = "inputs/base/model.safetensors"
OUTPUT_DIR = "out/T3"
SHARD_LIMIT_BYTES = 256 * 1024 * 1024  # 268,435,456 bytes

NUM_LAYERS = 16

# Exactly the projection-matrix names that must become bfloat16.
PROJECTION_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.(attention\.(query_key_value|dense)|mlp\.(dense_h_to_4h|dense_4h_to_h))\.weight$"
)

# The exact buffer names to delete (48 total: 3 per layer * 16 layers).
BUFFER_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(INPUT_PATH, framework="pt") as f:
        for key in f.keys():
            if BUFFER_RE.match(key):
                continue
            t = f.get_tensor(key)
            if PROJECTION_RE.match(key):
                tensors[key] = t.to(torch.bfloat16)
            else:
                tensors[key] = t.to(torch.float32)

    # ---- Required checks: fail loudly before writing anything ----
    bf16_keys = [k for k, t in tensors.items() if t.dtype == torch.bfloat16]
    if len(bf16_keys) != 64:
        raise AssertionError(f"expected exactly 64 bfloat16 tensors, got {len(bf16_keys)}")

    expected_bf16 = set()
    for i in range(NUM_LAYERS):
        expected_bf16.add(f"gpt_neox.layers.{i}.attention.query_key_value.weight")
        expected_bf16.add(f"gpt_neox.layers.{i}.attention.dense.weight")
        expected_bf16.add(f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight")
        expected_bf16.add(f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight")
    if set(bf16_keys) != expected_bf16:
        missing = expected_bf16 - set(bf16_keys)
        extra = set(bf16_keys) - expected_bf16
        raise AssertionError(f"bfloat16 tensor set mismatch: missing={missing} extra={extra}")

    if tensors["gpt_neox.layers.0.attention.query_key_value.weight"].dtype != torch.bfloat16:
        raise AssertionError("gpt_neox.layers.0.attention.query_key_value.weight is not bfloat16")

    if tensors["gpt_neox.embed_in.weight"].dtype != torch.float32:
        raise AssertionError("gpt_neox.embed_in.weight is not float32")

    if len(tensors) != 196:
        raise AssertionError(f"expected exactly 196 tensors, got {len(tensors)}")

    # No parameter (non-buffer) tensor should have been dropped: every key
    # that is not a matched buffer must be present.
    with safe_open(INPUT_PATH, framework="pt") as f:
        original_keys = set(f.keys())
    dropped = original_keys - set(tensors.keys())
    expected_dropped = {
        name
        for name in original_keys
        if BUFFER_RE.match(name)
    }
    if dropped != expected_dropped:
        raise AssertionError(
            f"dropped tensor set mismatch: dropped={dropped} expected={expected_dropped}"
        )
    if len(expected_dropped) != 48:
        raise AssertionError(f"expected exactly 48 buffers to drop, got {len(expected_dropped)}")

    # ---- Shard assignment ----
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # Preserve a stable order (as encountered) for reproducibility.
    ordered_keys = list(tensors.keys())

    shards: list[list[str]] = []
    current_shard: list[str] = []
    current_size = 0

    for key in ordered_keys:
        size = tensor_nbytes(tensors[key])
        if size > SHARD_LIMIT_BYTES:
            # Oversized tensor: gets its own shard.
            if current_shard:
                shards.append(current_shard)
                current_shard = []
                current_size = 0
            shards.append([key])
            continue
        if current_size + size > SHARD_LIMIT_BYTES and current_shard:
            shards.append(current_shard)
            current_shard = []
            current_size = 0
        current_shard.append(key)
        current_size += size

    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    weight_map: dict[str, str] = {}
    shard_filenames = [
        f"model-{i + 1:05d}-of-{num_shards:05d}.safetensors" for i in range(num_shards)
    ]

    for shard_idx, keys in enumerate(shards):
        shard_filename = shard_filenames[shard_idx]
        shard_tensors = {k: tensors[k].contiguous() for k in keys}
        save_file(shard_tensors, os.path.join(OUTPUT_DIR, shard_filename))
        for k in keys:
            weight_map[k] = shard_filename

    total_size = sum(tensor_nbytes(t) for t in tensors.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"Wrote {len(tensors)} tensors across {num_shards} shards to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
