"""
T3: Mixed-precision export with sharding (GPT-2 124M).

Cast the 48 large projection matrices to bfloat16, keep everything else
(embeddings, layer norms, biases) in float32, drop the non-parameter
causal-mask buffers, and write a sharded safetensors checkpoint with an
index file (<=64 MiB of tensor data per shard, oversized tensors alone).
"""

import json
import os

import torch
from safetensors.torch import save_file

IN_PATH = "inputs/base/model.safetensors"
OUT_DIR = "out/T3"
SHARD_LIMIT_BYTES = 64 * 1024 * 1024

N_LAYERS = 12
PROJ_SUFFIXES = [
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
]

BF16_NAMES = {f"h.{i}.{suf}" for i in range(N_LAYERS) for suf in PROJ_SUFFIXES}
BUFFER_NAMES = {f"h.{i}.attn.bias" for i in range(N_LAYERS)}


def main():
    from safetensors import safe_open

    tensors = {}
    with safe_open(IN_PATH, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    assert len(tensors) == 160, f"expected 160 input tensors, got {len(tensors)}"

    # Drop non-parameter buffers.
    for name in BUFFER_NAMES:
        assert name in tensors, f"expected buffer {name!r} not found in input"
        del tensors[name]

    # Cast the projection matrices to bfloat16; everything else stays float32.
    out = {}
    for name, t in tensors.items():
        if name in BF16_NAMES:
            out[name] = t.to(torch.bfloat16).contiguous()
        else:
            out[name] = t.to(torch.float32).contiguous()

    # --- Required checks ---
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    assert n_bf16 == 48, f"expected exactly 48 bfloat16 tensors, got {n_bf16}"
    assert out["h.0.attn.c_attn.weight"].dtype == torch.bfloat16, (
        "h.0.attn.c_attn.weight must be bfloat16"
    )
    assert out["wte.weight"].dtype == torch.float32, "wte.weight must be float32"
    assert len(out) == 148, f"expected exactly 148 output tensors, got {len(out)}"
    for name in BF16_NAMES:
        assert name in out and out[name].dtype == torch.bfloat16, (
            f"{name} must be present and bfloat16"
        )
    for name in BUFFER_NAMES:
        assert name not in out, f"buffer {name} must be dropped"

    # --- Build shards ---
    # Greedy bin-packing in tensor iteration order (dict insertion order,
    # i.e. original checkpoint order). Oversized tensors get their own shard.
    def tensor_nbytes(t):
        return t.numel() * t.element_size()

    shards = []  # list of dict name->tensor
    current = {}
    current_size = 0
    for name, t in out.items():
        size = tensor_nbytes(t)
        if size > SHARD_LIMIT_BYTES:
            if current:
                shards.append(current)
                current = {}
                current_size = 0
            shards.append({name: t})
            continue
        if current and current_size + size > SHARD_LIMIT_BYTES:
            shards.append(current)
            current = {}
            current_size = 0
        current[name] = t
        current_size += size
    if current:
        shards.append(current)

    os.makedirs(OUT_DIR, exist_ok=True)

    n_shards = len(shards)
    weight_map = {}
    total_size = 0
    for idx, shard in enumerate(shards, start=1):
        shard_name = f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
        shard_path = os.path.join(OUT_DIR, shard_name)
        save_file(shard, shard_path)
        for name in shard:
            weight_map[name] = shard_name
            total_size += tensor_nbytes(out[name])

    assert set(weight_map.keys()) == set(out.keys())

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"Wrote {len(out)} tensors across {n_shards} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
