"""
T3: Mixed-precision export with sharding (GPT-2, 124M)

- Cast the 48 large projection matrices (attn.c_attn, attn.c_proj,
  mlp.c_fc, mlp.c_proj weights, per layer) to bfloat16.
- Keep everything else (embeddings, layer norms, biases) in float32.
- Drop the non-parameter causal-mask buffers `h.<i>.attn.bias`.
- Write a sharded safetensors checkpoint with an index file, each shard
  holding at most 64 MiB of tensor data (a single oversized tensor gets
  its own shard).
"""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_PATH = "inputs/base/model.safetensors"
OUT_DIR = "out/T3"
INDEX_NAME = "model.safetensors.index.json"
SHARD_BYTE_LIMIT = 64 * 1024 * 1024  # 64 MiB, tensor data only
NUM_LAYERS = 12

BF16_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def is_bf16_target(name: str) -> bool:
    return any(name == f"h.{i}.{suf}" for i in range(NUM_LAYERS) for suf in BF16_SUFFIXES)


def is_dropped_buffer(name: str) -> bool:
    return any(name == f"h.{i}.attn.bias" for i in range(NUM_LAYERS))


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(IN_PATH, framework="pt") as f:
        for name in f.keys():
            if is_dropped_buffer(name):
                continue
            t = f.get_tensor(name)
            if is_bf16_target(name):
                t = t.to(torch.bfloat16)
            else:
                assert t.dtype == torch.float32, f"unexpected dtype for {name}: {t.dtype}"
            tensors[name] = t.contiguous()

    # --- Required checks (fail loudly before writing) ---
    bf16_names = [n for n, t in tensors.items() if t.dtype == torch.bfloat16]
    assert len(bf16_names) == 48, f"expected 48 bfloat16 tensors, got {len(bf16_names)}"
    assert tensors["h.0.attn.c_attn.weight"].dtype == torch.bfloat16, (
        "h.0.attn.c_attn.weight must be bfloat16"
    )
    assert tensors["wte.weight"].dtype == torch.float32, "wte.weight must be float32"
    assert len(tensors) == 148, f"expected 148 tensors in output, got {len(tensors)}"
    for name in tensors:
        assert not is_dropped_buffer(name), f"buffer {name} should have been dropped"

    # --- Greedy bin-packing into shards, oversized tensors alone ---
    names = list(tensors.keys())
    sizes = {n: tensor_nbytes(tensors[n]) for n in names}

    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name in names:
        size = sizes[name]
        if size > SHARD_BYTE_LIMIT:
            if current:
                shards.append(current)
                current, current_size = [], 0
            shards.append([name])
            continue
        if current and current_size + size > SHARD_BYTE_LIMIT:
            shards.append(current)
            current, current_size = [], 0
        current.append(name)
        current_size += size
    if current:
        shards.append(current)

    num_shards = len(shards)
    digits = max(5, len(str(num_shards)))
    weight_map: dict[str, str] = {}
    total_size = 0

    for idx, shard_names in enumerate(shards, start=1):
        shard_filename = (
            f"model-{idx:0{digits}d}-of-{num_shards:0{digits}d}.safetensors"
        )
        shard_tensors = {n: tensors[n] for n in shard_names}
        save_file(shard_tensors, os.path.join(OUT_DIR, shard_filename))
        for n in shard_names:
            weight_map[n] = shard_filename
            total_size += sizes[n]

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUT_DIR, INDEX_NAME), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"Wrote {len(tensors)} tensors across {num_shards} shard(s) to {OUT_DIR}/")


if __name__ == "__main__":
    main()
