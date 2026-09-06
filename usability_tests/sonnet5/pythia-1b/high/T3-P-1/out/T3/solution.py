"""
T3: Mixed-precision export with sharding (Pythia-1B).

Loads the base Pythia-1B checkpoint, casts the 64 large projection matrices
(query_key_value, dense, dense_h_to_4h, dense_4h_to_h per layer) to bfloat16,
upcasts everything else to float32, drops the 48 non-parameter buffers
(attention.bias, attention.masked_bias, attention.rotary_emb.inv_freq per
layer), and writes the result as a sharded safetensors checkpoint (<=256MiB
of tensor data per shard, oversized tensors alone in their own shard) plus
a model.safetensors.index.json.
"""

import json
import os

import torch
from safetensors.torch import load_file, save_file

IN_PATH = "inputs/base/model.safetensors"
OUT_DIR = "out/T3"
MAX_SHARD_BYTES = 256 * 1024 * 1024  # 256 MiB of tensor data per shard
N_LAYERS = 16

PROJECTION_SUFFIXES = [
    "attention.query_key_value.weight",
    "attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
]
BUFFER_SUFFIXES = [
    "attention.bias",
    "attention.masked_bias",
    "attention.rotary_emb.inv_freq",
]

bf16_keys = {f"gpt_neox.layers.{i}.{suf}" for i in range(N_LAYERS) for suf in PROJECTION_SUFFIXES}
drop_keys = {f"gpt_neox.layers.{i}.{suf}" for i in range(N_LAYERS) for suf in BUFFER_SUFFIXES}


def dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    state_dict = load_file(IN_PATH)

    out = {}
    for name, tensor in state_dict.items():
        if name in drop_keys:
            continue
        if name in bf16_keys:
            out[name] = tensor.to(torch.bfloat16).contiguous()
        else:
            out[name] = tensor.to(torch.float32).contiguous()

    # --- Required checks: fail loudly before writing anything ---
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    assert n_bf16 == 64, f"expected exactly 64 bfloat16 tensors, got {n_bf16}"

    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    assert out[probe].dtype == torch.bfloat16, f"{probe} is {out[probe].dtype}, expected bfloat16"

    assert out["gpt_neox.embed_in.weight"].dtype == torch.float32, (
        "gpt_neox.embed_in.weight is not float32"
    )

    assert len(out) == 196, f"expected exactly 196 tensors in the output, got {len(out)}"

    for i in range(N_LAYERS):
        for suf in BUFFER_SUFFIXES:
            key = f"gpt_neox.layers.{i}.{suf}"
            assert key not in out, f"buffer {key} should have been dropped"

    # --- Greedy sequential sharding: fill a shard until the next tensor
    # would push it over budget, then start a new one. A tensor bigger than
    # the budget on its own gets its own shard. ---
    names = list(out.keys())
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        tensor = out[name]
        size = tensor.numel() * dtype_size(tensor.dtype)
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
    if current:
        shards.append(current)

    n_shards = len(shards)
    weight_map = {}
    total_size = 0
    for shard_idx, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{shard_idx:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {name: out[name] for name in shard_names}
        save_file(shard_tensors, os.path.join(OUT_DIR, shard_filename))
        for name in shard_names:
            weight_map[name] = shard_filename
            total_size += out[name].numel() * dtype_size(out[name].dtype)

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"Wrote {len(out)} tensors across {n_shards} shard(s) to {OUT_DIR}/")


if __name__ == "__main__":
    main()
