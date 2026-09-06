"""
T3: Mixed-precision export with sharding (Pythia-1B).

Plain script on top of `safetensors` + `torch` (both in F-allowed.md).
No merging/adapter logic is needed here, just precise dtype control per
tensor and a sharded write with an index file, so a direct script is the
smallest, most auditable route through the allowed toolset.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
IN_PATH = HERE / "../../inputs/base/model.safetensors"
OUT_DIR = HERE
MAX_SHARD_BYTES = 256 * 1024 * 1024  # 268,435,456 bytes

NUM_LAYERS = 16

# Exactly the 64 projection matrices to cast to bfloat16.
PROJECTION_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.(attention\.query_key_value\.weight"
    r"|attention\.dense\.weight"
    r"|mlp\.dense_h_to_4h\.weight"
    r"|mlp\.dense_4h_to_h\.weight)$"
)

# The 48 non-parameter buffers to drop.
BUFFER_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)


def main() -> None:
    tensors: dict[str, torch.Tensor] = {}

    with safe_open(str(IN_PATH), framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            if BUFFER_RE.match(key):
                continue
            t = f.get_tensor(key)
            if PROJECTION_RE.match(key):
                t = t.to(torch.bfloat16)
            else:
                t = t.to(torch.float32)
            tensors[key] = t

    # --- Required checks: fail loudly before writing anything. ---
    bf16_keys = [k for k, t in tensors.items() if t.dtype == torch.bfloat16]
    if len(bf16_keys) != 64:
        raise AssertionError(f"expected 64 bfloat16 tensors, got {len(bf16_keys)}")

    qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tensors[qkv0].dtype != torch.bfloat16:
        raise AssertionError(f"{qkv0} must be bfloat16")

    embed_in = "gpt_neox.embed_in.weight"
    if tensors[embed_in].dtype != torch.float32:
        raise AssertionError(f"{embed_in} must be float32")

    if len(tensors) != 196:
        raise AssertionError(f"expected 196 tensors, got {len(tensors)}")

    dropped = [k for k in keys if BUFFER_RE.match(k)]
    if len(dropped) != 48:
        raise AssertionError(f"expected to drop 48 buffers, dropped {len(dropped)}")

    projected = [k for k in tensors if PROJECTION_RE.match(k)]
    if len(projected) != 64:
        raise AssertionError(f"expected 64 projection matrices, found {len(projected)}")
    if set(bf16_keys) != set(projected):
        raise AssertionError("bfloat16 tensors do not exactly match the projection set")

    # --- Shard assignment. ---
    # Greedy bin-packing in key order; any tensor over the limit alone gets
    # its own shard (only embed_in / embed_out qualify here, each ~206 MB).
    def tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[list[str]] = []
    shard_sizes: list[int] = []
    shard_closed: list[bool] = []  # True for a shard holding one oversized tensor alone
    for key in tensors:  # dict preserves insertion == original file order (minus drops)
        size = tensor_bytes(tensors[key])
        if size > MAX_SHARD_BYTES:
            shards.append([key])
            shard_sizes.append(size)
            shard_closed.append(True)
            continue
        placed = False
        for i, total in enumerate(shard_sizes):
            if shard_closed[i]:
                continue
            if total + size <= MAX_SHARD_BYTES:
                shards[i].append(key)
                shard_sizes[i] += size
                placed = True
                break
        if not placed:
            shards.append([key])
            shard_sizes.append(size)
            shard_closed.append(False)

    num_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    shard_names = []
    for idx, shard_keys in enumerate(shards, start=1):
        shard_name = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
        shard_names.append(shard_name)
        shard_tensors = {k: tensors[k].contiguous() for k in shard_keys}
        save_file(shard_tensors, str(OUT_DIR / shard_name))
        for k in shard_keys:
            weight_map[k] = shard_name
            total_size += tensor_bytes(tensors[k])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"wrote {len(tensors)} tensors across {num_shards} shards to {OUT_DIR}")
    for name, size in zip(shard_names, shard_sizes):
        print(f"  {name}: {size / (1024*1024):.1f} MiB")


if __name__ == "__main__":
    main()
