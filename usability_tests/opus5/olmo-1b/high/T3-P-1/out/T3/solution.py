"""T3: mixed-precision export with sharding for OLMo-1B-0724-hf.

Casts the 112 projection matrices to bfloat16, keeps everything else float32
with unchanged values, and writes a sharded safetensors checkpoint with an
index file.  Shards are streamed one at a time so peak memory stays around one
shard rather than the whole 5 GB checkpoint.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_DIR = "inputs/base"
OUT_DIR = "out/T3"
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024  # 268,435,456

# The projection matrices, and only these, become bfloat16.
PROJ_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def load_input_map() -> dict[str, str]:
    with open(os.path.join(IN_DIR, INDEX_NAME)) as fh:
        index = json.load(fh)
    return index["weight_map"]


def layer_indices(names) -> list[int]:
    idx = set()
    for name in names:
        parts = name.split(".")
        if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
            idx.add(int(parts[2]))
    return sorted(idx)


def main() -> None:
    weight_map_in = load_input_map()
    names = list(weight_map_in)

    # Exactly which tensors must be cast: built by construction from the layer
    # indices actually present, never by a loose ".*weight" pattern.
    projections = {
        f"model.layers.{i}.{suffix}"
        for i in layer_indices(names)
        for suffix in PROJ_SUFFIXES
    }
    missing = sorted(projections - set(names))
    if missing:
        raise SystemExit(f"FAIL: projection matrices absent from input: {missing[:5]}")

    # Read shapes and source dtypes without materialising any tensor data.
    shapes: dict[str, tuple[int, ...]] = {}
    src_dtype: dict[str, str] = {}
    for shard_file in sorted(set(weight_map_in.values())):
        with safe_open(os.path.join(IN_DIR, shard_file), framework="pt") as handle:
            for name in handle.keys():
                sl = handle.get_slice(name)
                shapes[name] = tuple(sl.get_shape())
                src_dtype[name] = sl.get_dtype()

    if set(shapes) != set(names):
        raise SystemExit("FAIL: index weight_map disagrees with the shard contents")
    non_f32 = sorted(n for n, d in src_dtype.items() if d != "F32")
    if non_f32:
        raise SystemExit(f"FAIL: input tensors are not all float32: {non_f32[:5]}")

    out_dtype = {
        name: (torch.bfloat16 if name in projections else torch.float32)
        for name in names
    }

    # --- Required checks, before anything is written -------------------------
    n_bf16 = sum(1 for d in out_dtype.values() if d is torch.bfloat16)
    if n_bf16 != 112:
        raise SystemExit(f"FAIL: {n_bf16} tensors would be bfloat16, expected 112")
    probe = "model.layers.0.self_attn.q_proj.weight"
    if out_dtype.get(probe) is not torch.bfloat16:
        raise SystemExit(f"FAIL: {probe} is not bfloat16")
    if out_dtype.get("model.embed_tokens.weight") is not torch.float32:
        raise SystemExit("FAIL: model.embed_tokens.weight is not float32")
    if len(out_dtype) != 114:
        raise SystemExit(f"FAIL: output would hold {len(out_dtype)} tensors, expected 114")
    # No buffers exist in this checkpoint, so nothing may be dropped.
    if set(out_dtype) != set(names):
        raise SystemExit("FAIL: the output key set differs from the input key set")

    # --- Shard planning ------------------------------------------------------
    def nbytes(name: str) -> int:
        n = 1
        for dim in shapes[name]:
            n *= dim
        return n * out_dtype[name].itemsize

    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name in names:
        size = nbytes(name)
        if size > MAX_SHARD_BYTES:
            # Oversized tensor: flush what we have, then give it its own shard.
            if current:
                shards.append(current)
                current, current_size = [], 0
            shards.append([name])
            continue
        if current and current_size + size > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_size = [], 0
        current.append(name)
        current_size += size
    if current:
        shards.append(current)

    total = len(shards)
    shard_names = [f"model-{i + 1:05d}-of-{total:05d}.safetensors" for i in range(total)]
    for shard, file_name in zip(shards, shard_names):
        size = sum(nbytes(n) for n in shard)
        if len(shard) > 1 and size > MAX_SHARD_BYTES:
            raise SystemExit(f"FAIL: {file_name} holds {size} bytes, over the shard budget")

    weight_map = {n: f for shard, f in zip(shards, shard_names) for n in shard}
    if len(weight_map) != 114:
        raise SystemExit(f"FAIL: weight_map covers {len(weight_map)} tensors, expected 114")

    # --- Write ---------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)

    # Only one shard's worth of tensors is held in memory at a time.
    handles = {
        f: safe_open(os.path.join(IN_DIR, f), framework="pt")
        for f in sorted(set(weight_map_in.values()))
    }
    written = 0
    for shard, file_name in zip(shards, shard_names):
        tensors = {}
        for name in shard:
            tensor = handles[weight_map_in[name]].get_tensor(name)
            if tensor.dtype is not torch.float32:
                raise SystemExit(f"FAIL: {name} arrived as {tensor.dtype}, expected float32")
            if out_dtype[name] is torch.bfloat16:
                tensor = tensor.to(torch.bfloat16)
            tensors[name] = tensor.contiguous().clone()
        save_file(tensors, os.path.join(OUT_DIR, file_name), metadata={"format": "pt"})
        written += len(tensors)
        del tensors

    if written != 114:
        raise SystemExit(f"FAIL: wrote {written} tensors, expected 114")

    total_size = sum(nbytes(n) for n in names)
    with open(os.path.join(OUT_DIR, INDEX_NAME), "w") as fh:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, fh, indent=2)
        fh.write("\n")

    # --- Verify what landed on disk -----------------------------------------
    seen_bf16 = 0
    seen: set[str] = set()
    for file_name in shard_names:
        with safe_open(os.path.join(OUT_DIR, file_name), framework="pt") as handle:
            for name in handle.keys():
                sl = handle.get_slice(name)
                dtype = sl.get_dtype()
                seen.add(name)
                if dtype == "BF16":
                    seen_bf16 += 1
                elif dtype != "F32":
                    raise SystemExit(f"FAIL: {name} written as {dtype}")
                if tuple(sl.get_shape()) != shapes[name]:
                    raise SystemExit(f"FAIL: {name} shape changed on write")
    if seen_bf16 != 112:
        raise SystemExit(f"FAIL: {seen_bf16} bfloat16 tensors on disk, expected 112")
    if len(seen) != 114 or seen != set(names):
        raise SystemExit(f"FAIL: {len(seen)} tensors on disk, expected the 114 input names")

    print(f"wrote {len(seen)} tensors ({seen_bf16} bfloat16) across {total} shards")
    for shard, file_name in zip(shards, shard_names):
        print(f"  {file_name}: {len(shard)} tensors, {sum(nbytes(n) for n in shard)} bytes")


if __name__ == "__main__":
    main()
