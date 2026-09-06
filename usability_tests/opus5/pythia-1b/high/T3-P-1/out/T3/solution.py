#!/usr/bin/env python
"""T3: mixed-precision export of Pythia-1B with sharding.

Casts the 64 projection matrices to bfloat16, upcasts everything else to
float32, drops the 48 non-parameter attention buffers, and writes a sharded
safetensors checkpoint with an index file to out/T3/.
"""

import json
import math
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_PATH = "inputs/base/model.safetensors"
OUT_DIR = "out/T3"
INDEX_NAME = "model.safetensors.index.json"
SHARD_LIMIT = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data

# Exactly the four projection matrices per layer. Anchored on both ends so
# that biases (".weight" only), embeddings and layer norms cannot match.
PROJ_RE = re.compile(
    r"^gpt_neox\.layers\.(\d+)\."
    r"(?:attention\.(?:query_key_value|dense)|mlp\.(?:dense_h_to_4h|dense_4h_to_h))"
    r"\.weight$"
)
# The three non-parameter buffers per layer. "attention.bias" is the causal
# mask, not a projection bias -- the projection biases are
# "attention.dense.bias" / "attention.query_key_value.bias" and must be kept.
BUFFER_RE = re.compile(
    r"^gpt_neox\.layers\.(\d+)\.attention\.(?:bias|masked_bias|rotary_emb\.inv_freq)$"
)

EXPECTED_PROJ_SHAPES = {
    "attention.query_key_value.weight": [6144, 2048],
    "attention.dense.weight": [2048, 2048],
    "mlp.dense_h_to_4h.weight": [8192, 2048],
    "mlp.dense_4h_to_h.weight": [2048, 8192],
}

DTYPE_SIZE = {torch.bfloat16: 2, torch.float32: 4}


def fail(msg):
    raise SystemExit(f"CHECK FAILED: {msg}")


def main():
    if not os.path.exists(IN_PATH):
        fail(f"missing input {IN_PATH}")
    os.makedirs(OUT_DIR, exist_ok=True)

    with safe_open(IN_PATH, framework="pt", device="cpu") as f:
        all_keys = sorted(f.keys())
        shapes = {k: list(f.get_slice(k).get_shape()) for k in all_keys}

    # ---- classify -------------------------------------------------------
    proj_keys = [k for k in all_keys if PROJ_RE.match(k)]
    buffer_keys = [k for k in all_keys if BUFFER_RE.match(k)]
    kept_keys = [k for k in all_keys if k not in set(buffer_keys)]

    print(f"input tensors: {len(all_keys)}")
    print(f"projection matrices matched: {len(proj_keys)}")
    print(f"buffers matched for deletion: {len(buffer_keys)}")

    if len(all_keys) != 244:
        fail(f"expected 244 input tensors, found {len(all_keys)}")
    if len(proj_keys) != 64:
        fail(f"expected 64 projection matrices, matched {len(proj_keys)}")
    if len(buffer_keys) != 48:
        fail(f"expected 48 buffers to delete, matched {len(buffer_keys)}")
    if set(proj_keys) & set(buffer_keys):
        fail("a tensor was classified as both projection and buffer")

    # No projection may be deleted; no bias/norm/embedding may be cast.
    for k in buffer_keys:
        if not k.startswith("gpt_neox.layers.") or ".attention." not in k:
            fail(f"unexpected buffer name {k}")
    for k in proj_keys:
        suffix = k.split(".", 3)[3]  # strip "gpt_neox.layers.<i>."
        if suffix not in EXPECTED_PROJ_SHAPES:
            fail(f"unexpected projection name {k}")
        if shapes[k] != EXPECTED_PROJ_SHAPES[suffix]:
            fail(f"{k}: expected shape {EXPECTED_PROJ_SHAPES[suffix]}, found {shapes[k]}")
    # Every layer contributes exactly 4 projections and 3 buffers.
    for i in range(16):
        n_p = sum(1 for k in proj_keys if PROJ_RE.match(k).group(1) == str(i))
        n_b = sum(1 for k in buffer_keys if BUFFER_RE.match(k).group(1) == str(i))
        if n_p != 4 or n_b != 3:
            fail(f"layer {i}: {n_p} projections (want 4), {n_b} buffers (want 3)")

    proj_set = set(proj_keys)
    out_dtype = {k: (torch.bfloat16 if k in proj_set else torch.float32) for k in kept_keys}

    # ---- required checks, before writing anything -----------------------
    n_bf16 = sum(1 for k in kept_keys if out_dtype[k] is torch.bfloat16)
    if n_bf16 != 64:
        fail(f"exactly 64 tensors must be bfloat16, plan has {n_bf16}")
    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    if out_dtype.get(probe) is not torch.bfloat16:
        fail(f"{probe} must be bfloat16, plan has {out_dtype.get(probe)}")
    if out_dtype.get("gpt_neox.embed_in.weight") is not torch.float32:
        fail(f"gpt_neox.embed_in.weight must be float32, plan has "
             f"{out_dtype.get('gpt_neox.embed_in.weight')}")
    if len(kept_keys) != 196:
        fail(f"output must have exactly 196 tensors, plan has {len(kept_keys)}")
    print("pre-write checks passed: 64 bf16, embed_in fp32, 196 tensors")

    # ---- plan the shards (greedy, in sorted key order) ------------------
    sizes = {k: math.prod(shapes[k]) * DTYPE_SIZE[out_dtype[k]] for k in kept_keys}
    for k, s in sizes.items():
        if s > SHARD_LIMIT:
            print(f"oversized tensor, gets its own shard: {k} ({s} bytes)")

    shards, cur, cur_size = [], [], 0
    for k in kept_keys:
        if cur and cur_size + sizes[k] > SHARD_LIMIT:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(k)
        cur_size += sizes[k]
    if cur:
        shards.append(cur)

    n_shards = len(shards)
    names = [f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)]
    for name, keys in zip(names, shards):
        total = sum(sizes[k] for k in keys)
        if total > SHARD_LIMIT and len(keys) != 1:
            fail(f"{name}: {total} bytes over the {SHARD_LIMIT} budget with {len(keys)} tensors")
        print(f"{name}: {len(keys):3d} tensors, {total:,} bytes")

    # ---- write ----------------------------------------------------------
    weight_map, total_size = {}, 0
    for name, keys in zip(names, shards):
        block = {}
        with safe_open(IN_PATH, framework="pt", device="cpu") as f:
            for k in keys:
                block[k] = f.get_tensor(k).to(out_dtype[k]).contiguous()
        for k in keys:
            if block[k].dtype is not out_dtype[k]:
                fail(f"{k}: wrote {block[k].dtype}, wanted {out_dtype[k]}")
            weight_map[k] = name
            total_size += sizes[k]
        save_file(block, os.path.join(OUT_DIR, name), metadata={"format": "pt"})
        del block

    with open(os.path.join(OUT_DIR, INDEX_NAME), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)
        f.write("\n")

    # ---- verify what landed on disk -------------------------------------
    seen, n_bf16_out = {}, 0
    for name in names:
        with safe_open(os.path.join(OUT_DIR, name), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in seen:
                    fail(f"{k} appears in both {seen[k]} and {name}")
                seen[k] = name
                t = f.get_slice(k)
                dt = t.get_dtype()
                if dt == "BF16":
                    n_bf16_out += 1
                elif dt != "F32":
                    fail(f"{k}: dtype {dt} on disk, expected BF16 or F32")
                if list(t.get_shape()) != shapes[k]:
                    fail(f"{k}: shape {t.get_shape()} on disk, expected {shapes[k]}")
    if len(seen) != 196:
        fail(f"{len(seen)} tensors on disk, expected 196")
    if n_bf16_out != 64:
        fail(f"{n_bf16_out} bfloat16 tensors on disk, expected 64")
    if seen != weight_map:
        fail("weight_map does not match what is actually stored in the shards")
    if set(seen) != set(kept_keys):
        fail("key set on disk differs from the planned key set")

    print(f"OK: {len(seen)} tensors in {n_shards} shards, {n_bf16_out} bfloat16, "
          f"{total_size:,} bytes of tensor data")


if __name__ == "__main__":
    sys.exit(main())
