"""T3: mixed-precision export of OLMo-1B-0724-hf with sharding.

Plain torch + safetensors script. Casts exactly the 112 per-layer projection
matrices to bfloat16, keeps every other tensor float32 and bit-identical,
enforces the required checks before writing, then writes greedy shards of at
most 256 MiB of tensor data plus model.safetensors.index.json.
"""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_DIR = "inputs/base"
OUT_DIR = "out/T3"
MAX_SHARD = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data

PROJ_RE = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def fail(msg):
    print("CHECK FAILED: " + msg, file=sys.stderr)
    sys.exit(1)


def main():
    index = json.load(open(os.path.join(IN_DIR, "model.safetensors.index.json")))
    weight_map = index["weight_map"]

    tensors = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(IN_DIR, shard), framework="pt") as f:
            for name in f.keys():
                tensors[name] = f.get_tensor(name)
    if len(tensors) != 114:
        fail(f"expected 114 input tensors, got {len(tensors)}")

    out = {}
    for name, t in tensors.items():
        if t.dtype != torch.float32:
            fail(f"{name} is {t.dtype} in the input, expected float32")
        if PROJ_RE.match(name):
            out[name] = t.to(torch.bfloat16).contiguous()
        else:
            out[name] = t.contiguous()

    # Required checks, before writing anything.
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 112:
        fail(f"expected exactly 112 bfloat16 tensors, got {n_bf16}")
    if out["model.layers.0.self_attn.q_proj.weight"].dtype != torch.bfloat16:
        fail("model.layers.0.self_attn.q_proj.weight is not bfloat16")
    if out["model.embed_tokens.weight"].dtype != torch.float32:
        fail("model.embed_tokens.weight is not float32")
    if len(out) != 114:
        fail(f"expected 114 output tensors, got {len(out)}")
    if set(out) != set(tensors):
        fail("tensor name set changed")
    for name, t in out.items():
        if t.dtype == torch.float32 and not torch.equal(t, tensors[name]):
            fail(f"{name} float32 values changed")
        if t.shape != tensors[name].shape:
            fail(f"{name} shape changed")

    # Greedy sharding in sorted name order; oversized tensors go alone.
    shards = []  # list of (names, nbytes)
    cur, cur_bytes = [], 0
    for name in sorted(out):
        nb = out[name].numel() * out[name].element_size()
        if nb > MAX_SHARD:
            if cur:
                shards.append((cur, cur_bytes))
                cur, cur_bytes = [], 0
            shards.append(([name], nb))
            continue
        if cur_bytes + nb > MAX_SHARD:
            shards.append((cur, cur_bytes))
            cur, cur_bytes = [], 0
        cur.append(name)
        cur_bytes += nb
    if cur:
        shards.append((cur, cur_bytes))
    for names, nb in shards:
        if nb > MAX_SHARD and len(names) != 1:
            fail("shard over budget")

    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(shards)
    out_map = {}
    total = 0
    for i, (names, nb) in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file({k: out[k] for k in names}, os.path.join(OUT_DIR, fname),
                  metadata={"format": "pt"})
        for k in names:
            out_map[k] = fname
        total += nb
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": out_map}, f,
                  indent=2, sort_keys=True)

    # Post-write verification of the on-disk result.
    seen = 0
    for fname in sorted(set(out_map.values())):
        with safe_open(os.path.join(OUT_DIR, fname), framework="pt") as f:
            keys = list(f.keys())
            size = 0
            for k in keys:
                t = f.get_tensor(k)
                if out_map[k] != fname:
                    fail(f"index mismatch for {k}")
                if t.dtype != out[k].dtype or not torch.equal(t, out[k]):
                    fail(f"on-disk mismatch for {k}")
                size += t.numel() * t.element_size()
            if size > MAX_SHARD and len(keys) != 1:
                fail(f"{fname} exceeds shard budget")
            seen += len(keys)
    if seen != 114 or len(out_map) != 114:
        fail(f"on-disk tensor count {seen}, index entries {len(out_map)}")
    print(f"OK: wrote {n} shards, {seen} tensors, {n_bf16} bfloat16, total {total} bytes")


if __name__ == "__main__":
    main()
