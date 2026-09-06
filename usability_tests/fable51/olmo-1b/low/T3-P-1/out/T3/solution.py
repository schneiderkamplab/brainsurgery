"""T3: mixed-precision export of OLMo-1B-0724-hf with sharding."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_DIR = os.path.join(ROOT, "inputs", "base")
OUT_DIR = os.path.join(ROOT, "out", "T3")
MAX_SHARD = 256 * 1024 * 1024

PROJ_RE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def fail(msg):
    print("CHECK FAILED:", msg, file=sys.stderr)
    sys.exit(1)


def main():
    with open(os.path.join(IN_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    tensors = {}
    for sf in shard_files:
        tensors.update(load_file(os.path.join(IN_DIR, sf)))

    out = {}
    for name, t in tensors.items():
        if PROJ_RE.match(name):
            out[name] = t.to(torch.bfloat16).contiguous()
        else:
            if t.dtype != torch.float32:
                fail(f"{name} is {t.dtype}, expected float32 input")
            out[name] = t.contiguous()

    # Required checks
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 112:
        fail(f"expected 112 bfloat16 tensors, got {n_bf16}")
    if out["model.layers.0.self_attn.q_proj.weight"].dtype != torch.bfloat16:
        fail("q_proj layer 0 not bfloat16")
    if out["model.embed_tokens.weight"].dtype != torch.float32:
        fail("embed_tokens not float32")
    if len(out) != 114:
        fail(f"expected 114 tensors, got {len(out)}")
    if set(out) != set(tensors):
        fail("tensor name set changed")

    # Shard: greedy fill in original order; oversized tensors go alone.
    shards = []
    cur, cur_bytes = {}, 0
    for name in tensors:  # preserve input ordering
        t = out[name]
        nb = t.numel() * t.element_size()
        if nb > MAX_SHARD:
            if cur:
                shards.append(cur)
                cur, cur_bytes = {}, 0
            shards.append({name: t})
            continue
        if cur_bytes + nb > MAX_SHARD:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[name] = t
        cur_bytes += nb
    if cur:
        shards.append(cur)

    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(shards)
    weight_map = {}
    total = 0
    for i, shard in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        size = sum(t.numel() * t.element_size() for t in shard.values())
        if size > MAX_SHARD and len(shard) > 1:
            fail(f"shard {fname} exceeds budget: {size}")
        total += size
        save_file(shard, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        for name in shard:
            weight_map[name] = fname
    if len(weight_map) != 114:
        fail(f"weight_map has {len(weight_map)} entries")
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
    print(f"wrote {n} shards, {len(weight_map)} tensors, {total} bytes to {OUT_DIR}")


if __name__ == "__main__":
    main()
