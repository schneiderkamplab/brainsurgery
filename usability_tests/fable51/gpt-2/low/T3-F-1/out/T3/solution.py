"""T3: mixed-precision sharded export of GPT-2 (124M)."""
import json
import os
import re

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
OUT = "out/T3"
SHARD_LIMIT = 64 * 1024 * 1024
CAST_RE = re.compile(r"^h\.\d+\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$")
DROP_RE = re.compile(r"^h\.\d+\.attn\.bias$")

sd = load_file(SRC)
assert len(sd) == 160, len(sd)

out = {}
for name, t in sd.items():
    if DROP_RE.match(name):
        continue
    out[name] = t.to(torch.bfloat16) if CAST_RE.match(name) else t

# Required checks (fail before writing).
n_bf16 = sum(t.dtype == torch.bfloat16 for t in out.values())
assert n_bf16 == 48, f"expected 48 bfloat16 tensors, got {n_bf16}"
assert out["h.0.attn.c_attn.weight"].dtype == torch.bfloat16
assert out["wte.weight"].dtype == torch.float32
assert len(out) == 148, f"expected 148 tensors, got {len(out)}"
assert all(t.dtype == torch.float32 for n, t in out.items() if not CAST_RE.match(n))

# Greedy sharding in original key order; oversized tensors go alone.
shards, cur, cur_size = [], {}, 0
for name, t in out.items():
    size = t.numel() * t.element_size()
    if cur and cur_size + size > SHARD_LIMIT:
        shards.append(cur)
        cur, cur_size = {}, 0
    cur[name] = t.contiguous()
    cur_size += size
    if size > SHARD_LIMIT:
        shards.append(cur)
        cur, cur_size = {}, 0
if cur:
    shards.append(cur)

os.makedirs(OUT, exist_ok=True)
weight_map = {}
total = 0
n = len(shards)
for i, shard in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    shard_bytes = sum(t.numel() * t.element_size() for t in shard.values())
    assert shard_bytes <= SHARD_LIMIT or len(shard) == 1, fname
    save_file(shard, os.path.join(OUT, fname), metadata={"format": "pt"})
    for name in shard:
        weight_map[name] = fname
    total += shard_bytes
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
print(f"wrote {n} shards, {len(weight_map)} tensors, {total} bytes")
