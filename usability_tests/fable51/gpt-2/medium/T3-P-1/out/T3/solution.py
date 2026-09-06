"""T3: mixed-precision export of GPT-2 (124M) with sharding.

- cast the 48 per-layer projection matrices to bfloat16
- keep every other parameter float32, values unchanged
- drop the 12 causal-mask buffers h.<i>.attn.bias
- write sharded safetensors (<= 64 MiB tensor data per shard, oversized
  tensors alone) plus model.safetensors.index.json
"""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "inputs", "base", "model.safetensors")
OUT_DIR = HERE
MAX_SHARD = 64 * 1024 * 1024  # 67,108,864 bytes of tensor data

N_LAYERS = 12
PROJ_RE = re.compile(r"^h\.(\d+)\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$")
BUFFER_RE = re.compile(r"^h\.(\d+)\.attn\.bias$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


sd = load_file(SRC)
if len(sd) != 160:
    fail(f"expected 160 input tensors, got {len(sd)}")

out = {}
n_cast = 0
n_dropped = 0
for name, t in sd.items():
    if BUFFER_RE.match(name):
        n_dropped += 1
        continue
    if PROJ_RE.match(name):
        out[name] = t.to(torch.bfloat16).contiguous()
        n_cast += 1
    else:
        if t.dtype != torch.float32:
            fail(f"{name} is {t.dtype}, expected float32 input")
        out[name] = t.contiguous()

# Required checks (before writing)
bf16 = [k for k, v in out.items() if v.dtype == torch.bfloat16]
if len(bf16) != 48:
    fail(f"expected exactly 48 bfloat16 tensors, got {len(bf16)}")
if n_cast != 4 * N_LAYERS or n_dropped != N_LAYERS:
    fail(f"cast {n_cast} matrices and dropped {n_dropped} buffers; expected 48 and 12")
if out["h.0.attn.c_attn.weight"].dtype != torch.bfloat16:
    fail("h.0.attn.c_attn.weight is not bfloat16")
if out["wte.weight"].dtype != torch.float32:
    fail("wte.weight is not float32")
if len(out) != 148:
    fail(f"expected 148 output tensors, got {len(out)}")
for k, v in out.items():
    if v.dtype not in (torch.bfloat16, torch.float32):
        fail(f"{k} has unexpected dtype {v.dtype}")
    if v.dtype == torch.float32 and not torch.equal(v, sd[k]):
        fail(f"{k} float32 values changed")

# Greedy sharding in key order (HF convention); oversized tensors sit alone.
shards = []
cur, cur_size = {}, 0
for name, t in out.items():
    size = t.numel() * t.element_size()
    if cur and cur_size + size > MAX_SHARD:
        shards.append(cur)
        cur, cur_size = {}, 0
    cur[name] = t
    cur_size += size
if cur:
    shards.append(cur)

for shard in shards:
    total = sum(t.numel() * t.element_size() for t in shard.values())
    if total > MAX_SHARD and len(shard) != 1:
        fail(f"shard exceeds budget with {len(shard)} tensors")

os.makedirs(OUT_DIR, exist_ok=True)
n = len(shards)
weight_map = {}
total_size = 0
for i, shard in enumerate(shards, start=1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    save_file(shard, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
    for name, t in shard.items():
        weight_map[name] = fname
        total_size += t.numel() * t.element_size()

index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2, sort_keys=True)
    f.write("\n")

# Verify what was written
reloaded = {}
for fname in sorted(set(weight_map.values())):
    reloaded.update(load_file(os.path.join(OUT_DIR, fname)))
if len(reloaded) != 148 or set(reloaded) != set(out):
    fail("reloaded key set differs from intended output")
for k, v in out.items():
    r = reloaded[k]
    if r.dtype != v.dtype or r.shape != v.shape or not torch.equal(r, v):
        fail(f"reloaded {k} differs")

print(f"OK: {len(out)} tensors, {len(bf16)} bfloat16, {n} shards, {total_size} bytes")
for i, shard in enumerate(shards, start=1):
    print(f"  shard {i}: {len(shard)} tensors, "
          f"{sum(t.numel() * t.element_size() for t in shard.values())} bytes")
