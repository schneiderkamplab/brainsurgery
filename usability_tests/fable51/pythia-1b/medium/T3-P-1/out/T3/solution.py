"""T3: mixed-precision export of Pythia-1B with sharding."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
OUT_DIR = "out/T3"
MAX_SHARD_BYTES = 256 * 1024 * 1024  # 268,435,456

PROJ_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\."
    r"(attention\.query_key_value|attention\.dense|mlp\.dense_h_to_4h|mlp\.dense_4h_to_h)"
    r"\.weight$"
)
BUFFER_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


sd = load_file(SRC)
if len(sd) != 244:
    fail(f"expected 244 input tensors, got {len(sd)}")

out = {}
n_proj = n_buf = 0
for name, t in sd.items():  # keep input order
    if BUFFER_RE.match(name):
        n_buf += 1
        continue
    if PROJ_RE.match(name):
        n_proj += 1
        out[name] = t.to(torch.float32).to(torch.bfloat16).contiguous()
    else:
        out[name] = t.to(torch.float32).contiguous()

# Required checks (fail before writing).
if n_proj != 64:
    fail(f"projection pattern matched {n_proj} tensors, expected 64")
if n_buf != 48:
    fail(f"buffer pattern matched {n_buf} tensors, expected 48")
n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
if n_bf16 != 64:
    fail(f"{n_bf16} bfloat16 tensors, expected 64")
if out["gpt_neox.layers.0.attention.query_key_value.weight"].dtype != torch.bfloat16:
    fail("layers.0 query_key_value.weight is not bfloat16")
if out["gpt_neox.embed_in.weight"].dtype != torch.float32:
    fail("embed_in.weight is not float32")
if any(t.dtype not in (torch.bfloat16, torch.float32) for t in out.values()):
    fail("unexpected dtype in output")
if len(out) != 196:
    fail(f"output has {len(out)} tensors, expected 196")
if set(out) != set(sd) - {k for k in sd if BUFFER_RE.match(k)}:
    fail("output key set does not equal input minus buffers")

# Greedy sharding in tensor order; oversized tensors go alone.
shards = []  # list of (names, size)
cur, cur_size = [], 0
for name, t in out.items():
    size = t.numel() * t.element_size()
    if size > MAX_SHARD_BYTES:
        if cur:
            shards.append((cur, cur_size))
            cur, cur_size = [], 0
        shards.append(([name], size))
        continue
    if cur and cur_size + size > MAX_SHARD_BYTES:
        shards.append((cur, cur_size))
        cur, cur_size = [], 0
    cur.append(name)
    cur_size += size
if cur:
    shards.append((cur, cur_size))

for names, size in shards:
    if size > MAX_SHARD_BYTES and len(names) != 1:
        fail("shard over budget")

os.makedirs(OUT_DIR, exist_ok=True)
n = len(shards)
weight_map = {}
total_size = 0
for i, (names, size) in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    save_file({k: out[k] for k in names}, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
    for k in names:
        weight_map[k] = fname
    total_size += size

index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2, sort_keys=True)

# Post-write verification.
reloaded = {}
for fname in sorted(set(weight_map.values())):
    reloaded.update(load_file(os.path.join(OUT_DIR, fname)))
if len(reloaded) != 196 or set(reloaded) != set(out):
    fail("reloaded key set mismatch")
for k, t in out.items():
    r = reloaded[k]
    if r.dtype != t.dtype or r.shape != t.shape or not torch.equal(r, t):
        fail(f"reloaded tensor mismatch for {k}")
print(f"OK: {len(out)} tensors in {n} shards, total {total_size} bytes")
for i, (names, size) in enumerate(shards, 1):
    print(f"  shard {i}: {len(names)} tensors, {size} bytes")
