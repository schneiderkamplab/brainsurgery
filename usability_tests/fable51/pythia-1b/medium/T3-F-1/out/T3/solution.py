"""T3: mixed-precision sharded export of Pythia-1B (plain torch + safetensors)."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
OUT = "out/T3"
MAX_SHARD = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data

PROJ_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.(attention\.(query_key_value|dense)"
    r"|mlp\.(dense_h_to_4h|dense_4h_to_h))\.weight$"
)
BUFFER_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)


def fail(msg):
    print("CHECK FAILED:", msg, file=sys.stderr)
    sys.exit(1)


src = load_file(SRC)
if len(src) != 244:
    fail(f"expected 244 input tensors, got {len(src)}")

out = {}
n_proj = n_buf = 0
for name, t in src.items():
    if BUFFER_RE.match(name):
        n_buf += 1
        continue
    if PROJ_RE.match(name):
        n_proj += 1
        out[name] = t.to(torch.bfloat16).contiguous()
    else:
        out[name] = t.to(torch.float32).contiguous()

# Required checks (before writing)
if n_buf != 48:
    fail(f"expected to drop 48 buffers, dropped {n_buf}")
if n_proj != 64:
    fail(f"expected 64 projection matrices, matched {n_proj}")
n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
if n_bf16 != 64:
    fail(f"expected exactly 64 bfloat16 tensors, got {n_bf16}")
if out["gpt_neox.layers.0.attention.query_key_value.weight"].dtype != torch.bfloat16:
    fail("layers.0 query_key_value.weight is not bfloat16")
if out["gpt_neox.embed_in.weight"].dtype != torch.float32:
    fail("embed_in.weight is not float32")
if len(out) != 196:
    fail(f"expected 196 output tensors, got {len(out)}")
if any(t.dtype not in (torch.bfloat16, torch.float32) for t in out.values()):
    fail("unexpected dtype in output")

# Greedy sharding in key order; a tensor over budget gets its own shard.
shards, cur, cur_size = [], {}, 0
for name, t in out.items():
    nb = t.numel() * t.element_size()
    if cur and cur_size + nb > MAX_SHARD:
        shards.append(cur)
        cur, cur_size = {}, 0
    cur[name] = t
    cur_size += nb
if cur:
    shards.append(cur)

os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    if f.endswith(".safetensors") or f == "model.safetensors.index.json":
        fail(f"output {OUT}/{f} already exists; refusing to overwrite")

weight_map, total = {}, 0
n = len(shards)
for i, shard in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    size = sum(t.numel() * t.element_size() for t in shard.values())
    assert size <= MAX_SHARD or len(shard) == 1, (fname, size)
    save_file(shard, os.path.join(OUT, fname), metadata={"format": "pt"})
    for name in shard:
        weight_map[name] = fname
    total += size

with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as fh:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, fh, indent=2)

# Post-write verification
reloaded = {}
for fname in set(weight_map.values()):
    reloaded.update(load_file(os.path.join(OUT, fname)))
assert set(reloaded) == set(out) == set(weight_map), "index/shard key mismatch"
for name, t in out.items():
    assert reloaded[name].dtype == t.dtype and torch.equal(reloaded[name], t), name
print(f"wrote {n} shards, {len(out)} tensors, {total} bytes to {OUT}")
