"""T3: mixed-precision export of GPT-2 with sharding."""
import json
import os
import re

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
OUT = "out/T3"
SHARD_BUDGET = 64 * 1024 * 1024  # 64 MiB of tensor data
NUM_LAYERS = 12

PROJ_RE = re.compile(r"^h\.\d+\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$")
BUFFER_RE = re.compile(r"^h\.\d+\.attn\.bias$")

sd = load_file(SRC)
assert len(sd) == 160, f"expected 160 input tensors, got {len(sd)}"

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
        assert t.dtype == torch.float32, f"{name} unexpected dtype {t.dtype}"
        out[name] = t.contiguous()

# Required checks
n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
assert n_bf16 == 48, f"expected 48 bfloat16 tensors, got {n_bf16}"
assert n_cast == 48 and n_dropped == NUM_LAYERS, (n_cast, n_dropped)
assert out["h.0.attn.c_attn.weight"].dtype == torch.bfloat16
assert out["wte.weight"].dtype == torch.float32
assert len(out) == 148, f"expected 148 output tensors, got {len(out)}"
assert all(t.dtype in (torch.float32, torch.bfloat16) for t in out.values())
# every non-projection tensor must still be float32
for name, t in out.items():
    if not PROJ_RE.match(name):
        assert t.dtype == torch.float32, name

# Shard: greedy in original key order; a tensor exceeding the budget gets its own shard.
shards = []  # list of (names, nbytes)
cur, cur_bytes = [], 0
for name, t in out.items():
    nb = t.numel() * t.element_size()
    if nb > SHARD_BUDGET:
        if cur:
            shards.append((cur, cur_bytes))
            cur, cur_bytes = [], 0
        shards.append(([name], nb))
        continue
    if cur and cur_bytes + nb > SHARD_BUDGET:
        shards.append((cur, cur_bytes))
        cur, cur_bytes = [], 0
    cur.append(name)
    cur_bytes += nb
if cur:
    shards.append((cur, cur_bytes))

for names, nb in shards:
    assert nb <= SHARD_BUDGET or len(names) == 1, (names[:3], nb)

os.makedirs(OUT, exist_ok=True)
total = len(shards)
weight_map = {}
total_size = 0
for i, (names, nb) in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{total:05d}.safetensors"
    save_file({n: out[n] for n in names}, os.path.join(OUT, fname), metadata={"format": "pt"})
    for n in names:
        weight_map[n] = fname
    total_size += nb

assert len(weight_map) == 148
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

# Verify round trip
check = {}
for fname in sorted(set(weight_map.values())):
    check.update(load_file(os.path.join(OUT, fname)))
assert len(check) == 148
for n, t in out.items():
    assert torch.equal(check[n], t), n
print(f"wrote {total} shards, {len(weight_map)} tensors, {total_size} bytes")
