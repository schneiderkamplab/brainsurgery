"""T3: mixed-precision export of GPT-2 (124M) with 64 MiB sharding."""

import json
import os
import re

import torch
from safetensors.torch import load_file, save_file

IN_PATH = "inputs/base/model.safetensors"
OUT_DIR = "out/T3"
SHARD_LIMIT = 64 * 1024 * 1024  # 67,108,864 bytes of tensor data

PROJ_RE = re.compile(r"^h\.\d+\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$")
BUFFER_RE = re.compile(r"^h\.\d+\.attn\.bias$")

state = load_file(IN_PATH)

out = {}
for name, t in state.items():
    if BUFFER_RE.match(name):
        continue  # non-parameter causal-mask buffer
    if PROJ_RE.match(name):
        out[name] = t.to(torch.bfloat16)
    else:
        out[name] = t.to(torch.float32)

# --- required checks, before writing ---
bf16 = [n for n, t in out.items() if t.dtype is torch.bfloat16]
assert len(bf16) == 48, f"expected 48 bfloat16 tensors, got {len(bf16)}"
assert out["h.0.attn.c_attn.weight"].dtype is torch.bfloat16, "h.0.attn.c_attn.weight not bfloat16"
assert out["wte.weight"].dtype is torch.float32, "wte.weight not float32"
assert len(out) == 148, f"expected 148 tensors, got {len(out)}"
dropped = set(state) - set(out)
assert all(BUFFER_RE.match(n) for n in dropped) and len(dropped) == 12, f"bad deletions: {dropped}"

# --- greedy sharding in original key order ---
names = list(out.keys())
sizes = {n: out[n].numel() * out[n].element_size() for n in names}

shards, cur, cur_bytes = [], [], 0
for n in names:
    s = sizes[n]
    if cur and cur_bytes + s > SHARD_LIMIT:
        shards.append(cur)
        cur, cur_bytes = [], 0
    cur.append(n)
    cur_bytes += s
if cur:
    shards.append(cur)

total = len(shards)
os.makedirs(OUT_DIR, exist_ok=True)
weight_map = {}
for i, keys in enumerate(shards, start=1):
    fname = f"model-{i:05d}-of-{total:05d}.safetensors"
    save_file({k: out[k].contiguous() for k in keys}, os.path.join(OUT_DIR, fname))
    for k in keys:
        weight_map[k] = fname

index = {
    "metadata": {"total_size": sum(sizes.values())},
    "weight_map": weight_map,
}
with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as fh:
    json.dump(index, fh, indent=2, sort_keys=True)
    fh.write("\n")

print(f"wrote {len(out)} tensors into {total} shards ({sum(sizes.values())} bytes)")
for i, keys in enumerate(shards, start=1):
    print(f"  shard {i}: {len(keys)} tensors, {sum(sizes[k] for k in keys)} bytes")
