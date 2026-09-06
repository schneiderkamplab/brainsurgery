"""T3: mixed-precision export with sharding for Pythia-1B."""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_PATH = "inputs/base/model.safetensors"
OUT_DIR = "out/T3"
MAX_SHARD_BYTES = 256 * 1024 * 1024

# The four projection matrices per layer, and nothing else.
PROJ_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\."
    r"(attention\.(query_key_value|dense)|mlp\.(dense_h_to_4h|dense_4h_to_h))"
    r"\.weight$"
)
BUFFER_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)

with safe_open(IN_PATH, framework="pt") as f:
    keys = list(f.keys())
    tensors = {k: f.get_tensor(k) for k in keys}

print(f"loaded {len(tensors)} tensors")

out = {}
for name in keys:
    if BUFFER_RE.match(name):
        continue
    t = tensors[name]
    if PROJ_RE.match(name):
        out[name] = t.to(torch.float32).to(torch.bfloat16).contiguous()
    else:
        out[name] = t.to(torch.float32).contiguous()

# ---- required checks, before writing anything ----
n_bf16 = sum(1 for t in out.values() if t.dtype is torch.bfloat16)
if n_bf16 != 64:
    raise SystemExit(f"CHECK FAILED: expected 64 bfloat16 tensors, got {n_bf16}")

probe = "gpt_neox.layers.0.attention.query_key_value.weight"
if out[probe].dtype is not torch.bfloat16:
    raise SystemExit(f"CHECK FAILED: {probe} is {out[probe].dtype}, expected bfloat16")

if out["gpt_neox.embed_in.weight"].dtype is not torch.float32:
    raise SystemExit(
        f"CHECK FAILED: embed_in.weight is {out['gpt_neox.embed_in.weight'].dtype}, "
        "expected float32"
    )

if len(out) != 196:
    raise SystemExit(f"CHECK FAILED: expected 196 output tensors, got {len(out)}")

n_dropped = len(tensors) - len(out)
if n_dropped != 48:
    raise SystemExit(f"CHECK FAILED: expected to drop 48 buffers, dropped {n_dropped}")

for name, t in out.items():
    if t.dtype not in (torch.float32, torch.bfloat16):
        raise SystemExit(f"CHECK FAILED: {name} has unexpected dtype {t.dtype}")

# ---- shard: greedy fill in key order; an oversized tensor lands alone ----
shards = []
cur, cur_bytes = {}, 0
for name, t in out.items():
    nbytes = t.numel() * t.element_size()
    if cur and cur_bytes + nbytes > MAX_SHARD_BYTES:
        shards.append(cur)
        cur, cur_bytes = {}, 0
    cur[name] = t
    cur_bytes += nbytes
if cur:
    shards.append(cur)

total = len(shards)
weight_map = {}
os.makedirs(OUT_DIR, exist_ok=True)
for i, shard in enumerate(shards, start=1):
    fname = f"model-{i:05d}-of-{total:05d}.safetensors"
    size = sum(t.numel() * t.element_size() for t in shard.values())
    if size > MAX_SHARD_BYTES and len(shard) > 1:
        raise SystemExit(f"CHECK FAILED: shard {fname} is {size} bytes with {len(shard)} tensors")
    save_file(shard, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
    for name in shard:
        weight_map[name] = fname
    print(f"{fname}: {len(shard)} tensors, {size} bytes")

if len(weight_map) != 196:
    raise SystemExit(f"CHECK FAILED: weight_map has {len(weight_map)} entries, expected 196")

index = {
    "metadata": {"total_size": sum(t.numel() * t.element_size() for t in out.values())},
    "weight_map": weight_map,
}
with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2, sort_keys=True)

print(f"wrote {total} shards + index to {OUT_DIR}/")
