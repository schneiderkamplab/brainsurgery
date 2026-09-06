"""T3: mixed-precision export with sharding for OLMo-1B-0724-hf."""

import json
import os
import re
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file

IN_DIR = "inputs/base"
OUT_DIR = "out/T3"
SHARD_LIMIT = 268_435_456  # 256 MiB of tensor data

CAST_RE = re.compile(
    r"^model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)

index_in = json.load(open(os.path.join(IN_DIR, "model.safetensors.index.json")))
weight_map_in = index_in["weight_map"]

tensors = {}
for shard in sorted(set(weight_map_in.values())):
    tensors.update(load_file(os.path.join(IN_DIR, shard)))

assert len(tensors) == 114, f"expected 114 input tensors, got {len(tensors)}"

out = OrderedDict()
for name in sorted(tensors):
    t = tensors[name]
    out[name] = t.to(torch.bfloat16) if CAST_RE.match(name) else t.to(torch.float32)

# --- required checks -------------------------------------------------------
n_bf16 = sum(1 for t in out.values() if t.dtype is torch.bfloat16)
assert n_bf16 == 112, f"expected 112 bfloat16 tensors, got {n_bf16}"
assert out["model.layers.0.self_attn.q_proj.weight"].dtype is torch.bfloat16
assert out["model.embed_tokens.weight"].dtype is torch.float32
assert len(out) == 114, f"expected 114 output tensors, got {len(out)}"

# --- shard (greedy, sorted key order; oversized tensor alone) --------------
shards, current, current_size = [], OrderedDict(), 0
for name, t in out.items():
    nbytes = t.numel() * t.element_size()
    if current and current_size + nbytes > SHARD_LIMIT:
        shards.append(current)
        current, current_size = OrderedDict(), 0
    current[name] = t
    current_size += nbytes
if current:
    shards.append(current)

n = len(shards)
os.makedirs(OUT_DIR, exist_ok=True)
weight_map, total_size = {}, 0
for i, shard in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    for name, t in shard.items():
        weight_map[name] = fname
        total_size += t.numel() * t.element_size()
    save_file({k: v.contiguous() for k, v in shard.items()},
              os.path.join(OUT_DIR, fname), metadata={"format": "pt"})

json.dump(
    {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w"),
    indent=2,
)

assert len(weight_map) == 114
print(f"wrote {n} shards, {len(weight_map)} tensors, {total_size} bytes")
