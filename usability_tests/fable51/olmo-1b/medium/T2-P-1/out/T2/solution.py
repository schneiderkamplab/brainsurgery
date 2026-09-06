"""T2: prune head 5 from every layer of OLMo-1B-0724-hf (16 -> 15 heads)."""
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

BASE = "inputs/base"
OUT = "out/T2/model.safetensors"
N_LAYERS, HEAD_DIM, PRUNE_HEAD = 16, 128, 5
KEEP = torch.cat(
    [torch.arange(0, PRUNE_HEAD * HEAD_DIM), torch.arange((PRUNE_HEAD + 1) * HEAD_DIM, 2048)]
)  # rows/cols 0..639, 768..2047


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# Load all shards into a single state dict.
with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
    index = json.load(f)
sd = {}
for shard in sorted(set(index["weight_map"].values())):
    sd.update(load_file(os.path.join(BASE, shard)))
if len(sd) != 114:
    fail(f"expected 114 input tensors, got {len(sd)}")

for i in range(N_LAYERS):
    p = f"model.layers.{i}.self_attn."
    for name in ("q_proj", "k_proj", "v_proj"):
        k = p + name + ".weight"
        w = sd[k]
        if tuple(w.shape) != (2048, 2048):
            fail(f"{k} unexpected shape {tuple(w.shape)}")
        sd[k] = w.index_select(0, KEEP).contiguous()
    k = p + "o_proj.weight"
    w = sd[k]
    if tuple(w.shape) != (2048, 2048):
        fail(f"{k} unexpected shape {tuple(w.shape)}")
    sd[k] = w.index_select(1, KEEP).contiguous()

# Required checks.
exp = {
    "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
}
for k, shape in exp.items():
    if tuple(sd[k].shape) != shape:
        fail(f"{k} has shape {tuple(sd[k].shape)}, expected {shape}")
# Also check every layer, not just layer 0.
for i in range(N_LAYERS):
    p = f"model.layers.{i}.self_attn."
    for name in ("q_proj", "k_proj", "v_proj"):
        if tuple(sd[p + name + ".weight"].shape) != (1920, 2048):
            fail(f"layer {i} {name} wrong shape")
    if tuple(sd[p + "o_proj.weight"].shape) != (2048, 1920):
        fail(f"layer {i} o_proj wrong shape")
if len(sd) != 114:
    fail(f"output has {len(sd)} tensors, expected 114")

save_file({k: v.contiguous() for k, v in sd.items()}, OUT, metadata={"format": "pt"})
print(f"wrote {OUT} with {len(sd)} tensors")
