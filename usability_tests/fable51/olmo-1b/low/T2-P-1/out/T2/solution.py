"""T2: prune attention head 5 from every layer of OLMo-1B-0724-hf."""
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_DIR = os.path.join(ROOT, "inputs", "base")
OUT_PATH = os.path.join(ROOT, "out", "T2", "model.safetensors")

N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN = 16, 16, 128, 2048
PRUNE_HEAD = 5


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# Load all shards.
with open(os.path.join(IN_DIR, "model.safetensors.index.json")) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))
sd = {}
for shard in shards:
    part = load_file(os.path.join(IN_DIR, shard))
    for k in part:
        if k in sd:
            fail(f"duplicate tensor across shards: {k}")
    sd.update(part)
if len(sd) != 114:
    fail(f"expected 114 input tensors, got {len(sd)}")

keep = torch.cat([
    torch.arange(0, PRUNE_HEAD * HEAD_DIM),
    torch.arange((PRUNE_HEAD + 1) * HEAD_DIM, N_HEADS * HEAD_DIM),
])
assert keep.numel() == (N_HEADS - 1) * HEAD_DIM

for i in range(N_LAYERS):
    p = f"model.layers.{i}.self_attn."
    for name in ("q_proj", "k_proj", "v_proj"):
        k = p + name + ".weight"
        w = sd[k]
        if tuple(w.shape) != (HIDDEN, HIDDEN):
            fail(f"{k} unexpected shape {tuple(w.shape)}")
        sd[k] = w.index_select(0, keep).contiguous()
    k = p + "o_proj.weight"
    w = sd[k]
    if tuple(w.shape) != (HIDDEN, HIDDEN):
        fail(f"{k} unexpected shape {tuple(w.shape)}")
    sd[k] = w.index_select(1, keep).contiguous()

# Required checks.
checks = {
    "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
}
for k, shape in checks.items():
    if tuple(sd[k].shape) != shape:
        fail(f"{k} has shape {tuple(sd[k].shape)}, expected {shape}")
if len(sd) != 114:
    fail(f"output has {len(sd)} tensors, expected 114")
for k, v in sd.items():
    if v.dtype != torch.float32:
        fail(f"{k} dtype {v.dtype} != float32")

save_file(sd, OUT_PATH, metadata={"format": "pt"})
print(f"wrote {OUT_PATH} with {len(sd)} tensors")
