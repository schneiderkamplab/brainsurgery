"""T2: prune attention head 5 from every layer of OLMo-1B-0724-hf (plain safetensors + torch)."""
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

BASE = "inputs/base"
OUT = "out/T2/model.safetensors"
HEAD, HEAD_DIM, N_LAYERS, HIDDEN = 5, 128, 16, 2048
KEEP = torch.cat([torch.arange(0, HEAD * HEAD_DIM),
                  torch.arange((HEAD + 1) * HEAD_DIM, HIDDEN)])  # 0..639, 768..2047
assert KEEP.numel() == 1920

with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
    index = json.load(f)
sd = {}
for shard in sorted(set(index["weight_map"].values())):
    sd.update(load_file(os.path.join(BASE, shard)))
assert len(sd) == 114, f"expected 114 input tensors, got {len(sd)}"

for i in range(N_LAYERS):
    p = f"model.layers.{i}.self_attn."
    for name in ("q_proj", "k_proj", "v_proj"):
        k = p + name + ".weight"
        assert sd[k].shape == (HIDDEN, HIDDEN), (k, sd[k].shape)
        sd[k] = sd[k][KEEP, :].contiguous()
    k = p + "o_proj.weight"
    assert sd[k].shape == (HIDDEN, HIDDEN), (k, sd[k].shape)
    sd[k] = sd[k][:, KEEP].contiguous()

# Required checks (fail loudly before writing).
def check(k, shape):
    if tuple(sd[k].shape) != shape:
        sys.exit(f"CHECK FAILED: {k} shape {tuple(sd[k].shape)} != {shape}")
check("model.layers.0.self_attn.q_proj.weight", (1920, 2048))
check("model.layers.0.self_attn.k_proj.weight", (1920, 2048))
check("model.layers.0.self_attn.v_proj.weight", (1920, 2048))
check("model.layers.0.self_attn.o_proj.weight", (2048, 1920))
if len(sd) != 114:
    sys.exit(f"CHECK FAILED: {len(sd)} tensors != 114")
for k, t in sd.items():
    assert t.dtype == torch.float32, (k, t.dtype)

save_file(sd, OUT, metadata={"format": "pt"})
print(f"wrote {OUT} with {len(sd)} tensors")
