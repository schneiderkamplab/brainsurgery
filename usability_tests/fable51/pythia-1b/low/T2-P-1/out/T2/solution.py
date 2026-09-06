"""T2: remove attention head 5 from every layer of Pythia-1B (GPT-NeoX layout)."""
import os
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T2/model.safetensors"
N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN, PRUNE = 16, 8, 256, 2048, 5
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head: q | k | v


def check(cond, msg):
    if not cond:
        print(f"CHECK FAILED: {msg}", file=sys.stderr)
        sys.exit(1)


sd = load_file(SRC)
check(len(sd) == 244, f"input has {len(sd)} tensors, expected 244")

keep_qkv = [h for h in range(N_HEADS) if h != PRUNE]
qkv_rows = torch.cat([torch.arange(h * QKV_BLOCK, (h + 1) * QKV_BLOCK) for h in keep_qkv])
dense_cols = torch.cat([torch.arange(h * HEAD_DIM, (h + 1) * HEAD_DIM) for h in keep_qkv])

for i in range(N_LAYERS):
    p = f"gpt_neox.layers.{i}.attention."
    w, b, d = sd[p + "query_key_value.weight"], sd[p + "query_key_value.bias"], sd[p + "dense.weight"]
    check(tuple(w.shape) == (N_HEADS * QKV_BLOCK, HIDDEN), f"{p}qkv.weight shape {tuple(w.shape)}")
    check(tuple(b.shape) == (N_HEADS * QKV_BLOCK,), f"{p}qkv.bias shape {tuple(b.shape)}")
    check(tuple(d.shape) == (HIDDEN, HIDDEN), f"{p}dense.weight shape {tuple(d.shape)}")
    sd[p + "query_key_value.weight"] = w[qkv_rows].contiguous()
    sd[p + "query_key_value.bias"] = b[qkv_rows].contiguous()
    sd[p + "dense.weight"] = d[:, dense_cols].contiguous()

# Required checks
p0 = "gpt_neox.layers.0.attention."
check(tuple(sd[p0 + "query_key_value.weight"].shape) == (5376, 2048), "layer0 qkv.weight shape")
check(tuple(sd[p0 + "query_key_value.bias"].shape) == (5376,), "layer0 qkv.bias shape")
check(tuple(sd[p0 + "dense.weight"].shape) == (2048, 1792), "layer0 dense.weight shape")
check(len(sd) == 244, f"output has {len(sd)} tensors, expected 244")

os.makedirs(os.path.dirname(DST), exist_ok=True)
save_file(sd, DST, metadata={"format": "pt"})

out = load_file(DST)
check(len(out) == 244, "reloaded output tensor count")
print(f"OK: wrote {DST} with {len(out)} tensors")
