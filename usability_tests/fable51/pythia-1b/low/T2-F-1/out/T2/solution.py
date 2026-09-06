"""T2: remove attention head 5 from every layer of Pythia-1B (GPT-NeoX layout)."""
import os
import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T2/model.safetensors"
LAYERS, HEADS, HEAD_DIM, PRUNE = 16, 8, 256, 5

sd = load_file(SRC)
assert len(sd) == 244, len(sd)

keep_heads = [h for h in range(HEADS) if h != PRUNE]
qkv_rows = torch.cat([torch.arange(768 * h, 768 * h + 768) for h in keep_heads])
dense_cols = torch.cat([torch.arange(HEAD_DIM * h, HEAD_DIM * h + HEAD_DIM) for h in keep_heads])

for i in range(LAYERS):
    p = f"gpt_neox.layers.{i}.attention."
    w, b, d = p + "query_key_value.weight", p + "query_key_value.bias", p + "dense.weight"
    assert sd[w].shape == (6144, 2048) and sd[b].shape == (6144,) and sd[d].shape == (2048, 2048)
    sd[w] = sd[w][qkv_rows].contiguous()
    sd[b] = sd[b][qkv_rows].contiguous()
    sd[d] = sd[d][:, dense_cols].contiguous()

# Required checks: fail loudly before writing.
l0 = "gpt_neox.layers.0.attention."
assert sd[l0 + "query_key_value.weight"].shape == (5376, 2048), sd[l0 + "query_key_value.weight"].shape
assert sd[l0 + "query_key_value.bias"].shape == (5376,), sd[l0 + "query_key_value.bias"].shape
assert sd[l0 + "dense.weight"].shape == (2048, 1792), sd[l0 + "dense.weight"].shape
assert len(sd) == 244, len(sd)
for i in range(LAYERS):
    p = f"gpt_neox.layers.{i}.attention."
    assert sd[p + "query_key_value.weight"].shape == (5376, 2048)
    assert sd[p + "query_key_value.bias"].shape == (5376,)
    assert sd[p + "dense.weight"].shape == (2048, 1792)

save_file(sd, DST, metadata={"format": "pt"})
print("wrote", DST, "tensors:", len(load_file(DST)))
