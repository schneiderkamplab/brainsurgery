"""T2: remove attention head 5 from every GPT-2 layer at the checkpoint level."""
import os
import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T2/model.safetensors"
N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN = 12, 12, 64, 768
PRUNE = 5

keep_heads = [h for h in range(N_HEADS) if h != PRUNE]
keep_idx = torch.tensor([h * HEAD_DIM + d for h in keep_heads for d in range(HEAD_DIM)])
# fused q|k|v: same head index pattern inside each 768-wide segment
qkv_idx = torch.cat([keep_idx + seg * HIDDEN for seg in range(3)])

sd = load_file(SRC)
assert len(sd) == 160, len(sd)
for i in range(N_LAYERS):
    p = f"h.{i}.attn."
    w = sd[p + "c_attn.weight"]; assert w.shape == (HIDDEN, 3 * HIDDEN), w.shape
    b = sd[p + "c_attn.bias"];   assert b.shape == (3 * HIDDEN,), b.shape
    o = sd[p + "c_proj.weight"]; assert o.shape == (HIDDEN, HIDDEN), o.shape
    sd[p + "c_attn.weight"] = w.index_select(1, qkv_idx).contiguous()
    sd[p + "c_attn.bias"] = b.index_select(0, qkv_idx).contiguous()
    sd[p + "c_proj.weight"] = o.index_select(0, keep_idx).contiguous()

# Required checks (fail loudly before writing)
assert tuple(sd["h.0.attn.c_attn.weight"].shape) == (768, 2112), sd["h.0.attn.c_attn.weight"].shape
assert tuple(sd["h.0.attn.c_attn.bias"].shape) == (2112,), sd["h.0.attn.c_attn.bias"].shape
assert tuple(sd["h.0.attn.c_proj.weight"].shape) == (704, 768), sd["h.0.attn.c_proj.weight"].shape
assert len(sd) == 160, len(sd)
for i in range(N_LAYERS):
    assert tuple(sd[f"h.{i}.attn.c_attn.weight"].shape) == (768, 2112)
    assert tuple(sd[f"h.{i}.attn.c_attn.bias"].shape) == (2112,)
    assert tuple(sd[f"h.{i}.attn.c_proj.weight"].shape) == (704, 768)

if os.path.exists(DST):
    raise SystemExit(f"destination exists: {DST}")
save_file(sd, DST)
out = load_file(DST)
assert len(out) == 160
print("wrote", DST, "with", len(out), "tensors")
