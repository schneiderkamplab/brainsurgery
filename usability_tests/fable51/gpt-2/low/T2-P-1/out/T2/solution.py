"""T2: remove head 5 from every GPT-2 layer at the checkpoint level."""
import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T2/model.safetensors"
N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN = 12, 12, 64, 768
PRUNE = 5

# Column/row indices kept within one 768-wide head-block segment.
keep_seg = torch.cat([
    torch.arange(0, PRUNE * HEAD_DIM),
    torch.arange((PRUNE + 1) * HEAD_DIM, HIDDEN),
])
assert keep_seg.numel() == (N_HEADS - 1) * HEAD_DIM
keep_qkv = torch.cat([keep_seg + s * HIDDEN for s in range(3)])  # q | k | v segments

sd = load_file(SRC)
assert len(sd) == 160, f"expected 160 input tensors, got {len(sd)}"

for i in range(N_LAYERS):
    w = f"h.{i}.attn.c_attn.weight"
    b = f"h.{i}.attn.c_attn.bias"
    o = f"h.{i}.attn.c_proj.weight"
    assert sd[w].shape == (HIDDEN, 3 * HIDDEN), (w, sd[w].shape)
    assert sd[b].shape == (3 * HIDDEN,), (b, sd[b].shape)
    assert sd[o].shape == (HIDDEN, HIDDEN), (o, sd[o].shape)
    sd[w] = sd[w][:, keep_qkv].contiguous()   # Conv1D [in, out]: heads on columns
    sd[b] = sd[b][keep_qkv].contiguous()
    sd[o] = sd[o][keep_seg, :].contiguous()   # output proj: heads on rows

# Required checks.
assert sd["h.0.attn.c_attn.weight"].shape == (768, 2112), sd["h.0.attn.c_attn.weight"].shape
assert sd["h.0.attn.c_attn.bias"].shape == (2112,), sd["h.0.attn.c_attn.bias"].shape
assert sd["h.0.attn.c_proj.weight"].shape == (704, 768), sd["h.0.attn.c_proj.weight"].shape
assert len(sd) == 160, len(sd)
for i in range(N_LAYERS):
    assert sd[f"h.{i}.attn.c_attn.weight"].shape == (768, 2112)
    assert sd[f"h.{i}.attn.c_attn.bias"].shape == (2112,)
    assert sd[f"h.{i}.attn.c_proj.weight"].shape == (704, 768)
    assert sd[f"h.{i}.attn.c_proj.bias"].shape == (768,)

save_file(sd, DST)
out = load_file(DST)
assert len(out) == 160, len(out)
print(f"wrote {DST} with {len(out)} tensors")
