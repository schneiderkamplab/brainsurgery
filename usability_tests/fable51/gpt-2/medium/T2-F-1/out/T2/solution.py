"""T2: remove attention head 5 from every layer of GPT-2 (124M) at the checkpoint level.

Plain torch + safetensors: slice the head blocks out of the fused c_attn
projection (columns / bias entries) and the c_proj output projection (rows),
copy every other tensor verbatim, check shapes, then write out/T2/model.safetensors.
"""
import os
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T2/model.safetensors"

N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN = 12, 12, 64, 768
PRUNE_HEAD = 5

# Indices kept within one 768-wide head segment: heads 0..4 and 6..11.
keep_in_segment = torch.cat(
    [torch.arange(0, PRUNE_HEAD * HEAD_DIM),
     torch.arange((PRUNE_HEAD + 1) * HEAD_DIM, N_HEADS * HEAD_DIM)]
)
assert keep_in_segment.tolist() == list(range(0, 320)) + list(range(384, 768))
# Same pattern repeated in each of the q, k, v segments of the fused projection.
keep_qkv = torch.cat([keep_in_segment + s * HIDDEN for s in range(3)])
assert keep_qkv.numel() == 3 * (N_HEADS - 1) * HEAD_DIM == 2112

sd = load_file(SRC)
if len(sd) != 160:
    sys.exit(f"unexpected input tensor count {len(sd)}")

out = {}
touched = 0
for name, t in sd.items():
    if name.endswith(".attn.c_attn.weight"):
        assert t.shape == (HIDDEN, 3 * HIDDEN), (name, t.shape)
        t = t.index_select(1, keep_qkv)
    elif name.endswith(".attn.c_attn.bias"):
        assert t.shape == (3 * HIDDEN,), (name, t.shape)
        t = t.index_select(0, keep_qkv)
    elif name.endswith(".attn.c_proj.weight"):
        assert t.shape == (HIDDEN, HIDDEN), (name, t.shape)
        t = t.index_select(0, keep_in_segment)
    else:
        out[name] = t.contiguous()
        continue
    touched += 1
    out[name] = t.contiguous()

# Required checks (fail loudly before writing).
if touched != 3 * N_LAYERS:
    sys.exit(f"expected to touch {3 * N_LAYERS} tensors, touched {touched}")
for i in range(N_LAYERS):
    expected = {
        f"h.{i}.attn.c_attn.weight": (HIDDEN, 2112),
        f"h.{i}.attn.c_attn.bias": (2112,),
        f"h.{i}.attn.c_proj.weight": (704, HIDDEN),
    }
    for k, shape in expected.items():
        if tuple(out[k].shape) != shape:
            sys.exit(f"{k}: shape {tuple(out[k].shape)} != {shape}")
        if out[k].dtype != sd[k].dtype:
            sys.exit(f"{k}: dtype changed")
# Spot-check block order on layer 0: kept columns equal the original ones.
w0, ow0 = sd["h.0.attn.c_attn.weight"], out["h.0.attn.c_attn.weight"]
assert torch.equal(ow0[:, :320], w0[:, :320])
assert torch.equal(ow0[:, 320:704], w0[:, 384:768])
assert torch.equal(ow0[:, 704:1024], w0[:, 768:1088])
assert torch.equal(ow0[:, 1408:1728], w0[:, 1536:1856])
assert torch.equal(ow0[:, 1728:], w0[:, 1920:])
p0, op0 = sd["h.0.attn.c_proj.weight"], out["h.0.attn.c_proj.weight"]
assert torch.equal(op0[:320], p0[:320]) and torch.equal(op0[320:], p0[384:])
if set(out) != set(sd) or len(out) != 160:
    sys.exit(f"key set changed: {len(out)} tensors")

os.makedirs(os.path.dirname(DST), exist_ok=True)
save_file(out, DST)

# Post-write verification of the file on disk.
chk = load_file(DST)
assert len(chk) == 160, len(chk)
assert tuple(chk["h.0.attn.c_attn.weight"].shape) == (768, 2112)
assert tuple(chk["h.0.attn.c_attn.bias"].shape) == (2112,)
assert tuple(chk["h.0.attn.c_proj.weight"].shape) == (704, 768)
for k in sd:
    if ".attn.c_attn." not in k and not k.endswith(".attn.c_proj.weight"):
        assert torch.equal(chk[k], sd[k]), k
print(f"wrote {DST}: {len(chk)} tensors, {touched} pruned")
