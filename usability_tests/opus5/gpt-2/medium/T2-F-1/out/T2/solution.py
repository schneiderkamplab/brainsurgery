"""T2: remove attention head 5 from every layer of GPT-2 (124M).

Plain torch + safetensors. GPT-2 uses Conv1D, so projections are stored as
[in, out]: heads are column blocks of c_attn.weight (three 768-wide q|k|v
segments) and row blocks of attn.c_proj.weight.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T2/model.safetensors")

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
PRUNE = 5

# Indices to keep inside one 768-wide segment: 0..319, 384..767.
keep_seg = torch.cat(
    [
        torch.arange(0, PRUNE * HEAD_DIM),
        torch.arange((PRUNE + 1) * HEAD_DIM, HIDDEN),
    ]
)
# Same, repeated for the fused q|k|v segments of c_attn.
keep_qkv = torch.cat([keep_seg + s * HIDDEN for s in range(3)])

assert keep_seg.numel() == 704, keep_seg.numel()
assert keep_qkv.numel() == 2112, keep_qkv.numel()

sd = load_file(str(SRC))
n_in = len(sd)
if n_in != 160:
    raise SystemExit(f"expected 160 input tensors, got {n_in}")

out = {}
touched = 0
for name, t in sd.items():
    parts = name.split(".")
    is_attn_layer = len(parts) >= 4 and parts[0] == "h" and parts[2] == "attn"
    leaf = ".".join(parts[3:]) if is_attn_layer else ""
    if leaf == "c_attn.weight":
        new = t.index_select(1, keep_qkv)
    elif leaf == "c_attn.bias":
        new = t.index_select(0, keep_qkv)
    elif leaf == "c_proj.weight":
        new = t.index_select(0, keep_seg)
    else:
        out[name] = t
        continue
    touched += 1
    out[name] = new.contiguous().clone()

expected_touched = 3 * N_LAYERS
if touched != expected_touched:
    raise SystemExit(f"expected to edit {expected_touched} tensors, edited {touched}")

# Required checks: fail loudly before writing.
checks = {
    "h.0.attn.c_attn.weight": (768, 2112),
    "h.0.attn.c_attn.bias": (2112,),
    "h.0.attn.c_proj.weight": (704, 768),
}
for key, shape in checks.items():
    got = tuple(out[key].shape)
    if got != shape:
        raise SystemExit(f"{key}: expected shape {shape}, got {got}")
if len(out) != 160:
    raise SystemExit(f"expected 160 output tensors, got {len(out)}")

# Every layer, not just layer 0; and nothing else changed.
for i in range(N_LAYERS):
    for key, shape in (
        (f"h.{i}.attn.c_attn.weight", (768, 2112)),
        (f"h.{i}.attn.c_attn.bias", (2112,)),
        (f"h.{i}.attn.c_proj.weight", (704, 768)),
    ):
        got = tuple(out[key].shape)
        if got != shape:
            raise SystemExit(f"{key}: expected shape {shape}, got {got}")
edited = {
    f"h.{i}.attn.{leaf}"
    for i in range(N_LAYERS)
    for leaf in ("c_attn.weight", "c_attn.bias", "c_proj.weight")
}
for name, t in sd.items():
    if name in edited:
        continue
    if out[name].shape != t.shape or out[name].dtype != t.dtype:
        raise SystemExit(f"{name}: unexpectedly modified")
    if not torch.equal(out[name], t):
        raise SystemExit(f"{name}: values unexpectedly modified")

DST.parent.mkdir(parents=True, exist_ok=True)
save_file(out, str(DST))
print(f"wrote {DST} with {len(out)} tensors ({touched} pruned)")
