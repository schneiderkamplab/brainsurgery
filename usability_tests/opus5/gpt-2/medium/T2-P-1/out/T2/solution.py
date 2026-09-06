"""T2: structured attention-head pruning for GPT-2 (124M).

Removes head 5 from every layer at the checkpoint level. GPT-2 uses Conv1D
projections stored as [in, out], so for the fused c_attn the heads are column
blocks inside each of the three 768-wide q/k/v segments, and for the output
c_proj the heads are row blocks.
"""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T2/model.safetensors")

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
PRUNE_HEAD = 5


def keep_index(n_segments: int) -> torch.Tensor:
    """Indices to keep along a head-bearing axis of `n_segments` q/k/v blocks."""
    keep = []
    for seg in range(n_segments):
        base = seg * HIDDEN
        for head in range(N_HEADS):
            if head == PRUNE_HEAD:
                continue
            start = base + head * HEAD_DIM
            keep.extend(range(start, start + HEAD_DIM))
    return torch.tensor(keep, dtype=torch.long)


QKV_KEEP = keep_index(3)  # c_attn: 2304 -> 2112
OUT_KEEP = keep_index(1)  # c_proj: 768 -> 704

# Sanity-check the index sets against the ranges spelled out in the task.
expected_qkv = [
    r
    for a, b in [(0, 319), (384, 767), (768, 1087), (1152, 1535), (1536, 1855), (1920, 2303)]
    for r in range(a, b + 1)
]
expected_out = [r for a, b in [(0, 319), (384, 767)] for r in range(a, b + 1)]
assert QKV_KEEP.tolist() == expected_qkv, "c_attn keep-index does not match the spec"
assert OUT_KEEP.tolist() == expected_out, "c_proj keep-index does not match the spec"

with safe_open(SRC, framework="pt") as f:
    metadata = f.metadata()
    names = list(f.keys())
    tensors = {name: f.get_tensor(name) for name in names}

n_in = len(tensors)
print(f"loaded {n_in} tensors from {SRC}")

out = {}
touched = 0
for name, t in tensors.items():
    parts = name.split(".")
    is_layer_attn = (
        len(parts) == 5
        and parts[0] == "h"
        and parts[1].isdigit()
        and parts[2] == "attn"
        and parts[3] in ("c_attn", "c_proj")
    )
    if is_layer_attn and parts[3] == "c_attn" and parts[4] in ("weight", "bias"):
        # weight is [768, 2304] (slice columns); bias is [2304] (slice rows)
        axis = t.dim() - 1
        new = t.index_select(axis, QKV_KEEP)
        touched += 1
    elif is_layer_attn and parts[3] == "c_proj" and parts[4] == "weight":
        # [768, 768] -> [704, 768]; heads are row blocks
        new = t.index_select(0, OUT_KEEP)
        touched += 1
    else:
        new = t
    out[name] = new.contiguous().clone()

expected_touched = N_LAYERS * 3
assert touched == expected_touched, f"pruned {touched} tensors, expected {expected_touched}"

# Required checks: fail loudly before writing anything.
for i in range(N_LAYERS):
    w = out[f"h.{i}.attn.c_attn.weight"].shape
    b = out[f"h.{i}.attn.c_attn.bias"].shape
    p = out[f"h.{i}.attn.c_proj.weight"].shape
    assert tuple(w) == (768, 2112), f"h.{i}.attn.c_attn.weight has shape {tuple(w)}, want (768, 2112)"
    assert tuple(b) == (2112,), f"h.{i}.attn.c_attn.bias has shape {tuple(b)}, want (2112,)"
    assert tuple(p) == (704, 768), f"h.{i}.attn.c_proj.weight has shape {tuple(p)}, want (704, 768)"

assert len(out) == 160, f"output has {len(out)} tensors, expected 160"
assert set(out) == set(tensors), "tensor names changed"
for name, t in out.items():
    assert t.dtype == tensors[name].dtype, f"{name}: dtype changed"
    if name not in {
        f"h.{i}.attn.{proj}.{suf}"
        for i in range(N_LAYERS)
        for proj, suf in (("c_attn", "weight"), ("c_attn", "bias"), ("c_proj", "weight"))
    }:
        assert torch.equal(t, tensors[name]), f"{name}: should have been left unchanged"

# Spot-check the values actually landed where they should for one layer.
ref = tensors["h.0.attn.c_attn.weight"]
got = out["h.0.attn.c_attn.weight"]
assert torch.equal(got[:, 0:320], ref[:, 0:320])
assert torch.equal(got[:, 320:704], ref[:, 384:768])
assert torch.equal(got[:, 704:1024], ref[:, 768:1088])
ref_p = tensors["h.0.attn.c_proj.weight"]
got_p = out["h.0.attn.c_proj.weight"]
assert torch.equal(got_p[0:320], ref_p[0:320])
assert torch.equal(got_p[320:704], ref_p[384:768])

DST.parent.mkdir(parents=True, exist_ok=True)
save_file(out, DST, metadata=metadata)
print(f"wrote {len(out)} tensors to {DST} (pruned head {PRUNE_HEAD} in {N_LAYERS} layers)")
