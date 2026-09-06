"""T2: remove attention head 5 from every layer of GPT-2 (124M).

GPT-2 Conv1D layout: weights are [in, out]. In c_attn the output axis is the
fused [q | k | v] projection, 768 wide each, with heads as 64-wide column
blocks inside each segment. In attn.c_proj the head axis is the input (row)
axis.
"""

import os
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T2/model.safetensors"

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
PRUNE_HEAD = 5
KEPT_HEADS = [h for h in range(N_HEADS) if h != PRUNE_HEAD]
NEW_HIDDEN = len(KEPT_HEADS) * HEAD_DIM  # 704


def check(cond, msg):
    if not cond:
        print(f"CHECK FAILED: {msg}", file=sys.stderr)
        sys.exit(1)


def keep_index_within_segment():
    """Indices 0..767 minus head 5's block, in ascending order."""
    return torch.cat([torch.arange(h * HEAD_DIM, (h + 1) * HEAD_DIM) for h in KEPT_HEADS])


def main():
    check(os.path.isfile(SRC), f"missing input {SRC}")
    sd = load_file(SRC)
    check(len(sd) == 160, f"expected 160 input tensors, got {len(sd)}")

    seg_idx = keep_index_within_segment()  # [704]
    # Fused q|k|v: same head removal in each 768-wide segment, segments in order.
    fused_idx = torch.cat([seg_idx + s * HIDDEN for s in range(3)])  # [2112]

    out = {}
    touched = 0
    for name, t in sd.items():
        for i in range(N_LAYERS):
            if name == f"h.{i}.attn.c_attn.weight":
                check(tuple(t.shape) == (HIDDEN, 3 * HIDDEN), f"{name} shape {tuple(t.shape)}")
                t = t.index_select(1, fused_idx)
                touched += 1
                break
            if name == f"h.{i}.attn.c_attn.bias":
                check(tuple(t.shape) == (3 * HIDDEN,), f"{name} shape {tuple(t.shape)}")
                t = t.index_select(0, fused_idx)
                touched += 1
                break
            if name == f"h.{i}.attn.c_proj.weight":
                check(tuple(t.shape) == (HIDDEN, HIDDEN), f"{name} shape {tuple(t.shape)}")
                t = t.index_select(0, seg_idx)
                touched += 1
                break
        out[name] = t.contiguous()

    check(touched == 3 * N_LAYERS, f"expected to edit {3 * N_LAYERS} tensors, edited {touched}")

    # Required checks.
    check(tuple(out["h.0.attn.c_attn.weight"].shape) == (768, 2112),
          f"h.0.attn.c_attn.weight shape {tuple(out['h.0.attn.c_attn.weight'].shape)}")
    check(tuple(out["h.0.attn.c_attn.bias"].shape) == (2112,),
          f"h.0.attn.c_attn.bias shape {tuple(out['h.0.attn.c_attn.bias'].shape)}")
    check(tuple(out["h.0.attn.c_proj.weight"].shape) == (704, 768),
          f"h.0.attn.c_proj.weight shape {tuple(out['h.0.attn.c_proj.weight'].shape)}")
    check(len(out) == 160, f"expected 160 output tensors, got {len(out)}")

    # Extra sanity: every layer, and values match a direct slice-and-concat.
    for i in range(N_LAYERS):
        w = sd[f"h.{i}.attn.c_attn.weight"]
        ref = torch.cat([w[:, 0:320], w[:, 384:768], w[:, 768:1088],
                         w[:, 1152:1536], w[:, 1536:1856], w[:, 1920:2304]], dim=1)
        check(torch.equal(out[f"h.{i}.attn.c_attn.weight"], ref), f"layer {i} c_attn.weight values")
        b = sd[f"h.{i}.attn.c_attn.bias"]
        ref = torch.cat([b[0:320], b[384:768], b[768:1088], b[1152:1536], b[1536:1856], b[1920:2304]])
        check(torch.equal(out[f"h.{i}.attn.c_attn.bias"], ref), f"layer {i} c_attn.bias values")
        p = sd[f"h.{i}.attn.c_proj.weight"]
        ref = torch.cat([p[0:320], p[384:768]], dim=0)
        check(torch.equal(out[f"h.{i}.attn.c_proj.weight"], ref), f"layer {i} c_proj.weight values")
        check(tuple(out[f"h.{i}.attn.c_proj.bias"].shape) == (768,), f"layer {i} c_proj.bias touched")
        check(tuple(out[f"h.{i}.attn.bias"].shape) == (1, 1, 1024, 1024), f"layer {i} attn.bias touched")
    for name, t in sd.items():
        check(out[name].dtype == t.dtype, f"{name} dtype changed")

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    save_file(out, DST, metadata={"format": "pt"})

    back = load_file(DST)
    check(len(back) == 160, f"reloaded output has {len(back)} tensors")
    check(tuple(back["h.0.attn.c_attn.weight"].shape) == (768, 2112), "reload c_attn.weight shape")
    check(tuple(back["h.0.attn.c_proj.weight"].shape) == (704, 768), "reload c_proj.weight shape")
    print(f"OK: wrote {DST} with {len(back)} tensors; edited {touched} tensors")


if __name__ == "__main__":
    main()
