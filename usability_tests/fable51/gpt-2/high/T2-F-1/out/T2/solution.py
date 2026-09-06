"""T2: prune attention head 5 from every layer of GPT-2 (124M) at checkpoint level.

Plain safetensors + torch script. GPT-2 uses Conv1D [in, out] layout, so
c_attn heads are column blocks (inside each of the q/k/v 768-wide segments)
and c_proj heads are row blocks.
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
EXPECTED_TENSORS = 160


def keep_index(n_heads: int, head: int, head_dim: int) -> torch.Tensor:
    """Indices 0..n_heads*head_dim-1 with the block of `head` removed."""
    idx = torch.arange(n_heads * head_dim)
    mask = (idx // head_dim) != head
    return idx[mask]


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"CHECK FAILED: {msg}")


def main() -> None:
    check(not os.path.exists(DST), f"destination already exists: {DST}")
    sd = load_file(SRC)
    check(len(sd) == EXPECTED_TENSORS, f"input has {len(sd)} tensors, expected {EXPECTED_TENSORS}")

    per_seg = keep_index(N_HEADS, PRUNE_HEAD, HEAD_DIM)  # 704 indices within a 768 segment
    # fused [q | k | v]: apply the same per-segment keep to each 768-wide block, in order
    qkv_keep = torch.cat([per_seg + s * HIDDEN for s in range(3)])
    check(qkv_keep.numel() == 3 * (HIDDEN - HEAD_DIM), "qkv keep index size")
    check(bool((qkv_keep[1:] > qkv_keep[:-1]).all()), "qkv keep index must be increasing")

    out = {}
    touched = 0
    for name, t in sd.items():
        if name.startswith("h.") and name.endswith(".attn.c_attn.weight"):
            check(tuple(t.shape) == (HIDDEN, 3 * HIDDEN), f"{name} shape {tuple(t.shape)}")
            new = t.index_select(1, qkv_keep)
            check(tuple(new.shape) == (768, 2112), f"{name} -> {tuple(new.shape)}")
        elif name.startswith("h.") and name.endswith(".attn.c_attn.bias"):
            check(tuple(t.shape) == (3 * HIDDEN,), f"{name} shape {tuple(t.shape)}")
            new = t.index_select(0, qkv_keep)
            check(tuple(new.shape) == (2112,), f"{name} -> {tuple(new.shape)}")
        elif name.startswith("h.") and name.endswith(".attn.c_proj.weight"):
            check(tuple(t.shape) == (HIDDEN, HIDDEN), f"{name} shape {tuple(t.shape)}")
            new = t.index_select(0, per_seg)
            check(tuple(new.shape) == (704, 768), f"{name} -> {tuple(new.shape)}")
        else:
            out[name] = t
            continue
        check(new.dtype == t.dtype, f"{name} dtype changed")
        out[name] = new.contiguous()
        touched += 1

    check(touched == 3 * N_LAYERS, f"touched {touched} tensors, expected {3 * N_LAYERS}")

    # Required checks (TASK.md), before writing.
    check(tuple(out["h.0.attn.c_attn.weight"].shape) == (768, 2112), "h.0 c_attn.weight shape")
    check(tuple(out["h.0.attn.c_attn.bias"].shape) == (2112,), "h.0 c_attn.bias shape")
    check(tuple(out["h.0.attn.c_proj.weight"].shape) == (704, 768), "h.0 c_proj.weight shape")
    check(len(out) == EXPECTED_TENSORS, f"output has {len(out)} tensors, expected {EXPECTED_TENSORS}")
    check(set(out) == set(sd), "key set changed")

    # Value spot-check against the explicit column ranges in TASK.md.
    w = sd["h.0.attn.c_attn.weight"]
    ranges = [(0, 320), (384, 768), (768, 1088), (1152, 1536), (1536, 1856), (1920, 2304)]
    ref = torch.cat([w[:, a:b] for a, b in ranges], dim=1)
    check(torch.equal(ref, out["h.0.attn.c_attn.weight"]), "h.0 c_attn.weight values")
    p = sd["h.0.attn.c_proj.weight"]
    check(torch.equal(torch.cat([p[0:320], p[384:768]]), out["h.0.attn.c_proj.weight"]),
          "h.0 c_proj.weight values")
    for name in sd:
        if ".attn.c_attn." not in name and ".attn.c_proj.weight" not in name:
            check(torch.equal(sd[name], out[name]), f"{name} should be unchanged")

    save_file(out, DST, metadata={"format": "pt"})

    # Post-write verification.
    back = load_file(DST)
    check(len(back) == EXPECTED_TENSORS, "reloaded tensor count")
    for name in out:
        check(torch.equal(back[name], out[name]), f"reloaded {name} differs")
    print(f"OK: wrote {DST} with {len(back)} tensors; pruned head {PRUNE_HEAD} in {N_LAYERS} layers")


if __name__ == "__main__":
    sys.exit(main())
