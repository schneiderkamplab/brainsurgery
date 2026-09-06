"""T2: remove attention head 5 from every layer of Pythia-1B (GPT-NeoX layout).

Plain safetensors slicing. No torch-side model loading, so values stay bit-exact.
"""
import os
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T2/model.safetensors"
N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN = 16, 8, 256, 2048
PRUNE_HEAD = 5
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head in the fused projection


def keep_index(n_heads: int, block: int, drop: int) -> torch.Tensor:
    """Indices of all positions except those belonging to head `drop`."""
    keep = [h for h in range(n_heads) if h != drop]
    return torch.cat([torch.arange(h * block, (h + 1) * block) for h in keep])


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"CHECK FAILED: {msg}")


def main() -> None:
    check(not os.path.exists(DST), f"destination already exists: {DST}")
    sd = load_file(SRC)
    check(len(sd) == 244, f"expected 244 input tensors, got {len(sd)}")

    qkv_idx = keep_index(N_HEADS, QKV_BLOCK, PRUNE_HEAD)
    dense_idx = keep_index(N_HEADS, HEAD_DIM, PRUNE_HEAD)
    touched = 0
    for i in range(N_LAYERS):
        p = f"gpt_neox.layers.{i}.attention."
        w, b, d = p + "query_key_value.weight", p + "query_key_value.bias", p + "dense.weight"
        check(tuple(sd[w].shape) == (6144, HIDDEN), f"{w} unexpected shape {tuple(sd[w].shape)}")
        check(tuple(sd[b].shape) == (6144,), f"{b} unexpected shape {tuple(sd[b].shape)}")
        check(tuple(sd[d].shape) == (HIDDEN, HIDDEN), f"{d} unexpected shape {tuple(sd[d].shape)}")
        sd[w] = sd[w].index_select(0, qkv_idx).contiguous()
        sd[b] = sd[b].index_select(0, qkv_idx).contiguous()
        sd[d] = sd[d].index_select(1, dense_idx).contiguous()
        touched += 3
    check(touched == 48, f"touched {touched} tensors, expected 48")

    # Required checks (TASK.md), enforced before writing.
    l0 = "gpt_neox.layers.0.attention."
    check(tuple(sd[l0 + "query_key_value.weight"].shape) == (5376, 2048), "layer0 qkv.weight shape")
    check(tuple(sd[l0 + "query_key_value.bias"].shape) == (5376,), "layer0 qkv.bias shape")
    check(tuple(sd[l0 + "dense.weight"].shape) == (2048, 1792), "layer0 dense.weight shape")
    check(len(sd) == 244, f"output has {len(sd)} tensors, expected 244")
    for i in range(N_LAYERS):
        p = f"gpt_neox.layers.{i}.attention."
        check(tuple(sd[p + "query_key_value.weight"].shape) == (5376, 2048), f"layer {i} qkv.weight")
        check(tuple(sd[p + "query_key_value.bias"].shape) == (5376,), f"layer {i} qkv.bias")
        check(tuple(sd[p + "dense.weight"].shape) == (2048, 1792), f"layer {i} dense.weight")
    for i in range(N_LAYERS):
        p = f"gpt_neox.layers.{i}.attention."
        for n in ("query_key_value.weight", "query_key_value.bias", "dense.weight"):
            check(sd[p + n].dtype == torch.float16, f"{p + n} dtype changed")

    save_file(sd, DST, metadata={"format": "pt"})

    # Post-write verification against the source: kept slices are bit-identical.
    out = load_file(DST)
    src = load_file(SRC)
    check(set(out) == set(src), "key set changed")
    for k in src:
        if k.endswith("attention.query_key_value.weight") or k.endswith("attention.query_key_value.bias"):
            ref = torch.cat([src[k][:3840], src[k][4608:]], 0)
        elif k.endswith("attention.dense.weight"):
            ref = torch.cat([src[k][:, :1280], src[k][:, 1536:]], 1)
        else:
            ref = src[k]
        check(out[k].dtype == ref.dtype and torch.equal(out[k], ref), f"{k} differs from expected")
    print(f"OK: wrote {DST} with {len(out)} tensors")


if __name__ == "__main__":
    sys.exit(main())
