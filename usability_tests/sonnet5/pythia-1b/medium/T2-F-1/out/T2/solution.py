"""T2: remove attention head 5 from every layer of Pythia-1B (GPT-NeoX).

Layout, per the task spec:
  - query_key_value.weight: [6144, 2048], rows grouped per head into 768-row
    blocks (head h owns rows 768*h .. 768*h+767; interleaved qkv within the
    block, but that internal structure doesn't matter here since we drop the
    whole block).
  - query_key_value.bias: [6144], same row grouping.
  - dense.weight: [2048, 2048], heads are 256-wide column blocks.

We plain-slice with torch/safetensors: no HF model construction needed since
the transform is a pure tensor-layout edit, and doing it directly avoids
depending on transformers' generic `prune_heads` (which does not know this
model's fused-qkv, GPT-NeoX-interleaved layout).
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HEAD_TO_PRUNE = 5
NUM_HEADS = 8
HEAD_DIM = 256
HIDDEN = 2048
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head in the fused qkv tensor

IN_PATH = Path(__file__).resolve().parents[2] / "inputs" / "base" / "model.safetensors"
OUT_PATH = Path(__file__).resolve().parent / "model.safetensors"


def qkv_row_keep_slices():
    """Rows to keep in query_key_value.{weight,bias}: drop the pruned head's 768-row block."""
    lo = HEAD_TO_PRUNE * QKV_BLOCK
    hi = lo + QKV_BLOCK
    return lo, hi  # [0..lo) and [hi..6144) are kept


def dense_col_keep_slices():
    """Columns to keep in dense.weight: drop the pruned head's 256-col block."""
    lo = HEAD_TO_PRUNE * HEAD_DIM
    hi = lo + HEAD_DIM
    return lo, hi  # [0..lo) and [hi..2048) are kept


def main():
    if not IN_PATH.exists():
        sys.exit(f"input not found: {IN_PATH}")

    qkv_lo, qkv_hi = qkv_row_keep_slices()
    dense_lo, dense_hi = dense_col_keep_slices()

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(IN_PATH), framework="pt") as f:
        keys = list(f.keys())
        assert len(keys) == 244, f"expected 244 input tensors, got {len(keys)}"
        for k in keys:
            t = f.get_tensor(k)
            if k.endswith("attention.query_key_value.weight"):
                assert t.shape == (6144, HIDDEN), (k, t.shape)
                t = torch.cat([t[:qkv_lo], t[qkv_hi:]], dim=0).contiguous()
                assert t.shape == (5376, HIDDEN), (k, t.shape)
            elif k.endswith("attention.query_key_value.bias"):
                assert t.shape == (6144,), (k, t.shape)
                t = torch.cat([t[:qkv_lo], t[qkv_hi:]], dim=0).contiguous()
                assert t.shape == (5376,), (k, t.shape)
            elif k.endswith("attention.dense.weight"):
                assert t.shape == (HIDDEN, HIDDEN), (k, t.shape)
                t = torch.cat([t[:, :dense_lo], t[:, dense_hi:]], dim=1).contiguous()
                assert t.shape == (HIDDEN, 1792), (k, t.shape)
            tensors[k] = t

    # Required checks (fail loudly before writing).
    assert tensors["gpt_neox.layers.0.attention.query_key_value.weight"].shape == (5376, 2048)
    assert tensors["gpt_neox.layers.0.attention.query_key_value.bias"].shape == (5376,)
    assert tensors["gpt_neox.layers.0.attention.dense.weight"].shape == (2048, 1792)
    assert len(tensors) == 244, f"expected 244 output tensors, got {len(tensors)}"

    for i in range(16):
        w_key = f"gpt_neox.layers.{i}.attention.query_key_value.weight"
        b_key = f"gpt_neox.layers.{i}.attention.query_key_value.bias"
        d_key = f"gpt_neox.layers.{i}.attention.dense.weight"
        assert tensors[w_key].shape == (5376, HIDDEN), (w_key, tensors[w_key].shape)
        assert tensors[b_key].shape == (5376,), (b_key, tensors[b_key].shape)
        assert tensors[d_key].shape == (HIDDEN, 1792), (d_key, tensors[d_key].shape)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(tensors)} tensors")


if __name__ == "__main__":
    main()
