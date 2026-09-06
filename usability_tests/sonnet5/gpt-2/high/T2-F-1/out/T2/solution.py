"""T2: structured attention-head pruning for GPT-2 (124M), condition F.

Removes head 5 (0-indexed) from every layer's attention block by slicing the
fused c_attn projection (columns, per q/k/v segment) and the c_proj
projection (rows). All other tensors, including the `attn.bias` causal-mask
buffer, are copied through unchanged.

Why a plain script instead of a toolkit call: this transformers version
(5.12.1) does not implement head pruning for GPT-2 -- GPT2Attention has no
`prune_heads` override, so `PreTrainedModel.prune_heads()` falls through to
the generic `_prune_heads`, which GPT2's model class does not implement.
mergekit and torch-state-bridge operate on whole tensors / renamed keys, not
sub-tensor column/row slices within a fused QKV block, so they don't fit
either. safetensors + torch give direct, checkable control over exactly
which columns/rows survive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent.parent
IN_PATH = SANDBOX / "inputs" / "base" / "model.safetensors"
OUT_PATH = HERE / "model.safetensors"

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
PRUNE_HEAD = 5


def keep_indices_excluding(total: int, head_dim: int, prune_head: int) -> list[int]:
    """Indices 0..total-1 with the [prune_head*head_dim, (prune_head+1)*head_dim) block removed."""
    start = prune_head * head_dim
    end = start + head_dim
    return [i for i in range(total) if not (start <= i < end)]


def qkv_keep_columns(prune_head: int) -> list[int]:
    """Column keep-list for c_attn.weight/bias: three HIDDEN-wide segments (q, k, v),
    each with the pruned head's 64-wide block removed, concatenated in order."""
    cols: list[int] = []
    for seg in range(3):  # q, k, v
        seg_start = seg * HIDDEN
        local_keep = keep_indices_excluding(HIDDEN, HEAD_DIM, prune_head)
        cols.extend(seg_start + i for i in local_keep)
    return cols


def main() -> None:
    if not IN_PATH.is_file():
        sys.exit(f"missing input checkpoint: {IN_PATH}")

    tensors = load_file(str(IN_PATH))
    if len(tensors) != 160:
        sys.exit(f"expected 160 input tensors, found {len(tensors)}")

    qkv_cols = torch.tensor(qkv_keep_columns(PRUNE_HEAD), dtype=torch.long)
    proj_rows = torch.tensor(
        keep_indices_excluding(HIDDEN, HEAD_DIM, PRUNE_HEAD), dtype=torch.long
    )

    out: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if name.endswith("attn.c_attn.weight"):
            assert tensor.shape == (HIDDEN, 3 * HIDDEN), (name, tensor.shape)
            new_t = tensor.index_select(1, qkv_cols).contiguous()
        elif name.endswith("attn.c_attn.bias"):
            assert tensor.shape == (3 * HIDDEN,), (name, tensor.shape)
            new_t = tensor.index_select(0, qkv_cols).contiguous()
        elif name.endswith("attn.c_proj.weight"):
            assert tensor.shape == (HIDDEN, HIDDEN), (name, tensor.shape)
            new_t = tensor.index_select(0, proj_rows).contiguous()
        else:
            # includes attn.c_proj.bias and the attn.bias mask buffer, which
            # are explicitly NOT per-head and must pass through untouched
            new_t = tensor.clone()
        out[name] = new_t

    # Required checks: fail loudly before writing anything.
    expected_qkv_width = 3 * HIDDEN - HEAD_DIM * 3  # 2304 - 192 = 2112
    expected_proj_rows = HIDDEN - HEAD_DIM  # 704

    def shape(name: str) -> torch.Size:
        return out[name].shape

    assert shape("h.0.attn.c_attn.weight") == (HIDDEN, expected_qkv_width), shape(
        "h.0.attn.c_attn.weight"
    )
    assert shape("h.0.attn.c_attn.bias") == (expected_qkv_width,), shape("h.0.attn.c_attn.bias")
    assert shape("h.0.attn.c_proj.weight") == (expected_proj_rows, HIDDEN), shape(
        "h.0.attn.c_proj.weight"
    )
    assert len(out) == 160, f"expected 160 output tensors, got {len(out)}"

    for i in range(N_LAYERS):
        assert out[f"h.{i}.attn.c_attn.weight"].shape == (HIDDEN, expected_qkv_width)
        assert out[f"h.{i}.attn.c_attn.bias"].shape == (expected_qkv_width,)
        assert out[f"h.{i}.attn.c_proj.weight"].shape == (expected_proj_rows, HIDDEN)
        # untouched tensors keep their original shape
        assert out[f"h.{i}.attn.c_proj.bias"].shape == (HIDDEN,)
        assert out[f"h.{i}.attn.bias"].shape == (1, 1, 1024, 1024)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"wrote {len(out)} tensors to {OUT_PATH}")


if __name__ == "__main__":
    main()
