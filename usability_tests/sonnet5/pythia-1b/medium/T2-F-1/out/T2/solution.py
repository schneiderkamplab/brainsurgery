"""T2: structured attention-head pruning for Pythia-1B.

Removes head 5 (0-indexed) from every layer's fused query_key_value
projection and the output dense projection, at the checkpoint level.

Layout notes (from TASK.md):
  - query_key_value.weight: [6144, 2048], rows grouped per head in GPT-NeoX
    interleaved order. Head h owns rows [768h, 768h+768) and inside that
    block holds q (256 rows), then k (256 rows), then v (256 rows).
  - query_key_value.bias: [6144], same row layout.
  - dense.weight: [2048, 2048] ([out, in] nn.Linear layout); heads are
    256-wide column blocks (the "in" side, since dense consumes head outputs
    concatenated along the input dimension).
  - dense.bias, attention buffers, MLP tensors: untouched.

We do not use transformers' GPTNeoXAttention.prune_heads (transformers 5.12
does not implement head pruning for the GPT-NeoX family), so this operates
directly on the safetensors state dict with plain torch slicing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
HIDDEN = 2048
QKV_BLOCK = 3 * HEAD_DIM  # 768: one head's q,k,v rows in the fused projection
HEAD_TO_PRUNE = 5

EXPECTED_TENSORS = 244


def head_row_range(h: int) -> tuple[int, int]:
    return h * QKV_BLOCK, (h + 1) * QKV_BLOCK


def head_col_range(h: int) -> tuple[int, int]:
    return h * HEAD_DIM, (h + 1) * HEAD_DIM


def kept_row_index(pruned_head: int) -> torch.Tensor:
    """Row indices into the 6144-row qkv tensors, keeping all heads but one."""
    parts = []
    for h in range(NUM_HEADS):
        if h == pruned_head:
            continue
        lo, hi = head_row_range(h)
        parts.append(torch.arange(lo, hi))
    return torch.cat(parts)


def kept_col_index(pruned_head: int) -> torch.Tensor:
    """Column indices into the 2048-col dense.weight input side."""
    parts = []
    for h in range(NUM_HEADS):
        if h == pruned_head:
            continue
        lo, hi = head_col_range(h)
        parts.append(torch.arange(lo, hi))
    return torch.cat(parts)


def prune(state_dict: dict[str, torch.Tensor], head: int) -> dict[str, torch.Tensor]:
    rows = kept_row_index(head)
    cols = kept_col_index(head)

    out: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.endswith("attention.query_key_value.weight") or key.endswith(
            "attention.query_key_value.bias"
        ):
            out[key] = tensor.index_select(0, rows).contiguous()
        elif key.endswith("attention.dense.weight"):
            out[key] = tensor.index_select(1, cols).contiguous()
        else:
            out[key] = tensor
    return out


def check(state_dict: dict[str, torch.Tensor]) -> None:
    expected_qkv_rows = NUM_HEADS * QKV_BLOCK - QKV_BLOCK  # 5376
    expected_dense_cols = NUM_HEADS * HEAD_DIM - HEAD_DIM  # 1792

    assert len(state_dict) == EXPECTED_TENSORS, (
        f"expected {EXPECTED_TENSORS} tensors, got {len(state_dict)}"
    )

    for i in range(NUM_LAYERS):
        w = state_dict[f"gpt_neox.layers.{i}.attention.query_key_value.weight"]
        assert w.shape == (expected_qkv_rows, HIDDEN), (
            f"layer {i} qkv.weight shape {tuple(w.shape)} != "
            f"({expected_qkv_rows}, {HIDDEN})"
        )
        b = state_dict[f"gpt_neox.layers.{i}.attention.query_key_value.bias"]
        assert b.shape == (expected_qkv_rows,), (
            f"layer {i} qkv.bias shape {tuple(b.shape)} != ({expected_qkv_rows},)"
        )
        d = state_dict[f"gpt_neox.layers.{i}.attention.dense.weight"]
        assert d.shape == (HIDDEN, expected_dense_cols), (
            f"layer {i} dense.weight shape {tuple(d.shape)} != "
            f"({HIDDEN}, {expected_dense_cols})"
        )
        db = state_dict[f"gpt_neox.layers.{i}.attention.dense.bias"]
        assert db.shape == (HIDDEN,), f"layer {i} dense.bias shape changed unexpectedly"


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "inputs" / "base" / "model.safetensors"
    dst_dir = root / "out" / "T2"
    dst = dst_dir / "model.safetensors"

    state_dict = load_file(str(src))
    pruned = prune(state_dict, HEAD_TO_PRUNE)
    check(pruned)

    dst_dir.mkdir(parents=True, exist_ok=True)
    save_file(pruned, str(dst))
    print(f"wrote {dst} with {len(pruned)} tensors")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)
