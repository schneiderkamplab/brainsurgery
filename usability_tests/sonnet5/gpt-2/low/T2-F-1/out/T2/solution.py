"""T2: prune head 5 from every layer of GPT-2 (124M) at the checkpoint level.

Uses plain torch + safetensors: for each layer, slice attn.c_attn.weight
(column blocks, one per q/k/v segment), attn.c_attn.bias (same layout), and
attn.c_proj.weight (row blocks) to drop head 5, keeping all other tensors
untouched. This is a direct tensor-surgery script on top of torch/safetensors
(both in F-allowed.md) rather than transformers' `prune_heads`, because the
required column order in TASK.md (concatenate q-kept, k-kept, v-kept
contiguous blocks) is exactly what explicit slicing gives with no ambiguity
about how a library helper orders/merges the segments.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HEAD = 5
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = 768
N_LAYERS = 12

IN_PATH = Path("inputs/base/model.safetensors")
OUT_DIR = Path("out/T2")
OUT_PATH = OUT_DIR / "model.safetensors"


def kept_head_ranges(n_heads: int, head_dim: int, drop_head: int) -> list[tuple[int, int]]:
    ranges = []
    for h in range(n_heads):
        if h == drop_head:
            continue
        start = h * head_dim
        ranges.append((start, start + head_dim))
    return ranges


def slice_concat(tensor: torch.Tensor, dim: int, ranges: list[tuple[int, int]]) -> torch.Tensor:
    pieces = []
    for start, end in ranges:
        idx = [slice(None)] * tensor.ndim
        idx[dim] = slice(start, end)
        pieces.append(tensor[tuple(idx)])
    return torch.cat(pieces, dim=dim)


def main() -> None:
    if not IN_PATH.exists():
        sys.exit(f"missing input: {IN_PATH}")

    tensors = load_file(str(IN_PATH))

    head_ranges = kept_head_ranges(N_HEADS, HEAD_DIM, HEAD)
    assert len(head_ranges) == N_HEADS - 1

    out = dict(tensors)  # unchanged tensors pass through by reference

    for i in range(N_LAYERS):
        attn_w_key = f"h.{i}.attn.c_attn.weight"
        attn_b_key = f"h.{i}.attn.c_attn.bias"
        proj_w_key = f"h.{i}.attn.c_proj.weight"

        attn_w = tensors[attn_w_key]
        attn_b = tensors[attn_b_key]
        proj_w = tensors[proj_w_key]

        assert attn_w.shape == (HIDDEN, 3 * HIDDEN), (attn_w_key, attn_w.shape)
        assert attn_b.shape == (3 * HIDDEN,), (attn_b_key, attn_b.shape)
        assert proj_w.shape == (HIDDEN, HIDDEN), (proj_w_key, proj_w.shape)

        # c_attn columns are [q(768) | k(768) | v(768)]; each segment's
        # columns are head blocks of width HEAD_DIM. Keep all heads except
        # HEAD, in order, within each segment, and keep the three segments
        # in q,k,v order.
        qkv_ranges = []
        for seg in range(3):
            offset = seg * HIDDEN
            qkv_ranges.extend((offset + s, offset + e) for s, e in head_ranges)

        new_attn_w = slice_concat(attn_w, dim=1, ranges=qkv_ranges)
        new_attn_b = slice_concat(attn_b, dim=0, ranges=qkv_ranges)
        new_proj_w = slice_concat(proj_w, dim=0, ranges=head_ranges)

        expected_qkv_width = 3 * (N_HEADS - 1) * HEAD_DIM
        expected_proj_rows = (N_HEADS - 1) * HEAD_DIM
        assert new_attn_w.shape == (HIDDEN, expected_qkv_width), new_attn_w.shape
        assert new_attn_b.shape == (expected_qkv_width,), new_attn_b.shape
        assert new_proj_w.shape == (expected_proj_rows, HIDDEN), new_proj_w.shape

        out[attn_w_key] = new_attn_w.contiguous()
        out[attn_b_key] = new_attn_b.contiguous()
        out[proj_w_key] = new_proj_w.contiguous()

    # --- Required checks (fail loudly before writing) ---
    assert out["h.0.attn.c_attn.weight"].shape == (768, 2112), out["h.0.attn.c_attn.weight"].shape
    assert out["h.0.attn.c_attn.bias"].shape == (2112,), out["h.0.attn.c_attn.bias"].shape
    assert out["h.0.attn.c_proj.weight"].shape == (704, 768), out["h.0.attn.c_proj.weight"].shape
    assert len(out) == 160, f"expected 160 tensors, got {len(out)}"
    assert len(tensors) == 160, f"input had {len(tensors)} tensors, expected 160"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(out)} tensors")


if __name__ == "__main__":
    main()
