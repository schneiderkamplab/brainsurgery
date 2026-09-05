"""
Structured attention-head pruning for GPT-2 (124M).

Removes head 5 (0-indexed) from every layer's attention block:
- h.<i>.attn.c_attn.weight  [768, 2304] -> [768, 2112]  (column blocks removed)
- h.<i>.attn.c_attn.bias    [2304]      -> [2112]        (row/element blocks removed)
- h.<i>.attn.c_proj.weight  [768, 768]  -> [704, 768]     (row blocks removed)

Everything else is copied unchanged.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

IN_PATH = Path("inputs/base/model.safetensors")
OUT_PATH = Path("out/T2/model.safetensors")

NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64
HIDDEN = NUM_HEADS * HEAD_DIM  # 768
PRUNE_HEAD = 5


def kept_head_indices(num_heads: int, pruned: int) -> list[int]:
    return [h for h in range(num_heads) if h != pruned]


def slice_ranges(num_heads: int, head_dim: int, pruned: int) -> list[tuple[int, int]]:
    """Column/row ranges to keep, in order, for a single 768-wide head-blocked axis."""
    return [(h * head_dim, (h + 1) * head_dim) for h in kept_head_indices(num_heads, pruned)]


def gather_cols(t: torch.Tensor, ranges: list[tuple[int, int]]) -> torch.Tensor:
    return torch.cat([t[:, a:b] for a, b in ranges], dim=1)


def gather_rows(t: torch.Tensor, ranges: list[tuple[int, int]]) -> torch.Tensor:
    return torch.cat([t[a:b, ...] for a, b in ranges], dim=0)


def gather_1d(t: torch.Tensor, ranges: list[tuple[int, int]]) -> torch.Tensor:
    return torch.cat([t[a:b] for a, b in ranges], dim=0)


def main() -> None:
    tensors = load_file(str(IN_PATH))
    assert len(tensors) == 160, f"expected 160 input tensors, got {len(tensors)}"

    head_ranges = slice_ranges(NUM_HEADS, HEAD_DIM, PRUNE_HEAD)  # 11 ranges of width 64

    out: dict[str, torch.Tensor] = {}

    for i in range(NUM_LAYERS):
        prefix = f"h.{i}.attn."

        # c_attn.weight: [768, 2304] fused [q|k|v], each 768-wide segment is head-blocked
        # on the column axis. Build the combined keep-ranges across all three segments.
        c_attn_w = tensors[prefix + "c_attn.weight"]
        assert c_attn_w.shape == (HIDDEN, 3 * HIDDEN), c_attn_w.shape
        qkv_ranges = []
        for seg in range(3):
            offset = seg * HIDDEN
            qkv_ranges.extend([(a + offset, b + offset) for a, b in head_ranges])
        new_c_attn_w = gather_cols(c_attn_w, qkv_ranges)
        out[prefix + "c_attn.weight"] = new_c_attn_w

        # c_attn.bias: [2304], same column layout as c_attn.weight
        c_attn_b = tensors[prefix + "c_attn.bias"]
        assert c_attn_b.shape == (3 * HIDDEN,), c_attn_b.shape
        new_c_attn_b = gather_1d(c_attn_b, qkv_ranges)
        out[prefix + "c_attn.bias"] = new_c_attn_b

        # c_proj.weight: [768, 768], heads are row blocks
        c_proj_w = tensors[prefix + "c_proj.weight"]
        assert c_proj_w.shape == (HIDDEN, HIDDEN), c_proj_w.shape
        new_c_proj_w = gather_rows(c_proj_w, head_ranges)
        out[prefix + "c_proj.weight"] = new_c_proj_w

        # Untouched per-layer attention tensors
        out[prefix + "c_proj.bias"] = tensors[prefix + "c_proj.bias"]
        out[prefix + "bias"] = tensors[prefix + "bias"]

    # Copy every other tensor unchanged (non-attention tensors, and anything
    # not already handled above).
    handled_suffixes = {
        "attn.c_attn.weight",
        "attn.c_attn.bias",
        "attn.c_proj.weight",
        "attn.c_proj.bias",
        "attn.bias",
    }
    layer_prefixes = tuple(f"h.{i}." for i in range(NUM_LAYERS))
    for name, tensor in tensors.items():
        if name.startswith(layer_prefixes):
            suffix = name[name.index(".", name.index(".") + 1) + 1 :]
            if suffix in handled_suffixes:
                continue
        if name in out:
            continue
        out[name] = tensor

    # --- Required checks ---
    assert out["h.0.attn.c_attn.weight"].shape == (768, 2112), (
        f"h.0.attn.c_attn.weight shape mismatch: {out['h.0.attn.c_attn.weight'].shape}"
    )
    assert out["h.0.attn.c_attn.bias"].shape == (2112,), (
        f"h.0.attn.c_attn.bias shape mismatch: {out['h.0.attn.c_attn.bias'].shape}"
    )
    assert out["h.0.attn.c_proj.weight"].shape == (704, 768), (
        f"h.0.attn.c_proj.weight shape mismatch: {out['h.0.attn.c_proj.weight'].shape}"
    )
    assert len(out) == 160, f"expected 160 output tensors, got {len(out)}"

    # Sanity-check for all layers, not just layer 0
    for i in range(NUM_LAYERS):
        prefix = f"h.{i}.attn."
        assert out[prefix + "c_attn.weight"].shape == (768, 2112)
        assert out[prefix + "c_attn.bias"].shape == (2112,)
        assert out[prefix + "c_proj.weight"].shape == (704, 768)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {k: v.contiguous() for k, v in out.items()}
    save_file(out, str(OUT_PATH))
    print(f"Wrote {len(out)} tensors to {OUT_PATH}")


if __name__ == "__main__":
    main()
