"""
T2: Structured attention-head pruning (GPT-2, 124M)

Removes head 5 (0-indexed) from every layer's attention block:
  - h.<i>.attn.c_attn.weight  [768, 2304] -> [768, 2112]  (columns; q|k|v fused, heads are 64-wide column blocks within each 768-wide segment)
  - h.<i>.attn.c_attn.bias    [2304]      -> [2112]        (same layout as the columns above)
  - h.<i>.attn.c_proj.weight  [768, 768]  -> [704, 768]    (rows; heads are 64-wide row blocks)

Everything else (including attn.c_proj.bias and the attn.bias mask buffer) is
copied through unchanged.
"""

import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INPUT_PATH = "inputs/base/model.safetensors"
OUTPUT_DIR = "out/T2"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "model.safetensors")

N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
HEAD_TO_REMOVE = 5


def keep_indices_within_segment(n_heads: int, head_dim: int, remove_head: int) -> torch.Tensor:
    """Indices (0..n_heads*head_dim-1) to keep after dropping one head's block."""
    idx = []
    for h in range(n_heads):
        if h == remove_head:
            continue
        start = h * head_dim
        idx.extend(range(start, start + head_dim))
    return torch.tensor(idx, dtype=torch.long)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tensors = {}
    with safe_open(INPUT_PATH, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    assert len(tensors) == 160, f"expected 160 input tensors, got {len(tensors)}"

    # Indices to keep within a single 768-wide head-bearing segment.
    keep = keep_indices_within_segment(N_HEADS, HEAD_DIM, HEAD_TO_REMOVE)
    assert keep.numel() == (N_HEADS - 1) * HEAD_DIM  # 704

    out = {}
    for key, t in tensors.items():
        if key.endswith("attn.c_attn.weight"):
            # [768, 2304]: three concatenated [q | k | v] segments of width 768 each,
            # heads are 64-wide column blocks within each segment.
            assert t.shape == (HIDDEN, 3 * HIDDEN), f"{key}: unexpected shape {tuple(t.shape)}"
            segments = [t.index_select(1, keep + seg * HIDDEN) for seg in range(3)]
            out[key] = torch.cat(segments, dim=1).contiguous()

        elif key.endswith("attn.c_attn.bias"):
            # [2304]: same q|k|v segment/column layout as the weight above.
            assert t.shape == (3 * HIDDEN,), f"{key}: unexpected shape {tuple(t.shape)}"
            segments = [t.index_select(0, keep + seg * HIDDEN) for seg in range(3)]
            out[key] = torch.cat(segments, dim=0).contiguous()

        elif key.endswith("attn.c_proj.weight"):
            # [768, 768]: heads are 64-wide row blocks.
            assert t.shape == (HIDDEN, HIDDEN), f"{key}: unexpected shape {tuple(t.shape)}"
            out[key] = t.index_select(0, keep).contiguous()

        else:
            # Not per-head (includes attn.c_proj.bias and the attn.bias mask buffer):
            # copy through unchanged.
            out[key] = t.contiguous()

    # Required checks (fail loudly before writing).
    def check_shape(key: str, expected: tuple) -> None:
        got = tuple(out[key].shape)
        assert got == expected, f"check failed: {key} has shape {got}, expected {expected}"

    check_shape("h.0.attn.c_attn.weight", (768, 2112))
    check_shape("h.0.attn.c_attn.bias", (2112,))
    check_shape("h.0.attn.c_proj.weight", (704, 768))
    assert len(out) == 160, f"check failed: output has {len(out)} tensors, expected 160"

    save_file(out, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH} with {len(out)} tensors")


if __name__ == "__main__":
    main()
