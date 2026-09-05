#!/usr/bin/env python
"""
T2: structured attention-head pruning for GPT-2 (124M).

Remove head 5 (0-indexed) from every layer's attention block:
  - h.<i>.attn.c_attn.weight [768, 2304] (Conv1D, [in, out]): fused q|k|v,
    heads are 64-wide column blocks inside each 768-wide q/k/v segment.
  - h.<i>.attn.c_attn.bias   [2304]: same column layout.
  - h.<i>.attn.c_proj.weight [768, 768]: heads are 64-wide row blocks.

Everything else (attn.c_proj.bias, attn.bias mask buffer, ln_*, mlp.*,
wte/wpe/ln_f) is copied through unchanged.

Plain torch + safetensors: no framework in F-allowed.md ships a working
head-pruning primitive for GPT-2 in this transformers version (5.12.1 has
neither GPT2Attention.prune_heads nor pytorch_utils.prune_conv1d_layer /
find_pruneable_heads_and_indices), so this slices directly.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64
HIDDEN = NUM_HEADS * HEAD_DIM  # 768
PRUNE_HEAD = 5

REPO_ROOT = Path(__file__).resolve().parents[2]
IN_PATH = REPO_ROOT / "inputs" / "base" / "model.safetensors"
OUT_PATH = REPO_ROOT / "out" / "T2" / "model.safetensors"


def keep_head_indices(num_heads: int, head_dim: int, prune_head: int) -> torch.Tensor:
    heads = [h for h in range(num_heads) if h != prune_head]
    idx = torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim) for h in heads])
    return idx


def main() -> None:
    state = load_file(str(IN_PATH))
    assert len(state) == 160, f"expected 160 input tensors, got {len(state)}"

    keep_within_segment = keep_head_indices(NUM_HEADS, HEAD_DIM, PRUNE_HEAD)  # 704 idx into 768
    assert keep_within_segment.numel() == (NUM_HEADS - 1) * HEAD_DIM == 704

    out = {}
    for name, tensor in state.items():
        if name.endswith("attn.c_attn.weight"):
            # [768, 2304] = [768, 3*768] (q|k|v), heads are column blocks per segment.
            assert tensor.shape == (HIDDEN, 3 * HIDDEN), (name, tensor.shape)
            col_idx = torch.cat([seg * HIDDEN + keep_within_segment for seg in range(3)])
            new_t = tensor.index_select(1, col_idx).contiguous()
            assert new_t.shape == (HIDDEN, 3 * (NUM_HEADS - 1) * HEAD_DIM), (name, new_t.shape)
            out[name] = new_t
        elif name.endswith("attn.c_attn.bias"):
            assert tensor.shape == (3 * HIDDEN,), (name, tensor.shape)
            col_idx = torch.cat([seg * HIDDEN + keep_within_segment for seg in range(3)])
            new_t = tensor.index_select(0, col_idx).contiguous()
            assert new_t.shape == (3 * (NUM_HEADS - 1) * HEAD_DIM,), (name, new_t.shape)
            out[name] = new_t
        elif name.endswith("attn.c_proj.weight"):
            # [768, 768], heads are row blocks.
            assert tensor.shape == (HIDDEN, HIDDEN), (name, tensor.shape)
            new_t = tensor.index_select(0, keep_within_segment).contiguous()
            assert new_t.shape == ((NUM_HEADS - 1) * HEAD_DIM, HIDDEN), (name, new_t.shape)
            out[name] = new_t
        else:
            out[name] = tensor.clone().contiguous()

    # Required checks (fail loudly before writing).
    assert out["h.0.attn.c_attn.weight"].shape == (768, 2112), out["h.0.attn.c_attn.weight"].shape
    assert out["h.0.attn.c_attn.bias"].shape == (2112,), out["h.0.attn.c_attn.bias"].shape
    assert out["h.0.attn.c_proj.weight"].shape == (704, 768), out["h.0.attn.c_proj.weight"].shape
    assert len(out) == 160, f"expected 160 output tensors, got {len(out)}"

    for i in range(NUM_LAYERS):
        assert out[f"h.{i}.attn.c_attn.weight"].shape == (768, 2112)
        assert out[f"h.{i}.attn.c_attn.bias"].shape == (2112,)
        assert out[f"h.{i}.attn.c_proj.weight"].shape == (704, 768)
        assert out[f"h.{i}.attn.c_proj.bias"].shape == (768,)
        assert out[f"h.{i}.attn.bias"].shape == state[f"h.{i}.attn.bias"].shape

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(out)} tensors", file=sys.stderr)


if __name__ == "__main__":
    main()
