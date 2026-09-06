"""
Structured attention-head pruning for GPT-2 (124M): remove head 5 from every
layer's attention projections.

GPT-2 stores c_attn as a fused [q | k | v] projection with columns
(out-features axis for Conv1D), 768 wide per segment, 12 heads of 64 dims
each. c_proj is the output projection with heads as row blocks.
"""

import torch
from safetensors.torch import load_file, save_file

HEAD = 5
NUM_HEADS = 12
HEAD_DIM = 64
HIDDEN = NUM_HEADS * HEAD_DIM  # 768
NUM_LAYERS = 12

IN_PATH = "inputs/base/model.safetensors"
OUT_PATH = "out/T2/model.safetensors"


def head_keep_indices(num_heads: int, head_dim: int, drop_head: int) -> torch.Tensor:
    """Indices (0-based) to keep along a HIDDEN-wide axis with heads as blocks."""
    keep = [
        h * head_dim + d
        for h in range(num_heads)
        if h != drop_head
        for d in range(head_dim)
    ]
    return torch.tensor(keep, dtype=torch.long)


def main():
    state_dict = load_file(IN_PATH)

    keep_idx = head_keep_indices(NUM_HEADS, HEAD_DIM, HEAD)  # length 704, block-ordered

    out = {}
    for name, tensor in state_dict.items():
        if name.endswith("attn.c_attn.weight"):
            # shape [768, 2304] = [in, 3*768]; heads are column blocks within
            # each 768-wide q/k/v segment.
            assert tensor.shape == (HIDDEN, 3 * HIDDEN), (name, tensor.shape)
            segments = []
            for seg in range(3):
                seg_cols = tensor[:, seg * HIDDEN : (seg + 1) * HIDDEN]
                segments.append(seg_cols[:, keep_idx])
            out[name] = torch.cat(segments, dim=1).contiguous()
        elif name.endswith("attn.c_attn.bias"):
            assert tensor.shape == (3 * HIDDEN,), (name, tensor.shape)
            segments = []
            for seg in range(3):
                seg_vals = tensor[seg * HIDDEN : (seg + 1) * HIDDEN]
                segments.append(seg_vals[keep_idx])
            out[name] = torch.cat(segments, dim=0).contiguous()
        elif name.endswith("attn.c_proj.weight"):
            # shape [768, 768]; heads are row blocks.
            assert tensor.shape == (HIDDEN, HIDDEN), (name, tensor.shape)
            out[name] = tensor[keep_idx, :].contiguous()
        else:
            out[name] = tensor

    # Required checks
    assert out["h.0.attn.c_attn.weight"].shape == (768, 2112), out["h.0.attn.c_attn.weight"].shape
    assert out["h.0.attn.c_attn.bias"].shape == (2112,), out["h.0.attn.c_attn.bias"].shape
    assert out["h.0.attn.c_proj.weight"].shape == (704, 768), out["h.0.attn.c_proj.weight"].shape
    assert len(out) == 160, f"expected 160 tensors, got {len(out)}"

    # Sanity: verify pruned layers actually match manual column selection for
    # layer 0 (explicit block ranges from TASK.md), to catch any indexing bug.
    orig = state_dict["h.0.attn.c_attn.weight"]
    expected_cols = list(range(0, 320)) + list(range(384, 768)) + \
        list(range(768, 1088)) + list(range(1152, 1536)) + \
        list(range(1536, 1856)) + list(range(1920, 2304))
    expected = orig[:, expected_cols]
    assert torch.equal(out["h.0.attn.c_attn.weight"], expected), "column selection mismatch"

    save_file(out, OUT_PATH)
    print(f"Wrote {OUT_PATH} with {len(out)} tensors")


if __name__ == "__main__":
    main()
