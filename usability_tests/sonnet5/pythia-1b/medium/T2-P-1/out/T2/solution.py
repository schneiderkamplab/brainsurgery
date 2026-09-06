"""Prune head 5 from every attention layer of Pythia-1B.

Pythia-1B (GPT-NeoX) has 16 layers, 8 heads of 256 dims each, hidden=2048.
The fused query_key_value projection is laid out per-head, interleaved:
for head h, rows [768*h : 768*h+768) hold that head's q (256), k (256), v (256)
rows, in that order. We remove head 5's 768-row block from qkv.weight/bias,
and remove head 5's 256-column block from attention.dense.weight.
"""

import torch
from safetensors.torch import load_file, save_file

IN_PATH = "inputs/base/model.safetensors"
OUT_PATH = "out/T2/model.safetensors"

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
HIDDEN = 2048
HEAD_TO_PRUNE = 5

QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head in fused qkv


def pruned_row_ranges(num_heads, head_block, head_to_prune):
    """Row index ranges to KEEP, in order, when removing one head's block."""
    start = head_to_prune * head_block
    end = start + head_block
    total = num_heads * head_block
    ranges = []
    if start > 0:
        ranges.append((0, start))
    if end < total:
        ranges.append((end, total))
    return ranges


def keep_rows(tensor, ranges):
    pieces = [tensor[s:e] for s, e in ranges]
    return torch.cat(pieces, dim=0)


def keep_cols(tensor, ranges):
    pieces = [tensor[:, s:e] for s, e in ranges]
    return torch.cat(pieces, dim=1)


def main():
    tensors = load_file(IN_PATH)
    assert len(tensors) == 244, f"expected 244 input tensors, got {len(tensors)}"

    qkv_row_ranges = pruned_row_ranges(NUM_HEADS, QKV_BLOCK, HEAD_TO_PRUNE)
    dense_col_ranges = pruned_row_ranges(NUM_HEADS, HEAD_DIM, HEAD_TO_PRUNE)

    # Sanity-check against the spec's explicit ranges.
    assert qkv_row_ranges == [(0, 3840), (4608, 6144)], qkv_row_ranges
    assert dense_col_ranges == [(0, 1280), (1536, 2048)], dense_col_ranges

    out = {}
    touched_prefixes = set()
    for i in range(NUM_LAYERS):
        prefix = f"gpt_neox.layers.{i}.attention"
        touched_prefixes.add(prefix)

        qkv_w_key = f"{prefix}.query_key_value.weight"
        qkv_b_key = f"{prefix}.query_key_value.bias"
        dense_w_key = f"{prefix}.dense.weight"

        qkv_w = tensors[qkv_w_key]
        qkv_b = tensors[qkv_b_key]
        dense_w = tensors[dense_w_key]

        assert qkv_w.shape == (NUM_HEADS * QKV_BLOCK, HIDDEN), (i, qkv_w.shape)
        assert qkv_b.shape == (NUM_HEADS * QKV_BLOCK,), (i, qkv_b.shape)
        assert dense_w.shape == (HIDDEN, HIDDEN), (i, dense_w.shape)

        new_qkv_w = keep_rows(qkv_w, qkv_row_ranges).contiguous()
        new_qkv_b = keep_rows(qkv_b, qkv_row_ranges).contiguous()
        new_dense_w = keep_cols(dense_w, dense_col_ranges).contiguous()

        out[qkv_w_key] = new_qkv_w
        out[qkv_b_key] = new_qkv_b
        out[dense_w_key] = new_dense_w

    # Everything else (including dense.bias, attention buffers, MLP tensors,
    # embeddings, norms, etc.) is copied through untouched.
    for key, tensor in tensors.items():
        if key in out:
            continue
        out[key] = tensor

    assert len(out) == len(tensors) == 244, (len(out), len(tensors))

    # Required checks before writing.
    assert out["gpt_neox.layers.0.attention.query_key_value.weight"].shape == (5376, 2048)
    assert out["gpt_neox.layers.0.attention.query_key_value.bias"].shape == (5376,)
    assert out["gpt_neox.layers.0.attention.dense.weight"].shape == (2048, 1792)
    assert len(out) == 244

    # Dtype and untouched-tensor sanity checks across all layers.
    for i in range(NUM_LAYERS):
        prefix = f"gpt_neox.layers.{i}.attention"
        assert out[f"{prefix}.query_key_value.weight"].shape == (5376, 2048)
        assert out[f"{prefix}.query_key_value.bias"].shape == (5376,)
        assert out[f"{prefix}.dense.weight"].shape == (2048, 1792)
        assert out[f"{prefix}.dense.weight"].dtype == torch.float16
        assert out[f"{prefix}.query_key_value.weight"].dtype == torch.float16
        # dense.bias must be unchanged
        assert torch.equal(out[f"{prefix}.dense.bias"], tensors[f"{prefix}.dense.bias"])
        assert out[f"{prefix}.dense.bias"].shape == (2048,)

    save_file(out, OUT_PATH)
    print(f"Wrote {OUT_PATH} with {len(out)} tensors.")


if __name__ == "__main__":
    main()
