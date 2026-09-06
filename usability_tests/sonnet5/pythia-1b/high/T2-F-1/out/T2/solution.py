"""T2: prune head 5 from every layer of Pythia-1B (GPT-NeoX architecture).

Plain script on top of `safetensors` + `torch` (both on the F-allowed list).
`transformers.prune_heads` was considered but GPTNeoX's fused, interleaved
query_key_value layout (per-head 768-row blocks of q/k/v, not a global
[q|k|v] split) is not something the generic `PreTrainedModel.prune_heads`
API models, so a direct slice-and-reassemble on the safetensors state dict is
the reliable route here: it lets every row/column range be pinned exactly to
the layout described in TASK.md and checked before writing.

Usage: python solution.py <in_file> <out_file>
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
HIDDEN = NUM_HEADS * HEAD_DIM  # 2048
QKV_BLOCK = 3 * HEAD_DIM  # 768: one head's q,k,v rows in the fused projection
PRUNE_HEAD = 5
EXPECTED_TENSOR_COUNT = 244


def kept_row_ranges(head_dim_block: int, num_heads: int, prune_head: int) -> list[range]:
    """Row/col ranges to keep when dropping `prune_head`, in original order."""
    ranges = []
    if prune_head > 0:
        ranges.append(range(0, prune_head * head_dim_block))
    if prune_head < num_heads - 1:
        ranges.append(range((prune_head + 1) * head_dim_block, num_heads * head_dim_block))
    return ranges


def prune_head_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(state_dict)

    qkv_ranges = kept_row_ranges(QKV_BLOCK, NUM_HEADS, PRUNE_HEAD)  # rows of qkv weight/bias
    dense_ranges = kept_row_ranges(HEAD_DIM, NUM_HEADS, PRUNE_HEAD)  # columns of dense.weight

    for i in range(NUM_LAYERS):
        prefix = f"gpt_neox.layers.{i}.attention"

        qkv_w_key = f"{prefix}.query_key_value.weight"
        qkv_b_key = f"{prefix}.query_key_value.bias"
        dense_w_key = f"{prefix}.dense.weight"

        for key in (qkv_w_key, qkv_b_key, dense_w_key):
            if key not in state_dict:
                raise KeyError(f"missing expected tensor: {key}")

        qkv_w = state_dict[qkv_w_key]
        if qkv_w.shape != (NUM_HEADS * QKV_BLOCK, HIDDEN):
            raise ValueError(f"{qkv_w_key} has shape {tuple(qkv_w.shape)}, expected {(NUM_HEADS * QKV_BLOCK, HIDDEN)}")
        out[qkv_w_key] = torch.cat([qkv_w[r] for r in qkv_ranges], dim=0).contiguous()

        qkv_b = state_dict[qkv_b_key]
        if qkv_b.shape != (NUM_HEADS * QKV_BLOCK,):
            raise ValueError(f"{qkv_b_key} has shape {tuple(qkv_b.shape)}, expected {(NUM_HEADS * QKV_BLOCK,)}")
        out[qkv_b_key] = torch.cat([qkv_b[r] for r in qkv_ranges], dim=0).contiguous()

        dense_w = state_dict[dense_w_key]
        if dense_w.shape != (HIDDEN, HIDDEN):
            raise ValueError(f"{dense_w_key} has shape {tuple(dense_w.shape)}, expected {(HIDDEN, HIDDEN)}")
        out[dense_w_key] = torch.cat([dense_w[:, r] for r in dense_ranges], dim=1).contiguous()

    return out


def check_result(out: dict[str, torch.Tensor]) -> None:
    expected_qkv_w = (NUM_HEADS - 1) * QKV_BLOCK  # 5376
    expected_qkv_b = expected_qkv_w
    expected_dense_w = (HIDDEN, (NUM_HEADS - 1) * HEAD_DIM)  # (2048, 1792)

    layer0_qkv_w = out["gpt_neox.layers.0.attention.query_key_value.weight"]
    if tuple(layer0_qkv_w.shape) != (expected_qkv_w, HIDDEN):
        raise AssertionError(
            f"layer 0 query_key_value.weight shape {tuple(layer0_qkv_w.shape)} != {(expected_qkv_w, HIDDEN)}"
        )

    layer0_qkv_b = out["gpt_neox.layers.0.attention.query_key_value.bias"]
    if tuple(layer0_qkv_b.shape) != (expected_qkv_b,):
        raise AssertionError(f"layer 0 query_key_value.bias shape {tuple(layer0_qkv_b.shape)} != {(expected_qkv_b,)}")

    layer0_dense_w = out["gpt_neox.layers.0.attention.dense.weight"]
    if tuple(layer0_dense_w.shape) != expected_dense_w:
        raise AssertionError(f"layer 0 dense.weight shape {tuple(layer0_dense_w.shape)} != {expected_dense_w}")

    if len(out) != EXPECTED_TENSOR_COUNT:
        raise AssertionError(f"output has {len(out)} tensors, expected {EXPECTED_TENSOR_COUNT}")

    # All other per-layer head-bearing tensors, checked for every layer, not just layer 0.
    for i in range(NUM_LAYERS):
        prefix = f"gpt_neox.layers.{i}.attention"
        w = out[f"{prefix}.query_key_value.weight"]
        b = out[f"{prefix}.query_key_value.bias"]
        d = out[f"{prefix}.dense.weight"]
        if tuple(w.shape) != (expected_qkv_w, HIDDEN):
            raise AssertionError(f"layer {i} query_key_value.weight shape {tuple(w.shape)} != {(expected_qkv_w, HIDDEN)}")
        if tuple(b.shape) != (expected_qkv_b,):
            raise AssertionError(f"layer {i} query_key_value.bias shape {tuple(b.shape)} != {(expected_qkv_b,)}")
        if tuple(d.shape) != expected_dense_w:
            raise AssertionError(f"layer {i} dense.weight shape {tuple(d.shape)} != {expected_dense_w}")


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python solution.py <in_file> <out_file>", file=sys.stderr)
        raise SystemExit(2)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    state_dict = load_file(str(in_path))
    if len(state_dict) != EXPECTED_TENSOR_COUNT:
        raise AssertionError(f"input has {len(state_dict)} tensors, expected {EXPECTED_TENSOR_COUNT}")

    pruned = prune_head_from_state_dict(state_dict)
    check_result(pruned)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(pruned, str(out_path))
    print(f"wrote {out_path} with {len(pruned)} tensors")


if __name__ == "__main__":
    main()
