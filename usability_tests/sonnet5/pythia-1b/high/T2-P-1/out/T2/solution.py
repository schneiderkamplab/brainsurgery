"""
Structured attention-head pruning for Pythia-1B.

Removes head 5 (0-indexed) from every layer's attention block:
  - attention.query_key_value.weight / .bias: drop the 768-row block
    belonging to head 5 out of the 8 interleaved q/k/v blocks.
  - attention.dense.weight: drop the corresponding 256-column block
    (dense consumes concatenated per-head outputs in head order).
Everything else is copied through unchanged.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
IN_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUT_PATH = HERE / "model.safetensors"

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head in the fused qkv projection
HEAD_TO_PRUNE = 5

EXPECTED_TENSOR_COUNT = 244


def rows_to_keep(num_heads: int, head_block: int, pruned_head: int) -> list[int]:
    """Row/col indices to keep, in order, after dropping one head's block."""
    keep = []
    for h in range(num_heads):
        if h == pruned_head:
            continue
        start = h * head_block
        keep.extend(range(start, start + head_block))
    return keep


def main() -> None:
    state_dict = load_file(str(IN_PATH))

    qkv_keep = rows_to_keep(NUM_HEADS, QKV_BLOCK, HEAD_TO_PRUNE)
    dense_keep = rows_to_keep(NUM_HEADS, HEAD_DIM, HEAD_TO_PRUNE)
    qkv_keep_t = torch.tensor(qkv_keep, dtype=torch.long)
    dense_keep_t = torch.tensor(dense_keep, dtype=torch.long)

    assert qkv_keep == list(range(0, 3840)) + list(range(4608, 6144)), (
        "qkv row selection does not match the expected keep ranges"
    )
    assert dense_keep == list(range(0, 1280)) + list(range(1536, 2048)), (
        "dense column selection does not match the expected keep ranges"
    )

    out_state_dict = {}
    for name, tensor in state_dict.items():
        is_qkv_weight = False
        is_qkv_bias = False
        is_dense_weight = False
        for i in range(NUM_LAYERS):
            prefix = f"gpt_neox.layers.{i}.attention."
            if name == prefix + "query_key_value.weight":
                is_qkv_weight = True
                break
            if name == prefix + "query_key_value.bias":
                is_qkv_bias = True
                break
            if name == prefix + "dense.weight":
                is_dense_weight = True
                break

        if is_qkv_weight:
            assert tensor.shape == (6144, 2048), (
                f"{name}: expected [6144, 2048], got {list(tensor.shape)}"
            )
            new_tensor = tensor.index_select(0, qkv_keep_t).contiguous()
        elif is_qkv_bias:
            assert tensor.shape == (6144,), f"{name}: expected [6144], got {list(tensor.shape)}"
            new_tensor = tensor.index_select(0, qkv_keep_t).contiguous()
        elif is_dense_weight:
            assert tensor.shape == (2048, 2048), (
                f"{name}: expected [2048, 2048], got {list(tensor.shape)}"
            )
            new_tensor = tensor.index_select(1, dense_keep_t).contiguous()
        else:
            new_tensor = tensor

        out_state_dict[name] = new_tensor

    # Required checks: fail loudly before writing anything.
    l0 = "gpt_neox.layers.0.attention."
    assert out_state_dict[l0 + "query_key_value.weight"].shape == (5376, 2048), (
        "layer 0 query_key_value.weight has wrong shape after pruning"
    )
    assert out_state_dict[l0 + "query_key_value.bias"].shape == (5376,), (
        "layer 0 query_key_value.bias has wrong shape after pruning"
    )
    assert out_state_dict[l0 + "dense.weight"].shape == (2048, 1792), (
        "layer 0 dense.weight has wrong shape after pruning"
    )
    assert len(out_state_dict) == EXPECTED_TENSOR_COUNT, (
        f"expected {EXPECTED_TENSOR_COUNT} tensors, got {len(out_state_dict)}"
    )
    assert len(state_dict) == len(out_state_dict), "tensor count changed unexpectedly"
    assert set(state_dict.keys()) == set(out_state_dict.keys()), "tensor names changed"

    for i in range(NUM_LAYERS):
        prefix = f"gpt_neox.layers.{i}.attention."
        assert out_state_dict[prefix + "query_key_value.weight"].shape == (5376, 2048)
        assert out_state_dict[prefix + "query_key_value.bias"].shape == (5376,)
        assert out_state_dict[prefix + "dense.weight"].shape == (2048, 1792)
        assert out_state_dict[prefix + "dense.bias"].shape == (2048,)
        assert out_state_dict[prefix + "dense.bias"].equal(state_dict[prefix + "dense.bias"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_state_dict, str(OUT_PATH))

    print(f"Wrote {len(out_state_dict)} tensors to {OUT_PATH}")


if __name__ == "__main__":
    main()
