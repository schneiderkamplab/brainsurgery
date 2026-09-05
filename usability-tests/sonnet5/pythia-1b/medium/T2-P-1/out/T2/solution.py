"""
Prune head 5 (0-indexed) from every attention layer of Pythia-1B.

Pythia-1B / GPT-NeoX attention layout:
  - hidden_size = 2048, num_heads = 8, head_dim = 256
  - query_key_value.weight: [6144, 2048]. Rows are grouped per head into
    768-row blocks (interleaved q/k/v per head, not [q|k|v] segments):
    head h owns rows [768*h : 768*h+768), and within that block the first
    256 rows are q, next 256 are k, last 256 are v.
  - query_key_value.bias: [6144], same row layout.
  - dense.weight: [2048, 2048], nn.Linear [out, in]; heads are 256-wide
    column (input) blocks: head h owns columns [256*h : 256*h+256).
  - dense.bias: [2048], not per-head -> untouched.

Removing head H=5 means dropping its 768-row block from qkv weight/bias
and its 256-column block from dense.weight, for every one of the 16 layers.
"""

import sys

import torch
from safetensors.torch import load_file, save_file

HEAD_TO_PRUNE = 5
NUM_LAYERS = 16
HIDDEN_SIZE = 2048
NUM_HEADS = 8
HEAD_DIM = 256
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head in the fused qkv projection

IN_PATH = "inputs/base/model.safetensors"
OUT_PATH = "out/T2/model.safetensors"


def drop_row_block(tensor: torch.Tensor, block_size: int, head: int) -> torch.Tensor:
    """Remove rows [block_size*head : block_size*head+block_size) from dim 0."""
    start = block_size * head
    end = start + block_size
    return torch.cat([tensor[:start], tensor[end:]], dim=0).contiguous()


def drop_col_block(tensor: torch.Tensor, block_size: int, head: int) -> torch.Tensor:
    """Remove columns [block_size*head : block_size*head+block_size) from dim 1."""
    start = block_size * head
    end = start + block_size
    return torch.cat([tensor[:, :start], tensor[:, end:]], dim=1).contiguous()


def main() -> None:
    state_dict = load_file(IN_PATH)
    original_count = len(state_dict)

    out_state_dict = {}
    for name, tensor in state_dict.items():
        qkv_w_suffix = "attention.query_key_value.weight"
        qkv_b_suffix = "attention.query_key_value.bias"
        dense_w_suffix = "attention.dense.weight"

        if name.endswith(qkv_w_suffix):
            assert tensor.shape == (6144, 2048), (
                f"{name}: expected [6144, 2048], got {list(tensor.shape)}"
            )
            new_tensor = drop_row_block(tensor, QKV_BLOCK, HEAD_TO_PRUNE)
        elif name.endswith(qkv_b_suffix):
            assert tensor.shape == (6144,), f"{name}: expected [6144], got {list(tensor.shape)}"
            new_tensor = drop_row_block(tensor, QKV_BLOCK, HEAD_TO_PRUNE)
        elif name.endswith(dense_w_suffix):
            assert tensor.shape == (2048, 2048), (
                f"{name}: expected [2048, 2048], got {list(tensor.shape)}"
            )
            new_tensor = drop_col_block(tensor, HEAD_DIM, HEAD_TO_PRUNE)
        else:
            new_tensor = tensor
        out_state_dict[name] = new_tensor

    # Sanity: every layer's head-bearing tensors were touched.
    for i in range(NUM_LAYERS):
        qkv_w_name = f"gpt_neox.layers.{i}.attention.query_key_value.weight"
        qkv_b_name = f"gpt_neox.layers.{i}.attention.query_key_value.bias"
        dense_w_name = f"gpt_neox.layers.{i}.attention.dense.weight"
        assert qkv_w_name in out_state_dict, f"missing {qkv_w_name}"
        assert qkv_b_name in out_state_dict, f"missing {qkv_b_name}"
        assert dense_w_name in out_state_dict, f"missing {dense_w_name}"

    # Required checks (fail loudly before writing).
    qkv0_w = out_state_dict["gpt_neox.layers.0.attention.query_key_value.weight"]
    qkv0_b = out_state_dict["gpt_neox.layers.0.attention.query_key_value.bias"]
    dense0_w = out_state_dict["gpt_neox.layers.0.attention.dense.weight"]

    assert tuple(qkv0_w.shape) == (5376, 2048), (
        f"layer 0 qkv.weight: expected [5376, 2048], got {list(qkv0_w.shape)}"
    )
    assert tuple(qkv0_b.shape) == (5376,), (
        f"layer 0 qkv.bias: expected [5376], got {list(qkv0_b.shape)}"
    )
    assert tuple(dense0_w.shape) == (2048, 1792), (
        f"layer 0 dense.weight: expected [2048, 1792], got {list(dense0_w.shape)}"
    )
    assert len(out_state_dict) == 244, (
        f"expected 244 tensors in output, got {len(out_state_dict)}"
    )
    assert original_count == 244, f"expected 244 tensors in input, got {original_count}"

    # Also verify shapes hold for every layer, not just layer 0.
    for i in range(NUM_LAYERS):
        w = out_state_dict[f"gpt_neox.layers.{i}.attention.query_key_value.weight"]
        b = out_state_dict[f"gpt_neox.layers.{i}.attention.query_key_value.bias"]
        d = out_state_dict[f"gpt_neox.layers.{i}.attention.dense.weight"]
        assert tuple(w.shape) == (5376, 2048), f"layer {i} qkv.weight shape {list(w.shape)}"
        assert tuple(b.shape) == (5376,), f"layer {i} qkv.bias shape {list(b.shape)}"
        assert tuple(d.shape) == (2048, 1792), f"layer {i} dense.weight shape {list(d.shape)}"
        assert w.dtype == torch.float16 and b.dtype == torch.float16 and d.dtype == torch.float16

    save_file(out_state_dict, OUT_PATH)
    print(f"Wrote {OUT_PATH} with {len(out_state_dict)} tensors.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
