"""T2: structured attention-head pruning for OLMo-1B-0724-hf.

Removes head 5 (0-indexed) from every layer's q/k/v/o attention projections
by slicing out its 128-row/column block, keeping the surrounding blocks in
order. Everything else is copied unchanged.

Uses safetensors directly (not transformers `prune_heads`) so the transform
is an exact, auditable slice-and-concat operation with no model
construction, dtype casting, or config side effects in the loop.
"""

import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

INPUT_DIR = Path(__file__).resolve().parents[2] / "inputs" / "base"
OUTPUT_PATH = Path(__file__).resolve().parent / "model.safetensors"

NUM_LAYERS = 16
NUM_HEADS = 16
HEAD_DIM = 128
HEAD_TO_REMOVE = 5
HIDDEN = NUM_HEADS * HEAD_DIM  # 2048
EXPECTED_ROWS = (NUM_HEADS - 1) * HEAD_DIM  # 1920

ROW_TENSOR_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
)
COL_TENSOR_SUFFIX = "self_attn.o_proj.weight"


def torch_cat_rows(tensor, start, end):
    import torch

    return torch.cat([tensor[:start, :], tensor[end:, :]], dim=0)


def torch_cat_cols(tensor, start, end):
    import torch

    return torch.cat([tensor[:, :start], tensor[:, end:]], dim=1)


def main():
    index_path = INPUT_DIR / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    expected_keys = set(weight_map.keys())
    assert len(expected_keys) == 114, f"expected 114 tensors in index, got {len(expected_keys)}"

    # Open all shard files referenced by the index.
    shard_files = sorted(set(weight_map.values()))
    handles = {shard: safe_open(str(INPUT_DIR / shard), framework="pt") for shard in shard_files}

    result = {}
    row_start = HEAD_TO_REMOVE * HEAD_DIM
    row_end = row_start + HEAD_DIM

    for key, shard in weight_map.items():
        tensor = handles[shard].get_tensor(key)

        if key.endswith(ROW_TENSOR_SUFFIXES):
            assert tensor.shape == (HIDDEN, HIDDEN), (
                f"{key}: expected shape ({HIDDEN}, {HIDDEN}), got {tuple(tensor.shape)}"
            )
            tensor = torch_cat_rows(tensor, row_start, row_end)
            assert tensor.shape == (EXPECTED_ROWS, HIDDEN), (
                f"{key}: expected shape ({EXPECTED_ROWS}, {HIDDEN}) after pruning, "
                f"got {tuple(tensor.shape)}"
            )
        elif key.endswith(COL_TENSOR_SUFFIX):
            assert tensor.shape == (HIDDEN, HIDDEN), (
                f"{key}: expected shape ({HIDDEN}, {HIDDEN}), got {tuple(tensor.shape)}"
            )
            tensor = torch_cat_cols(tensor, row_start, row_end)
            assert tensor.shape == (HIDDEN, EXPECTED_ROWS), (
                f"{key}: expected shape ({HIDDEN}, {EXPECTED_ROWS}) after pruning, "
                f"got {tuple(tensor.shape)}"
            )

        result[key] = tensor.contiguous()

    assert len(result) == 114, f"expected 114 output tensors, got {len(result)}"

    # Required checks: shapes for layer 0's projections, exact tensor count.
    for name, suffix, expected_shape in [
        ("q_proj", "self_attn.q_proj.weight", (EXPECTED_ROWS, HIDDEN)),
        ("k_proj", "self_attn.k_proj.weight", (EXPECTED_ROWS, HIDDEN)),
        ("v_proj", "self_attn.v_proj.weight", (EXPECTED_ROWS, HIDDEN)),
        ("o_proj", "self_attn.o_proj.weight", (HIDDEN, EXPECTED_ROWS)),
    ]:
        key = f"model.layers.0.{suffix}"
        actual_shape = tuple(result[key].shape)
        assert actual_shape == expected_shape, (
            f"required check failed: {key} has shape {actual_shape}, expected {expected_shape}"
        )

    assert len(result) == 114, "required check failed: output must have exactly 114 tensors"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(result, str(OUTPUT_PATH))
    print(f"Wrote {len(result)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
