"""
Structured attention-head pruning for OLMo-1B-0724-hf.

Removes head 5 (0-indexed) from every layer's self-attention block by
slicing the corresponding rows (for q/k/v projections) or columns (for
o_proj) out of the projection weight matrices. All other tensors are
copied through unchanged.
"""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INPUT_DIR = "inputs/base"
OUTPUT_DIR = "out/T2"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "model.safetensors")

NUM_LAYERS = 16
NUM_HEADS = 16
HEAD_DIM = 128
HIDDEN_SIZE = 2048
HEAD_TO_PRUNE = 5

assert NUM_HEADS * HEAD_DIM == HIDDEN_SIZE

# Row/column indices to KEEP: everything except the pruned head's block.
prune_start = HEAD_TO_PRUNE * HEAD_DIM  # 640
prune_end = prune_start + HEAD_DIM  # 768
keep_indices = torch.cat(
    [
        torch.arange(0, prune_start),
        torch.arange(prune_end, HIDDEN_SIZE),
    ]
)
assert keep_indices.numel() == HIDDEN_SIZE - HEAD_DIM == 1920
assert prune_start == 640 and prune_end == 768


def load_all_tensors(input_dir):
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    tensors = {}
    shard_files = sorted(set(weight_map.values()))
    for shard_file in shard_files:
        shard_path = os.path.join(input_dir, shard_file)
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # sanity: every key in the index was actually loaded
    assert set(tensors.keys()) == set(weight_map.keys())
    return tensors


def main():
    tensors = load_all_tensors(INPUT_DIR)

    expected_num_tensors = len(tensors)
    print(f"Loaded {expected_num_tensors} tensors from {INPUT_DIR}")

    output = {}
    pruned_row_keys = set()
    pruned_col_keys = set()
    for i in range(NUM_LAYERS):
        pruned_row_keys.add(f"model.layers.{i}.self_attn.q_proj.weight")
        pruned_row_keys.add(f"model.layers.{i}.self_attn.k_proj.weight")
        pruned_row_keys.add(f"model.layers.{i}.self_attn.v_proj.weight")
        pruned_col_keys.add(f"model.layers.{i}.self_attn.o_proj.weight")

    for key, tensor in tensors.items():
        if key in pruned_row_keys:
            assert tensor.shape == (HIDDEN_SIZE, HIDDEN_SIZE), (
                f"{key} has unexpected shape {tuple(tensor.shape)}"
            )
            new_tensor = tensor.index_select(0, keep_indices).contiguous()
            assert new_tensor.shape == (1920, HIDDEN_SIZE)
        elif key in pruned_col_keys:
            assert tensor.shape == (HIDDEN_SIZE, HIDDEN_SIZE), (
                f"{key} has unexpected shape {tuple(tensor.shape)}"
            )
            new_tensor = tensor.index_select(1, keep_indices).contiguous()
            assert new_tensor.shape == (HIDDEN_SIZE, 1920)
        else:
            new_tensor = tensor
        output[key] = new_tensor

    # Required checks -----------------------------------------------------
    assert output["model.layers.0.self_attn.q_proj.weight"].shape == (1920, 2048)
    assert output["model.layers.0.self_attn.k_proj.weight"].shape == (1920, 2048)
    assert output["model.layers.0.self_attn.v_proj.weight"].shape == (1920, 2048)
    assert output["model.layers.0.self_attn.o_proj.weight"].shape == (2048, 1920)
    assert len(output) == 114, f"expected 114 tensors, got {len(output)}"
    assert len(output) == expected_num_tensors

    # Check every layer's pruned tensors, not just layer 0
    for i in range(NUM_LAYERS):
        for name, expected_shape in (
            (f"model.layers.{i}.self_attn.q_proj.weight", (1920, 2048)),
            (f"model.layers.{i}.self_attn.k_proj.weight", (1920, 2048)),
            (f"model.layers.{i}.self_attn.v_proj.weight", (1920, 2048)),
            (f"model.layers.{i}.self_attn.o_proj.weight", (2048, 1920)),
        ):
            assert output[name].shape == expected_shape, (
                f"{name}: expected {expected_shape}, got {tuple(output[name].shape)}"
            )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_file(output, OUTPUT_PATH)
    print(f"Wrote {len(output)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
