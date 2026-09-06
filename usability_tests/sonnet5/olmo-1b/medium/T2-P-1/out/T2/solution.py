"""
T2: Structured attention-head pruning (OLMo-1B-0724-hf)

Remove head 5 (0-indexed) from every layer's q_proj, k_proj, v_proj (row
blocks, out-features axis) and o_proj (column blocks, in-features axis).

Head layout: 16 heads x 128 dims = 2048. Head 5 occupies rows/cols
[640:768). Kept ranges: [0:640) and [768:2048).
"""

import json
import os

import torch
from safetensors.torch import save_file

INPUT_DIR = "inputs/base"
OUTPUT_DIR = "out/T2"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "model.safetensors")

NUM_LAYERS = 16
NUM_HEADS = 16
HEAD_DIM = 128
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM  # 2048
PRUNE_HEAD = 5

KEEP_START = PRUNE_HEAD * HEAD_DIM  # 640
KEEP_END = (PRUNE_HEAD + 1) * HEAD_DIM  # 768


def load_all_tensors(input_dir):
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    tensors = {}
    shard_files = sorted(set(index["weight_map"].values()))
    for shard_file in shard_files:
        shard_path = os.path.join(input_dir, shard_file)
        from safetensors import safe_open

        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    expected_keys = set(index["weight_map"].keys())
    assert set(tensors.keys()) == expected_keys, (
        f"Loaded tensor keys do not match index: "
        f"missing={expected_keys - set(tensors.keys())}, "
        f"extra={set(tensors.keys()) - expected_keys}"
    )
    return tensors


def prune_row_blocks(tensor):
    """Remove rows [KEEP_START:KEEP_END) — used for q/k/v_proj (out-features axis)."""
    assert tensor.shape[0] == HIDDEN_SIZE, f"expected {HIDDEN_SIZE} rows, got {tensor.shape[0]}"
    kept = torch.cat([tensor[:KEEP_START], tensor[KEEP_END:]], dim=0)
    return kept.contiguous()


def prune_col_blocks(tensor):
    """Remove columns [KEEP_START:KEEP_END) — used for o_proj (in-features axis)."""
    assert tensor.shape[1] == HIDDEN_SIZE, f"expected {HIDDEN_SIZE} cols, got {tensor.shape[1]}"
    kept = torch.cat([tensor[:, :KEEP_START], tensor[:, KEEP_END:]], dim=1)
    return kept.contiguous()


def main():
    tensors = load_all_tensors(INPUT_DIR)
    original_count = len(tensors)

    output = dict(tensors)  # start as a copy; per-layer projections get overwritten

    for i in range(NUM_LAYERS):
        prefix = f"model.layers.{i}.self_attn."
        q_key = prefix + "q_proj.weight"
        k_key = prefix + "k_proj.weight"
        v_key = prefix + "v_proj.weight"
        o_key = prefix + "o_proj.weight"

        for key in (q_key, k_key, v_key, o_key):
            assert key in tensors, f"missing expected tensor: {key}"

        output[q_key] = prune_row_blocks(tensors[q_key])
        output[k_key] = prune_row_blocks(tensors[k_key])
        output[v_key] = prune_row_blocks(tensors[v_key])
        output[o_key] = prune_col_blocks(tensors[o_key])

    # --- Required checks ---
    expected_reduced_shape = (HIDDEN_SIZE - HEAD_DIM, HIDDEN_SIZE)  # (1920, 2048)
    expected_o_shape = (HIDDEN_SIZE, HIDDEN_SIZE - HEAD_DIM)  # (2048, 1920)

    q0 = output["model.layers.0.self_attn.q_proj.weight"]
    k0 = output["model.layers.0.self_attn.k_proj.weight"]
    v0 = output["model.layers.0.self_attn.v_proj.weight"]
    o0 = output["model.layers.0.self_attn.o_proj.weight"]

    assert tuple(q0.shape) == expected_reduced_shape, (
        f"q_proj.weight shape check failed: got {tuple(q0.shape)}, expected {expected_reduced_shape}"
    )
    assert tuple(k0.shape) == expected_reduced_shape, (
        f"k_proj.weight shape check failed: got {tuple(k0.shape)}, expected {expected_reduced_shape}"
    )
    assert tuple(v0.shape) == expected_reduced_shape, (
        f"v_proj.weight shape check failed: got {tuple(v0.shape)}, expected {expected_reduced_shape}"
    )
    assert tuple(o0.shape) == expected_o_shape, (
        f"o_proj.weight shape check failed: got {tuple(o0.shape)}, expected {expected_o_shape}"
    )
    assert len(output) == original_count == 114, (
        f"tensor count check failed: got {len(output)}, expected 114"
    )

    # Also verify all layers, not just layer 0, to catch off-by-one/indexing bugs early.
    for i in range(NUM_LAYERS):
        prefix = f"model.layers.{i}.self_attn."
        assert tuple(output[prefix + "q_proj.weight"].shape) == expected_reduced_shape
        assert tuple(output[prefix + "k_proj.weight"].shape) == expected_reduced_shape
        assert tuple(output[prefix + "v_proj.weight"].shape) == expected_reduced_shape
        assert tuple(output[prefix + "o_proj.weight"].shape) == expected_o_shape

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_file(output, OUTPUT_FILE)
    print(f"Wrote {len(output)} tensors to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
