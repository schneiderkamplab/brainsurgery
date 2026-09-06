"""
T2: Structured attention-head pruning (OLMo-1B-0724-hf)

Removes head index 5 (0-indexed) from every layer's q_proj, k_proj, v_proj
(row blocks) and o_proj (column blocks), leaving 15 heads of 128 dims each.
Every other tensor is copied through unchanged.
"""

import json
import os

import torch
from safetensors.torch import save_file
from safetensors import safe_open

INPUT_DIR = "inputs/base"
INDEX_FILE = os.path.join(INPUT_DIR, "model.safetensors.index.json")
OUTPUT_DIR = "out/T2"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "model.safetensors")

NUM_LAYERS = 16
NUM_HEADS = 16
HEAD_DIM = 128
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM  # 2048
PRUNE_HEAD = 5

# Row/col range removed by pruning head 5: [640, 768)
REMOVE_START = PRUNE_HEAD * HEAD_DIM
REMOVE_END = REMOVE_START + HEAD_DIM


def load_all_tensors():
    with open(INDEX_FILE) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    tensors = {}
    shard_files = sorted(set(weight_map.values()))
    handles = {
        shard: safe_open(os.path.join(INPUT_DIR, shard), framework="pt")
        for shard in shard_files
    }
    try:
        for name, shard in weight_map.items():
            tensors[name] = handles[shard].get_tensor(name)
    finally:
        # safe_open handles don't need explicit closing, but drop refs anyway.
        del handles
    return tensors


def prune_rows(t: torch.Tensor) -> torch.Tensor:
    # heads are row blocks: keep rows 0..639 and 768..2047
    return torch.cat([t[:REMOVE_START], t[REMOVE_END:]], dim=0).contiguous()


def prune_cols(t: torch.Tensor) -> torch.Tensor:
    # heads are column blocks: keep cols 0..639 and 768..2047
    return torch.cat([t[:, :REMOVE_START], t[:, REMOVE_END:]], dim=1).contiguous()


def main():
    tensors = load_all_tensors()
    original_count = len(tensors)

    out_tensors = {}
    for name, t in tensors.items():
        is_q = name.endswith("self_attn.q_proj.weight")
        is_k = name.endswith("self_attn.k_proj.weight")
        is_v = name.endswith("self_attn.v_proj.weight")
        is_o = name.endswith("self_attn.o_proj.weight")

        if is_q or is_k or is_v:
            assert t.shape == (HIDDEN_SIZE, HIDDEN_SIZE), (
                f"{name}: expected [{HIDDEN_SIZE}, {HIDDEN_SIZE}], got {tuple(t.shape)}"
            )
            new_t = prune_rows(t)
        elif is_o:
            assert t.shape == (HIDDEN_SIZE, HIDDEN_SIZE), (
                f"{name}: expected [{HIDDEN_SIZE}, {HIDDEN_SIZE}], got {tuple(t.shape)}"
            )
            new_t = prune_cols(t)
        else:
            new_t = t

        out_tensors[name] = new_t

    # --- Required checks ---
    assert len(out_tensors) == original_count == 114, (
        f"expected 114 tensors, got {len(out_tensors)} (input had {original_count})"
    )

    expected_qkv_shape = (HIDDEN_SIZE - HEAD_DIM, HIDDEN_SIZE)  # [1920, 2048]
    expected_o_shape = (HIDDEN_SIZE, HIDDEN_SIZE - HEAD_DIM)  # [2048, 1920]

    for proj, expected in [
        ("q_proj", expected_qkv_shape),
        ("k_proj", expected_qkv_shape),
        ("v_proj", expected_qkv_shape),
        ("o_proj", expected_o_shape),
    ]:
        name = f"model.layers.0.self_attn.{proj}.weight"
        actual = tuple(out_tensors[name].shape)
        assert actual == expected, f"{name}: expected {expected}, got {actual}"

    for i in range(NUM_LAYERS):
        for proj, expected in [
            ("q_proj", expected_qkv_shape),
            ("k_proj", expected_qkv_shape),
            ("v_proj", expected_qkv_shape),
            ("o_proj", expected_o_shape),
        ]:
            name = f"model.layers.{i}.self_attn.{proj}.weight"
            actual = tuple(out_tensors[name].shape)
            assert actual == expected, f"{name}: expected {expected}, got {actual}"

    # Names must be unchanged.
    assert set(out_tensors.keys()) == set(tensors.keys())

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_file(out_tensors, OUTPUT_FILE)
    print(f"Wrote {len(out_tensors)} tensors to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
