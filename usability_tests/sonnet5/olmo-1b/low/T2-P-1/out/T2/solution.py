"""Prune attention head 5 from every layer of OLMo-1B-0724-hf.

Standalone script: loads the sharded safetensors checkpoint, slices head 5
out of q_proj/k_proj/v_proj (row blocks) and o_proj (column blocks) for each
of the 16 layers, checks shapes, and writes a single-file safetensors
checkpoint.
"""

import json
import os

import torch
from safetensors.torch import save_file
from safetensors import safe_open

HEAD_DIM = 128
NUM_HEADS = 16
HIDDEN = NUM_HEADS * HEAD_DIM  # 2048
PRUNE_HEAD = 5
NUM_LAYERS = 16

IN_DIR = "inputs/base"
OUT_DIR = "out/T2"
OUT_FILE = os.path.join(OUT_DIR, "model.safetensors")


def load_index(in_dir):
    with open(os.path.join(in_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    return index["weight_map"]


def load_all_tensors(in_dir, weight_map):
    tensors = {}
    shard_files = sorted(set(weight_map.values()))
    for shard in shard_files:
        path = os.path.join(in_dir, shard)
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def rows_to_keep(prune_head):
    start = prune_head * HEAD_DIM
    end = start + HEAD_DIM
    keep = list(range(0, start)) + list(range(end, HIDDEN))
    return keep


def main():
    weight_map = load_index(IN_DIR)
    tensors = load_all_tensors(IN_DIR, weight_map)

    keep_idx = torch.tensor(rows_to_keep(PRUNE_HEAD), dtype=torch.long)
    assert keep_idx.numel() == HIDDEN - HEAD_DIM == 1920

    out_tensors = {}
    for key, tensor in tensors.items():
        is_row_proj = any(
            key == f"model.layers.{i}.self_attn.{name}.weight"
            for i in range(NUM_LAYERS)
            for name in ("q_proj", "k_proj", "v_proj")
        )
        is_col_proj = any(
            key == f"model.layers.{i}.self_attn.o_proj.weight" for i in range(NUM_LAYERS)
        )
        if is_row_proj:
            assert tensor.shape == (HIDDEN, HIDDEN), f"{key} has unexpected shape {tensor.shape}"
            out_tensors[key] = tensor.index_select(0, keep_idx).contiguous()
        elif is_col_proj:
            assert tensor.shape == (HIDDEN, HIDDEN), f"{key} has unexpected shape {tensor.shape}"
            out_tensors[key] = tensor.index_select(1, keep_idx).contiguous()
        else:
            out_tensors[key] = tensor

    # Required checks
    for name in ("q_proj", "k_proj", "v_proj"):
        shape = tuple(out_tensors[f"model.layers.0.self_attn.{name}.weight"].shape)
        assert shape == (1920, 2048), f"model.layers.0.self_attn.{name}.weight has shape {shape}"
    shape = tuple(out_tensors["model.layers.0.self_attn.o_proj.weight"].shape)
    assert shape == (2048, 1920), f"model.layers.0.self_attn.o_proj.weight has shape {shape}"
    assert len(out_tensors) == 114, f"expected 114 tensors, got {len(out_tensors)}"

    os.makedirs(OUT_DIR, exist_ok=True)
    save_file(out_tensors, OUT_FILE)
    print(f"Wrote {OUT_FILE} with {len(out_tensors)} tensors")


if __name__ == "__main__":
    main()
