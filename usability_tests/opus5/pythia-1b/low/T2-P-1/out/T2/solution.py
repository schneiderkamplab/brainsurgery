"""T2: remove attention head 5 from every layer of Pythia-1B."""

import os

import torch
from safetensors.torch import load_file, save_file

IN_PATH = "inputs/base/model.safetensors"
OUT_DIR = "out/T2"
OUT_PATH = os.path.join(OUT_DIR, "model.safetensors")

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
PRUNE_HEAD = 5
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head in the fused projection


def keep_index(total, start, stop):
    """Indices of [0, total) with [start, stop) removed, order preserved."""
    return torch.cat([torch.arange(0, start), torch.arange(stop, total)])


def main():
    tensors = load_file(IN_PATH)
    n_in = len(tensors)

    qkv_keep = keep_index(NUM_HEADS * QKV_BLOCK, PRUNE_HEAD * QKV_BLOCK, (PRUNE_HEAD + 1) * QKV_BLOCK)
    dense_keep = keep_index(NUM_HEADS * HEAD_DIM, PRUNE_HEAD * HEAD_DIM, (PRUNE_HEAD + 1) * HEAD_DIM)

    for i in range(NUM_LAYERS):
        p = f"gpt_neox.layers.{i}.attention."
        w, b, d = p + "query_key_value.weight", p + "query_key_value.bias", p + "dense.weight"
        for k in (w, b, d):
            if k not in tensors:
                raise KeyError(f"missing expected tensor {k}")
        if tuple(tensors[w].shape) != (6144, 2048):
            raise ValueError(f"{w} has shape {tuple(tensors[w].shape)}, expected (6144, 2048)")
        if tuple(tensors[b].shape) != (6144,):
            raise ValueError(f"{b} has shape {tuple(tensors[b].shape)}, expected (6144,)")
        if tuple(tensors[d].shape) != (2048, 2048):
            raise ValueError(f"{d} has shape {tuple(tensors[d].shape)}, expected (2048, 2048)")

        tensors[w] = tensors[w][qkv_keep, :].contiguous()
        tensors[b] = tensors[b][qkv_keep].contiguous()
        tensors[d] = tensors[d][:, dense_keep].contiguous()

    # Required checks.
    checks = {
        "gpt_neox.layers.0.attention.query_key_value.weight": (5376, 2048),
        "gpt_neox.layers.0.attention.query_key_value.bias": (5376,),
        "gpt_neox.layers.0.attention.dense.weight": (2048, 1792),
    }
    for k, want in checks.items():
        got = tuple(tensors[k].shape)
        if got != want:
            raise AssertionError(f"{k}: shape {got}, expected {want}")
    if len(tensors) != 244:
        raise AssertionError(f"output has {len(tensors)} tensors, expected 244")
    if n_in != 244:
        raise AssertionError(f"input had {n_in} tensors, expected 244")

    os.makedirs(OUT_DIR, exist_ok=True)
    save_file(tensors, OUT_PATH)
    print(f"wrote {OUT_PATH} with {len(tensors)} tensors")


if __name__ == "__main__":
    main()
