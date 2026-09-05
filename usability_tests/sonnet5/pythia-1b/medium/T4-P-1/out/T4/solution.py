"""
T4: Task-vector merge of two Pythia-1B fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])
for the 64 MLP tensors, computed in float32 and cast back to float16.
Every other tensor is copied unchanged from base.
"""

import re
import sys

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
NUM_LAYERS = 16

BASE_PATH = "inputs/base/model.safetensors"
FT1_PATH = "inputs/ft1/model.safetensors"
FT2_PATH = "inputs/ft2/model.safetensors"
OUT_PATH = "out/T4/model.safetensors"

MLP_SUFFIXES = (
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    "mlp.dense_4h_to_h.bias",
)


def mlp_tensor_names():
    names = set()
    for i in range(NUM_LAYERS):
        for suffix in MLP_SUFFIXES:
            names.add(f"gpt_neox.layers.{i}.{suffix}")
    return names


def main():
    base = load_file(BASE_PATH)
    ft1 = load_file(FT1_PATH)
    ft2 = load_file(FT2_PATH)

    base_keys = set(base.keys())
    ft1_keys = set(ft1.keys())
    ft2_keys = set(ft2.keys())

    if not (base_keys == ft1_keys == ft2_keys):
        missing_ft1 = base_keys - ft1_keys
        extra_ft1 = ft1_keys - base_keys
        missing_ft2 = base_keys - ft2_keys
        extra_ft2 = ft2_keys - base_keys
        raise RuntimeError(
            "Tensor name mismatch between checkpoints. "
            f"ft1 missing={missing_ft1} extra={extra_ft1}; "
            f"ft2 missing={missing_ft2} extra={extra_ft2}"
        )

    if len(base_keys) != 244:
        raise RuntimeError(f"Expected 244 tensors in base, found {len(base_keys)}")

    mlp_names = mlp_tensor_names()
    if len(mlp_names) != 64:
        raise RuntimeError(f"Expected 64 MLP tensor names, computed {len(mlp_names)}")
    for name in mlp_names:
        if name not in base_keys:
            raise RuntimeError(f"Expected MLP tensor {name!r} not found in checkpoint")

    non_mlp_names = base_keys - mlp_names

    # Step 1: every tensor outside the 64 MLP tensors must be identical
    # (bit-exact) across all three checkpoints.
    for name in sorted(non_mlp_names):
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.dtype != f1.dtype or not torch.equal(b, f1):
            raise RuntimeError(f"Non-MLP tensor {name!r} differs between base and ft1")
        if b.shape != f2.shape or b.dtype != f2.dtype or not torch.equal(b, f2):
            raise RuntimeError(f"Non-MLP tensor {name!r} differs between base and ft2")

    out = {}
    merged_count = 0

    for name in sorted(mlp_names):
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            raise RuntimeError(f"Shape mismatch for MLP tensor {name!r}")
        if b.dtype != torch.float16 or f1.dtype != torch.float16 or f2.dtype != torch.float16:
            raise RuntimeError(f"Expected float16 for MLP tensor {name!r}")

        b32 = b.to(torch.float32)
        f1_32 = f1.to(torch.float32)
        f2_32 = f2.to(torch.float32)

        merged32 = b32 + LAMBDA * (f1_32 - b32) + LAMBDA * (f2_32 - b32)
        out[name] = merged32.to(torch.float16).contiguous()
        merged_count += 1

    if merged_count != 64:
        raise RuntimeError(f"Expected to merge exactly 64 tensors, merged {merged_count}")

    for name in sorted(non_mlp_names):
        out[name] = base[name].clone().contiguous()

    if len(out) != 244:
        raise RuntimeError(f"Expected 244 output tensors, produced {len(out)}")

    save_file(out, OUT_PATH)
    print(f"Wrote {OUT_PATH} with {len(out)} tensors ({merged_count} merged, "
          f"{len(out) - merged_count} unchanged).")


if __name__ == "__main__":
    main()
