#!/usr/bin/env python
"""T4: task-vector merge of two Pythia-1B fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])
for the 64 MLP tensors; everything else copied unchanged from base.

Plain script on torch + safetensors (condition F allowlist). Verifies the
frozen-backbone precondition before touching anything.
"""

import sys

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
NUM_LAYERS = 16
MLP_SUFFIXES = [
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    "mlp.dense_4h_to_h.bias",
]


def mlp_tensor_names():
    names = set()
    for i in range(NUM_LAYERS):
        for suf in MLP_SUFFIXES:
            names.add(f"gpt_neox.layers.{i}.{suf}")
    return names


def fail(msg):
    print(f"ABORT: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    base = load_file("inputs/base/model.safetensors")
    ft1 = load_file("inputs/ft1/model.safetensors")
    ft2 = load_file("inputs/ft2/model.safetensors")

    mlp_names = mlp_tensor_names()

    # Step 1: same tensor names across all three checkpoints.
    names_base, names_ft1, names_ft2 = set(base), set(ft1), set(ft2)
    if not (names_base == names_ft1 == names_ft2):
        fail(
            "tensor name mismatch across checkpoints: "
            f"base only={names_base - names_ft1 - names_ft2}, "
            f"ft1 only={names_ft1 - names_base}, ft2 only={names_ft2 - names_base}"
        )

    missing_mlp = mlp_names - names_base
    if missing_mlp:
        fail(f"expected MLP tensor names not present in checkpoints: {missing_mlp}")

    # Everything outside the 64 MLP tensors must be bit-identical across all three.
    non_mlp = names_base - mlp_names
    mismatched = []
    for name in non_mlp:
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatched.append((name, "shape"))
            continue
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            mismatched.append((name, "dtype"))
            continue
        if not (torch.equal(b, f1) and torch.equal(b, f2)):
            mismatched.append((name, "value"))
    if mismatched:
        fail(f"non-MLP tensors differ across checkpoints (frozen-backbone check failed): {mismatched[:10]}")

    # Step 2/3: merge the 64 MLP tensors, copy the rest unchanged.
    out = {}
    merged_count = 0
    for name in names_base:
        if name in mlp_names:
            b = base[name].to(torch.float32)
            f1 = ft1[name].to(torch.float32)
            f2 = ft2[name].to(torch.float32)
            if b.shape != f1.shape or b.shape != f2.shape:
                fail(f"shape mismatch on MLP tensor {name}")
            merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
            out[name] = merged.to(base[name].dtype).contiguous()
            merged_count += 1
        else:
            out[name] = base[name].contiguous()

    if merged_count != 64:
        fail(f"expected exactly 64 merged MLP tensors, got {merged_count}")
    if len(out) != 244:
        fail(f"expected exactly 244 output tensors, got {len(out)}")

    save_file(out, "out/T4/model.safetensors")
    print(f"OK: wrote out/T4/model.safetensors with {len(out)} tensors ({merged_count} merged)")


if __name__ == "__main__":
    main()
