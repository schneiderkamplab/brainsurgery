#!/usr/bin/env python
"""T4: task-vector merge of two GPT-2 fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])   for the 48 MLP tensors
out[X] = base[X]                                                       for everything else

Verifies before touching anything:
  - base, ft1, ft2 have exactly the same tensor names
  - every tensor outside the 48 MLP tensors is bit-identical across all three checkpoints
Aborts loudly (raises) if any of that does not hold, or if the merged/output
tensor counts are not exactly 48 / 160.
"""

import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
N_LAYERS = 12
BASE = "inputs/base/model.safetensors"
FT1 = "inputs/ft1/model.safetensors"
FT2 = "inputs/ft2/model.safetensors"
OUT = "out/T4/model.safetensors"


def mlp_names(n_layers: int) -> set[str]:
    names = set()
    for i in range(n_layers):
        names.add(f"h.{i}.mlp.c_fc.weight")
        names.add(f"h.{i}.mlp.c_fc.bias")
        names.add(f"h.{i}.mlp.c_proj.weight")
        names.add(f"h.{i}.mlp.c_proj.bias")
    return names


def load_all(path: str) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    base = load_all(BASE)
    ft1 = load_all(FT1)
    ft2 = load_all(FT2)

    # Step 1: same tensor names across all three.
    base_keys, ft1_keys, ft2_keys = set(base), set(ft1), set(ft2)
    if not (base_keys == ft1_keys == ft2_keys):
        missing_in_ft1 = base_keys - ft1_keys
        extra_in_ft1 = ft1_keys - base_keys
        missing_in_ft2 = base_keys - ft2_keys
        extra_in_ft2 = ft2_keys - base_keys
        raise AssertionError(
            "Tensor name mismatch across checkpoints:\n"
            f"  missing in ft1: {sorted(missing_in_ft1)}\n"
            f"  extra in ft1:   {sorted(extra_in_ft1)}\n"
            f"  missing in ft2: {sorted(missing_in_ft2)}\n"
            f"  extra in ft2:   {sorted(extra_in_ft2)}"
        )

    expected_mlp = mlp_names(N_LAYERS)
    missing_mlp = expected_mlp - base_keys
    if missing_mlp:
        raise AssertionError(f"Expected MLP tensors absent from checkpoints: {sorted(missing_mlp)}")

    non_mlp_keys = base_keys - expected_mlp

    # Step 1 (continued): every non-MLP tensor must be bit-identical across the three checkpoints.
    mismatched = []
    for key in sorted(non_mlp_keys):
        b, f1, f2 = base[key], ft1[key], ft2[key]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatched.append(f"{key}: shape mismatch base={tuple(b.shape)} ft1={tuple(f1.shape)} ft2={tuple(f2.shape)}")
            continue
        if not torch.equal(b, f1):
            mismatched.append(f"{key}: differs between base and ft1")
        elif not torch.equal(b, f2):
            mismatched.append(f"{key}: differs between base and ft2")
    if mismatched:
        raise AssertionError(
            "Non-MLP tensors are not identical across base/ft1/ft2 "
            "(frozen-backbone assumption violated):\n  " + "\n  ".join(mismatched)
        )

    # Step 2: task-vector merge for the 48 MLP tensors, each vector taken against
    # the unmodified base (not against a base already touched by the other merge).
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for key in sorted(expected_mlp):
        b = base[key].to(torch.float32)
        f1 = ft1[key].to(torch.float32)
        f2 = ft2[key].to(torch.float32)
        if b.shape != f1.shape or b.shape != f2.shape:
            raise AssertionError(f"{key}: shape mismatch base={tuple(b.shape)} ft1={tuple(f1.shape)} ft2={tuple(f2.shape)}")
        merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
        out[key] = merged.contiguous()
        merged_count += 1

    if merged_count != 48:
        raise AssertionError(f"Expected exactly 48 merged tensors, got {merged_count}")

    # Step 3: everything else is the base, unchanged.
    for key in sorted(non_mlp_keys):
        out[key] = base[key].clone().contiguous()

    if len(out) != 160:
        raise AssertionError(f"Expected exactly 160 output tensors, got {len(out)}")
    if set(out) != base_keys:
        raise AssertionError("Output tensor name set does not match input tensor name set")

    save_file(out, OUT)
    print(f"Wrote {OUT}: {len(out)} tensors ({merged_count} merged, {len(out) - merged_count} unchanged).")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
