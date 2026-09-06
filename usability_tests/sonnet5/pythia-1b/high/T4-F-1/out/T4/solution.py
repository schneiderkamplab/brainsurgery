#!/usr/bin/env python3
"""T4: task-vector merge of two Pythia-1B fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the
64 MLP tensors (lambda=0.4), computed in float32 and cast back to float16.
Every other tensor is copied from base unchanged.

Plain script on top of `safetensors` and `torch` (both allowed packages).
Written deliberately as a script rather than through a merge-config tool so
that the three-way precondition check in step 1 is explicit and unskippable.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
NUM_LAYERS = 16

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root (out/T4/solution.py -> sandbox)
INPUTS = ROOT / "inputs"
OUT_DIR = ROOT / "out" / "T4"
OUT_FILE = OUT_DIR / "model.safetensors"


def die(msg: str) -> None:
    print(f"ABORT: {msg}", file=sys.stderr)
    sys.exit(1)


def expected_mlp_names() -> set[str]:
    names = set()
    for i in range(NUM_LAYERS):
        p = f"gpt_neox.layers.{i}.mlp"
        names.add(f"{p}.dense_h_to_4h.weight")
        names.add(f"{p}.dense_h_to_4h.bias")
        names.add(f"{p}.dense_4h_to_h.weight")
        names.add(f"{p}.dense_4h_to_h.bias")
    return names


def main() -> None:
    base = load_file(INPUTS / "base" / "model.safetensors")
    ft1 = load_file(INPUTS / "ft1" / "model.safetensors")
    ft2 = load_file(INPUTS / "ft2" / "model.safetensors")

    # --- Step 1: precondition checks, abort loudly if violated ---------
    base_keys, ft1_keys, ft2_keys = set(base), set(ft1), set(ft2)
    if not (base_keys == ft1_keys == ft2_keys):
        missing_ft1 = base_keys - ft1_keys
        extra_ft1 = ft1_keys - base_keys
        missing_ft2 = base_keys - ft2_keys
        extra_ft2 = ft2_keys - base_keys
        die(
            "tensor name mismatch across checkpoints: "
            f"ft1 missing={sorted(missing_ft1)} extra={sorted(extra_ft1)}; "
            f"ft2 missing={sorted(missing_ft2)} extra={sorted(extra_ft2)}"
        )

    mlp_names = expected_mlp_names()
    if not mlp_names.issubset(base_keys):
        missing = mlp_names - base_keys
        die(f"expected MLP tensor names not found in base checkpoint: {sorted(missing)}")

    non_mlp_names = base_keys - mlp_names

    for name in non_mlp_names:
        b, s1, s2 = base[name], ft1[name], ft2[name]
        if b.shape != s1.shape or b.dtype != s1.dtype or not torch.equal(b, s1):
            die(f"non-MLP tensor '{name}' differs between base and ft1; frozen-backbone "
                "assumption violated")
        if b.shape != s2.shape or b.dtype != s2.dtype or not torch.equal(b, s2):
            die(f"non-MLP tensor '{name}' differs between base and ft2; frozen-backbone "
                "assumption violated")

    # Also confirm the MLP tensors actually have matching shapes/dtypes
    # (arithmetic below would otherwise fail with a less clear error).
    for name in mlp_names:
        for tag, ck in (("ft1", ft1), ("ft2", ft2)):
            if name not in ck:
                die(f"MLP tensor '{name}' missing from {tag}")
            if ck[name].shape != base[name].shape or ck[name].dtype != base[name].dtype:
                die(f"MLP tensor '{name}' shape/dtype mismatch between base and {tag}")

    # --- Step 2/3: merge ------------------------------------------------
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in base_keys:
        if name in mlp_names:
            b32 = base[name].to(torch.float32)
            t1 = (ft1[name].to(torch.float32) - b32) * LAMBDA
            t2 = (ft2[name].to(torch.float32) - b32) * LAMBDA
            merged = (b32 + t1 + t2).to(base[name].dtype)
            out[name] = merged.contiguous()
            merged_count += 1
        else:
            out[name] = base[name].clone().contiguous()

    # --- Required checks -------------------------------------------------
    if merged_count != 64:
        die(f"expected exactly 64 merged MLP tensors, got {merged_count}")
    if len(out) != 244:
        die(f"expected exactly 244 output tensors, got {len(out)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, OUT_FILE, metadata={"format": "pt"})
    print(f"wrote {OUT_FILE} with {len(out)} tensors ({merged_count} merged)")


if __name__ == "__main__":
    main()
