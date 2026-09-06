"""
T4: task-vector merge of two GPT-2 fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])   for the 48 MLP tensors
out[X] = base[X]                                                       for every other tensor

Verifies before merging that all three checkpoints share the same tensor
names and that every non-MLP tensor is bit-identical across all three.
Aborts loudly (non-zero exit) if any required check fails.
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
NUM_LAYERS = 12

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
FT1 = HERE.parent.parent / "inputs" / "ft1" / "model.safetensors"
FT2 = HERE.parent.parent / "inputs" / "ft2" / "model.safetensors"
OUT = HERE / "model.safetensors"


def mlp_tensor_names():
    names = set()
    for i in range(NUM_LAYERS):
        names.add(f"h.{i}.mlp.c_fc.weight")
        names.add(f"h.{i}.mlp.c_fc.bias")
        names.add(f"h.{i}.mlp.c_proj.weight")
        names.add(f"h.{i}.mlp.c_proj.bias")
    return names


def load_all(path):
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def fail(msg):
    print(f"ABORT: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    base = load_all(BASE)
    ft1 = load_all(FT1)
    ft2 = load_all(FT2)

    mlp_names = mlp_tensor_names()

    # --- Step 1: verify same key sets ---
    keys_base, keys_ft1, keys_ft2 = set(base), set(ft1), set(ft2)
    if not (keys_base == keys_ft1 == keys_ft2):
        only_ft1 = keys_ft1 - keys_base
        only_ft2 = keys_ft2 - keys_base
        missing_ft1 = keys_base - keys_ft1
        missing_ft2 = keys_base - keys_ft2
        fail(
            "tensor name mismatch across checkpoints: "
            f"ft1 extra={sorted(only_ft1)} ft1 missing={sorted(missing_ft1)} "
            f"ft2 extra={sorted(only_ft2)} ft2 missing={sorted(missing_ft2)}"
        )

    if not mlp_names.issubset(keys_base):
        fail(f"expected MLP tensor names not found in base: {sorted(mlp_names - keys_base)}")

    if len(mlp_names) != 48:
        fail(f"expected exactly 48 MLP tensors, computed {len(mlp_names)}")

    # --- Step 1 continued: every non-MLP tensor must be identical across all three ---
    non_mlp = keys_base - mlp_names
    mismatched = []
    for name in sorted(non_mlp):
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatched.append(f"{name}: shape mismatch base={b.shape} ft1={f1.shape} ft2={f2.shape}")
            continue
        if not (torch.equal(b, f1) and torch.equal(b, f2)):
            mismatched.append(f"{name}: values differ outside the MLP tensors")
    if mismatched:
        fail(
            "non-MLP tensors differ between checkpoints (frozen-backbone assumption violated): "
            + "; ".join(mismatched)
        )

    # --- Step 2: merge the 48 MLP tensors ---
    out = {}
    merged_count = 0
    for name in sorted(non_mlp):
        out[name] = base[name].clone()

    for name in sorted(mlp_names):
        b = base[name].to(torch.float32)
        f1 = ft1[name].to(torch.float32)
        f2 = ft2[name].to(torch.float32)
        out[name] = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
        merged_count += 1

    if merged_count != 48:
        fail(f"expected to merge exactly 48 tensors, merged {merged_count}")

    if len(out) != 160:
        fail(f"expected output to have exactly 160 tensors, got {len(out)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT))
    print(f"wrote {OUT} with {len(out)} tensors ({merged_count} merged, "
          f"{len(out) - merged_count} unchanged)")


if __name__ == "__main__":
    main()
