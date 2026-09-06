"""
T4: Task-vector merge of two fine-tunes (GPT-2 124M).

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])
for the 48 MLP tensors, lambda = 0.4. Everything else is copied from base
unchanged, after verifying it is identical across all three checkpoints.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent.parent / "inputs"
LAMBDA = 0.4


def mlp_tensor_names():
    names = set()
    for i in range(12):
        names.add(f"h.{i}.mlp.c_fc.weight")
        names.add(f"h.{i}.mlp.c_fc.bias")
        names.add(f"h.{i}.mlp.c_proj.weight")
        names.add(f"h.{i}.mlp.c_proj.bias")
    return names


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    base_path = INPUTS / "base" / "model.safetensors"
    ft1_path = INPUTS / "ft1" / "model.safetensors"
    ft2_path = INPUTS / "ft2" / "model.safetensors"

    base = load_file(str(base_path))
    ft1 = load_file(str(ft1_path))
    ft2 = load_file(str(ft2_path))

    if len(base) != 160:
        fail(f"expected 160 tensors in base, found {len(base)}")

    # Step 1: verify same tensor names across all three checkpoints.
    base_keys = set(base.keys())
    ft1_keys = set(ft1.keys())
    ft2_keys = set(ft2.keys())
    if base_keys != ft1_keys or base_keys != ft2_keys:
        only_in_base = base_keys - (ft1_keys & ft2_keys)
        only_in_ft1 = ft1_keys - base_keys
        only_in_ft2 = ft2_keys - base_keys
        fail(
            "tensor name mismatch between checkpoints: "
            f"base-only={sorted(only_in_base)} ft1-only={sorted(only_in_ft1)} "
            f"ft2-only={sorted(only_in_ft2)}"
        )

    mlp_names = mlp_tensor_names()
    missing_mlp = mlp_names - base_keys
    if missing_mlp:
        fail(f"expected MLP tensor names not found in checkpoint: {sorted(missing_mlp)}")
    if len(mlp_names) != 48:
        fail(f"expected 48 MLP tensor names, computed {len(mlp_names)}")

    # Verify every non-MLP tensor is identical (same shape, dtype, bit-exact
    # values) across base, ft1 and ft2.
    non_mlp_names = base_keys - mlp_names
    mismatched = []
    for name in sorted(non_mlp_names):
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatched.append(f"{name}: shape mismatch base={b.shape} ft1={f1.shape} ft2={f2.shape}")
            continue
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            mismatched.append(f"{name}: dtype mismatch base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")
            continue
        if not torch.equal(b, f1) or not torch.equal(b, f2):
            mismatched.append(f"{name}: values differ outside the declared MLP tensors")
    if mismatched:
        fail(
            "non-MLP tensors differ across checkpoints (fine-tunes were not "
            "frozen-backbone as assumed):\n  " + "\n  ".join(mismatched)
        )

    # Step 2 & 3: build the output state dict.
    out = {}
    merged_count = 0
    for name in base_keys:
        if name in mlp_names:
            b = base[name].to(torch.float32)
            f1 = ft1[name].to(torch.float32)
            f2 = ft2[name].to(torch.float32)
            merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
            out[name] = merged.to(base[name].dtype).contiguous()
            merged_count += 1
        else:
            out[name] = base[name].clone().contiguous()

    if merged_count != 48:
        fail(f"expected exactly 48 merged tensors, merged {merged_count}")

    if len(out) != 160:
        fail(f"expected output to have exactly 160 tensors, has {len(out)}")

    out_dir = HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "model.safetensors"))
    print(f"Wrote {out_dir / 'model.safetensors'} with {len(out)} tensors "
          f"({merged_count} merged, {len(out) - merged_count} copied unchanged).")


if __name__ == "__main__":
    main()
