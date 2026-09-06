"""
T4: Task-vector merge of two Pythia-1B fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the
64 MLP tensors, computed in float32 and cast back to float16. All other
tensors are copied unchanged from base. Verifies before merging that the
three checkpoints share the same tensor names/shapes/dtypes and that every
non-MLP tensor is bit-identical across all three.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
NUM_LAYERS = 16

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # out/T4 -> out -> sandbox root
INPUTS = ROOT / "inputs"
OUT_DIR = HERE
OUT_FILE = OUT_DIR / "model.safetensors"


def mlp_tensor_names():
    names = set()
    for i in range(NUM_LAYERS):
        prefix = f"gpt_neox.layers.{i}.mlp"
        names.add(f"{prefix}.dense_h_to_4h.weight")
        names.add(f"{prefix}.dense_h_to_4h.bias")
        names.add(f"{prefix}.dense_4h_to_h.weight")
        names.add(f"{prefix}.dense_4h_to_h.bias")
    return names


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    base_path = INPUTS / "base" / "model.safetensors"
    ft1_path = INPUTS / "ft1" / "model.safetensors"
    ft2_path = INPUTS / "ft2" / "model.safetensors"

    for p in (base_path, ft1_path, ft2_path):
        if not p.exists():
            fail(f"missing input file: {p}")

    base = load_file(str(base_path))
    ft1 = load_file(str(ft1_path))
    ft2 = load_file(str(ft2_path))

    # 1. Verify same tensor names across all three checkpoints.
    base_keys = set(base.keys())
    ft1_keys = set(ft1.keys())
    ft2_keys = set(ft2.keys())
    if not (base_keys == ft1_keys == ft2_keys):
        missing_from_ft1 = base_keys - ft1_keys
        missing_from_ft2 = base_keys - ft2_keys
        extra_in_ft1 = ft1_keys - base_keys
        extra_in_ft2 = ft2_keys - base_keys
        fail(
            "tensor name mismatch across checkpoints: "
            f"missing_from_ft1={missing_from_ft1}, missing_from_ft2={missing_from_ft2}, "
            f"extra_in_ft1={extra_in_ft1}, extra_in_ft2={extra_in_ft2}"
        )

    mlp_names = mlp_tensor_names()
    if not mlp_names.issubset(base_keys):
        fail(f"expected MLP tensor names not found in base checkpoint: {mlp_names - base_keys}")

    non_mlp_names = base_keys - mlp_names

    # Verify every non-MLP tensor is identical (shape, dtype, values) across all three.
    for name in non_mlp_names:
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            fail(f"shape mismatch for non-MLP tensor {name!r}: base={b.shape}, ft1={f1.shape}, ft2={f2.shape}")
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            fail(f"dtype mismatch for non-MLP tensor {name!r}: base={b.dtype}, ft1={f1.dtype}, ft2={f2.dtype}")
        if not torch.equal(b, f1):
            fail(f"non-MLP tensor {name!r} differs between base and ft1")
        if not torch.equal(b, f2):
            fail(f"non-MLP tensor {name!r} differs between base and ft2")

    # Also verify MLP tensors match in shape/dtype across checkpoints (even though values may differ).
    for name in mlp_names:
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            fail(f"shape mismatch for MLP tensor {name!r}: base={b.shape}, ft1={f1.shape}, ft2={f2.shape}")
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            fail(f"dtype mismatch for MLP tensor {name!r}: base={b.dtype}, ft1={f1.dtype}, ft2={f2.dtype}")

    # 2. Merge the 64 MLP tensors via task arithmetic, taken against the
    # unmodified base (both deltas computed independently, then combined).
    out = {}
    merged_count = 0
    for name in mlp_names:
        b32 = base[name].to(torch.float32)
        f1_32 = ft1[name].to(torch.float32)
        f2_32 = ft2[name].to(torch.float32)
        merged32 = b32 + LAMBDA * (f1_32 - b32) + LAMBDA * (f2_32 - b32)
        out[name] = merged32.to(base[name].dtype).contiguous()
        merged_count += 1

    if merged_count != 64:
        fail(f"expected exactly 64 merged MLP tensors, got {merged_count}")

    # 3. Everything else comes from base, unchanged.
    for name in non_mlp_names:
        out[name] = base[name].contiguous()

    if len(out) != 244:
        fail(f"expected exactly 244 output tensors, got {len(out)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_FILE))
    print(f"Wrote {len(out)} tensors ({merged_count} merged) to {OUT_FILE}")


if __name__ == "__main__":
    main()
