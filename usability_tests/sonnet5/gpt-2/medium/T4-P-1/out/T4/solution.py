"""
T4: Task-vector merge of two GPT-2 fine-tunes.

out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])

for the 48 MLP tensors (per layer i in 0..11: mlp.c_fc.weight/bias,
mlp.c_proj.weight/bias). Every other tensor is copied unchanged from base,
after verifying it is bit-identical across base/ft1/ft2.
"""

import re
import sys

import torch
from safetensors.torch import load_file, save_file

INPUTS = "inputs"
OUT_PATH = "out/T4/model.safetensors"
LAMBDA = 0.4

MLP_RE = re.compile(r"^h\.(\d+)\.mlp\.(c_fc|c_proj)\.(weight|bias)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def is_mlp_tensor(name: str) -> bool:
    return MLP_RE.match(name) is not None


def main() -> None:
    base = load_file(f"{INPUTS}/base/model.safetensors")
    ft1 = load_file(f"{INPUTS}/ft1/model.safetensors")
    ft2 = load_file(f"{INPUTS}/ft2/model.safetensors")

    # --- Step 1: verify same tensor names across all three checkpoints ---
    base_keys = set(base.keys())
    ft1_keys = set(ft1.keys())
    ft2_keys = set(ft2.keys())

    if not (base_keys == ft1_keys == ft2_keys):
        missing_ft1 = base_keys - ft1_keys
        extra_ft1 = ft1_keys - base_keys
        missing_ft2 = base_keys - ft2_keys
        extra_ft2 = ft2_keys - base_keys
        fail(
            "tensor name mismatch across checkpoints: "
            f"ft1 missing={sorted(missing_ft1)} extra={sorted(extra_ft1)}; "
            f"ft2 missing={sorted(missing_ft2)} extra={sorted(extra_ft2)}"
        )

    if len(base_keys) != 160:
        fail(f"expected 160 tensors in base checkpoint, found {len(base_keys)}")

    mlp_names = {name for name in base_keys if is_mlp_tensor(name)}
    non_mlp_names = base_keys - mlp_names

    if len(mlp_names) != 48:
        fail(f"expected 48 MLP tensors, found {len(mlp_names)}: {sorted(mlp_names)}")

    # Every tensor outside the 48 MLP tensors must be identical (shape,
    # dtype, and bit-exact values) across base, ft1, ft2.
    for name in sorted(non_mlp_names):
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            fail(f"shape mismatch on non-MLP tensor {name!r}: "
                 f"base={tuple(b.shape)} ft1={tuple(f1.shape)} ft2={tuple(f2.shape)}")
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            fail(f"dtype mismatch on non-MLP tensor {name!r}: "
                 f"base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")
        if not torch.equal(b, f1):
            fail(f"non-MLP tensor {name!r} differs between base and ft1, "
                 "but fine-tunes are expected to share a frozen backbone")
        if not torch.equal(b, f2):
            fail(f"non-MLP tensor {name!r} differs between base and ft2, "
                 "but fine-tunes are expected to share a frozen backbone")

    # --- Step 2: compute the merge for the 48 MLP tensors ---
    out = {}
    merged_count = 0
    for name in sorted(mlp_names):
        b = base[name].to(torch.float32)
        f1 = ft1[name].to(torch.float32)
        f2 = ft2[name].to(torch.float32)
        if b.shape != f1.shape or b.shape != f2.shape:
            fail(f"shape mismatch on MLP tensor {name!r}: "
                 f"base={tuple(b.shape)} ft1={tuple(f1.shape)} ft2={tuple(f2.shape)}")
        merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
        out[name] = merged.contiguous()
        merged_count += 1

    if merged_count != 48:
        fail(f"expected to merge exactly 48 tensors, merged {merged_count}")

    # --- Step 3: copy every other tensor unchanged from base ---
    for name in sorted(non_mlp_names):
        out[name] = base[name].contiguous()

    # --- Step 4: write output, exactly 160 tensors ---
    if len(out) != 160:
        fail(f"expected output to have exactly 160 tensors, got {len(out)}")

    save_file(out, OUT_PATH)
    print(f"Wrote {OUT_PATH} with {len(out)} tensors "
          f"({merged_count} merged, {len(out) - merged_count} unchanged).")


if __name__ == "__main__":
    main()
