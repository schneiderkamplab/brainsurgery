"""
T4: Task-vector merge of two GPT-2 fine-tunes.

out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])

for the 48 MLP tensors (per layer i in 0..11: mlp.c_fc.{weight,bias},
mlp.c_proj.{weight,bias}); every other tensor is copied unchanged from base.
"""

import re
import sys

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
BASE_PATH = "inputs/base/model.safetensors"
FT1_PATH = "inputs/ft1/model.safetensors"
FT2_PATH = "inputs/ft2/model.safetensors"
OUT_PATH = "out/T4/model.safetensors"

MLP_RE = re.compile(r"^h\.(\d+)\.mlp\.(c_fc|c_proj)\.(weight|bias)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base = load_file(BASE_PATH)
    ft1 = load_file(FT1_PATH)
    ft2 = load_file(FT2_PATH)

    # --- Step 1: verify shared structure before touching anything ---
    base_keys = set(base.keys())
    ft1_keys = set(ft1.keys())
    ft2_keys = set(ft2.keys())

    if not (base_keys == ft1_keys == ft2_keys):
        fail(
            "tensor name mismatch: "
            f"base-only={base_keys - ft1_keys - ft2_keys}, "
            f"ft1-only={ft1_keys - base_keys}, "
            f"ft2-only={ft2_keys - base_keys}"
        )

    mlp_keys = {k for k in base_keys if MLP_RE.match(k)}
    if len(mlp_keys) != 48:
        fail(f"expected exactly 48 MLP tensors, found {len(mlp_keys)}: {sorted(mlp_keys)}")

    non_mlp_keys = base_keys - mlp_keys

    for k in sorted(non_mlp_keys):
        b, f1, f2 = base[k], ft1[k], ft2[k]
        if b.shape != f1.shape or b.shape != f2.shape:
            fail(f"shape mismatch outside MLP tensors at {k!r}: {b.shape} vs {f1.shape} vs {f2.shape}")
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            fail(f"dtype mismatch outside MLP tensors at {k!r}: {b.dtype} vs {f1.dtype} vs {f2.dtype}")
        if not torch.equal(b, f1):
            fail(f"non-MLP tensor {k!r} differs between base and ft1")
        if not torch.equal(b, f2):
            fail(f"non-MLP tensor {k!r} differs between base and ft2")

    # Also check shape/dtype agreement on the MLP tensors themselves.
    for k in sorted(mlp_keys):
        b, f1, f2 = base[k], ft1[k], ft2[k]
        if b.shape != f1.shape or b.shape != f2.shape:
            fail(f"shape mismatch on MLP tensor {k!r}: {b.shape} vs {f1.shape} vs {f2.shape}")
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            fail(f"dtype mismatch on MLP tensor {k!r}: {b.dtype} vs {f1.dtype} vs {f2.dtype}")

    # --- Step 2 & 3: compute the merge ---
    out: dict[str, torch.Tensor] = {}
    merged_count = 0

    for k in base_keys:
        if k in mlp_keys:
            b = base[k].to(torch.float32)
            f1 = ft1[k].to(torch.float32)
            f2 = ft2[k].to(torch.float32)
            merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
            out[k] = merged.to(base[k].dtype).contiguous()
            merged_count += 1
        else:
            out[k] = base[k].contiguous()

    if merged_count != 48:
        fail(f"expected to merge exactly 48 tensors, merged {merged_count}")

    # --- Step 4: write output ---
    if len(out) != 160:
        fail(f"expected exactly 160 tensors in output, got {len(out)}")

    save_file(out, OUT_PATH)

    # Sanity re-read.
    check = load_file(OUT_PATH)
    if len(check) != 160:
        fail(f"output file on disk has {len(check)} tensors, expected 160")

    print(f"OK: wrote {OUT_PATH} with {len(out)} tensors ({merged_count} merged).")


if __name__ == "__main__":
    main()
