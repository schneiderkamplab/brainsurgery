"""
T4: Task-vector merge of two Pythia-1B fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the
64 MLP tensors (computed in float32, cast back to float16); everything else
is copied unchanged from the base. Verifies the frozen-backbone assumption
before doing anything else.
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

MLP_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.mlp\.(dense_h_to_4h|dense_4h_to_h)\.(weight|bias)$"
)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base = load_file(BASE_PATH)
    ft1 = load_file(FT1_PATH)
    ft2 = load_file(FT2_PATH)

    # --- Step 1: verify shared structure and identity outside the MLP tensors ---
    names_base = set(base.keys())
    names_ft1 = set(ft1.keys())
    names_ft2 = set(ft2.keys())
    if names_base != names_ft1 or names_base != names_ft2:
        fail(
            "tensor name mismatch across checkpoints: "
            f"base-only={names_base - names_ft1 - names_ft2}, "
            f"ft1-only={names_ft1 - names_base}, ft2-only={names_ft2 - names_base}"
        )

    mlp_names = {n for n in names_base if MLP_RE.match(n)}
    non_mlp_names = names_base - mlp_names

    for name in non_mlp_names:
        b, s1, s2 = base[name], ft1[name], ft2[name]
        if b.shape != s1.shape or b.shape != s2.shape:
            fail(f"shape mismatch on non-MLP tensor {name!r}")
        if b.dtype != s1.dtype or b.dtype != s2.dtype:
            fail(f"dtype mismatch on non-MLP tensor {name!r}")
        if not torch.equal(b, s1) or not torch.equal(b, s2):
            fail(
                f"non-MLP tensor {name!r} differs between checkpoints; "
                "frozen-backbone assumption violated"
            )

    if len(mlp_names) != 64:
        fail(f"expected 64 MLP tensors to merge, found {len(mlp_names)}")

    # --- Step 2 & 3: merge MLP tensors, copy everything else unchanged ---
    out = {}
    merged_count = 0
    for name in names_base:
        if name in mlp_names:
            b = base[name].to(torch.float32)
            f1 = ft1[name].to(torch.float32)
            f2 = ft2[name].to(torch.float32)
            merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
            out[name] = merged.to(base[name].dtype)
            merged_count += 1
        else:
            out[name] = base[name].clone()

    if merged_count != 64:
        fail(f"internal error: merged {merged_count} tensors, expected 64")

    # --- Step 4: write output, exactly 244 tensors ---
    if len(out) != 244:
        fail(f"output has {len(out)} tensors, expected 244")

    save_file(out, OUT_PATH)
    print(f"Wrote {OUT_PATH} with {len(out)} tensors ({merged_count} merged).")


if __name__ == "__main__":
    main()
