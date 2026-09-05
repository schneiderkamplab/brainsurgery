"""
T4: Task-vector merge of two Pythia-1B fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])   for the 64 MLP tensors
out[X] = base[X]                                                       for everything else

Plain script on top of `safetensors` + `torch` (both in F-allowed.md). No
mergekit config is used because the required checks (identical key sets,
bit-identical non-MLP tensors across all three checkpoints, exactly 64
merged tensors, exactly 244 output tensors) need to be enforced explicitly
and fail loudly; a hand-rolled verification pass makes that unambiguous.
"""

import re
import sys

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
MLP_RE = re.compile(
    r"^gpt_neox\.layers\.(\d+)\.mlp\.(dense_h_to_4h|dense_4h_to_h)\.(weight|bias)$"
)


def is_mlp_key(key: str) -> bool:
    m = MLP_RE.match(key)
    return bool(m) and 0 <= int(m.group(1)) <= 15


def main() -> None:
    base = load_file("inputs/base/model.safetensors")
    ft1 = load_file("inputs/ft1/model.safetensors")
    ft2 = load_file("inputs/ft2/model.safetensors")

    # Step 1: same key set across all three checkpoints.
    base_keys, ft1_keys, ft2_keys = set(base), set(ft1), set(ft2)
    if not (base_keys == ft1_keys == ft2_keys):
        missing_in_ft1 = base_keys - ft1_keys
        missing_in_ft2 = base_keys - ft2_keys
        extra_in_ft1 = ft1_keys - base_keys
        extra_in_ft2 = ft2_keys - base_keys
        raise SystemExit(
            "Tensor name mismatch across checkpoints:\n"
            f"  missing in ft1: {sorted(missing_in_ft1)}\n"
            f"  missing in ft2: {sorted(missing_in_ft2)}\n"
            f"  extra in ft1:   {sorted(extra_in_ft1)}\n"
            f"  extra in ft2:   {sorted(extra_in_ft2)}"
        )

    all_keys = sorted(base_keys)
    mlp_keys = [k for k in all_keys if is_mlp_key(k)]
    other_keys = [k for k in all_keys if k not in set(mlp_keys)]

    # Also verify shapes/dtypes match across the three, for every key.
    for k in all_keys:
        b, f1, f2 = base[k], ft1[k], ft2[k]
        if not (b.shape == f1.shape == f2.shape):
            raise SystemExit(f"Shape mismatch on {k}: base={b.shape} ft1={f1.shape} ft2={f2.shape}")
        if not (b.dtype == f1.dtype == f2.dtype):
            raise SystemExit(f"Dtype mismatch on {k}: base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")

    # Step 1 continued: every non-MLP tensor must be bit-identical across all three.
    mismatched_other = []
    for k in other_keys:
        b, f1, f2 = base[k], ft1[k], ft2[k]
        if not torch.equal(b, f1) or not torch.equal(b, f2):
            mismatched_other.append(k)
    if mismatched_other:
        raise SystemExit(
            "Fine-tune(s) modified tensors outside the expected 64 MLP tensors "
            f"(frozen-backbone assumption violated): {mismatched_other}"
        )

    if len(mlp_keys) != 64:
        raise SystemExit(f"Expected exactly 64 MLP tensors to merge, found {len(mlp_keys)}: {mlp_keys}")

    # Step 2: merge the 64 MLP tensors in float32, cast back to the base dtype.
    out = {}
    for k in mlp_keys:
        base_dtype = base[k].dtype
        b32 = base[k].to(torch.float32)
        f1_32 = ft1[k].to(torch.float32)
        f2_32 = ft2[k].to(torch.float32)
        merged = b32 + LAMBDA * (f1_32 - b32) + LAMBDA * (f2_32 - b32)
        out[k] = merged.to(base_dtype).contiguous()

    # Step 3: everything else is the base, unchanged.
    for k in other_keys:
        out[k] = base[k].contiguous()

    # Required checks, restated on the actual output about to be written.
    if len(out) != 244:
        raise SystemExit(f"Expected exactly 244 output tensors, got {len(out)}")
    merged_count = sum(1 for k in out if is_mlp_key(k))
    if merged_count != 64:
        raise SystemExit(f"Expected exactly 64 merged tensors in output, got {merged_count}")

    save_file(out, "out/T4/model.safetensors")
    print(f"Wrote out/T4/model.safetensors: {len(out)} tensors ({merged_count} merged, "
          f"{len(out) - merged_count} copied from base).")


if __name__ == "__main__":
    main()
