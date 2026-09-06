#!/usr/bin/env python
"""T4: task-vector merge of two fine-tunes of Pythia-1B.

out[X] = base[X] + lam * (ft1[X] - base[X]) + lam * (ft2[X] - base[X])

for the 64 MLP tensors only; every other tensor is copied bit-exact from the
base. Each task vector is taken against the *unmodified* base: the two deltas
are computed from the freshly read base tensor and accumulated in float32
before a single cast back to the base dtype, so no intermediate result is ever
fed back in as a base.

Every requirement in TASK.md "Required checks" is an assertion here; the script
exits non-zero (traceback / MergeError) if any of them does not hold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
NUM_LAYERS = 16
EXPECTED_TOTAL = 244
EXPECTED_MERGED = 64

BASE = Path("inputs/base/model.safetensors")
FT1 = Path("inputs/ft1/model.safetensors")
FT2 = Path("inputs/ft2/model.safetensors")
OUT = Path("out/T4/model.safetensors")

MLP_SUFFIXES = (
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    "mlp.dense_4h_to_h.bias",
)


class MergeError(RuntimeError):
    """A required precondition or post-condition of the merge failed."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise MergeError(message)


def bit_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare raw bytes, so NaN/-0.0 are handled as identity, not as values."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    # reshape(-1) first: a 0-dim tensor cannot be bit-cast to a wider dtype.
    ab = a.contiguous().reshape(-1).view(torch.uint8)
    bb = b.contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(ab, bb)


def expected_mlp_names() -> set[str]:
    return {
        f"gpt_neox.layers.{i}.{suffix}"
        for i in range(NUM_LAYERS)
        for suffix in MLP_SUFFIXES
    }


def main() -> int:
    with (
        safe_open(BASE, framework="pt") as base,
        safe_open(FT1, framework="pt") as ft1,
        safe_open(FT2, framework="pt") as ft2,
    ):
        # --- step 1: verification, before anything is touched -----------------
        base_keys = set(base.keys())
        ft1_keys = set(ft1.keys())
        ft2_keys = set(ft2.keys())

        check(
            base_keys == ft1_keys,
            f"base/ft1 tensor names differ: only in base {sorted(base_keys - ft1_keys)[:5]}, "
            f"only in ft1 {sorted(ft1_keys - base_keys)[:5]}",
        )
        check(
            base_keys == ft2_keys,
            f"base/ft2 tensor names differ: only in base {sorted(base_keys - ft2_keys)[:5]}, "
            f"only in ft2 {sorted(ft2_keys - base_keys)[:5]}",
        )
        check(
            len(base_keys) == EXPECTED_TOTAL,
            f"expected {EXPECTED_TOTAL} tensors in the inputs, found {len(base_keys)}",
        )

        # The 64 MLP tensors, named explicitly rather than pattern-matched.
        mlp_keys = expected_mlp_names()
        missing = sorted(mlp_keys - base_keys)
        check(not missing, f"MLP tensors missing from the checkpoints: {missing[:5]}")
        check(
            len(mlp_keys) == EXPECTED_MERGED,
            f"expected {EXPECTED_MERGED} MLP tensor names, built {len(mlp_keys)}",
        )
        # Guard against the name list under-matching what is actually an MLP tensor.
        found_mlp = {k for k in base_keys if ".mlp." in k}
        check(
            found_mlp == mlp_keys,
            f"checkpoint contains MLP tensors outside the merge set: "
            f"{sorted(found_mlp - mlp_keys)[:5]}",
        )

        shared_keys = sorted(base_keys - mlp_keys)
        check(
            len(shared_keys) == EXPECTED_TOTAL - EXPECTED_MERGED,
            f"expected {EXPECTED_TOTAL - EXPECTED_MERGED} shared tensors, "
            f"found {len(shared_keys)}",
        )

        out: dict[str, torch.Tensor] = {}
        n_verified = 0
        for key in shared_keys:
            b = base.get_tensor(key)
            f1 = ft1.get_tensor(key)
            f2 = ft2.get_tensor(key)
            check(
                b.shape == f1.shape == f2.shape,
                f"{key}: shape mismatch base {tuple(b.shape)} ft1 {tuple(f1.shape)} "
                f"ft2 {tuple(f2.shape)}",
            )
            check(
                b.dtype == f1.dtype == f2.dtype,
                f"{key}: dtype mismatch base {b.dtype} ft1 {f1.dtype} ft2 {f2.dtype}",
            )
            check(bit_identical(b, f1), f"{key}: ft1 differs from base outside the MLP tensors")
            check(bit_identical(b, f2), f"{key}: ft2 differs from base outside the MLP tensors")
            n_verified += 1
            out[key] = b  # step 3: unchanged tensors come from the base
        check(
            n_verified == EXPECTED_TOTAL - EXPECTED_MERGED,
            f"verified {n_verified} shared tensors, expected {EXPECTED_TOTAL - EXPECTED_MERGED}",
        )
        print(f"[ok] step 1: {n_verified} non-MLP tensors identical in base, ft1 and ft2")

        # --- step 2: the merge ------------------------------------------------
        n_merged = 0
        for key in sorted(mlp_keys):
            b = base.get_tensor(key)
            f1 = ft1.get_tensor(key)
            f2 = ft2.get_tensor(key)
            check(
                b.shape == f1.shape == f2.shape,
                f"{key}: shape mismatch base {tuple(b.shape)} ft1 {tuple(f1.shape)} "
                f"ft2 {tuple(f2.shape)}",
            )
            check(
                b.dtype == f1.dtype == f2.dtype,
                f"{key}: dtype mismatch base {b.dtype} ft1 {f1.dtype} ft2 {f2.dtype}",
            )
            b32 = b.to(torch.float32)
            merged = b32 + LAMBDA * (f1.to(torch.float32) - b32) + LAMBDA * (
                f2.to(torch.float32) - b32
            )
            check(
                torch.isfinite(merged).all().item(),
                f"{key}: merged tensor contains non-finite values",
            )
            out[key] = merged.to(b.dtype)  # cast back to the base dtype (float16)
            n_merged += 1

        check(
            n_merged == EXPECTED_MERGED,
            f"merged {n_merged} tensors, expected exactly {EXPECTED_MERGED}",
        )
        print(f"[ok] step 2: {n_merged} MLP tensors merged with lambda={LAMBDA} in float32")

        check(
            len(out) == EXPECTED_TOTAL,
            f"output holds {len(out)} tensors, expected exactly {EXPECTED_TOTAL}",
        )

        OUT.parent.mkdir(parents=True, exist_ok=True)
        save_file(out, str(OUT), metadata={"format": "pt"})
        print(f"[ok] step 4: wrote {OUT} with {len(out)} tensors")

    # --- post-condition: re-read what was written -----------------------------
    with (
        safe_open(BASE, framework="pt") as base,
        safe_open(OUT, framework="pt") as written,
    ):
        written_keys = set(written.keys())
        check(
            len(written_keys) == EXPECTED_TOTAL,
            f"output file has {len(written_keys)} tensors, expected {EXPECTED_TOTAL}",
        )
        check(
            written_keys == set(base.keys()),
            "output tensor names differ from the base tensor names",
        )
        mlp_keys = expected_mlp_names()
        n_same, n_changed = 0, 0
        for key in sorted(written_keys):
            b = base.get_tensor(key)
            w = written.get_tensor(key)
            check(
                b.shape == w.shape and b.dtype == w.dtype,
                f"{key}: output shape/dtype {tuple(w.shape)}/{w.dtype} != base "
                f"{tuple(b.shape)}/{b.dtype}",
            )
            if key in mlp_keys:
                n_changed += 1
            else:
                check(bit_identical(b, w), f"{key}: non-MLP tensor is not bit-identical to base")
                n_same += 1
        check(
            n_same == EXPECTED_TOTAL - EXPECTED_MERGED and n_changed == EXPECTED_MERGED,
            f"output split is {n_same} unchanged / {n_changed} merged, expected "
            f"{EXPECTED_TOTAL - EXPECTED_MERGED} / {EXPECTED_MERGED}",
        )
        print(
            f"[ok] verified output: {n_same} tensors bit-identical to the base, "
            f"{n_changed} merged, {len(written_keys)} total"
        )

    print("T4 merge complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
