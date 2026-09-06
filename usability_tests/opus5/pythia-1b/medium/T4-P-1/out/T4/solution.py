"""T4: task-vector merge of two Pythia-1B fine-tunes into the base checkpoint.

out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])
for the 64 MLP tensors, every other tensor copied from the base untouched.
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
N_LAYERS = 16
EXPECTED_TOTAL = 244
EXPECTED_MERGED = 64

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASE = ROOT / "inputs" / "base" / "model.safetensors"
FT1 = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2 = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT = ROOT / "out" / "T4" / "model.safetensors"

MLP_SUFFIXES = (
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    "mlp.dense_4h_to_h.bias",
)
MLP_KEYS = {
    f"gpt_neox.layers.{i}.{suffix}"
    for i in range(N_LAYERS)
    for suffix in MLP_SUFFIXES
}


def fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def main() -> None:
    for path in (BASE, FT1, FT2):
        if not path.is_file():
            fail(f"missing input checkpoint: {path}")

    with (
        safe_open(BASE, framework="pt", device="cpu") as base,
        safe_open(FT1, framework="pt", device="cpu") as ft1,
        safe_open(FT2, framework="pt", device="cpu") as ft2,
    ):
        base_keys = set(base.keys())
        ft1_keys = set(ft1.keys())
        ft2_keys = set(ft2.keys())

        # --- step 1: the three checkpoints must agree on names ---------------
        if base_keys != ft1_keys:
            diff = sorted(base_keys ^ ft1_keys)
            fail(f"base and ft1 tensor names differ ({len(diff)}): {diff[:10]}")
        if base_keys != ft2_keys:
            diff = sorted(base_keys ^ ft2_keys)
            fail(f"base and ft2 tensor names differ ({len(diff)}): {diff[:10]}")

        if len(base_keys) != EXPECTED_TOTAL:
            fail(f"expected {EXPECTED_TOTAL} tensors in the base, found {len(base_keys)}")

        missing = MLP_KEYS - base_keys
        if missing:
            fail(f"expected MLP tensors absent from the checkpoints: {sorted(missing)[:10]}")
        if len(MLP_KEYS) != EXPECTED_MERGED:
            fail(f"MLP key set has {len(MLP_KEYS)} names, expected {EXPECTED_MERGED}")

        shared_keys = sorted(base_keys - MLP_KEYS)

        # --- step 1 (cont.): everything outside the MLP must be identical ----
        mismatched = []
        for key in shared_keys:
            b = base.get_tensor(key)
            a = ft1.get_tensor(key)
            c = ft2.get_tensor(key)
            for name, other in (("ft1", a), ("ft2", c)):
                if other.shape != b.shape or other.dtype != b.dtype:
                    mismatched.append(f"{key}: {name} {tuple(other.shape)}/{other.dtype} "
                                      f"vs base {tuple(b.shape)}/{b.dtype}")
                elif not torch.equal(other, b):
                    mismatched.append(f"{key}: {name} differs from base outside the MLP")
        if mismatched:
            fail(
                f"{len(mismatched)} non-MLP tensor(s) are not shared across the three "
                f"checkpoints: " + "; ".join(mismatched[:10])
            )

        # --- shapes/dtypes of the MLP tensors must line up as well ----------
        for key in sorted(MLP_KEYS):
            b = base.get_slice(key)
            for name, handle in (("ft1", ft1), ("ft2", ft2)):
                o = handle.get_slice(key)
                if o.get_shape() != b.get_shape() or o.get_dtype() != b.get_dtype():
                    fail(
                        f"{key}: {name} {o.get_shape()}/{o.get_dtype()} vs "
                        f"base {b.get_shape()}/{b.get_dtype()}"
                    )

        # --- step 2/3: merge the MLP tensors, copy the rest -----------------
        out: dict[str, torch.Tensor] = {}
        merged = 0
        for key in sorted(base_keys):
            b = base.get_tensor(key)
            if key in MLP_KEYS:
                b32 = b.to(torch.float32)
                # both task vectors are taken against the *unmodified* base
                tv1 = ft1.get_tensor(key).to(torch.float32) - b32
                tv2 = ft2.get_tensor(key).to(torch.float32) - b32
                out[key] = (b32 + LAMBDA * tv1 + LAMBDA * tv2).to(b.dtype).contiguous()
                merged += 1
            else:
                out[key] = b.clone().contiguous()

        metadata = base.metadata() or {}

    # --- required checks ----------------------------------------------------
    if merged != EXPECTED_MERGED:
        fail(f"merged {merged} tensors, expected exactly {EXPECTED_MERGED}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected exactly {EXPECTED_TOTAL}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata=metadata)

    # --- verify what actually landed on disk --------------------------------
    with safe_open(OUT, framework="pt", device="cpu") as written:
        written_keys = set(written.keys())
    if written_keys != base_keys:
        fail("written checkpoint does not have the base's tensor names")
    if len(written_keys) != EXPECTED_TOTAL:
        fail(f"written checkpoint has {len(written_keys)} tensors, expected {EXPECTED_TOTAL}")

    print(f"verified {len(shared_keys)} shared tensors identical across base/ft1/ft2")
    print(f"merged {merged} MLP tensors with lambda={LAMBDA}")
    print(f"wrote {len(written_keys)} tensors to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
