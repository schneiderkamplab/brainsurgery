"""T4: task-vector merge of two GPT-2 fine-tunes.

out[X] = base[X] + lambda*(ft1[X] - base[X]) + lambda*(ft2[X] - base[X])
for the 48 MLP tensors; every other tensor is copied from the base verbatim.

Both task vectors are taken against the *unmodified* base: the merged value is
computed from the three source tensors in one expression and only then written,
so the first task vector can never contaminate the second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
N_LAYERS = 12
EXPECTED_TOTAL = 160
EXPECTED_MERGED = 48

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
IN = ROOT / "inputs"
OUT = HERE / "model.safetensors"

MLP_KEYS = [
    f"h.{i}.mlp.{proj}.{kind}"
    for i in range(N_LAYERS)
    for proj in ("c_fc", "c_proj")
    for kind in ("weight", "bias")
]


class CheckFailed(RuntimeError):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailed(msg)


def main() -> None:
    handles = {
        name: safe_open(IN / name / "model.safetensors", framework="pt")
        for name in ("base", "ft1", "ft2")
    }
    base, ft1, ft2 = handles["base"], handles["ft1"], handles["ft2"]

    # --- step 1: same tensor names in all three checkpoints ---------------
    keys = {name: set(h.keys()) for name, h in handles.items()}
    for name in ("ft1", "ft2"):
        missing = sorted(keys["base"] - keys[name])
        extra = sorted(keys[name] - keys["base"])
        check(
            not missing and not extra,
            f"{name} key set differs from base: missing={missing[:5]} extra={extra[:5]}",
        )
    all_keys = sorted(keys["base"])
    check(
        len(all_keys) == EXPECTED_TOTAL,
        f"expected {EXPECTED_TOTAL} tensors in the base, found {len(all_keys)}",
    )

    # the 48 MLP names must actually be present
    missing_mlp = sorted(set(MLP_KEYS) - keys["base"])
    check(not missing_mlp, f"MLP tensors absent from the checkpoints: {missing_mlp}")
    check(
        len(set(MLP_KEYS)) == EXPECTED_MERGED,
        f"MLP key list is not {EXPECTED_MERGED} names, got {len(set(MLP_KEYS))}",
    )
    mlp = set(MLP_KEYS)

    # --- step 1 (cont.) + merge ------------------------------------------
    out: dict[str, torch.Tensor] = {}
    n_merged = 0
    n_copied = 0
    for key in all_keys:
        b = base.get_tensor(key)
        a = ft1.get_tensor(key)
        c = ft2.get_tensor(key)
        for name, t in (("ft1", a), ("ft2", c)):
            check(
                t.shape == b.shape,
                f"{key}: shape {tuple(t.shape)} in {name} != {tuple(b.shape)} in base",
            )
            check(
                t.dtype == b.dtype,
                f"{key}: dtype {t.dtype} in {name} != {b.dtype} in base",
            )
        if key in mlp:
            check(
                b.dtype == torch.float32,
                f"{key}: expected float32 for the merge, got {b.dtype}",
            )
            bf = b.to(torch.float32)
            out[key] = (
                bf + LAMBDA * (a.to(torch.float32) - bf) + LAMBDA * (c.to(torch.float32) - bf)
            )
            n_merged += 1
        else:
            # frozen backbone: must be bit-identical in all three checkpoints
            check(torch.equal(a, b), f"non-MLP tensor {key} differs between base and ft1")
            check(torch.equal(c, b), f"non-MLP tensor {key} differs between base and ft2")
            out[key] = b.clone()
            n_copied += 1

    # --- required checks --------------------------------------------------
    check(n_merged == EXPECTED_MERGED, f"merged {n_merged} tensors, expected {EXPECTED_MERGED}")
    check(
        n_copied == EXPECTED_TOTAL - EXPECTED_MERGED,
        f"copied {n_copied} tensors, expected {EXPECTED_TOTAL - EXPECTED_MERGED}",
    )
    check(len(out) == EXPECTED_TOTAL, f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata={"format": "pt"})

    # --- post-write verification -----------------------------------------
    with safe_open(OUT, framework="pt") as f:
        written = sorted(f.keys())
        check(
            len(written) == EXPECTED_TOTAL,
            f"written file has {len(written)} tensors, expected {EXPECTED_TOTAL}",
        )
        check(written == all_keys, "written key set differs from the input key set")
        for key in written:
            t = f.get_tensor(key)
            ref = out[key]
            check(t.shape == ref.shape and t.dtype == ref.dtype, f"{key}: bad shape/dtype on disk")
            if key not in mlp:
                check(torch.equal(t, base.get_tensor(key)), f"{key}: not bit-identical to the base")

    print(f"merged {n_merged} MLP tensors (lambda={LAMBDA}), copied {n_copied} unchanged")
    print(f"wrote {OUT} with {len(written)} tensors")


if __name__ == "__main__":
    try:
        main()
    except CheckFailed as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
