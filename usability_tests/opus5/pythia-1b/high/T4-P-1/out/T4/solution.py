#!/usr/bin/env python
"""T4: task-vector merge of two Pythia-1B fine-tunes onto the base.

    out[X] = base[X] + lam * (ft1[X] - base[X]) + lam * (ft2[X] - base[X])

for the 64 MLP tensors, every other tensor copied bit-exactly from the base.
Both task vectors are taken against the *unmodified* base.
"""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # sandbox root
IN = ROOT / "inputs"
OUT_DIR = ROOT / "out" / "T4"
OUT_FILE = OUT_DIR / "model.safetensors"

LAMBDA = 0.4
N_LAYERS = 16
N_TENSORS = 244
N_MLP = 64


class CheckFailed(RuntimeError):
    """A required check did not hold."""


def check(cond, msg):
    if not cond:
        raise CheckFailed(msg)


def mlp_names():
    names = []
    for i in range(N_LAYERS):
        p = f"gpt_neox.layers.{i}.mlp."
        names += [
            p + "dense_h_to_4h.weight",
            p + "dense_h_to_4h.bias",
            p + "dense_4h_to_h.weight",
            p + "dense_4h_to_h.bias",
        ]
    return names


def raw_bits(t):
    """Flat uint8 view of a tensor's storage, for bit-exact comparison
    (NaN-safe, unlike torch.equal)."""
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def main():
    handles = {}
    for name in ("base", "ft1", "ft2"):
        path = IN / name / "model.safetensors"
        check(path.is_file(), f"missing input checkpoint: {path}")
        handles[name] = safe_open(path, framework="pt", device="cpu")

    base, ft1, ft2 = handles["base"], handles["ft1"], handles["ft2"]

    # --- step 1: structural verification -------------------------------------
    keys = {n: set(h.keys()) for n, h in handles.items()}
    for other in ("ft1", "ft2"):
        check(
            keys["base"] == keys[other],
            f"tensor names differ between base and {other}: "
            f"only in base={sorted(keys['base'] - keys[other])[:5]}, "
            f"only in {other}={sorted(keys[other] - keys['base'])[:5]}",
        )
    all_keys = sorted(keys["base"])
    check(
        len(all_keys) == N_TENSORS,
        f"expected {N_TENSORS} tensors per checkpoint, found {len(all_keys)}",
    )

    mlp = mlp_names()
    check(len(set(mlp)) == N_MLP, f"MLP name list is not {N_MLP} distinct names")
    missing = [k for k in mlp if k not in keys["base"]]
    check(not missing, f"MLP tensors absent from the checkpoints: {missing}")
    mlp_set = set(mlp)

    shared = [k for k in all_keys if k not in mlp_set]
    check(
        len(shared) == N_TENSORS - N_MLP,
        f"expected {N_TENSORS - N_MLP} non-MLP tensors, found {len(shared)}",
    )

    # every tensor outside the MLP set must be identical in all three
    for k in shared:
        tb, t1, t2 = base.get_tensor(k), ft1.get_tensor(k), ft2.get_tensor(k)
        for other, t in (("ft1", t1), ("ft2", t2)):
            check(
                tb.shape == t.shape,
                f"shape mismatch on shared tensor {k!r}: base {tuple(tb.shape)} "
                f"vs {other} {tuple(t.shape)}",
            )
            check(
                tb.dtype == t.dtype,
                f"dtype mismatch on shared tensor {k!r}: base {tb.dtype} vs {other} {t.dtype}",
            )
            check(
                torch.equal(raw_bits(tb), raw_bits(t)),
                f"shared tensor {k!r} is not identical between base and {other}; "
                "the frozen-backbone precondition does not hold",
            )
    del tb, t1, t2

    # the MLP tensors must at least agree on shape and dtype across the three
    for k in mlp:
        tb, t1, t2 = base.get_tensor(k), ft1.get_tensor(k), ft2.get_tensor(k)
        for other, t in (("ft1", t1), ("ft2", t2)):
            check(
                tb.shape == t.shape and tb.dtype == t.dtype,
                f"MLP tensor {k!r}: base {tuple(tb.shape)}/{tb.dtype} vs "
                f"{other} {tuple(t.shape)}/{t.dtype}",
            )
    del tb, t1, t2

    # --- step 2/3: merge -----------------------------------------------------
    out = {}
    merged = 0
    for k in all_keys:
        if k in mlp_set:
            tb = base.get_tensor(k)
            b32 = tb.to(torch.float32)
            v1 = ft1.get_tensor(k).to(torch.float32) - b32
            v2 = ft2.get_tensor(k).to(torch.float32) - b32
            out[k] = (b32 + LAMBDA * v1 + LAMBDA * v2).to(tb.dtype)
            merged += 1
        else:
            out[k] = base.get_tensor(k).clone()

    check(merged == N_MLP, f"merged {merged} tensors, expected exactly {N_MLP}")
    check(len(out) == N_TENSORS, f"output has {len(out)} tensors, expected {N_TENSORS}")

    # --- step 4: write -------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_FILE), metadata={"format": "pt"})

    # read back and confirm what landed on disk
    with safe_open(OUT_FILE, framework="pt", device="cpu") as h:
        wrote = sorted(h.keys())
    check(
        len(wrote) == N_TENSORS,
        f"written file has {len(wrote)} tensors, expected {N_TENSORS}",
    )
    check(wrote == all_keys, "written key set differs from the base key set")

    print(f"verified {len(shared)} shared tensors identical across base/ft1/ft2")
    print(f"merged {merged} MLP tensors with lambda={LAMBDA} (both vectors vs. base)")
    print(f"wrote {len(wrote)} tensors to {OUT_FILE}")


if __name__ == "__main__":
    main()
