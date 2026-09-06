"""T4: task-vector merge of two GPT-2 fine-tunes onto the base.

out[X] = base[X] + lam*(ft1[X]-base[X]) + lam*(ft2[X]-base[X]) for the 48 MLP
tensors; every other tensor is copied from the base bit-exactly.
Each task vector is taken against the *unmodified* base.
"""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAM = 0.4
N_LAYERS = 12
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "inputs/base/model.safetensors"
FT1 = ROOT / "inputs/ft1/model.safetensors"
FT2 = ROOT / "inputs/ft2/model.safetensors"
OUT = ROOT / "out/T4/model.safetensors"

MLP_KEYS = [
    f"h.{i}.mlp.{mod}.{p}"
    for i in range(N_LAYERS)
    for mod in ("c_fc", "c_proj")
    for p in ("weight", "bias")
]


class CheckFailed(RuntimeError):
    pass


def require(cond, msg):
    if not cond:
        raise CheckFailed(msg)


def main() -> None:
    with (
        safe_open(BASE, framework="pt") as b,
        safe_open(FT1, framework="pt") as f1,
        safe_open(FT2, framework="pt") as f2,
    ):
        kb, k1, k2 = set(b.keys()), set(f1.keys()), set(f2.keys())

        # --- step 1: identical key sets ------------------------------------
        require(kb == k1, f"ft1 key set differs from base: {sorted(kb ^ k1)[:10]}")
        require(kb == k2, f"ft2 key set differs from base: {sorted(kb ^ k2)[:10]}")
        require(len(kb) == 160, f"expected 160 tensors in base, got {len(kb)}")

        require(len(MLP_KEYS) == 48, f"MLP key list has {len(MLP_KEYS)} entries, expected 48")
        missing = sorted(set(MLP_KEYS) - kb)
        require(not missing, f"MLP tensors absent from the checkpoints: {missing}")

        mlp = set(MLP_KEYS)
        shared = sorted(kb - mlp)
        require(
            len(shared) == 112, f"expected 112 non-MLP tensors, got {len(shared)}"
        )

        # --- step 1 (cont.): non-MLP tensors identical in all three --------
        out = {}
        for k in shared:
            tb, t1, t2 = b.get_tensor(k), f1.get_tensor(k), f2.get_tensor(k)
            require(
                tb.shape == t1.shape == t2.shape,
                f"shape mismatch outside the MLPs for {k}: "
                f"{tuple(tb.shape)} / {tuple(t1.shape)} / {tuple(t2.shape)}",
            )
            require(
                tb.dtype == t1.dtype == t2.dtype,
                f"dtype mismatch outside the MLPs for {k}: "
                f"{tb.dtype} / {t1.dtype} / {t2.dtype}",
            )
            require(
                torch.equal(tb, t1),
                f"non-MLP tensor {k} differs between base and ft1 "
                f"(max |delta| = {(t1.float() - tb.float()).abs().max().item():.3e})",
            )
            require(
                torch.equal(tb, t2),
                f"non-MLP tensor {k} differs between base and ft2 "
                f"(max |delta| = {(t2.float() - tb.float()).abs().max().item():.3e})",
            )
            out[k] = tb.clone()  # unchanged, bit-exact copy of the base

        # --- step 2: task-vector merge, both vectors against the base -----
        merged = 0
        for k in MLP_KEYS:
            tb, t1, t2 = b.get_tensor(k), f1.get_tensor(k), f2.get_tensor(k)
            require(
                tb.shape == t1.shape == t2.shape,
                f"shape mismatch for MLP tensor {k}: "
                f"{tuple(tb.shape)} / {tuple(t1.shape)} / {tuple(t2.shape)}",
            )
            require(
                tb.dtype == torch.float32,
                f"expected float32 for {k}, got {tb.dtype}",
            )
            base32 = tb.to(torch.float32)
            tv1 = t1.to(torch.float32) - base32
            tv2 = t2.to(torch.float32) - base32
            out[k] = (base32 + LAM * tv1 + LAM * tv2).to(tb.dtype).contiguous()
            merged += 1

        require(merged == 48, f"merged {merged} tensors, expected exactly 48")
        require(len(out) == 160, f"output has {len(out)} tensors, expected exactly 160")
        require(
            set(out) == kb, "output key set does not match the input key set"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT))

    # --- post-write verification of the required checks -------------------
    with safe_open(OUT, framework="pt") as o:
        ko = set(o.keys())
        require(len(ko) == 160, f"written file has {len(ko)} tensors, expected 160")
        require(ko == kb, "written key set does not match the input key set")
        changed = 0
        with safe_open(BASE, framework="pt") as b:
            for k in sorted(ko):
                to, tb = o.get_tensor(k), b.get_tensor(k)
                require(
                    to.shape == tb.shape and to.dtype == tb.dtype,
                    f"shape/dtype changed for {k}",
                )
                if not torch.equal(to, tb):
                    changed += 1
                    require(k in mlp, f"non-MLP tensor {k} was modified")
        require(changed <= 48, f"{changed} tensors differ from the base, expected <= 48")

    print(f"wrote {OUT} : 160 tensors, 48 merged (lambda={LAM}), 112 copied from base")


if __name__ == "__main__":
    main()
