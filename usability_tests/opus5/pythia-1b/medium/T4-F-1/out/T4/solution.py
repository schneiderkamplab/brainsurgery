"""T4: task-vector merge of two fine-tunes of Pythia-1B.

out[X] = base[X] + lam*(ft1[X]-base[X]) + lam*(ft2[X]-base[X]) for the 64 MLP
tensors, computed in float32 against the *unmodified* base and cast back to
float16; every other tensor is copied bit-exactly from the base.

Plain torch + safetensors: each tensor is streamed one at a time, so the three
2 GB checkpoints are never all in memory.
"""

from __future__ import annotations

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
ROOT = HERE.parent.parent  # sandbox root
BASE = ROOT / "inputs/base/model.safetensors"
FT1 = ROOT / "inputs/ft1/model.safetensors"
FT2 = ROOT / "inputs/ft2/model.safetensors"
OUT = ROOT / "out/T4/model.safetensors"


class CheckFailed(RuntimeError):
    pass


def mlp_keys() -> set[str]:
    keys = set()
    for i in range(N_LAYERS):
        for proj in ("dense_h_to_4h", "dense_4h_to_h"):
            for suffix in ("weight", "bias"):
                keys.add(f"gpt_neox.layers.{i}.mlp.{proj}.{suffix}")
    return keys


def main() -> int:
    for path in (BASE, FT1, FT2):
        if not path.is_file():
            raise CheckFailed(f"missing input checkpoint: {path}")

    with (
        safe_open(BASE, framework="pt") as base,
        safe_open(FT1, framework="pt") as ft1,
        safe_open(FT2, framework="pt") as ft2,
    ):
        kb, k1, k2 = set(base.keys()), set(ft1.keys()), set(ft2.keys())

        # --- step 1a: identical tensor name sets -----------------------------
        if kb != k1 or kb != k2:
            raise CheckFailed(
                "tensor name sets differ: "
                f"base\\ft1={sorted(kb - k1)[:5]} ft1\\base={sorted(k1 - kb)[:5]} "
                f"base\\ft2={sorted(kb - k2)[:5]} ft2\\base={sorted(k2 - kb)[:5]}"
            )
        if len(kb) != EXPECTED_TOTAL:
            raise CheckFailed(f"expected {EXPECTED_TOTAL} tensors in base, found {len(kb)}")

        merge_keys = mlp_keys()
        missing = merge_keys - kb
        if missing:
            raise CheckFailed(f"MLP tensors absent from the checkpoints: {sorted(missing)}")
        if len(merge_keys) != EXPECTED_MERGED:
            raise CheckFailed(f"MLP key list has {len(merge_keys)} entries, expected {EXPECTED_MERGED}")

        tensors: dict[str, torch.Tensor] = {}
        merged = 0
        copied = 0

        for key in sorted(kb):
            b = base.get_tensor(key)
            t1 = ft1.get_tensor(key)
            t2 = ft2.get_tensor(key)

            if t1.shape != b.shape or t2.shape != b.shape:
                raise CheckFailed(
                    f"shape mismatch for {key}: base={tuple(b.shape)} "
                    f"ft1={tuple(t1.shape)} ft2={tuple(t2.shape)}"
                )
            if t1.dtype != b.dtype or t2.dtype != b.dtype:
                raise CheckFailed(
                    f"dtype mismatch for {key}: base={b.dtype} ft1={t1.dtype} ft2={t2.dtype}"
                )

            if key in merge_keys:
                # --- step 2: task vectors, both against the unmodified base ---
                b32 = b.to(torch.float32)
                out = b32 + LAMBDA * (t1.to(torch.float32) - b32) + LAMBDA * (t2.to(torch.float32) - b32)
                tensors[key] = out.to(b.dtype).contiguous()
                merged += 1
            else:
                # --- step 1b: frozen backbone must be bit-identical in all 3 --
                if not torch.equal(b, t1) or not torch.equal(b, t2):
                    raise CheckFailed(
                        f"non-MLP tensor {key} differs between the checkpoints; the frozen-backbone "
                        "precondition of a task-vector merge does not hold"
                    )
                # --- step 3: copied unchanged from the base -------------------
                tensors[key] = b.clone().contiguous()
                copied += 1

    # --- required checks -----------------------------------------------------
    if merged != EXPECTED_MERGED:
        raise CheckFailed(f"merged {merged} tensors, expected exactly {EXPECTED_MERGED}")
    if len(tensors) != EXPECTED_TOTAL:
        raise CheckFailed(f"output holds {len(tensors)} tensors, expected {EXPECTED_TOTAL}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT), metadata={"format": "pt"})

    # --- verify what actually landed on disk ---------------------------------
    with safe_open(OUT, framework="pt") as written:
        wk = set(written.keys())
    if wk != kb:
        raise CheckFailed("written checkpoint does not have the input key set")
    if len(wk) != EXPECTED_TOTAL:
        raise CheckFailed(f"written checkpoint has {len(wk)} tensors, expected {EXPECTED_TOTAL}")

    print(f"merged {merged} MLP tensors (lambda={LAMBDA}), copied {copied} unchanged")
    print(f"wrote {OUT} with {len(wk)} tensors")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailed as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
