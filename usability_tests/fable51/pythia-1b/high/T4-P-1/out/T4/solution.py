"""T4: task-vector merge of two fine-tunes into the Pythia-1B base.

out[X] = base[X] + lam * (ft1[X] - base[X]) + lam * (ft2[X] - base[X])
for the 64 MLP tensors; every other tensor is verified identical across the
three checkpoints and copied from the base unchanged.
"""

import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "inputs" / "base" / "model.safetensors"
FT1 = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2 = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT = ROOT / "out" / "T4" / "model.safetensors"

LAMBDA = 0.4
NUM_LAYERS = 16
EXPECTED_TOTAL = 244
EXPECTED_MERGED = 64

MLP_RE = re.compile(
    r"^gpt_neox\.layers\.(\d+)\.mlp\.(dense_h_to_4h|dense_4h_to_h)\.(weight|bias)$"
)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    with safe_open(BASE, "pt") as base, safe_open(FT1, "pt") as ft1, safe_open(FT2, "pt") as ft2:
        base_keys = set(base.keys())
        ft1_keys = set(ft1.keys())
        ft2_keys = set(ft2.keys())

        # --- Step 1: key sets must match exactly.
        if base_keys != ft1_keys:
            fail(
                f"key set mismatch base vs ft1: only-base={sorted(base_keys - ft1_keys)} "
                f"only-ft1={sorted(ft1_keys - base_keys)}"
            )
        if base_keys != ft2_keys:
            fail(
                f"key set mismatch base vs ft2: only-base={sorted(base_keys - ft2_keys)} "
                f"only-ft2={sorted(ft2_keys - base_keys)}"
            )
        if len(base_keys) != EXPECTED_TOTAL:
            fail(f"expected {EXPECTED_TOTAL} tensors in base, found {len(base_keys)}")

        mlp_keys = set()
        for k in base_keys:
            m = MLP_RE.match(k)
            if m and 0 <= int(m.group(1)) < NUM_LAYERS:
                mlp_keys.add(k)
        if len(mlp_keys) != EXPECTED_MERGED:
            fail(f"expected {EXPECTED_MERGED} MLP tensors, matched {len(mlp_keys)}")
        shared_keys = base_keys - mlp_keys

        # --- Step 1 (cont.): every non-MLP tensor identical in all three.
        bad = []
        for k in sorted(shared_keys):
            b = base.get_tensor(k)
            for name, f in (("ft1", ft1), ("ft2", ft2)):
                t = f.get_tensor(k)
                if t.shape != b.shape or t.dtype != b.dtype:
                    bad.append(f"{k}: {name} shape/dtype {tuple(t.shape)}/{t.dtype} "
                               f"vs base {tuple(b.shape)}/{b.dtype}")
                elif not torch.equal(t, b):
                    bad.append(f"{k}: {name} values differ from base")
        if bad:
            fail("shared tensors are not identical across checkpoints:\n  " + "\n  ".join(bad))
        print(f"verified {len(shared_keys)} shared tensors identical across base/ft1/ft2")

        # --- Step 2: merge the MLP tensors against the unmodified base.
        out = {}
        merged = 0
        for k in sorted(base_keys):
            b = base.get_tensor(k)
            if k in mlp_keys:
                t1 = ft1.get_tensor(k)
                t2 = ft2.get_tensor(k)
                if t1.shape != b.shape or t2.shape != b.shape:
                    fail(f"{k}: shape mismatch base={tuple(b.shape)} "
                         f"ft1={tuple(t1.shape)} ft2={tuple(t2.shape)}")
                if t1.dtype != b.dtype or t2.dtype != b.dtype:
                    fail(f"{k}: dtype mismatch base={b.dtype} ft1={t1.dtype} ft2={t2.dtype}")
                b32 = b.to(torch.float32)
                tv1 = t1.to(torch.float32) - b32
                tv2 = t2.to(torch.float32) - b32
                out[k] = (b32 + LAMBDA * tv1 + LAMBDA * tv2).to(b.dtype).contiguous()
                merged += 1
            else:
                out[k] = b.contiguous()

    # --- Required checks.
    if merged != EXPECTED_MERGED:
        fail(f"merged {merged} tensors, expected {EXPECTED_MERGED}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata={"format": "pt"})

    # Re-open and confirm the written file.
    with safe_open(OUT, "pt") as f:
        n = len(list(f.keys()))
    if n != EXPECTED_TOTAL:
        fail(f"written file has {n} tensors, expected {EXPECTED_TOTAL}")
    print(f"merged {merged} MLP tensors with lambda={LAMBDA}; wrote {n} tensors to {OUT}")


if __name__ == "__main__":
    main()
