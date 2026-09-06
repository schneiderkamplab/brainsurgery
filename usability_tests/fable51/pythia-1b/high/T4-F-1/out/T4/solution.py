"""T4: task-vector merge of two frozen-backbone fine-tunes into a Pythia-1B base.

out[X] = base[X] + lam*(ft1[X]-base[X]) + lam*(ft2[X]-base[X])   for the 64 MLP tensors,
computed in float32 and cast back to the base dtype; every other tensor is copied from base.
Both task vectors are taken against the unmodified base.
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
N_LAYERS = 16
EXPECTED_TOTAL = 244
EXPECTED_MERGED = 64
MLP_RE = re.compile(
    r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_(h_to_4h|4h_to_h)\.(weight|bias)$"
)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    for p in (BASE, FT1, FT2):
        if not p.is_file():
            fail(f"missing input {p}")
    if OUT.exists():
        fail(f"output already exists: {OUT}")

    with safe_open(BASE, "pt") as fb, safe_open(FT1, "pt") as f1, safe_open(FT2, "pt") as f2:
        kb, k1, k2 = set(fb.keys()), set(f1.keys()), set(f2.keys())
        # Step 1a: identical tensor names in all three checkpoints.
        if not (kb == k1 == k2):
            fail(
                "tensor name sets differ: "
                f"ft1-base={sorted(k1 - kb)} base-ft1={sorted(kb - k1)} "
                f"ft2-base={sorted(k2 - kb)} base-ft2={sorted(kb - k2)}"
            )
        names = sorted(kb)
        if len(names) != EXPECTED_TOTAL:
            fail(f"expected {EXPECTED_TOTAL} tensors in base, found {len(names)}")

        mlp_names = [n for n in names if MLP_RE.match(n)]
        if len(mlp_names) != EXPECTED_MERGED:
            fail(f"expected {EXPECTED_MERGED} MLP tensors, matched {len(mlp_names)}")
        layers = {int(MLP_RE.match(n).group(1)) for n in mlp_names}
        if layers != set(range(N_LAYERS)):
            fail(f"MLP tensors cover layers {sorted(layers)}, expected 0..{N_LAYERS - 1}")
        mlp_set = set(mlp_names)

        # Step 1b: every non-MLP tensor is bit-identical (shape, dtype, values) in all three.
        # Also check that MLP tensors agree in shape and dtype across checkpoints.
        for n in names:
            tb, t1, t2 = fb.get_tensor(n), f1.get_tensor(n), f2.get_tensor(n)
            for tag, t in (("ft1", t1), ("ft2", t2)):
                if t.shape != tb.shape or t.dtype != tb.dtype:
                    fail(
                        f"{n}: {tag} has shape {tuple(t.shape)} dtype {t.dtype}, "
                        f"base has {tuple(tb.shape)} {tb.dtype}"
                    )
            if n not in mlp_set:
                if not torch.equal(tb, t1):
                    fail(f"shared tensor {n} differs between base and ft1")
                if not torch.equal(tb, t2):
                    fail(f"shared tensor {n} differs between base and ft2")
        print(f"verified: {len(names)} names identical, {len(names) - len(mlp_set)} shared "
              f"tensors bit-identical across base/ft1/ft2")

        # Step 2 + 3: merge the 64 MLP tensors, copy everything else from base.
        out: dict[str, torch.Tensor] = {}
        merged = 0
        for n in names:
            tb = fb.get_tensor(n)
            if n in mlp_set:
                b32 = tb.to(torch.float32)
                tv1 = f1.get_tensor(n).to(torch.float32) - b32
                tv2 = f2.get_tensor(n).to(torch.float32) - b32
                out[n] = (b32 + LAMBDA * tv1 + LAMBDA * tv2).to(tb.dtype).contiguous()
                merged += 1
            else:
                out[n] = tb.contiguous()

    if merged != EXPECTED_MERGED:
        fail(f"merged {merged} tensors, expected {EXPECTED_MERGED}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata={"format": "pt"})

    # Step 4: re-open the written file and verify the tensor count and key set.
    with safe_open(OUT, "pt") as fo:
        written = set(fo.keys())
    if len(written) != EXPECTED_TOTAL or written != set(names):
        fail(f"written file has {len(written)} tensors / key set mismatch")
    print(f"wrote {OUT} with {len(written)} tensors ({merged} merged, lambda={LAMBDA})")


if __name__ == "__main__":
    main()
