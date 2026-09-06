"""T4: task-vector merge of two GPT-2 fine-tunes into the base (lambda = 0.4)."""

import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "inputs" / "base" / "model.safetensors"
FT1 = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2 = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT = ROOT / "out" / "T4" / "model.safetensors"

LAMBDA = 0.4
N_LAYERS = 12
MLP_RE = re.compile(r"^h\.(\d+)\.mlp\.(c_fc|c_proj)\.(weight|bias)$")
EXPECTED_TOTAL = 160
EXPECTED_MLP = 4 * N_LAYERS  # 48


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base = load_file(str(BASE))
    ft1 = load_file(str(FT1))
    ft2 = load_file(str(FT2))

    # ---- Step 1: verify shared layout and identical non-MLP tensors -------
    names = set(base)
    if set(ft1) != names or set(ft2) != names:
        fail(
            "tensor name sets differ: "
            f"base^ft1={sorted(names ^ set(ft1))[:5]}, base^ft2={sorted(names ^ set(ft2))[:5]}"
        )
    if len(names) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} tensors in base, found {len(names)}")

    mlp_names = {n for n in names if MLP_RE.match(n)}
    if len(mlp_names) != EXPECTED_MLP:
        fail(f"expected {EXPECTED_MLP} MLP tensors, matched {len(mlp_names)}")
    for i in range(N_LAYERS):
        for leaf in ("c_fc.weight", "c_fc.bias", "c_proj.weight", "c_proj.bias"):
            if f"h.{i}.mlp.{leaf}" not in mlp_names:
                fail(f"missing MLP tensor h.{i}.mlp.{leaf}")

    for n in sorted(names):
        b, a, c = base[n], ft1[n], ft2[n]
        if not (b.shape == a.shape == c.shape):
            fail(f"shape mismatch for {n}: {tuple(b.shape)} {tuple(a.shape)} {tuple(c.shape)}")
        if not (b.dtype == a.dtype == c.dtype):
            fail(f"dtype mismatch for {n}: {b.dtype} {a.dtype} {c.dtype}")
        if n in mlp_names:
            continue
        # bit-exact comparison (torch.equal would treat NaN != NaN; compare raw bits)
        if not (
            torch.equal(b.view(torch.int32) if b.dtype == torch.float32 else b,
                        a.view(torch.int32) if a.dtype == torch.float32 else a)
            and torch.equal(b.view(torch.int32) if b.dtype == torch.float32 else b,
                            c.view(torch.int32) if c.dtype == torch.float32 else c)
        ):
            fail(f"non-MLP tensor {n} differs between base and a fine-tune")
    print(f"verified: {len(names)} shared names, {len(names) - len(mlp_names)} non-MLP tensors identical")

    # ---- Step 2/3: merge MLP tensors against the unmodified base ----------
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for n in sorted(names):
        if n in mlp_names:
            b = base[n].to(torch.float32)
            tv1 = ft1[n].to(torch.float32) - b
            tv2 = ft2[n].to(torch.float32) - b
            out[n] = (b + LAMBDA * tv1 + LAMBDA * tv2).to(base[n].dtype).contiguous()
            merged += 1
        else:
            out[n] = base[n].contiguous()

    if merged != EXPECTED_MLP:
        fail(f"merged {merged} tensors, expected {EXPECTED_MLP}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata={"format": "pt"})

    # ---- Post-write sanity: reload and re-check counts --------------------
    reloaded = load_file(str(OUT))
    if len(reloaded) != EXPECTED_TOTAL or set(reloaded) != names:
        fail("reloaded output does not have the expected 160 tensor names")
    print(f"merged {merged} MLP tensors with lambda={LAMBDA}; wrote {len(reloaded)} tensors to {OUT}")


if __name__ == "__main__":
    main()
