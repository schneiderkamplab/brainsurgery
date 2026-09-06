"""T4: task-vector merge of two frozen-backbone fine-tunes of GPT-2 (124M).

out[X] = base[X] + lam*(ft1[X]-base[X]) + lam*(ft2[X]-base[X])   for the 48 MLP tensors
out[X] = base[X]                                                   for everything else

Every task vector is taken against the *unmodified* base (base is never
mutated). All checks raise; nothing is written unless they all pass.
"""
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "inputs/base/model.safetensors"
FT1 = ROOT / "inputs/ft1/model.safetensors"
FT2 = ROOT / "inputs/ft2/model.safetensors"
OUT = ROOT / "out/T4/model.safetensors"

LAM = 0.4
N_LAYERS = 12
MLP_NAMES = frozenset(
    f"h.{i}.mlp.{p}"
    for i in range(N_LAYERS)
    for p in ("c_fc.weight", "c_fc.bias", "c_proj.weight", "c_proj.bias")
)
EXPECTED_MERGED = 48
EXPECTED_TOTAL = 160


def fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def load(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def main() -> None:
    if len(MLP_NAMES) != EXPECTED_MERGED:
        fail(f"MLP allowlist has {len(MLP_NAMES)} names, expected {EXPECTED_MERGED}")
    if OUT.exists():
        fail(f"output already exists: {OUT}")

    base, ft1, ft2 = load(BASE), load(FT1), load(FT2)

    # ---- step 1: verify before touching anything --------------------------
    kb, k1, k2 = set(base), set(ft1), set(ft2)
    if not (kb == k1 == k2):
        fail(
            "tensor name sets differ: "
            f"base-only={sorted(kb - k1 - k2)}, ft1-only={sorted(k1 - kb)}, ft2-only={sorted(k2 - kb)}"
        )
    if len(kb) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} tensors in inputs, found {len(kb)}")
    missing = MLP_NAMES - kb
    if missing:
        fail(f"MLP tensors missing from inputs: {sorted(missing)}")

    for name in sorted(kb):
        b, a1, a2 = base[name], ft1[name], ft2[name]
        for tag, t in (("ft1", a1), ("ft2", a2)):
            if t.shape != b.shape or t.dtype != b.dtype:
                fail(f"{name}: {tag} shape/dtype {tuple(t.shape)}/{t.dtype} != base {tuple(b.shape)}/{b.dtype}")
        if name in MLP_NAMES:
            continue
        # bit-exact comparison of every non-MLP tensor across the three checkpoints
        if not (torch.equal(b, a1) and torch.equal(b, a2)):
            fail(f"non-MLP tensor differs between checkpoints: {name}")

    for name in sorted(MLP_NAMES):
        if base[name].dtype != torch.float32:
            fail(f"{name}: expected float32, got {base[name].dtype}")

    # ---- step 2/3: merge ---------------------------------------------------
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for name in sorted(kb):
        b = base[name]
        if name in MLP_NAMES:
            tv1 = ft1[name] - b   # against unmodified base
            tv2 = ft2[name] - b   # against unmodified base
            m = b + LAM * tv1 + LAM * tv2
            if m.dtype != torch.float32 or m.shape != b.shape:
                fail(f"{name}: merged tensor has wrong dtype/shape")
            out[name] = m.contiguous()
            merged += 1
        else:
            out[name] = b.contiguous()

    if merged != EXPECTED_MERGED:
        fail(f"merged {merged} tensors, expected {EXPECTED_MERGED}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"output dict has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    # ---- step 4: save and re-verify from disk ------------------------------
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata={"format": "pt"})

    disk = load(OUT)
    if len(disk) != EXPECTED_TOTAL:
        fail(f"saved file has {len(disk)} tensors, expected {EXPECTED_TOTAL}")
    if set(disk) != kb:
        fail("saved key set differs from input key set")
    n_changed = 0
    for name in sorted(kb):
        d = disk[name]
        if d.shape != base[name].shape or d.dtype != base[name].dtype:
            fail(f"{name}: saved shape/dtype differs from base")
        if name in MLP_NAMES:
            ref = base[name] + LAM * (ft1[name] - base[name]) + LAM * (ft2[name] - base[name])
            if not torch.equal(d, ref):
                fail(f"{name}: saved merged tensor != recomputed merge")
            n_changed += 1
        elif not torch.equal(d, base[name]):
            fail(f"{name}: saved unchanged tensor != base")
    if n_changed != EXPECTED_MERGED:
        fail(f"post-save: {n_changed} merged tensors verified, expected {EXPECTED_MERGED}")

    print(f"OK: wrote {OUT} with {len(disk)} tensors, {n_changed} merged (lambda={LAM})")


if __name__ == "__main__":
    sys.exit(main())
