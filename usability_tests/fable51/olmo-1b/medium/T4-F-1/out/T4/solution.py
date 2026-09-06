"""T4: task-vector merge of two fine-tunes onto OLMo-1B base (lambda = 0.4).

Plain safetensors + torch script. Steps:
  1. Verify all three checkpoints have identical key sets and that every
     non-MLP tensor is bit-identical (shape, dtype, values) across the three.
  2. out[X] = base[X] + l*(ft1[X]-base[X]) + l*(ft2[X]-base[X]) for the 48 MLP
     tensors, every task vector taken against the unmodified base, in float32.
  3. All other tensors copied from base unchanged.
  4. Write a single out/T4/model.safetensors with exactly 114 tensors; re-open
     it and check the count and that unchanged tensors are bit-exact.
"""
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
FT1 = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2 = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT = ROOT / "out" / "T4" / "model.safetensors"

LAMBDA = 0.4
EXPECTED_TOTAL = 114
EXPECTED_MERGED = 48
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


class Sharded:
    """Read-only view over a (possibly sharded) safetensors checkpoint."""

    def __init__(self, files: list[Path]):
        self.handles = [safe_open(str(f), framework="pt", device="cpu") for f in files]
        self.where: dict[str, object] = {}
        for h in self.handles:
            for k in h.keys():
                if k in self.where:
                    fail(f"duplicate key across shards: {k}")
                self.where[k] = h

    def keys(self) -> set[str]:
        return set(self.where)

    def get(self, k: str) -> torch.Tensor:
        return self.where[k].get_tensor(k)


def load_base() -> Sharded:
    index = json.loads((BASE_DIR / "model.safetensors.index.json").read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    return Sharded([BASE_DIR / s for s in shard_names])


def main() -> None:
    if OUT.exists():
        fail(f"output already exists: {OUT}")

    base = load_base()
    ft1 = Sharded([FT1])
    ft2 = Sharded([FT2])

    # ---- Step 1: shared-tensor verification, before anything else ----------
    kb, k1, k2 = base.keys(), ft1.keys(), ft2.keys()
    if not (kb == k1 == k2):
        fail(
            "tensor name sets differ: "
            f"base-only={sorted(kb - k1 - k2)[:5]} ft1-only={sorted(k1 - kb)[:5]} "
            f"ft2-only={sorted(k2 - kb)[:5]}"
        )
    if len(kb) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} tensors, found {len(kb)}")

    mlp_keys = sorted(k for k in kb if MLP_RE.match(k))
    if len(mlp_keys) != EXPECTED_MERGED:
        fail(f"expected {EXPECTED_MERGED} MLP tensors, matched {len(mlp_keys)}")
    shared_keys = sorted(kb - set(mlp_keys))

    for k in shared_keys:
        tb, t1, t2 = base.get(k), ft1.get(k), ft2.get(k)
        for name, t in (("ft1", t1), ("ft2", t2)):
            if t.shape != tb.shape or t.dtype != tb.dtype:
                fail(f"{k}: {name} shape/dtype {tuple(t.shape)}/{t.dtype} "
                     f"!= base {tuple(tb.shape)}/{tb.dtype}")
            if not torch.equal(t, tb):
                fail(f"{k}: {name} differs from base but is outside the MLP set")
    print(f"verified {len(shared_keys)} shared tensors identical across base/ft1/ft2")

    # ---- Step 2 + 3: merge MLP tensors, copy the rest ----------------------
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for k in sorted(kb):
        tb = base.get(k)
        if k in mlp_keys:
            t1, t2 = ft1.get(k), ft2.get(k)
            for name, t in (("ft1", t1), ("ft2", t2)):
                if t.shape != tb.shape or t.dtype != tb.dtype:
                    fail(f"{k}: {name} shape/dtype mismatch vs base")
            if tb.dtype != torch.float32:
                fail(f"{k}: expected float32, got {tb.dtype}")
            # Both task vectors are taken against the *original* base tensor tb.
            tv1 = t1.float() - tb
            tv2 = t2.float() - tb
            out[k] = (tb + LAMBDA * tv1 + LAMBDA * tv2).to(torch.float32).contiguous()
            merged += 1
        else:
            out[k] = tb.contiguous()

    if merged != EXPECTED_MERGED:
        fail(f"merged {merged} tensors, expected {EXPECTED_MERGED}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    # ---- Step 4: write and re-verify ---------------------------------------
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata={"format": "pt"})

    with safe_open(str(OUT), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        if keys != kb:
            fail("written key set differs from base key set")
        if len(keys) != EXPECTED_TOTAL:
            fail(f"written file has {len(keys)} tensors, expected {EXPECTED_TOTAL}")
        for k in shared_keys:
            if not torch.equal(f.get_tensor(k), base.get(k)):
                fail(f"{k}: written tensor is not bit-identical to base")
        for k in mlp_keys:
            w = f.get_tensor(k)
            if w.dtype != torch.float32 or w.shape != base.get(k).shape:
                fail(f"{k}: written merged tensor has wrong dtype/shape")
    print(f"OK: wrote {OUT} with {EXPECTED_TOTAL} tensors ({merged} merged, "
          f"{len(shared_keys)} unchanged)")


if __name__ == "__main__":
    main()
