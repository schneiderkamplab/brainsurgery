#!/usr/bin/env python
"""T4: task-vector merge of two fine-tunes of OLMo-1B-0724-hf.

out[X] = base[X] + LAMBDA * (ft1[X] - base[X]) + LAMBDA * (ft2[X] - base[X])
for the 48 MLP tensors; every other tensor is copied from the base unchanged.

Step 1 (before any arithmetic): the three checkpoints must have identical
tensor names, and every non-MLP tensor must be identical (shape, dtype and
bit-exact values) in all three. Any violation aborts with an error.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]  # sandbox root
BASE_DIR = ROOT / "inputs" / "base"
FT1 = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2 = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT = ROOT / "out" / "T4" / "model.safetensors"

LAMBDA = 0.4
NUM_LAYERS = 16
EXPECTED_TOTAL = 114
EXPECTED_MLP = 48
MLP_NAMES = frozenset(
    f"model.layers.{i}.mlp.{proj}.weight"
    for i in range(NUM_LAYERS)
    for proj in ("gate_proj", "up_proj", "down_proj")
)


def fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


class Checkpoint:
    """Uniform read access to a single-file or sharded safetensors checkpoint."""

    def __init__(self, label: str, files: list[Path]):
        self.label = label
        self.handles = [safe_open(str(f), framework="pt", device="cpu") for f in files]
        self.where: dict[str, object] = {}
        for h in self.handles:
            for k in h.keys():
                if k in self.where:
                    fail(f"{label}: tensor {k!r} appears in more than one shard")
                self.where[k] = h

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self.where)

    def get(self, name: str) -> torch.Tensor:
        return self.where[name].get_tensor(name)


def open_base() -> Checkpoint:
    index = json.loads((BASE_DIR / "model.safetensors.index.json").read_text())
    shard_files = sorted(set(index["weight_map"].values()))
    ckpt = Checkpoint("base", [BASE_DIR / s for s in shard_files])
    if ckpt.names != frozenset(index["weight_map"]):
        fail("base: shard contents do not match model.safetensors.index.json")
    return ckpt


def main() -> None:
    base = open_base()
    ft1 = Checkpoint("ft1", [FT1])
    ft2 = Checkpoint("ft2", [FT2])

    # ---- Step 1: verify names and shared (non-MLP) tensors -----------------
    if not (base.names == ft1.names == ft2.names):
        for a, b in ((base, ft1), (base, ft2)):
            only_a = sorted(a.names - b.names)
            only_b = sorted(b.names - a.names)
            if only_a or only_b:
                print(f"{a.label} vs {b.label}: only in {a.label}={only_a[:5]}, "
                      f"only in {b.label}={only_b[:5]}", file=sys.stderr)
        fail("tensor name sets differ between base, ft1 and ft2")
    names = sorted(base.names)
    if len(names) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} tensors, found {len(names)}")
    missing_mlp = sorted(MLP_NAMES - base.names)
    if missing_mlp:
        fail(f"MLP tensors missing from checkpoints: {missing_mlp}")

    shared_names = [n for n in names if n not in MLP_NAMES]
    base_shared: dict[str, torch.Tensor] = {}
    for n in shared_names:
        tb, t1, t2 = base.get(n), ft1.get(n), ft2.get(n)
        for label, t in (("ft1", t1), ("ft2", t2)):
            if t.shape != tb.shape or t.dtype != tb.dtype:
                fail(f"shared tensor {n!r}: base {tuple(tb.shape)}/{tb.dtype} vs "
                     f"{label} {tuple(t.shape)}/{t.dtype}")
            if not torch.equal(tb, t):
                fail(f"shared tensor {n!r} differs between base and {label}; "
                     "fine-tune did not keep the backbone frozen")
        base_shared[n] = tb
    print(f"step 1 OK: {len(names)} names match; {len(shared_names)} shared tensors "
          f"identical in base, ft1 and ft2")

    # Also make sure the MLP tensors agree on shape/dtype before merging.
    for n in sorted(MLP_NAMES):
        sb = base.where[n].get_slice(n)
        for ck in (ft1, ft2):
            s = ck.where[n].get_slice(n)
            if s.get_shape() != sb.get_shape() or s.get_dtype() != sb.get_dtype():
                fail(f"MLP tensor {n!r}: shape/dtype mismatch between base and {ck.label}")
        if sb.get_dtype() != "F32":
            fail(f"MLP tensor {n!r} is {sb.get_dtype()}, expected F32")

    # ---- Step 2: task arithmetic against the unmodified base ---------------
    merged: dict[str, torch.Tensor] = {}
    for n in sorted(MLP_NAMES):
        b = base.get(n).to(torch.float32)
        d1 = ft1.get(n).to(torch.float32) - b   # task vector 1, against pristine base
        d2 = ft2.get(n).to(torch.float32) - b   # task vector 2, against pristine base
        out = b + LAMBDA * d1 + LAMBDA * d2
        if out.dtype != torch.float32 or out.shape != b.shape:
            fail(f"merged tensor {n!r} has unexpected dtype/shape")
        merged[n] = out.contiguous()
    if len(merged) != EXPECTED_MLP:
        fail(f"expected to merge {EXPECTED_MLP} tensors, merged {len(merged)}")
    print(f"step 2 OK: merged {len(merged)} MLP tensors with lambda={LAMBDA}")

    # ---- Step 3/4: assemble and write ---------------------------------------
    result: dict[str, torch.Tensor] = {}
    for n in names:
        result[n] = merged[n] if n in MLP_NAMES else base_shared[n].contiguous()
    if len(result) != EXPECTED_TOTAL:
        fail(f"output has {len(result)} tensors, expected {EXPECTED_TOTAL}")
    if set(result) != base.names:
        fail("output names differ from input names")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(result, str(OUT), metadata={"format": "pt"})

    # ---- Post-write verification of the file on disk -----------------------
    with safe_open(str(OUT), framework="pt", device="cpu") as f:
        out_names = set(f.keys())
        if len(out_names) != EXPECTED_TOTAL or out_names != base.names:
            fail(f"written file has {len(out_names)} tensors / wrong names")
        for n in shared_names:
            if not torch.equal(f.get_tensor(n), base_shared[n]):
                fail(f"written shared tensor {n!r} is not bit-identical to base")
        for n in sorted(MLP_NAMES)[:3]:
            t = f.get_tensor(n)
            if t.dtype != torch.float32 or not torch.equal(t, merged[n]):
                fail(f"written merged tensor {n!r} does not match computed value")
    print(f"wrote {OUT} with {len(out_names)} tensors "
          f"({len(shared_names)} unchanged, {len(merged)} merged)")


if __name__ == "__main__":
    main()
