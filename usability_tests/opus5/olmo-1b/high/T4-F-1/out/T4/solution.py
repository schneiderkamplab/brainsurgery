#!/usr/bin/env python3
"""T4: task-vector merge of two fine-tunes of OLMo-1B-0724-hf.

    out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])

for the 48 MLP tensors only; every other tensor is copied from the base
unchanged.  Both task vectors are taken against the *unmodified* base: the
base tensor is loaded fresh per name and never written to.

Every requirement of the task is enforced by an assertion that aborts the run.
Tensors are streamed one name at a time (safetensors lazy `safe_open`), so
peak memory is a few tensors, not three 5 GB checkpoints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
N_LAYERS = 16
N_TOTAL = 114
N_MLP = 48

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # sandbox root
INPUTS = ROOT / "inputs"
OUT_DIR = ROOT / "out" / "T4"
OUT_FILE = OUT_DIR / "model.safetensors"


class CheckFailed(RuntimeError):
    """A required check did not hold; the run must abort."""


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailed(msg)


class Checkpoint:
    """Lazy read-only view over a single-file or sharded safetensors checkpoint."""

    def __init__(self, label: str, path: Path):
        self.label = label
        index = path / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            files = sorted({path / f for f in weight_map.values()})
        elif (path / "model.safetensors").is_file():
            files = [path / "model.safetensors"]
        else:
            raise CheckFailed(f"{label}: no model.safetensors[.index.json] under {path}")

        self._handles = {}
        self._owner: dict[str, object] = {}
        for f in files:
            h = safe_open(str(f), framework="pt", device="cpu")
            self._handles[f] = h
            for key in h.keys():
                require(key not in self._owner, f"{label}: duplicate tensor name {key!r}")
                self._owner[key] = h

    @property
    def keys(self) -> set[str]:
        return set(self._owner)

    def get(self, name: str) -> torch.Tensor:
        return self._owner[name].get_tensor(name)

    def meta(self, name: str) -> tuple[tuple[int, ...], torch.dtype]:
        sl = self._owner[name].get_slice(name)
        return tuple(sl.get_shape()), self.get_dtype(sl)

    _DTYPES = {
        "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
        "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
        "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
    }

    @classmethod
    def get_dtype(cls, sl) -> torch.dtype:
        raw = sl.get_dtype()
        require(raw in cls._DTYPES, f"unsupported safetensors dtype {raw!r}")
        return cls._DTYPES[raw]


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Exact bit-level comparison (stricter than ==: separates NaN and -0.0)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(
        a.contiguous().view(torch.uint8).reshape(-1),
        b.contiguous().view(torch.uint8).reshape(-1),
    )


def expected_mlp_names() -> set[str]:
    return {
        f"model.layers.{i}.mlp.{proj}.weight"
        for i in range(N_LAYERS)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }


def main() -> int:
    base = Checkpoint("base", INPUTS / "base")
    ft1 = Checkpoint("ft1", INPUTS / "ft1")
    ft2 = Checkpoint("ft2", INPUTS / "ft2")

    # --- Check 1a: identical tensor name sets across all three checkpoints ---
    for other in (ft1, ft2):
        only_base = sorted(base.keys - other.keys)
        only_other = sorted(other.keys - base.keys)
        require(
            not only_base and not only_other,
            f"tensor names differ between base and {other.label}: "
            f"missing from {other.label}={only_base[:5]}, extra in {other.label}={only_other[:5]}",
        )
    names = sorted(base.keys)
    require(
        len(names) == N_TOTAL,
        f"expected {N_TOTAL} tensors in the base, found {len(names)}",
    )

    # --- Check 1b: the 48 MLP tensors are exactly the ones the task names ---
    mlp = expected_mlp_names()
    missing = sorted(mlp - base.keys)
    require(not missing, f"MLP tensors absent from the checkpoints: {missing}")
    require(len(mlp) == N_MLP, f"expected {N_MLP} MLP tensor names, built {len(mlp)}")

    # --- Check 1c: shapes and dtypes agree across the three checkpoints ---
    for name in names:
        bm = base.meta(name)
        for other in (ft1, ft2):
            om = other.meta(name)
            require(
                bm == om,
                f"{name}: base has shape/dtype {bm}, {other.label} has {om}",
            )

    # --- Check 1d: every tensor outside the 48 MLP tensors is identical ---
    shared = [n for n in names if n not in mlp]
    require(
        len(shared) == N_TOTAL - N_MLP,
        f"expected {N_TOTAL - N_MLP} shared tensors, found {len(shared)}",
    )
    for name in shared:
        b = base.get(name)
        for other in (ft1, ft2):
            require(
                bitwise_equal(b, other.get(name)),
                f"backbone assumption violated: {name} differs between base and {other.label}",
            )
    print(f"[ok] {len(shared)} non-MLP tensors are bit-identical in base, ft1 and ft2")

    # --- Merge ---
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for name in names:
        b = base.get(name)
        if name in mlp:
            require(b.dtype == torch.float32, f"{name}: expected float32, got {b.dtype}")
            # base is read fresh here and never mutated, so both task vectors
            # are taken against the unmodified base.
            b32 = b.to(torch.float32)
            tv1 = ft1.get(name).to(torch.float32) - b32
            tv2 = ft2.get(name).to(torch.float32) - b32
            out[name] = (b32 + LAMBDA * tv1 + LAMBDA * tv2).contiguous()
            merged += 1
        else:
            out[name] = b.contiguous()

    # --- Required check: exactly 48 tensors were merged ---
    require(merged == N_MLP, f"merged {merged} tensors, expected exactly {N_MLP}")
    # --- Required check: the output has exactly 114 tensors ---
    require(len(out) == N_TOTAL, f"output holds {len(out)} tensors, expected {N_TOTAL}")
    print(f"[ok] merged {merged} MLP tensors with lambda={LAMBDA}, "
          f"copied {len(out) - merged} unchanged")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_FILE))
    del out

    # --- Post-write verification against the file that was actually written ---
    written = Checkpoint("out", OUT_DIR)
    require(
        len(written.keys) == N_TOTAL,
        f"written file holds {len(written.keys)} tensors, expected {N_TOTAL}",
    )
    require(
        written.keys == base.keys,
        "written tensor names do not match the base tensor names",
    )
    changed = 0
    for name in names:
        b = base.get(name)
        w = written.get(name)
        require(
            (w.shape, w.dtype) == (b.shape, b.dtype),
            f"{name}: written shape/dtype {(w.shape, w.dtype)} != base {(b.shape, b.dtype)}",
        )
        if name in mlp:
            expect = b.to(torch.float32) + LAMBDA * (
                ft1.get(name).to(torch.float32) - b.to(torch.float32)
            ) + LAMBDA * (ft2.get(name).to(torch.float32) - b.to(torch.float32))
            err = torch.linalg.vector_norm(w.to(torch.float32) - expect)
            denom = torch.linalg.vector_norm(expect)
            rel = float(err / denom) if float(denom) > 0 else float(err)
            require(rel <= 1e-6, f"{name}: written value off by relative {rel:.3e}")
            changed += 1
        else:
            require(
                bitwise_equal(b, w),
                f"{name}: should have been copied from the base unchanged, but differs",
            )
    require(changed == N_MLP, f"{changed} tensors changed in the output, expected {N_MLP}")
    print(f"[ok] verified {OUT_FILE}: {N_TOTAL} tensors, "
          f"{changed} merged, {N_TOTAL - changed} bit-identical to the base")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailed as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
