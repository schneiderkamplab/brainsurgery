#!/usr/bin/env python
"""T4: task-vector merge of two fine-tunes of OLMo-1B-0724-hf.

    out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])

for the 48 MLP tensors; every other tensor is copied from the base verbatim.
Each task vector is taken against the *unmodified* base, so the merge is a
single pure expression over freshly loaded base tensors -- nothing is written
back into the base state dict before both differences are formed.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_DIR = os.path.join(ROOT, "inputs", "base")
FT1_FILE = os.path.join(ROOT, "inputs", "ft1", "model.safetensors")
FT2_FILE = os.path.join(ROOT, "inputs", "ft2", "model.safetensors")
OUT_FILE = os.path.join(ROOT, "out", "T4", "model.safetensors")

LAMBDA = 0.4
EXPECTED_TOTAL = 114
EXPECTED_MERGED = 48
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


class CheckFailed(RuntimeError):
    """A required check did not hold."""


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailed(msg)


def log(msg: str) -> None:
    print(msg, flush=True)


class Checkpoint:
    """Read-only view over one checkpoint, sharded or single-file."""

    def __init__(self, name: str, files: list[str], stack: contextlib.ExitStack) -> None:
        self.name = name
        self._handles = {
            path: stack.enter_context(safe_open(path, framework="pt")) for path in files
        }
        self._where: dict[str, str] = {}
        for path, handle in self._handles.items():
            for key in handle.keys():
                check(
                    key not in self._where,
                    f"{name}: tensor {key!r} appears in more than one shard",
                )
                self._where[key] = path

    @property
    def keys(self) -> set[str]:
        return set(self._where)

    def get(self, key: str) -> torch.Tensor:
        return self._handles[self._where[key]].get_tensor(key)


def base_shard_files() -> list[str]:
    index_path = os.path.join(BASE_DIR, "model.safetensors.index.json")
    with open(index_path) as fh:
        index = json.load(fh)
    shards = sorted(set(index["weight_map"].values()))
    return [os.path.join(BASE_DIR, shard) for shard in shards]


def same_shape_dtype(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype


def same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-exact comparison (unlike ==, this also separates -0.0/0.0 and NaNs)."""
    if not same_shape_dtype(a, b):
        return False
    va = a.contiguous().numpy().view(np.uint8)
    vb = b.contiguous().numpy().view(np.uint8)
    return bool(np.array_equal(va, vb))


def main() -> int:
    with contextlib.ExitStack() as stack:
        return run(stack)


def run(stack: contextlib.ExitStack) -> int:
    log("== loading checkpoint headers ==")
    base = Checkpoint("base", base_shard_files(), stack)
    ft1 = Checkpoint("ft1", [FT1_FILE], stack)
    ft2 = Checkpoint("ft2", [FT2_FILE], stack)

    # ---- step 1a: identical tensor name sets -------------------------------
    for other in (ft1, ft2):
        missing = sorted(base.keys - other.keys)
        extra = sorted(other.keys - base.keys)
        check(
            not missing and not extra,
            f"{other.name} tensor names differ from base: "
            f"missing={missing[:5]} extra={extra[:5]}",
        )
    keys = sorted(base.keys)
    check(
        len(keys) == EXPECTED_TOTAL,
        f"expected {EXPECTED_TOTAL} tensors in base, found {len(keys)}",
    )
    log(f"tensor names match across base/ft1/ft2 ({len(keys)} tensors)")

    # ---- classify the MLP tensors ------------------------------------------
    mlp_keys = sorted(k for k in keys if MLP_RE.match(k))
    layers = sorted({int(MLP_RE.match(k).group(1)) for k in mlp_keys})
    expected_mlp = {
        f"model.layers.{i}.mlp.{proj}.weight"
        for i in layers
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    check(
        set(mlp_keys) == expected_mlp,
        "MLP tensor set is not exactly gate/up/down for every layer: "
        f"{sorted(expected_mlp.symmetric_difference(mlp_keys))[:5]}",
    )
    check(
        len(mlp_keys) == EXPECTED_MERGED,
        f"expected {EXPECTED_MERGED} MLP tensors, found {len(mlp_keys)}",
    )
    other_keys = [k for k in keys if k not in expected_mlp]
    log(f"{len(mlp_keys)} MLP tensors, {len(other_keys)} tensors outside the MLPs")

    # ---- step 1c: everything outside the MLPs is identical in all three ----
    out_tensors: dict[str, torch.Tensor] = {}
    for n, key in enumerate(other_keys, start=1):
        b = base.get(key)
        for other in (ft1, ft2):
            o = other.get(key)
            check(
                same_shape_dtype(b, o),
                f"{key}: base is {tuple(b.shape)}/{b.dtype}, "
                f"{other.name} is {tuple(o.shape)}/{o.dtype}",
            )
            check(
                same_bytes(b, o),
                f"shared-tensor check failed: {key} differs between base and {other.name}",
            )
            del o
        out_tensors[key] = b
        if n % 20 == 0 or n == len(other_keys):
            log(f"  verified {n}/{len(other_keys)} shared tensors")
    log(f"all {len(other_keys)} non-MLP tensors are bit-identical in base, ft1 and ft2")

    # ---- step 2: the task-vector merge -------------------------------------
    merged = 0
    for key in mlp_keys:
        b = base.get(key)
        check(
            b.dtype == torch.float32,
            f"{key}: expected float32 input, got {b.dtype}",
        )
        f1, f2 = ft1.get(key), ft2.get(key)
        for other, o in (("ft1", f1), ("ft2", f2)):
            check(
                same_shape_dtype(b, o),
                f"{key}: base is {tuple(b.shape)}/{b.dtype}, "
                f"{other} is {tuple(o.shape)}/{o.dtype}",
            )
        b32 = b.to(torch.float32)
        tv1 = f1.to(torch.float32) - b32
        tv2 = f2.to(torch.float32) - b32
        merged_t = b32 + LAMBDA * tv1 + LAMBDA * tv2
        check(
            torch.isfinite(merged_t).all().item(),
            f"{key}: merged tensor contains non-finite values",
        )
        out_tensors[key] = merged_t.to(b.dtype).contiguous()
        merged += 1
        if merged % 12 == 0 or merged == len(mlp_keys):
            log(f"  merged {merged}/{len(mlp_keys)} MLP tensors")
    check(
        merged == EXPECTED_MERGED,
        f"merged {merged} tensors, expected exactly {EXPECTED_MERGED}",
    )

    # ---- step 3/4: write a single file with all 114 tensors ----------------
    check(
        len(out_tensors) == EXPECTED_TOTAL,
        f"output holds {len(out_tensors)} tensors, expected {EXPECTED_TOTAL}",
    )
    check(
        set(out_tensors) == set(keys),
        "output tensor names differ from the base tensor names",
    )
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    out_tensors = {k: v.contiguous() for k, v in out_tensors.items()}
    save_file(out_tensors, OUT_FILE, metadata={"format": "pt"})
    log(f"wrote {OUT_FILE} ({os.path.getsize(OUT_FILE)} bytes)")

    # ---- read the file back and check it against what we intended ---------
    written = Checkpoint("out", [OUT_FILE], stack)
    check(
        len(written.keys) == EXPECTED_TOTAL,
        f"written file has {len(written.keys)} tensors, expected {EXPECTED_TOTAL}",
    )
    check(written.keys == set(keys), "written file has the wrong tensor names")
    for key in keys:
        check(
            same_bytes(written.get(key), out_tensors[key]),
            f"{key}: tensor read back from the output does not match what was computed",
        )
    log(f"read back {EXPECTED_TOTAL} tensors from the output; all match")
    log(f"OK: {merged} MLP tensors merged with lambda={LAMBDA}, "
        f"{len(other_keys)} tensors copied from the base")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailed as exc:
        print(f"FAILED CHECK: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
