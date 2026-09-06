#!/usr/bin/env python
"""T4: task-vector merge of two OLMo-1B fine-tunes against a shared base.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the
48 MLP tensors; every other tensor is copied unchanged from base. Verifies
before merging that all three checkpoints share the same tensor names and
that every non-MLP tensor is bit-identical across all three (frozen-backbone
precondition for task arithmetic). Plain script on torch + safetensors, no
brainsurgery.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
MLP_RE = re.compile(r"^model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root (out/T4 -> out -> root)
INPUTS = ROOT / "inputs"
OUT_DIR = HERE
OUT_FILE = OUT_DIR / "model.safetensors"


def load_sharded_or_single(dir_path: Path) -> dict[str, torch.Tensor]:
    index_path = dir_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index["weight_map"].values()))
        tensors: dict[str, torch.Tensor] = {}
        for shard_name in shard_names:
            tensors.update(load_file(str(dir_path / shard_name)))
        return tensors
    return load_file(str(dir_path / "model.safetensors"))


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base = load_sharded_or_single(INPUTS / "base")
    ft1 = load_sharded_or_single(INPUTS / "ft1")
    ft2 = load_sharded_or_single(INPUTS / "ft2")

    # --- Step 1: shared-tensor verification ---
    base_keys, ft1_keys, ft2_keys = set(base), set(ft1), set(ft2)
    if not (base_keys == ft1_keys == ft2_keys):
        only_base = base_keys - ft1_keys - ft2_keys
        only_ft1 = ft1_keys - base_keys
        only_ft2 = ft2_keys - base_keys
        fail(
            "tensor name sets differ across checkpoints: "
            f"base-only(sample)={sorted(only_base)[:5]} "
            f"ft1-only(sample)={sorted(only_ft1)[:5]} "
            f"ft2-only(sample)={sorted(only_ft2)[:5]}"
        )

    mlp_keys = {k for k in base_keys if MLP_RE.match(k)}
    if len(mlp_keys) != 48:
        fail(f"expected exactly 48 MLP tensors, found {len(mlp_keys)}: {sorted(mlp_keys)}")

    non_mlp_keys = base_keys - mlp_keys
    mismatched = []
    for k in sorted(non_mlp_keys):
        b, f1, f2 = base[k], ft1[k], ft2[k]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatched.append((k, "shape", b.shape, f1.shape, f2.shape))
            continue
        if not (torch.equal(b, f1) and torch.equal(b, f2)):
            mismatched.append((k, "value", None, None, None))
    if mismatched:
        fail(
            "non-MLP tensors are not identical across base/ft1/ft2 "
            f"(frozen-backbone assumption violated): {mismatched[:10]}"
        )

    # --- Step 2 & 3: merge ---
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for k in base_keys:
        b = base[k]
        if k in mlp_keys:
            if b.dtype != torch.float32:
                fail(f"expected float32 for {k}, got {b.dtype}")
            f1, f2 = ft1[k], ft2[k]
            if b.shape != f1.shape or b.shape != f2.shape:
                fail(f"shape mismatch for MLP tensor {k}: base={b.shape} ft1={f1.shape} ft2={f2.shape}")
            merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
            out[k] = merged.contiguous()
            merged_count += 1
        else:
            out[k] = b.clone().contiguous()

    if merged_count != 48:
        fail(f"expected to merge exactly 48 tensors, merged {merged_count}")
    if len(out) != 114:
        fail(f"expected exactly 114 output tensors, got {len(out)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_FILE), metadata={"format": "pt"})
    print(f"OK: wrote {len(out)} tensors ({merged_count} merged) to {OUT_FILE}")


if __name__ == "__main__":
    main()
