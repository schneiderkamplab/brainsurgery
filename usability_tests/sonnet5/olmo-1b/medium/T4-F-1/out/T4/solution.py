"""
T4: task-vector merge of two OLMo-1B fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])   for the 48 MLP tensors
out[X] = base[X]                                                       for everything else

Plain torch/safetensors script (no merge toolkit needed for a two-file
task-vector add): this keeps every check explicit and auditable, which is
exactly what the "abort loudly if the precondition fails" requirement in
TASK.md wants.

Usage: python solution.py
Reads from ../../inputs (relative to this file), writes out/T4/model.safetensors
next to it (i.e. two directories up from this file, at out/T4/).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
NUM_LAYERS = 16

HERE = Path(__file__).resolve().parent  # out/T4
TASK_DIR = HERE.parent.parent  # sandbox root
INPUTS = TASK_DIR / "inputs"
OUT_FILE = HERE / "model.safetensors"

MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def load_sharded_or_single(path: Path) -> dict[str, torch.Tensor]:
    """Load a checkpoint directory that is either a single model.safetensors
    file or a sharded one with model.safetensors.index.json."""
    index_path = path / "model.safetensors.index.json"
    tensors: dict[str, torch.Tensor] = {}
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shard_files = sorted(set(weight_map.values()))
        for shard in shard_files:
            with safe_open(path / shard, framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
        if set(tensors) != set(weight_map):
            die(f"{path}: index.json weight_map keys do not match tensors found in shards")
    else:
        single = path / "model.safetensors"
        if not single.exists():
            die(f"{path}: neither model.safetensors.index.json nor model.safetensors found")
        with safe_open(single, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def expected_mlp_names() -> set[str]:
    names = set()
    for i in range(NUM_LAYERS):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            names.add(f"model.layers.{i}.mlp.{proj}.weight")
    return names


def main() -> None:
    base = load_sharded_or_single(INPUTS / "base")
    ft1 = load_sharded_or_single(INPUTS / "ft1")
    ft2 = load_sharded_or_single(INPUTS / "ft2")

    # --- Step 1: verify identical key sets across all three checkpoints ---
    keys_base, keys_ft1, keys_ft2 = set(base), set(ft1), set(ft2)
    if not (keys_base == keys_ft1 == keys_ft2):
        die(
            "checkpoints do not share the same tensor names: "
            f"base-only={keys_base - keys_ft1 - keys_ft2}, "
            f"ft1-only={keys_ft1 - keys_base - keys_ft2}, "
            f"ft2-only={keys_ft2 - keys_base - keys_ft1}"
        )
    if len(keys_base) != 114:
        die(f"expected 114 tensors, base has {len(keys_base)}")

    mlp_names = expected_mlp_names()
    found_mlp_names = {k for k in keys_base if MLP_RE.match(k)}
    if found_mlp_names != mlp_names:
        die(
            "MLP tensor names don't match the expected 48: "
            f"missing={mlp_names - found_mlp_names}, unexpected={found_mlp_names - mlp_names}"
        )
    if len(mlp_names) != 48:
        die(f"expected 48 MLP tensors, computed {len(mlp_names)}")

    non_mlp_names = keys_base - mlp_names

    # --- Step 1 (cont.): every non-MLP tensor must be bit-identical in all three ---
    mismatches = []
    for name in non_mlp_names:
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatches.append(f"{name}: shape mismatch base={b.shape} ft1={f1.shape} ft2={f2.shape}")
            continue
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            mismatches.append(f"{name}: dtype mismatch base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")
            continue
        if not torch.equal(b, f1) or not torch.equal(b, f2):
            mismatches.append(f"{name}: values differ outside the MLP tensor set")
    if mismatches:
        die(
            "non-MLP tensors are not identical across base/ft1/ft2 "
            f"({len(mismatches)} mismatches), first few:\n" + "\n".join(mismatches[:10])
        )

    # --- Step 2: merge the 48 MLP tensors by task arithmetic, in float32 ---
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in mlp_names:
        b = base[name].to(torch.float32)
        f1 = ft1[name].to(torch.float32)
        f2 = ft2[name].to(torch.float32)
        if b.shape != f1.shape or b.shape != f2.shape:
            die(f"{name}: shape mismatch between base/ft1/ft2")
        merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
        out[name] = merged.to(base[name].dtype).contiguous()
        merged_count += 1

    if merged_count != 48:
        die(f"expected to merge exactly 48 tensors, merged {merged_count}")

    # --- Step 3: everything else copied unchanged from base ---
    for name in non_mlp_names:
        out[name] = base[name].contiguous()

    # --- Step 4: write a single output file with exactly 114 tensors ---
    if len(out) != 114:
        die(f"output has {len(out)} tensors, expected 114")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_FILE), metadata={"format": "pt"})

    with safe_open(OUT_FILE, framework="pt") as f:
        written = list(f.keys())
    if len(written) != 114:
        die(f"wrote {len(written)} tensors to {OUT_FILE}, expected 114")

    print(f"OK: wrote {len(written)} tensors ({merged_count} merged, "
          f"{len(non_mlp_names)} copied unchanged) to {OUT_FILE}")


if __name__ == "__main__":
    main()
