#!/usr/bin/env python3
"""T4: task-vector merge of two OLMo-1B-0724-hf fine-tunes.

out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])

for the 48 MLP tensors (gate_proj/up_proj/down_proj weights across the 16
layers); every other tensor is copied unchanged from base. Aborts loudly if
the frozen-backbone precondition does not hold.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # out/T4 -> out -> sandbox root
INPUTS = REPO_ROOT / "inputs"
OUT_DIR = REPO_ROOT / "out" / "T4"
LAMBDA = 0.4

MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def die(msg: str) -> None:
    print(f"ABORT: {msg}", file=sys.stderr)
    sys.exit(1)


def load_sharded(dir_path: Path) -> dict[str, torch.Tensor]:
    """Load a safetensors checkpoint that may be sharded (index.json) or a
    single model.safetensors file."""
    index_path = dir_path / "model.safetensors.index.json"
    tensors: dict[str, torch.Tensor] = {}
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        for shard_name in shard_files:
            with safe_open(dir_path / shard_name, framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
    else:
        single = dir_path / "model.safetensors"
        if not single.exists():
            die(f"no model.safetensors or model.safetensors.index.json under {dir_path}")
        with safe_open(single, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    base = load_sharded(INPUTS / "base")
    ft1 = load_sharded(INPUTS / "ft1")
    ft2 = load_sharded(INPUTS / "ft2")

    # --- Step 1: verify same tensor names across all three checkpoints. ---
    base_keys, ft1_keys, ft2_keys = set(base), set(ft1), set(ft2)
    if not (base_keys == ft1_keys == ft2_keys):
        only_in_base = base_keys - (ft1_keys & ft2_keys)
        only_in_ft1 = ft1_keys - (base_keys & ft2_keys)
        only_in_ft2 = ft2_keys - (base_keys & ft1_keys)
        die(
            "tensor name sets differ across checkpoints: "
            f"base-only={sorted(only_in_base)[:5]} "
            f"ft1-only={sorted(only_in_ft1)[:5]} "
            f"ft2-only={sorted(only_in_ft2)[:5]}"
        )
    if len(base_keys) != 114:
        die(f"expected 114 tensors, base has {len(base_keys)}")

    mlp_keys = {k for k in base_keys if MLP_RE.match(k)}
    expected_mlp = {
        f"model.layers.{i}.mlp.{proj}.weight"
        for i in range(16)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    if mlp_keys != expected_mlp:
        die(
            "MLP tensor set does not match the expected 48 names "
            f"(missing={sorted(expected_mlp - mlp_keys)[:5]}, "
            f"unexpected={sorted(mlp_keys - expected_mlp)[:5]})"
        )
    if len(mlp_keys) != 48:
        die(f"expected exactly 48 MLP tensors, found {len(mlp_keys)}")

    # --- Verify every non-MLP tensor is identical (dtype, shape, values) in
    # --- all three checkpoints -- the frozen-backbone precondition.
    non_mlp_keys = base_keys - mlp_keys
    mismatches = []
    for key in sorted(non_mlp_keys):
        b, f1, f2 = base[key], ft1[key], ft2[key]
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            mismatches.append(f"{key}: dtype mismatch base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")
            continue
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatches.append(f"{key}: shape mismatch base={b.shape} ft1={f1.shape} ft2={f2.shape}")
            continue
        if not torch.equal(b, f1):
            mismatches.append(f"{key}: base != ft1 (expected identical, frozen backbone)")
        if not torch.equal(b, f2):
            mismatches.append(f"{key}: base != ft2 (expected identical, frozen backbone)")
    if mismatches:
        die(
            "frozen-backbone precondition violated for non-MLP tensors:\n  "
            + "\n  ".join(mismatches[:20])
            + (f"\n  ... and {len(mismatches) - 20} more" if len(mismatches) > 20 else "")
        )

    # --- Step 2 & 3: compute the merge. ---
    out: dict[str, torch.Tensor] = {}
    for key in non_mlp_keys:
        out[key] = base[key].clone()

    merged_count = 0
    for key in sorted(mlp_keys):
        b = base[key].to(torch.float32)
        f1 = ft1[key].to(torch.float32)
        f2 = ft2[key].to(torch.float32)
        if b.shape != f1.shape or b.shape != f2.shape:
            die(f"{key}: shape mismatch base={b.shape} ft1={f1.shape} ft2={f2.shape}")
        merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
        out[key] = merged.to(base[key].dtype).contiguous()
        merged_count += 1

    # --- Required checks ---
    if merged_count != 48:
        die(f"expected exactly 48 merged tensors, computed {merged_count}")
    if len(out) != 114:
        die(f"expected output with exactly 114 tensors, got {len(out)}")
    if set(out) != base_keys:
        die("output tensor name set does not match input tensor name set")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "model.safetensors"
    # Ensure all tensors are contiguous CPU tensors before saving.
    out = {k: v.contiguous() for k, v in out.items()}
    save_file(out, str(out_path))

    print(f"OK: wrote {len(out)} tensors ({merged_count} merged) to {out_path}")


if __name__ == "__main__":
    main()
