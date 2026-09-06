"""T4: task-vector merge of two GPT-2 fine-tunes.

Plain script on top of `safetensors` and `torch` (both in F-allowed.md).
Not using mergekit's task-arithmetic YAML here because that requires
HF-format model directories with configs on both sides of the merge and
gives less direct control over the exact three-way equality check this task
needs before touching anything; a short script expresses the required
checks and the arithmetic more transparently and auditably.

    out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])

for the 48 MLP tensors, base[X] unchanged everywhere else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
NUM_LAYERS = 12
MLP_SUFFIXES = ["mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias"]


def mlp_tensor_names() -> set[str]:
    names = set()
    for i in range(NUM_LAYERS):
        for suffix in MLP_SUFFIXES:
            names.add(f"h.{i}.{suffix}")
    return names


def load_all(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    base_path = root / "inputs" / "base" / "model.safetensors"
    ft1_path = root / "inputs" / "ft1" / "model.safetensors"
    ft2_path = root / "inputs" / "ft2" / "model.safetensors"
    out_path = root / "out" / "T4" / "model.safetensors"

    base = load_all(base_path)
    ft1 = load_all(ft1_path)
    ft2 = load_all(ft2_path)

    # Step 1: same tensor names across all three checkpoints.
    base_keys, ft1_keys, ft2_keys = set(base), set(ft1), set(ft2)
    if not (base_keys == ft1_keys == ft2_keys):
        only_base = base_keys - ft1_keys - ft2_keys
        only_ft1 = ft1_keys - base_keys
        only_ft2 = ft2_keys - base_keys
        fail(
            "tensor name sets differ across checkpoints: "
            f"only_in_base={sorted(only_base)[:5]} "
            f"only_in_ft1={sorted(only_ft1)[:5]} only_in_ft2={sorted(only_ft2)[:5]}"
        )

    expected_mlp = mlp_tensor_names()
    missing_mlp = expected_mlp - base_keys
    if missing_mlp:
        fail(f"expected MLP tensors missing from checkpoints: {sorted(missing_mlp)}")

    non_mlp_keys = base_keys - expected_mlp

    # Step 1 (cont'd): every non-MLP tensor identical (shape, dtype, values) in all three.
    mismatches = []
    for key in non_mlp_keys:
        b, f1, f2 = base[key], ft1[key], ft2[key]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatches.append(f"{key}: shape mismatch base={tuple(b.shape)} "
                               f"ft1={tuple(f1.shape)} ft2={tuple(f2.shape)}")
            continue
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            mismatches.append(f"{key}: dtype mismatch base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")
            continue
        if not torch.equal(b, f1) or not torch.equal(b, f2):
            mismatches.append(f"{key}: values differ outside the declared MLP tensor set")
    if mismatches:
        fail(
            "non-MLP tensors are not identical across base/ft1/ft2 "
            f"({len(mismatches)} mismatch(es)), e.g.: {mismatches[:5]}"
        )

    # Step 2: task-arithmetic merge of the 48 MLP tensors, against the unmodified base.
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for key in expected_mlp:
        b = base[key].to(torch.float32)
        f1 = ft1[key].to(torch.float32)
        f2 = ft2[key].to(torch.float32)
        merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
        out[key] = merged.to(base[key].dtype).contiguous()
        merged_count += 1

    if merged_count != 48:
        fail(f"expected to merge exactly 48 MLP tensors, merged {merged_count}")

    # Step 3: everything else comes from base, unchanged.
    for key in non_mlp_keys:
        out[key] = base[key].contiguous()

    if len(out) != 160:
        fail(f"expected exactly 160 output tensors, got {len(out)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_path))
    print(f"wrote {out_path} with {len(out)} tensors ({merged_count} merged, "
          f"{len(non_mlp_keys)} unchanged)")


if __name__ == "__main__":
    main()
