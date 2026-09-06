"""
T4: Task-vector merge of two Pythia-1B fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the 64 MLP
tensors, computed in float32 and cast back to float16. Everything else is copied
unchanged from base. Fails loudly if the preconditions (identical key sets, identical
non-MLP tensors across all three checkpoints) do not hold.
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent.parent
BASE_PATH = SANDBOX / "inputs" / "base" / "model.safetensors"
FT1_PATH = SANDBOX / "inputs" / "ft1" / "model.safetensors"
FT2_PATH = SANDBOX / "inputs" / "ft2" / "model.safetensors"
OUT_PATH = SANDBOX / "out" / "T4" / "model.safetensors"

LAMBDA = 0.4
NUM_LAYERS = 16
EXPECTED_TOTAL_TENSORS = 244
EXPECTED_MLP_TENSORS = 64


def mlp_tensor_names() -> set[str]:
    names = set()
    for i in range(NUM_LAYERS):
        prefix = f"gpt_neox.layers.{i}.mlp"
        names.add(f"{prefix}.dense_h_to_4h.weight")
        names.add(f"{prefix}.dense_h_to_4h.bias")
        names.add(f"{prefix}.dense_4h_to_h.weight")
        names.add(f"{prefix}.dense_4h_to_h.bias")
    return names


def load_all(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base = load_all(BASE_PATH)
    ft1 = load_all(FT1_PATH)
    ft2 = load_all(FT2_PATH)

    base_keys = set(base.keys())
    ft1_keys = set(ft1.keys())
    ft2_keys = set(ft2.keys())

    if not (base_keys == ft1_keys == ft2_keys):
        missing_ft1 = base_keys - ft1_keys
        extra_ft1 = ft1_keys - base_keys
        missing_ft2 = base_keys - ft2_keys
        extra_ft2 = ft2_keys - base_keys
        fail(
            "tensor name mismatch across checkpoints: "
            f"ft1 missing={sorted(missing_ft1)} extra={sorted(extra_ft1)}; "
            f"ft2 missing={sorted(missing_ft2)} extra={sorted(extra_ft2)}"
        )

    if len(base_keys) != EXPECTED_TOTAL_TENSORS:
        fail(f"expected {EXPECTED_TOTAL_TENSORS} tensors, base has {len(base_keys)}")

    mlp_names = mlp_tensor_names()
    if not mlp_names.issubset(base_keys):
        missing = mlp_names - base_keys
        fail(f"expected MLP tensor names not found in checkpoint: {sorted(missing)}")
    if len(mlp_names) != EXPECTED_MLP_TENSORS:
        fail(f"expected {EXPECTED_MLP_TENSORS} MLP tensor names, computed {len(mlp_names)}")

    non_mlp_names = base_keys - mlp_names

    # Step 1: every tensor outside the 64 MLP tensors must be identical (shape,
    # dtype, and bit-exact values) across all three checkpoints.
    mismatches = []
    for name in non_mlp_names:
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatches.append(f"{name}: shape mismatch base={tuple(b.shape)} "
                               f"ft1={tuple(f1.shape)} ft2={tuple(f2.shape)}")
            continue
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            mismatches.append(f"{name}: dtype mismatch base={b.dtype} "
                               f"ft1={f1.dtype} ft2={f2.dtype}")
            continue
        if not torch.equal(b, f1):
            mismatches.append(f"{name}: differs between base and ft1 but is not an MLP tensor")
            continue
        if not torch.equal(b, f2):
            mismatches.append(f"{name}: differs between base and ft2 but is not an MLP tensor")
            continue
    if mismatches:
        fail(
            "non-MLP tensors are not identical across checkpoints (frozen-backbone "
            "assumption violated):\n  " + "\n  ".join(mismatches)
        )

    # Also verify shapes/dtypes match for the MLP tensors themselves before merging.
    for name in mlp_names:
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            fail(f"{name}: MLP tensor shape mismatch base={tuple(b.shape)} "
                 f"ft1={tuple(f1.shape)} ft2={tuple(f2.shape)}")
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            fail(f"{name}: MLP tensor dtype mismatch base={b.dtype} "
                 f"ft1={f1.dtype} ft2={f2.dtype}")

    # Step 2: merge the MLP tensors via task arithmetic, each task vector taken
    # against the unmodified base (not against an already-merged tensor).
    output: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in mlp_names:
        base_dtype = base[name].dtype
        b32 = base[name].to(torch.float32)
        f1_32 = ft1[name].to(torch.float32)
        f2_32 = ft2[name].to(torch.float32)
        merged32 = b32 + LAMBDA * (f1_32 - b32) + LAMBDA * (f2_32 - b32)
        output[name] = merged32.to(base_dtype)
        merged_count += 1

    if merged_count != EXPECTED_MLP_TENSORS:
        fail(f"expected to merge {EXPECTED_MLP_TENSORS} tensors, merged {merged_count}")

    # Step 3: everything else is copied unchanged from base.
    for name in non_mlp_names:
        output[name] = base[name].clone()

    if len(output) != EXPECTED_TOTAL_TENSORS:
        fail(f"output has {len(output)} tensors, expected {EXPECTED_TOTAL_TENSORS}")

    # Make sure everything is contiguous for safetensors.
    output = {k: v.contiguous() for k, v in output.items()}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, str(OUT_PATH))

    print(f"OK: wrote {len(output)} tensors ({merged_count} merged, "
          f"{len(non_mlp_names)} unchanged) to {OUT_PATH}")


if __name__ == "__main__":
    main()
