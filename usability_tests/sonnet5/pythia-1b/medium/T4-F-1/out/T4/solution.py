"""T4: task-vector merge of two Pythia-1B fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the 64 MLP tensors
out[X] = base[X]                                                      for everything else

Verifies before touching anything that base/ft1/ft2 share the same tensor
names and that every non-MLP tensor is bit-identical across all three.
"""

import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
NUM_LAYERS = 16

BASE = Path("inputs/base/model.safetensors")
FT1 = Path("inputs/ft1/model.safetensors")
FT2 = Path("inputs/ft2/model.safetensors")
OUT = Path("out/T4/model.safetensors")

MLP_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_(h_to_4h|4h_to_h)\.(weight|bias)$")


def load_all(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def is_mlp_tensor(name: str) -> bool:
    m = MLP_RE.match(name)
    return m is not None and int(m.group(1)) < NUM_LAYERS


def main() -> None:
    base = load_all(BASE)
    ft1 = load_all(FT1)
    ft2 = load_all(FT2)

    # --- Step 1: verify shared structure before touching anything ---
    base_keys, ft1_keys, ft2_keys = set(base), set(ft1), set(ft2)
    if not (base_keys == ft1_keys == ft2_keys):
        missing_ft1 = base_keys - ft1_keys
        missing_ft2 = base_keys - ft2_keys
        extra_ft1 = ft1_keys - base_keys
        extra_ft2 = ft2_keys - base_keys
        sys.exit(
            "ABORT: tensor name sets differ between checkpoints.\n"
            f"  missing from ft1: {sorted(missing_ft1)}\n"
            f"  missing from ft2: {sorted(missing_ft2)}\n"
            f"  extra in ft1: {sorted(extra_ft1)}\n"
            f"  extra in ft2: {sorted(extra_ft2)}"
        )

    mlp_names = {name for name in base_keys if is_mlp_tensor(name)}
    non_mlp_names = base_keys - mlp_names

    mismatches = []
    for name in sorted(non_mlp_names):
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatches.append(f"{name}: shape mismatch base={tuple(b.shape)} ft1={tuple(f1.shape)} ft2={tuple(f2.shape)}")
            continue
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            mismatches.append(f"{name}: dtype mismatch base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")
            continue
        if not torch.equal(b, f1):
            mismatches.append(f"{name}: differs between base and ft1 outside the declared MLP tensor set")
        if not torch.equal(b, f2):
            mismatches.append(f"{name}: differs between base and ft2 outside the declared MLP tensor set")

    if mismatches:
        sys.exit("ABORT: non-MLP tensors are not identical across checkpoints:\n" + "\n".join(mismatches))

    if len(mlp_names) != 64:
        sys.exit(f"ABORT: expected exactly 64 MLP tensors, found {len(mlp_names)}: {sorted(mlp_names)}")

    # --- Step 2 & 3: compute the merge ---
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in base_keys:
        b = base[name]
        if name in mlp_names:
            f1, f2 = ft1[name], ft2[name]
            b32, f1_32, f2_32 = b.float(), f1.float(), f2.float()
            merged32 = b32 + LAMBDA * (f1_32 - b32) + LAMBDA * (f2_32 - b32)
            out[name] = merged32.to(b.dtype)
            merged_count += 1
        else:
            out[name] = b.clone()

    if merged_count != 64:
        sys.exit(f"ABORT: merged {merged_count} tensors, expected exactly 64")

    if len(out) != 244:
        sys.exit(f"ABORT: output has {len(out)} tensors, expected exactly 244")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT))
    print(f"Wrote {OUT} with {len(out)} tensors ({merged_count} merged, {len(out) - merged_count} unchanged).")


if __name__ == "__main__":
    main()
