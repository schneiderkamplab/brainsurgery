"""
T4: Task-vector merge of two fine-tunes (OLMo-1B-0724-hf).

out[X] = base[X] + 0.4*(ft1[X]-base[X]) + 0.4*(ft2[X]-base[X])  for the 48 MLP tensors
out[X] = base[X]                                                for everything else
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent.parent / "inputs"
OUT_DIR = HERE
LAMBDA = 0.4
N_LAYERS = 16

MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def load_base(base_dir: Path) -> dict[str, torch.Tensor]:
    index_path = base_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    shard_names = sorted(set(index["weight_map"].values()))
    tensors: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard = load_file(base_dir / shard_name)
        tensors.update(shard)
    return tensors


def expected_mlp_names() -> set[str]:
    names = set()
    for i in range(N_LAYERS):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            names.add(f"model.layers.{i}.mlp.{proj}.weight")
    return names


def main() -> None:
    base = load_base(INPUTS / "base")
    ft1 = load_file(INPUTS / "ft1" / "model.safetensors")
    ft2 = load_file(INPUTS / "ft2" / "model.safetensors")

    # Step 1: verify same tensor names across all three checkpoints.
    names_base, names_ft1, names_ft2 = set(base), set(ft1), set(ft2)
    if not (names_base == names_ft1 == names_ft2):
        missing_ft1 = names_base - names_ft1
        extra_ft1 = names_ft1 - names_base
        missing_ft2 = names_base - names_ft2
        extra_ft2 = names_ft2 - names_base
        raise RuntimeError(
            "Tensor name mismatch between checkpoints: "
            f"ft1 missing={missing_ft1} extra={extra_ft1}; "
            f"ft2 missing={missing_ft2} extra={extra_ft2}"
        )

    expected_mlp = expected_mlp_names()
    actual_mlp = {n for n in names_base if MLP_RE.match(n)}
    if actual_mlp != expected_mlp:
        raise RuntimeError(
            f"MLP tensor set mismatch. Missing: {expected_mlp - actual_mlp}, "
            f"unexpected: {actual_mlp - expected_mlp}"
        )
    if len(actual_mlp) != 48:
        raise RuntimeError(f"Expected exactly 48 MLP tensors, found {len(actual_mlp)}")

    non_mlp_names = names_base - actual_mlp

    # Verify every non-MLP tensor is identical (dtype, shape, values) across all three.
    for name in sorted(non_mlp_names):
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            raise RuntimeError(f"Shape mismatch outside MLP tensors for {name!r}")
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            raise RuntimeError(f"Dtype mismatch outside MLP tensors for {name!r}")
        if not torch.equal(b, f1):
            raise RuntimeError(f"Non-MLP tensor {name!r} differs between base and ft1")
        if not torch.equal(b, f2):
            raise RuntimeError(f"Non-MLP tensor {name!r} differs between base and ft2")

    # Step 2 & 3: build output.
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name, b in base.items():
        if name in actual_mlp:
            f1, f2 = ft1[name], ft2[name]
            b32, f1_32, f2_32 = b.to(torch.float32), f1.to(torch.float32), f2.to(torch.float32)
            merged32 = b32 + LAMBDA * (f1_32 - b32) + LAMBDA * (f2_32 - b32)
            out[name] = merged32.to(b.dtype)
            merged_count += 1
        else:
            out[name] = b.clone()

    if merged_count != 48:
        raise RuntimeError(f"Expected to merge exactly 48 tensors, merged {merged_count}")
    if len(out) != 114:
        raise RuntimeError(f"Expected exactly 114 output tensors, got {len(out)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, OUT_DIR / "model.safetensors")
    print(f"Wrote {len(out)} tensors ({merged_count} merged) to {OUT_DIR / 'model.safetensors'}")


if __name__ == "__main__":
    main()
