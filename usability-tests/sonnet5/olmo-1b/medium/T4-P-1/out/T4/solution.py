"""
Task-vector merge of two fine-tunes of OLMo-1B-0724-hf.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the
48 MLP tensors; every other tensor is copied unchanged from base.
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

MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def load_sharded(dirpath: Path) -> dict[str, torch.Tensor]:
    index_path = dirpath / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index["weight_map"].values()))
        tensors: dict[str, torch.Tensor] = {}
        for shard_name in shard_names:
            tensors.update(load_file(dirpath / shard_name))
        return tensors
    return load_file(dirpath / "model.safetensors")


def is_mlp_tensor(name: str) -> bool:
    m = MLP_RE.match(name)
    return bool(m) and 0 <= int(m.group(1)) <= 15


def main() -> None:
    base = load_sharded(INPUTS / "base")
    ft1 = load_sharded(INPUTS / "ft1")
    ft2 = load_sharded(INPUTS / "ft2")

    # 1. Same tensor names across all three checkpoints.
    names_base, names_ft1, names_ft2 = set(base), set(ft1), set(ft2)
    if not (names_base == names_ft1 == names_ft2):
        missing_ft1 = names_base - names_ft1
        missing_ft2 = names_base - names_ft2
        extra_ft1 = names_ft1 - names_base
        extra_ft2 = names_ft2 - names_base
        raise AssertionError(
            "Tensor name sets differ across checkpoints: "
            f"missing_in_ft1={missing_ft1}, extra_in_ft1={extra_ft1}, "
            f"missing_in_ft2={missing_ft2}, extra_in_ft2={extra_ft2}"
        )

    mlp_names = {name for name in base if is_mlp_tensor(name)}
    expected_mlp_names = {
        f"model.layers.{i}.mlp.{proj}.weight"
        for i in range(16)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    if mlp_names != expected_mlp_names:
        raise AssertionError(
            f"MLP tensor name detection mismatch: found {len(mlp_names)}, "
            f"expected {len(expected_mlp_names)}; "
            f"diff={mlp_names.symmetric_difference(expected_mlp_names)}"
        )
    if len(mlp_names) != 48:
        raise AssertionError(f"expected 48 MLP tensors, found {len(mlp_names)}")

    non_mlp_names = names_base - mlp_names

    # Every non-MLP tensor must be bit-identical across all three checkpoints
    # (same shape, dtype and values) -- this is the frozen-backbone precondition.
    for name in non_mlp_names:
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            raise AssertionError(f"shape mismatch outside MLP tensors for {name!r}")
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            raise AssertionError(f"dtype mismatch outside MLP tensors for {name!r}")
        if not torch.equal(b, f1):
            raise AssertionError(f"non-MLP tensor {name!r} differs between base and ft1")
        if not torch.equal(b, f2):
            raise AssertionError(f"non-MLP tensor {name!r} differs between base and ft2")

    # 2/3. Build the output: base unchanged outside MLP, merged inside MLP.
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in base:
        if name in mlp_names:
            b = base[name].to(torch.float32)
            f1 = ft1[name].to(torch.float32)
            f2 = ft2[name].to(torch.float32)
            merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
            out[name] = merged.to(base[name].dtype).contiguous()
            merged_count += 1
        else:
            out[name] = base[name].clone().contiguous()

    if merged_count != 48:
        raise AssertionError(f"expected to merge exactly 48 tensors, merged {merged_count}")
    if len(out) != 114:
        raise AssertionError(f"expected 114 output tensors, got {len(out)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, OUT_DIR / "model.safetensors")
    print(f"Wrote {len(out)} tensors ({merged_count} merged) to {OUT_DIR / 'model.safetensors'}")


if __name__ == "__main__":
    main()
