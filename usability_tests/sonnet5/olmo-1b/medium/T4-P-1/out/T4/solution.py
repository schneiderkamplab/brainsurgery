"""
T4: Task-vector merge of two fine-tunes (OLMo-1B-0724-hf).

Standalone script: loads the base checkpoint (sharded safetensors) and two
fine-tune checkpoints (single-file safetensors each), verifies that the only
tensors differing from the base are the 48 MLP tensors, computes the task-
vector merge for those, and writes a single-file safetensors checkpoint with
all 114 tensors.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent.parent / "inputs"
BASE_DIR = INPUTS / "base"
FT1_PATH = INPUTS / "ft1" / "model.safetensors"
FT2_PATH = INPUTS / "ft2" / "model.safetensors"
OUT_PATH = HERE / "model.safetensors"

LAMBDA = 0.4
MLP_RE = re.compile(r"^model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def load_sharded(base_dir: Path) -> dict[str, torch.Tensor]:
    index_path = base_dir / "model.safetensors.index.json"
    if not index_path.exists():
        die(f"missing index file: {index_path}")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    shards = {name: load_file(base_dir / name) for name in shard_names}
    tensors: dict[str, torch.Tensor] = {}
    for name, shard in weight_map.items():
        if name not in shards[shard]:
            die(f"tensor {name!r} declared in index but missing from shard {shard}")
        tensors[name] = shards[shard][name]
    return tensors


def main() -> None:
    base = load_sharded(BASE_DIR)
    if not FT1_PATH.exists():
        die(f"missing file: {FT1_PATH}")
    if not FT2_PATH.exists():
        die(f"missing file: {FT2_PATH}")
    ft1 = load_file(FT1_PATH)
    ft2 = load_file(FT2_PATH)

    # --- Step 1: verify same tensor names across all three checkpoints ---
    base_keys = set(base.keys())
    ft1_keys = set(ft1.keys())
    ft2_keys = set(ft2.keys())
    if base_keys != ft1_keys:
        die(
            "tensor name mismatch between base and ft1: "
            f"base-only={sorted(base_keys - ft1_keys)} ft1-only={sorted(ft1_keys - base_keys)}"
        )
    if base_keys != ft2_keys:
        die(
            "tensor name mismatch between base and ft2: "
            f"base-only={sorted(base_keys - ft2_keys)} ft2-only={sorted(ft2_keys - base_keys)}"
        )

    if len(base_keys) != 114:
        die(f"expected 114 tensors in base, found {len(base_keys)}")

    mlp_keys = {name for name in base_keys if MLP_RE.match(name)}
    if len(mlp_keys) != 48:
        die(f"expected 48 MLP tensors matching the naming pattern, found {len(mlp_keys)}")

    non_mlp_keys = base_keys - mlp_keys

    # Verify every tensor outside the 48 MLP tensors is identical across all three.
    mismatched = []
    for name in sorted(non_mlp_keys):
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.dtype != f1.dtype or not torch.equal(b, f1):
            mismatched.append((name, "ft1"))
        if b.shape != f2.shape or b.dtype != f2.dtype or not torch.equal(b, f2):
            mismatched.append((name, "ft2"))
    if mismatched:
        die(
            "non-MLP tensors differ from base (expected identical across all "
            f"three checkpoints): {mismatched[:10]}"
            + (" ..." if len(mismatched) > 10 else "")
        )

    # Also verify MLP tensors match shape/dtype across checkpoints (sanity, not
    # required to be equal in value).
    for name in mlp_keys:
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if not (b.shape == f1.shape == f2.shape):
            die(f"shape mismatch for MLP tensor {name!r}: base={b.shape} ft1={f1.shape} ft2={f2.shape}")
        if not (b.dtype == f1.dtype == f2.dtype):
            die(f"dtype mismatch for MLP tensor {name!r}: base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")
        if b.dtype != torch.float32:
            die(f"expected float32 tensors, got {b.dtype} for {name!r}")

    # --- Step 2 & 3: compute the merge ---
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in base_keys:
        if name in mlp_keys:
            b = base[name].to(torch.float32)
            f1 = ft1[name].to(torch.float32)
            f2 = ft2[name].to(torch.float32)
            out[name] = (b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)).contiguous()
            merged_count += 1
        else:
            out[name] = base[name].contiguous()

    if merged_count != 48:
        die(f"expected to merge exactly 48 tensors, merged {merged_count}")

    # --- Step 4: write output ---
    if len(out) != 114:
        die(f"expected exactly 114 tensors in output, got {len(out)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))

    print(f"OK: wrote {len(out)} tensors ({merged_count} merged) to {OUT_PATH}")


if __name__ == "__main__":
    main()
