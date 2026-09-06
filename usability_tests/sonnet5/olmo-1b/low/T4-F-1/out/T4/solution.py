"""T4: task-vector merge of two OLMo-1B fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the
48 MLP tensors; everything else copied unchanged from base. Fails loudly if
the shared-tensor precondition, the merged-tensor count, or the output tensor
count does not hold.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root (out/T4 -> out -> root)
INPUTS = ROOT / "inputs"
LAMBDA = 0.4
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def load_sharded(dirpath: Path) -> dict[str, torch.Tensor]:
    index_path = dirpath / "model.safetensors.index.json"
    tensors: dict[str, torch.Tensor] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_files = sorted(set(index["weight_map"].values()))
        for shard in shard_files:
            with safe_open(dirpath / shard, framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
    else:
        (single,) = list(dirpath.glob("*.safetensors"))
        with safe_open(single, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    base = load_sharded(INPUTS / "base")
    ft1 = load_sharded(INPUTS / "ft1")
    ft2 = load_sharded(INPUTS / "ft2")

    # --- Step 1: verify same key sets ---
    keys_base, keys_ft1, keys_ft2 = set(base), set(ft1), set(ft2)
    if not (keys_base == keys_ft1 == keys_ft2):
        missing_1 = keys_base - keys_ft1
        extra_1 = keys_ft1 - keys_base
        missing_2 = keys_base - keys_ft2
        extra_2 = keys_ft2 - keys_base
        sys.exit(
            "ABORT: tensor name sets differ across checkpoints.\n"
            f"  base vs ft1: missing={sorted(missing_1)} extra={sorted(extra_1)}\n"
            f"  base vs ft2: missing={sorted(missing_2)} extra={sorted(extra_2)}"
        )

    mlp_keys = {k for k in keys_base if MLP_RE.match(k)}
    non_mlp_keys = keys_base - mlp_keys

    # Verify every non-MLP tensor is identical (bit-exact) across all three.
    mismatches = []
    for k in sorted(non_mlp_keys):
        b, f1, f2 = base[k], ft1[k], ft2[k]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatches.append(f"{k}: shape mismatch base={b.shape} ft1={f1.shape} ft2={f2.shape}")
            continue
        if not torch.equal(b, f1):
            mismatches.append(f"{k}: differs between base and ft1")
        if not torch.equal(b, f2):
            mismatches.append(f"{k}: differs between base and ft2")
    if mismatches:
        sys.exit("ABORT: non-MLP tensors are not identical across checkpoints:\n" + "\n".join(mismatches))

    # Sanity-check the expected MLP tensor set: 16 layers * 3 tensors = 48.
    expected_mlp = {
        f"model.layers.{i}.mlp.{proj}.weight"
        for i in range(16)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    if mlp_keys != expected_mlp:
        sys.exit(
            "ABORT: MLP tensor set does not match expectation.\n"
            f"  missing={sorted(expected_mlp - mlp_keys)} extra={sorted(mlp_keys - expected_mlp)}"
        )
    if len(mlp_keys) != 48:
        sys.exit(f"ABORT: expected 48 MLP tensors, found {len(mlp_keys)}")

    # --- Step 2/3: merge ---
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for k in keys_base:
        if k in mlp_keys:
            b = base[k].to(torch.float32)
            f1 = ft1[k].to(torch.float32)
            f2 = ft2[k].to(torch.float32)
            merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
            out[k] = merged.to(base[k].dtype)
            merged_count += 1
        else:
            out[k] = base[k].clone()

    if merged_count != 48:
        sys.exit(f"ABORT: merged {merged_count} tensors, expected exactly 48")
    if len(out) != 114:
        sys.exit(f"ABORT: output has {len(out)} tensors, expected exactly 114")

    out_dir = HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "model.safetensors"))
    print(f"OK: wrote {len(out)} tensors ({merged_count} merged) to {out_dir / 'model.safetensors'}")


if __name__ == "__main__":
    main()
