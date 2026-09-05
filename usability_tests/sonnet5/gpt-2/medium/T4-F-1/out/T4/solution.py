"""
T4: Task-vector merge of two GPT-2 fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the 48 MLP tensors
out[X] = base[X]                                                     for everything else

Uses only `safetensors` + `torch` (both on the F-allowed list). A plain script
was chosen over mergekit's task-arithmetic merge method because this task
needs a bespoke precondition check (identical non-MLP tensors across all
three checkpoints, bit-exact) before any arithmetic happens, and mergekit's
task_arithmetic merge method does not expose that as a built-in gate - it
would have to be done separately anyway, so a single script doing both the
check and the arithmetic is the smaller, more auditable path.
"""

import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent.parent / "inputs"
OUT_DIR = HERE
OUT_FILE = OUT_DIR / "model.safetensors"

MLP_RE = re.compile(r"^h\.\d+\.mlp\.(c_fc|c_proj)\.(weight|bias)$")


def is_mlp_tensor(name: str) -> bool:
    return bool(MLP_RE.match(name))


def fail(msg: str) -> None:
    print(f"ABORT: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base = load_file(INPUTS / "base" / "model.safetensors")
    ft1 = load_file(INPUTS / "ft1" / "model.safetensors")
    ft2 = load_file(INPUTS / "ft2" / "model.safetensors")

    # --- Step 1: verify shared structure and non-MLP identity ---
    base_keys, ft1_keys, ft2_keys = set(base), set(ft1), set(ft2)
    if not (base_keys == ft1_keys == ft2_keys):
        missing_in_ft1 = base_keys - ft1_keys
        missing_in_ft2 = base_keys - ft2_keys
        extra_in_ft1 = ft1_keys - base_keys
        extra_in_ft2 = ft2_keys - base_keys
        fail(
            "tensor name mismatch across checkpoints: "
            f"missing_in_ft1={sorted(missing_in_ft1)} missing_in_ft2={sorted(missing_in_ft2)} "
            f"extra_in_ft1={sorted(extra_in_ft1)} extra_in_ft2={sorted(extra_in_ft2)}"
        )

    mlp_keys = {k for k in base_keys if is_mlp_tensor(k)}
    non_mlp_keys = base_keys - mlp_keys

    if len(mlp_keys) != 48:
        fail(f"expected exactly 48 MLP tensors, found {len(mlp_keys)}: {sorted(mlp_keys)}")

    for k in non_mlp_keys:
        b, f1, f2 = base[k], ft1[k], ft2[k]
        if b.shape != f1.shape or b.shape != f2.shape:
            fail(f"shape mismatch outside MLP tensors for {k!r}: "
                 f"base={tuple(b.shape)} ft1={tuple(f1.shape)} ft2={tuple(f2.shape)}")
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            fail(f"dtype mismatch outside MLP tensors for {k!r}: "
                 f"base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")
        if not torch.equal(b, f1):
            fail(f"non-MLP tensor {k!r} differs between base and ft1; "
                 "frozen-backbone assumption violated")
        if not torch.equal(b, f2):
            fail(f"non-MLP tensor {k!r} differs between base and ft2; "
                 "frozen-backbone assumption violated")

    # --- Step 2/3: compute merge, each task vector taken against the unmodified base ---
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for k in base_keys:
        b = base[k]
        if k in mlp_keys:
            b32 = b.to(torch.float32)
            f1_32 = ft1[k].to(torch.float32)
            f2_32 = ft2[k].to(torch.float32)
            merged = b32 + LAMBDA * (f1_32 - b32) + LAMBDA * (f2_32 - b32)
            out[k] = merged.to(b.dtype).contiguous()
            merged_count += 1
        else:
            out[k] = b.contiguous()

    if merged_count != 48:
        fail(f"internal error: merged {merged_count} tensors, expected 48")

    # --- Step 4: write output, exactly 160 tensors ---
    if len(out) != 160:
        fail(f"output has {len(out)} tensors, expected 160")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_FILE))

    # Re-load and sanity check the file we just wrote.
    check = load_file(OUT_FILE)
    if len(check) != 160:
        fail(f"written file has {len(check)} tensors, expected 160")
    if sum(1 for k in check if is_mlp_tensor(k)) != 48:
        fail("written file does not have exactly 48 MLP tensors")

    print(f"OK: wrote {OUT_FILE} with {len(out)} tensors "
          f"({merged_count} merged, {len(out) - merged_count} copied unchanged)")


if __name__ == "__main__":
    main()
