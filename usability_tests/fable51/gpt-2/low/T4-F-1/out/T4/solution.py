"""T4: task-vector merge of two GPT-2 fine-tunes (lambda = 0.4)."""
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
LAMBDA = 0.4
MLP_RE = re.compile(r"^h\.(\d+)\.mlp\.(c_fc|c_proj)\.(weight|bias)$")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base = load_file(ROOT / "inputs/base/model.safetensors")
    ft1 = load_file(ROOT / "inputs/ft1/model.safetensors")
    ft2 = load_file(ROOT / "inputs/ft2/model.safetensors")

    # Step 1: same names, and everything outside the MLP tensors identical.
    if not (base.keys() == ft1.keys() == ft2.keys()):
        fail("tensor name sets differ between base, ft1 and ft2")
    mlp_keys = {k for k in base if MLP_RE.match(k)}
    if len(mlp_keys) != 48:
        fail(f"expected 48 MLP tensors, found {len(mlp_keys)}")
    for k in base:
        for name, ft in (("ft1", ft1), ("ft2", ft2)):
            if ft[k].shape != base[k].shape or ft[k].dtype != base[k].dtype:
                fail(f"{k}: shape/dtype mismatch in {name}")
            if k not in mlp_keys and not torch.equal(base[k], ft[k]):
                fail(f"{k}: non-MLP tensor differs between base and {name}")

    # Step 2/3: merge each task vector against the unmodified base.
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for k, b in base.items():
        if k in mlp_keys:
            b32 = b.float()
            out[k] = (b32 + LAMBDA * (ft1[k].float() - b32) + LAMBDA * (ft2[k].float() - b32)).to(b.dtype)
            merged += 1
        else:
            out[k] = b
    if merged != 48:
        fail(f"merged {merged} tensors, expected 48")
    if len(out) != 160:
        fail(f"output has {len(out)} tensors, expected 160")

    dest = ROOT / "out/T4/model.safetensors"
    save_file({k: v.contiguous() for k, v in out.items()}, dest)
    check = load_file(dest)
    if len(check) != 160 or check.keys() != base.keys():
        fail("written file does not have the expected 160 tensor names")
    print(f"OK: merged {merged} tensors, wrote {len(check)} tensors to {dest}")


if __name__ == "__main__":
    main()
