"""T4: task-vector merge of two Pythia-1B fine-tunes onto the base.

Plain torch + safetensors: the three checkpoints fit in memory and every
required check is an explicit assertion here.
"""
import re
import sys

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
MLP_RE = re.compile(r"^gpt_neox\.layers\.\d+\.mlp\.dense_(h_to_4h|4h_to_h)\.(weight|bias)$")


def fail(msg):
    raise SystemExit(f"ERROR: {msg}")


def main():
    base = load_file("inputs/base/model.safetensors")
    ft1 = load_file("inputs/ft1/model.safetensors")
    ft2 = load_file("inputs/ft2/model.safetensors")

    # Step 1: same names everywhere.
    if not (set(base) == set(ft1) == set(ft2)):
        fail(
            "tensor name sets differ: "
            f"ft1 only={sorted(set(ft1) ^ set(base))[:5]} ft2 only={sorted(set(ft2) ^ set(base))[:5]}"
        )

    mlp_keys = sorted(k for k in base if MLP_RE.match(k))
    if len(mlp_keys) != 64:
        fail(f"expected 64 MLP tensors, matched {len(mlp_keys)}")

    # Step 1 cont.: everything outside the MLP tensors must be bit-identical.
    mlp_set = set(mlp_keys)
    for k in sorted(base):
        if k in mlp_set:
            continue
        for name, sd in (("ft1", ft1), ("ft2", ft2)):
            if base[k].shape != sd[k].shape or base[k].dtype != sd[k].dtype:
                fail(f"{k}: shape/dtype differs between base and {name}")
            if not torch.equal(base[k], sd[k]):
                fail(f"non-MLP tensor {k} differs between base and {name}")

    # Step 2: every task vector is taken against the *unmodified* base.
    out = dict(base)
    merged = 0
    for k in mlp_keys:
        b = base[k]
        for name, sd in (("ft1", ft1), ("ft2", ft2)):
            if sd[k].shape != b.shape or sd[k].dtype != b.dtype:
                fail(f"{k}: shape/dtype differs between base and {name}")
        b32 = b.to(torch.float32)
        merged32 = b32 + LAMBDA * (ft1[k].to(torch.float32) - b32) + LAMBDA * (
            ft2[k].to(torch.float32) - b32
        )
        out[k] = merged32.to(b.dtype)
        merged += 1

    if merged != 64:
        fail(f"merged {merged} tensors, expected exactly 64")
    if len(out) != 244:
        fail(f"output has {len(out)} tensors, expected exactly 244")

    out = {k: v.contiguous() for k, v in out.items()}
    save_file(out, "out/T4/model.safetensors", metadata={"format": "pt"})

    check = load_file("out/T4/model.safetensors")
    if len(check) != 244:
        fail(f"written file has {len(check)} tensors, expected 244")
    for k in check:
        if k in mlp_set:
            continue
        if not torch.equal(check[k], base[k]):
            fail(f"non-MLP tensor {k} was modified in the output")
    print(f"OK: 244 tensors written, {merged} merged, lambda={LAMBDA}")


if __name__ == "__main__":
    sys.exit(main())
