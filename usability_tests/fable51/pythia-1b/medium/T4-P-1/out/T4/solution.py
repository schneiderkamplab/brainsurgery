"""T4: task-vector merge of two Pythia-1B fine-tunes into the base (lambda = 0.4)."""

import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
FT1 = os.path.join(ROOT, "inputs", "ft1", "model.safetensors")
FT2 = os.path.join(ROOT, "inputs", "ft2", "model.safetensors")
OUT = os.path.join(HERE, "model.safetensors")

LAMBDA = 0.4
N_LAYERS = 16
MLP_RE = re.compile(
    r"^gpt_neox\.layers\.(\d+)\.mlp\.(dense_h_to_4h|dense_4h_to_h)\.(weight|bias)$"
)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base = load_file(BASE)
    ft1 = load_file(FT1)
    ft2 = load_file(FT2)

    # Step 1: verify names, then every non-MLP tensor is identical across all three.
    names = set(base)
    if set(ft1) != names or set(ft2) != names:
        fail(
            "tensor name sets differ: "
            f"base^ft1={sorted(names ^ set(ft1))[:5]} base^ft2={sorted(names ^ set(ft2))[:5]}"
        )
    if len(names) != 244:
        fail(f"expected 244 tensors in base, found {len(names)}")

    mlp_names = {n for n in names if MLP_RE.match(n)}
    expected_mlp = {
        f"gpt_neox.layers.{i}.mlp.{proj}.{kind}"
        for i in range(N_LAYERS)
        for proj in ("dense_h_to_4h", "dense_4h_to_h")
        for kind in ("weight", "bias")
    }
    if mlp_names != expected_mlp:
        fail(f"MLP tensor set mismatch: {sorted(mlp_names ^ expected_mlp)[:5]}")

    for n in sorted(names):
        for tag, other in (("ft1", ft1), ("ft2", ft2)):
            b, o = base[n], other[n]
            if b.shape != o.shape or b.dtype != o.dtype:
                fail(f"{n}: {tag} shape/dtype {tuple(o.shape)}/{o.dtype} != base {tuple(b.shape)}/{b.dtype}")
            if n not in mlp_names and not torch.equal(b, o):
                fail(f"shared tensor {n} differs between base and {tag}")

    # Step 2: merge the 64 MLP tensors, each task vector taken against the unmodified base.
    out = {}
    merged = 0
    for n in sorted(names):
        b = base[n]
        if n in mlp_names:
            b32 = b.float()
            tv1 = ft1[n].float() - b32
            tv2 = ft2[n].float() - b32
            out[n] = (b32 + LAMBDA * tv1 + LAMBDA * tv2).to(b.dtype).contiguous()
            merged += 1
        else:
            out[n] = b.contiguous()

    if merged != 64:
        fail(f"merged {merged} tensors, expected 64")
    if len(out) != 244:
        fail(f"output has {len(out)} tensors, expected 244")

    os.makedirs(HERE, exist_ok=True)
    save_file(out, OUT, metadata={"format": "pt"})

    # Re-read and confirm the file on disk.
    check = load_file(OUT)
    if len(check) != 244 or set(check) != names:
        fail("written file does not have the expected 244 tensor names")
    for n in names:
        if check[n].shape != base[n].shape or check[n].dtype != base[n].dtype:
            fail(f"written {n} shape/dtype mismatch")
        if n not in mlp_names and not torch.equal(check[n], base[n]):
            fail(f"written shared tensor {n} differs from base")
    print(f"OK: wrote {OUT} with {len(check)} tensors, {merged} merged (lambda={LAMBDA})")


if __name__ == "__main__":
    main()
