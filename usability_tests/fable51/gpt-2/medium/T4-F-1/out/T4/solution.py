"""T4: task-vector merge of two GPT-2 fine-tunes (lambda = 0.4).

out[X] = base[X] + lam*(ft1[X]-base[X]) + lam*(ft2[X]-base[X]) for the 48 MLP
tensors; every other tensor copied bit-exact from the base, after verifying the
three checkpoints agree on names and on all non-MLP tensors.
"""
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
LAM = 0.4
N_LAYERS = 12
MLP_RE = re.compile(r"^h\.(\d+)\.mlp\.(c_fc|c_proj)\.(weight|bias)$")
EXPECTED_MLP = {
    f"h.{i}.mlp.{m}.{p}" for i in range(N_LAYERS) for m in ("c_fc", "c_proj") for p in ("weight", "bias")
}
OUT = ROOT / "out" / "T4" / "model.safetensors"


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base = load_file(ROOT / "inputs" / "base" / "model.safetensors")
    ft1 = load_file(ROOT / "inputs" / "ft1" / "model.safetensors")
    ft2 = load_file(ROOT / "inputs" / "ft2" / "model.safetensors")

    # Step 1: identical name sets, identical non-MLP tensors, before any merging.
    if not (base.keys() == ft1.keys() == ft2.keys()):
        fail(
            "tensor name sets differ: "
            f"base^ft1={sorted(set(base) ^ set(ft1))}, base^ft2={sorted(set(base) ^ set(ft2))}"
        )
    if len(base) != 160:
        fail(f"expected 160 tensors in base, found {len(base)}")

    mlp_names = {k for k in base if MLP_RE.match(k)}
    if mlp_names != EXPECTED_MLP:
        fail(f"MLP tensor set mismatch: missing={sorted(EXPECTED_MLP - mlp_names)}, "
             f"extra={sorted(mlp_names - EXPECTED_MLP)}")

    for name in sorted(set(base) - mlp_names):
        for tag, other in (("ft1", ft1), ("ft2", ft2)):
            b, o = base[name], other[name]
            if b.shape != o.shape or b.dtype != o.dtype:
                fail(f"{tag}[{name}] shape/dtype {tuple(o.shape)}/{o.dtype} != base "
                     f"{tuple(b.shape)}/{b.dtype}")
            if not torch.equal(b, o):
                fail(f"non-MLP tensor {name} differs between base and {tag}")

    # Step 2/3: merge the MLP tensors against the unmodified base; copy the rest.
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for name, b in base.items():
        if name in mlp_names:
            for tag, other in (("ft1", ft1), ("ft2", ft2)):
                if other[name].shape != b.shape or other[name].dtype != b.dtype:
                    fail(f"{tag}[{name}] shape/dtype mismatch with base")
            if b.dtype != torch.float32:
                fail(f"{name} is {b.dtype}, expected float32")
            b32, f1, f2 = b.float(), ft1[name].float(), ft2[name].float()
            out[name] = (b32 + LAM * (f1 - b32) + LAM * (f2 - b32)).contiguous()
            merged += 1
        else:
            out[name] = b.contiguous()

    if merged != 48:
        fail(f"merged {merged} tensors, expected 48")
    if len(out) != 160:
        fail(f"output has {len(out)} tensors, expected 160")
    if out.keys() != base.keys():
        fail("output key set differs from base")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata={"format": "pt"})

    # Re-read and check the written file.
    written = load_file(OUT)
    if len(written) != 160:
        fail(f"written file has {len(written)} tensors, expected 160")
    for name in set(base) - mlp_names:
        if not torch.equal(written[name], base[name]):
            fail(f"written unchanged tensor {name} differs from base")
    print(f"OK: wrote {OUT} with {len(written)} tensors ({merged} merged, lambda={LAM})")


if __name__ == "__main__":
    main()
