"""T4: task-vector merge of two GPT-2 fine-tunes into the base."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parents[2]
INPUTS = HERE / "inputs"
OUT = HERE / "out" / "T4" / "model.safetensors"

LAMBDA = 0.4
MLP_SUFFIXES = (
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    "mlp.c_proj.bias",
)
MLP_KEYS = {f"h.{i}.{s}" for i in range(12) for s in MLP_SUFFIXES}

base = load_file(INPUTS / "base" / "model.safetensors")
ft1 = load_file(INPUTS / "ft1" / "model.safetensors")
ft2 = load_file(INPUTS / "ft2" / "model.safetensors")

# 1. same tensor names everywhere
for name, sd in (("ft1", ft1), ("ft2", ft2)):
    if set(sd) != set(base):
        missing = sorted(set(base) - set(sd))
        extra = sorted(set(sd) - set(base))
        raise SystemExit(f"{name} key set differs from base: missing={missing} extra={extra}")

if len(base) != 160:
    raise SystemExit(f"expected 160 tensors in base, got {len(base)}")

missing_mlp = sorted(MLP_KEYS - set(base))
if missing_mlp:
    raise SystemExit(f"expected MLP tensors absent from base: {missing_mlp}")
if len(MLP_KEYS) != 48:
    raise SystemExit(f"expected 48 MLP tensor names, got {len(MLP_KEYS)}")

# shapes and dtypes must agree on every tensor
for key in base:
    for name, sd in (("ft1", ft1), ("ft2", ft2)):
        if sd[key].shape != base[key].shape or sd[key].dtype != base[key].dtype:
            raise SystemExit(
                f"{name}[{key}] has {tuple(sd[key].shape)}/{sd[key].dtype}, "
                f"base has {tuple(base[key].shape)}/{base[key].dtype}"
            )

# every non-MLP tensor must be bit-identical in all three checkpoints
differing = [
    key
    for key in sorted(set(base) - MLP_KEYS)
    if not (torch.equal(base[key], ft1[key]) and torch.equal(base[key], ft2[key]))
]
if differing:
    raise SystemExit(
        f"{len(differing)} non-MLP tensors differ between checkpoints, "
        f"frozen-backbone assumption violated: {differing[:10]}"
    )

# 2./3. merge; every task vector is taken against the unmodified base
out = {}
merged = 0
for key, tensor in base.items():
    if key in MLP_KEYS:
        b = tensor.to(torch.float32)
        out[key] = (
            b + LAMBDA * (ft1[key].to(torch.float32) - b) + LAMBDA * (ft2[key].to(torch.float32) - b)
        ).to(tensor.dtype).contiguous()
        merged += 1
    else:
        out[key] = tensor.clone().contiguous()

if merged != 48:
    raise SystemExit(f"expected to merge 48 tensors, merged {merged}")
if len(out) != 160:
    raise SystemExit(f"expected 160 output tensors, got {len(out)}")

OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(out, OUT)

check = load_file(OUT)
if len(check) != 160:
    raise SystemExit(f"written file has {len(check)} tensors, expected 160")
print(f"wrote {OUT} with {len(check)} tensors, {merged} merged (lambda={LAMBDA})")
