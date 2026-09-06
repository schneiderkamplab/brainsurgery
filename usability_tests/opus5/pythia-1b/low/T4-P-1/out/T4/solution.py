"""T4: task-vector merge of two Pythia-1B fine-tunes."""

import sys

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
BASE = "inputs/base/model.safetensors"
FT1 = "inputs/ft1/model.safetensors"
FT2 = "inputs/ft2/model.safetensors"
OUT = "out/T4/model.safetensors"

MLP_SUFFIXES = (
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    "mlp.dense_4h_to_h.bias",
)


def die(msg):
    raise SystemExit(f"ERROR: {msg}")


base = load_file(BASE)
ft1 = load_file(FT1)
ft2 = load_file(FT2)

# 1. same tensor names everywhere
if set(base) != set(ft1) or set(base) != set(ft2):
    die(
        "tensor name sets differ: "
        f"ft1 only={sorted(set(ft1) - set(base))} base only={sorted(set(base) - set(ft1))} "
        f"ft2 only={sorted(set(ft2) - set(base))}"
    )

mlp_keys = {
    f"gpt_neox.layers.{i}.{suffix}" for i in range(16) for suffix in MLP_SUFFIXES
}
missing = mlp_keys - set(base)
if missing:
    die(f"expected MLP tensors are absent from the checkpoints: {sorted(missing)}")
if len(mlp_keys) != 64:
    die(f"expected 64 MLP tensor names, built {len(mlp_keys)}")

# every tensor outside the MLP set must be identical in all three
for key in sorted(set(base) - mlp_keys):
    b, a, c = base[key], ft1[key], ft2[key]
    if b.shape != a.shape or b.shape != c.shape:
        die(f"shape mismatch on shared tensor {key}: {b.shape} / {a.shape} / {c.shape}")
    if b.dtype != a.dtype or b.dtype != c.dtype:
        die(f"dtype mismatch on shared tensor {key}: {b.dtype} / {a.dtype} / {c.dtype}")
    if not torch.equal(b, a):
        die(f"shared tensor {key} differs between base and ft1")
    if not torch.equal(b, c):
        die(f"shared tensor {key} differs between base and ft2")

# shapes/dtypes must also line up on the tensors we are about to merge
for key in sorted(mlp_keys):
    b, a, c = base[key], ft1[key], ft2[key]
    if b.shape != a.shape or b.shape != c.shape:
        die(f"shape mismatch on MLP tensor {key}: {b.shape} / {a.shape} / {c.shape}")
    if b.dtype != a.dtype or b.dtype != c.dtype:
        die(f"dtype mismatch on MLP tensor {key}: {b.dtype} / {a.dtype} / {c.dtype}")

# 2./3. merge against the unmodified base; everything else copied straight over
out = {}
merged = 0
for key, tensor in base.items():
    if key in mlp_keys:
        b32 = tensor.to(torch.float32)
        merged_val = (
            b32
            + LAMBDA * (ft1[key].to(torch.float32) - b32)
            + LAMBDA * (ft2[key].to(torch.float32) - b32)
        )
        out[key] = merged_val.to(tensor.dtype)
        merged += 1
    else:
        out[key] = tensor.clone()

if merged != 64:
    die(f"merged {merged} tensors, expected exactly 64")
if len(out) != 244:
    die(f"output has {len(out)} tensors, expected exactly 244")

save_file(out, OUT)

check = load_file(OUT)
if len(check) != 244:
    die(f"written file has {len(check)} tensors, expected exactly 244")
print(f"wrote {OUT}: {len(check)} tensors, {merged} merged, lambda={LAMBDA}")
