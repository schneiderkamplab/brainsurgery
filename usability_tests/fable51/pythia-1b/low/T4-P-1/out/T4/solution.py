"""T4: task-vector merge of two Pythia-1B fine-tunes into the base (MLP tensors only)."""
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "inputs/base/model.safetensors"
FT1 = ROOT / "inputs/ft1/model.safetensors"
FT2 = ROOT / "inputs/ft2/model.safetensors"
OUT = ROOT / "out/T4/model.safetensors"
LAMBDA = 0.4
MLP_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.mlp\.(dense_h_to_4h|dense_4h_to_h)\.(weight|bias)$")


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


base = load_file(str(BASE))
ft1 = load_file(str(FT1))
ft2 = load_file(str(FT2))

# Step 1: verify names, shapes, dtypes, and non-MLP equality before touching anything.
if not (set(base) == set(ft1) == set(ft2)):
    fail("tensor name sets differ between base/ft1/ft2")
if len(base) != 244:
    fail(f"expected 244 tensors in base, got {len(base)}")

mlp_names = sorted(n for n in base if MLP_RE.match(n))
if len(mlp_names) != 64:
    fail(f"expected 64 MLP tensors, found {len(mlp_names)}")

for name in base:
    for tag, sd in (("ft1", ft1), ("ft2", ft2)):
        if sd[name].shape != base[name].shape or sd[name].dtype != base[name].dtype:
            fail(f"{tag}[{name}] shape/dtype differs from base")
        if name not in mlp_names and not torch.equal(sd[name], base[name]):
            fail(f"shared tensor {name} differs between base and {tag}")

# Step 2/3: merge MLP tensors against the unmodified base; copy everything else.
out = {}
merged = 0
for name, b in base.items():
    if name in mlp_names:
        b32 = b.float()
        m = b32 + LAMBDA * (ft1[name].float() - b32) + LAMBDA * (ft2[name].float() - b32)
        out[name] = m.to(b.dtype).contiguous()
        merged += 1
    else:
        out[name] = b.contiguous()

if merged != 64:
    fail(f"merged {merged} tensors, expected 64")
if len(out) != 244:
    fail(f"output has {len(out)} tensors, expected 244")

OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(out, str(OUT), metadata={"format": "pt"})

# Post-write check.
n = len(load_file(str(OUT)))
if n != 244:
    fail(f"written file has {n} tensors, expected 244")
print(f"OK: merged {merged} MLP tensors, wrote {n} tensors to {OUT}")
