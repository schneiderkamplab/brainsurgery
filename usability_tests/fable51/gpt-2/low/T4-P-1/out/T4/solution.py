"""T4: task-vector merge of two GPT-2 fine-tunes into the base (lambda = 0.4)."""
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAMBDA = 0.4
MLP_RE = re.compile(r"^h\.(\d+)\.mlp\.(c_fc|c_proj)\.(weight|bias)$")


def fail(msg):
    raise SystemExit(f"ERROR: {msg}")


base = load_file(os.path.join(ROOT, "inputs", "base", "model.safetensors"))
ft1 = load_file(os.path.join(ROOT, "inputs", "ft1", "model.safetensors"))
ft2 = load_file(os.path.join(ROOT, "inputs", "ft2", "model.safetensors"))

# Step 1: verify layouts and shared tensors before touching anything.
if not (set(base) == set(ft1) == set(ft2)):
    fail("tensor name sets differ between base, ft1 and ft2")
if len(base) != 160:
    fail(f"expected 160 tensors in base, got {len(base)}")

mlp_names = sorted(n for n in base if MLP_RE.match(n))
if len(mlp_names) != 48:
    fail(f"expected 48 MLP tensors, found {len(mlp_names)}")

for name in base:
    b, f1, f2 = base[name], ft1[name], ft2[name]
    for tag, t in (("ft1", f1), ("ft2", f2)):
        if t.shape != b.shape or t.dtype != b.dtype:
            fail(f"{name}: shape/dtype mismatch in {tag}: {tuple(t.shape)}/{t.dtype} "
                 f"vs base {tuple(b.shape)}/{b.dtype}")
    if name not in mlp_names:
        if not torch.equal(b, f1):
            fail(f"non-MLP tensor {name} differs between base and ft1")
        if not torch.equal(b, f2):
            fail(f"non-MLP tensor {name} differs between base and ft2")

# Step 2/3: merge MLP tensors against the unmodified base; copy everything else.
out = {}
merged = 0
for name in base:
    b = base[name]
    if name in mlp_names:
        if b.dtype != torch.float32:
            fail(f"{name}: expected float32, got {b.dtype}")
        tv1 = ft1[name] - b
        tv2 = ft2[name] - b
        out[name] = (b + LAMBDA * tv1 + LAMBDA * tv2).contiguous()
        merged += 1
    else:
        out[name] = b.clone().contiguous()

if merged != 48:
    fail(f"merged {merged} tensors, expected 48")
if len(out) != 160:
    fail(f"output has {len(out)} tensors, expected 160")

dest = os.path.join(ROOT, "out", "T4", "model.safetensors")
save_file(out, dest)

check = load_file(dest)
if len(check) != 160:
    fail(f"saved output has {len(check)} tensors, expected 160")
print(f"OK: merged {merged} MLP tensors, wrote {len(check)} tensors to {dest}")
