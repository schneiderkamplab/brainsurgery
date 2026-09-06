"""T4: task-vector merge of two Pythia-1B fine-tunes (lambda = 0.4)."""
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
LAMBDA = 0.4
MLP_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_(h_to_4h|4h_to_h)\.(weight|bias)$")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


base = load_file(ROOT / "inputs/base/model.safetensors")
ft1 = load_file(ROOT / "inputs/ft1/model.safetensors")
ft2 = load_file(ROOT / "inputs/ft2/model.safetensors")

# Step 1: same names, and everything outside the MLP tensors identical.
if not (set(base) == set(ft1) == set(ft2)):
    fail("tensor name sets differ between checkpoints")
if len(base) != 244:
    fail(f"expected 244 tensors in base, got {len(base)}")

mlp_names = {n for n in base if MLP_RE.match(n)}
if len(mlp_names) != 64:
    fail(f"expected 64 MLP tensors, matched {len(mlp_names)}")

for name in base:
    for tag, ft in (("ft1", ft1), ("ft2", ft2)):
        if ft[name].shape != base[name].shape or ft[name].dtype != base[name].dtype:
            fail(f"{tag}[{name}] shape/dtype differs from base")
        if name not in mlp_names and not torch.equal(ft[name], base[name]):
            fail(f"shared tensor {name} differs between base and {tag}")

# Step 2/3: merge, each task vector taken against the unmodified base.
out = {}
merged = 0
for name, b in base.items():
    if name in mlp_names:
        b32 = b.float()
        t = b32 + LAMBDA * (ft1[name].float() - b32) + LAMBDA * (ft2[name].float() - b32)
        out[name] = t.to(b.dtype).contiguous()
        merged += 1
    else:
        out[name] = b.contiguous()

if merged != 64:
    fail(f"merged {merged} tensors, expected 64")
if len(out) != 244:
    fail(f"output has {len(out)} tensors, expected 244")

dest = ROOT / "out/T4/model.safetensors"
save_file(out, dest, metadata={"format": "pt"})

# Re-read and confirm the file on disk.
check = load_file(dest)
if len(check) != 244 or set(check) != set(base):
    fail("written file does not have the expected 244 tensor names")
print(f"OK: wrote {dest} with {len(check)} tensors, {merged} merged")
