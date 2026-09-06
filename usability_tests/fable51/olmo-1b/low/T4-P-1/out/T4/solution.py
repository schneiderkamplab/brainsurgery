"""T4: task-vector merge of two fine-tunes (OLMo-1B) with lambda = 0.4."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_DIR = os.path.join(ROOT, "inputs", "base")
FT1 = os.path.join(ROOT, "inputs", "ft1", "model.safetensors")
FT2 = os.path.join(ROOT, "inputs", "ft2", "model.safetensors")
OUT = os.path.join(HERE, "model.safetensors")
LAMBDA = 0.4
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_sharded(d: str) -> dict:
    with open(os.path.join(d, "model.safetensors.index.json")) as f:
        index = json.load(f)
    tensors: dict = {}
    for shard in sorted(set(index["weight_map"].values())):
        part = load_file(os.path.join(d, shard))
        if set(part) & set(tensors):
            fail(f"duplicate tensor names across shards in {shard}")
        tensors.update(part)
    if set(tensors) != set(index["weight_map"]):
        fail("index weight_map does not match shard contents")
    return tensors


base = load_sharded(BASE_DIR)
ft1 = load_file(FT1)
ft2 = load_file(FT2)

# Step 1: same names in all three, and all non-MLP tensors identical.
if not (set(base) == set(ft1) == set(ft2)):
    fail("tensor name sets differ between base, ft1 and ft2")
if len(base) != 114:
    fail(f"expected 114 tensors in base, found {len(base)}")

mlp_names = sorted(n for n in base if MLP_RE.match(n))
if len(mlp_names) != 48:
    fail(f"expected 48 MLP tensors, found {len(mlp_names)}")

for name in base:
    b, a1, a2 = base[name], ft1[name], ft2[name]
    if not (b.shape == a1.shape == a2.shape and b.dtype == a1.dtype == a2.dtype):
        fail(f"shape/dtype mismatch for {name}")
    if name in mlp_names:
        continue
    if not (torch.equal(b, a1) and torch.equal(b, a2)):
        fail(f"non-MLP tensor differs between checkpoints: {name}")

# Step 2/3: merge MLP tensors against the unmodified base; copy the rest.
out: dict = {}
merged = 0
for name in base:
    b = base[name]
    if name in mlp_names:
        if b.dtype != torch.float32:
            fail(f"{name} is {b.dtype}, expected float32")
        tv = LAMBDA * (ft1[name] - b) + LAMBDA * (ft2[name] - b)
        out[name] = (b + tv).contiguous()
        merged += 1
    else:
        out[name] = b.contiguous()

if merged != 48:
    fail(f"merged {merged} tensors, expected 48")
if len(out) != 114:
    fail(f"output has {len(out)} tensors, expected 114")

os.makedirs(HERE, exist_ok=True)
save_file(out, OUT, metadata={"format": "pt"})

# Post-write check.
check = load_file(OUT)
if len(check) != 114 or set(check) != set(base):
    fail("written file does not contain the expected 114 tensors")
print(f"OK: merged {merged} MLP tensors, wrote {len(check)} tensors to {OUT}")
