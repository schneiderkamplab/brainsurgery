"""T4: task-vector merge of two fine-tunes onto OLMo-1B base (lambda = 0.4)."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.abspath(os.path.join(ROOT, "..", ".."))
BASE_DIR = os.path.join(SANDBOX, "inputs", "base")
FT1_PATH = os.path.join(SANDBOX, "inputs", "ft1", "model.safetensors")
FT2_PATH = os.path.join(SANDBOX, "inputs", "ft2", "model.safetensors")
OUT_PATH = os.path.join(ROOT, "model.safetensors")

LAMBDA = 0.4
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
EXPECTED_TOTAL = 114
EXPECTED_MLP = 48


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_sharded(directory):
    with open(os.path.join(directory, "model.safetensors.index.json")) as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))
    sd = {}
    for shard in shards:
        part = load_file(os.path.join(directory, shard))
        dup = set(part) & set(sd)
        if dup:
            fail(f"duplicate tensor names across shards: {sorted(dup)[:5]}")
        sd.update(part)
    missing = set(index["weight_map"]) - set(sd)
    if missing:
        fail(f"index lists tensors not found in shards: {sorted(missing)[:5]}")
    return sd


base = load_sharded(BASE_DIR)
ft1 = load_file(FT1_PATH)
ft2 = load_file(FT2_PATH)

# Step 1: verify names and shared tensors before touching anything.
names = set(base)
if set(ft1) != names or set(ft2) != names:
    fail(
        "tensor name sets differ: "
        f"ft1-base={sorted(set(ft1) - names)[:5]}, base-ft1={sorted(names - set(ft1))[:5]}, "
        f"ft2-base={sorted(set(ft2) - names)[:5]}, base-ft2={sorted(names - set(ft2))[:5]}"
    )
if len(names) != EXPECTED_TOTAL:
    fail(f"expected {EXPECTED_TOTAL} tensors, found {len(names)}")

mlp_names = sorted(n for n in names if MLP_RE.match(n))
if len(mlp_names) != EXPECTED_MLP:
    fail(f"expected {EXPECTED_MLP} MLP tensors, found {len(mlp_names)}")
shared_names = sorted(names - set(mlp_names))

for n in names:
    for tag, sd in (("ft1", ft1), ("ft2", ft2)):
        if sd[n].shape != base[n].shape or sd[n].dtype != base[n].dtype:
            fail(f"{tag}[{n}] shape/dtype {tuple(sd[n].shape)}/{sd[n].dtype} != "
                 f"base {tuple(base[n].shape)}/{base[n].dtype}")

differing = []
for n in shared_names:
    if not torch.equal(base[n], ft1[n]):
        differing.append(("ft1", n))
    if not torch.equal(base[n], ft2[n]):
        differing.append(("ft2", n))
if differing:
    fail(f"{len(differing)} non-MLP tensors differ from base: {differing[:10]}")
print(f"verified: {len(shared_names)} shared tensors identical across base/ft1/ft2")

# Step 2: merge the 48 MLP tensors against the unmodified base, in float32.
out = {}
merged = 0
for n in names:
    b = base[n]
    if n in mlp_names:
        if b.dtype != torch.float32:
            fail(f"{n} is {b.dtype}, expected float32")
        b32 = b.float()
        out[n] = (b32 + LAMBDA * (ft1[n].float() - b32) + LAMBDA * (ft2[n].float() - b32)).contiguous()
        merged += 1
    else:
        out[n] = b.contiguous()

if merged != EXPECTED_MLP:
    fail(f"merged {merged} tensors, expected {EXPECTED_MLP}")
if len(out) != EXPECTED_TOTAL:
    fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

save_file(out, OUT_PATH, metadata={"format": "pt"})

# Re-open the written file and confirm the count.
written = load_file(OUT_PATH)
if len(written) != EXPECTED_TOTAL or set(written) != names:
    fail(f"written file has {len(written)} tensors / name mismatch")
print(f"merged {merged} MLP tensors, wrote {len(written)} tensors to {OUT_PATH}")
