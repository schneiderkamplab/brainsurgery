"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 layers, dropping blocks 2, 6, 10, 14."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

BASE = "inputs/base"
OUT = "out/T1/model.safetensors"
DROP = {2, 6, 10, 14}
NUM_OLD = 16
NUM_NEW = NUM_OLD - len(DROP)
EXPECTED_TENSORS = 86
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# Load all shards via the index.
with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))
sd = {}
for shard in shards:
    part = load_file(os.path.join(BASE, shard))
    if set(part) & set(sd):
        fail(f"duplicate keys across shards in {shard}")
    sd.update(part)
if len(sd) != 114:
    fail(f"expected 114 input tensors, got {len(sd)}")

# Build the old->new index map in ascending order (no collisions possible
# because we build a fresh dict rather than renaming in place).
keep = [i for i in range(NUM_OLD) if i not in DROP]
remap = {old: new for new, old in enumerate(keep)}
if remap != {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 7: 5, 8: 6, 9: 7, 11: 8, 12: 9, 13: 10, 15: 11}:
    fail(f"unexpected remap {remap}")

new_sd = {}
for name, t in sd.items():
    m = LAYER_RE.match(name)
    if m is None:
        new_sd[name] = t
        continue
    old = int(m.group(1))
    if old >= NUM_OLD:
        fail(f"unexpected layer index {old} in {name}")
    if old in DROP:
        continue
    new_name = f"model.layers.{remap[old]}.{m.group(2)}"
    if new_name in new_sd:
        fail(f"collision on {new_name}")
    new_sd[new_name] = t

# Required checks.
for name in new_sd:
    m = LAYER_RE.match(name)
    if m and int(m.group(1)) >= NUM_NEW:
        fail(f"tensor of block >= {NUM_NEW} remains: {name}")
q_count = sum(1 for n in new_sd if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", n))
if q_count != NUM_NEW:
    fail(f"expected {NUM_NEW} blocks, found {q_count} q_proj tensors")
blocks = {int(LAYER_RE.match(n).group(1)) for n in new_sd if LAYER_RE.match(n)}
if blocks != set(range(NUM_NEW)):
    fail(f"block indices not contiguous: {sorted(blocks)}")
for i in range(NUM_NEW):
    n_i = sum(1 for n in new_sd if n.startswith(f"model.layers.{i}."))
    if n_i != 7:
        fail(f"block {i} has {n_i} tensors, expected 7")
if len(new_sd) != EXPECTED_TENSORS:
    fail(f"expected {EXPECTED_TENSORS} output tensors, got {len(new_sd)}")
# Values/shapes/dtypes preserved: verify against the source mapping.
for old, new in remap.items():
    for name, t in sd.items():
        if name.startswith(f"model.layers.{old}."):
            nt = new_sd[name.replace(f"model.layers.{old}.", f"model.layers.{new}.", 1)]
            if nt.shape != t.shape or nt.dtype != t.dtype or not torch.equal(nt, t):
                fail(f"mismatch for {name}")
for name in ("model.embed_tokens.weight", "lm_head.weight"):
    if name not in new_sd or not torch.equal(new_sd[name], sd[name]):
        fail(f"non-block tensor {name} altered or missing")

save_file({k: v.contiguous() for k, v in new_sd.items()}, OUT, metadata={"format": "pt"})
print(f"wrote {OUT} with {len(new_sd)} tensors, {NUM_NEW} blocks")
