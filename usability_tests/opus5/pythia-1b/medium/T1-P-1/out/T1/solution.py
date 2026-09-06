"""T1: depth-prune Pythia-1B from 16 to 12 blocks with contiguous renumbering."""

import os
import re
import sys

from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(SANDBOX, "inputs", "base", "model.safetensors")
DST = os.path.join(HERE, "model.safetensors")

DROP = {2, 6, 10, 14}
N_OLD = 16
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def die(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


# old index -> new index, surviving blocks in original order
keep = [i for i in range(N_OLD) if i not in DROP]
remap = {old: new for new, old in enumerate(keep)}

tensors = {}
with safe_open(SRC, framework="pt") as f:
    src_keys = list(f.keys())
    for key in src_keys:
        m = LAYER_RE.match(key)
        if m is None:
            new_key = key
        else:
            old = int(m.group(1))
            if old in DROP:
                continue
            if old not in remap:
                die(f"unexpected block index {old} in {key}")
            new_key = f"gpt_neox.layers.{remap[old]}.{m.group(2)}"
        if new_key in tensors:
            die(f"destination collision on {new_key}")
        tensors[new_key] = f.get_tensor(key)

print(f"read {len(src_keys)} tensors, produced {len(tensors)}")

# --- required checks -------------------------------------------------------
for i in (12, 13, 14, 15):
    stale = [k for k in tensors if k.startswith(f"gpt_neox.layers.{i}.")]
    if stale:
        die(f"{len(stale)} tensors of block {i} remain, e.g. {stale[0]}")

qkv = [k for k in tensors if LAYER_RE.match(k) and k.endswith(".attention.query_key_value.weight")]
if len(qkv) != 12:
    die(f"expected 12 query_key_value.weight tensors, found {len(qkv)}")

indices = sorted(int(LAYER_RE.match(k).group(1)) for k in qkv)
if indices != list(range(12)):
    die(f"block indices are not contiguous 0..11: {indices}")

if len(tensors) != 184:
    die(f"expected 184 output tensors, found {len(tensors)}")

# every surviving block must still own its full 15 tensors
for new in range(12):
    n = len([k for k in tensors if k.startswith(f"gpt_neox.layers.{new}.")])
    if n != 15:
        die(f"block {new} has {n} tensors, expected 15")

save_file(tensors, DST, metadata={"format": "pt"})
print(f"wrote {DST} with {len(tensors)} tensors")
