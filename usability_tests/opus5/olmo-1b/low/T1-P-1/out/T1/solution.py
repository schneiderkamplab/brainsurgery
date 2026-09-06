"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 blocks, renumbering contiguously."""

import json
import os
import re
import sys

from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, "..", "..", "inputs", "base")
OUT_PATH = os.path.join(HERE, "model.safetensors")

DROP = {2, 6, 10, 14}
N_LAYERS = 16
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# --- load all shards via the index -------------------------------------------------
with open(os.path.join(IN_DIR, "model.safetensors.index.json")) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))

state = {}
for shard in shards:
    part = load_file(os.path.join(IN_DIR, shard))
    for k, v in part.items():
        if k in state:
            die(f"duplicate tensor across shards: {k}")
        state[k] = v

if len(state) != 114:
    die(f"expected 114 input tensors, got {len(state)}")

# --- build the old -> new block index map ------------------------------------------
survivors = [i for i in range(N_LAYERS) if i not in DROP]
if len(survivors) != 12:
    die(f"expected 12 surviving blocks, got {len(survivors)}")
remap = {old: new for new, old in enumerate(survivors)}

# --- rename ------------------------------------------------------------------------
out = {}
for key, tensor in state.items():
    m = LAYER_RE.match(key)
    if m is None:
        new_key = key  # embed_tokens / lm_head
    else:
        old_idx = int(m.group(1))
        if old_idx not in remap:
            continue  # dropped block
        new_key = f"model.layers.{remap[old_idx]}.{m.group(2)}"
    if new_key in out:
        die(f"renumbering collision: {new_key} produced twice")
    out[new_key] = tensor

# --- required checks ---------------------------------------------------------------
present = set()
for key in out:
    m = LAYER_RE.match(key)
    if m:
        present.add(int(m.group(1)))

stale = sorted(i for i in present if i >= 12)
if stale:
    die(f"tensors of blocks {stale} still present")

q_proj = [k for k in out if LAYER_RE.match(k) and k.endswith(".self_attn.q_proj.weight")]
if len(q_proj) != 12:
    die(f"expected 12 q_proj tensors, got {len(q_proj)}")

if present != set(range(12)):
    die(f"block indices are not contiguous 0..11: {sorted(present)}")

if len(out) != 86:
    die(f"expected 86 output tensors, got {len(out)}")

non_block = {k for k in out if not LAYER_RE.match(k)}
if non_block != {"model.embed_tokens.weight", "lm_head.weight"}:
    die(f"unexpected non-block tensors: {sorted(non_block)}")

# values/shapes/dtypes must be untouched: verify each output tensor is the same
# object as its source, under the expected source name.
inv = {new: old for old, new in remap.items()}
for key, tensor in out.items():
    m = LAYER_RE.match(key)
    src = f"model.layers.{inv[int(m.group(1))]}.{m.group(2)}" if m else key
    if state[src] is not tensor:
        die(f"{key} does not carry the tensor of {src}")

# --- write -------------------------------------------------------------------------
save_file({k: v.contiguous() for k, v in out.items()}, OUT_PATH)
print(f"wrote {OUT_PATH}: {len(out)} tensors, blocks 0..{max(present)}")
