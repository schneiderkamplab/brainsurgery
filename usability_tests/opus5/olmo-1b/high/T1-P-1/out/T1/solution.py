"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 transformer blocks.

Drops blocks 2, 6, 10, 14 and renumbers the survivors to 0..11, preserving
order, values, shapes and dtypes. Writes out/T1/model.safetensors.
"""

import json
import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / "inputs" / "base"
DST = HERE / "model.safetensors"

DROP = {2, 6, 10, 14}
N_LAYERS_IN = 16
EXPECTED_TENSORS = 86
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def die(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


# --- load every shard listed in the index -----------------------------------
index = json.loads((SRC / "model.safetensors.index.json").read_text())
weight_map = index["weight_map"]

state = {}
for shard in sorted(set(weight_map.values())):
    for name, tensor in load_file(SRC / shard).items():
        if name in state:
            die(f"duplicate tensor across shards: {name}")
        state[name] = tensor

if set(state) != set(weight_map):
    die("loaded tensors do not match the index weight_map")

# --- build the old -> new block index map -----------------------------------
survivors = [i for i in range(N_LAYERS_IN) if i not in DROP]
remap = {old: new for new, old in enumerate(survivors)}

out = {}
for name, tensor in state.items():
    m = LAYER_RE.match(name)
    if m is None:
        out[name] = tensor  # non-block tensor, carried over unchanged
        continue
    old = int(m.group(1))
    if old >= N_LAYERS_IN:
        die(f"block index {old} outside 0..{N_LAYERS_IN - 1}: {name}")
    if old in DROP:
        continue
    new_name = f"model.layers.{remap[old]}.{m.group(2)}"
    if new_name in out:
        die(f"renumbering collision: {name} -> {new_name}")
    out[new_name] = tensor

# --- required checks (before anything is written) ---------------------------
blocks = sorted({int(m.group(1)) for m in map(LAYER_RE.match, out) if m})

stale = [b for b in blocks if b >= 12]
if stale:
    die(f"tensors of blocks {stale} remain; expected none at index >= 12")

if blocks != list(range(12)):
    die(f"block indices are not exactly 0..11: {blocks}")

n_q = sum(
    1 for k in out if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", k)
)
if n_q != 12:
    die(f"expected exactly 12 blocks (q_proj tensors), found {n_q}")

if len(out) != EXPECTED_TENSORS:
    die(f"expected exactly {EXPECTED_TENSORS} tensors, found {len(out)}")

for old, new in remap.items():
    for suffix in sorted(k.split(".", 3)[3] for k in state if k.startswith(f"model.layers.{old}.")):
        src, dst = f"model.layers.{old}.{suffix}", f"model.layers.{new}.{suffix}"
        if dst not in out:
            die(f"missing renamed tensor {dst}")
        if out[dst].data_ptr() != state[src].data_ptr():
            die(f"{dst} is not the original tensor of {src}")

for name in ("model.embed_tokens.weight", "lm_head.weight"):
    if name not in out or out[name].data_ptr() != state[name].data_ptr():
        die(f"non-block tensor {name} is missing or altered")

# --- write ------------------------------------------------------------------
# clone so tied/shared storages (e.g. tied embeddings) are separate buffers
save_file({k: v.clone().contiguous() for k, v in out.items()}, DST, metadata={"format": "pt"})

# --- verify what landed on disk ---------------------------------------------
back = load_file(DST)
if set(back) != set(out) or len(back) != EXPECTED_TENSORS:
    die("written file does not match the intended key set")
for k, v in out.items():
    if back[k].shape != v.shape or back[k].dtype != v.dtype or not back[k].equal(v):
        die(f"written tensor differs from source: {k}")

print(f"OK: wrote {len(back)} tensors to {DST}")
