"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 layers, renumbering contiguously."""
import json
import os
import re
import sys

from safetensors import safe_open
from safetensors.torch import save_file

BASE = os.path.join("inputs", "base")
OUT_DIR = os.path.join("out", "T1")
OUT = os.path.join(OUT_DIR, "model.safetensors")
REMOVE = {2, 6, 10, 14}
N_OLD, N_NEW, PER_BLOCK, N_TOTAL = 16, 12, 7, 86
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# Load all tensors from the shards.
with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))
src = {}
for shard in shards:
    with safe_open(os.path.join(BASE, shard), framework="pt") as f:
        for k in f.keys():
            if k in src:
                fail(f"duplicate key across shards: {k}")
            src[k] = f.get_tensor(k)
if len(src) != PER_BLOCK * N_OLD + 2:
    fail(f"expected {PER_BLOCK * N_OLD + 2} input tensors, got {len(src)}")

# Build old->new index map from surviving blocks in original order.
survivors = [i for i in range(N_OLD) if i not in REMOVE]
remap = {old: new for new, old in enumerate(survivors)}
if len(remap) != N_NEW:
    fail(f"expected {N_NEW} surviving blocks, got {len(remap)}")

# Build the new state dict into a fresh dict, so renaming cannot collide.
dst = {}
for k, t in src.items():
    m = LAYER_RE.match(k)
    if m is None:
        new_k = k
    else:
        old = int(m.group(1))
        if old >= N_OLD:
            fail(f"unexpected layer index in {k}")
        if old in REMOVE:
            continue
        new_k = f"model.layers.{remap[old]}.{m.group(2)}"
    if new_k in dst:
        fail(f"collision on destination key {new_k}")
    dst[new_k] = t.contiguous()

# Required checks.
for k in dst:
    m = LAYER_RE.match(k)
    if m and int(m.group(1)) >= N_NEW:
        fail(f"block index >= {N_NEW} remains: {k}")
q_keys = [k for k in dst if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", k)]
if len(q_keys) != N_NEW:
    fail(f"expected {N_NEW} q_proj tensors, got {len(q_keys)}")
blocks = {int(LAYER_RE.match(k).group(1)) for k in dst if LAYER_RE.match(k)}
if blocks != set(range(N_NEW)):
    fail(f"block indices not contiguous 0..{N_NEW - 1}: {sorted(blocks)}")
if len(dst) != N_TOTAL:
    fail(f"expected {N_TOTAL} output tensors, got {len(dst)}")
# Value/shape/dtype integrity against the source.
for old, new in remap.items():
    for rest in ("self_attn.q_proj.weight", "mlp.down_proj.weight"):
        a, b = src[f"model.layers.{old}.{rest}"], dst[f"model.layers.{new}.{rest}"]
        if a.shape != b.shape or a.dtype != b.dtype or not a.equal(b):
            fail(f"mismatch renaming layer {old} -> {new} ({rest})")
for k in ("model.embed_tokens.weight", "lm_head.weight"):
    if not src[k].equal(dst[k]):
        fail(f"non-block tensor changed: {k}")

os.makedirs(OUT_DIR, exist_ok=True)
save_file(dst, OUT, metadata={"format": "pt"})

with safe_open(OUT, framework="pt") as f:
    n = len(list(f.keys()))
if n != N_TOTAL:
    os.remove(OUT)
    fail(f"written file has {n} tensors, expected {N_TOTAL}")
print(f"OK: wrote {OUT} with {n} tensors, blocks 0..{N_NEW - 1}")
