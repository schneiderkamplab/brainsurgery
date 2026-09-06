"""T1: depth-prune GPT-2 (124M) from 12 to 9 blocks and renumber contiguously."""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent.parent
SRC = SANDBOX / "inputs" / "base" / "model.safetensors"
DST = HERE / "model.safetensors"

DROP = {2, 5, 8}
N_BLOCKS_IN = 12
BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


state = load_file(str(SRC))
print(f"loaded {len(state)} tensors from {SRC}")

keep = [i for i in range(N_BLOCKS_IN) if i not in DROP]
remap = {old: new for new, old in enumerate(keep)}
print(f"block remap: {remap}")

# Build a fresh dict; never rename in place, so a shifted block can never
# overwrite a surviving one.
out = {}
for name, tensor in state.items():
    m = BLOCK_RE.match(name)
    if m is None:
        out[name] = tensor  # wte / wpe / ln_f.*
        continue
    old = int(m.group(1))
    if old in DROP:
        continue
    new_name = f"h.{remap[old]}.{m.group(2)}"
    if new_name in out:
        fail(f"renumbering collision on {new_name}")
    out[new_name] = tensor

# --- required checks, before anything is written ---

survivors = {}
for name in out:
    m = BLOCK_RE.match(name)
    if m is not None:
        survivors.setdefault(int(m.group(1)), []).append(name)

stale = sorted(i for i in survivors if i >= 9)
if stale:
    fail(f"tensors of blocks {stale} still present")

n_c_attn = sum(1 for name in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", name))
if n_c_attn != 9:
    fail(f"expected 9 blocks by h.<i>.attn.c_attn.weight, found {n_c_attn}")

if sorted(survivors) != list(range(9)):
    fail(f"block indices are not contiguous 0..8: {sorted(survivors)}")

for i, names in sorted(survivors.items()):
    if len(names) != 13:
        fail(f"block {i} has {len(names)} tensors, expected 13")

if len(out) != 121:
    fail(f"output has {len(out)} tensors, expected 121")

# values/shapes/dtypes must be untouched
for new_name, tensor in out.items():
    m = BLOCK_RE.match(new_name)
    src_name = new_name
    if m is not None:
        src_name = f"h.{keep[int(m.group(1))]}.{m.group(2)}"
    src = state[src_name]
    if src.shape != tensor.shape or src.dtype != tensor.dtype:
        fail(f"{new_name}: shape/dtype changed vs {src_name}")
    if src.data_ptr() != tensor.data_ptr():
        fail(f"{new_name}: not the original tensor from {src_name}")

DST.parent.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous() for k, v in out.items()}, str(DST))

check = load_file(str(DST))
if len(check) != 121 or set(check) != set(out):
    fail("written file does not match the in-memory result")
for k in check:
    if not check[k].equal(out[k]) or check[k].dtype != out[k].dtype:
        fail(f"{k}: written values differ")

print(f"wrote {len(check)} tensors to {DST}")
print("OK")
