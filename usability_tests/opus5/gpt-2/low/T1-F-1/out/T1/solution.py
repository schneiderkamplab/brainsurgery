"""T1: depth-prune GPT-2 blocks 2, 5, 8 and renumber survivors contiguously."""
import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T1/model.safetensors")
DROP = {2, 5, 8}
N_OLD = 12
BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


keep = [i for i in range(N_OLD) if i not in DROP]
remap = {old: new for new, old in enumerate(keep)}

with safe_open(SRC, framework="pt") as f:
    src = {k: f.get_tensor(k) for k in f.keys()}

out = {}
for k, v in src.items():
    m = BLOCK_RE.match(k)
    if m is None:
        out[k] = v
        continue
    idx, rest = int(m.group(1)), m.group(2)
    if idx in DROP:
        continue
    nk = f"h.{remap[idx]}.{rest}"
    if nk in out:
        fail(f"renumbering collision on {nk}")
    out[nk] = v

# Required checks.
for bad in (9, 10, 11):
    stray = [k for k in out if BLOCK_RE.match(k) and int(BLOCK_RE.match(k).group(1)) == bad]
    if stray:
        fail(f"tensors of block {bad} remain: {stray[:3]}")
anchors = [k for k in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", k)]
if len(anchors) != 9:
    fail(f"expected 9 blocks, found {len(anchors)}")
indices = sorted(int(BLOCK_RE.match(k).group(1)) for k in anchors)
if indices != list(range(9)):
    fail(f"block indices not contiguous 0..8: {indices}")
if len(out) != 121:
    fail(f"expected 121 tensors, got {len(out)}")

DST.parent.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous() for k, v in out.items()}, str(DST))
print(f"wrote {DST} with {len(out)} tensors, {len(anchors)} blocks")
