"""T1: depth-prune GPT-2 (drop blocks 2, 5, 8) and renumber blocks 0..8."""
import os
import re
import sys

from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"
DROP = {2, 5, 8}
BLOCK = re.compile(r"^h\.(\d+)\.(.+)$")

sd = load_file(SRC)
assert len(sd) == 160, len(sd)

kept_ids = sorted({int(m.group(1)) for k in sd if (m := BLOCK.match(k))} - DROP)
remap = {old: new for new, old in enumerate(kept_ids)}

out = {}
for k, v in sd.items():
    m = BLOCK.match(k)
    if m is None:
        out[k] = v
        continue
    idx = int(m.group(1))
    if idx in DROP:
        continue
    nk = f"h.{remap[idx]}.{m.group(2)}"
    if nk in out:
        sys.exit(f"collision on {nk}")
    out[nk] = v

# Required checks (before writing anything).
ids = {int(m.group(1)) for k in out if (m := BLOCK.match(k))}
assert ids.isdisjoint({9, 10, 11}), f"stale block indices: {ids & {9, 10, 11}}"
assert ids == set(range(9)), f"block ids not 0..8: {sorted(ids)}"
assert sum(k.endswith(".attn.c_attn.weight") for k in out) == 9
assert len(out) == 121, len(out)
for old, new in remap.items():
    for k in sd:
        m = BLOCK.match(k)
        if m and int(m.group(1)) == old:
            nk = f"h.{new}.{m.group(2)}"
            assert out[nk].data_ptr() == sd[k].data_ptr()
for k in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"):
    assert out[k].data_ptr() == sd[k].data_ptr()

os.makedirs(os.path.dirname(DST), exist_ok=True)
save_file({k: v.contiguous() for k, v in out.items()}, DST, metadata={"format": "pt"})
print(f"wrote {DST} with {len(out)} tensors")
