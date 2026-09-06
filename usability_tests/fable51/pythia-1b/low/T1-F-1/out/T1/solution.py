"""T1: depth-prune Pythia-1B 16 -> 12 layers, renumber contiguously, verify, save."""
import os
import re
import sys

from safetensors import safe_open
from safetensors.torch import save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"
REMOVE = {2, 6, 10, 14}
N_OLD, N_NEW = 16, 12
PAT = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")

keep = [i for i in range(N_OLD) if i not in REMOVE]
remap = {old: new for new, old in enumerate(keep)}  # old 3 -> 2, 4 -> 3, ...

out = {}
with safe_open(SRC, framework="pt") as f:
    keys = list(f.keys())
    assert len(keys) == 244, f"expected 244 input tensors, got {len(keys)}"
    for k in keys:
        m = PAT.match(k)
        if m is None:
            new_k = k  # non-block tensor, unchanged
        else:
            old = int(m.group(1))
            if old in REMOVE:
                continue
            new_k = f"gpt_neox.layers.{remap[old]}.{m.group(2)}"
        assert new_k not in out, f"collision on {new_k}"
        out[new_k] = f.get_tensor(k).contiguous()

# Required checks (fail before anything is written).
idx = sorted({int(PAT.match(k).group(1)) for k in out if PAT.match(k)})
assert idx == list(range(N_NEW)), f"block indices not 0..{N_NEW-1}: {idx}"
assert not any(i >= N_NEW for i in idx), "tensors of blocks >= 12 remain"
qkv = [k for k in out if k.endswith(".attention.query_key_value.weight")]
assert len(qkv) == N_NEW, f"expected 12 qkv weights, got {len(qkv)}"
assert len(out) == 184, f"expected 184 tensors, got {len(out)}"
for i in range(N_NEW):
    n = sum(1 for k in out if k.startswith(f"gpt_neox.layers.{i}."))
    assert n == 15, f"block {i} has {n} tensors, expected 15"

if os.path.exists(DST):
    sys.exit(f"refusing to overwrite existing {DST}")
save_file(out, DST, metadata={"format": "pt"})
print(f"wrote {DST} with {len(out)} tensors")
