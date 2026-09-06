"""T1: drop blocks 2,6,10,14 from OLMo-1B (16 layers) and renumber to 0..11."""
import json, os, re, sys
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "inputs/base"
OUT = "out/T1/model.safetensors"
DROP = {2, 6, 10, 14}
N_OLD, N_NEW = 16, 12
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

survivors = [i for i in range(N_OLD) if i not in DROP]
assert len(survivors) == N_NEW
remap = {old: new for new, old in enumerate(survivors)}  # order-preserving, collision-free

index = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
shards = sorted(set(index["weight_map"].values()))
out = {}
n_in = 0
for shard in shards:
    with safe_open(os.path.join(SRC, shard), framework="pt") as f:
        for k in f.keys():
            n_in += 1
            m = LAYER_RE.match(k)
            if m is None:
                nk = k
            else:
                old = int(m.group(1))
                if old in DROP:
                    continue
                nk = f"model.layers.{remap[old]}.{m.group(2)}"
            if nk in out:
                sys.exit(f"collision: {nk} already present (from {k})")
            out[nk] = f.get_tensor(k).contiguous()
assert n_in == 114, f"expected 114 input tensors, got {n_in}"

# Required checks (run before writing anything)
for k in out:
    m = LAYER_RE.match(k)
    if m and int(m.group(1)) >= N_NEW:
        sys.exit(f"check failed: block index out of range in {k}")
q = [k for k in out if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", k)]
if len(q) != N_NEW:
    sys.exit(f"check failed: expected {N_NEW} blocks, found {len(q)}")
if sorted(int(LAYER_RE.match(k).group(1)) for k in q) != list(range(N_NEW)):
    sys.exit("check failed: block indices not contiguous 0..11")
if len(out) != 86:
    sys.exit(f"check failed: expected 86 tensors, got {len(out)}")

save_file(out, OUT, metadata={"format": "pt"})
print(f"wrote {OUT} with {len(out)} tensors")
