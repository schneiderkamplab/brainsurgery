"""T3: mixed-precision export of OLMo-1B-0724-hf with sharding."""
import json
import re
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "inputs" / "base"
OUT_DIR = ROOT / "out" / "T3"
MAX_SHARD_BYTES = 256 * 1024 * 1024  # 268,435,456

PROJ_RE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)

# --- load, preserving the index order ---
index = json.loads((IN_DIR / "model.safetensors.index.json").read_text())
shard_files = sorted(set(index["weight_map"].values()))
state = OrderedDict()
for f in shard_files:
    state.update(load_file(str(IN_DIR / f)))
assert len(state) == 114, f"expected 114 input tensors, got {len(state)}"

# --- transform ---
out = OrderedDict()
n_cast = 0
for name, t in state.items():
    if PROJ_RE.match(name):
        assert t.dtype == torch.float32 and t.ndim == 2, name
        out[name] = t.to(torch.bfloat16).contiguous()
        n_cast += 1
    else:
        assert t.dtype == torch.float32, f"{name} is {t.dtype}, expected float32"
        out[name] = t.contiguous()

# --- required checks (before writing) ---
n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
assert n_bf16 == 112, f"expected exactly 112 bfloat16 tensors, got {n_bf16}"
assert n_cast == 112, f"regex matched {n_cast} tensors, expected 112"
assert out["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
assert out["model.embed_tokens.weight"].dtype == torch.float32
assert out["lm_head.weight"].dtype == torch.float32
assert len(out) == 114, f"expected 114 output tensors, got {len(out)}"
assert set(out) == set(state), "tensor names changed"

# --- shard: greedy in order, oversized tensors alone ---
shards = []  # list of (names, nbytes)
cur, cur_bytes = [], 0
for name, t in out.items():
    nb = t.numel() * t.element_size()
    if nb > MAX_SHARD_BYTES:
        if cur:
            shards.append((cur, cur_bytes))
            cur, cur_bytes = [], 0
        shards.append(([name], nb))
        continue
    if cur and cur_bytes + nb > MAX_SHARD_BYTES:
        shards.append((cur, cur_bytes))
        cur, cur_bytes = [], 0
    cur.append(name)
    cur_bytes += nb
if cur:
    shards.append((cur, cur_bytes))

for names, nb in shards:
    assert nb <= MAX_SHARD_BYTES or len(names) == 1, (names, nb)

# --- write ---
OUT_DIR.mkdir(parents=True, exist_ok=True)
n_shards = len(shards)
weight_map = {}
total = 0
for i, (names, nb) in enumerate(shards, start=1):
    fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
    save_file({n: out[n] for n in names}, str(OUT_DIR / fname), metadata={"format": "pt"})
    for n in names:
        weight_map[n] = fname
    total += nb
assert len(weight_map) == 114
(OUT_DIR / "model.safetensors.index.json").write_text(
    json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n"
)

print(f"wrote {n_shards} shards, {len(weight_map)} tensors, {total} bytes to {OUT_DIR}")
for i, (names, nb) in enumerate(shards, start=1):
    print(f"  shard {i}: {len(names)} tensors, {nb} bytes")
