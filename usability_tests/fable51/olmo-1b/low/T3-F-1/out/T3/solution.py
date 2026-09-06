"""T3: mixed-precision export with sharding for OLMo-1B-0724-hf.

Plain torch + safetensors. Casts exactly the 112 per-layer projection
matrices to bfloat16, keeps everything else float32, and writes a sharded
safetensors checkpoint with a 256 MiB per-shard budget (oversized tensors
get their own shard). Every required check raises before anything is written.
"""
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "inputs" / "base"
DST = ROOT / "out" / "T3"
SHARD_BUDGET = 256 * 1024 * 1024

PROJ_RE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)

# ---- load (keys sorted by original index order for stability) -------------
index = json.loads((SRC / "model.safetensors.index.json").read_text())
names = list(index["weight_map"].keys())
tensors: dict[str, torch.Tensor] = {}
for shard in sorted(set(index["weight_map"].values())):
    with safe_open(SRC / shard, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
assert set(tensors) == set(names), "index and shards disagree on tensor names"
assert len(tensors) == 114, f"expected 114 input tensors, got {len(tensors)}"

# ---- transform --------------------------------------------------------------
cast = [k for k in names if PROJ_RE.match(k)]
assert len(cast) == 112, f"projection pattern matched {len(cast)} tensors, expected 112"
out = {k: (v.to(torch.bfloat16) if k in cast else v) for k, v in tensors.items()}

# ---- required checks (fail before writing) ----------------------------------
n_bf16 = sum(v.dtype == torch.bfloat16 for v in out.values())
assert n_bf16 == 112, f"{n_bf16} bfloat16 tensors, expected 112"
assert out["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
assert out["model.embed_tokens.weight"].dtype == torch.float32
assert out["lm_head.weight"].dtype == torch.float32
assert all(v.dtype == torch.float32 for k, v in out.items() if k not in cast)
assert all(torch.equal(out[k], tensors[k]) for k in names if k not in cast)
assert len(out) == 114, f"output has {len(out)} tensors, expected 114"
assert set(out) == set(names), "tensor names changed"

# ---- shard: greedy by index order, oversized tensors alone ------------------
shards: list[list[str]] = []
cur: list[str] = []
cur_bytes = 0
for k in names:
    nbytes = out[k].numel() * out[k].element_size()
    if nbytes > SHARD_BUDGET:
        if cur:
            shards.append(cur)
            cur, cur_bytes = [], 0
        shards.append([k])
        continue
    if cur and cur_bytes + nbytes > SHARD_BUDGET:
        shards.append(cur)
        cur, cur_bytes = [], 0
    cur.append(k)
    cur_bytes += nbytes
if cur:
    shards.append(cur)
for s in shards:
    total = sum(out[k].numel() * out[k].element_size() for k in s)
    assert total <= SHARD_BUDGET or len(s) == 1, f"shard over budget: {total}"

# ---- write ------------------------------------------------------------------
DST.mkdir(parents=True, exist_ok=True)
for p in DST.glob("model-*.safetensors"):
    p.unlink()
n = len(shards)
weight_map: dict[str, str] = {}
total_size = 0
for i, s in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    save_file({k: out[k].contiguous() for k in s}, DST / fname, metadata={"format": "pt"})
    for k in s:
        weight_map[k] = fname
        total_size += out[k].numel() * out[k].element_size()
(DST / "model.safetensors.index.json").write_text(
    json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2)
)

# ---- verify what landed on disk ------------------------------------------------
seen = 0
for fname in sorted(set(weight_map.values())):
    with safe_open(DST / fname, framework="pt") as f:
        keys = list(f.keys())
        seen += len(keys)
        for k in keys:
            assert weight_map[k] == fname
            t = f.get_tensor(k)
            assert torch.equal(t, out[k]) and t.dtype == out[k].dtype, k
assert seen == 114, f"{seen} tensors on disk, expected 114"
print(f"wrote {seen} tensors in {n} shards to {DST}")
