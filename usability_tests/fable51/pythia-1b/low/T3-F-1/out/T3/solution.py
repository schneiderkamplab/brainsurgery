"""T3: mixed-precision sharded export of Pythia-1B (plain torch + safetensors)."""
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T3")
MAX_SHARD = 256 * 1024 * 1024

PROJ_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.(attention\.(query_key_value|dense)|mlp\.(dense_h_to_4h|dense_4h_to_h))\.weight$"
)
BUF_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)

src = load_file(str(SRC))
assert len(src) == 244, len(src)

out: dict[str, torch.Tensor] = {}
dropped = 0
for name, t in src.items():
    if BUF_RE.match(name):
        dropped += 1
        continue
    if PROJ_RE.match(name):
        out[name] = t.to(torch.bfloat16).contiguous()
    else:
        out[name] = t.to(torch.float32).contiguous()

# Required checks (fail loudly before writing).
n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
assert n_bf16 == 64, f"expected 64 bf16 tensors, got {n_bf16}"
assert out["gpt_neox.layers.0.attention.query_key_value.weight"].dtype == torch.bfloat16
assert out["gpt_neox.embed_in.weight"].dtype == torch.float32
assert len(out) == 196, f"expected 196 tensors, got {len(out)}"
assert dropped == 48, dropped
assert all(t.dtype == torch.float32 for n, t in out.items() if not PROJ_RE.match(n))

# Shard: greedy in original key order; oversized tensors get their own shard.
shards: list[list[str]] = []
cur: list[str] = []
cur_size = 0
for name, t in out.items():
    size = t.numel() * t.element_size()
    if size > MAX_SHARD:
        if cur:
            shards.append(cur)
            cur, cur_size = [], 0
        shards.append([name])
        continue
    if cur_size + size > MAX_SHARD:
        shards.append(cur)
        cur, cur_size = [], 0
    cur.append(name)
    cur_size += size
if cur:
    shards.append(cur)

DST.mkdir(parents=True, exist_ok=True)
for old in DST.glob("model*.safetensors*"):
    old.unlink()
n = len(shards)
weight_map: dict[str, str] = {}
total = 0
for i, names in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    part = {k: out[k] for k in names}
    sz = sum(t.numel() * t.element_size() for t in part.values())
    assert sz <= MAX_SHARD or len(part) == 1, (fname, sz)
    total += sz
    save_file(part, str(DST / fname), metadata={"format": "pt"})
    for k in names:
        weight_map[k] = fname
assert len(weight_map) == 196
(DST / "model.safetensors.index.json").write_text(
    json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2, sort_keys=True)
)
print(f"wrote {n} shards, {len(weight_map)} tensors, {total} bytes to {DST}")
