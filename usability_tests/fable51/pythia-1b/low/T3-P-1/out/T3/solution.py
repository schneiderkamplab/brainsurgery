import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "inputs" / "base" / "model.safetensors"
OUT = ROOT / "out" / "T3"
MAX_SHARD = 268_435_456

PROJ = re.compile(
    r"^gpt_neox\.layers\.\d+\."
    r"(attention\.query_key_value|attention\.dense|mlp\.dense_h_to_4h|mlp\.dense_4h_to_h)\.weight$"
)
BUF = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)

sd = load_file(str(SRC))
assert len(sd) == 244, len(sd)

out = {}
for name, t in sd.items():
    if BUF.match(name):
        continue
    if PROJ.match(name):
        out[name] = t.to(torch.bfloat16).contiguous()
    else:
        out[name] = t.to(torch.float32).contiguous()

# Required checks
n_bf16 = sum(t.dtype == torch.bfloat16 for t in out.values())
assert n_bf16 == 64, f"expected 64 bfloat16 tensors, got {n_bf16}"
assert out["gpt_neox.layers.0.attention.query_key_value.weight"].dtype == torch.bfloat16
assert out["gpt_neox.embed_in.weight"].dtype == torch.float32
assert len(out) == 196, f"expected 196 tensors, got {len(out)}"
assert all(t.dtype == torch.float32 for n, t in out.items() if not PROJ.match(n))
assert set(sd) - set(out) == {n for n in sd if BUF.match(n)}
assert len(set(sd) - set(out)) == 48

# Shard greedily in input order
shards = [[]]
sizes = [0]
for name, t in out.items():
    nbytes = t.numel() * t.element_size()
    if sizes[-1] + nbytes > MAX_SHARD and shards[-1]:
        shards.append([])
        sizes.append(0)
    shards[-1].append(name)
    sizes[-1] += nbytes
assert all(s <= MAX_SHARD or len(names) == 1 for s, names in zip(sizes, shards))

OUT.mkdir(parents=True, exist_ok=True)
for p in OUT.glob("model-*.safetensors"):
    p.unlink()
n = len(shards)
weight_map = {}
total = 0
for i, names in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    save_file({k: out[k] for k in names}, str(OUT / fname), metadata={"format": "pt"})
    for k in names:
        weight_map[k] = fname
    total += sizes[i - 1]
index = {"metadata": {"total_size": total}, "weight_map": weight_map}
(OUT / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

# Verify round trip
back = {}
for fname in sorted(set(weight_map.values())):
    back.update(load_file(str(OUT / fname)))
assert set(back) == set(out) and len(back) == 196
for k in out:
    assert back[k].dtype == out[k].dtype and torch.equal(back[k], out[k]), k
print(f"wrote {n} shards, {len(out)} tensors, {total} bytes")
