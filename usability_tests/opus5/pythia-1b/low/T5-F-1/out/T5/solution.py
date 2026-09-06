"""T5: fold a PEFT LoRA adapter into Pythia-1B base weights, write sharded safetensors."""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
LORA = os.path.join(ROOT, "inputs", "lora", "adapter_model.safetensors")
CFG = os.path.join(ROOT, "inputs", "lora", "adapter_config.json")
OUT = os.path.join(ROOT, "out", "T5")

SHARD_LIMIT = 512 * 1024 * 1024

cfg = json.load(open(CFG))
r, alpha, fan_in_fan_out = cfg["r"], cfg["lora_alpha"], cfg["fan_in_fan_out"]
scale = alpha / r

base = {}
with safe_open(BASE, framework="pt") as f:
    order = list(f.keys())
    for k in order:
        base[k] = f.get_tensor(k)

adapter = {}
with safe_open(LORA, framework="pt") as f:
    for k in f.keys():
        adapter[k] = f.get_tensor(k)

# Map adapter names -> base names: strip the PEFT "base_model.model." prefix
# and the ".lora_A/.lora_B" factor suffix.
PAT = re.compile(r"^base_model\.model\.(?P<mod>.+)\.lora_(?P<ab>[AB])\.weight$")
pairs = {}
for k in adapter:
    m = PAT.match(k)
    if not m:
        raise ValueError(f"unrecognised adapter tensor name: {k}")
    pairs.setdefault(m.group("mod"), {})[m.group("ab")] = k

merged = 0
for mod, ab in sorted(pairs.items()):
    if set(ab) != {"A", "B"}:
        raise ValueError(f"incomplete LoRA pair for {mod}: {sorted(ab)}")
    target = f"{mod}.weight"
    if target not in base:
        raise KeyError(f"adapter targets missing base tensor {target}")
    A = adapter[ab["A"]].to(torch.float32)
    B = adapter[ab["B"]].to(torch.float32)
    if A.shape[0] != r or B.shape[1] != r:
        raise ValueError(f"{mod}: factor rank mismatch A={tuple(A.shape)} B={tuple(B.shape)} r={r}")
    delta = scale * (B @ A)          # [out, in], nn.Linear layout
    if fan_in_fan_out:
        delta = delta.T
    W = base[target]
    if delta.shape != W.shape:
        raise ValueError(f"{mod}: delta {tuple(delta.shape)} != base {tuple(W.shape)}")
    base[target] = (W.to(torch.float32) + delta).to(W.dtype)
    merged += 1

# ---- required checks -------------------------------------------------------
assert merged == 16, f"expected 16 adapter pairs merged, got {merged}"
bad = [k for k in base if "lora_" in k]
assert not bad, f"adapter tensors leaked into the output: {bad}"
qkv = "gpt_neox.layers.0.attention.query_key_value.weight"
assert tuple(base[qkv].shape) == (6144, 2048), f"{qkv} has shape {tuple(base[qkv].shape)}"
assert len(base) == 244, f"expected 244 output tensors, got {len(base)}"
assert base[qkv].dtype == torch.float16, base[qkv].dtype

# ---- shard --------------------------------------------------------------
def nbytes(t):
    return t.numel() * t.element_size()

shards, cur, cur_size = [], [], 0
for k in order:
    n = nbytes(base[k])
    if cur and cur_size + n > SHARD_LIMIT:
        shards.append(cur)
        cur, cur_size = [], 0
    cur.append(k)
    cur_size += n
if cur:
    shards.append(cur)

if os.path.isdir(OUT):
    for name in os.listdir(OUT):
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            os.remove(os.path.join(OUT, name))
os.makedirs(OUT, exist_ok=True)

total = len(shards)
weight_map, total_size = {}, 0
for i, keys in enumerate(shards, start=1):
    fname = f"model-{i:05d}-of-{total:05d}.safetensors"
    save_file({k: base[k].contiguous() for k in keys}, os.path.join(OUT, fname))
    for k in keys:
        weight_map[k] = fname
        total_size += nbytes(base[k])

with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

# ---- post-write verification ----------------------------------------------
seen = {}
for fname in sorted(set(weight_map.values())):
    path = os.path.join(OUT, fname)
    size = 0
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            seen[k] = t
            size += nbytes(t)
    if size > SHARD_LIMIT:
        with safe_open(path, framework="pt") as f:
            assert len(f.keys()) == 1, f"{fname} holds {size} B over budget in >1 tensor"

assert len(seen) == 244, f"round-trip found {len(seen)} tensors"
assert set(seen) == set(base), "round-trip key set differs"
assert not [k for k in seen if "lora_" in k]
print(f"merged {merged} LoRA pairs (scale={scale}); wrote {total} shards, {len(seen)} tensors to {OUT}")
