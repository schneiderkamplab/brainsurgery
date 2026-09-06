"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights, write sharded."""
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = os.path.join(ROOT, "inputs", "base")
LORA = os.path.join(ROOT, "inputs", "lora")
OUT = os.path.join(ROOT, "out", "T5")
MAX_SHARD = 512 * 1024 * 1024


def load_dir(index_path):
    index = json.load(open(index_path))
    order = list(index["weight_map"])
    tensors = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(os.path.join(BASE, shard), framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    return tensors, order


base, order = load_dir(os.path.join(BASE, "model.safetensors.index.json"))

adapter = {}
with safe_open(os.path.join(LORA, "adapter_model.safetensors"), framework="pt") as f:
    for k in f.keys():
        adapter[k] = f.get_tensor(k)

cfg = json.load(open(os.path.join(LORA, "adapter_config.json")))
scale = cfg["lora_alpha"] / cfg["r"]
assert not cfg["fan_in_fan_out"], "fan_in_fan_out=True is not handled"

# Pair up lora_A/lora_B by their base target name.
pairs = {}
for name in adapter:
    if ".lora_A.weight" in name:
        stem, side = name.split(".lora_A.weight")[0], "A"
    elif ".lora_B.weight" in name:
        stem, side = name.split(".lora_B.weight")[0], "B"
    else:
        raise SystemExit(f"unexpected adapter tensor: {name}")
    target = stem[len("base_model.model.") :] + ".weight"
    pairs.setdefault(target, {})[side] = adapter[name]

merged = 0
for target, ab in sorted(pairs.items()):
    if set(ab) != {"A", "B"}:
        raise SystemExit(f"incomplete adapter pair for {target}: {sorted(ab)}")
    if target not in base:
        raise SystemExit(f"adapter target missing from base: {target}")
    A, B = ab["A"].float(), ab["B"].float()
    W = base[target]
    delta = scale * (B @ A)
    if delta.shape != W.shape:
        raise SystemExit(f"delta shape {tuple(delta.shape)} != {tuple(W.shape)} for {target}")
    base[target] = (W.float() + delta).to(W.dtype)
    merged += 1

# Required checks.
if merged != 32:
    raise SystemExit(f"expected 32 merged adapter pairs, got {merged}")
bad = [k for k in base if "lora_" in k]
if bad:
    raise SystemExit(f"adapter tensors leaked into output: {bad}")
q0 = base["model.layers.0.self_attn.q_proj.weight"]
if tuple(q0.shape) != (2048, 2048):
    raise SystemExit(f"q_proj shape changed: {tuple(q0.shape)}")
if len(base) != 114:
    raise SystemExit(f"expected 114 output tensors, got {len(base)}")
if set(order) != set(base):
    raise SystemExit("output key set differs from base key set")

# Greedy sharding: a tensor larger than the limit gets its own shard.
shards, cur, cur_size = [], {}, 0
for name in order:
    t = base[name]
    n = t.numel() * t.element_size()
    if cur and cur_size + n > MAX_SHARD:
        shards.append(cur)
        cur, cur_size = {}, 0
    cur[name] = t
    cur_size += n
if cur:
    shards.append(cur)

total = len(shards)
weight_map, total_size = {}, 0
os.makedirs(OUT, exist_ok=True)
for i, shard in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{total:05d}.safetensors"
    size = sum(t.numel() * t.element_size() for t in shard.values())
    if size > MAX_SHARD and len(shard) > 1:
        raise SystemExit(f"shard {fname} is {size} bytes over the limit with {len(shard)} tensors")
    total_size += size
    save_file({k: v.contiguous() for k, v in shard.items()}, os.path.join(OUT, fname))
    for k in shard:
        weight_map[k] = fname

with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

print(f"merged {merged} adapter pairs (scale={scale}) into {len(base)} tensors")
print(f"wrote {total} shards, total_size={total_size}")
