"""T5: merge a PEFT LoRA adapter into Pythia-1B base weights and write sharded safetensors."""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
LORA = os.path.join(ROOT, "inputs", "lora", "adapter_model.safetensors")
CFG = os.path.join(ROOT, "inputs", "lora", "adapter_config.json")
OUT = os.path.join(ROOT, "out", "T5")
SHARD_LIMIT = 512 * 1024 * 1024


def fail(msg):
    print("FAIL:", msg, file=sys.stderr)
    sys.exit(1)


cfg = json.load(open(CFG))
scale = cfg["lora_alpha"] / cfg["r"]
fan_in_fan_out = cfg.get("fan_in_fan_out", False)

with safe_open(BASE, "pt") as f:
    state = {k: f.get_tensor(k) for k in f.keys()}
with safe_open(LORA, "pt") as f:
    lora = {k: f.get_tensor(k) for k in f.keys()}

pat = re.compile(r"^base_model\.model\.(.*)\.lora_A\.weight$")
merged = 0
for ak in sorted(lora):
    m = pat.match(ak)
    if not m:
        continue
    prefix = m.group(1)
    bk = ak.replace(".lora_A.", ".lora_B.")
    target = prefix + ".weight"
    if bk not in lora:
        fail(f"missing lora_B for {ak}")
    if target not in state:
        fail(f"base tensor {target} not found for adapter {ak}")
    A = lora[ak].float()
    B = lora[bk].float()
    delta = scale * (B @ A)
    if fan_in_fan_out:
        delta = delta.T
    W = state[target]
    if delta.shape != W.shape:
        fail(f"shape mismatch for {target}: {tuple(delta.shape)} vs {tuple(W.shape)}")
    state[target] = (W.float() + delta).to(W.dtype).contiguous()
    merged += 1

if merged != 16:
    fail(f"expected 16 adapter pairs, merged {merged}")
if any("lora_" in k for k in state):
    fail("lora_ tensor present in output")
if tuple(state["gpt_neox.layers.0.attention.query_key_value.weight"].shape) != (6144, 2048):
    fail("layer 0 qkv weight has wrong shape")
if len(state) != 244:
    fail(f"expected 244 tensors, got {len(state)}")

# Sharding: keep base key order; the embedding matrices go alone in their own shard,
# everything else is packed greedily under the 512 MiB budget.
ALONE = {"gpt_neox.embed_in.weight", "embed_out.weight"}
shards, cur, cur_bytes = [], {}, 0
for k, t in state.items():
    nb = t.numel() * t.element_size()
    if k in ALONE or nb > SHARD_LIMIT:
        if cur:
            shards.append(cur)
        shards.append({k: t})
        cur, cur_bytes = {}, 0
        continue
    if cur and cur_bytes + nb > SHARD_LIMIT:
        shards.append(cur)
        cur, cur_bytes = {}, 0
    cur[k] = t
    cur_bytes += nb
if cur:
    shards.append(cur)
for shard in shards:
    if sum(t.numel() * t.element_size() for t in shard.values()) > SHARD_LIMIT:
        fail("shard exceeds 512 MiB")

os.makedirs(OUT, exist_ok=True)
n = len(shards)
weight_map = {}
total = 0
for i, shard in enumerate(shards, 1):
    name = f"model-{i:05d}-of-{n:05d}.safetensors"
    save_file(shard, os.path.join(OUT, name), metadata={"format": "pt"})
    for k, t in shard.items():
        weight_map[k] = name
        total += t.numel() * t.element_size()
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)

print(f"merged {merged} adapters, wrote {len(weight_map)} tensors in {n} shards to {OUT}")
