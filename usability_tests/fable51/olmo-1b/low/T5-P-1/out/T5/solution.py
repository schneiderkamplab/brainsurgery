"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights and write a sharded checkpoint."""
import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "inputs/base"
LORA = "inputs/lora"
OUT = "out/T5"
SHARD_BUDGET = 512 * 1024 * 1024  # bytes of tensor data per shard

# --- load base ---------------------------------------------------------------
index = json.load(open(os.path.join(BASE, "model.safetensors.index.json")))
base = {}
for shard in sorted(set(index["weight_map"].values())):
    with safe_open(os.path.join(BASE, shard), "pt") as f:
        for k in f.keys():
            base[k] = f.get_tensor(k)
assert len(base) == 114, f"expected 114 base tensors, got {len(base)}"

# --- load adapter ------------------------------------------------------------
cfg = json.load(open(os.path.join(LORA, "adapter_config.json")))
scale = cfg["lora_alpha"] / cfg["r"]
assert cfg["fan_in_fan_out"] is False, "unexpected fan_in_fan_out layout"
lora = {}
with safe_open(os.path.join(LORA, "adapter_model.safetensors"), "pt") as f:
    for k in f.keys():
        lora[k] = f.get_tensor(k)

pat = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")
merged = 0
for k in sorted(lora):
    m = pat.match(k)
    if not m:
        continue
    target = m.group(1) + ".weight"
    a = lora[k]
    b = lora["base_model.model." + m.group(1) + ".lora_B.weight"]
    assert target in base, f"adapter target missing in base: {target}"
    w = base[target]
    assert w.dtype == torch.float32 and a.dtype == torch.float32 and b.dtype == torch.float32
    delta = scale * (b @ a)
    assert delta.shape == w.shape, f"{target}: delta {delta.shape} vs base {w.shape}"
    base[target] = (w + delta).contiguous()
    merged += 1

# --- required checks ---------------------------------------------------------
assert merged == 32, f"expected 32 merged adapter pairs, got {merged}"
assert not any("lora_" in k for k in base), "adapter tensor leaked into output"
assert tuple(base["model.layers.0.self_attn.q_proj.weight"].shape) == (2048, 2048)
assert len(base) == 114, f"expected 114 output tensors, got {len(base)}"

# --- shard and write ---------------------------------------------------------
shards, cur, cur_size = [], {}, 0
for k in sorted(base):
    t = base[k]
    n = t.numel() * t.element_size()
    if cur and cur_size + n > SHARD_BUDGET:
        shards.append(cur)
        cur, cur_size = {}, 0
    cur[k] = t
    cur_size += n
if cur:
    shards.append(cur)

os.makedirs(OUT, exist_ok=True)
weight_map = {}
total = 0
for i, shard in enumerate(shards, 1):
    name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
    save_file(shard, os.path.join(OUT, name), metadata={"format": "pt"})
    for k, t in shard.items():
        weight_map[k] = name
        total += t.numel() * t.element_size()
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
print(f"merged {merged} pairs, wrote {len(weight_map)} tensors in {len(shards)} shards")
