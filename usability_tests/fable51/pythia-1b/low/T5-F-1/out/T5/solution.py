"""T5: merge a PEFT LoRA adapter into Pythia-1B base weights and write sharded safetensors."""
import json
import os
import re

import torch
from safetensors.torch import load_file, save_file

BASE = "inputs/base/model.safetensors"
LORA = "inputs/lora/adapter_model.safetensors"
CFG = "inputs/lora/adapter_config.json"
OUT = "out/T5"
MAX_SHARD = 512 * 1024 * 1024

cfg = json.load(open(CFG))
scale = cfg["lora_alpha"] / cfg["r"]
assert not cfg["fan_in_fan_out"], "fan_in_fan_out=true not handled"

base = load_file(BASE)
lora = load_file(LORA)
assert len(base) == 244, f"expected 244 base tensors, got {len(base)}"

pat = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")
merged = 0
for a_name, A in lora.items():
    m = pat.match(a_name)
    if not m:
        assert ".lora_B." in a_name, f"unexpected adapter tensor {a_name}"
        continue
    b_name = a_name.replace(".lora_A.", ".lora_B.")
    target = m.group(1) + ".weight"
    assert b_name in lora, f"missing {b_name}"
    assert target in base, f"no base tensor {target}"
    B = lora[b_name]
    W = base[target]
    assert A.shape[0] == cfg["r"] and B.shape[1] == cfg["r"], (A.shape, B.shape)
    delta = scale * (B.float() @ A.float())
    assert delta.shape == W.shape, (delta.shape, W.shape)
    base[target] = (W.float() + delta).to(W.dtype).contiguous()
    merged += 1

# Required checks
assert merged == 16, f"merged {merged} adapter pairs, expected 16"
assert not any("lora_" in k for k in base), "lora_ tensor in output"
assert tuple(base["gpt_neox.layers.0.attention.query_key_value.weight"].shape) == (6144, 2048)
assert base["gpt_neox.layers.0.attention.query_key_value.weight"].dtype == torch.float16
assert len(base) == 244, f"output has {len(base)} tensors"

# Shard by tensor-data size (greedy, in base key order).
shards, cur, cur_size = [], {}, 0
for k, t in base.items():
    n = t.numel() * t.element_size()
    if cur and cur_size + n > MAX_SHARD:
        shards.append(cur)
        cur, cur_size = {}, 0
    cur[k] = t
    cur_size += n
if cur:
    shards.append(cur)

os.makedirs(OUT, exist_ok=True)
weight_map, total = {}, 0
for i, sh in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
    size = sum(t.numel() * t.element_size() for t in sh.values())
    assert size <= MAX_SHARD, (fname, size)
    save_file(sh, os.path.join(OUT, fname), metadata={"format": "pt"})
    for k in sh:
        weight_map[k] = fname
    total += size
assert len(weight_map) == 244
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
print(f"merged={merged} shards={len(shards)} tensors={len(weight_map)} bytes={total}")
