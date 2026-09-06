"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights and write a sharded checkpoint."""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "inputs/base"
LORA = "inputs/lora"
OUT = "out/T5"
SHARD_BUDGET = 512 * 1024 * 1024  # bytes of tensor data per shard
EXPECTED_PAIRS = 32
EXPECTED_TENSORS = 114
PREFIX = "base_model.model."


def fail(msg):
    print("FAIL:", msg, file=sys.stderr)
    sys.exit(1)


# ---- load base ----
index = json.load(open(os.path.join(BASE, "model.safetensors.index.json")))
state = {}
for shard in sorted(set(index["weight_map"].values())):
    with safe_open(os.path.join(BASE, shard), framework="pt") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
if len(state) != EXPECTED_TENSORS:
    fail(f"base has {len(state)} tensors, expected {EXPECTED_TENSORS}")

# ---- load adapter ----
cfg = json.load(open(os.path.join(LORA, "adapter_config.json")))
scale = cfg["lora_alpha"] / cfg["r"]
fan_in_fan_out = cfg.get("fan_in_fan_out", False)
adapter = {}
with safe_open(os.path.join(LORA, "adapter_model.safetensors"), framework="pt") as f:
    for k in f.keys():
        adapter[k] = f.get_tensor(k)

# ---- merge ----
merged = 0
for a_key, A in adapter.items():
    if ".lora_A." not in a_key:
        continue
    b_key = a_key.replace(".lora_A.", ".lora_B.")
    if b_key not in adapter:
        fail(f"missing lora_B for {a_key}")
    B = adapter[b_key]
    if not a_key.startswith(PREFIX):
        fail(f"unexpected adapter key prefix: {a_key}")
    base_key = a_key[len(PREFIX):].replace(".lora_A.weight", ".weight")
    if base_key not in state:
        fail(f"adapter targets missing base tensor {base_key}")
    W = state[base_key]
    delta = (B.float() @ A.float()) * scale
    if fan_in_fan_out:
        delta = delta.T
    if delta.shape != W.shape:
        fail(f"delta shape {tuple(delta.shape)} != base {tuple(W.shape)} for {base_key}")
    if W.dtype != torch.float32:
        fail(f"{base_key} dtype {W.dtype}, expected float32")
    state[base_key] = (W + delta).contiguous()
    merged += 1

# ---- required checks ----
if merged != EXPECTED_PAIRS:
    fail(f"merged {merged} adapter pairs, expected {EXPECTED_PAIRS}")
if any("lora_" in k for k in state):
    fail("lora_ tensor present in output")
q0 = state["model.layers.0.self_attn.q_proj.weight"]
if tuple(q0.shape) != (2048, 2048):
    fail(f"q_proj layer 0 shape {tuple(q0.shape)}")
if len(state) != EXPECTED_TENSORS:
    fail(f"output has {len(state)} tensors, expected {EXPECTED_TENSORS}")

# ---- shard ----
def nbytes(t):
    return t.numel() * t.element_size()

shards, cur, cur_size = [], {}, 0
for k in state:  # keep base order
    t = state[k]
    if cur and cur_size + nbytes(t) > SHARD_BUDGET:
        shards.append(cur)
        cur, cur_size = {}, 0
    cur[k] = t
    cur_size += nbytes(t)
if cur:
    shards.append(cur)

os.makedirs(OUT, exist_ok=True)
n = len(shards)
weight_map, total = {}, 0
for i, sd in enumerate(shards, 1):
    name = f"model-{i:05d}-of-{n:05d}.safetensors"
    size = sum(nbytes(t) for t in sd.values())
    if size > SHARD_BUDGET:
        fail(f"shard {name} holds {size} bytes of tensor data > budget")
    save_file(sd, os.path.join(OUT, name), metadata={"format": "pt"})
    for k in sd:
        weight_map[k] = name
    total += size
json.dump(
    {"metadata": {"total_size": total}, "weight_map": weight_map},
    open(os.path.join(OUT, "model.safetensors.index.json"), "w"),
    indent=2,
)
print(f"OK: merged {merged} pairs, wrote {len(weight_map)} tensors in {n} shards to {OUT}")
