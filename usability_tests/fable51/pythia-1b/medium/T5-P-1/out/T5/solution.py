"""T5: merge a PEFT LoRA adapter into Pythia-1B base weights and write a sharded checkpoint."""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
LORA = os.path.join(ROOT, "inputs", "lora", "adapter_model.safetensors")
CFG = os.path.join(ROOT, "inputs", "lora", "adapter_config.json")
OUT = os.path.join(ROOT, "out", "T5")
MAX_SHARD = 512 * 1024 * 1024
PREFIX = "base_model.model."
EXPECTED_PAIRS = 16
EXPECTED_TENSORS = 244


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


cfg = json.load(open(CFG))
scale = cfg["lora_alpha"] / cfg["r"]
fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))

# Load base into memory (float16 tensors), preserve original order.
state = {}
with safe_open(BASE, framework="pt") as f:
    for k in f.keys():
        state[k] = f.get_tensor(k)
if len(state) != EXPECTED_TENSORS:
    fail(f"base has {len(state)} tensors, expected {EXPECTED_TENSORS}")

# Load adapter and pair A/B by module.
lora = {}
with safe_open(LORA, framework="pt") as f:
    for k in f.keys():
        lora[k] = f.get_tensor(k)

pat = re.compile(r"^(?P<mod>.+)\.lora_(?P<ab>[AB])\.weight$")
pairs = {}
for k in lora:
    m = pat.match(k)
    if not m:
        fail(f"unrecognised adapter tensor name: {k}")
    pairs.setdefault(m.group("mod"), {})[m.group("ab")] = k
for mod, d in pairs.items():
    if set(d) != {"A", "B"}:
        fail(f"incomplete adapter pair for {mod}: {sorted(d)}")
if len(pairs) != EXPECTED_PAIRS:
    fail(f"found {len(pairs)} adapter pairs, expected {EXPECTED_PAIRS}")

merged = 0
for mod, d in pairs.items():
    if not mod.startswith(PREFIX):
        fail(f"adapter module without PEFT prefix: {mod}")
    base_name = mod[len(PREFIX):] + ".weight"
    if base_name not in state:
        fail(f"adapter target {base_name} not in base checkpoint")
    A = lora[d["A"]].float()
    B = lora[d["B"]].float()
    delta = scale * (B @ A)
    if fan_in_fan_out:
        delta = delta.T
    W = state[base_name]
    if delta.shape != W.shape:
        fail(f"delta shape {tuple(delta.shape)} != base shape {tuple(W.shape)} for {base_name}")
    state[base_name] = (W.float() + delta).to(W.dtype).contiguous()
    merged += 1
if merged != EXPECTED_PAIRS:
    fail(f"merged {merged} pairs, expected {EXPECTED_PAIRS}")

# Required checks before writing.
if any("lora_" in k for k in state):
    fail("adapter tensor name in output")
qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
if tuple(state[qkv0].shape) != (6144, 2048):
    fail(f"{qkv0} has shape {tuple(state[qkv0].shape)}")
if state[qkv0].dtype != torch.float16:
    fail(f"{qkv0} has dtype {state[qkv0].dtype}")
if len(state) != EXPECTED_TENSORS:
    fail(f"output has {len(state)} tensors, expected {EXPECTED_TENSORS}")

# Shard: greedy in base order, budget on tensor bytes only.
shards = []
cur, cur_bytes = [], 0
for k, t in state.items():
    nbytes = t.numel() * t.element_size()
    if cur and cur_bytes + nbytes > MAX_SHARD:
        shards.append(cur)
        cur, cur_bytes = [], 0
    cur.append(k)
    cur_bytes += nbytes
if cur:
    shards.append(cur)

os.makedirs(OUT, exist_ok=True)
n = len(shards)
weight_map = {}
total_size = 0
for i, names in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    path = os.path.join(OUT, fname)
    if os.path.exists(path):
        fail(f"destination exists: {path}")
    tensors = {k: state[k] for k in names}
    size = sum(t.numel() * t.element_size() for t in tensors.values())
    if size > MAX_SHARD:
        fail(f"shard {fname} exceeds budget: {size}")
    save_file(tensors, path, metadata={"format": "pt"})
    for k in names:
        weight_map[k] = fname
    total_size += size

if len(weight_map) != EXPECTED_TENSORS:
    fail(f"weight_map has {len(weight_map)} entries")
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

print(f"merged {merged} pairs (scale={scale}), wrote {n} shards, {len(weight_map)} tensors, {total_size} bytes")
