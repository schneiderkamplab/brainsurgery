"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights and write a sharded
safetensors checkpoint. Uses only safetensors + torch (no model instantiation)."""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.abspath(os.path.join(ROOT, "..", ".."))
BASE = os.path.join(SANDBOX, "inputs", "base")
LORA = os.path.join(SANDBOX, "inputs", "lora")
OUT = os.path.join(SANDBOX, "out", "T5")
SHARD_BUDGET = 512 * 1024 * 1024  # bytes of tensor data per shard, headers excluded
STANDALONE = {"model.embed_tokens.weight", "lm_head.weight"}  # per TASK.md, each in its own shard
EXPECTED_PAIRS = 32
EXPECTED_TENSORS = 114


def fail(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


# --- adapter config ---------------------------------------------------------
cfg = json.load(open(os.path.join(LORA, "adapter_config.json")))
scale = cfg["lora_alpha"] / cfg["r"]
fan_in_fan_out = cfg.get("fan_in_fan_out", False)
print(f"scale = {scale}, fan_in_fan_out = {fan_in_fan_out}")

# --- load base (all 114 tensors, keep index order) --------------------------
index = json.load(open(os.path.join(BASE, "model.safetensors.index.json")))
order = list(index["weight_map"].keys())
state = {}
for shard in sorted(set(index["weight_map"].values())):
    with safe_open(os.path.join(BASE, shard), "pt") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
if set(state) != set(order) or len(state) != EXPECTED_TENSORS:
    fail(f"base has {len(state)} tensors, expected {EXPECTED_TENSORS}")

# --- load adapter, pair A/B factors ----------------------------------------
PREFIX = "base_model.model."
lora = {}
with safe_open(os.path.join(LORA, "adapter_model.safetensors"), "pt") as f:
    for k in f.keys():
        lora[k] = f.get_tensor(k)
pairs = {}
for k in lora:
    if not k.startswith(PREFIX):
        fail(f"unexpected adapter key {k}")
    rest = k[len(PREFIX):]
    for factor in ("lora_A", "lora_B"):
        suffix = f".{factor}.weight"
        if rest.endswith(suffix):
            pairs.setdefault(rest[: -len(suffix)] + ".weight", {})[factor] = lora[k]
            break
    else:
        fail(f"adapter key {k} is neither lora_A nor lora_B")

# --- merge -------------------------------------------------------------------
merged = 0
for base_name, ab in sorted(pairs.items()):
    if set(ab) != {"lora_A", "lora_B"}:
        fail(f"incomplete pair for {base_name}: {sorted(ab)}")
    if base_name not in state:
        fail(f"adapter targets missing base tensor {base_name}")
    A, B = ab["lora_A"].float(), ab["lora_B"].float()
    W = state[base_name]
    if W.dtype != torch.float32:
        fail(f"{base_name} dtype {W.dtype}, expected float32")
    delta = (B @ A) * scale
    if fan_in_fan_out:
        delta = delta.T
    if delta.shape != W.shape:
        fail(f"delta shape {tuple(delta.shape)} != weight shape {tuple(W.shape)} for {base_name}")
    state[base_name] = (W + delta).contiguous()
    merged += 1
print(f"merged {merged} adapter pairs")

# --- required checks ---------------------------------------------------------
if merged != EXPECTED_PAIRS:
    fail(f"merged {merged} pairs, expected {EXPECTED_PAIRS}")
if any("lora_" in k for k in state):
    fail("adapter tensor names present in output")
q0 = state["model.layers.0.self_attn.q_proj.weight"]
if tuple(q0.shape) != (2048, 2048) or q0.dtype != torch.float32:
    fail(f"layers.0 q_proj has shape {tuple(q0.shape)} dtype {q0.dtype}")
if len(state) != EXPECTED_TENSORS:
    fail(f"output has {len(state)} tensors, expected {EXPECTED_TENSORS}")
if set(state) != set(order):
    fail("output key set differs from base key set")

# --- shard plan --------------------------------------------------------------
def nbytes(t):
    return t.numel() * t.element_size()

shards, cur, cur_size = [], [], 0
for name in order:
    size = nbytes(state[name])
    if name in STANDALONE or size > SHARD_BUDGET:
        if cur:
            shards.append(cur)
        shards.append([name])
        cur, cur_size = [], 0
        continue
    if cur and cur_size + size > SHARD_BUDGET:
        shards.append(cur)
        cur, cur_size = [], 0
    cur.append(name)
    cur_size += size
if cur:
    shards.append(cur)
for s in shards:
    total = sum(nbytes(state[n]) for n in s)
    if total > SHARD_BUDGET and len(s) > 1:
        fail(f"shard exceeds budget: {total} bytes")

# --- write -------------------------------------------------------------------
n = len(shards)
weight_map = {}
for i, names in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    save_file({k: state[k] for k in names}, os.path.join(OUT, fname), metadata={"format": "pt"})
    for k in names:
        weight_map[k] = fname
    print(f"{fname}: {len(names)} tensors, {sum(nbytes(state[k]) for k in names)} bytes")
total_size = sum(nbytes(t) for t in state.values())
json.dump(
    {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    open(os.path.join(OUT, "model.safetensors.index.json"), "w"),
    indent=2,
)
for extra in ("config.json", "generation_config.json", "special_tokens_map.json",
              "tokenizer_config.json", "tokenizer.json"):
    src = os.path.join(BASE, extra)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, extra))

# --- verify written output ---------------------------------------------------
seen = set()
for fname in sorted(set(weight_map.values())):
    with safe_open(os.path.join(OUT, fname), "pt") as f:
        seen.update(f.keys())
if len(seen) != EXPECTED_TENSORS or seen != set(order):
    fail("written checkpoint does not contain exactly the expected tensors")
if any("lora_" in k for k in seen):
    fail("lora_ tensor written")
print(f"OK: {len(seen)} tensors in {n} shards -> {OUT}")
