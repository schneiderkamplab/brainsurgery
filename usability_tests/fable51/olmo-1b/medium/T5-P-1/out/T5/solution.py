"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights and write a sharded checkpoint."""
import json
import os
import sys
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.abspath(os.path.join(ROOT, "..", ".."))
BASE_DIR = os.path.join(SANDBOX, "inputs", "base")
LORA_DIR = os.path.join(SANDBOX, "inputs", "lora")
OUT_DIR = os.path.join(SANDBOX, "out", "T5")
MAX_SHARD_BYTES = 512 * 1024 * 1024
PEFT_PREFIX = "base_model.model."


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# ---- load base ---------------------------------------------------------------
with open(os.path.join(BASE_DIR, "model.safetensors.index.json")) as f:
    base_index = json.load(f)
base_names = sorted(base_index["weight_map"])
state = OrderedDict()
for shard in sorted(set(base_index["weight_map"].values())):
    state.update(load_file(os.path.join(BASE_DIR, shard)))
if set(state) != set(base_names):
    fail("base shards do not match base index")
print(f"loaded base: {len(state)} tensors")

# ---- load adapter ------------------------------------------------------------
with open(os.path.join(LORA_DIR, "adapter_config.json")) as f:
    cfg = json.load(f)
r, alpha = cfg["r"], cfg["lora_alpha"]
fan_in_fan_out = cfg.get("fan_in_fan_out", False)
scale = alpha / r
print(f"adapter: r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")
adapter = load_file(os.path.join(LORA_DIR, "adapter_model.safetensors"))

pairs = {}
for name, t in adapter.items():
    if ".lora_A." in name:
        key, kind = name.replace(".lora_A.", "."), "A"
    elif ".lora_B." in name:
        key, kind = name.replace(".lora_B.", "."), "B"
    else:
        fail(f"unexpected adapter tensor {name}")
    if not key.startswith(PEFT_PREFIX):
        fail(f"adapter name without PEFT prefix: {name}")
    pairs.setdefault(key[len(PEFT_PREFIX):], {})[kind] = t

# ---- merge -------------------------------------------------------------------
merged = 0
for base_name, ab in sorted(pairs.items()):
    if set(ab) != {"A", "B"}:
        fail(f"incomplete LoRA pair for {base_name}: {sorted(ab)}")
    if base_name not in state:
        fail(f"adapter target {base_name} not in base")
    A, B = ab["A"].float(), ab["B"].float()
    W = state[base_name]
    if W.dtype != torch.float32:
        fail(f"{base_name} dtype {W.dtype}, expected float32")
    if A.shape[0] != r or B.shape[1] != r:
        fail(f"rank mismatch for {base_name}: A{tuple(A.shape)} B{tuple(B.shape)}")
    delta = scale * (B @ A)
    if fan_in_fan_out:
        delta = delta.T
    if delta.shape != W.shape:
        fail(f"delta shape {tuple(delta.shape)} != weight shape {tuple(W.shape)} for {base_name}")
    state[base_name] = (W + delta).contiguous()
    merged += 1
print(f"merged {merged} adapter pairs")

# ---- required checks ---------------------------------------------------------
if merged != 32:
    fail(f"expected 32 merged adapter pairs, got {merged}")
lora_names = [n for n in state if "lora_" in n]
if lora_names:
    fail(f"adapter tensors in output: {lora_names[:5]}")
q0 = state["model.layers.0.self_attn.q_proj.weight"]
if tuple(q0.shape) != (2048, 2048):
    fail(f"q_proj layer 0 shape {tuple(q0.shape)}")
if q0.dtype != torch.float32:
    fail(f"q_proj layer 0 dtype {q0.dtype}")
if len(state) != 114:
    fail(f"expected 114 output tensors, got {len(state)}")
if set(state) != set(base_names):
    fail("output key set differs from base")

# ---- shard -------------------------------------------------------------------
shards = []  # list of (names, bytes)
cur, cur_bytes = [], 0
for name in base_names:
    t = state[name]
    nbytes = t.numel() * t.element_size()
    # Tensors that dominate the budget (the 412 MB embedding and lm_head) go alone,
    # as the task specifies; anything else is packed greedily under the cap.
    if nbytes > MAX_SHARD_BYTES // 2:
        if cur:
            shards.append((cur, cur_bytes))
            cur, cur_bytes = [], 0
        shards.append(([name], nbytes))
        continue
    if cur_bytes + nbytes > MAX_SHARD_BYTES:
        shards.append((cur, cur_bytes))
        cur, cur_bytes = [], 0
    cur.append(name)
    cur_bytes += nbytes
if cur:
    shards.append((cur, cur_bytes))

n = len(shards)
os.makedirs(OUT_DIR, exist_ok=True)
weight_map = {}
total = 0
for i, (names, nbytes) in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    path = os.path.join(OUT_DIR, fname)
    if os.path.exists(path):
        fail(f"output shard already exists: {path}")
    save_file({k: state[k].contiguous() for k in names}, path, metadata={"format": "pt"})
    for k in names:
        weight_map[k] = fname
    total += nbytes
    print(f"wrote {fname}: {len(names)} tensors, {nbytes} bytes")
if len(weight_map) != 114:
    fail(f"weight_map has {len(weight_map)} entries")
with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2, sort_keys=True)
print(f"done: {n} shards, {total} bytes, index written")
