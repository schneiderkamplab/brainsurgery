"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and write a sharded checkpoint."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
LORA = os.path.join(ROOT, "inputs", "lora", "adapter_model.safetensors")
CFG = os.path.join(ROOT, "inputs", "lora", "adapter_config.json")
OUT = os.path.join(ROOT, "out", "T5")
MAX_SHARD = 104_857_600  # 100 MiB of tensor data per shard


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


base = load_file(BASE)
lora = load_file(LORA)
cfg = json.load(open(CFG))
scale = cfg["lora_alpha"] / cfg["r"]
fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
print(f"base tensors: {len(base)}, adapter tensors: {len(lora)}, scale={scale}, "
      f"fan_in_fan_out={fan_in_fan_out}")

pat = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")
merged = 0
for a_name in sorted(lora):
    m = pat.match(a_name)
    if not m:
        continue
    module = m.group(1)
    b_name = f"base_model.model.{module}.lora_B.weight"
    w_name = f"{module}.weight"
    if b_name not in lora:
        fail(f"missing lora_B for {a_name}")
    if w_name not in base:
        fail(f"base tensor {w_name} not found for adapter {a_name}")
    A = lora[a_name].float()
    B = lora[b_name].float()
    delta = scale * (B @ A)  # [out, in], nn.Linear convention
    if fan_in_fan_out:
        delta = delta.T  # base is Conv1D [in, out]
    W = base[w_name]
    if W.dtype != torch.float32:
        fail(f"{w_name} dtype {W.dtype} != float32")
    if delta.shape != W.shape:
        fail(f"delta shape {tuple(delta.shape)} != base shape {tuple(W.shape)} for {w_name}")
    base[w_name] = (W + delta).contiguous()
    merged += 1

# unmatched adapter tensors are an error too
for k in lora:
    if not pat.match(k) and not k.endswith(".lora_B.weight"):
        fail(f"unrecognised adapter tensor {k}")

# --- required checks ---
if merged != 12:
    fail(f"expected 12 merged adapter pairs, got {merged}")
if any("lora_" in k for k in base):
    fail("adapter tensor name leaked into output")
if tuple(base["h.0.attn.c_attn.weight"].shape) != (768, 2304):
    fail(f"h.0.attn.c_attn.weight shape {tuple(base['h.0.attn.c_attn.weight'].shape)}")
if len(base) != 160:
    fail(f"expected 160 output tensors, got {len(base)}")
print(f"merged {merged} adapter pairs; checks passed")

# --- shard: greedy in name order, oversized tensors alone ---
shards = []  # list of (names, size)
cur, cur_size = [], 0
for name in sorted(base):
    t = base[name]
    size = t.numel() * t.element_size()
    if cur and cur_size + size > MAX_SHARD:
        shards.append(cur)
        cur, cur_size = [], 0
    cur.append(name)
    cur_size += size
if cur:
    shards.append(cur)

os.makedirs(OUT, exist_ok=True)
n = len(shards)
weight_map = {}
total = 0
for i, names in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    sd = {k: base[k].contiguous() for k in names}
    ssize = sum(v.numel() * v.element_size() for v in sd.values())
    if ssize > MAX_SHARD and len(sd) > 1:
        fail(f"shard {fname} exceeds budget with {len(sd)} tensors")
    save_file(sd, os.path.join(OUT, fname), metadata={"format": "pt"})
    for k in names:
        weight_map[k] = fname
    total += ssize
    print(f"{fname}: {len(names)} tensors, {ssize} bytes")

index = {"metadata": {"total_size": total}, "weight_map": weight_map}
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2, sort_keys=True)
if len(weight_map) != 160:
    fail(f"index has {len(weight_map)} entries")
print(f"wrote {n} shards + index to {OUT}")
