"""T5: merge a PEFT LoRA adapter into GPT-2 Conv1D weights and write a sharded safetensors checkpoint."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
LORA_DIR = os.path.join(ROOT, "inputs", "lora")
OUT = os.path.join(ROOT, "out", "T5")
SHARD_BUDGET = 100 * 1024 * 1024
EXPECTED_PAIRS = 12
EXPECTED_TENSORS = 160


def fail(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


base = load_file(BASE)
adapter = load_file(os.path.join(LORA_DIR, "adapter_model.safetensors"))
with open(os.path.join(LORA_DIR, "adapter_config.json")) as f:
    cfg = json.load(f)
scale = cfg["lora_alpha"] / cfg["r"]
fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))

pat = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")
merged = 0
for a_name, A in adapter.items():
    m = pat.match(a_name)
    if not m:
        continue
    module = m.group(1)
    b_name = f"base_model.model.{module}.lora_B.weight"
    if b_name not in adapter:
        fail(f"missing lora_B for {a_name}")
    B = adapter[b_name]
    target = f"{module}.weight"
    if target not in base:
        fail(f"adapter target {target} not in base")
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        fail(f"adapter factors for {module} are not float32")
    delta = scale * (B.float() @ A.float())  # [out, in]
    if fan_in_fan_out:
        delta = delta.T  # Conv1D base layout [in, out]
    W = base[target]
    if W.shape != delta.shape:
        fail(f"shape mismatch for {target}: {tuple(W.shape)} vs {tuple(delta.shape)}")
    base[target] = (W.float() + delta).contiguous()
    merged += 1

# leftover adapter tensors that are not part of an A/B pair
unpaired = [k for k in adapter if "lora_A" not in k and "lora_B" not in k]
if unpaired:
    fail(f"unrecognised adapter tensors: {unpaired}")
if merged != EXPECTED_PAIRS:
    fail(f"expected {EXPECTED_PAIRS} merged pairs, got {merged}")
if any("lora_" in k for k in base):
    fail("lora_ tensor in output")
if tuple(base["h.0.attn.c_attn.weight"].shape) != (768, 2304):
    fail(f"h.0.attn.c_attn.weight has shape {tuple(base['h.0.attn.c_attn.weight'].shape)}")
if base["h.0.attn.c_attn.weight"].dtype != torch.float32:
    fail("merged weight is not float32")
if len(base) != EXPECTED_TENSORS:
    fail(f"expected {EXPECTED_TENSORS} tensors, got {len(base)}")

# shard: greedy by insertion order, budget on tensor bytes only
shards, cur, cur_bytes = [], {}, 0
for name, t in base.items():
    nbytes = t.numel() * t.element_size()
    if cur and cur_bytes + nbytes > SHARD_BUDGET:
        shards.append(cur)
        cur, cur_bytes = {}, 0
    cur[name] = t
    cur_bytes += nbytes
if cur:
    shards.append(cur)

os.makedirs(OUT, exist_ok=True)
n = len(shards)
weight_map, total = {}, 0
for i, shard in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    save_file(shard, os.path.join(OUT, fname), metadata={"format": "pt"})
    for k, t in shard.items():
        weight_map[k] = fname
        total += t.numel() * t.element_size()
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
if len(weight_map) != EXPECTED_TENSORS:
    fail(f"index has {len(weight_map)} entries")
print(f"merged {merged} pairs, wrote {n} shards, {len(weight_map)} tensors")
