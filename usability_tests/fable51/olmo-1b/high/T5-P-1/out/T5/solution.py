"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights and write a sharded checkpoint."""

import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.abspath(os.path.join(ROOT, "..", ".."))
BASE_DIR = os.path.join(SANDBOX, "inputs", "base")
LORA_DIR = os.path.join(SANDBOX, "inputs", "lora")
OUT_DIR = os.path.join(SANDBOX, "out", "T5")

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912
# Tensors at least this large get a shard of their own (task: embed_tokens and
# lm_head, 412 MB each, are stored alone). Anything above MAX_SHARD_BYTES is
# necessarily alone; this lower bound keeps the two big matrices alone as well.
ALONE_MIN_BYTES = 256 * 1024 * 1024
EXPECTED_PAIRS = 32
EXPECTED_TENSORS = 114
DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1, "BOOL": 1}


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# --- adapter config ---------------------------------------------------------
with open(os.path.join(LORA_DIR, "adapter_config.json")) as f:
    cfg = json.load(f)
r, alpha = cfg["r"], cfg["lora_alpha"]
fan_in_fan_out = cfg.get("fan_in_fan_out", False)
scale = alpha / r
print(f"r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

# --- adapter tensors --------------------------------------------------------
lora_A, lora_B = {}, {}
pat = re.compile(r"^base_model\.model\.(.+)\.lora_([AB])\.weight$")
with safe_open(os.path.join(LORA_DIR, "adapter_model.safetensors"), framework="pt") as f:
    for k in f.keys():
        m = pat.match(k)
        if m is None:
            fail(f"unrecognized adapter tensor name: {k}")
        base_name = m.group(1) + ".weight"
        (lora_A if m.group(2) == "A" else lora_B)[base_name] = f.get_tensor(k)
if set(lora_A) != set(lora_B):
    fail(f"unpaired adapter tensors: {set(lora_A) ^ set(lora_B)}")
pairs = sorted(lora_A)
if len(pairs) != EXPECTED_PAIRS:
    fail(f"expected {EXPECTED_PAIRS} adapter pairs, found {len(pairs)}")

# --- base index -------------------------------------------------------------
with open(os.path.join(BASE_DIR, "model.safetensors.index.json")) as f:
    base_index = json.load(f)
base_map = base_index["weight_map"]
names = sorted(base_map)
if len(names) != EXPECTED_TENSORS:
    fail(f"base has {len(names)} tensors, expected {EXPECTED_TENSORS}")
missing = [n for n in pairs if n not in base_map]
if missing:
    fail(f"adapter targets not in base: {missing}")

# tensor sizes from headers, without loading data
sizes = {}
shapes = {}
by_shard = {}
for n, s in base_map.items():
    by_shard.setdefault(s, []).append(n)
for s, ns in by_shard.items():
    with safe_open(os.path.join(BASE_DIR, s), framework="pt") as f:
        for n in ns:
            sl = f.get_slice(n)
            shape = sl.get_shape()
            nbytes = DTYPE_BYTES[sl.get_dtype()]
            for d in shape:
                nbytes *= d
            sizes[n] = nbytes
            shapes[n] = shape

# --- shard assignment (HF-style greedy, sorted names) -----------------------
shards = [[]]
cur = 0
for n in names:
    alone = sizes[n] >= ALONE_MIN_BYTES
    if shards[-1] and (alone or cur + sizes[n] > MAX_SHARD_BYTES):
        shards.append([])
        cur = 0
    shards[-1].append(n)
    cur += sizes[n]
    if alone:
        shards.append([])
        cur = 0
shards = [s for s in shards if s]
num_shards = len(shards)
shard_files = [f"model-{i + 1:05d}-of-{num_shards:05d}.safetensors" for i in range(num_shards)]
for i, ns in enumerate(shards):
    total = sum(sizes[n] for n in ns)
    if total > MAX_SHARD_BYTES and len(ns) != 1:
        fail(f"shard {shard_files[i]} exceeds budget with {len(ns)} tensors")

# --- merge + write ----------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
for fn in shard_files + ["model.safetensors.index.json"]:
    if os.path.exists(os.path.join(OUT_DIR, fn)):
        fail(f"output already exists: {os.path.join(OUT_DIR, fn)}")

merged = 0
weight_map = {}
total_size = 0
open_files = {}
try:
    for s in by_shard:
        open_files[s] = safe_open(os.path.join(BASE_DIR, s), framework="pt").__enter__()
    for i, ns in enumerate(shards):
        out = {}
        for n in ns:
            t = open_files[base_map[n]].get_tensor(n)
            if n in lora_A:
                if t.dtype != torch.float32:
                    fail(f"{n} is {t.dtype}, expected float32")
                A, B = lora_A[n].float(), lora_B[n].float()
                delta = B @ A
                if fan_in_fan_out:
                    delta = delta.T
                if delta.shape != t.shape:
                    fail(f"{n}: delta shape {tuple(delta.shape)} != {tuple(t.shape)}")
                t = t + scale * delta
                merged += 1
            out[n] = t.contiguous()
            weight_map[n] = shard_files[i]
            total_size += t.numel() * t.element_size()
        # checks before writing this shard
        for n in out:
            if "lora_" in n:
                fail(f"adapter tensor in output: {n}")
        data_bytes = sum(v.numel() * v.element_size() for v in out.values())
        if data_bytes > MAX_SHARD_BYTES and len(out) != 1:
            fail(f"shard {shard_files[i]} would exceed budget")
        save_file(out, os.path.join(OUT_DIR, shard_files[i]), metadata={"format": "pt"})
        print(f"wrote {shard_files[i]}: {len(out)} tensors, {data_bytes} bytes")
        del out
finally:
    for f in open_files.values():
        f.__exit__(None, None, None)

# --- final checks -----------------------------------------------------------
if merged != EXPECTED_PAIRS:
    fail(f"merged {merged} pairs, expected {EXPECTED_PAIRS}")
if any("lora_" in n for n in weight_map):
    fail("lora_ tensor in output")
q0 = "model.layers.0.self_attn.q_proj.weight"
if list(shapes[q0]) != [2048, 2048]:
    fail(f"{q0} has shape {shapes[q0]}")
if len(weight_map) != EXPECTED_TENSORS:
    fail(f"output has {len(weight_map)} tensors, expected {EXPECTED_TENSORS}")
if set(weight_map) != set(names):
    fail("output key set differs from base")

# verify the written shards match the plan
seen = set()
for fn in shard_files:
    with safe_open(os.path.join(OUT_DIR, fn), framework="pt") as f:
        keys = list(f.keys())
        if f.get_slice(q0).get_shape() != [2048, 2048] if q0 in keys else False:
            fail(f"{q0} shape changed on disk")
        for k in keys:
            if weight_map[k] != fn:
                fail(f"{k} written to wrong shard")
        seen.update(keys)
if seen != set(names):
    fail("written tensors do not match expected key set")

with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)
print(f"OK: merged {merged} pairs, {len(weight_map)} tensors in {num_shards} shards, "
      f"total_size={total_size}")
