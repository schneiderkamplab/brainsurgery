#!/usr/bin/env python
"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights and export sharded.

Reads the sharded base under inputs/base/, the adapter under inputs/lora/,
folds  W += (alpha / r) * B @ A  for every adapted module, and writes a plain
dense sharded safetensors checkpoint to out/T5/.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_DIR = "inputs/base"
LORA_FILE = "inputs/lora/adapter_model.safetensors"
LORA_CONFIG = "inputs/lora/adapter_config.json"
OUT_DIR = "out/T5"

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912
LORA_PREFIX = "base_model.model."
EXPECTED_PAIRS = 32
EXPECTED_TENSORS = 114


def die(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


# ---------------------------------------------------------------- base index

with open(os.path.join(BASE_DIR, "model.safetensors.index.json")) as f:
    base_index = json.load(f)
weight_map = base_index["weight_map"]

# name -> (shard file, shape, dtype), in a deterministic (sorted) order.
base_meta: "OrderedDict[str, tuple[str, tuple[int, ...], str]]" = OrderedDict()
handles: dict[str, object] = {}
for name in sorted(weight_map):
    shard = weight_map[name]
    path = os.path.join(BASE_DIR, shard)
    if path not in handles:
        handles[path] = safe_open(path, framework="pt", device="cpu")
    sl = handles[path].get_slice(name)
    base_meta[name] = (path, tuple(sl.get_shape()), str(sl.get_dtype()))

if len(base_meta) != EXPECTED_TENSORS:
    die(f"base has {len(base_meta)} tensors, expected {EXPECTED_TENSORS}")

# ------------------------------------------------------------- adapter pairs

with open(LORA_CONFIG) as f:
    cfg = json.load(f)
r = int(cfg["r"])
alpha = float(cfg["lora_alpha"])
scale = alpha / r
if cfg.get("fan_in_fan_out", False):
    die("fan_in_fan_out=true is not handled by this script")

lora = safe_open(LORA_FILE, framework="pt", device="cpu")
lora_keys = list(lora.keys())

pairs: dict[str, dict[str, str]] = {}
for key in lora_keys:
    if ".lora_A.weight" in key:
        role, stem = "A", key[: -len(".lora_A.weight")]
    elif ".lora_B.weight" in key:
        role, stem = "B", key[: -len(".lora_B.weight")]
    else:
        die(f"unrecognised adapter tensor {key!r}")
    if not stem.startswith(LORA_PREFIX):
        die(f"adapter key {key!r} does not start with {LORA_PREFIX!r}")
    base_name = stem[len(LORA_PREFIX) :] + ".weight"
    slot = pairs.setdefault(base_name, {})
    if role in slot:
        die(f"duplicate lora_{role} for {base_name!r}")
    slot[role] = key

for base_name, slot in pairs.items():
    if set(slot) != {"A", "B"}:
        die(f"incomplete adapter pair for {base_name!r}: {sorted(slot)}")
    if base_name not in base_meta:
        die(f"adapter targets {base_name!r} which is not in the base checkpoint")

# Check 1: exactly 32 adapter pairs.
if len(pairs) != EXPECTED_PAIRS:
    die(f"found {len(pairs)} adapter pairs, expected {EXPECTED_PAIRS}")
if len(lora_keys) != 2 * EXPECTED_PAIRS:
    die(f"adapter has {len(lora_keys)} tensors, expected {2 * EXPECTED_PAIRS}")


def merged(name: str) -> torch.Tensor:
    """Load a base tensor, folding in its adapter delta when it has one."""
    path, _, _ = base_meta[name]
    w = handles[path].get_tensor(name)
    if name not in pairs:
        return w
    a = lora.get_tensor(pairs[name]["A"])  # [r, in]
    b = lora.get_tensor(pairs[name]["B"])  # [out, r]
    if a.shape[0] != r or b.shape[1] != r:
        die(f"{name}: adapter rank mismatch, A{tuple(a.shape)} B{tuple(b.shape)} vs r={r}")
    delta = (b.to(torch.float32) @ a.to(torch.float32)) * scale
    if delta.shape != w.shape:
        die(f"{name}: delta shape {tuple(delta.shape)} != base shape {tuple(w.shape)}")
    if w.dtype != torch.float32:
        die(f"{name}: base dtype {w.dtype} is not float32")
    return (w.to(torch.float32) + delta).to(w.dtype)


# ------------------------------------------------------------------ sharding

# safe_open reports dtypes as safetensors names ("F32", "BF16", ...).
ITEMSIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2,
            "I8": 1, "U8": 1, "BOOL": 1}


def nbytes(shape: tuple[int, ...], dtype: str) -> int:
    if dtype not in ITEMSIZE:
        die(f"unknown dtype {dtype!r}")
    n = 1
    for d in shape:
        n *= d
    return n * ITEMSIZE[dtype]


shards: list[list[str]] = []
current: list[str] = []
current_bytes = 0
for name, (_, shape, dtype) in base_meta.items():
    size = nbytes(shape, dtype)
    if current and current_bytes + size > MAX_SHARD_BYTES:
        shards.append(current)
        current, current_bytes = [], 0
    current.append(name)
    current_bytes += size
if current:
    shards.append(current)

for i, names in enumerate(shards):
    total = sum(nbytes(base_meta[n][1], base_meta[n][2]) for n in names)
    if total > MAX_SHARD_BYTES and len(names) > 1:
        die(f"shard {i} holds {total} bytes over the budget with {len(names)} tensors")

# ------------------------------------------------------------ pre-write checks

planned = [n for names in shards for n in names]
if sorted(planned) != sorted(base_meta):
    die("shard plan does not cover exactly the base tensor names")

# Check 2: no adapter tensor in the output.
bad = [n for n in planned if "lora_" in n]
if bad:
    die(f"adapter tensors would be written: {bad[:5]}")

# Check 3: the first merged q_proj keeps its shape.
probe = "model.layers.0.self_attn.q_proj.weight"
if probe not in pairs:
    die(f"{probe} was not among the merged tensors")
if base_meta[probe][2] != "F32":
    die(f"{probe} has dtype {base_meta[probe][2]}, expected F32")
if base_meta[probe][1] != (2048, 2048):
    die(f"{probe} has shape {base_meta[probe][1]}, expected (2048, 2048)")

# Check 4: exactly 114 tensors.
if len(planned) != EXPECTED_TENSORS:
    die(f"output would hold {len(planned)} tensors, expected {EXPECTED_TENSORS}")

# ---------------------------------------------------------------------- write

if os.path.isdir(OUT_DIR):
    for entry in os.listdir(OUT_DIR):
        if entry.endswith(".safetensors") or entry == "model.safetensors.index.json":
            os.remove(os.path.join(OUT_DIR, entry))
os.makedirs(OUT_DIR, exist_ok=True)

n_shards = len(shards)
out_map: dict[str, str] = {}
total_size = 0
for i, names in enumerate(shards, start=1):
    filename = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
    tensors = {}
    for name in names:
        t = merged(name).contiguous().clone()
        tensors[name] = t
        total_size += t.numel() * t.element_size()
        out_map[name] = filename
    save_file(tensors, os.path.join(OUT_DIR, filename), metadata={"format": "pt"})
    del tensors

with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(
        {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(out_map.items()))},
        f,
        indent=2,
    )
    f.write("\n")

for extra in ("config.json", "generation_config.json"):
    src = os.path.join(BASE_DIR, extra)
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(OUT_DIR, extra))

# ------------------------------------------------------------ verify on disk

seen: dict[str, tuple[tuple[int, ...], str]] = {}
for i in range(1, n_shards + 1):
    filename = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
    with safe_open(os.path.join(OUT_DIR, filename), framework="pt", device="cpu") as fh:
        for name in fh.keys():
            if "lora_" in name:
                die(f"adapter tensor {name!r} landed in {filename}")
            sl = fh.get_slice(name)
            seen[name] = (tuple(sl.get_shape()), str(sl.get_dtype()))
            if out_map[name] != filename:
                die(f"index maps {name!r} to {out_map[name]} but it is in {filename}")

if len(seen) != EXPECTED_TENSORS:
    die(f"wrote {len(seen)} tensors, expected {EXPECTED_TENSORS}")
if set(seen) != set(base_meta):
    die("written tensor names differ from the base tensor names")
if seen[probe][0] != (2048, 2048):
    die(f"{probe} was written with shape {seen[probe][0]}")
for name, (shape, dtype) in seen.items():
    if shape != base_meta[name][1]:
        die(f"{name}: wrote shape {shape}, base has {base_meta[name][1]}")
    if dtype != base_meta[name][2]:
        die(f"{name}: wrote dtype {dtype}, base has {base_meta[name][2]}")

print(f"merged {len(pairs)} adapter pairs with scale = {alpha} / {r} = {scale}")
print(f"wrote {len(seen)} tensors into {n_shards} shards ({total_size} bytes) under {OUT_DIR}")
for i, names in enumerate(shards, start=1):
    total = sum(nbytes(base_meta[n][1], base_meta[n][2]) for n in names)
    print(f"  shard {i:>2}: {len(names):>3} tensors, {total:>11} bytes")
