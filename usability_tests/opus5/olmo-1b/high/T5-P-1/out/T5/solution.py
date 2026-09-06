"""T5: merge a PEFT LoRA adapter into the OLMo-1B base weights, write sharded.

Reads inputs/base (sharded safetensors) and inputs/lora, folds
scale * B @ A into every adapted base weight, and writes out/T5 as a
sharded safetensors checkpoint with a 512 MiB per-shard budget.

Every required check raises before any output file is written; the merge
arithmetic itself is checked again while shards are produced.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASE_DIR = ROOT / "inputs" / "base"
LORA_DIR = ROOT / "inputs" / "lora"
OUT_DIR = ROOT / "out" / "T5"

SHARD_LIMIT = 512 * 1024 * 1024  # 536,870,912 bytes of tensor data
# The task requires the two 412 MB embedding matrices to sit alone in their own
# shard, even though neither exceeds the budget above. Generalise that as: a
# tensor taking more than half a shard is never packed with anything else.
SOLO_LIMIT = SHARD_LIMIT // 2
EXPECTED_PAIRS = 32
EXPECTED_TENSORS = 114
PEFT_PREFIX = "base_model.model."

ITEMSIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}


def fail(msg: str) -> None:
    raise SystemExit(f"CHECK FAILED: {msg}")


# ---------------------------------------------------------------- base index

index_path = BASE_DIR / "model.safetensors.index.json"
base_index = json.loads(index_path.read_text())
base_map: dict[str, str] = base_index["weight_map"]  # name -> shard file

# name -> (shape, dtype string, byte size), read from the shard headers only.
base_meta: dict[str, tuple[list[int], str, int]] = {}
for shard_file in sorted(set(base_map.values())):
    with safe_open(BASE_DIR / shard_file, framework="pt") as f:
        for name in f.keys():
            sl = f.get_slice(name)
            shape = list(sl.get_shape())
            dtype = sl.get_dtype()
            if dtype not in ITEMSIZE:
                fail(f"unsupported dtype {dtype!r} for {name}")
            base_meta[name] = (shape, dtype, math.prod(shape) * ITEMSIZE[dtype])

if set(base_meta) != set(base_map):
    fail("base shard contents do not match the base index weight_map")
print(f"base: {len(base_meta)} tensors in {len(set(base_map.values()))} shards")

# ------------------------------------------------------------------- adapter

cfg = json.loads((LORA_DIR / "adapter_config.json").read_text())
r = int(cfg["r"])
alpha = float(cfg["lora_alpha"])
fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
if r <= 0:
    fail(f"adapter r must be positive, got {r}")
scale = alpha / r
print(f"adapter: r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

adapter = load_file(LORA_DIR / "adapter_model.safetensors")

# base weight name -> (A, B), derived from the lora_A keys.
pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
for key in sorted(adapter):
    if not key.endswith(".lora_A.weight"):
        continue
    b_key = key[: -len(".lora_A.weight")] + ".lora_B.weight"
    if b_key not in adapter:
        fail(f"{key} has no matching lora_B tensor")
    if not key.startswith(PEFT_PREFIX):
        fail(f"adapter key {key} does not start with {PEFT_PREFIX!r}")
    base_name = key[len(PEFT_PREFIX) : -len(".lora_A.weight")] + ".weight"
    if base_name not in base_meta:
        fail(f"adapter targets {base_name}, which is not in the base checkpoint")
    if base_name in pairs:
        fail(f"two adapter pairs map onto {base_name}")
    pairs[base_name] = (adapter[key], adapter[b_key])

# Every adapter tensor must be accounted for by exactly one pair.
if len(adapter) != 2 * len(pairs):
    fail(f"adapter has {len(adapter)} tensors but only {len(pairs)} A/B pairs")

# Shapes must line up with the base layout before anything is computed.
for base_name, (a, b) in pairs.items():
    shape, dtype, _ = base_meta[base_name]
    if a.ndim != 2 or b.ndim != 2:
        fail(f"{base_name}: adapter factors must be 2-D, got {list(a.shape)} / {list(b.shape)}")
    if a.shape[0] != r or b.shape[1] != r:
        fail(f"{base_name}: factor rank {list(a.shape)}/{list(b.shape)} disagrees with r={r}")
    out_dim, in_dim = (shape[1], shape[0]) if fan_in_fan_out else (shape[0], shape[1])
    if b.shape[0] != out_dim or a.shape[1] != in_dim:
        fail(
            f"{base_name}: B@A is [{b.shape[0]}, {a.shape[1]}] but the base weight "
            f"needs [{out_dim}, {in_dim}] (fan_in_fan_out={fan_in_fan_out})"
        )
    if dtype != "F32":
        fail(f"{base_name}: expected a float32 base weight, got {dtype}")

# --- required check 1: exactly 32 adapter pairs were found -------------------
if len(pairs) != EXPECTED_PAIRS:
    fail(f"expected {EXPECTED_PAIRS} adapter pairs, found {len(pairs)}")
print(f"matched {len(pairs)} adapter pairs onto base weights")

# ---------------------------------------------------- planned output contents

out_names = list(base_map)  # same names as the base, in base index order

# --- required check 2: no adapter tensor in the output -----------------------
leaked = [n for n in out_names if "lora_" in n]
if leaked:
    fail(f"adapter tensors would be written to the output: {leaked[:5]}")

# --- required check 3: the probe weight keeps its shape ----------------------
probe = "model.layers.0.self_attn.q_proj.weight"
if probe not in base_meta:
    fail(f"{probe} is missing from the base checkpoint")
if base_meta[probe][0] != [2048, 2048]:
    fail(f"{probe} has shape {base_meta[probe][0]}, expected [2048, 2048]")

# --- required check 4: exactly 114 tensors in the output ---------------------
if len(out_names) != EXPECTED_TENSORS:
    fail(f"output would have {len(out_names)} tensors, expected {EXPECTED_TENSORS}")
if len(set(out_names)) != len(out_names):
    fail("duplicate tensor names in the planned output")

# ------------------------------------------------------------------ sharding

shards: list[list[str]] = []
current: list[str] = []
current_bytes = 0
for name in out_names:
    nbytes = base_meta[name][2]
    if nbytes > SOLO_LIMIT:
        # Oversized (or near-oversized) tensor: it gets a shard of its own.
        if current:
            shards.append(current)
            current, current_bytes = [], 0
        shards.append([name])
        continue
    if current and current_bytes + nbytes > SHARD_LIMIT:
        shards.append(current)
        current, current_bytes = [], 0
    current.append(name)
    current_bytes += nbytes
if current:
    shards.append(current)

total = len(shards)
shard_names = [f"model-{i + 1:05d}-of-{total:05d}.safetensors" for i in range(total)]
for i, group in enumerate(shards):
    size = sum(base_meta[n][2] for n in group)
    if len(group) > 1 and size > SHARD_LIMIT:
        fail(f"shard {shard_names[i]} holds {size} bytes, over the {SHARD_LIMIT} budget")
    print(f"  {shard_names[i]}: {len(group):3d} tensors, {size / 2**20:8.1f} MiB")

# ------------------------------------------------------------------- writing

OUT_DIR.mkdir(parents=True, exist_ok=True)
# Clear checkpoint files from an earlier run; this directory also holds the script.
for stale in list(OUT_DIR.glob("*.safetensors")) + list(OUT_DIR.glob("*.index.json")):
    stale.unlink()

# Reverse lookup so each output shard reads only the base shards it needs.
handles = {
    shard_file: safe_open(BASE_DIR / shard_file, framework="pt")
    for shard_file in sorted(set(base_map.values()))
}

weight_map: dict[str, str] = {}
total_size = 0
merged = 0
try:
    for shard_file, group in zip(shard_names, shards):
        tensors: dict[str, torch.Tensor] = {}
        for name in group:
            t = handles[base_map[name]].get_tensor(name)
            if name in pairs:
                a, b = pairs[name]
                delta = scale * (b.to(torch.float32) @ a.to(torch.float32))
                if fan_in_fan_out:
                    delta = delta.T
                if delta.shape != t.shape:
                    fail(f"{name}: delta {list(delta.shape)} vs base {list(t.shape)}")
                t = (t.to(torch.float32) + delta).contiguous()
                merged += 1
            if list(t.shape) != base_meta[name][0]:
                fail(f"{name}: shape changed to {list(t.shape)}")
            if t.dtype is not torch.float32:
                fail(f"{name}: dtype changed to {t.dtype}")
            tensors[name] = t
            weight_map[name] = shard_file
            total_size += t.numel() * t.element_size()
        save_file(tensors, OUT_DIR / shard_file, metadata={"format": "pt"})
        del tensors
finally:
    for h in handles.values():
        h.__exit__(None, None, None)

if merged != EXPECTED_PAIRS:
    fail(f"merged {merged} weights, expected {EXPECTED_PAIRS}")

(OUT_DIR / "model.safetensors.index.json").write_text(
    json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2)
    + "\n"
)

# -------------------------------------------------------- verify what landed

written = json.loads((OUT_DIR / "model.safetensors.index.json").read_text())["weight_map"]
if set(written) != set(base_map):
    fail("the written index does not have the same names as the base")
if len(written) != EXPECTED_TENSORS:
    fail(f"the written index has {len(written)} tensors, expected {EXPECTED_TENSORS}")
if any("lora_" in n for n in written):
    fail("an adapter tensor reached the written output")

seen: set[str] = set()
for shard_file in sorted(set(written.values())):
    path = OUT_DIR / shard_file
    if not path.exists():
        fail(f"index references {shard_file}, which was not written")
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        size = sum(
            math.prod(f.get_slice(k).get_shape()) * ITEMSIZE[f.get_slice(k).get_dtype()]
            for k in keys
        )
    if len(keys) > 1 and size > SHARD_LIMIT:
        fail(f"{shard_file} holds {size} bytes of tensor data, over the budget")
    seen.update(keys)
if seen != set(written):
    fail("shard contents and index weight_map disagree")

stray = sorted(p.name for p in OUT_DIR.iterdir() if p.suffix == ".safetensors") 
if stray != sorted(set(written.values())):
    fail(f"unexpected shard files in the output: {stray}")

print(
    f"OK: wrote {len(written)} tensors into {len(set(written.values()))} shards "
    f"({total_size / 2**20:.1f} MiB) at {OUT_DIR}"
)
