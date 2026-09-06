"""T3: mixed-precision sharded export of OLMo-1B-0724-hf.

Plain torch + safetensors script. Casts exactly the 112 per-layer projection
matrices to bfloat16, keeps every other tensor float32 and bit-identical,
and writes a greedy-filled sharded safetensors checkpoint (<= 256 MiB of
tensor data per shard; oversized tensors alone in their own shard) with a
model.safetensors.index.json.
"""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.dirname(os.path.dirname(HERE))
IN_DIR = os.path.join(SANDBOX, "inputs", "base")
OUT_DIR = HERE  # out/T3
MAX_SHARD = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data

NUM_LAYERS = 16
PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
# Explicit, fully anchored set: no wildcard that could hit embeddings/norms.
BF16_KEYS = {f"model.layers.{i}.{p}.weight" for i in range(NUM_LAYERS) for p in PROJ}
assert len(BF16_KEYS) == 112


def fail(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def sort_key(name):
    m = re.match(r"model\.layers\.(\d+)\.", name)
    if name == "model.embed_tokens.weight":
        return (0, 0, name)
    if m:
        return (1, int(m.group(1)), name)
    return (2, 0, name)


# ---- load ---------------------------------------------------------------
with open(os.path.join(IN_DIR, "model.safetensors.index.json")) as f:
    in_index = json.load(f)["weight_map"]
tensors = {}
for shard in sorted(set(in_index.values())):
    with safe_open(os.path.join(IN_DIR, shard), framework="pt") as sf:
        for k in sf.keys():
            if k in tensors:
                fail(f"duplicate key across input shards: {k}")
            tensors[k] = sf.get_tensor(k)
if len(tensors) != 114:
    fail(f"expected 114 input tensors, got {len(tensors)}")
missing = BF16_KEYS - set(tensors)
if missing:
    fail(f"projection keys missing from input: {sorted(missing)[:5]} ...")

# ---- transform ----------------------------------------------------------
out = {}
for k in sorted(tensors, key=sort_key):
    t = tensors[k]
    if t.dtype != torch.float32:
        fail(f"{k} is {t.dtype}, expected float32 input")
    out[k] = t.to(torch.bfloat16).contiguous() if k in BF16_KEYS else t.contiguous()

# ---- required checks (before writing) -----------------------------------
n_bf16 = sum(v.dtype == torch.bfloat16 for v in out.values())
if n_bf16 != 112:
    fail(f"expected exactly 112 bfloat16 tensors, got {n_bf16}")
if out["model.layers.0.self_attn.q_proj.weight"].dtype != torch.bfloat16:
    fail("model.layers.0.self_attn.q_proj.weight is not bfloat16")
if out["model.embed_tokens.weight"].dtype != torch.float32:
    fail("model.embed_tokens.weight is not float32")
if out["lm_head.weight"].dtype != torch.float32:
    fail("lm_head.weight is not float32")
if len(out) != 114:
    fail(f"expected exactly 114 output tensors, got {len(out)}")
if set(out) != set(tensors):
    fail("output key set differs from input key set")
for k in out:
    if k not in BF16_KEYS and not torch.equal(out[k], tensors[k]):
        fail(f"{k} changed value")
    if out[k].shape != tensors[k].shape:
        fail(f"{k} changed shape")

# ---- shard (greedy, in order) -------------------------------------------
shards, cur, cur_size = [], [], 0
for k, v in out.items():
    size = v.numel() * v.element_size()
    if size > MAX_SHARD:
        if cur:
            shards.append(cur)
            cur, cur_size = [], 0
        shards.append([k])
        continue
    if cur_size + size > MAX_SHARD:
        shards.append(cur)
        cur, cur_size = [], 0
    cur.append(k)
    cur_size += size
if cur:
    shards.append(cur)

n = len(shards)
weight_map, total = {}, 0
for i, keys in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    data = {k: out[k] for k in keys}
    size = sum(v.numel() * v.element_size() for v in data.values())
    if size > MAX_SHARD and len(data) != 1:
        fail(f"shard {fname} exceeds budget with {len(data)} tensors")
    total += size
    for k in keys:
        weight_map[k] = fname

# ---- write --------------------------------------------------------------
for old in os.listdir(OUT_DIR):
    if old.endswith(".safetensors") or old == "model.safetensors.index.json":
        os.remove(os.path.join(OUT_DIR, old))
for i, keys in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    save_file({k: out[k] for k in keys}, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2, sort_keys=True)
    f.write("\n")

# ---- verify what was written --------------------------------------------
seen = 0
for fname in sorted(set(weight_map.values())):
    with safe_open(os.path.join(OUT_DIR, fname), framework="pt") as sf:
        keys = list(sf.keys())
        seen += len(keys)
        for k in keys:
            if weight_map[k] != fname:
                fail(f"index mismatch for {k}")
            t = sf.get_tensor(k)
            if not torch.equal(t, out[k]):
                fail(f"round-trip mismatch for {k}")
if seen != 114 or len(weight_map) != 114:
    fail(f"written checkpoint has {seen} tensors / {len(weight_map)} index entries")
print(f"OK: {seen} tensors, {n} shards, {n_bf16} bf16, {total} bytes of tensor data")
for i, keys in enumerate(shards, 1):
    print(f"  shard {i}: {len(keys)} tensors, "
          f"{sum(out[k].numel() * out[k].element_size() for k in keys)} bytes")
