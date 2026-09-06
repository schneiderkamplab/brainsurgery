"""T3: mixed-precision export of GPT-2 (124M) with sharding.

Casts the 48 projection matrices to bfloat16, keeps everything else float32,
drops the 12 causal-mask buffers, and writes a sharded safetensors checkpoint
with an index file to out/T3/.
"""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(HERE, "..", "..", "inputs", "base", "model.safetensors")
OUT_DIR = HERE
MAX_SHARD_BYTES = 64 * 1024 * 1024  # 67_108_864

N_LAYERS = 12
PROJ_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)

# Explicit name sets: no regex, so nothing can leak onto wte/wpe/ln_*/biases.
PROJECTIONS = {
    f"h.{i}.{suffix}" for i in range(N_LAYERS) for suffix in PROJ_SUFFIXES
}
BUFFERS = {f"h.{i}.attn.bias" for i in range(N_LAYERS)}


def die(msg):
    raise SystemExit(f"FAIL: {msg}")


# ---------------------------------------------------------------- load
with safe_open(IN_PATH, framework="pt") as f:
    src = {k: f.get_tensor(k) for k in f.keys()}

print(f"loaded {len(src)} tensors from {IN_PATH}")

# Guard the assumptions the task states about the input.
missing = sorted((PROJECTIONS | BUFFERS) - set(src))
if missing:
    die(f"input is missing {len(missing)} expected tensors, e.g. {missing[:3]}")

# Nothing outside the enumerated layer range should look like a projection or
# a mask buffer; if the input has more layers than assumed, stop.
stray = sorted(
    k
    for k in src
    if re.fullmatch(r"h\.\d+\.(attn\.bias|attn\.c_attn\.weight|attn\.c_proj\.weight"
                    r"|mlp\.c_fc\.weight|mlp\.c_proj\.weight)", k)
    and k not in PROJECTIONS
    and k not in BUFFERS
)
if stray:
    die(f"input has projections/buffers outside layers 0..{N_LAYERS - 1}: {stray}")

# ---------------------------------------------------------------- transform
out = {}
for name, tensor in src.items():
    if name in BUFFERS:
        continue  # drop the causal-mask buffers
    if name in PROJECTIONS:
        out[name] = tensor.to(torch.bfloat16).contiguous()
    else:
        out[name] = tensor.to(torch.float32).contiguous()

# ---------------------------------------------------------------- checks
n_bf16 = sum(1 for t in out.values() if t.dtype is torch.bfloat16)
if n_bf16 != 48:
    die(f"expected exactly 48 bfloat16 tensors, got {n_bf16}")

if out["h.0.attn.c_attn.weight"].dtype is not torch.bfloat16:
    die(f"h.0.attn.c_attn.weight is {out['h.0.attn.c_attn.weight'].dtype}, want bfloat16")

if out["wte.weight"].dtype is not torch.float32:
    die(f"wte.weight is {out['wte.weight'].dtype}, want float32")

if len(out) != 148:
    die(f"expected 148 output tensors, got {len(out)}")

# Extra safety: no parameter was dropped, and every non-projection kept its
# exact float32 bits.
dropped = set(src) - set(out)
if dropped != BUFFERS:
    die(f"dropped the wrong set of tensors: {sorted(dropped ^ BUFFERS)}")
for name, tensor in out.items():
    if name in PROJECTIONS:
        continue
    if not torch.equal(tensor, src[name]):
        die(f"{name} changed value but should have been passed through unchanged")
    if tensor.shape != src[name].shape:
        die(f"{name} changed shape")

print(f"checks passed: {len(out)} tensors, {n_bf16} bfloat16")

# ---------------------------------------------------------------- shard
def nbytes(t):
    return t.numel() * t.element_size()


shards = []  # list of list-of-names
cur, cur_size = [], 0
for name in sorted(out):  # deterministic order; the input file order is arbitrary
    size = nbytes(out[name])
    if size > MAX_SHARD_BYTES:
        # Oversized tensor goes alone in its own shard.
        if cur:
            shards.append(cur)
        shards.append([name])
        cur, cur_size = [], 0
        continue
    if cur and cur_size + size > MAX_SHARD_BYTES:
        shards.append(cur)
        cur, cur_size = [], 0
    cur.append(name)
    cur_size += size
if cur:
    shards.append(cur)

total = len(shards)
weight_map = {}
total_size = 0
for idx, names in enumerate(shards, start=1):
    fname = f"model-{idx:05d}-of-{total:05d}.safetensors"
    shard_bytes = sum(nbytes(out[n]) for n in names)
    if len(names) > 1 and shard_bytes > MAX_SHARD_BYTES:
        die(f"{fname} holds {shard_bytes} bytes, over the {MAX_SHARD_BYTES} budget")
    total_size += shard_bytes
    for n in names:
        weight_map[n] = fname
    save_file({n: out[n] for n in names}, os.path.join(OUT_DIR, fname))
    print(f"wrote {fname}: {len(names)} tensors, {shard_bytes} bytes")

if len(weight_map) != len(out):
    die(f"weight_map covers {len(weight_map)} of {len(out)} tensors")

index = {
    "metadata": {"total_size": total_size},
    "weight_map": {k: weight_map[k] for k in sorted(weight_map)},
}
with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2, sort_keys=False)
    f.write("\n")

print(f"wrote model.safetensors.index.json: {total} shards, {total_size} bytes total")
