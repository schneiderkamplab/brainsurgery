"""T3: mixed-precision export of GPT-2 (124M) with sharding.

Plain torch + safetensors. Casts exactly the 48 projection matrices to
bfloat16, drops the 12 causal-mask buffers, keeps everything else float32,
and writes 64 MiB shards plus model.safetensors.index.json.
"""
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "inputs" / "base" / "model.safetensors"
OUT = ROOT / "out" / "T3"
SHARD_BUDGET = 64 * 1024 * 1024  # 67,108,864 bytes of tensor data

PROJ = re.compile(r"^h\.(\d+)\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$")
BUF = re.compile(r"^h\.(\d+)\.attn\.bias$")


def fail(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


sd = load_file(str(SRC))
if len(sd) != 160:
    fail(f"expected 160 input tensors, got {len(sd)}")

out = {}
n_cast = n_drop = 0
for name, t in sd.items():
    if BUF.match(name):
        n_drop += 1
        continue
    if t.dtype != torch.float32:
        fail(f"{name} is {t.dtype}, expected float32 input")
    if PROJ.match(name):
        out[name] = t.to(torch.bfloat16).contiguous()
        n_cast += 1
    else:
        out[name] = t.contiguous()

# Required checks (before writing).
n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
if n_bf16 != 48 or n_cast != 48:
    fail(f"expected exactly 48 bfloat16 tensors, got {n_bf16} (cast {n_cast})")
if out["h.0.attn.c_attn.weight"].dtype != torch.bfloat16:
    fail("h.0.attn.c_attn.weight is not bfloat16")
if out["wte.weight"].dtype != torch.float32:
    fail("wte.weight is not float32")
if len(out) != 148:
    fail(f"expected 148 output tensors, got {len(out)}")
if n_drop != 12:
    fail(f"expected to drop 12 buffers, dropped {n_drop}")
for name, t in out.items():
    if t.dtype not in (torch.float32, torch.bfloat16):
        fail(f"{name} has unexpected dtype {t.dtype}")
    if t.dtype == torch.float32 and not torch.equal(t, sd[name]):
        fail(f"{name} changed value")

# Sharding: greedy in original key order; a tensor over budget gets its own shard.
shards = []  # list of (names, nbytes)
cur, cur_bytes = [], 0
for name, t in out.items():
    nb = t.numel() * t.element_size()
    if cur and cur_bytes + nb > SHARD_BUDGET:
        shards.append((cur, cur_bytes))
        cur, cur_bytes = [], 0
    cur.append(name)
    cur_bytes += nb
    if nb > SHARD_BUDGET:  # oversized tensor stays alone
        shards.append((cur, cur_bytes))
        cur, cur_bytes = [], 0
if cur:
    shards.append((cur, cur_bytes))

n = len(shards)
weight_map = {}
total_size = 0
OUT.mkdir(parents=True, exist_ok=True)
for existing in OUT.glob("model*.safetensors*"):
    existing.unlink()
for i, (names, nb) in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    if nb > SHARD_BUDGET and len(names) != 1:
        fail(f"shard {fname} exceeds budget with {len(names)} tensors")
    save_file({k: out[k] for k in names}, str(OUT / fname), metadata={"format": "pt"})
    for k in names:
        weight_map[k] = fname
    total_size += nb

index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
(OUT / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

# Post-write verification.
seen = {}
for fname in sorted(set(weight_map.values())):
    seen.update(load_file(str(OUT / fname)))
if set(seen) != set(out) or len(seen) != 148:
    fail("re-read key set differs from expected")
for k, t in out.items():
    if seen[k].dtype != t.dtype or seen[k].shape != t.shape or not torch.equal(seen[k], t):
        fail(f"re-read mismatch for {k}")
print(f"OK: {len(out)} tensors, {n_bf16} bfloat16, {n_drop} buffers dropped, {n} shards")
