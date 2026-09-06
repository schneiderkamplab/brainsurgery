"""T3: mixed-precision sharded export of GPT-2 (124M).

Plain torch + safetensors. Casts exactly the 48 projection matrices to
bfloat16, drops the 12 causal-mask buffers, keeps everything else float32,
and writes a sharded safetensors checkpoint with an index file.
"""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
OUT = "out/T3"
SHARD_BUDGET = 64 * 1024 * 1024  # 67,108,864 bytes of tensor data per shard

PROJ_RE = re.compile(r"^h\.(\d+)\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$")
BUFFER_RE = re.compile(r"^h\.(\d+)\.attn\.bias$")


def natural_key(name: str):
    """Sort non-layer tensors first, then layers by numeric index, then by name."""
    m = re.match(r"^h\.(\d+)\.(.*)$", name)
    if m:
        return (1, int(m.group(1)), m.group(2))
    return (0, 0, name)


def fail(msg: str):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    src = load_file(SRC)
    if len(src) != 160:
        fail(f"expected 160 input tensors, got {len(src)}")

    out: dict[str, torch.Tensor] = {}
    n_cast = n_dropped = 0
    for name, t in src.items():
        if BUFFER_RE.match(name):
            n_dropped += 1
            continue
        if PROJ_RE.match(name):
            if t.dtype != torch.float32:
                fail(f"{name} expected float32 input, got {t.dtype}")
            out[name] = t.to(torch.bfloat16).contiguous()
            n_cast += 1
        else:
            if t.dtype != torch.float32:
                fail(f"{name} expected float32, got {t.dtype}")
            out[name] = t.contiguous()

    # ---- Required checks (before writing anything) ----
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 48:
        fail(f"expected exactly 48 bfloat16 tensors, got {n_bf16}")
    if n_cast != 48:
        fail(f"cast {n_cast} projection matrices, expected 48")
    if n_dropped != 12:
        fail(f"dropped {n_dropped} buffers, expected 12")
    if out["h.0.attn.c_attn.weight"].dtype != torch.bfloat16:
        fail("h.0.attn.c_attn.weight is not bfloat16")
    if out["wte.weight"].dtype != torch.float32:
        fail("wte.weight is not float32")
    if len(out) != 148:
        fail(f"expected 148 output tensors, got {len(out)}")
    n_f32 = sum(1 for t in out.values() if t.dtype == torch.float32)
    if n_f32 != 100:
        fail(f"expected 100 float32 tensors, got {n_f32}")
    # Everything not cast must be bit-identical to the source.
    for name, t in out.items():
        if t.dtype == torch.float32 and not torch.equal(t, src[name]):
            fail(f"{name} changed value")

    # ---- Sharding: greedy in natural key order, oversized tensors alone ----
    shards: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for name in sorted(out, key=natural_key):
        nbytes = out[name].numel() * out[name].element_size()
        if nbytes > SHARD_BUDGET:
            if cur:
                shards.append(cur)
                cur, cur_bytes = [], 0
            shards.append([name])
            continue
        if cur_bytes + nbytes > SHARD_BUDGET:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(name)
        cur_bytes += nbytes
    if cur:
        shards.append(cur)

    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    os.makedirs(OUT, exist_ok=True)
    existing = [f for f in os.listdir(OUT) if f.endswith(".safetensors")]
    if existing:
        fail(f"{OUT} already contains shard files: {existing}")

    for i, names in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        chunk = {n: out[n] for n in names}
        size = sum(t.numel() * t.element_size() for t in chunk.values())
        if len(chunk) > 1 and size > SHARD_BUDGET:
            fail(f"shard {fname} exceeds budget: {size}")
        save_file(chunk, os.path.join(OUT, fname), metadata={"format": "pt"})
        for n in names:
            weight_map[n] = fname
        total_size += size
        print(f"{fname}: {len(names)} tensors, {size} bytes")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    # ---- Post-write verification: reload and compare ----
    reloaded: dict[str, torch.Tensor] = {}
    for fname in sorted(set(weight_map.values())):
        with safe_open(os.path.join(OUT, fname), framework="pt") as f:
            for k in f.keys():
                if weight_map[k] != fname:
                    fail(f"index mismatch for {k}")
                reloaded[k] = f.get_tensor(k)
    if set(reloaded) != set(out) or len(reloaded) != 148:
        fail("reloaded key set differs from expected")
    for k, t in out.items():
        r = reloaded[k]
        if r.dtype != t.dtype or r.shape != t.shape or not torch.equal(r, t):
            fail(f"reloaded {k} differs")
    print(f"OK: {len(out)} tensors, {n_bf16} bf16, {n_f32} f32, {n_shards} shards, "
          f"{total_size} bytes total")


if __name__ == "__main__":
    main()
