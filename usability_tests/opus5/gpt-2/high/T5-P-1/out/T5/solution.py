#!/usr/bin/env python
"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and export sharded.

base (Conv1D)  h.<i>.attn.c_attn.weight : [in=768, out=2304]
adapter (Linear conv.) lora_A [r, in], lora_B [out, r]

fan_in_fan_out = true -> delta for the Linear layout is (B @ A) with shape
[out, in]; the base is stored transposed, so we add its transpose:

    W[in, out] += (alpha / r) * (B @ A).T
"""

import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

BASE_PATH = os.path.join(ROOT, "inputs", "base", "model.safetensors")
LORA_PATH = os.path.join(ROOT, "inputs", "lora", "adapter_model.safetensors")
CONFIG_PATH = os.path.join(ROOT, "inputs", "lora", "adapter_config.json")
OUT_DIR = os.path.join(ROOT, "out", "T5")

MAX_SHARD_BYTES = 100 * 1024 * 1024  # 104,857,600
EXPECTED_PAIRS = 12
EXPECTED_TENSORS = 160

# base_model.model.h.<i>.<module>.lora_{A,B}.weight
ADAPTER_RE = re.compile(r"^base_model\.model\.(.+)\.lora_([AB])\.weight$")


def fail(msg):
    raise SystemExit(f"CHECK FAILED: {msg}")


def load_all(path):
    """Load every tensor, preserving the checkpoint's own key order."""
    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def nbytes(t):
    return t.numel() * t.element_size()


def main():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    fan_in_fan_out = bool(cfg["fan_in_fan_out"])
    if r <= 0:
        fail(f"adapter r must be positive, got {r}")
    scale = alpha / r
    print(f"adapter: r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")
    if not fan_in_fan_out:
        fail("this script assumes fan_in_fan_out=true (Conv1D base layout)")

    base = load_all(BASE_PATH)
    lora = load_all(LORA_PATH)
    print(f"loaded base: {len(base)} tensors; adapter: {len(lora)} tensors")

    if len(base) != EXPECTED_TENSORS:
        fail(f"base has {len(base)} tensors, expected {EXPECTED_TENSORS}")

    # --- group adapter tensors into (A, B) pairs keyed by the base name ---
    pairs = {}
    for name, t in lora.items():
        m = ADAPTER_RE.match(name)
        if m is None:
            fail(f"unrecognised adapter tensor name: {name!r}")
        stem, side = m.group(1), m.group(2)
        target = f"{stem}.weight"
        slot = pairs.setdefault(target, {})
        if side in slot:
            fail(f"duplicate lora_{side} for {target}")
        slot[side] = t

    for target, slot in pairs.items():
        if set(slot) != {"A", "B"}:
            fail(f"incomplete adapter pair for {target}: have {sorted(slot)}")

    # CHECK 1: exactly 12 adapter pairs
    if len(pairs) != EXPECTED_PAIRS:
        fail(f"found {len(pairs)} adapter pairs, expected {EXPECTED_PAIRS}")

    # --- merge ---
    merged = dict(base)  # same key order as the base checkpoint
    for target in sorted(pairs, key=lambda s: (len(s), s)):
        A = pairs[target]["A"]
        B = pairs[target]["B"]
        if target not in merged:
            fail(f"adapter targets {target!r}, which is not in the base checkpoint")
        W = merged[target]

        if A.ndim != 2 or B.ndim != 2:
            fail(f"{target}: expected 2-D adapter factors, got {A.shape} / {B.shape}")
        if A.shape[0] != r or B.shape[1] != r:
            fail(f"{target}: adapter rank mismatch, A={tuple(A.shape)} B={tuple(B.shape)} r={r}")
        # delta in Linear layout is [out, in]; base is Conv1D [in, out]
        if (B.shape[0], A.shape[1]) != (W.shape[1], W.shape[0]):
            fail(
                f"{target}: layout mismatch, (B@A)={ (B.shape[0], A.shape[1]) } "
                f"vs base {tuple(W.shape)} (expected the transpose)"
            )
        if W.dtype != torch.float32 or A.dtype != torch.float32 or B.dtype != torch.float32:
            fail(f"{target}: expected float32 throughout")

        delta = (B.to(torch.float32) @ A.to(torch.float32)).T  # [in, out]
        new_W = (W.to(torch.float32) + scale * delta).contiguous()

        if new_W.shape != W.shape:
            fail(f"{target}: shape changed {tuple(W.shape)} -> {tuple(new_W.shape)}")
        if new_W.dtype != torch.float32:
            fail(f"{target}: dtype changed to {new_W.dtype}")
        merged[target] = new_W
        print(f"  merged {target} {tuple(new_W.shape)}")

    # --- required checks on the final state dict ---
    # CHECK 2: no adapter tensor survives
    leaked = [k for k in merged if "lora_" in k]
    if leaked:
        fail(f"adapter tensors leaked into the output: {leaked[:5]}")
    # CHECK 3: the probe tensor keeps its shape
    probe = "h.0.attn.c_attn.weight"
    if probe not in merged:
        fail(f"{probe} missing from the output")
    if tuple(merged[probe].shape) != (768, 2304):
        fail(f"{probe} has shape {tuple(merged[probe].shape)}, expected (768, 2304)")
    # CHECK 4: tensor count
    if len(merged) != EXPECTED_TENSORS:
        fail(f"output has {len(merged)} tensors, expected {EXPECTED_TENSORS}")
    if set(merged) != set(base):
        fail("output key set differs from the base key set")
    for k in base:
        if merged[k].shape != base[k].shape or merged[k].dtype != base[k].dtype:
            fail(f"{k}: shape/dtype drifted from the base")
    print(f"checks passed: {len(pairs)} pairs merged, {len(merged)} tensors, no lora_ keys")

    # --- shard: greedy over the base key order, oversized tensors land alone ---
    shards = []
    cur, cur_bytes = {}, 0
    for k, t in merged.items():
        n = nbytes(t)
        if cur and cur_bytes + n > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[k] = t
        cur_bytes += n
    if cur:
        shards.append(cur)

    total = len(shards)
    names = [f"model-{i + 1:05d}-of-{total:05d}.safetensors" for i in range(total)]

    for name, shard in zip(names, shards):
        size = sum(nbytes(t) for t in shard.values())
        if size > MAX_SHARD_BYTES and len(shard) != 1:
            fail(f"{name}: {size} bytes over budget with {len(shard)} tensors")
        print(f"  {name}: {len(shard):3d} tensors, {size:,} bytes")

    weight_map = {k: name for name, shard in zip(names, shards) for k in shard}
    if len(weight_map) != EXPECTED_TENSORS:
        fail(f"weight_map covers {len(weight_map)} tensors, expected {EXPECTED_TENSORS}")

    # --- write ---
    os.makedirs(OUT_DIR, exist_ok=True)
    for stale in os.listdir(OUT_DIR):
        if stale.endswith(".safetensors") or stale == "model.safetensors.index.json":
            os.remove(os.path.join(OUT_DIR, stale))

    for name, shard in zip(names, shards):
        save_file(shard, os.path.join(OUT_DIR, name), metadata={"format": "pt"})

    index = {
        "metadata": {"total_size": sum(nbytes(t) for t in merged.values())},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")

    # --- read back and re-verify what was actually written ---
    seen = {}
    for name in names:
        with safe_open(os.path.join(OUT_DIR, name), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in seen:
                    fail(f"{k} written to more than one shard")
                seen[k] = f.get_slice(k).get_shape()
    if len(seen) != EXPECTED_TENSORS:
        fail(f"read back {len(seen)} tensors, expected {EXPECTED_TENSORS}")
    if set(seen) != set(base):
        fail("read-back key set differs from the base key set")
    if any("lora_" in k for k in seen):
        fail("adapter tensors present in the written shards")
    if tuple(seen[probe]) != (768, 2304):
        fail(f"{probe} written with shape {tuple(seen[probe])}")

    print(f"wrote {total} shards + index to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
