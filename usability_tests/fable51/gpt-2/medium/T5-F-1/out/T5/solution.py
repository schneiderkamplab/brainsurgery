"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and write a sharded safetensors checkpoint.

Plain safetensors + torch: reads base and adapter tensors, applies
    W += (alpha / r) * (B @ A).T      (fan_in_fan_out = true, Conv1D base layout [in, out])
in float32, verifies the required invariants, then writes shards of at most 100 MiB of
tensor data each plus model.safetensors.index.json.
"""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
LORA_W = os.path.join(ROOT, "inputs", "lora", "adapter_model.safetensors")
LORA_CFG = os.path.join(ROOT, "inputs", "lora", "adapter_config.json")
OUT = os.path.join(ROOT, "out", "T5")
MAX_SHARD = 100 * 1024 * 1024  # 104,857,600 bytes of tensor data
EXPECTED_PAIRS = 12
EXPECTED_TENSORS = 160


def fail(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    cfg = json.load(open(LORA_CFG))
    scale = cfg["lora_alpha"] / cfg["r"]
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    print(f"scale={scale} fan_in_fan_out={fan_in_fan_out} target_modules={cfg['target_modules']}")

    with safe_open(BASE, framework="pt") as f:
        names = list(f.keys())
        base = {k: f.get_tensor(k) for k in names}
    with safe_open(LORA_W, framework="pt") as f:
        lora = {k: f.get_tensor(k) for k in f.keys()}

    # Pair adapter factors: base_model.model.<name>.lora_A.weight / lora_B.weight -> <name>.weight
    pat = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")
    merged = 0
    for a_name, A in lora.items():
        m = pat.match(a_name)
        if not m:
            continue
        module = m.group(1)
        b_name = f"base_model.model.{module}.lora_B.weight"
        if b_name not in lora:
            fail(f"lora_B missing for {a_name}")
        B = lora[b_name]
        target = f"{module}.weight"
        if target not in base:
            fail(f"base tensor {target} not found for adapter {a_name}")
        W = base[target]
        if W.dtype != torch.float32 or A.dtype != torch.float32 or B.dtype != torch.float32:
            fail(f"non-float32 tensors for {target}")
        delta = scale * (B.float() @ A.float())  # [out, in]
        if fan_in_fan_out:
            delta = delta.T  # Conv1D base is [in, out]
        if delta.shape != W.shape:
            fail(f"delta shape {tuple(delta.shape)} != base shape {tuple(W.shape)} for {target}")
        base[target] = (W + delta).contiguous()
        merged += 1
    unpaired = [k for k in lora if "lora_A" not in k and "lora_B" not in k]
    if unpaired:
        fail(f"unexpected adapter tensors: {unpaired}")
    if merged != EXPECTED_PAIRS:
        fail(f"merged {merged} adapter pairs, expected {EXPECTED_PAIRS}")
    if len(lora) != 2 * merged:
        fail(f"adapter has {len(lora)} tensors, expected {2 * merged}")

    # Required checks before writing.
    if any("lora_" in k for k in base):
        fail("a tensor name containing 'lora_' is in the output")
    if tuple(base["h.0.attn.c_attn.weight"].shape) != (768, 2304):
        fail(f"h.0.attn.c_attn.weight has shape {tuple(base['h.0.attn.c_attn.weight'].shape)}")
    if len(base) != EXPECTED_TENSORS:
        fail(f"output has {len(base)} tensors, expected {EXPECTED_TENSORS}")
    if set(base) != set(names):
        fail("output key set differs from base key set")

    # Greedy sharding in base order; an oversized tensor goes alone in its own shard.
    shards, cur, cur_bytes = [], [], 0
    for k in names:
        nbytes = base[k].numel() * base[k].element_size()
        if cur and cur_bytes + nbytes > MAX_SHARD:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(k)
        cur_bytes += nbytes
    if cur:
        shards.append(cur)
    for s in shards:
        total = sum(base[k].numel() * base[k].element_size() for k in s)
        if total > MAX_SHARD and len(s) != 1:
            fail(f"shard exceeds budget with more than one tensor: {total} bytes")

    os.makedirs(OUT, exist_ok=True)
    for old in os.listdir(OUT):
        if old.endswith(".safetensors") or old == "model.safetensors.index.json":
            os.remove(os.path.join(OUT, old))
    n = len(shards)
    weight_map = {}
    for i, s in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file({k: base[k].contiguous() for k in s}, os.path.join(OUT, fname), metadata={"format": "pt"})
        for k in s:
            weight_map[k] = fname
        print(f"wrote {fname}: {len(s)} tensors, "
              f"{sum(base[k].numel() * base[k].element_size() for k in s)} bytes")
    total_size = sum(t.numel() * t.element_size() for t in base.values())
    with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, fh, indent=2)

    # Post-write verification: reload and check count, names, no lora_, shard budgets.
    seen = {}
    for fname in sorted(set(weight_map.values())):
        with safe_open(os.path.join(OUT, fname), framework="pt") as f:
            keys = list(f.keys())
            tb = 0
            for k in keys:
                t = f.get_tensor(k)
                tb += t.numel() * t.element_size()
                seen[k] = fname
            if tb > MAX_SHARD and len(keys) != 1:
                fail(f"written shard {fname} over budget")
    if seen != weight_map:
        fail("index weight_map does not match shard contents")
    if len(seen) != EXPECTED_TENSORS or any("lora_" in k for k in seen):
        fail("post-write tensor set check failed")
    print(f"OK: {merged} adapters merged, {len(seen)} tensors in {n} shards at {OUT}")


if __name__ == "__main__":
    main()
