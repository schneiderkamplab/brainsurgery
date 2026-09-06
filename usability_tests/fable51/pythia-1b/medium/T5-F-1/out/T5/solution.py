"""T5: merge a PEFT LoRA adapter into Pythia-1B base weights and write a sharded
safetensors checkpoint. Plain torch + safetensors; no model instantiation."""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.dirname(os.path.dirname(ROOT))
BASE = os.path.join(SANDBOX, "inputs", "base", "model.safetensors")
LORA_DIR = os.path.join(SANDBOX, "inputs", "lora")
OUT = ROOT
MAX_SHARD = 512 * 1024 * 1024  # bytes of tensor data per shard
PREFIX = "base_model.model."
EXPECTED_PAIRS = 16
EXPECTED_TENSORS = 244
SOLO_TENSORS = {"gpt_neox.embed_in.weight", "embed_out.weight"}


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    with open(os.path.join(LORA_DIR, "adapter_config.json")) as f:
        cfg = json.load(f)
    scale = cfg["lora_alpha"] / cfg["r"]
    fan_in_fan_out = cfg.get("fan_in_fan_out", False)
    if scale != 2.0:
        fail(f"unexpected scale {scale}")

    base = {}
    with safe_open(BASE, framework="pt") as f:
        for k in f.keys():
            base[k] = f.get_tensor(k)
    if len(base) != EXPECTED_TENSORS:
        fail(f"base has {len(base)} tensors, expected {EXPECTED_TENSORS}")

    lora = {}
    with safe_open(os.path.join(LORA_DIR, "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            lora[k] = f.get_tensor(k)

    # Pair lora_A / lora_B by module name.
    modules = {}
    for k in lora:
        if not k.startswith(PREFIX):
            fail(f"adapter key without expected prefix: {k}")
        stem = k[len(PREFIX):]
        for tag in ("lora_A", "lora_B"):
            suffix = f".{tag}.weight"
            if stem.endswith(suffix):
                modules.setdefault(stem[: -len(suffix)], {})[tag] = lora[k]
                break
        else:
            fail(f"unrecognized adapter key: {k}")

    merged = 0
    for mod, ab in sorted(modules.items()):
        if set(ab) != {"lora_A", "lora_B"}:
            fail(f"incomplete adapter pair for {mod}: {sorted(ab)}")
        target = f"{mod}.weight"
        if target not in base:
            fail(f"adapter target {target} not in base")
        w = base[target]
        delta = (ab["lora_B"].float() @ ab["lora_A"].float()) * scale
        if fan_in_fan_out:
            delta = delta.T
        if delta.shape != w.shape:
            fail(f"delta shape {tuple(delta.shape)} != base {tuple(w.shape)} for {target}")
        base[target] = (w.float() + delta).to(w.dtype).contiguous()
        merged += 1

    # Required checks.
    if merged != EXPECTED_PAIRS:
        fail(f"merged {merged} adapter pairs, expected {EXPECTED_PAIRS}")
    if any("lora_" in k for k in base):
        fail("lora_ tensor present in output")
    qkv0 = base["gpt_neox.layers.0.attention.query_key_value.weight"]
    if tuple(qkv0.shape) != (6144, 2048) or qkv0.dtype != torch.float16:
        fail(f"layer 0 qkv has shape {tuple(qkv0.shape)} dtype {qkv0.dtype}")
    if len(base) != EXPECTED_TENSORS:
        fail(f"output has {len(base)} tensors, expected {EXPECTED_TENSORS}")

    # Shard: greedy fill in base key order. A tensor exceeding the budget on
    # its own, or one of the large embedding matrices the task says must be
    # stored alone, gets a shard by itself.
    shards, cur, cur_size = [], {}, 0
    for k, t in base.items():
        nbytes = t.numel() * t.element_size()
        if nbytes > MAX_SHARD or k in SOLO_TENSORS:
            if cur:
                shards.append(cur)
                cur, cur_size = {}, 0
            shards.append({k: t})
            continue
        if cur and cur_size + nbytes > MAX_SHARD:
            shards.append(cur)
            cur, cur_size = {}, 0
        cur[k] = t
        cur_size += nbytes
    if cur:
        shards.append(cur)

    n = len(shards)
    weight_map, total = {}, 0
    for i, shard in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{n:05d}.safetensors"
        size = sum(t.numel() * t.element_size() for t in shard.values())
        if len(shard) > 1 and size > MAX_SHARD:
            fail(f"shard {name} exceeds budget: {size}")
        total += size
        save_file(shard, os.path.join(OUT, name), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = name
    if len(weight_map) != EXPECTED_TENSORS:
        fail(f"weight_map has {len(weight_map)} entries")
    with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
    print(f"OK: merged {merged} pairs, wrote {n} shards, {len(weight_map)} tensors, {total} bytes")


if __name__ == "__main__":
    main()
