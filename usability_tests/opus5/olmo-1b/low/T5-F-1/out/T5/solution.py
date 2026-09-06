"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights, export sharded.

Plain script on torch + safetensors (condition F). Chosen over peft's
merge_and_unload because that route needs instantiating the full model and
gives no control over the shard layout required here.
"""

import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "inputs/base"
LORA = "inputs/lora"
OUT = "out/T5"
MAX_SHARD = 536_870_912  # 512 MiB of tensor data


def die(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def load_dir(d):
    index = json.load(open(os.path.join(d, "model.safetensors.index.json")))
    sd = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(os.path.join(d, shard), framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
    return sd


def main():
    cfg = json.load(open(os.path.join(LORA, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    if cfg.get("fan_in_fan_out", False):
        die("fan_in_fan_out=true not handled by this script")

    base = load_dir(BASE)
    n_base = len(base)
    with safe_open(os.path.join(LORA, "adapter_model.safetensors"), framework="pt") as f:
        adapter = {k: f.get_tensor(k) for k in f.keys()}

    merged = 0
    for name in list(adapter):
        if not name.endswith(".lora_A.weight"):
            continue
        stem = name[: -len(".lora_A.weight")]
        b_name = stem + ".lora_B.weight"
        if b_name not in adapter:
            die(f"missing {b_name}")
        prefix = "base_model.model."
        if not stem.startswith(prefix):
            die(f"unexpected adapter name {stem}")
        target = stem[len(prefix):] + ".weight"
        if target not in base:
            die(f"adapter target {target} not in base")
        A = adapter[name].to(torch.float32)
        B = adapter[b_name].to(torch.float32)
        W = base[target]
        if W.dtype != torch.float32:
            die(f"{target} is {W.dtype}, expected float32")
        delta = scale * (B @ A)
        if delta.shape != W.shape:
            die(f"delta {tuple(delta.shape)} != {tuple(W.shape)} for {target}")
        base[target] = (W + delta).to(torch.float32)
        merged += 1

    # Required checks.
    if merged != 32:
        die(f"merged {merged} adapter pairs, expected 32")
    if any("lora_" in k for k in base):
        die("adapter tensor leaked into output")
    probe = "model.layers.0.self_attn.q_proj.weight"
    if tuple(base[probe].shape) != (2048, 2048):
        die(f"{probe} has shape {tuple(base[probe].shape)}")
    if len(base) != 114:
        die(f"output has {len(base)} tensors, expected 114")
    if len(base) != n_base:
        die("tensor count changed relative to base")

    # Greedy sharding in sorted key order; a tensor bigger than the limit
    # ends up alone in its own shard.
    shards, cur, cur_size = [], {}, 0
    for k in sorted(base):
        size = base[k].numel() * base[k].element_size()
        if cur and cur_size + size > MAX_SHARD:
            shards.append(cur)
            cur, cur_size = {}, 0
        cur[k] = base[k]
        cur_size += size
    if cur:
        shards.append(cur)

    os.makedirs(OUT, exist_ok=True)
    n = len(shards)
    weight_map, total = {}, 0
    for i, shard in enumerate(shards, 1):
        fn = f"model-{i:05d}-of-{n:05d}.safetensors"
        size = sum(t.numel() * t.element_size() for t in shard.values())
        if size > MAX_SHARD and len(shard) > 1:
            die(f"shard {fn} exceeds limit with {len(shard)} tensors")
        total += size
        for k in shard:
            weight_map[k] = fn
        save_file({k: v.contiguous() for k, v in shard.items()}, os.path.join(OUT, fn),
                  metadata={"format": "pt"})
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=2)
    print(f"merged {merged} pairs, wrote {len(weight_map)} tensors in {n} shards")


main()
