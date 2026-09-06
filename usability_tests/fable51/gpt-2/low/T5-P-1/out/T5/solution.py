"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and export sharded safetensors."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
LORA = os.path.join(ROOT, "inputs", "lora", "adapter_model.safetensors")
CFG = os.path.join(ROOT, "inputs", "lora", "adapter_config.json")
OUT = os.path.join(ROOT, "out", "T5")
SHARD_BUDGET = 100 * 1024 * 1024


def fail(msg):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    with open(CFG) as f:
        cfg = json.load(f)
    scale = cfg["lora_alpha"] / cfg["r"]
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    print(f"scale={scale} fan_in_fan_out={fan_in_fan_out}")

    base = load_file(BASE)
    lora = load_file(LORA)
    if len(base) != 160:
        fail(f"expected 160 base tensors, got {len(base)}")

    pat = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")
    merged = 0
    for a_name, A in lora.items():
        m = pat.match(a_name)
        if not m:
            continue
        module = m.group(1)
        b_name = f"base_model.model.{module}.lora_B.weight"
        if b_name not in lora:
            fail(f"missing lora_B for {a_name}")
        B = lora[b_name]
        target = f"{module}.weight"
        if target not in base:
            fail(f"target {target} not in base")
        W = base[target]
        if W.dtype != torch.float32:
            fail(f"{target} is {W.dtype}, expected float32")
        delta = scale * (B.float() @ A.float())  # [out, in]
        if fan_in_fan_out:
            delta = delta.T  # base is Conv1D [in, out]
        if delta.shape != W.shape:
            fail(f"delta shape {tuple(delta.shape)} != {tuple(W.shape)} for {target}")
        base[target] = (W + delta).contiguous()
        merged += 1

    # Required checks.
    if merged != 12:
        fail(f"expected 12 merged adapter pairs, got {merged}")
    if any("lora_" in k for k in base):
        fail("lora_ tensor present in output")
    if tuple(base["h.0.attn.c_attn.weight"].shape) != (768, 2304):
        fail(f"h.0.attn.c_attn.weight has shape {tuple(base['h.0.attn.c_attn.weight'].shape)}")
    if len(base) != 160:
        fail(f"expected 160 output tensors, got {len(base)}")

    # Shard: greedy fill up to the byte budget; oversized tensors go alone.
    shards, cur, cur_bytes = [], {}, 0
    for name in base:  # keep base order
        t = base[name]
        nbytes = t.numel() * t.element_size()
        if cur and cur_bytes + nbytes > SHARD_BUDGET:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[name] = t
        cur_bytes += nbytes
    if cur:
        shards.append(cur)

    os.makedirs(OUT, exist_ok=True)
    n = len(shards)
    weight_map = {}
    total = 0
    for i, shard in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file(shard, os.path.join(OUT, fname), metadata={"format": "pt"})
        for k, t in shard.items():
            weight_map[k] = fname
            total += t.numel() * t.element_size()
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    if len(weight_map) != 160:
        fail(f"index maps {len(weight_map)} tensors, expected 160")
    print(f"wrote {n} shards, {len(weight_map)} tensors, {total} bytes to {OUT}")


if __name__ == "__main__":
    main()
