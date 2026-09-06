"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and export sharded safetensors.

Plain torch + safetensors script: the task is a deterministic file-level rewrite
with a custom shard budget, which is easier to state exactly here than to coax
out of peft/transformers save_pretrained.
"""

import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "inputs/base/model.safetensors"
LORA = "inputs/lora/adapter_model.safetensors"
CFG = "inputs/lora/adapter_config.json"
OUT = "out/T5"
SHARD_BUDGET = 100 * 1024 * 1024  # 104,857,600 bytes of tensor data


def load(path):
    with safe_open(path, framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def main():
    cfg = json.load(open(CFG))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / r
    fan_in_fan_out = cfg["fan_in_fan_out"]

    base = load(BASE)
    lora = load(LORA)

    # group adapter tensors by target base name
    pat = re.compile(r"^base_model\.model\.(?P<name>.+)\.lora_(?P<ab>[AB])\.weight$")
    pairs = {}
    for k in lora:
        m = pat.match(k)
        if m is None:
            raise SystemExit(f"unrecognised adapter tensor name: {k}")
        pairs.setdefault(m["name"], {})[m["ab"]] = k

    merged = 0
    for name, ab in sorted(pairs.items()):
        if set(ab) != {"A", "B"}:
            raise SystemExit(f"incomplete adapter pair for {name}: {sorted(ab)}")
        target = f"{name}.weight"
        if target not in base:
            raise SystemExit(f"adapter targets missing base tensor {target}")
        A = lora[ab["A"]].to(torch.float32)
        B = lora[ab["B"]].to(torch.float32)
        if A.shape[0] != r or B.shape[1] != r:
            raise SystemExit(f"{name}: factor rank does not match r={r}: {A.shape} {B.shape}")
        delta = scale * (B @ A)  # [out, in]
        if fan_in_fan_out:
            delta = delta.T  # base uses Conv1D [in, out]
        W = base[target]
        if W.shape != delta.shape:
            raise SystemExit(f"{name}: delta {tuple(delta.shape)} != base {tuple(W.shape)}")
        if W.dtype != torch.float32:
            raise SystemExit(f"{name}: base dtype {W.dtype} is not float32")
        base[target] = (W.to(torch.float32) + delta).contiguous()
        merged += 1

    # ---- required checks, before writing ----
    assert merged == 12, f"expected 12 adapter pairs merged, got {merged}"
    assert not [k for k in base if "lora_" in k], "adapter tensor leaked into output"
    assert tuple(base["h.0.attn.c_attn.weight"].shape) == (768, 2304), "c_attn shape changed"
    assert len(base) == 160, f"expected 160 output tensors, got {len(base)}"

    # ---- shard, greedily, in base key order ----
    shards, cur, cur_bytes = [], {}, 0
    for k, t in base.items():
        n = t.numel() * t.element_size()
        if n > SHARD_BUDGET:
            if cur:
                shards.append(cur)
                cur, cur_bytes = {}, 0
            shards.append({k: t})  # oversized tensor alone
            continue
        if cur and cur_bytes + n > SHARD_BUDGET:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[k] = t
        cur_bytes += n
    if cur:
        shards.append(cur)

    total = len(shards)
    for i, s in enumerate(shards, 1):
        sz = sum(t.numel() * t.element_size() for t in s.values())
        assert sz <= SHARD_BUDGET or len(s) == 1, f"shard {i} over budget: {sz}"

    if os.path.isdir(OUT):
        for f in os.listdir(OUT):
            if f.endswith(".safetensors") or f == "model.safetensors.index.json":
                os.remove(os.path.join(OUT, f))
    os.makedirs(OUT, exist_ok=True)

    weight_map, total_size = {}, 0
    for i, s in enumerate(shards, 1):
        fn = f"model-{i:05d}-of-{total:05d}.safetensors"
        save_file(s, os.path.join(OUT, fn), metadata={"format": "pt"})
        for k, t in s.items():
            weight_map[k] = fn
            total_size += t.numel() * t.element_size()
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=False)

    assert len(weight_map) == 160, f"weight_map has {len(weight_map)} entries"
    print(f"merged {merged} pairs (scale={scale}), wrote {total} shards, {len(weight_map)} tensors")


if __name__ == "__main__":
    main()
