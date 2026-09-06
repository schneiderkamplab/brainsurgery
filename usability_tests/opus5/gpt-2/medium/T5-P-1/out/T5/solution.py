"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and write a sharded checkpoint."""

import json
import os
import re
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "inputs/base/model.safetensors"
ADAPTER = "inputs/lora/adapter_model.safetensors"
ADAPTER_CFG = "inputs/lora/adapter_config.json"
OUT_DIR = "out/T5"
SHARD_LIMIT = 100 * 1024 * 1024  # 100 MiB of tensor data per shard


def load_st(path):
    tensors = OrderedDict()
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def main():
    cfg = json.load(open(ADAPTER_CFG))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    fan_in_fan_out = cfg["fan_in_fan_out"]
    scale = alpha / r

    base = load_st(BASE)
    adapter = load_st(ADAPTER)

    # Pair up lora_A / lora_B by their common prefix, then map to a base name.
    pat = re.compile(r"^(?:base_model\.model\.)?(.*)\.lora_(A|B)(?:\.default)?\.weight$")
    pairs = {}
    for name in adapter:
        m = pat.match(name)
        if m is None:
            raise RuntimeError(f"unrecognised adapter tensor name: {name}")
        pairs.setdefault(m.group(1), {})[m.group(2)] = name

    merged = 0
    for stem, sides in sorted(pairs.items()):
        if set(sides) != {"A", "B"}:
            raise RuntimeError(f"incomplete LoRA pair for {stem}: {sorted(sides)}")
        target = stem + ".weight"
        if target not in base:
            raise RuntimeError(f"no base tensor for adapter {stem} (looked for {target})")

        A = adapter[sides["A"]].to(torch.float32)  # [r, in]
        B = adapter[sides["B"]].to(torch.float32)  # [out, r]
        if A.shape[0] != r or B.shape[1] != r:
            raise RuntimeError(f"{stem}: rank mismatch, A={tuple(A.shape)} B={tuple(B.shape)} r={r}")

        delta = scale * (B @ A)  # [out, in], nn.Linear convention
        if fan_in_fan_out:
            delta = delta.T  # base uses Conv1D [in, out]

        W = base[target]
        if W.shape != delta.shape:
            raise RuntimeError(
                f"{stem}: delta {tuple(delta.shape)} does not match base {tuple(W.shape)}"
            )
        if W.dtype != torch.float32:
            raise RuntimeError(f"{stem}: base dtype {W.dtype}, expected float32")
        base[target] = (W.to(torch.float32) + delta).to(torch.float32)
        merged += 1

    # --- required checks -------------------------------------------------
    if merged != 12:
        raise AssertionError(f"expected 12 merged adapter pairs, got {merged}")
    bad = [k for k in base if "lora_" in k]
    if bad:
        raise AssertionError(f"adapter tensors leaked into the output: {bad}")
    probe = "h.0.attn.c_attn.weight"
    if tuple(base[probe].shape) != (768, 2304):
        raise AssertionError(f"{probe} has shape {tuple(base[probe].shape)}, expected (768, 2304)")
    if len(base) != 160:
        raise AssertionError(f"output has {len(base)} tensors, expected 160")

    # --- shard: greedy fill in checkpoint order, oversized tensors alone ---
    def nbytes(t):
        return t.numel() * t.element_size()

    shards, cur, cur_size = [], [], 0
    for name, t in base.items():
        size = nbytes(t)
        if cur and cur_size + size > SHARD_LIMIT:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(name)
        cur_size += size
        if cur_size > SHARD_LIMIT:  # single oversized tensor: seal it alone
            shards.append(cur)
            cur, cur_size = [], 0
    if cur:
        shards.append(cur)

    n = len(shards)
    os.makedirs(OUT_DIR, exist_ok=True)
    weight_map, total_size = {}, 0
    for i, names in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        payload = {k: base[k].contiguous().clone() for k in names}
        save_file(payload, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        for k in names:
            weight_map[k] = fname
            total_size += nbytes(base[k])

    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

    print(f"merged {merged} LoRA pairs (scale={scale}, fan_in_fan_out={fan_in_fan_out})")
    print(f"wrote {len(base)} tensors into {n} shards, {total_size} bytes")


if __name__ == "__main__":
    main()
