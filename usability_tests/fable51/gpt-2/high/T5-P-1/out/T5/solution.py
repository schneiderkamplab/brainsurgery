"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and write a sharded safetensors checkpoint."""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
ADAPTER = os.path.join(ROOT, "inputs", "lora", "adapter_model.safetensors")
ADAPTER_CFG = os.path.join(ROOT, "inputs", "lora", "adapter_config.json")
OUT_DIR = os.path.join(ROOT, "out", "T5")
SHARD_BUDGET = 100 * 1024 * 1024  # 104,857,600 bytes of tensor data per shard
EXPECTED_PAIRS = 12
EXPECTED_TENSORS = 160


def fail(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    with open(ADAPTER_CFG) as f:
        cfg = json.load(f)
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    scale = alpha / r
    print(f"r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

    # Load base fully into memory (500 MB, fine on CPU).
    state = {}
    with safe_open(BASE, framework="pt") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
    print(f"base tensors: {len(state)}")

    adapter = {}
    with safe_open(ADAPTER, framework="pt") as f:
        for k in f.keys():
            adapter[k] = f.get_tensor(k)
    print(f"adapter tensors: {len(adapter)}")

    # Pair lora_A / lora_B by module name; strip the PEFT prefix to get the base name.
    pat = re.compile(r"^(?:base_model\.model\.)?(.*)\.lora_([AB])\.weight$")
    pairs = {}
    for k in adapter:
        m = pat.match(k)
        if not m:
            fail(f"unrecognised adapter tensor name: {k}")
        pairs.setdefault(m.group(1), {})[m.group(2)] = k
    for mod, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            fail(f"incomplete adapter pair for {mod}: {sorted(ab)}")

    merged = 0
    for mod, ab in sorted(pairs.items()):
        A = adapter[ab["A"]].to(torch.float32)  # [r, in]
        B = adapter[ab["B"]].to(torch.float32)  # [out, r]
        base_name = f"{mod}.weight"
        if base_name not in state:
            fail(f"adapter targets {base_name} which is not in the base checkpoint")
        W = state[base_name]
        if W.dtype != torch.float32:
            fail(f"{base_name} has dtype {W.dtype}, expected float32")
        if A.shape[0] != r or B.shape[1] != r:
            fail(f"rank mismatch for {mod}: A{tuple(A.shape)} B{tuple(B.shape)} r={r}")
        delta = scale * (B @ A)  # [out, in], nn.Linear convention
        if fan_in_fan_out:
            delta = delta.T  # base is Conv1D [in, out]
        if delta.shape != W.shape:
            fail(f"delta shape {tuple(delta.shape)} != base shape {tuple(W.shape)} for {base_name}")
        state[base_name] = (W + delta).contiguous()
        merged += 1
        print(f"merged {base_name}: |delta|_F={delta.norm().item():.4f}")

    # ---- required checks ----
    if merged != EXPECTED_PAIRS:
        fail(f"expected {EXPECTED_PAIRS} merged adapter pairs, got {merged}")
    lora_keys = [k for k in state if "lora_" in k]
    if lora_keys:
        fail(f"adapter tensors leaked into output: {lora_keys[:5]}")
    probe = "h.0.attn.c_attn.weight"
    if probe not in state or tuple(state[probe].shape) != (768, 2304):
        fail(f"{probe} missing or wrong shape: {tuple(state[probe].shape) if probe in state else None}")
    if state[probe].dtype != torch.float32:
        fail(f"{probe} dtype {state[probe].dtype} != float32")
    if len(state) != EXPECTED_TENSORS:
        fail(f"expected {EXPECTED_TENSORS} output tensors, got {len(state)}")

    # ---- sharding: greedy fill in base key order; oversized tensors go alone ----
    shards = []
    cur, cur_size = {}, 0
    for k, t in state.items():
        nbytes = t.numel() * t.element_size()
        if cur and cur_size + nbytes > SHARD_BUDGET:
            shards.append(cur)
            cur, cur_size = {}, 0
        cur[k] = t
        cur_size += nbytes
    if cur:
        shards.append(cur)
    n = len(shards)
    for s in shards:
        size = sum(t.numel() * t.element_size() for t in s.values())
        if size > SHARD_BUDGET and len(s) != 1:
            fail(f"shard exceeds budget with {len(s)} tensors ({size} bytes)")

    os.makedirs(OUT_DIR, exist_ok=True)
    weight_map = {}
    total_size = 0
    for i, s in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file(s, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        for k, t in s.items():
            weight_map[k] = fname
            total_size += t.numel() * t.element_size()
        print(f"wrote {fname}: {len(s)} tensors, "
              f"{sum(t.numel() * t.element_size() for t in s.values())} bytes")
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"done: {len(weight_map)} tensors in {n} shards, total_size={total_size}")


if __name__ == "__main__":
    main()
