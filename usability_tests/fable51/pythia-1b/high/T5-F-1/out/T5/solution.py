#!/usr/bin/env python
"""T5: merge a PEFT LoRA adapter into Pythia-1B base weights and write a sharded
safetensors checkpoint. Plain torch + safetensors; no model instantiation.

Usage: python out/T5/solution.py
"""
import json
import math
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "inputs" / "base" / "model.safetensors"
LORA = ROOT / "inputs" / "lora" / "adapter_model.safetensors"
LORA_CFG = ROOT / "inputs" / "lora" / "adapter_config.json"
OUT = ROOT / "out" / "T5"

SHARD_BUDGET = 512 * 1024 * 1024  # 536,870,912 bytes of tensor data per shard
EXPECTED_PAIRS = 16
EXPECTED_TENSORS = 244
PROBE = "gpt_neox.layers.0.attention.query_key_value.weight"
PROBE_SHAPE = (6144, 2048)
# TASK.md: these are stored alone in their own shard.
STANDALONE = {"gpt_neox.embed_in.weight", "embed_out.weight"}


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    cfg = json.loads(LORA_CFG.read_text())
    if cfg.get("peft_type") != "LORA":
        fail(f"unexpected peft_type {cfg.get('peft_type')!r}")
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scale = alpha / r
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    print(f"r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

    # --- collect adapter pairs, map to base names ---------------------------
    pat = re.compile(r"^base_model\.model\.(?P<base>.+)\.lora_(?P<ab>[AB])\.weight$")
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(LORA), "pt") as f:
        for k in f.keys():
            m = pat.match(k)
            if m is None:
                fail(f"adapter tensor with unrecognised name: {k}")
            pairs.setdefault(m["base"] + ".weight", {})[m["ab"]] = f.get_tensor(k)
    for name, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            fail(f"incomplete adapter pair for {name}: {sorted(ab)}")
    if len(pairs) != EXPECTED_PAIRS:
        fail(f"expected {EXPECTED_PAIRS} adapter pairs, found {len(pairs)}")

    # --- load base, merge -----------------------------------------------------
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(BASE), "pt") as f:
        base_keys = list(f.keys())
        for k in base_keys:
            tensors[k] = f.get_tensor(k)

    merged = 0
    for name, ab in pairs.items():
        if name not in tensors:
            fail(f"adapter targets missing base tensor {name}")
        w = tensors[name]
        A, B = ab["A"].float(), ab["B"].float()
        if A.shape[0] != r or B.shape[1] != r:
            fail(f"{name}: rank mismatch A{tuple(A.shape)} B{tuple(B.shape)} r={r}")
        delta = scale * (B @ A)  # [out, in]
        if fan_in_fan_out:
            delta = delta.T
        if delta.shape != w.shape:
            fail(f"{name}: delta shape {tuple(delta.shape)} != base {tuple(w.shape)}")
        tensors[name] = (w.float() + delta).to(w.dtype).contiguous()
        merged += 1
    if merged != EXPECTED_PAIRS:
        fail(f"merged {merged} pairs, expected {EXPECTED_PAIRS}")

    # --- required checks before writing --------------------------------------
    lora_keys = [k for k in tensors if "lora_" in k]
    if lora_keys:
        fail(f"adapter tensors would be written: {lora_keys[:3]}")
    if PROBE not in tensors or tuple(tensors[PROBE].shape) != PROBE_SHAPE:
        fail(f"{PROBE} shape {tuple(tensors.get(PROBE, torch.empty(0)).shape)} != {PROBE_SHAPE}")
    if tensors[PROBE].dtype != torch.float16:
        fail(f"{PROBE} dtype {tensors[PROBE].dtype} != float16")
    if len(tensors) != EXPECTED_TENSORS:
        fail(f"output has {len(tensors)} tensors, expected {EXPECTED_TENSORS}")
    if set(tensors) != set(base_keys):
        fail("output key set differs from base key set")

    # --- shard: greedy in base order, budget on tensor bytes -----------------
    def nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for k in base_keys:
        n = nbytes(tensors[k])
        if n > SHARD_BUDGET or k in STANDALONE:
            if cur:
                shards.append(cur)
            shards.append([k])  # oversized / standalone tensor alone in its own shard
            cur, cur_bytes = [], 0
            continue
        if cur and cur_bytes + n > SHARD_BUDGET:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(k)
        cur_bytes += n
    if cur:
        shards.append(cur)

    for s in shards:
        tot = sum(nbytes(tensors[k]) for k in s)
        if tot > SHARD_BUDGET and len(s) != 1:
            fail(f"shard exceeds budget: {tot} bytes, {len(s)} tensors")

    # --- write ---------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.safetensors"):
        old.unlink()
    idx_path = OUT / "model.safetensors.index.json"
    if idx_path.exists():
        idx_path.unlink()

    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, s in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        save_file({k: tensors[k] for k in s}, str(OUT / fname), metadata={"format": "pt"})
        for k in s:
            weight_map[k] = fname
            total_size += nbytes(tensors[k])
        print(f"{fname}: {len(s)} tensors, {sum(nbytes(tensors[k]) for k in s)} bytes")
    idx_path.write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2)
        + "\n"
    )

    # --- post-write verification ---------------------------------------------
    seen: dict[str, torch.Tensor] = {}
    for fname in sorted(set(weight_map.values())):
        with safe_open(str(OUT / fname), "pt") as f:
            for k in f.keys():
                if k in seen:
                    fail(f"{k} written to more than one shard")
                seen[k] = f.get_tensor(k)
    if set(seen) != set(base_keys) or len(seen) != EXPECTED_TENSORS:
        fail("written key set differs from expected")
    if any("lora_" in k for k in seen):
        fail("lora_ tensor found in written output")
    with safe_open(str(BASE), "pt") as f:
        for k in base_keys:
            t = f.get_tensor(k)
            if k in pairs:
                ref = (t.float() + scale * (pairs[k]["B"].float() @ pairs[k]["A"].float())).half()
                rel = (seen[k].float() - ref.float()).norm() / ref.float().norm()
                if rel > 1e-3:
                    fail(f"{k}: relative error {rel:.3e}")
            else:
                if seen[k].dtype != t.dtype or seen[k].shape != t.shape or not torch.equal(seen[k], t):
                    fail(f"{k}: unchanged tensor differs from base")
    print(f"OK: {len(seen)} tensors in {n_shards} shards, {merged} LoRA pairs merged")


if __name__ == "__main__":
    main()
