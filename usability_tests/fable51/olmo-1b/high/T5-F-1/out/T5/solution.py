#!/usr/bin/env python
"""T5: merge a PEFT LoRA adapter into OLMo-1B base weights, write sharded safetensors.

Works directly on the checkpoint files (no model instantiation): tensors are
streamed shard by shard from the base, the 32 adapted projections get
W += scale * B @ A in float32, and the result is re-sharded to <= 512 MiB of
tensor data per shard with a model.safetensors.index.json.

Uses only torch + safetensors (json for the configs). Fails loudly on every
"Required check" before any output file is written, and re-verifies the
written checkpoint afterwards.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root
BASE_DIR = ROOT / "inputs" / "base"
LORA_DIR = ROOT / "inputs" / "lora"
OUT_DIR = ROOT / "out" / "T5"

SHARD_BUDGET = 512 * 1024 * 1024  # 536,870,912 bytes of tensor data per shard
# Tensors bigger than half the budget get a shard of their own (embed_tokens, lm_head).
SOLO_THRESHOLD = SHARD_BUDGET // 2

EXPECTED_PAIRS = 32
EXPECTED_TENSORS = 114
DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "I64": 8, "I32": 4, "I8": 1, "U8": 1, "BOOL": 1}


def fail(msg: str) -> None:
    raise SystemExit(f"CHECK FAILED: {msg}")


def natural_key(name: str):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def main() -> None:
    # ---------------------------------------------------------------- inputs
    index = json.loads((BASE_DIR / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    base_names = sorted(weight_map, key=natural_key)
    if len(base_names) != EXPECTED_TENSORS:
        fail(f"base has {len(base_names)} tensors, expected {EXPECTED_TENSORS}")

    base_files = {shard: safe_open(BASE_DIR / shard, "pt", device="cpu")
                  for shard in sorted(set(weight_map.values()))}
    # Shapes/dtypes from headers only (no data read yet).
    meta: dict[str, tuple[list[int], str]] = {}
    for name in base_names:
        sl = base_files[weight_map[name]].get_slice(name)
        meta[name] = (list(sl.get_shape()), sl.get_dtype())

    cfg = json.loads((LORA_DIR / "adapter_config.json").read_text())
    if cfg.get("peft_type") != "LORA":
        fail(f"adapter is not LoRA: {cfg.get('peft_type')}")
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scale = alpha / math.sqrt(r) if cfg.get("use_rslora", False) else alpha / r
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    if cfg.get("bias", "none") != "none":
        fail("adapter has bias terms; this script only handles bias='none'")
    print(f"adapter: r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

    lora = safe_open(LORA_DIR / "adapter_model.safetensors", "pt", device="cpu")
    lora_names = list(lora.keys())

    # ------------------------------------------- map adapter names -> base names
    pat = re.compile(r"^base_model\.model\.(?P<base>.+)\.lora_(?P<which>[AB])\.weight$")
    pairs: dict[str, dict[str, str]] = {}
    for n in lora_names:
        m = pat.match(n)
        if not m:
            fail(f"unrecognised adapter tensor name: {n}")
        pairs.setdefault(m["base"] + ".weight", {})[m["which"]] = n
    for base_name, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            fail(f"incomplete LoRA pair for {base_name}: {sorted(ab)}")
        if base_name not in meta:
            fail(f"adapter targets {base_name}, which is not in the base checkpoint")
        w_shape, w_dtype = meta[base_name]
        a_shape = list(lora.get_slice(ab["A"]).get_shape())
        b_shape = list(lora.get_slice(ab["B"]).get_shape())
        out_f, in_f = (w_shape[1], w_shape[0]) if fan_in_fan_out else (w_shape[0], w_shape[1])
        if a_shape != [r, in_f] or b_shape != [out_f, r]:
            fail(f"{base_name}: A{a_shape} B{b_shape} do not fit W{w_shape} with r={r}")
        if w_dtype != "F32":
            fail(f"{base_name}: base dtype {w_dtype}, expected F32")
    if len(pairs) != EXPECTED_PAIRS:
        fail(f"found {len(pairs)} adapter pairs, expected {EXPECTED_PAIRS}")
    if 2 * len(pairs) != len(lora_names):
        fail(f"{len(lora_names)} adapter tensors but {len(pairs)} pairs; leftovers exist")
    print(f"matched {len(pairs)} LoRA pairs to base tensors")

    # ------------------------------------------------------- output name checks
    out_names = base_names  # merged weights keep their names; nothing added
    bad = [n for n in out_names if "lora_" in n]
    if bad:
        fail(f"adapter names would leak into the output: {bad[:3]}")
    if meta["model.layers.0.self_attn.q_proj.weight"][0] != [2048, 2048]:
        fail(f"layer0 q_proj shape {meta['model.layers.0.self_attn.q_proj.weight'][0]}")
    if len(out_names) != EXPECTED_TENSORS:
        fail(f"output would have {len(out_names)} tensors, expected {EXPECTED_TENSORS}")

    # -------------------------------------------------------------- shard plan
    def nbytes(name: str) -> int:
        shape, dt = meta[name]
        return math.prod(shape) * DTYPE_BYTES[dt]

    shards: list[list[str]] = []
    cur: list[str] = []
    cur_size = 0
    for name in out_names:
        sz = nbytes(name)
        if sz > SHARD_BUDGET:
            fail(f"{name} ({sz} bytes) exceeds the shard budget on its own")
        if sz > SOLO_THRESHOLD:
            if cur:
                shards.append(cur)
            shards.append([name])
            cur, cur_size = [], 0
            continue
        if cur and cur_size + sz > SHARD_BUDGET:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(name)
        cur_size += sz
    if cur:
        shards.append(cur)
    n_shards = len(shards)
    shard_file = {i: f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)}
    for i, names in enumerate(shards):
        tot = sum(nbytes(n) for n in names)
        if tot > SHARD_BUDGET:
            fail(f"planned shard {shard_file[i]} holds {tot} bytes > budget")
    print(f"planned {n_shards} shards")

    # ------------------------------------------------------- destination state
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in OUT_DIR.iterdir()
                if p.suffix == ".safetensors" or p.name == "model.safetensors.index.json"]
    if existing:
        fail(f"destination already holds checkpoint files: {existing[:3]} (remove them first)")

    # ---------------------------------------------------------- merge + write
    merged_count = 0
    out_weight_map: dict[str, str] = {}
    total_size = 0
    for i, names in enumerate(shards):
        tensors: dict[str, torch.Tensor] = {}
        for name in names:
            w = base_files[weight_map[name]].get_tensor(name)
            if name in pairs:
                a = lora.get_tensor(pairs[name]["A"]).to(torch.float32)
                b = lora.get_tensor(pairs[name]["B"]).to(torch.float32)
                delta = scale * (b @ a)
                if fan_in_fan_out:
                    delta = delta.T
                if delta.shape != w.shape:
                    fail(f"{name}: delta {tuple(delta.shape)} vs W {tuple(w.shape)}")
                w = (w.to(torch.float32) + delta).to(torch.float32)
                merged_count += 1
            tensors[name] = w.contiguous()
            out_weight_map[name] = shard_file[i]
            total_size += w.numel() * w.element_size()
        save_file(tensors, OUT_DIR / shard_file[i], metadata={"format": "pt"})
        print(f"wrote {shard_file[i]}: {len(names)} tensors, "
              f"{sum(t.numel() * t.element_size() for t in tensors.values())} bytes")
        del tensors
    if merged_count != EXPECTED_PAIRS:
        fail(f"merged {merged_count} tensors, expected {EXPECTED_PAIRS}")
    (OUT_DIR / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": out_weight_map},
                   indent=2, sort_keys=True) + "\n")

    # ------------------------------------------------------------ verification
    print("verifying written checkpoint ...")
    idx2 = json.loads((OUT_DIR / "model.safetensors.index.json").read_text())
    wm2 = idx2["weight_map"]
    if set(wm2) != set(base_names):
        fail("index weight_map key set differs from base")
    if any("lora_" in n for n in wm2):
        fail("lora_ name in output index")
    seen: set[str] = set()
    for fname in sorted(set(wm2.values())):
        with safe_open(OUT_DIR / fname, "pt", device="cpu") as f:
            keys = list(f.keys())
            data = 0
            for k in keys:
                if wm2.get(k) != fname:
                    fail(f"{k} in {fname} but index says {wm2.get(k)}")
                sl = f.get_slice(k)
                shape, dt = list(sl.get_shape()), sl.get_dtype()
                if (shape, dt) != tuple(meta[k]) and [shape, dt] != list(meta[k]):
                    fail(f"{k}: shape/dtype {shape} {dt} vs base {meta[k]}")
                data += math.prod(shape) * DTYPE_BYTES[dt]
                t = f.get_tensor(k)
                ref = base_files[weight_map[k]].get_tensor(k)
                if k in pairs:
                    a = lora.get_tensor(pairs[k]["A"]); b = lora.get_tensor(pairs[k]["B"])
                    d = scale * (b @ a)
                    exp = ref + (d.T if fan_in_fan_out else d)
                    rel = (t.double() - exp.double()).norm() / exp.double().norm()
                    if not rel <= 1e-6:
                        fail(f"{k}: merged relative error {rel}")
                    if torch.equal(t, ref):
                        fail(f"{k}: merged tensor is identical to base (delta not applied)")
                else:
                    if not torch.equal(t, ref):
                        fail(f"{k}: unchanged tensor differs from base")
                seen.add(k)
            if data > SHARD_BUDGET:
                fail(f"{fname} holds {data} bytes of tensor data > {SHARD_BUDGET}")
            print(f"  {fname}: {len(keys)} tensors, {data} bytes OK")
    if len(seen) != EXPECTED_TENSORS:
        fail(f"output has {len(seen)} tensors, expected {EXPECTED_TENSORS}")
    if seen != set(wm2):
        fail("index lists tensors that are not in the shard files")
    with safe_open(OUT_DIR / wm2["model.layers.0.self_attn.q_proj.weight"], "pt") as f:
        shp = list(f.get_slice("model.layers.0.self_attn.q_proj.weight").get_shape())
        if shp != [2048, 2048]:
            fail(f"layer0 q_proj output shape {shp}")
    print(f"OK: {len(seen)} tensors in {len(set(wm2.values()))} shards, "
          f"{merged_count} merged, total_size={total_size}")


if __name__ == "__main__":
    sys.exit(main())
