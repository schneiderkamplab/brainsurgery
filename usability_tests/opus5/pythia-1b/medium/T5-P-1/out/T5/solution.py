"""T5: merge a PEFT LoRA adapter into Pythia-1B and export a sharded checkpoint."""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
LORA = os.path.join(ROOT, "inputs", "lora", "adapter_model.safetensors")
LORA_CFG = os.path.join(ROOT, "inputs", "lora", "adapter_config.json")
OUT_DIR = os.path.join(ROOT, "out", "T5")

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912
PREFIX = "base_model.model."
EXPECTED_PAIRS = 16
EXPECTED_TENSORS = 244

DTYPE_BYTES = {
    torch.float16: 2, torch.bfloat16: 2, torch.float32: 4, torch.float64: 8,
    torch.int64: 8, torch.int32: 4, torch.int16: 2, torch.int8: 1,
    torch.uint8: 1, torch.bool: 1,
}


def fail(msg):
    raise SystemExit(f"CHECK FAILED: {msg}")


def main():
    with open(LORA_CFG) as fh:
        cfg = json.load(fh)
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    fan_in_fan_out = cfg.get("fan_in_fan_out", False)
    scale = alpha / r
    if fan_in_fan_out:
        fail("fan_in_fan_out is true; this script assumes the [out, in] layout")
    print(f"adapter: r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

    base = safe_open(BASE, framework="pt")
    base_keys = list(base.keys())
    lora = safe_open(LORA, framework="pt")
    lora_keys = list(lora.keys())

    # Map adapter A/B pairs onto base weight names.
    pairs = {}
    for key in lora_keys:
        for tag in ("lora_A", "lora_B"):
            suffix = f".{tag}.weight"
            if key.endswith(suffix):
                stem = key[: -len(suffix)]
                if not stem.startswith(PREFIX):
                    fail(f"adapter tensor {key!r} lacks the {PREFIX!r} prefix")
                target = stem[len(PREFIX):] + ".weight"
                pairs.setdefault(target, {})[tag] = key
                break
        else:
            fail(f"unrecognised adapter tensor name: {key!r}")

    # Check 1: exactly 16 complete pairs, each with a base weight to merge into.
    complete = {t: p for t, p in pairs.items() if set(p) == {"lora_A", "lora_B"}}
    if len(pairs) != len(complete):
        fail(f"incomplete adapter pairs: {sorted(set(pairs) - set(complete))}")
    if len(complete) != EXPECTED_PAIRS:
        fail(f"expected {EXPECTED_PAIRS} adapter pairs, found {len(complete)}")
    missing = [t for t in complete if t not in base_keys]
    if missing:
        fail(f"adapter targets absent from the base checkpoint: {missing}")

    merged = {}
    for target, p in sorted(complete.items()):
        w = base.get_tensor(target)
        a = lora.get_tensor(p["lora_A"])
        b = lora.get_tensor(p["lora_B"])
        if a.shape[0] != r or b.shape[1] != r:
            fail(f"{target}: factor ranks {tuple(a.shape)} / {tuple(b.shape)} disagree with r={r}")
        delta = (b.to(torch.float32) @ a.to(torch.float32)) * scale
        if delta.shape != w.shape:
            fail(f"{target}: delta {tuple(delta.shape)} != base {tuple(w.shape)}")
        out = (w.to(torch.float32) + delta).to(w.dtype).contiguous()
        merged[target] = out
        print(f"merged {target}: {tuple(out.shape)} {out.dtype}")
    if len(merged) != EXPECTED_PAIRS:
        fail(f"merged {len(merged)} tensors, expected {EXPECTED_PAIRS}")

    # Output key set = base key set; adapter tensors are dropped.
    out_keys = list(base_keys)

    # Check 2: no adapter or intermediate tensor survives.
    leaked = [k for k in out_keys if "lora_" in k]
    if leaked:
        fail(f"adapter tensor names present in the output: {leaked}")

    # Check 3: the probe weight keeps its shape.
    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    if probe not in merged:
        fail(f"{probe} was not merged")
    if tuple(merged[probe].shape) != (6144, 2048):
        fail(f"{probe} has shape {tuple(merged[probe].shape)}, expected (6144, 2048)")
    if merged[probe].dtype != torch.float16:
        fail(f"{probe} has dtype {merged[probe].dtype}, expected torch.float16")

    # Check 4: exactly 244 tensors.
    if len(out_keys) != EXPECTED_TENSORS:
        fail(f"output has {len(out_keys)} tensors, expected {EXPECTED_TENSORS}")

    # Sizes for the shard plan (merged tensors keep the base dtype and shape).
    st_dtype_bytes = {
        "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
        "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    }
    sizes = {}
    for k in out_keys:
        t = merged.get(k)
        if t is not None:
            sizes[k] = t.numel() * DTYPE_BYTES[t.dtype]
            continue
        sl = base.get_slice(k)
        st_dtype = sl.get_dtype()
        if st_dtype not in st_dtype_bytes:
            fail(f"{k}: unhandled dtype {st_dtype}")
        numel = 1
        for d in sl.get_shape():
            numel *= d
        sizes[k] = numel * st_dtype_bytes[st_dtype]
    total_size = sum(sizes.values())

    # Greedy shard plan in checkpoint key order; a tensor over budget gets its own shard.
    shards = []
    cur, cur_bytes = [], 0
    for k in out_keys:
        n = sizes[k]
        if cur and cur_bytes + n > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(k)
        cur_bytes += n
    if cur:
        shards.append(cur)
    for i, sh in enumerate(shards):
        b = sum(sizes[k] for k in sh)
        if b > MAX_SHARD_BYTES and len(sh) > 1:
            fail(f"shard {i} holds {b} bytes over the {MAX_SHARD_BYTES} budget")

    total_shards = len(shards)
    names = [f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors" for i in range(total_shards)]
    weight_map = {k: names[i] for i, sh in enumerate(shards) for k in sh}
    if len(weight_map) != EXPECTED_TENSORS:
        fail(f"weight_map covers {len(weight_map)} tensors, expected {EXPECTED_TENSORS}")

    os.makedirs(OUT_DIR, exist_ok=True)
    for i, sh in enumerate(shards):
        data = {}
        for k in sh:
            data[k] = merged[k] if k in merged else base.get_tensor(k).contiguous()
        save_file(data, os.path.join(OUT_DIR, names[i]), metadata={"format": "pt"})
        print(f"wrote {names[i]}: {len(sh)} tensors, {sum(sizes[k] for k in sh)} bytes")
        del data

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {k: weight_map[k] for k in out_keys},
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=False)
        fh.write("\n")

    # Verify what landed on disk.
    seen = []
    for name in names:
        with safe_open(os.path.join(OUT_DIR, name), framework="pt") as fh:
            seen.extend(fh.keys())
    if sorted(seen) != sorted(out_keys):
        fail("tensors on disk do not match the planned key set")
    if any("lora_" in k for k in seen):
        fail("adapter tensor names present on disk")
    if len(seen) != EXPECTED_TENSORS:
        fail(f"{len(seen)} tensors on disk, expected {EXPECTED_TENSORS}")
    print(f"OK: {len(seen)} tensors in {total_shards} shards, {total_size} bytes total")


if __name__ == "__main__":
    main()
