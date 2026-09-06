"""T5: merge a PEFT LoRA adapter into Pythia-1B base weights and write a sharded checkpoint."""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
LORA = os.path.join(ROOT, "inputs", "lora", "adapter_model.safetensors")
LORA_CFG = os.path.join(ROOT, "inputs", "lora", "adapter_config.json")
OUT_DIR = os.path.join(ROOT, "out", "T5")

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912
PREFIX = "base_model.model."
EXPECTED_PAIRS = 16
EXPECTED_TENSORS = 244
# TASK.md: the embedding matrices are stored alone in their own shard.
STANDALONE = {"gpt_neox.embed_in.weight", "embed_out.weight"}


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    with open(LORA_CFG) as f:
        cfg = json.load(f)
    r, alpha = cfg["r"], cfg["lora_alpha"]
    fan_in_fan_out = cfg.get("fan_in_fan_out", False)
    scale = alpha / r
    print(f"r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

    # Load base (keep key order of the file).
    base = {}
    with safe_open(BASE, framework="pt") as f:
        base_keys = list(f.keys())
        for k in base_keys:
            base[k] = f.get_tensor(k)
    print(f"base: {len(base)} tensors")

    # Load adapter and group into (A, B) pairs keyed by the base tensor name.
    pairs = {}
    with safe_open(LORA, framework="pt") as f:
        for k in f.keys():
            if not k.startswith(PREFIX):
                fail(f"unexpected adapter key without PEFT prefix: {k}")
            stem = k[len(PREFIX):]
            if stem.endswith(".lora_A.weight"):
                which, mod = "A", stem[: -len(".lora_A.weight")]
            elif stem.endswith(".lora_B.weight"):
                which, mod = "B", stem[: -len(".lora_B.weight")]
            else:
                fail(f"unexpected adapter key: {k}")
            pairs.setdefault(mod, {})[which] = f.get_tensor(k)

    for mod, d in pairs.items():
        if set(d) != {"A", "B"}:
            fail(f"incomplete adapter pair for {mod}: {sorted(d)}")
    if len(pairs) != EXPECTED_PAIRS:
        fail(f"expected {EXPECTED_PAIRS} adapter pairs, found {len(pairs)}")

    merged = 0
    for mod, d in sorted(pairs.items()):
        name = mod + ".weight"
        if name not in base:
            fail(f"adapter target {name} not in base checkpoint")
        w = base[name]
        A, B = d["A"].float(), d["B"].float()
        delta = scale * (B @ A)  # [out, in]
        if fan_in_fan_out:
            delta = delta.T
        if delta.shape != w.shape:
            fail(f"{name}: delta shape {tuple(delta.shape)} != weight shape {tuple(w.shape)}")
        base[name] = (w.float() + delta).to(w.dtype).contiguous()
        merged += 1
    if merged != EXPECTED_PAIRS:
        fail(f"merged {merged} pairs, expected {EXPECTED_PAIRS}")
    print(f"merged {merged} adapter pairs")

    # Required checks before writing.
    lora_keys = [k for k in base if "lora_" in k]
    if lora_keys:
        fail(f"adapter tensors present in output: {lora_keys[:5]}")
    qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tuple(base[qkv0].shape) != (6144, 2048):
        fail(f"{qkv0} has shape {tuple(base[qkv0].shape)}, expected (6144, 2048)")
    if base[qkv0].dtype != torch.float16:
        fail(f"{qkv0} has dtype {base[qkv0].dtype}, expected float16")
    if len(base) != EXPECTED_TENSORS:
        fail(f"output has {len(base)} tensors, expected {EXPECTED_TENSORS}")
    if set(base) != set(base_keys):
        fail("output key set differs from base key set")

    # Plan shards: greedy in file order, cap on tensor bytes; oversized tensors go alone.
    shards = []  # list of lists of names
    cur, cur_bytes = [], 0
    for k in base_keys:
        nb = base[k].numel() * base[k].element_size()
        if nb > MAX_SHARD_BYTES or k in STANDALONE:
            if cur:
                shards.append(cur)
                cur, cur_bytes = [], 0
            shards.append([k])
            continue
        if cur and cur_bytes + nb > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(k)
        cur_bytes += nb
    if cur:
        shards.append(cur)

    n = len(shards)
    weight_map = {}
    total_size = 0
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, names in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        tensors = {k: base[k] for k in names}
        nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
        if nbytes > MAX_SHARD_BYTES and len(tensors) > 1:
            fail(f"shard {fname} exceeds cap with {nbytes} bytes")
        save_file(tensors, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        for k in names:
            weight_map[k] = fname
        total_size += nbytes
        print(f"wrote {fname}: {len(names)} tensors, {nbytes} bytes")

    if len(weight_map) != EXPECTED_TENSORS:
        fail(f"weight_map has {len(weight_map)} entries, expected {EXPECTED_TENSORS}")
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    print(f"wrote index: {n} shards, {len(weight_map)} tensors, total_size={total_size}")

    # Read back and verify against the inputs.
    seen = {}
    for fname in sorted(set(weight_map.values())):
        with safe_open(os.path.join(OUT_DIR, fname), framework="pt") as f:
            for k in f.keys():
                if k in seen:
                    fail(f"{k} appears in both {seen[k]} and {fname}")
                seen[k] = fname
                if weight_map.get(k) != fname:
                    fail(f"index maps {k} to {weight_map.get(k)}, found in {fname}")
                t = f.get_tensor(k)
                if "lora_" in k:
                    fail(f"adapter tensor {k} in output")
                if k + "" in [m + ".weight" for m in pairs]:
                    mod = k[: -len(".weight")]
                    A, B = pairs[mod]["A"].float(), pairs[mod]["B"].float()
                    with safe_open(BASE, framework="pt") as fb:
                        ref = fb.get_tensor(k).float() + scale * (B @ A)
                    err = (t.float() - ref).norm() / ref.norm()
                    if t.dtype != torch.float16 or err > 1e-3:
                        fail(f"{k}: dtype {t.dtype}, rel err {err:.3e}")
                else:
                    with safe_open(BASE, framework="pt") as fb:
                        ref = fb.get_tensor(k)
                    if t.dtype != ref.dtype or t.shape != ref.shape or not torch.equal(t, ref):
                        fail(f"{k}: unchanged tensor differs from base")
    if len(seen) != EXPECTED_TENSORS:
        fail(f"read back {len(seen)} tensors, expected {EXPECTED_TENSORS}")
    print(f"verified {len(seen)} tensors against inputs")


if __name__ == "__main__":
    main()
