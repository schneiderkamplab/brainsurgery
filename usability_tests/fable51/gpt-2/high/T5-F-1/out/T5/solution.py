"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and write a sharded safetensors checkpoint.

Plain safetensors + torch. The transformers/peft `merge_and_unload` route was rejected because
`save_pretrained` would drop the `h.<i>.attn.bias` mask buffers and add a `transformer.` prefix,
breaking the exact key set required by the task.
"""
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "inputs" / "base" / "model.safetensors"
ADAPTER = ROOT / "inputs" / "lora" / "adapter_model.safetensors"
ADAPTER_CFG = ROOT / "inputs" / "lora" / "adapter_config.json"
OUT = ROOT / "out" / "T5"
SHARD_BUDGET = 100 * 1024 * 1024  # 104,857,600 bytes of tensor data per shard

EXPECTED_PAIRS = 12
EXPECTED_TENSORS = 160


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    cfg = json.loads(ADAPTER_CFG.read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / r
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    print(f"adapter: r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

    # --- load base ---
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(BASE), "pt") as f:
        base_keys = list(f.keys())
        for k in base_keys:
            tensors[k] = f.get_tensor(k)
    print(f"base: {len(tensors)} tensors")

    # --- load adapter and pair A/B ---
    pat = re.compile(r"^base_model\.model\.(?P<target>.+)\.lora_(?P<ab>[AB])\.weight$")
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(ADAPTER), "pt") as f:
        adapter_keys = list(f.keys())
        for k in adapter_keys:
            m = pat.match(k)
            if m is None:
                fail(f"unrecognised adapter tensor name {k!r}")
            pairs.setdefault(m["target"], {})[m["ab"]] = f.get_tensor(k)
    for target, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            fail(f"incomplete adapter pair for {target}: have {sorted(ab)}")
    if len(pairs) != EXPECTED_PAIRS:
        fail(f"expected {EXPECTED_PAIRS} adapter pairs, found {len(pairs)}")

    # --- merge ---
    merged = 0
    for target, ab in sorted(pairs.items()):
        name = f"{target}.weight"
        if name not in tensors:
            fail(f"adapter target {target} has no base tensor {name!r}")
        A, B = ab["A"], ab["B"]
        W = tensors[name]
        if A.dtype != torch.float32 or B.dtype != torch.float32 or W.dtype != torch.float32:
            fail(f"{name}: expected float32 everywhere, got W={W.dtype} A={A.dtype} B={B.dtype}")
        if A.shape[0] != r or B.shape[1] != r:
            fail(f"{name}: rank mismatch A={tuple(A.shape)} B={tuple(B.shape)} r={r}")
        delta = (B @ A) * scale  # [out, in], nn.Linear convention
        if fan_in_fan_out:
            delta = delta.T  # base is Conv1D [in, out]
        if delta.shape != W.shape:
            fail(f"{name}: delta shape {tuple(delta.shape)} != base shape {tuple(W.shape)}")
        tensors[name] = (W + delta).contiguous()
        merged += 1
    if merged != EXPECTED_PAIRS:
        fail(f"merged {merged} pairs, expected {EXPECTED_PAIRS}")
    print(f"merged {merged} adapter pairs")

    # --- required checks, before writing ---
    lora_names = [k for k in tensors if "lora_" in k]
    if lora_names:
        fail(f"adapter tensors present in output: {lora_names[:5]}")
    probe = tensors["h.0.attn.c_attn.weight"]
    if tuple(probe.shape) != (768, 2304):
        fail(f"h.0.attn.c_attn.weight has shape {tuple(probe.shape)}, expected (768, 2304)")
    if probe.dtype != torch.float32:
        fail(f"h.0.attn.c_attn.weight dtype {probe.dtype}, expected float32")
    if len(tensors) != EXPECTED_TENSORS:
        fail(f"output has {len(tensors)} tensors, expected {EXPECTED_TENSORS}")
    if list(tensors) != base_keys:
        fail("output key set differs from base key set")

    # --- plan shards: greedy in base key order, oversized tensors alone ---
    shards: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for k in base_keys:
        nbytes = tensors[k].numel() * tensors[k].element_size()
        if nbytes > SHARD_BUDGET:
            if cur:
                shards.append(cur)
                cur, cur_bytes = [], 0
            shards.append([k])
            continue
        if cur_bytes + nbytes > SHARD_BUDGET:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(k)
        cur_bytes += nbytes
    if cur:
        shards.append(cur)

    n = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    plan: list[tuple[str, list[str], int]] = []
    for i, keys in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        size = sum(tensors[k].numel() * tensors[k].element_size() for k in keys)
        if size > SHARD_BUDGET and len(keys) != 1:
            fail(f"shard {fname} holds {size} bytes across {len(keys)} tensors")
        for k in keys:
            weight_map[k] = fname
        total_size += size
        plan.append((fname, keys, size))
    if len(weight_map) != EXPECTED_TENSORS:
        fail(f"weight_map has {len(weight_map)} entries, expected {EXPECTED_TENSORS}")

    # --- write ---
    OUT.mkdir(parents=True, exist_ok=True)
    stale = [p for p in OUT.glob("*.safetensors")] + [p for p in OUT.glob("*.index.json")]
    for p in stale:
        p.unlink()
    for fname, keys, size in plan:
        save_file({k: tensors[k] for k in keys}, str(OUT / fname), metadata={"format": "pt"})
        print(f"wrote {fname}: {len(keys)} tensors, {size} bytes")
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (OUT / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote model.safetensors.index.json ({n} shards, total_size={total_size})")

    # --- verify what landed on disk ---
    seen: dict[str, str] = {}
    for fname in sorted(set(weight_map.values())):
        with safe_open(str(OUT / fname), "pt") as f:
            for k in f.keys():
                if k in seen:
                    fail(f"{k} appears in both {seen[k]} and {fname}")
                seen[k] = fname
                if "lora_" in k:
                    fail(f"{k} written to {fname}")
    if seen != weight_map:
        fail("on-disk tensors do not match weight_map")
    with safe_open(str(OUT / weight_map["h.0.attn.c_attn.weight"]), "pt") as f:
        s = f.get_slice("h.0.attn.c_attn.weight")
        if s.get_shape() != [768, 2304] or s.get_dtype() != "F32":
            fail(f"on-disk h.0.attn.c_attn.weight is {s.get_shape()} {s.get_dtype()}")
    print(f"OK: {len(seen)} tensors in {n} shards")


if __name__ == "__main__":
    main()
