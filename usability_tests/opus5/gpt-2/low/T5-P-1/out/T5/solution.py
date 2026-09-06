"""T5: merge a PEFT LoRA adapter into GPT-2 base weights and write a sharded checkpoint."""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
LORA = os.path.join(ROOT, "inputs", "lora", "adapter_model.safetensors")
CFG = os.path.join(ROOT, "inputs", "lora", "adapter_config.json")
OUT = os.path.join(ROOT, "out", "T5")

SHARD_LIMIT = 100 * 1024 * 1024


def load(path):
    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def main():
    with open(CFG) as f:
        cfg = json.load(f)
    r, alpha = cfg["r"], cfg["lora_alpha"]
    if not cfg.get("fan_in_fan_out"):
        raise SystemExit("expected fan_in_fan_out = true")
    scale = alpha / r

    base = load(BASE)
    adapter = load(LORA)

    # collect A/B pairs keyed by the base tensor name they apply to
    pairs = {}
    pat = re.compile(r"^base_model\.model\.(.+)\.lora_(A|B)\.weight$")
    for name, t in adapter.items():
        m = pat.match(name)
        if m is None:
            raise SystemExit(f"unrecognised adapter tensor name: {name}")
        target = m.group(1) + ".weight"
        pairs.setdefault(target, {})[m.group(2)] = t

    if len(pairs) != 12:
        raise SystemExit(f"expected 12 adapter pairs, found {len(pairs)}")
    for target, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            raise SystemExit(f"incomplete adapter pair for {target}: {sorted(ab)}")
        if target not in base:
            raise SystemExit(f"adapter target {target} is not a base tensor")

    merged = 0
    for target, ab in sorted(pairs.items()):
        A, B = ab["A"].float(), ab["B"].float()
        W = base[target]
        if W.dtype != torch.float32:
            raise SystemExit(f"{target} is {W.dtype}, expected float32")
        if A.shape[0] != r or B.shape[1] != r:
            raise SystemExit(f"rank mismatch for {target}: {tuple(A.shape)} {tuple(B.shape)}")
        delta = (scale * (B @ A)).T  # [out, in] -> Conv1D [in, out]
        if delta.shape != W.shape:
            raise SystemExit(f"delta {tuple(delta.shape)} != base {tuple(W.shape)} for {target}")
        base[target] = (W + delta).contiguous()
        merged += 1
    if merged != 12:
        raise SystemExit(f"merged {merged} pairs, expected 12")

    # required checks
    bad = [k for k in base if "lora_" in k]
    if bad:
        raise SystemExit(f"adapter tensors leaked into the output: {bad}")
    probe = "h.0.attn.c_attn.weight"
    if tuple(base[probe].shape) != (768, 2304):
        raise SystemExit(f"{probe} has shape {tuple(base[probe].shape)}, expected (768, 2304)")
    if len(base) != 160:
        raise SystemExit(f"output has {len(base)} tensors, expected 160")

    # greedy sharding, at most SHARD_LIMIT bytes of tensor data per shard;
    # any single tensor above the limit gets a shard of its own
    def nbytes(t):
        return t.numel() * t.element_size()

    shards, cur, cur_size = [], {}, 0
    for k, t in base.items():
        n = nbytes(t)
        if cur and (cur_size + n > SHARD_LIMIT):
            shards.append(cur)
            cur, cur_size = {}, 0
        cur[k] = t
        cur_size += n
        if cur_size >= SHARD_LIMIT:
            shards.append(cur)
            cur, cur_size = {}, 0
    if cur:
        shards.append(cur)

    total = len(shards)
    names = [f"model-{i + 1:05d}-of-{total:05d}.safetensors" for i in range(total)]
    for shard, name in zip(shards, names):
        size = sum(nbytes(t) for t in shard.values())
        if size > SHARD_LIMIT and len(shard) > 1:
            raise SystemExit(f"shard {name} holds {size} bytes over the limit with {len(shard)} tensors")

    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            os.remove(os.path.join(OUT, f))

    weight_map = {}
    for shard, name in zip(shards, names):
        save_file({k: v.contiguous() for k, v in shard.items()}, os.path.join(OUT, name))
        for k in shard:
            weight_map[k] = name

    if len(weight_map) != 160:
        raise SystemExit(f"weight_map has {len(weight_map)} entries, expected 160")

    index = {
        "metadata": {"total_size": sum(nbytes(t) for t in base.values())},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"merged {merged} adapter pairs, wrote {len(base)} tensors into {total} shards")


if __name__ == "__main__":
    main()
