"""T5: merge a PEFT LoRA adapter into Pythia-1B and write a sharded checkpoint."""

import json
import os

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
LORA = os.path.join(ROOT, "inputs", "lora", "adapter_model.safetensors")
CFG = os.path.join(ROOT, "inputs", "lora", "adapter_config.json")
OUT = os.path.join(ROOT, "out", "T5")

SHARD_LIMIT = 536_870_912  # 512 MiB of tensor data


def main() -> None:
    cfg = json.load(open(CFG))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / r
    assert not cfg["fan_in_fan_out"], "fan_in_fan_out=True needs a transpose"

    base = load_file(BASE)
    adapter = load_file(LORA)
    n_base = len(base)

    # Group adapter tensors by the base weight they adapt.
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, t in adapter.items():
        if ".lora_A.weight" in name:
            key, side = name.replace(".lora_A.weight", ""), "A"
        elif ".lora_B.weight" in name:
            key, side = name.replace(".lora_B.weight", ""), "B"
        else:
            raise SystemExit(f"unexpected adapter tensor: {name}")
        key = key.removeprefix("base_model.model.") + ".weight"
        pairs.setdefault(key, {})[side] = t

    if len(pairs) != 16:
        raise SystemExit(f"expected 16 adapter pairs, found {len(pairs)}")
    for key, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            raise SystemExit(f"incomplete adapter pair for {key}: {sorted(ab)}")
        if key not in base:
            raise SystemExit(f"adapter targets missing base tensor {key}")

    for key, ab in sorted(pairs.items()):
        w = base[key]
        a, b = ab["A"].float(), ab["B"].float()
        if b.shape[1] != r or a.shape[0] != r:
            raise SystemExit(f"{key}: factor rank mismatch {b.shape} @ {a.shape}")
        delta = scale * (b @ a)
        if delta.shape != w.shape:
            raise SystemExit(f"{key}: delta {tuple(delta.shape)} != base {tuple(w.shape)}")
        base[key] = (w.float() + delta).to(w.dtype)

    # Required checks before writing.
    bad = [n for n in base if "lora_" in n]
    if bad:
        raise SystemExit(f"adapter tensors leaked into the output: {bad[:5]}")
    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tuple(base[probe].shape) != (6144, 2048):
        raise SystemExit(f"{probe} has shape {tuple(base[probe].shape)}")
    if base[probe].dtype != torch.float16:
        raise SystemExit(f"{probe} has dtype {base[probe].dtype}")
    if len(base) != 244 or len(base) != n_base:
        raise SystemExit(f"expected 244 tensors, have {len(base)}")

    # Greedy sharding in checkpoint order, at most SHARD_LIMIT bytes per shard.
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name, t in base.items():
        nbytes = t.numel() * t.element_size()
        if current and current_size + nbytes > SHARD_LIMIT:
            shards.append(current)
            current, current_size = [], 0
        current.append(name)
        current_size += nbytes
    if current:
        shards.append(current)

    os.makedirs(OUT, exist_ok=True)
    total = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, names in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{total:05d}.safetensors"
        save_file({n: base[n].contiguous() for n in names}, os.path.join(OUT, fname))
        for n in names:
            weight_map[n] = fname
            total_size += base[n].numel() * base[n].element_size()

    with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

    print(f"merged 16 adapter pairs (scale={scale}), wrote {len(weight_map)} tensors "
          f"in {total} shards to {OUT}")


if __name__ == "__main__":
    main()
