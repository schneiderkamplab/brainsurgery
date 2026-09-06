"""T5: merge a PEFT LoRA adapter into a Pythia-1B checkpoint and export it sharded.

Plain torch + safetensors on the checkpoint files (no model instantiation),
with HuggingFace's own shard planner for the 512 MiB sharding rule.
"""

import json
import os
import re

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file

BASE = "inputs/base/model.safetensors"
LORA = "inputs/lora/adapter_model.safetensors"
LORA_CFG = "inputs/lora/adapter_config.json"
OUT_DIR = "out/T5"
MAX_SHARD_BYTES = 512 * 1024 * 1024
EXPECTED_PAIRS = 16
EXPECTED_TENSORS = 244

ADAPTER_RE = re.compile(r"^base_model\.model\.(?P<base>.+)\.lora_(?P<factor>[AB])\.weight$")


def main() -> None:
    cfg = json.load(open(LORA_CFG))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))
    if r <= 0:
        raise ValueError(f"invalid r: {r}")
    scale = alpha / r

    base = load_file(BASE)
    adapter = load_file(LORA)

    # Group adapter tensors into (A, B) pairs keyed by the base weight name.
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in adapter.items():
        m = ADAPTER_RE.match(name)
        if m is None:
            raise ValueError(f"unrecognised adapter tensor name: {name}")
        pairs.setdefault(m["base"] + ".weight", {})[m["factor"]] = tensor

    incomplete = [k for k, v in pairs.items() if set(v) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"incomplete lora pairs: {incomplete}")
    if len(pairs) != EXPECTED_PAIRS:
        raise ValueError(f"expected {EXPECTED_PAIRS} adapter pairs, found {len(pairs)}")

    merged = 0
    for target, fac in sorted(pairs.items()):
        if target not in base:
            raise KeyError(f"adapter targets missing base tensor: {target}")
        w = base[target]
        a, b = fac["A"].float(), fac["B"].float()
        if a.shape[0] != r or b.shape[1] != r:
            raise ValueError(f"{target}: factor rank mismatch, A={tuple(a.shape)} B={tuple(b.shape)}")
        delta = scale * (b @ a)  # [out, in], nn.Linear layout
        if fan_in_fan_out:
            delta = delta.T
        if delta.shape != w.shape:
            raise ValueError(f"{target}: delta {tuple(delta.shape)} != base {tuple(w.shape)}")
        base[target] = (w.float() + delta).to(w.dtype)
        merged += 1

    if merged != EXPECTED_PAIRS:
        raise ValueError(f"merged {merged} pairs, expected {EXPECTED_PAIRS}")

    # Required checks, before writing anything.
    leftover = [k for k in base if "lora_" in k]
    if leftover:
        raise ValueError(f"adapter tensors present in output: {leftover}")
    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tuple(base[probe].shape) != (6144, 2048):
        raise ValueError(f"{probe} has shape {tuple(base[probe].shape)}, expected (6144, 2048)")
    if base[probe].dtype != torch.float16:
        raise ValueError(f"{probe} has dtype {base[probe].dtype}, expected torch.float16")
    if len(base) != EXPECTED_TENSORS:
        raise ValueError(f"output has {len(base)} tensors, expected {EXPECTED_TENSORS}")

    plan = split_torch_state_dict_into_shards(
        base, max_shard_size=MAX_SHARD_BYTES, filename_pattern="model{suffix}.safetensors"
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    for filename, keys in plan.filename_to_tensors.items():
        shard = {k: base[k].contiguous() for k in keys}
        nbytes = sum(t.numel() * t.element_size() for t in shard.values())
        if nbytes > MAX_SHARD_BYTES and len(shard) > 1:
            raise ValueError(f"{filename}: {nbytes} bytes over budget with {len(shard)} tensors")
        save_file(shard, os.path.join(OUT_DIR, filename), metadata={"format": "pt"})

    index = {"metadata": plan.metadata, "weight_map": plan.tensor_to_filename}
    if set(index["weight_map"]) != set(base):
        raise ValueError("weight_map does not cover exactly the output tensors")
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)

    print(f"merged {merged} pairs (scale={scale}), wrote {len(base)} tensors "
          f"in {len(plan.filename_to_tensors)} shards to {OUT_DIR}/")


if __name__ == "__main__":
    main()
