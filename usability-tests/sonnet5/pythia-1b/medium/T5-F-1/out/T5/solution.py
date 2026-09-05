"""
T5: LoRA adapter merge with sharded export (Pythia-1B).

Plain script on top of `safetensors` + `torch` (both in F-allowed.md). Chosen
over `peft.merge_and_unload` because that path requires instantiating the
full HF model just to fold two rank-16 factors into 16 linear layers; a
direct safetensors-to-safetensors rewrite is simpler and does not need the
model class to load correctly. Sharding is written manually, following the
same sequential bin-filling rule HF's `save_pretrained` sharding uses: at
most 512 MiB (536,870,912 bytes) of tensor data per shard, oversized tensors
get their own shard.

Usage: python solution.py
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_PATH = Path("inputs/base/model.safetensors")
ADAPTER_PATH = Path("inputs/lora/adapter_model.safetensors")
ADAPTER_CONFIG_PATH = Path("inputs/lora/adapter_config.json")
OUT_DIR = Path("out/T5")
MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912 bytes, tensor data only

ADAPTER_NAME_RE = re.compile(
    r"^base_model\.model\.(?P<base>.+)\.lora_(?P<part>[AB])\.weight$"
)


def load_base():
    tensors = {}
    with safe_open(BASE_PATH, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def load_adapter_pairs():
    """Group adapter tensors into {base_name: {"A": tensor, "B": tensor}}."""
    raw = {}
    with safe_open(ADAPTER_PATH, framework="pt") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in raw.items():
        m = ADAPTER_NAME_RE.match(name)
        if m is None:
            raise ValueError(f"Unrecognized adapter tensor name: {name}")
        base_name = m.group("base") + ".weight"
        part = m.group("part")
        pairs.setdefault(base_name, {})[part] = tensor

    for base_name, parts in pairs.items():
        if set(parts) != {"A", "B"}:
            raise ValueError(f"Incomplete lora pair for {base_name}: got {set(parts)}")
    return pairs


def merge(base_tensors, pairs, scale):
    merged_names = set()
    for base_name, parts in pairs.items():
        if base_name not in base_tensors:
            raise KeyError(f"Adapter targets missing base tensor: {base_name}")
        A = parts["A"].to(torch.float32)  # [r, in]
        B = parts["B"].to(torch.float32)  # [out, r]
        base = base_tensors[base_name]
        delta = scale * (B @ A)  # [out, in], same layout as base ([out, in])
        if delta.shape != base.shape:
            raise ValueError(
                f"Shape mismatch for {base_name}: base {tuple(base.shape)} "
                f"vs delta {tuple(delta.shape)}"
            )
        merged = (base.to(torch.float32) + delta).to(base.dtype)
        base_tensors[base_name] = merged
        merged_names.add(base_name)
    return merged_names


def shard(tensors: dict[str, torch.Tensor]):
    """Sequential bin-filling: accumulate in key order until the next tensor
    would push the running total over MAX_SHARD_BYTES, then start a new
    shard. A tensor larger than the budget on its own gets its own shard."""
    names = list(tensors.keys())
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    def tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    for name in names:
        size = tensor_bytes(tensors[name])
        if size > MAX_SHARD_BYTES:
            # oversized tensor: flush whatever is pending, then give it a
            # shard entirely to itself
            if current:
                shards.append(current)
                current = []
                current_bytes = 0
            shards.append([name])
            continue
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def main():
    adapter_config = json.loads(ADAPTER_CONFIG_PATH.read_text())
    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    fan_in_fan_out = adapter_config.get("fan_in_fan_out", False)
    if fan_in_fan_out:
        raise ValueError("This script assumes fan_in_fan_out = false")
    scale = lora_alpha / r

    base_tensors = load_base()
    n_base = len(base_tensors)

    pairs = load_adapter_pairs()
    if len(pairs) != 16:
        raise AssertionError(f"Expected exactly 16 adapter pairs, found {len(pairs)}")

    merged_names = merge(base_tensors, pairs, scale)
    if len(merged_names) != 16:
        raise AssertionError(f"Expected 16 merged tensors, merged {len(merged_names)}")

    # required checks
    for name in base_tensors:
        if "lora_" in name:
            raise AssertionError(f"Adapter tensor leaked into output: {name}")

    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tuple(base_tensors[probe].shape) != (6144, 2048):
        raise AssertionError(
            f"{probe} has shape {tuple(base_tensors[probe].shape)}, expected (6144, 2048)"
        )

    if len(base_tensors) != n_base or len(base_tensors) != 244:
        raise AssertionError(f"Expected 244 tensors in output, got {len(base_tensors)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_groups = shard(base_tensors)

    weight_map = {}
    total_size = 0
    n_shards = len(shard_groups)
    digits = len(str(n_shards))
    for i, names in enumerate(shard_groups, start=1):
        shard_filename = f"model-{i:0{digits}d}-of-{n_shards:0{digits}d}.safetensors"
        shard_tensors = {name: base_tensors[name].contiguous() for name in names}
        save_file(shard_tensors, OUT_DIR / shard_filename)
        for name, t in shard_tensors.items():
            weight_map[name] = shard_filename
            total_size += t.numel() * t.element_size()

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    print(f"Merged {len(merged_names)} adapter pairs into {len(base_tensors)} base tensors.")
    print(f"Wrote {n_shards} shards to {OUT_DIR}.")


if __name__ == "__main__":
    main()
