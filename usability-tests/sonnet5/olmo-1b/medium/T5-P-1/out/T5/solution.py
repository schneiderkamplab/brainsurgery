"""
T5: LoRA adapter merge with sharded export (OLMo-1B-0724-hf)

Loads the base checkpoint and the PEFT LoRA adapter, merges the adapter into
the base q_proj/v_proj weights for every layer (weight += (alpha/r) * B @ A),
and writes a sharded safetensors checkpoint with no adapter tensors and a
512 MiB per-shard tensor-data budget (oversized tensors get their own shard).
"""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_DIR = os.path.join(REPO_ROOT, "inputs", "base")
LORA_DIR = os.path.join(REPO_ROOT, "inputs", "lora")
OUT_DIR = HERE

SHARD_BUDGET_BYTES = 512 * 1024 * 1024  # 512 MiB, tensor data only

ADAPTER_PREFIX = "base_model.model.model."
ADAPTED_MODULES = ("self_attn.q_proj", "self_attn.v_proj")
NUM_LAYERS = 16


def load_base_state_dict():
    """Load every tensor from the sharded base checkpoint into a dict."""
    index_path = os.path.join(BASE_DIR, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Group tensor names by shard file so each shard is opened once.
    shard_to_names = {}
    for name, shard_file in weight_map.items():
        shard_to_names.setdefault(shard_file, []).append(name)

    state_dict = {}
    for shard_file, names in shard_to_names.items():
        shard_path = os.path.join(BASE_DIR, shard_file)
        with safe_open(shard_path, framework="pt") as f:
            for name in names:
                state_dict[name] = f.get_tensor(name)

    assert len(state_dict) == len(weight_map), (
        f"expected {len(weight_map)} tensors from index, loaded {len(state_dict)}"
    )
    return state_dict


def load_adapter_state_dict():
    adapter_path = os.path.join(LORA_DIR, "adapter_model.safetensors")
    state_dict = {}
    with safe_open(adapter_path, framework="pt") as f:
        for name in f.keys():
            state_dict[name] = f.get_tensor(name)
    return state_dict


def load_adapter_config():
    with open(os.path.join(LORA_DIR, "adapter_config.json")) as f:
        return json.load(f)


def merge_lora(base_sd, adapter_sd, config):
    r = config["r"]
    lora_alpha = config["lora_alpha"]
    fan_in_fan_out = config["fan_in_fan_out"]
    assert not fan_in_fan_out, (
        "fan_in_fan_out=True is not handled by this script (would need A/B transposed)"
    )
    scale = lora_alpha / r

    merged_pairs = 0
    consumed_adapter_keys = set()

    for i in range(NUM_LAYERS):
        for module in ADAPTED_MODULES:
            a_key = f"{ADAPTER_PREFIX}layers.{i}.{module}.lora_A.weight"
            b_key = f"{ADAPTER_PREFIX}layers.{i}.{module}.lora_B.weight"
            base_key = f"model.layers.{i}.{module}.weight"

            if a_key not in adapter_sd or b_key not in adapter_sd:
                continue

            A = adapter_sd[a_key].to(torch.float32)
            B = adapter_sd[b_key].to(torch.float32)
            assert base_key in base_sd, f"base tensor {base_key} not found"

            base_weight = base_sd[base_key]
            assert base_weight.dtype == torch.float32, (
                f"{base_key} expected float32, got {base_weight.dtype}"
            )

            delta = scale * (B @ A)  # [out, in], same layout as base_weight
            assert delta.shape == base_weight.shape, (
                f"shape mismatch for {base_key}: delta {delta.shape} vs base {base_weight.shape}"
            )

            base_sd[base_key] = base_weight + delta

            consumed_adapter_keys.add(a_key)
            consumed_adapter_keys.add(b_key)
            merged_pairs += 1

    # Every adapter tensor must have been used; nothing left over.
    leftover = set(adapter_sd.keys()) - consumed_adapter_keys
    assert not leftover, f"adapter tensors were not merged: {sorted(leftover)}"

    assert merged_pairs == 32, f"expected exactly 32 adapter pairs merged, got {merged_pairs}"

    return base_sd, merged_pairs


def plan_shards(state_dict):
    """Greedily bin-pack tensors into shards under the byte budget.

    A tensor larger than half the budget is kept alone in its own shard: two
    such tensors could never share a shard anyway, and packing one next to
    smaller tensors would leave the shard's fate hostage to iteration order.
    Everything else is first-fit-decreasing packed under the byte budget.
    """
    items = list(state_dict.items())
    # Sort by size descending for reasonably tight packing (deterministic).
    items.sort(key=lambda kv: kv[1].numel() * kv[1].element_size(), reverse=True)

    shards = []  # list of list[name]
    shard_sizes = []  # running byte total per shard
    shard_open = []  # whether more tensors may still be packed into a shard

    for name, tensor in items:
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > SHARD_BUDGET_BYTES // 2:
            shards.append([name])
            shard_sizes.append(nbytes)
            shard_open.append(False)  # sealed: too big to share with anything
            continue

        placed = False
        for idx, size in enumerate(shard_sizes):
            if shard_open[idx] and size + nbytes <= SHARD_BUDGET_BYTES:
                shards[idx].append(name)
                shard_sizes[idx] += nbytes
                placed = True
                break
        if not placed:
            shards.append([name])
            shard_sizes.append(nbytes)
            shard_open.append(True)

    return shards


def save_sharded(state_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    shard_name_lists = plan_shards(state_dict)
    num_shards = len(shard_name_lists)

    weight_map = {}
    total_size = 0
    for shard_idx, names in enumerate(shard_name_lists, start=1):
        shard_filename = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {}
        for name in names:
            tensor = state_dict[name].contiguous()
            shard_tensors[name] = tensor
            weight_map[name] = shard_filename
            total_size += tensor.numel() * tensor.element_size()
        save_file(shard_tensors, os.path.join(out_dir, shard_filename))

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    return weight_map


def main():
    base_sd = load_base_state_dict()
    adapter_sd = load_adapter_state_dict()
    config = load_adapter_config()

    # Required check: no unexpected adapter tensor names (e.g. bias) sneak in.
    for key in adapter_sd:
        assert re.match(
            rf"^{re.escape(ADAPTER_PREFIX)}layers\.\d+\.self_attn\.[qv]_proj\.lora_[AB]\.weight$",
            key,
        ), f"unexpected adapter tensor name: {key}"
    assert len(adapter_sd) == 64, f"expected 64 adapter tensors, found {len(adapter_sd)}"

    merged_sd, merged_pairs = merge_lora(base_sd, adapter_sd, config)

    # --- Required checks (fail loudly before writing) ---
    assert merged_pairs == 32, f"expected 32 merged pairs, got {merged_pairs}"

    lora_names = [name for name in merged_sd if "lora_" in name]
    assert not lora_names, f"adapter tensors leaked into output: {lora_names}"

    probe_key = "model.layers.0.self_attn.q_proj.weight"
    assert merged_sd[probe_key].shape == (2048, 2048), (
        f"{probe_key} has shape {tuple(merged_sd[probe_key].shape)}, expected (2048, 2048)"
    )

    assert len(merged_sd) == 114, f"expected 114 tensors in output, got {len(merged_sd)}"

    weight_map = save_sharded(merged_sd, OUT_DIR)

    assert len(weight_map) == 114, f"expected 114 tensors in weight_map, got {len(weight_map)}"

    print(f"Merged {merged_pairs} LoRA pairs into base weights.")
    print(f"Wrote {len(weight_map)} tensors across "
          f"{len(set(weight_map.values()))} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
