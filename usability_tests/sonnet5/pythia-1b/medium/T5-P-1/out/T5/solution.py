"""
T5: LoRA adapter merge with sharded export (Pythia-1B).

Loads the base Pythia-1B checkpoint and a PEFT-style LoRA adapter, merges
the adapter into the base attention.query_key_value weights, verifies the
result, and writes a sharded safetensors checkpoint under out/T5/.
"""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_PATH = os.path.join(TASK_DIR, "inputs", "base", "model.safetensors")
ADAPTER_PATH = os.path.join(TASK_DIR, "inputs", "lora", "adapter_model.safetensors")
ADAPTER_CONFIG_PATH = os.path.join(TASK_DIR, "inputs", "lora", "adapter_config.json")
OUT_DIR = HERE

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912 bytes
ADAPTER_PREFIX = "base_model.model."
NUM_LAYERS = 16
ADAPTED_MODULE = "attention.query_key_value"


def load_adapter_config():
    with open(ADAPTER_CONFIG_PATH) as f:
        return json.load(f)


def main():
    cfg = load_adapter_config()
    r = cfg["r"]
    lora_alpha = cfg["lora_alpha"]
    fan_in_fan_out = cfg["fan_in_fan_out"]
    if fan_in_fan_out:
        raise NotImplementedError("fan_in_fan_out=True is not handled by this script")
    scale = lora_alpha / r

    # Load base tensors.
    base_tensors = {}
    with safe_open(BASE_PATH, framework="pt") as f:
        for key in f.keys():
            base_tensors[key] = f.get_tensor(key)

    # Load adapter tensors.
    adapter_tensors = {}
    with safe_open(ADAPTER_PATH, framework="pt") as f:
        for key in f.keys():
            adapter_tensors[key] = f.get_tensor(key)

    # Group adapter tensors by base weight name.
    pairs = {}  # base_name -> {"A": tensor, "B": tensor}
    for key, tensor in adapter_tensors.items():
        if not key.startswith(ADAPTER_PREFIX):
            raise ValueError(f"Unexpected adapter tensor name (no expected prefix): {key}")
        rest = key[len(ADAPTER_PREFIX):]
        if rest.endswith(".lora_A.weight"):
            base_name = rest[: -len(".lora_A.weight")] + ".weight"
            slot = "A"
        elif rest.endswith(".lora_B.weight"):
            base_name = rest[: -len(".lora_B.weight")] + ".weight"
            slot = "B"
        else:
            raise ValueError(f"Unexpected adapter tensor name (no lora_A/lora_B suffix): {key}")
        pairs.setdefault(base_name, {})[slot] = tensor

    # Verify every expected pair is present and matches the documented naming.
    expected_base_names = {
        f"gpt_neox.layers.{i}.{ADAPTED_MODULE}.weight" for i in range(NUM_LAYERS)
    }
    found_names = set(pairs.keys())
    if found_names != expected_base_names:
        missing = expected_base_names - found_names
        extra = found_names - expected_base_names
        raise AssertionError(
            f"Adapter pair name mismatch. Missing: {sorted(missing)}, extra: {sorted(extra)}"
        )
    if len(pairs) != NUM_LAYERS:
        raise AssertionError(f"Expected {NUM_LAYERS} adapter pairs, found {len(pairs)}")
    for base_name, slots in pairs.items():
        if "A" not in slots or "B" not in slots:
            raise AssertionError(f"Incomplete adapter pair for {base_name}: {sorted(slots)}")

    # Merge.
    merged_count = 0
    for base_name, slots in pairs.items():
        if base_name not in base_tensors:
            raise KeyError(f"Base checkpoint is missing adapted tensor {base_name}")
        base_weight = base_tensors[base_name]
        base_dtype = base_weight.dtype
        base_shape = tuple(base_weight.shape)

        A = slots["A"]  # [r, in]
        B = slots["B"]  # [out, r]
        if tuple(A.shape) != (r, base_shape[1]):
            raise AssertionError(f"{base_name}: lora_A shape {tuple(A.shape)} unexpected")
        if tuple(B.shape) != (base_shape[0], r):
            raise AssertionError(f"{base_name}: lora_B shape {tuple(B.shape)} unexpected")

        delta = scale * (B.to(torch.float32) @ A.to(torch.float32))  # [out, in]
        if tuple(delta.shape) != base_shape:
            raise AssertionError(
                f"{base_name}: computed delta shape {tuple(delta.shape)} != base shape {base_shape}"
            )

        merged = base_weight.to(torch.float32) + delta
        merged = merged.to(base_dtype)
        if tuple(merged.shape) != base_shape:
            raise AssertionError(f"{base_name}: merged shape changed from {base_shape}")
        base_tensors[base_name] = merged
        merged_count += 1

    # --- Required checks ---
    if merged_count != 16:
        raise AssertionError(f"Expected to merge exactly 16 adapter pairs, merged {merged_count}")

    for name in base_tensors:
        if "lora_" in name:
            raise AssertionError(f"Adapter tensor leaked into output: {name}")

    qkv0_name = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tuple(base_tensors[qkv0_name].shape) != (6144, 2048):
        raise AssertionError(
            f"{qkv0_name} has shape {tuple(base_tensors[qkv0_name].shape)}, expected (6144, 2048)"
        )

    if len(base_tensors) != 244:
        raise AssertionError(f"Expected 244 tensors in output, got {len(base_tensors)}")

    # --- Shard and write ---
    os.makedirs(OUT_DIR, exist_ok=True)

    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # Preserve base file's key order for determinism.
    names_in_order = list(base_tensors.keys())

    shards = []  # list of dict(name -> tensor)
    current_shard = {}
    current_bytes = 0
    for name in names_in_order:
        t = base_tensors[name]
        nbytes = tensor_nbytes(t)
        if nbytes > MAX_SHARD_BYTES:
            # Oversized tensor gets its own shard.
            if current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_bytes = 0
            shards.append({name: t})
            continue
        if current_shard and current_bytes + nbytes > MAX_SHARD_BYTES:
            shards.append(current_shard)
            current_shard = {}
            current_bytes = 0
        current_shard[name] = t
        current_bytes += nbytes
    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    weight_map = {}
    shard_filenames = []
    for idx, shard in enumerate(shards, start=1):
        filename = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
        shard_filenames.append(filename)
        for name in shard:
            weight_map[name] = filename

    for filename, shard in zip(shard_filenames, shards):
        # Tensors must be contiguous for safetensors.
        contiguous_shard = {k: v.contiguous() for k, v in shard.items()}
        save_file(contiguous_shard, os.path.join(OUT_DIR, filename))

    total_size = sum(tensor_nbytes(t) for t in base_tensors.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"Merged {merged_count} adapter pairs.")
    print(f"Wrote {len(base_tensors)} tensors across {num_shards} shards to {OUT_DIR}")


if __name__ == "__main__":
    main()
