"""
T5: LoRA adapter merge with sharded export (OLMo-1B-0724-hf).

Loads the sharded base checkpoint and the PEFT-style LoRA adapter, merges
adapter deltas (scale * B @ A) into the targeted q_proj/v_proj weights of
every layer, and writes a plain dense, re-sharded safetensors checkpoint
with no adapter or intermediate tensors.
"""

import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
TASK_DIR = HERE.parent.parent
BASE_DIR = TASK_DIR / "inputs" / "base"
LORA_DIR = TASK_DIR / "inputs" / "lora"
OUT_DIR = HERE  # out/T5/

SHARD_MAX_BYTES = 512 * 1024 * 1024  # 536,870,912 bytes of tensor data per shard

ADAPTED_MODULES = ["self_attn.q_proj", "self_attn.v_proj"]
NUM_LAYERS = 16

DTYPE_SIZE = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}


def main():
    # --- Load base index ---
    with open(BASE_DIR / "model.safetensors.index.json") as f:
        base_index = json.load(f)
    base_weight_map = base_index["weight_map"]  # name -> shard filename

    base_names = list(base_weight_map.keys())
    if len(base_names) != 114:
        raise AssertionError(f"expected 114 base tensors, found {len(base_names)}")

    # Open all base shard files with safe_open (lazy access).
    shard_files = sorted(set(base_weight_map.values()))
    base_handles = {
        shard: safe_open(str(BASE_DIR / shard), framework="pt", device="cpu")
        for shard in shard_files
    }

    def get_base_tensor(name):
        shard = base_weight_map[name]
        return base_handles[shard].get_tensor(name)

    def get_base_shape(name):
        shard = base_weight_map[name]
        return base_handles[shard].get_slice(name).get_shape()

    def get_base_dtype_str(name):
        shard = base_weight_map[name]
        return base_handles[shard].get_slice(name).get_dtype()

    # --- Load LoRA adapter config and weights ---
    with open(LORA_DIR / "adapter_config.json") as f:
        adapter_config = json.load(f)

    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    fan_in_fan_out = adapter_config.get("fan_in_fan_out", False)
    target_modules = set(adapter_config["target_modules"])

    if fan_in_fan_out:
        raise NotImplementedError(
            "fan_in_fan_out=True is not handled by this script"
        )
    expected_leaf_modules = {m.split(".")[-1] for m in ADAPTED_MODULES}
    if set(target_modules) != expected_leaf_modules:
        raise AssertionError(
            f"unexpected target_modules in adapter_config.json: {target_modules}"
        )

    scale = lora_alpha / r

    lora_tensors = {}
    with safe_open(str(LORA_DIR / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_tensors[key] = f.get_tensor(key)

    if len(lora_tensors) != 64:
        raise AssertionError(f"expected 64 adapter tensors, found {len(lora_tensors)}")

    # base_model.model.model.layers.<i>.<module>.lora_A.weight / lora_B.weight
    lora_name_re = re.compile(
        r"^base_model\.model\.model\.layers\.(\d+)\.(self_attn\.[a-z_]+)\.lora_([AB])\.weight$"
    )

    # Map (layer, module) -> {"A": tensor, "B": tensor}
    pairs = {}
    for key, tensor in lora_tensors.items():
        m = lora_name_re.match(key)
        if m is None:
            raise AssertionError(f"unrecognized adapter tensor name: {key}")
        layer_idx = int(m.group(1))
        module = m.group(2)
        ab = m.group(3)
        if module.split(".")[-1] not in target_modules:
            raise AssertionError(f"adapter tensor for untargeted module: {key}")
        pairs.setdefault((layer_idx, module), {})[ab] = tensor

    if len(pairs) != 32:
        raise AssertionError(f"expected 32 adapter pairs (16 layers x 2 modules), found {len(pairs)}")

    for (layer_idx, module), d in pairs.items():
        if "A" not in d or "B" not in d:
            raise AssertionError(f"incomplete adapter pair for layer {layer_idx} {module}")

    merged_pair_count = 0
    merged_deltas = {}  # base_name -> merged tensor (float32)

    for (layer_idx, module), d in pairs.items():
        base_name = f"model.layers.{layer_idx}.{module}.weight"
        if base_name not in base_weight_map:
            raise AssertionError(f"base tensor not found for adapter pair: {base_name}")

        A = d["A"].to(torch.float32)  # [r, in]
        B = d["B"].to(torch.float32)  # [out, r]

        if A.shape != (r, 2048):
            raise AssertionError(f"unexpected lora_A shape for {base_name}: {tuple(A.shape)}")
        if B.shape != (2048, r):
            raise AssertionError(f"unexpected lora_B shape for {base_name}: {tuple(B.shape)}")

        base_tensor = get_base_tensor(base_name).to(torch.float32)  # [out, in], no transpose (fan_in_fan_out=False)

        if base_tensor.shape != (2048, 2048):
            raise AssertionError(f"unexpected base shape for {base_name}: {tuple(base_tensor.shape)}")

        delta = scale * (B @ A)  # [out, in]
        merged = base_tensor + delta

        if merged.dtype != torch.float32:
            raise AssertionError(f"merged tensor for {base_name} is not float32")
        if merged.shape != (2048, 2048):
            raise AssertionError(f"merged tensor for {base_name} has wrong shape: {tuple(merged.shape)}")

        merged_deltas[base_name] = merged
        merged_pair_count += 1

    if merged_pair_count != 32:
        raise AssertionError(f"expected to merge exactly 32 adapter pairs, merged {merged_pair_count}")

    # --- Required checks ---
    for name in merged_deltas:
        if "lora_" in name:
            raise AssertionError("adapter tensor name leaked into merged output")

    q0 = "model.layers.0.self_attn.q_proj.weight"
    if tuple(get_base_shape(q0)) != (2048, 2048):
        raise AssertionError("model.layers.0.self_attn.q_proj.weight shape check failed")

    if len(base_names) != 114:
        raise AssertionError("base tensor count check failed")

    # --- Build final tensor list: name -> torch.Tensor, dtype, nbytes ---
    def dtype_from_str(dtype_str):
        mapping = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
        if dtype_str not in mapping:
            raise AssertionError(f"unsupported dtype: {dtype_str}")
        return mapping[dtype_str]

    tensor_info = []  # list of (name, nbytes)
    for name in base_names:
        if name in merged_deltas:
            nbytes = merged_deltas[name].numel() * 4
        else:
            shape = get_base_shape(name)
            dtype = dtype_from_str(get_base_dtype_str(name))
            numel = 1
            for s in shape:
                numel *= s
            nbytes = numel * DTYPE_SIZE[dtype]
        tensor_info.append((name, nbytes))

    def load_final_tensor(name):
        if name in merged_deltas:
            return merged_deltas[name]
        return get_base_tensor(name)

    # --- Greedy shard packing (in base name order) ---
    shards = []  # list of list of names
    current_shard = []
    current_size = 0
    for name, nbytes in tensor_info:
        if nbytes > SHARD_MAX_BYTES:
            if current_shard:
                shards.append(current_shard)
                current_shard = []
                current_size = 0
            shards.append([name])
            continue
        if current_shard and current_size + nbytes > SHARD_MAX_BYTES:
            shards.append(current_shard)
            current_shard = []
            current_size = 0
        current_shard.append(name)
        current_size += nbytes
    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    total_size = sum(nbytes for _, nbytes in tensor_info)

    os.makedirs(OUT_DIR, exist_ok=True)

    weight_map = {}
    for shard_idx, names in enumerate(shards, start=1):
        shard_filename = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
        tensors = {}
        for name in names:
            t = load_final_tensor(name)
            tensors[name] = t.contiguous()
            weight_map[name] = shard_filename
        save_file(tensors, str(OUT_DIR / shard_filename), metadata={"format": "pt"})

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    if len(weight_map) != 114:
        raise AssertionError(f"output has {len(weight_map)} tensors, expected 114")
    if any("lora_" in name for name in weight_map):
        raise AssertionError("adapter tensor name found in output weight_map")

    print(f"Merged {merged_pair_count} adapter pairs.")
    print(f"Wrote {len(weight_map)} tensors across {num_shards} shards to {OUT_DIR}")


if __name__ == "__main__":
    main()
