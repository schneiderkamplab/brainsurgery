"""
T5: LoRA adapter merge with sharded export (OLMo-1B-0724-hf)

Standalone script: merges a PEFT-style LoRA adapter into the base OLMo-1B
checkpoint ("merge and unload") and writes the result as a sharded
safetensors checkpoint, with no adapter or intermediate tensors in the
output.
"""

import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_DIR = os.path.join(REPO_ROOT, "inputs", "base")
LORA_DIR = os.path.join(REPO_ROOT, "inputs", "lora")
OUT_DIR = os.path.join(REPO_ROOT, "out", "T5")

SHARD_BUDGET_BYTES = 512 * 1024 * 1024  # 536,870,912 bytes, tensor data only

ADAPTER_NAME_RE = re.compile(
    r"^base_model\.model\.model\.layers\.(\d+)\.(self_attn\.(?:q_proj|v_proj))\.lora_([AB])\.weight$"
)


def load_base_index():
    with open(os.path.join(BASE_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)
    return index["weight_map"]


def load_adapter_config():
    with open(os.path.join(LORA_DIR, "adapter_config.json")) as f:
        return json.load(f)


def load_base_tensor(name, weight_map):
    shard_file = weight_map[name]
    path = os.path.join(BASE_DIR, shard_file)
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(name)


def main():
    cfg = load_adapter_config()
    assert cfg["fan_in_fan_out"] is False, (
        "this script only implements the fan_in_fan_out=false layout"
    )
    r = cfg["r"]
    lora_alpha = cfg["lora_alpha"]
    scale = lora_alpha / r

    weight_map = load_base_index()

    # --- Parse adapter tensors into per-(layer, module) A/B pairs ---
    adapter_path = os.path.join(LORA_DIR, "adapter_model.safetensors")
    lora_A = {}  # (layer, module) -> tensor
    lora_B = {}
    with safe_open(adapter_path, framework="pt") as f:
        adapter_names = list(f.keys())
        for name in adapter_names:
            m = ADAPTER_NAME_RE.match(name)
            if m is None:
                raise ValueError(f"unrecognized adapter tensor name: {name}")
            layer_idx, module, ab = m.group(1), m.group(2), m.group(3)
            key = (layer_idx, module)
            tensor = f.get_tensor(name).to(torch.float32)
            if ab == "A":
                lora_A[key] = tensor
            else:
                lora_B[key] = tensor

    if len(adapter_names) != 64:
        raise AssertionError(
            f"expected 64 adapter tensors (32 A + 32 B), found {len(adapter_names)}"
        )

    pairs = sorted(set(lora_A.keys()) & set(lora_B.keys()))
    if set(lora_A.keys()) != set(lora_B.keys()):
        raise AssertionError("mismatched lora_A / lora_B keys in adapter")
    if len(pairs) != 32:
        raise AssertionError(f"expected exactly 32 adapter pairs, found {len(pairs)}")

    # Map each (layer, module) adapter pair to its base tensor name.
    merge_map = {}
    for layer_idx, module in pairs:
        base_name = f"model.layers.{layer_idx}.{module}.weight"
        if base_name not in weight_map:
            raise KeyError(f"base tensor {base_name!r} not found in base checkpoint index")
        merge_map[base_name] = (layer_idx, module)

    # --- Build the merged state dict ---
    merged = {}
    merged_count = 0
    for name in weight_map:
        base_tensor = load_base_tensor(name, weight_map)
        if name in merge_map:
            layer_idx, module = merge_map[name]
            A = lora_A[(layer_idx, module)]  # [r, in]
            B = lora_B[(layer_idx, module)]  # [out, r]
            if base_tensor.dtype != torch.float32:
                raise AssertionError(f"{name} is not float32 ({base_tensor.dtype})")
            base_f32 = base_tensor.to(torch.float32)
            delta = scale * (B @ A)  # [out, in], same layout as base (fan_in_fan_out=false)
            if delta.shape != base_f32.shape:
                raise AssertionError(
                    f"shape mismatch merging {name}: base {base_f32.shape} vs delta {delta.shape}"
                )
            result = (base_f32 + delta).contiguous()
            merged[name] = result
            merged_count += 1
        else:
            merged[name] = base_tensor.contiguous()

    # --- Required checks (fail loudly before writing anything) ---
    if merged_count != 32:
        raise AssertionError(f"expected to merge exactly 32 adapter pairs, merged {merged_count}")

    lora_leftover = [n for n in merged if "lora_" in n]
    if lora_leftover:
        raise AssertionError(f"adapter tensor names leaked into output: {lora_leftover}")

    probe_name = "model.layers.0.self_attn.q_proj.weight"
    if tuple(merged[probe_name].shape) != (2048, 2048):
        raise AssertionError(
            f"{probe_name} has shape {tuple(merged[probe_name].shape)}, expected (2048, 2048)"
        )

    if len(merged) != 114:
        raise AssertionError(f"expected 114 tensors in output, got {len(merged)}")

    # --- Shard by size budget (bin-pack in name order; oversized tensors get their own shard) ---
    def tensor_bytes(t):
        return t.element_size() * t.nelement()

    names_in_order = list(weight_map.keys())  # preserves base checkpoint's own ordering
    shards = []  # list[list[name]]
    current = []
    current_size = 0
    for name in names_in_order:
        size = tensor_bytes(merged[name])
        if size > SHARD_BUDGET_BYTES:
            if current:
                shards.append(current)
                current = []
                current_size = 0
            shards.append([name])
            continue
        if current and current_size + size > SHARD_BUDGET_BYTES:
            shards.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += size
    if current:
        shards.append(current)

    os.makedirs(OUT_DIR, exist_ok=True)

    num_shards = len(shards)
    weight_map_out = {}
    total_size = 0
    for i, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {name: merged[name] for name in shard_names}
        save_file(shard_tensors, os.path.join(OUT_DIR, shard_filename), metadata={"format": "pt"})
        for name, t in shard_tensors.items():
            weight_map_out[name] = shard_filename
            total_size += tensor_bytes(t)

    index_out = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map_out,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index_out, f, indent=2)

    print(f"Merged {merged_count} adapter pairs into {len(merged)} tensors.")
    print(f"Wrote {num_shards} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
