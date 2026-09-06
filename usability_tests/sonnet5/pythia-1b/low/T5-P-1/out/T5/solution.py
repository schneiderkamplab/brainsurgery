"""Merge a LoRA adapter into the Pythia-1B base checkpoint and write a
sharded safetensors output, with no adapter tensors left over.
"""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_PATH = os.path.join(TASK_DIR, "inputs", "base", "model.safetensors")
ADAPTER_PATH = os.path.join(TASK_DIR, "inputs", "lora", "adapter_model.safetensors")
ADAPTER_CONFIG_PATH = os.path.join(TASK_DIR, "inputs", "lora", "adapter_config.json")
OUT_DIR = os.path.join(HERE)  # out/T5/

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912 bytes

LORA_A_RE = re.compile(
    r"^base_model\.model\.(gpt_neox\.layers\.(\d+)\.attention\.query_key_value)\.lora_A\.weight$"
)


def main():
    with open(ADAPTER_CONFIG_PATH) as f:
        adapter_config = json.load(f)

    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    fan_in_fan_out = adapter_config.get("fan_in_fan_out", False)
    assert not fan_in_fan_out, "this solution assumes fan_in_fan_out = False"
    scale = lora_alpha / r

    # Load adapter tensors.
    adapter_tensors = {}
    with safe_open(ADAPTER_PATH, framework="pt") as f:
        for key in f.keys():
            adapter_tensors[key] = f.get_tensor(key)

    # Find matching A/B pairs.
    pairs = {}  # base_module_name -> (A, B)
    for key in adapter_tensors:
        m = LORA_A_RE.match(key)
        if m is None:
            continue
        base_module = m.group(1) + ".weight"
        b_key = key.replace("lora_A.weight", "lora_B.weight")
        assert b_key in adapter_tensors, f"missing matching lora_B for {key}"
        pairs[base_module] = (adapter_tensors[key], adapter_tensors[b_key])

    assert len(pairs) == 16, f"expected exactly 16 adapter pairs, found {len(pairs)}"

    # Load base tensors.
    base_tensors = {}
    with safe_open(BASE_PATH, framework="pt") as f:
        base_keys = list(f.keys())
        for key in base_keys:
            base_tensors[key] = f.get_tensor(key)

    assert len(base_tensors) == 244, f"expected 244 base tensors, found {len(base_tensors)}"

    merged_count = 0
    for base_module, (A, B) in pairs.items():
        assert base_module in base_tensors, f"base tensor {base_module} not found"
        weight = base_tensors[base_module]
        base_dtype = weight.dtype
        assert weight.shape == (6144, 2048), f"unexpected base shape {weight.shape}"
        assert A.shape == (16, 2048), f"unexpected lora_A shape {A.shape}"
        assert B.shape == (6144, 16), f"unexpected lora_B shape {B.shape}"

        delta = scale * (B.to(torch.float32) @ A.to(torch.float32))
        merged = weight.to(torch.float32) + delta
        base_tensors[base_module] = merged.to(base_dtype)
        merged_count += 1

    assert merged_count == 16, f"expected to merge 16 weights, merged {merged_count}"

    # Required checks.
    assert not any("lora_" in k for k in base_tensors), "adapter tensor leaked into output"
    assert base_tensors["gpt_neox.layers.0.attention.query_key_value.weight"].shape == (
        6144,
        2048,
    ), "layer 0 qkv weight shape changed unexpectedly"
    assert len(base_tensors) == 244, f"expected 244 output tensors, got {len(base_tensors)}"

    # Shard: greedy bin-packing in key order, each tensor whose own size
    # exceeds the shard budget gets its own shard.
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards = []  # list of dict name->tensor
    current_shard = {}
    current_bytes = 0
    for key in base_keys:
        t = base_tensors[key]
        nbytes = tensor_nbytes(t)
        if nbytes > MAX_SHARD_BYTES:
            if current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_bytes = 0
            shards.append({key: t})
            continue
        if current_bytes + nbytes > MAX_SHARD_BYTES and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_bytes = 0
        current_shard[key] = t
        current_bytes += nbytes
    if current_shard:
        shards.append(current_shard)

    os.makedirs(OUT_DIR, exist_ok=True)

    weight_map = {}
    total_size = 0
    num_shards = len(shards)
    shard_filenames = []
    for i, shard in enumerate(shards, start=1):
        filename = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        shard_filenames.append(filename)
        for key in shard:
            weight_map[key] = filename
        total_size += sum(tensor_nbytes(t) for t in shard.values())

    for filename, shard in zip(shard_filenames, shards):
        save_file(shard, os.path.join(OUT_DIR, filename), metadata={"format": "pt"})

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"Merged {merged_count} adapter pairs into base weights.")
    print(f"Wrote {len(shards)} shard(s), {len(weight_map)} tensors total.")


if __name__ == "__main__":
    main()
