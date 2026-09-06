"""T5: LoRA adapter merge with sharded export (OLMo-1B-0724-hf).

Plain script on top of `safetensors` + `torch` + `huggingface_hub`'s sharding
helper (the same shard-splitting algorithm `transformers.save_pretrained`
uses internally, imported directly for full control over output paths and
so the required checks run before anything is written).

peft's `PeftModel.merge_and_unload` was considered (see REPORT.md) but
requires instantiating the full HF model just to merge four numbers'
worth of arithmetic per adapted module; a direct state-dict merge is more
direct and equally correct given the layout guarantees in TASK.md.
"""

import json
import os
import re

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(HERE, "..", "..", "inputs", "base")
LORA_DIR = os.path.join(HERE, "..", "..", "inputs", "lora")
OUT_DIR = HERE

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912

TARGET_MODULES = ["self_attn.q_proj", "self_attn.v_proj"]
NUM_LAYERS = 16


def load_base_state_dict():
    with open(os.path.join(BASE_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    state_dict = {}
    for shard_file in shard_files:
        with safe_open(os.path.join(BASE_DIR, shard_file), framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    assert set(state_dict.keys()) == set(weight_map.keys())
    return state_dict


def load_adapter_tensors():
    tensors = {}
    with safe_open(os.path.join(LORA_DIR, "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main():
    with open(os.path.join(LORA_DIR, "adapter_config.json")) as f:
        adapter_config = json.load(f)
    assert adapter_config["fan_in_fan_out"] is False, "layout assumption violated"
    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    scale = lora_alpha / r

    base_state_dict = load_base_state_dict()
    adapter_tensors = load_adapter_tensors()

    pair_re = re.compile(
        r"^base_model\.model\.model\.layers\.(\d+)\.(self_attn\.(?:q|v)_proj)\.lora_A\.weight$"
    )

    merged_pairs = 0
    for key in list(adapter_tensors.keys()):
        m = pair_re.match(key)
        if m is None:
            continue
        layer_idx, module = m.group(1), m.group(2)
        key_a = key
        key_b = f"base_model.model.model.layers.{layer_idx}.{module}.lora_B.weight"
        base_key = f"model.layers.{layer_idx}.{module}.weight"

        assert base_key in base_state_dict, f"missing base tensor {base_key}"
        assert key_b in adapter_tensors, f"missing paired lora_B for {key_a}"

        A = adapter_tensors[key_a].to(torch.float32)
        B = adapter_tensors[key_b].to(torch.float32)
        delta = scale * (B @ A)

        base = base_state_dict[base_key]
        assert base.dtype == torch.float32
        assert delta.shape == base.shape, (
            f"shape mismatch for {base_key}: base {tuple(base.shape)} vs delta {tuple(delta.shape)}"
        )
        base_state_dict[base_key] = base + delta
        merged_pairs += 1

    # --- Required checks (fail loudly before writing anything) ---
    expected_pairs = NUM_LAYERS * len(TARGET_MODULES)
    assert merged_pairs == expected_pairs, (
        f"expected {expected_pairs} adapter pairs, merged {merged_pairs}"
    )
    assert not any("lora_" in k for k in base_state_dict), "adapter tensor leaked into output"
    assert base_state_dict["model.layers.0.self_attn.q_proj.weight"].shape == (2048, 2048)
    assert len(base_state_dict) == 114, f"expected 114 tensors, got {len(base_state_dict)}"

    # --- Sharded export ---
    split = split_torch_state_dict_into_shards(
        base_state_dict,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=MAX_SHARD_BYTES,
    )

    for filename, keys in split.filename_to_tensors.items():
        shard = {k: base_state_dict[k] for k in keys}
        save_file(shard, os.path.join(OUT_DIR, filename), metadata={"format": "pt"})

    index = {
        "metadata": split.metadata,
        "weight_map": split.tensor_to_filename,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"merged {merged_pairs} adapter pairs into {len(base_state_dict)} tensors")
    print(f"wrote {len(split.filename_to_tensors)} shard file(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
