"""
T5: LoRA adapter merge with sharded export (Pythia-1B).

Merges a PEFT-style LoRA adapter into the base gpt_neox.*.attention.query_key_value
weights ("merge and unload"), then writes the result as a sharded safetensors
checkpoint with a model.safetensors.index.json, respecting a 512 MiB per-shard
tensor-data budget.
"""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "inputs", "base")
LORA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "inputs", "lora")
OUT_DIR = os.path.join(os.path.dirname(__file__))

BASE_PATH = os.path.abspath(os.path.join(BASE_DIR, "model.safetensors"))
ADAPTER_PATH = os.path.abspath(os.path.join(LORA_DIR, "adapter_model.safetensors"))
ADAPTER_CONFIG_PATH = os.path.abspath(os.path.join(LORA_DIR, "adapter_config.json"))

SHARD_BYTE_LIMIT = 512 * 1024 * 1024  # 536,870,912 bytes

LORA_A_RE = re.compile(
    r"^base_model\.model\.(?P<base_name>.+)\.lora_A\.weight$"
)
LORA_B_RE = re.compile(
    r"^base_model\.model\.(?P<base_name>.+)\.lora_B\.weight$"
)


def load_adapter_config():
    with open(ADAPTER_CONFIG_PATH) as f:
        cfg = json.load(f)
    return cfg


def load_state_dict(path):
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main():
    cfg = load_adapter_config()
    r = cfg["r"]
    lora_alpha = cfg["lora_alpha"]
    fan_in_fan_out = cfg["fan_in_fan_out"]
    if fan_in_fan_out:
        raise NotImplementedError(
            "fan_in_fan_out=true is not handled by this script; "
            "the transpose convention would need to be applied to B @ A."
        )
    scale = lora_alpha / r

    base_sd = load_state_dict(BASE_PATH)
    adapter_sd = load_state_dict(ADAPTER_PATH)

    n_base_tensors_before = len(base_sd)

    # Pair up lora_A / lora_B tensors by their base tensor name.
    lora_pairs = {}  # base_name -> {"A": tensor, "B": tensor}
    for key, tensor in adapter_sd.items():
        m_a = LORA_A_RE.match(key)
        m_b = LORA_B_RE.match(key)
        if m_a:
            base_name = m_a.group("base_name") + ".weight"
            lora_pairs.setdefault(base_name, {})["A"] = tensor
        elif m_b:
            base_name = m_b.group("base_name") + ".weight"
            lora_pairs.setdefault(base_name, {})["B"] = tensor
        else:
            raise ValueError(f"Unrecognized adapter tensor name: {key}")

    for base_name, pair in lora_pairs.items():
        if "A" not in pair or "B" not in pair:
            raise ValueError(f"Incomplete LoRA pair for {base_name}: {pair.keys()}")

    # Required check: exactly 16 adapter pairs found.
    if len(lora_pairs) != 16:
        raise AssertionError(
            f"Expected exactly 16 adapter pairs, found {len(lora_pairs)}"
        )

    merged_count = 0
    for base_name, pair in lora_pairs.items():
        if base_name not in base_sd:
            raise KeyError(f"Base tensor {base_name} referenced by adapter not found")

        base_weight = base_sd[base_name]
        base_dtype = base_weight.dtype

        A = pair["A"].to(torch.float32)  # [r, in]
        B = pair["B"].to(torch.float32)  # [out, r]

        delta = scale * (B @ A)  # [out, in], matches nn.Linear [out, in] layout
        if delta.shape != base_weight.shape:
            raise ValueError(
                f"Shape mismatch merging {base_name}: base {base_weight.shape}, "
                f"delta {delta.shape}"
            )

        merged_fp32 = base_weight.to(torch.float32) + delta
        base_sd[base_name] = merged_fp32.to(base_dtype)
        merged_count += 1

    if merged_count != 16:
        raise AssertionError(f"Expected to merge 16 tensors, merged {merged_count}")

    # Required check: no adapter/intermediate tensor leaked into the output.
    if any("lora_" in name for name in base_sd.keys()):
        raise AssertionError("Adapter tensor(s) leaked into merged state dict")

    # Required check: layer-0 qkv weight shape unchanged.
    qkv0_name = "gpt_neox.layers.0.attention.query_key_value.weight"
    expected_qkv_shape = torch.Size([6144, 2048])
    if base_sd[qkv0_name].shape != expected_qkv_shape:
        raise AssertionError(
            f"{qkv0_name} has shape {base_sd[qkv0_name].shape}, "
            f"expected {tuple(expected_qkv_shape)}"
        )

    # Required check: same tensor count as base checkpoint (244, unchanged names).
    if len(base_sd) != n_base_tensors_before:
        raise AssertionError(
            f"Tensor count changed: started with {n_base_tensors_before}, "
            f"ended with {len(base_sd)}"
        )
    if len(base_sd) != 244:
        raise AssertionError(f"Expected 244 tensors in output, got {len(base_sd)}")

    # --- Shard and write ---
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # Deterministic order: sorted tensor names.
    names = sorted(base_sd.keys())

    # Per the spec (Required result, item 4), gpt_neox.embed_in.weight and
    # embed_out.weight are called out by name to be stored alone in their own
    # shard even though, at 206 MB each, neither individually exceeds the
    # 512 MiB budget on its own. Treat that as an explicit requirement rather
    # than only falling back to the generic oversized-tensor rule below.
    SOLO_TENSOR_NAMES = {"gpt_neox.embed_in.weight", "embed_out.weight"}

    shards = []  # list of dict[name -> tensor]
    current_shard = {}
    current_size = 0
    for name in names:
        t = base_sd[name]
        size = tensor_nbytes(t)
        if size > SHARD_BYTE_LIMIT or name in SOLO_TENSOR_NAMES:
            # Oversized (or explicitly called-out) tensor gets its own shard.
            if current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
            shards.append({name: t})
            continue
        if current_size + size > SHARD_BYTE_LIMIT and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[name] = t
        current_size += size
    if current_shard:
        shards.append(current_shard)

    n_shards = len(shards)
    os.makedirs(OUT_DIR, exist_ok=True)

    weight_map = {}
    total_size = 0
    for idx, shard in enumerate(shards, start=1):
        shard_filename = f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
        shard_path = os.path.join(OUT_DIR, shard_filename)
        # Ensure contiguous tensors for safetensors.
        contiguous_shard = {k: v.contiguous() for k, v in shard.items()}
        save_file(contiguous_shard, shard_path)
        for name, t in shard.items():
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(t)

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path = os.path.join(OUT_DIR, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"Merged {merged_count} LoRA pairs into base weights.")
    print(f"Wrote {len(weight_map)} tensors across {n_shards} shard(s) to {OUT_DIR}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
