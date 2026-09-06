"""
T5: LoRA adapter merge with sharded export (GPT-2, 124M)

Merges a PEFT-style LoRA adapter into the base GPT-2 checkpoint ("merge and
unload") and writes the result as a sharded safetensors checkpoint with a
model.safetensors.index.json, with no shard exceeding 100 MiB of tensor data
(a single oversized tensor gets its own shard).
"""

import json
import os

import torch
from safetensors.torch import save_file

BASE_DIR = "inputs/base"
LORA_DIR = "inputs/lora"
OUT_DIR = "out/T5"
SHARD_LIMIT_BYTES = 100 * 1024 * 1024  # 100 MiB, tensor data only

TARGET_MODULES = ["attn.c_attn"]
NUM_LAYERS = 12


def load_safetensors(path):
    from safetensors import safe_open

    tensors = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main():
    base = load_safetensors(os.path.join(BASE_DIR, "model.safetensors"))
    adapter = load_safetensors(os.path.join(LORA_DIR, "adapter_model.safetensors"))

    with open(os.path.join(LORA_DIR, "adapter_config.json")) as f:
        adapter_config = json.load(f)

    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    fan_in_fan_out = adapter_config["fan_in_fan_out"]
    assert fan_in_fan_out is True, (
        f"expected fan_in_fan_out=True (Conv1D base layout), got {fan_in_fan_out}"
    )
    scale = lora_alpha / r

    merged_count = 0
    for layer_idx in range(NUM_LAYERS):
        for module in TARGET_MODULES:
            prefix = f"base_model.model.h.{layer_idx}.{module}"
            a_key = f"{prefix}.lora_A.weight"
            b_key = f"{prefix}.lora_B.weight"
            if a_key not in adapter or b_key not in adapter:
                continue

            base_key = f"h.{layer_idx}.{module}.weight"
            assert base_key in base, f"missing base tensor {base_key} for adapter pair"

            A = adapter[a_key].to(torch.float32)  # [r, in] = [16, 768]
            B = adapter[b_key].to(torch.float32)  # [out, r] = [2304, 16]
            assert A.shape == (r, 768), f"unexpected lora_A shape for {a_key}: {A.shape}"
            assert B.shape == (2304, r), f"unexpected lora_B shape for {b_key}: {B.shape}"

            base_weight = base[base_key].to(torch.float32)
            assert base_weight.shape == (768, 2304), (
                f"unexpected base shape for {base_key}: {base_weight.shape}"
            )

            delta = scale * (B @ A).T  # [768, 2304], Conv1D [in, out] layout
            assert delta.shape == base_weight.shape

            base[base_key] = (base_weight + delta).contiguous()
            merged_count += 1

    # --- Required checks ---
    assert merged_count == 12, f"expected exactly 12 adapter pairs merged, got {merged_count}"

    assert not any("lora_" in name for name in base), (
        "output must not contain any tensor name with 'lora_'"
    )

    assert base["h.0.attn.c_attn.weight"].shape == torch.Size([768, 2304]), (
        f"h.0.attn.c_attn.weight has wrong shape: {base['h.0.attn.c_attn.weight'].shape}"
    )

    assert len(base) == 160, f"expected exactly 160 tensors in output, got {len(base)}"

    # --- Shard and write ---
    os.makedirs(OUT_DIR, exist_ok=True)

    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # Deterministic ordering for reproducible shard assignment.
    names = sorted(base.keys())

    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for name in names:
        tensor = base[name]
        nbytes = tensor_nbytes(tensor)

        if nbytes > SHARD_LIMIT_BYTES:
            # Oversized tensor gets its own shard.
            if current:
                shards.append(current)
                current = {}
                current_bytes = 0
            shards.append({name: tensor})
            continue

        if current and current_bytes + nbytes > SHARD_LIMIT_BYTES:
            shards.append(current)
            current = {}
            current_bytes = 0

        current[name] = tensor
        current_bytes += nbytes

    if current:
        shards.append(current)

    num_shards = len(shards)
    weight_map = {}
    total_size = 0

    for i, shard in enumerate(shards, start=1):
        shard_filename = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        shard_path = os.path.join(OUT_DIR, shard_filename)
        save_file(shard, shard_path, metadata={"format": "pt"})
        for name, tensor in shard.items():
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(tensor)

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"Merged {merged_count} adapter pairs into {len(base)} tensors.")
    print(f"Wrote {num_shards} shard(s) to {OUT_DIR}/")


if __name__ == "__main__":
    main()
