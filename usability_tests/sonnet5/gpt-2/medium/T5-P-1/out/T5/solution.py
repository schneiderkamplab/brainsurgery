"""
T5: LoRA adapter merge with sharded export (GPT-2, 124M).

Merges a PEFT-style LoRA adapter (target_modules=["c_attn"], fan_in_fan_out=True)
into the base GPT-2 checkpoint, then writes the result as a sharded
safetensors checkpoint under out/T5/, with each shard holding at most
100 MiB (104,857,600 bytes) of tensor data (oversized tensors get their own
shard).
"""

import json
import os
import re

import torch
from safetensors.torch import save_file
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_PATH = os.path.join(TASK_DIR, "inputs", "base", "model.safetensors")
LORA_PATH = os.path.join(TASK_DIR, "inputs", "lora", "adapter_model.safetensors")
LORA_CONFIG_PATH = os.path.join(TASK_DIR, "inputs", "lora", "adapter_config.json")
OUT_DIR = os.path.join(HERE)

SHARD_MAX_BYTES = 100 * 1024 * 1024  # 104,857,600 bytes of tensor data per shard

LORA_A_RE = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")
LORA_B_RE = re.compile(r"^base_model\.model\.(.+)\.lora_B\.weight$")


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    with open(LORA_CONFIG_PATH) as f:
        lora_config = json.load(f)

    fan_in_fan_out = lora_config["fan_in_fan_out"]
    assert fan_in_fan_out is True, (
        f"expected fan_in_fan_out=true, got {fan_in_fan_out!r}"
    )
    scale = lora_config["lora_alpha"] / lora_config["r"]

    base = load_state_dict(BASE_PATH)
    lora = load_state_dict(LORA_PATH)

    # Pair up lora_A / lora_B tensors by base module name.
    a_names = {}
    b_names = {}
    for key in lora:
        m = LORA_A_RE.match(key)
        if m:
            a_names[m.group(1)] = key
            continue
        m = LORA_B_RE.match(key)
        if m:
            b_names[m.group(1)] = key
            continue
        raise ValueError(f"unexpected tensor name in adapter checkpoint: {key!r}")

    assert set(a_names) == set(b_names), (
        f"lora_A/lora_B module sets differ: only in A={set(a_names) - set(b_names)}, "
        f"only in B={set(b_names) - set(a_names)}"
    )

    merged_count = 0
    for module_name, a_key in a_names.items():
        base_key = f"{module_name}.weight"
        assert base_key in base, f"base checkpoint is missing {base_key!r}"

        A = lora[a_key].to(torch.float32)  # [r, in]
        B = lora[b_names[module_name]].to(torch.float32)  # [out, r]

        base_weight = base[base_key]
        assert base_weight.dtype == torch.float32, (
            f"{base_key} expected float32, got {base_weight.dtype}"
        )

        delta = scale * (B @ A)  # [out, in], nn.Linear convention
        delta = delta.T  # -> [in, out], Conv1D convention (fan_in_fan_out=True)

        assert delta.shape == base_weight.shape, (
            f"shape mismatch merging into {base_key}: "
            f"delta {tuple(delta.shape)} vs base {tuple(base_weight.shape)}"
        )

        base[base_key] = (base_weight.to(torch.float32) + delta).contiguous()
        merged_count += 1

    # --- Required checks ---
    assert merged_count == 12, f"expected 12 adapter pairs merged, got {merged_count}"
    assert not any("lora_" in name for name in base), (
        "output must not contain any tensor whose name contains 'lora_'"
    )
    assert base["h.0.attn.c_attn.weight"].shape == (768, 2304), (
        f"h.0.attn.c_attn.weight has shape {tuple(base['h.0.attn.c_attn.weight'].shape)}, "
        "expected (768, 2304)"
    )
    assert len(base) == 160, f"expected 160 tensors in output, got {len(base)}"

    # --- Shard the checkpoint ---
    # Greedily bin-pack tensors into shards of at most SHARD_MAX_BYTES of tensor
    # data each (in insertion order), except a tensor that alone exceeds the
    # budget gets its own shard.
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[list[str]] = []
    current_shard: list[str] = []
    current_bytes = 0

    for name, tensor in base.items():
        nbytes = tensor_nbytes(tensor)
        if nbytes > SHARD_MAX_BYTES:
            if current_shard:
                shards.append(current_shard)
                current_shard = []
                current_bytes = 0
            shards.append([name])
            continue
        if current_shard and current_bytes + nbytes > SHARD_MAX_BYTES:
            shards.append(current_shard)
            current_shard = []
            current_bytes = 0
        current_shard.append(name)
        current_bytes += nbytes

    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    weight_map = {}
    total_size = 0
    os.makedirs(OUT_DIR, exist_ok=True)

    for i, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {name: base[name] for name in shard_names}
        save_file(shard_tensors, os.path.join(OUT_DIR, shard_filename))
        for name in shard_names:
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(base[name])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")

    print(f"Merged {merged_count} LoRA adapter pairs into base checkpoint.")
    print(f"Wrote {len(base)} tensors across {num_shards} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
