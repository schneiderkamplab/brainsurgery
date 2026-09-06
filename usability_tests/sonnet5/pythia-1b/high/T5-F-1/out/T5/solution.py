#!/usr/bin/env python3
"""T5: merge a PEFT-style LoRA adapter into the Pythia-1B base checkpoint and
write the result as a sharded safetensors checkpoint.

Approach: plain script directly on the checkpoint files (torch + safetensors
only). No model is instantiated; peft's `merge_and_unload` requires loading
the full HF model into memory and hands back a single (unsharded) state
dict, which then still needs custom sharding logic to satisfy the exact
shard-size and single-oversized-tensor rules below -- so it buys nothing
over operating on the raw tensors, and this way every step (name mapping,
scale, add, shard packing) is visible and checked explicitly.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent.parent
BASE_PATH = SANDBOX / "inputs" / "base" / "model.safetensors"
LORA_PATH = SANDBOX / "inputs" / "lora" / "adapter_model.safetensors"
ADAPTER_CONFIG_PATH = SANDBOX / "inputs" / "lora" / "adapter_config.json"
OUT_DIR = HERE
MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912 bytes of tensor data

# TASK.md, Required result #4 names these two tensors explicitly as always
# alone in their own shard. Note this does not follow from the stated general
# rule ("a tensor larger than the 512 MiB cap gets its own shard"): each is
# 206,045,184 bytes (~196.5 MiB), under the 536,870,912-byte cap, so ordinary
# greedy bin-packing (verified here against huggingface_hub's
# split_torch_state_dict_into_shards and mergekit's TensorWriter, both of
# which only isolate a tensor that individually exceeds max_shard_size) packs
# them alongside other tensors under any tensor ordering. Since the spec
# names them explicitly as a required fact about this checkpoint's output,
# they are isolated here directly rather than relying on the general rule.
ALWAYS_ISOLATE = {"gpt_neox.embed_in.weight", "embed_out.weight"}

LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"
ADAPTER_PREFIX = "base_model.model."


def load_all(path):
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def module_name_from_adapter_key(key, suffix):
    assert key.startswith(ADAPTER_PREFIX), f"unexpected adapter key prefix: {key}"
    assert key.endswith(suffix), f"unexpected adapter key suffix: {key}"
    return key[len(ADAPTER_PREFIX) : -len(suffix)]


def main():
    adapter_config = json.loads(ADAPTER_CONFIG_PATH.read_text())
    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    fan_in_fan_out = adapter_config["fan_in_fan_out"]
    assert fan_in_fan_out is False, (
        "solution assumes fan_in_fan_out=false (nn.Linear [out, in] layout); "
        f"got {fan_in_fan_out!r}"
    )
    scale = lora_alpha / r

    base = load_all(BASE_PATH)
    lora = load_all(LORA_PATH)

    lora_a = {}
    lora_b = {}
    for key, tensor in lora.items():
        if key.endswith(LORA_A_SUFFIX):
            lora_a[module_name_from_adapter_key(key, LORA_A_SUFFIX)] = tensor
        elif key.endswith(LORA_B_SUFFIX):
            lora_b[module_name_from_adapter_key(key, LORA_B_SUFFIX)] = tensor
        else:
            raise AssertionError(f"adapter tensor with unexpected name: {key}")

    modules_a = set(lora_a)
    modules_b = set(lora_b)
    assert modules_a == modules_b, (
        f"lora_A/lora_B module mismatch: only in A {modules_a - modules_b}, "
        f"only in B {modules_b - modules_a}"
    )

    # Required check: exactly 16 adapter pairs found.
    assert len(modules_a) == 16, f"expected exactly 16 adapter pairs, found {len(modules_a)}"

    merged_count = 0
    for module in sorted(modules_a):
        base_key = f"{module}.weight"
        assert base_key in base, f"no base tensor for adapter module {module!r} ({base_key})"

        base_weight = base[base_key]
        base_dtype = base_weight.dtype

        a = lora_a[module].to(torch.float32)
        b = lora_b[module].to(torch.float32)
        delta = scale * (b @ a)  # [out, in], same layout as base_weight (fan_in_fan_out=False)

        assert delta.shape == base_weight.shape, (
            f"delta shape {tuple(delta.shape)} != base shape {tuple(base_weight.shape)} "
            f"for {base_key}"
        )

        merged = (base_weight.to(torch.float32) + delta).to(base_dtype)
        base[base_key] = merged.contiguous()
        merged_count += 1

    assert merged_count == 16, f"expected to merge 16 weights, merged {merged_count}"

    # Required check: layer 0 qkv weight still has the expected shape.
    probe_key = "gpt_neox.layers.0.attention.query_key_value.weight"
    assert base[probe_key].shape == (6144, 2048), (
        f"{probe_key} has shape {tuple(base[probe_key].shape)}, expected (6144, 2048)"
    )

    # Required check: no adapter tensor leaks into the output.
    leaked = [k for k in base if "lora_" in k]
    assert not leaked, f"adapter tensor names leaked into output: {leaked}"

    # Required check: tensor count unchanged (244, same names as base).
    assert len(base) == 244, f"expected 244 tensors in output, got {len(base)}"

    write_sharded(base, OUT_DIR)
    print(f"Wrote {len(base)} tensors, {merged_count} merged, to {OUT_DIR}")


def write_sharded(tensors: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    def nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # Deterministic order (sorted by name) makes shard assignment reproducible.
    names = sorted(tensors.keys())

    # Greedy bin-packing: fill a shard until the next tensor would push it over
    # budget, then start a new one. A tensor that alone exceeds the budget, or
    # is in ALWAYS_ISOLATE (see comment above), gets its own shard.
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        size = nbytes(tensors[name])
        if name in ALWAYS_ISOLATE or size > MAX_SHARD_BYTES:
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

    for shard_names in shards:
        total = sum(nbytes(tensors[n]) for n in shard_names)
        assert total <= MAX_SHARD_BYTES or len(shard_names) == 1, (
            "a multi-tensor shard exceeds the 512 MiB budget"
        )

    num_shards = len(shards)
    weight_map = {}
    total_size = 0
    shard_filenames = []
    for i, shard_names in enumerate(shards, start=1):
        filename = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        shard_filenames.append(filename)
        shard_tensors = {n: tensors[n] for n in shard_names}
        save_file(shard_tensors, str(out_dir / filename), metadata={"format": "pt"})
        for n in shard_names:
            weight_map[n] = filename
            total_size += nbytes(tensors[n])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    assert len(weight_map) == len(tensors), "weight_map is missing some tensors"


if __name__ == "__main__":
    main()
