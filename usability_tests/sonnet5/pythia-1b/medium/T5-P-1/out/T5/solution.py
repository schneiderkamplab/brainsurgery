"""
T5: LoRA adapter merge with sharded export (Pythia-1B).

Loads the base Pythia-1B checkpoint and a PEFT-style LoRA adapter, merges
the adapter into the base attention.query_key_value weights
(weight += (lora_alpha / r) * B @ A, computed in float32, cast back to the
base dtype), verifies the required invariants, then writes the resulting
dense checkpoint as a sharded safetensors checkpoint (max 512 MiB of tensor
data per shard, oversized tensors get their own shard).
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent.parent
INPUTS = SANDBOX / "inputs"
OUT_DIR = SANDBOX / "out" / "T5"

BASE_PATH = INPUTS / "base" / "model.safetensors"
ADAPTER_PATH = INPUTS / "lora" / "adapter_model.safetensors"
ADAPTER_CONFIG_PATH = INPUTS / "lora" / "adapter_config.json"

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912 bytes of tensor data

ADAPTER_PREFIX = "base_model.model."
LORA_A_RE = re.compile(r"^(?P<base>.+)\.lora_A\.weight$")
LORA_B_RE = re.compile(r"^(?P<base>.+)\.lora_B\.weight$")


def load_all(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    with open(ADAPTER_CONFIG_PATH) as f:
        adapter_config = json.load(f)

    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    fan_in_fan_out = adapter_config["fan_in_fan_out"]
    if fan_in_fan_out:
        raise NotImplementedError(
            "fan_in_fan_out=True is not handled by this script"
        )
    scale = lora_alpha / r

    base_tensors = load_all(BASE_PATH)
    adapter_tensors = load_all(ADAPTER_PATH)

    # Map lora_A keys to lora_B keys via their shared base module name.
    a_keys = {}
    for key in adapter_tensors:
        m = LORA_A_RE.match(key)
        if m:
            a_keys[m.group("base")] = key

    b_keys = {}
    for key in adapter_tensors:
        m = LORA_B_RE.match(key)
        if m:
            b_keys[m.group("base")] = key

    if set(a_keys.keys()) != set(b_keys.keys()):
        raise RuntimeError(
            f"lora_A/lora_B key mismatch: "
            f"A-only={set(a_keys) - set(b_keys)}, B-only={set(b_keys) - set(a_keys)}"
        )

    merged_count = 0
    merged_base_names = set()

    for adapter_base_name, a_key in a_keys.items():
        b_key = b_keys[adapter_base_name]

        # adapter_base_name looks like:
        # base_model.model.gpt_neox.layers.<i>.attention.query_key_value
        if not adapter_base_name.startswith(ADAPTER_PREFIX):
            raise RuntimeError(f"unexpected adapter key prefix: {adapter_base_name}")
        base_name = adapter_base_name[len(ADAPTER_PREFIX) :] + ".weight"

        if base_name not in base_tensors:
            raise RuntimeError(
                f"adapter targets base tensor {base_name!r} which does not exist "
                f"in the base checkpoint"
            )

        A = adapter_tensors[a_key]
        B = adapter_tensors[b_key]
        base_weight = base_tensors[base_name]

        if A.shape[0] != r or B.shape[1] != r:
            raise RuntimeError(
                f"unexpected lora rank for {adapter_base_name}: "
                f"A.shape={tuple(A.shape)}, B.shape={tuple(B.shape)}, r={r}"
            )
        if A.shape[1] != base_weight.shape[1] or B.shape[0] != base_weight.shape[0]:
            raise RuntimeError(
                f"lora factor shapes {tuple(A.shape)}/{tuple(B.shape)} do not match "
                f"base weight shape {tuple(base_weight.shape)} for {base_name}"
            )

        delta = scale * (B.to(torch.float32) @ A.to(torch.float32))
        merged = base_weight.to(torch.float32) + delta
        merged = merged.to(base_weight.dtype)

        if merged.shape != base_weight.shape:
            raise RuntimeError(
                f"merged shape {tuple(merged.shape)} != base shape "
                f"{tuple(base_weight.shape)} for {base_name}"
            )

        base_tensors[base_name] = merged.contiguous()
        merged_count += 1
        merged_base_names.add(base_name)

    # --- Required checks -------------------------------------------------

    if merged_count != 16:
        raise AssertionError(f"expected 16 adapter pairs merged, got {merged_count}")

    lora_leftover = [k for k in base_tensors if "lora_" in k]
    if lora_leftover:
        raise AssertionError(f"lora_ tensors leaked into output: {lora_leftover}")

    probe_name = "gpt_neox.layers.0.attention.query_key_value.weight"
    if probe_name not in base_tensors:
        raise AssertionError(f"missing expected tensor {probe_name!r}")
    if tuple(base_tensors[probe_name].shape) != (6144, 2048):
        raise AssertionError(
            f"{probe_name} has shape {tuple(base_tensors[probe_name].shape)}, "
            f"expected (6144, 2048)"
        )

    if len(base_tensors) != 244:
        raise AssertionError(f"expected 244 tensors in output, got {len(base_tensors)}")

    # --- Sharded save ------------------------------------------------------

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # Deterministic ordering (base checkpoint order).
    names = list(base_tensors.keys())

    shards: list[list[str]] = []
    current_shard: list[str] = []
    current_bytes = 0

    for name in names:
        nbytes = tensor_nbytes(base_tensors[name])
        if nbytes > MAX_SHARD_BYTES:
            # Oversized tensor gets its own shard.
            if current_shard:
                shards.append(current_shard)
                current_shard = []
                current_bytes = 0
            shards.append([name])
            continue
        if current_shard and current_bytes + nbytes > MAX_SHARD_BYTES:
            shards.append(current_shard)
            current_shard = []
            current_bytes = 0
        current_shard.append(name)
        current_bytes += nbytes

    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    weight_map = {}
    shard_filenames = []

    for shard_idx, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
        shard_filenames.append(shard_filename)
        shard_tensors = {name: base_tensors[name] for name in shard_names}
        save_file(shard_tensors, str(OUT_DIR / shard_filename))
        for name in shard_names:
            weight_map[name] = shard_filename

    if set(weight_map.keys()) != set(names):
        raise AssertionError("weight_map does not cover all tensors")

    total_size = sum(tensor_nbytes(t) for t in base_tensors.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")

    print(f"Merged {merged_count} adapter pairs.")
    print(f"Wrote {len(names)} tensors across {num_shards} shards to {OUT_DIR}")


if __name__ == "__main__":
    main()
