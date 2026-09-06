#!/usr/bin/env python
"""T5: LoRA merge-and-unload with sharded safetensors export (OLMo-1B-0724-hf).

Plain torch + safetensors script (condition F, no brainsurgery). Rationale for
this route over `peft.merge_and_unload` is in REPORT.md: it lets the exact
sharding rule (512 MiB budget, oversize tensors alone) be enforced directly,
instead of trusting a library's own shard splitter to match the spec.

Steps:
  1. Load adapter_config.json, validate the assumptions the task states
     (r=16, lora_alpha=32, fan_in_fan_out=False, the two target modules).
  2. Load all 64 adapter tensors, pair them up by (layer, module) into 32
     (A, B) pairs, and map each pair to its base tensor name.
  3. Load the sharded base checkpoint into one dict (float32, ~5 GiB - fits
     comfortably in memory).
  4. For each of the 32 pairs: base += scale * B @ A, computed in float32.
  5. Run the required checks before writing anything.
  6. Re-shard greedily under a 512 MiB (536,870,912 bytes) per-shard tensor
     budget, with any single tensor over that budget alone in its own shard.
  7. Write shards + model.safetensors.index.json.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]  # out/T5/solution.py -> out/T5 -> out -> sandbox root
INPUTS = REPO_ROOT / "inputs"
BASE_DIR = INPUTS / "base"
LORA_DIR = INPUTS / "lora"
OUT_DIR = REPO_ROOT / "out" / "T5"

SHARD_LIMIT_BYTES = 512 * 1024 * 1024  # 536,870,912
TARGET_MODULES = ["self_attn.q_proj", "self_attn.v_proj"]
NUM_LAYERS = 16


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def load_adapter_config() -> dict:
    cfg = json.loads((LORA_DIR / "adapter_config.json").read_text())
    if cfg.get("r") != 16:
        fail(f"expected r=16, got {cfg.get('r')!r}")
    if cfg.get("lora_alpha") != 32:
        fail(f"expected lora_alpha=32, got {cfg.get('lora_alpha')!r}")
    if cfg.get("fan_in_fan_out", False) is not False:
        fail(f"expected fan_in_fan_out=False, got {cfg.get('fan_in_fan_out')!r}")
    target = set(cfg.get("target_modules", []))
    expected = {"q_proj", "v_proj"}
    if target != expected:
        fail(f"expected target_modules={sorted(expected)}, got {sorted(target)}")
    return cfg


def load_base_state_dict() -> dict[str, torch.Tensor]:
    index = json.loads((BASE_DIR / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        with safe_open(BASE_DIR / shard_name, framework="pt") as f:
            for key in f.keys():
                if state_dict.setdefault(key, None) is not None:
                    fail(f"duplicate tensor name across base shards: {key}")
                state_dict[key] = f.get_tensor(key)
    if set(state_dict.keys()) != set(weight_map.keys()):
        fail("base shard contents do not match the index's weight_map")
    return state_dict


def load_adapter_tensors() -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(LORA_DIR / "adapter_model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    cfg = load_adapter_config()
    scale = cfg["lora_alpha"] / cfg["r"]

    base_sd = load_base_state_dict()
    adapter_tensors = load_adapter_tensors()

    if len(adapter_tensors) != 64:
        fail(f"expected 64 adapter tensors, found {len(adapter_tensors)}")

    merged_count = 0
    consumed_adapter_keys: set[str] = set()

    for layer in range(NUM_LAYERS):
        for module in TARGET_MODULES:
            prefix = f"base_model.model.model.layers.{layer}.{module}"
            a_key = f"{prefix}.lora_A.weight"
            b_key = f"{prefix}.lora_B.weight"
            if a_key not in adapter_tensors or b_key not in adapter_tensors:
                fail(f"missing adapter pair for layer {layer}, module {module}")

            A = adapter_tensors[a_key].to(torch.float32)
            B = adapter_tensors[b_key].to(torch.float32)
            if A.shape != (16, 2048):
                fail(f"{a_key}: expected shape [16, 2048], got {list(A.shape)}")
            if B.shape != (2048, 16):
                fail(f"{b_key}: expected shape [2048, 16], got {list(B.shape)}")

            base_key = f"model.layers.{layer}.{module}.weight"
            if base_key not in base_sd:
                fail(f"base checkpoint is missing {base_key}")
            base_tensor = base_sd[base_key]
            if base_tensor.dtype != torch.float32:
                fail(f"{base_key}: expected float32 base tensor, got {base_tensor.dtype}")
            if base_tensor.shape != (2048, 2048):
                fail(f"{base_key}: expected shape [2048, 2048], got {list(base_tensor.shape)}")

            delta = scale * (B @ A)
            if delta.shape != base_tensor.shape:
                fail(f"{base_key}: delta shape {list(delta.shape)} != base shape")

            base_sd[base_key] = (base_tensor + delta).contiguous()
            consumed_adapter_keys.add(a_key)
            consumed_adapter_keys.add(b_key)
            merged_count += 1

    # --- Required checks (fail loudly before writing) ---
    if merged_count != 32:
        fail(f"expected exactly 32 adapter pairs merged, got {merged_count}")
    if consumed_adapter_keys != set(adapter_tensors.keys()):
        leftover = set(adapter_tensors.keys()) - consumed_adapter_keys
        fail(f"adapter tensors left unconsumed: {sorted(leftover)}")
    if any("lora_" in name for name in base_sd.keys()):
        fail("an adapter tensor name leaked into the output state dict")
    q0 = base_sd.get("model.layers.0.self_attn.q_proj.weight")
    if q0 is None or tuple(q0.shape) != (2048, 2048):
        fail("model.layers.0.self_attn.q_proj.weight has the wrong shape after merge")
    if len(base_sd) != 114:
        fail(f"expected exactly 114 tensors in the output, got {len(base_sd)}")

    # --- Shard: greedy bin-packing under the 512 MiB tensor-data budget ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted(base_sd.keys())
    # The task names these two tensors explicitly as the ones that must be
    # alone in their own shard (they are the largest tensors in the
    # checkpoint). Isolate them unconditionally, on top of the generic
    # over-budget rule below, rather than relying on bin-packing order to
    # happen to separate them from everything else.
    ALWAYS_ALONE = {"model.embed_tokens.weight", "lm_head.weight"}

    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name in names:
        nbytes = base_sd[name].numel() * base_sd[name].element_size()
        if nbytes > SHARD_LIMIT_BYTES or name in ALWAYS_ALONE:
            if current:
                shards.append(current)
                current, current_size = [], 0
            shards.append([name])
            continue
        if current and current_size + nbytes > SHARD_LIMIT_BYTES:
            shards.append(current)
            current, current_size = [], 0
        current.append(name)
        current_size += nbytes
    if current:
        shards.append(current)

    num_shards = len(shards)
    weight_map = {}
    total_size = 0
    for shard_idx, shard_names_list in enumerate(shards, start=1):
        shard_file = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {}
        for name in shard_names_list:
            tensor = base_sd[name]
            shard_tensors[name] = tensor
            weight_map[name] = shard_file
            total_size += tensor.numel() * tensor.element_size()
        save_file(shard_tensors, OUT_DIR / shard_file, metadata={"format": "pt"})

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    print(f"OK: merged {merged_count} adapter pairs, wrote {num_shards} shards, "
          f"{len(base_sd)} tensors, {total_size} bytes total.")


if __name__ == "__main__":
    main()
