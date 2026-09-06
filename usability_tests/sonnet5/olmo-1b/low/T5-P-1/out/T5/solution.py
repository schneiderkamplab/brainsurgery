"""
T5: LoRA adapter merge with sharded export (OLMo-1B-0724-hf).

Standalone PyTorch/safetensors script: merges a PEFT-style LoRA adapter into
the base checkpoint's q_proj/v_proj weights and writes the result as a
re-sharded safetensors checkpoint (max 512 MiB tensor data per shard, oversized
tensors alone in their own shard).
"""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_DIR = os.path.join(TASK_ROOT, "inputs", "base")
LORA_DIR = os.path.join(TASK_ROOT, "inputs", "lora")
OUT_DIR = os.path.join(TASK_ROOT, "out", "T5")

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912
LORA_ALPHA = 32
LORA_R = 16
SCALE = LORA_ALPHA / LORA_R
FAN_IN_FAN_OUT = False

LORA_NAME_RE = re.compile(
    r"^base_model\.model\.(model\.layers\.\d+\.self_attn\.(?:q|v)_proj)\.lora_([AB])\.weight$"
)


def load_base_state_dict():
    index_path = os.path.join(BASE_DIR, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    state = {}
    shard_files = sorted(set(weight_map.values()))
    for shard_file in shard_files:
        with safe_open(os.path.join(BASE_DIR, shard_file), framework="pt") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)

    assert set(state.keys()) == set(weight_map.keys()), "base tensor set mismatch vs index"
    return state


def load_lora_pairs():
    path = os.path.join(LORA_DIR, "adapter_model.safetensors")
    pairs = {}  # module_base_name -> {"A": tensor, "B": tensor}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            m = LORA_NAME_RE.match(key)
            if m is None:
                raise ValueError(f"unrecognized adapter tensor name: {key}")
            module_name, ab = m.group(1), m.group(2)
            pairs.setdefault(module_name, {})[ab] = f.get_tensor(key)
    return pairs


def merge(state, pairs):
    merged_count = 0
    for module_name, ab in pairs.items():
        assert "A" in ab and "B" in ab, f"incomplete lora pair for {module_name}"
        base_key = f"{module_name}.weight"
        assert base_key in state, f"base tensor {base_key} not found"

        A = ab["A"].to(torch.float32)
        B = ab["B"].to(torch.float32)
        assert A.shape == (LORA_R, 2048), f"unexpected lora_A shape for {module_name}: {A.shape}"
        assert B.shape == (2048, LORA_R), f"unexpected lora_B shape for {module_name}: {B.shape}"

        delta = SCALE * (B @ A)  # [out, in] = [2048, 2048]
        if FAN_IN_FAN_OUT:
            delta = delta.t()

        base = state[base_key]
        assert base.shape == delta.shape, f"shape mismatch merging {base_key}"
        state[base_key] = (base.to(torch.float32) + delta).to(base.dtype)
        merged_count += 1

    return merged_count


SOLO_SHARD_NAMES = {"model.embed_tokens.weight", "lm_head.weight"}


def shard_state_dict(state):
    """Greedy-pack tensors into shards of at most MAX_SHARD_BYTES tensor bytes.
    Any tensor whose own size exceeds the cap, or that is named in
    SOLO_SHARD_NAMES, is stored alone in its own shard."""
    names = sorted(state.keys())
    shards = []  # list of list[name]
    current = []
    current_bytes = 0

    for name in names:
        t = state[name]
        nbytes = t.numel() * t.element_size()
        if nbytes > MAX_SHARD_BYTES or name in SOLO_SHARD_NAMES:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
            continue
        if current and current_bytes + nbytes > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += nbytes

    if current:
        shards.append(current)

    return shards


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    state = load_base_state_dict()
    assert len(state) == 114, f"expected 114 base tensors, got {len(state)}"

    pairs = load_lora_pairs()
    assert len(pairs) == 32, f"expected 32 adapter pairs, got {len(pairs)}"

    merged_count = merge(state, pairs)
    assert merged_count == 32, f"expected 32 merges, got {merged_count}"

    assert not any("lora_" in name for name in state), "adapter tensor leaked into output"
    assert state["model.layers.0.self_attn.q_proj.weight"].shape == (
        2048,
        2048,
    ), "q_proj.weight shape changed"
    assert len(state) == 114, f"expected 114 output tensors, got {len(state)}"

    shards = shard_state_dict(state)
    num_shards = len(shards)
    weight_map = {}
    total_size = 0

    for shard_idx, names in enumerate(shards, start=1):
        shard_filename = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {name: state[name].contiguous() for name in names}
        save_file(shard_tensors, os.path.join(OUT_DIR, shard_filename))
        for name in names:
            weight_map[name] = shard_filename
            total_size += state[name].numel() * state[name].element_size()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"Merged {merged_count} adapter pairs into {len(state)} tensors, "
          f"written as {num_shards} shards to {OUT_DIR}")


if __name__ == "__main__":
    main()
