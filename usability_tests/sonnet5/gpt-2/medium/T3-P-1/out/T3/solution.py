"""
T3: Mixed-precision export with sharding (GPT-2 (124M))

Cast the 48 large projection matrices to bfloat16, keep everything else
(embeddings, layer norms, biases) as float32, drop the 12 non-parameter
causal-mask buffers, and write a sharded safetensors checkpoint (<=64MiB of
tensor data per shard, oversized tensors get their own shard) plus an index
file.
"""

import json
import os
import re

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(HERE, "..", "..", "inputs", "base", "model.safetensors")
OUTPUT_DIR = HERE
SHARD_LIMIT_BYTES = 64 * 1024 * 1024  # 64 MiB

# Projection matrices to cast to bfloat16: exactly these four per layer.
BF16_PATTERN = re.compile(
    r"^h\.\d+\.(attn\.c_attn\.weight|attn\.c_proj\.weight|mlp\.c_fc\.weight|mlp\.c_proj\.weight)$"
)
# Non-parameter buffers to drop.
BUFFER_PATTERN = re.compile(r"^h\.\d+\.attn\.bias$")


def load_state_dict(path):
    from safetensors import safe_open

    state_dict = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def main():
    state_dict = load_state_dict(INPUT_PATH)
    print(f"Loaded {len(state_dict)} tensors from {INPUT_PATH}")

    out = {}
    n_bf16 = 0
    n_dropped = 0
    for name, tensor in state_dict.items():
        if BUFFER_PATTERN.match(name):
            n_dropped += 1
            continue
        if BF16_PATTERN.match(name):
            out[name] = tensor.to(torch.bfloat16).contiguous()
            n_bf16 += 1
        else:
            out[name] = tensor.to(torch.float32).contiguous()

    print(f"Dropped {n_dropped} buffers, cast {n_bf16} tensors to bfloat16")

    # --- Required checks: fail loudly before writing ---
    actual_bf16 = [k for k, v in out.items() if v.dtype == torch.bfloat16]
    assert len(actual_bf16) == 48, f"expected 48 bfloat16 tensors, got {len(actual_bf16)}"
    assert out["h.0.attn.c_attn.weight"].dtype == torch.bfloat16, (
        "h.0.attn.c_attn.weight must be bfloat16"
    )
    assert out["wte.weight"].dtype == torch.float32, "wte.weight must be float32"
    assert len(out) == 148, f"expected 148 tensors in output, got {len(out)}"
    for name in state_dict:
        if not BUFFER_PATTERN.match(name):
            assert name in out, f"parameter {name} was dropped"
    for name in out:
        assert not BUFFER_PATTERN.match(name), f"buffer {name} was not dropped"
    print("All required checks passed.")

    # --- Sharding ---
    # Pack tensors into shards greedily in original order, respecting the
    # byte budget; a tensor larger than the budget gets its own shard.
    def tensor_nbytes(t):
        return t.numel() * t.element_size()

    names = list(out.keys())
    shards = []  # list of list of names
    current = []
    current_bytes = 0
    for name in names:
        nbytes = tensor_nbytes(out[name])
        if nbytes > SHARD_LIMIT_BYTES:
            if current:
                shards.append(current)
                current = []
                current_bytes = 0
            shards.append([name])
            continue
        if current and current_bytes + nbytes > SHARD_LIMIT_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += nbytes
    if current:
        shards.append(current)

    n_shards = len(shards)
    weight_map = {}
    total_size = 0
    for i, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {name: out[name] for name in shard_names}
        save_file(shard_tensors, os.path.join(OUTPUT_DIR, shard_filename))
        for name in shard_names:
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(out[name])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"Wrote {n_shards} shards and index to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
