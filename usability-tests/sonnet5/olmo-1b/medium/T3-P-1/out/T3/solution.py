"""
T3: Mixed-precision export with sharding (OLMo-1B-0724-hf).

Loads the float32 safetensors checkpoint under inputs/base, casts the 112
attention/MLP projection matrices to bfloat16 (round-to-nearest-even via
tensor.to(torch.bfloat16)), leaves everything else (embeddings, lm_head) in
float32 with unchanged values, and writes a sharded safetensors checkpoint
under out/T3/ with a model.safetensors.index.json index.

Shard packing: greedy fill, each shard holds at most MAX_SHARD_BYTES of
tensor data; a tensor larger than that limit is written alone in its own
shard.
"""

import json
import os
import re

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
INPUT_DIR = os.path.join(REPO_ROOT, "inputs", "base")
OUTPUT_DIR = HERE

MAX_SHARD_BYTES = 256 * 1024 * 1024  # 256 MiB

# Exactly the projection matrices named in TASK.md: per-layer q/k/v/o_proj
# under self_attn, and gate/up/down_proj under mlp.
PROJ_PATTERN = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj\.weight"
    r"|mlp\.(gate|up|down)_proj\.weight)$"
)


def load_state_dict(input_dir: str) -> dict[str, torch.Tensor]:
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_names = sorted(set(weight_map.values()))
    tensors: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        from safetensors import safe_open

        shard_path = os.path.join(input_dir, shard_name)
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    assert set(tensors.keys()) == set(weight_map.keys()), "key mismatch vs index"
    return tensors


def main() -> None:
    state_dict = load_state_dict(INPUT_DIR)
    assert len(state_dict) == 114, f"expected 114 input tensors, got {len(state_dict)}"

    out_state_dict: dict[str, torch.Tensor] = {}
    proj_keys = []
    for name, tensor in state_dict.items():
        if PROJ_PATTERN.match(name):
            out_state_dict[name] = tensor.to(torch.bfloat16)
            proj_keys.append(name)
        else:
            assert tensor.dtype == torch.float32, f"unexpected non-fp32 input tensor {name}"
            out_state_dict[name] = tensor

    # --- Required checks (fail loudly before writing anything) ---
    n_bf16 = sum(1 for t in out_state_dict.values() if t.dtype == torch.bfloat16)
    assert n_bf16 == 112, f"expected exactly 112 bfloat16 tensors, got {n_bf16}"
    assert len(proj_keys) == 112, f"expected 112 projection matrices, matched {len(proj_keys)}"
    assert out_state_dict["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
    assert out_state_dict["model.embed_tokens.weight"].dtype == torch.float32
    assert len(out_state_dict) == 114, f"expected 114 output tensors, got {len(out_state_dict)}"

    # Sanity: non-projection tensors carry unchanged values (bit-identical fp32).
    for name, tensor in state_dict.items():
        if name not in proj_keys:
            assert torch.equal(tensor, out_state_dict[name]), f"value changed for {name}"

    # --- Shard packing: greedy fill, oversized tensors get their own shard ---
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    names = list(out_state_dict.keys())
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        nbytes = tensor_nbytes(out_state_dict[name])
        if nbytes > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current = []
                current_bytes = 0
            shards.append([name])
            continue
        if current and current_bytes + nbytes > MAX_SHARD_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += nbytes
    if current:
        shards.append(current)

    for shard_names in shards:
        total = sum(tensor_nbytes(out_state_dict[n]) for n in shard_names)
        if len(shard_names) > 1:
            assert total <= MAX_SHARD_BYTES, f"shard exceeds {MAX_SHARD_BYTES} bytes"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {name: out_state_dict[name] for name in shard_names}
        # safetensors requires contiguous tensors.
        shard_tensors = {n: t.contiguous() for n, t in shard_tensors.items()}
        save_file(shard_tensors, os.path.join(OUTPUT_DIR, shard_filename))
        for name in shard_names:
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(out_state_dict[name])

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    assert len(weight_map) == 114, f"expected 114 tensors in weight_map, got {len(weight_map)}"

    print(f"Wrote {n_shards} shards, {len(weight_map)} tensors to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
