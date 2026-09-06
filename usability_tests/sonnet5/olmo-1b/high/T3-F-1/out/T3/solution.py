#!/usr/bin/env python3
"""T3: mixed-precision export with sharding for OLMo-1B-0724-hf.

Casts the 112 attention/MLP projection matrices to bfloat16, keeps every
other tensor (here: model.embed_tokens.weight, lm_head.weight) in float32,
and writes a sharded safetensors checkpoint with an index file. Shards are
packed greedily to <= 256 MiB of tensor data each; any single tensor that
alone exceeds that budget gets its own shard.

Plain script on top of `torch` + `safetensors` (both allowed under
condition F) -- no HF `save_pretrained` sharding is used because it does not
give per-tensor dtype control, and no other listed tool exposes tensor-level
dtype casting plus custom shard packing together.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
IN_DIR = HERE.parent.parent / "inputs" / "base"
OUT_DIR = HERE
SHARD_BUDGET_BYTES = 256 * 1024 * 1024  # 256 MiB, tensor data only

NUM_LAYERS = 16
PROJ_SUFFIXES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]
BF16_NAMES = {
    f"model.layers.{i}.{suffix}" for i in range(NUM_LAYERS) for suffix in PROJ_SUFFIXES
}
assert len(BF16_NAMES) == 112, f"expected 112 projection names, built {len(BF16_NAMES)}"


def load_input_index() -> dict:
    with open(IN_DIR / "model.safetensors.index.json") as f:
        return json.load(f)


def load_all_tensors(weight_map: dict) -> dict:
    tensors = {}
    shard_names = sorted(set(weight_map.values()))
    for shard in shard_names:
        with safe_open(IN_DIR / shard, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def pack_shards(names: list[str], sizes: dict[str, int]) -> list[list[str]]:
    """Greedy bin-pack in given order; oversized tensors get a solo shard."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        size = sizes[name]
        if size > SHARD_BUDGET_BYTES:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
            continue
        if current and current_bytes + size > SHARD_BUDGET_BYTES:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def main() -> None:
    in_index = load_input_index()
    weight_map_in = in_index["weight_map"]
    all_names = list(weight_map_in.keys())
    assert len(all_names) == 114, f"expected 114 input tensors, found {len(all_names)}"

    tensors = load_all_tensors(weight_map_in)
    assert set(tensors.keys()) == set(all_names)

    out_tensors: dict[str, torch.Tensor] = {}
    for name, t in tensors.items():
        if name in BF16_NAMES:
            out_tensors[name] = t.to(torch.bfloat16).contiguous()
        else:
            assert t.dtype == torch.float32, f"{name}: expected float32 input, got {t.dtype}"
            out_tensors[name] = t.contiguous()

    # ---- Required checks: fail loudly before writing anything -------------
    bf16_count = sum(1 for t in out_tensors.values() if t.dtype == torch.bfloat16)
    assert bf16_count == 112, f"expected exactly 112 bfloat16 tensors, got {bf16_count}"
    assert out_tensors["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16, (
        "model.layers.0.self_attn.q_proj.weight must be bfloat16"
    )
    assert out_tensors["model.embed_tokens.weight"].dtype == torch.float32, (
        "model.embed_tokens.weight must be float32"
    )
    assert len(out_tensors) == 114, f"expected 114 output tensors, got {len(out_tensors)}"
    assert out_tensors["lm_head.weight"].dtype == torch.float32
    for name in BF16_NAMES:
        assert out_tensors[name].dtype == torch.bfloat16, f"{name} did not get cast"
    for name in all_names:
        if name not in BF16_NAMES:
            assert torch.equal(out_tensors[name], tensors[name]), f"{name} value changed"
    # -------------------------------------------------------------------------

    sizes = {name: t.numel() * t.element_size() for name, t in out_tensors.items()}
    shards = pack_shards(all_names, sizes)
    num_shards = len(shards)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weight_map_out: dict[str, str] = {}
    total_size = 0
    for shard_idx, names in enumerate(shards, start=1):
        shard_filename = f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {name: out_tensors[name] for name in names}
        save_file(shard_tensors, OUT_DIR / shard_filename, metadata={"format": "pt"})
        for name in names:
            weight_map_out[name] = shard_filename
            total_size += sizes[name]

    assert set(weight_map_out.keys()) == set(all_names)
    out_index = {"metadata": {"total_size": total_size}, "weight_map": weight_map_out}
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(out_index, f, indent=2, sort_keys=True)

    print(f"Wrote {num_shards} shard(s), {len(out_tensors)} tensors, "
          f"{bf16_count} bfloat16, total_size={total_size} bytes", file=sys.stderr)


if __name__ == "__main__":
    main()
