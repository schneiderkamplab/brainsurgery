#!/usr/bin/env python
"""T3: mixed-precision export with sharding for OLMo-1B-0724-hf.

Plain script on top of `safetensors` + `torch` (both in the condition-F
allow-list). No merging or key-rewriting is needed here -- this is a
per-tensor dtype cast plus a re-shard -- so `mergekit` / `torch-state-bridge`
/ `transformers.save_pretrained` would only add indirection for something a
direct safetensors read/write already does exactly.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_DIR = Path("inputs/base")
OUT_DIR = Path("out/T3")
SHARD_BUDGET = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data per shard

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
PROJ_RE = re.compile(
    r"^model\.layers\.(\d+)\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight$"
)


def is_projection(name: str) -> bool:
    m = PROJ_RE.match(name)
    if not m:
        return False
    layer = int(m.group(1))
    return 0 <= layer < NUM_LAYERS


def main() -> None:
    with open(IN_DIR / "model.safetensors.index.json") as f:
        in_index = json.load(f)
    weight_map_in = in_index["weight_map"]
    shard_files = sorted(set(weight_map_in.values()))

    tensors: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        with safe_open(IN_DIR / shard, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    if set(tensors) != set(weight_map_in):
        raise RuntimeError("loaded tensor keys do not match index weight_map")

    expected_proj = {
        f"model.layers.{i}.{suffix}" for i in range(NUM_LAYERS) for suffix in PROJ_SUFFIXES
    }
    found_proj = {k for k in tensors if is_projection(k)}
    if found_proj != expected_proj:
        missing = expected_proj - found_proj
        extra = found_proj - expected_proj
        raise RuntimeError(f"projection set mismatch: missing={missing} extra={extra}")

    out_tensors: dict[str, torch.Tensor] = {}
    for name, t in tensors.items():
        if name in expected_proj:
            out_tensors[name] = t.to(torch.bfloat16).contiguous()
        else:
            if t.dtype != torch.float32:
                raise RuntimeError(f"expected float32 input for {name}, got {t.dtype}")
            out_tensors[name] = t.contiguous()

    # --- required checks: fail loudly before writing anything ---
    bf16_count = sum(1 for t in out_tensors.values() if t.dtype == torch.bfloat16)
    assert bf16_count == 112, f"expected 112 bfloat16 tensors, got {bf16_count}"
    assert out_tensors["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16, (
        "model.layers.0.self_attn.q_proj.weight must be bfloat16"
    )
    assert out_tensors["model.embed_tokens.weight"].dtype == torch.float32, (
        "model.embed_tokens.weight must be float32"
    )
    assert len(out_tensors) == 114, f"expected 114 tensors total, got {len(out_tensors)}"
    for name in tensors:
        assert out_tensors[name].shape == tensors[name].shape, f"shape changed for {name}"

    # --- shard: greedy bin-packing by declaration order, budget in tensor bytes ---
    def tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    names = list(out_tensors.keys())
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name in names:
        size = tensor_bytes(out_tensors[name])
        if size > SHARD_BUDGET:
            if current:
                shards.append(current)
                current, current_size = [], 0
            shards.append([name])
            continue
        if current and current_size + size > SHARD_BUDGET:
            shards.append(current)
            current, current_size = [], 0
        current.append(name)
        current_size += size
    if current:
        shards.append(current)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    num_shards = len(shards)
    weight_map_out: dict[str, str] = {}
    total_size = 0
    digits = max(5, len(str(num_shards)))
    for i, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{i:0{digits}d}-of-{num_shards:0{digits}d}.safetensors"
        shard_dict = {name: out_tensors[name] for name in shard_names}
        save_file(shard_dict, OUT_DIR / shard_filename, metadata={"format": "pt"})
        for name in shard_names:
            weight_map_out[name] = shard_filename
            total_size += tensor_bytes(out_tensors[name])

    if num_shards == 1:
        # normalize single-shard naming is unnecessary here (won't happen for this
        # checkpoint given the two >256MiB tensors), but keep index consistent regardless.
        pass

    out_index = {"metadata": {"total_size": total_size}, "weight_map": weight_map_out}
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(out_index, f, indent=2, sort_keys=True)

    print(f"wrote {len(out_tensors)} tensors across {num_shards} shards to {OUT_DIR}")


if __name__ == "__main__":
    main()
