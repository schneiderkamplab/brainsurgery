"""
T3: Mixed-precision export with sharding (OLMo-1B-0724-hf)

Loads the fp32 sharded checkpoint under inputs/base, casts exactly the 112
projection matrices (q/k/v/o_proj, gate/up/down_proj across 16 layers) to
bfloat16, leaves everything else (embed_tokens, lm_head) untouched in
float32, and re-shards the result into out/T3 with a 256 MiB per-shard cap
on tensor data (oversized single tensors get their own shard).
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
IN_DIR = HERE.parent.parent / "inputs" / "base"
OUT_DIR = HERE.parent / "T3"

SHARD_LIMIT_BYTES = 256 * 1024 * 1024  # 268,435,456 bytes

# Exactly the 112 projection matrices: 7 per layer, layers 0..15.
PROJ_PATTERN = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def load_index(base_dir: Path) -> dict:
    with open(base_dir / "model.safetensors.index.json") as f:
        return json.load(f)


def main() -> None:
    index = load_index(IN_DIR)
    weight_map = index["weight_map"]
    tensor_names = list(weight_map.keys())

    # Load every tensor, casting the 112 projection matrices to bfloat16
    # and leaving everything else as-is (float32, unchanged values).
    tensors: dict[str, torch.Tensor] = {}
    open_files = {}
    try:
        for shard_file in sorted(set(weight_map.values())):
            open_files[shard_file] = safe_open(
                IN_DIR / shard_file, framework="pt", device="cpu"
            )
        for name in tensor_names:
            f = open_files[weight_map[name]]
            t = f.get_tensor(name)
            if PROJ_PATTERN.match(name):
                t = t.to(torch.bfloat16)
            else:
                assert t.dtype == torch.float32, f"unexpected input dtype for {name}: {t.dtype}"
            tensors[name] = t.contiguous()
    finally:
        del open_files  # safe_open handles have no explicit close; drop refs

    # --- Required checks: fail loudly before writing anything ---
    bf16_names = [n for n, t in tensors.items() if t.dtype == torch.bfloat16]
    assert len(bf16_names) == 112, f"expected 112 bfloat16 tensors, got {len(bf16_names)}"
    assert set(bf16_names) == {n for n in tensor_names if PROJ_PATTERN.match(n)}, (
        "bfloat16 tensor set does not match the intended projection matrices"
    )
    assert tensors["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16, (
        "model.layers.0.self_attn.q_proj.weight must be bfloat16"
    )
    assert tensors["model.embed_tokens.weight"].dtype == torch.float32, (
        "model.embed_tokens.weight must be float32"
    )
    assert len(tensors) == 114, f"expected 114 tensors total, got {len(tensors)}"
    for name, t in tensors.items():
        if name not in bf16_names:
            assert t.dtype == torch.float32, f"{name} should be float32, got {t.dtype}"

    # --- Shard assignment ---
    # Oversized tensors (> shard limit) get a shard to themselves. Everything
    # else is greedily packed, in a stable order, so each shard's tensor data
    # stays under SHARD_LIMIT_BYTES.
    def byte_size(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    sizes = {name: byte_size(t) for name, t in tensors.items()}

    oversized = [n for n in tensor_names if sizes[n] > SHARD_LIMIT_BYTES]
    normal = [n for n in tensor_names if sizes[n] <= SHARD_LIMIT_BYTES]

    shard_groups: list[list[str]] = [[n] for n in oversized]

    current: list[str] = []
    current_size = 0
    for name in normal:
        size = sizes[name]
        if current and current_size + size > SHARD_LIMIT_BYTES:
            shard_groups.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += size
    if current:
        shard_groups.append(current)

    assert sum(len(g) for g in shard_groups) == 114, "shard groups lost or duplicated tensors"
    for group in shard_groups:
        group_size = sum(sizes[n] for n in group)
        if len(group) > 1:
            assert group_size <= SHARD_LIMIT_BYTES, "shard exceeds the 256 MiB limit"

    # --- Write shards + index ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    num_shards = len(shard_groups)
    new_weight_map: dict[str, str] = {}
    for i, group in enumerate(shard_groups, start=1):
        shard_name = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        shard_tensors = {name: tensors[name] for name in group}
        save_file(shard_tensors, OUT_DIR / shard_name, metadata={"format": "pt"})
        for name in group:
            new_weight_map[name] = shard_name

    assert len(new_weight_map) == 114
    total_size = sum(sizes.values())
    index_out = {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map,
    }
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index_out, f, indent=2, sort_keys=True)

    print(f"Wrote {len(new_weight_map)} tensors across {num_shards} shards to {OUT_DIR}")


if __name__ == "__main__":
    main()
