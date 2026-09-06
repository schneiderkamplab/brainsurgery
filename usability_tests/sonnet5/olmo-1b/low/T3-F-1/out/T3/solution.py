"""
T3: Mixed-precision export with sharding (OLMo-1B-0724-hf).

Plain script on top of `safetensors` + `torch` (both in F-allowed.md). Loads
the input shards, casts exactly the 112 projection matrices to bfloat16,
leaves everything else (embeddings, lm_head) as float32, re-shards to a
256 MiB tensor-data budget per shard, and writes an index file. Fails loudly
via assertions before writing anything.
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_DIR = Path("inputs/base")
OUT_DIR = Path("out/T3")
MAX_SHARD_BYTES = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data

PROJ_RE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def load_all(in_dir: Path) -> dict[str, torch.Tensor]:
    index = json.loads((in_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    tensors: dict[str, torch.Tensor] = {}
    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        with safe_open(in_dir / shard_name, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    assert set(tensors) == set(weight_map), "loaded keys don't match index weight_map"
    return tensors


def main() -> None:
    tensors = load_all(IN_DIR)
    assert len(tensors) == 114, f"expected 114 tensors in input, got {len(tensors)}"

    proj_keys = {k for k in tensors if PROJ_RE.match(k)}
    assert len(proj_keys) == 112, f"expected 112 projection matrices, found {len(proj_keys)}"

    out_tensors: dict[str, torch.Tensor] = {}
    for key, t in tensors.items():
        if key in proj_keys:
            out_tensors[key] = t.to(torch.bfloat16).contiguous()
        else:
            assert t.dtype == torch.float32, f"{key} expected float32 input, got {t.dtype}"
            out_tensors[key] = t.contiguous()

    # Required checks: fail loudly before writing.
    n_bf16 = sum(1 for t in out_tensors.values() if t.dtype == torch.bfloat16)
    assert n_bf16 == 112, f"expected exactly 112 bfloat16 tensors, got {n_bf16}"
    assert out_tensors["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
    assert out_tensors["model.embed_tokens.weight"].dtype == torch.float32
    assert len(out_tensors) == 114, f"expected 114 output tensors, got {len(out_tensors)}"

    def tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # Greedy bin-packing into shards with a fixed byte budget; a single
    # tensor larger than the budget gets its own shard.
    items = sorted(tensors.keys())  # stable, deterministic order
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for key in items:
        size = tensor_bytes(out_tensors[key])
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([key])
            continue
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(key)
        current_bytes += size
    if current:
        shards.append(current)

    n_shards = len(shards)
    digits = max(5, len(str(n_shards)))
    weight_map: dict[str, str] = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_size = 0
    for i, shard_keys in enumerate(shards, start=1):
        shard_name = f"model-{i:0{digits}d}-of-{n_shards:0{digits}d}.safetensors"
        shard_tensors = {k: out_tensors[k] for k in shard_keys}
        save_file(shard_tensors, OUT_DIR / shard_name, metadata={"format": "pt"})
        for k in shard_keys:
            weight_map[k] = shard_name
            total_size += tensor_bytes(out_tensors[k])

    assert set(weight_map) == set(out_tensors), "weight_map does not cover all tensors"

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    print(f"Wrote {len(out_tensors)} tensors across {n_shards} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
