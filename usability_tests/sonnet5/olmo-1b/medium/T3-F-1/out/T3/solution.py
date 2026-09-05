"""
T3: Mixed-precision export with sharding (OLMo-1B-0724-hf)

Casts the 112 per-layer projection matrices to bfloat16, keeps every other
tensor (embeddings, lm_head) in float32, and writes a sharded safetensors
checkpoint (<=256 MiB of tensor data per shard, except lone oversized
tensors) with an index file, mirroring HuggingFace's sharded export layout.

Tools: `safetensors` for I/O, `torch` for the cast. Chosen over
`transformers ... save_pretrained(dtype=...)` because that casts the whole
model to one dtype; doing the cast at the state-dict level with an explicit
per-tensor predicate gives exact control over which 112 tensors move to
bfloat16 while everything else stays float32, and over shard packing.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_DIR = Path(__file__).resolve().parents[2] / "inputs" / "base"
OUT_DIR = Path(__file__).resolve().parent
MAX_SHARD_BYTES = 256 * 1024 * 1024  # 268,435,456

PROJ_RE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def load_state_dict(in_dir: Path) -> dict[str, torch.Tensor]:
    with open(in_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    state_dict: dict[str, torch.Tensor] = {}
    for shard, keys in shard_to_keys.items():
        with safe_open(in_dir / shard, framework="pt") as f:
            for key in keys:
                state_dict[key] = f.get_tensor(key)
    # Preserve the index's declared key order for deterministic sharding.
    return {key: state_dict[key] for key in weight_map}


def main() -> None:
    state_dict = load_state_dict(IN_DIR)

    out_dict: dict[str, torch.Tensor] = {}
    cast_count = 0
    for key, tensor in state_dict.items():
        if PROJ_RE.match(key):
            out_dict[key] = tensor.to(torch.bfloat16).contiguous()
            cast_count += 1
        else:
            out_dict[key] = tensor.to(torch.float32).contiguous()

    # --- Required checks: fail loudly before writing anything. ---
    n_bf16 = sum(1 for t in out_dict.values() if t.dtype == torch.bfloat16)
    assert cast_count == 112, f"expected to cast 112 tensors, cast {cast_count}"
    assert n_bf16 == 112, f"expected exactly 112 bfloat16 tensors, got {n_bf16}"
    assert out_dict["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16, (
        "model.layers.0.self_attn.q_proj.weight must be bfloat16"
    )
    assert out_dict["model.embed_tokens.weight"].dtype == torch.float32, (
        "model.embed_tokens.weight must be float32"
    )
    assert len(out_dict) == 114, f"expected 114 tensors, got {len(out_dict)}"
    # No buffers exist in this checkpoint per TASK.md; nothing to drop, and we
    # must not delete anything -- verify the key set is unchanged.
    assert set(out_dict) == set(state_dict), "tensor name set changed"

    # --- Shard packing: greedy bin-pack in key order, oversized tensors alone. ---
    def tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for key, tensor in out_dict.items():
        size = tensor_bytes(tensor)
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current, current_size = [], 0
            shards.append([key])
            continue
        if current and current_size + size > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_size = [], 0
        current.append(key)
        current_size += size
    if current:
        shards.append(current)

    for shard_keys in shards:
        sizes = [tensor_bytes(out_dict[k]) for k in shard_keys]
        assert sum(sizes) <= MAX_SHARD_BYTES or len(shard_keys) == 1, (
            "shard exceeds byte budget"
        )

    n_shards = len(shards)
    digits = max(5, len(str(n_shards)))
    shard_names = [
        f"model-{i + 1:0{digits}d}-of-{n_shards:0{digits}d}.safetensors"
        for i in range(n_shards)
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard_name, shard_keys in zip(shard_names, shards):
        shard_dict = {k: out_dict[k] for k in shard_keys}
        save_file(shard_dict, OUT_DIR / shard_name, metadata={"format": "pt"})
        for k in shard_keys:
            weight_map[k] = shard_name
            total_size += tensor_bytes(out_dict[k])

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"Wrote {len(out_dict)} tensors across {n_shards} shards to {OUT_DIR}")
    print(f"Cast {cast_count} projection tensors to bfloat16")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAILED CHECK: {e}", file=sys.stderr)
        sys.exit(1)
