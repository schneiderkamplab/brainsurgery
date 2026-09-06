"""
T3: Mixed-precision export with sharding (Pythia-1B).

Casts the 64 large projection matrices to bfloat16, upcasts everything else
to float32, drops the 48 non-parameter buffers, and writes a sharded
safetensors checkpoint (<=256 MiB tensor data per shard, oversized tensors
alone in their own shard) with an index file.
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
INPUT_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUT_DIR = HERE
SHARD_LIMIT = 256 * 1024 * 1024  # 256 MiB, tensor data only

BUFFER_SUFFIXES = (
    ".attention.bias",
    ".attention.masked_bias",
    ".attention.rotary_emb.inv_freq",
)

PROJECTION_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.(attention\.query_key_value|attention\.dense|"
    r"mlp\.dense_h_to_4h|mlp\.dense_4h_to_h)\.weight$"
)


def is_buffer(name: str) -> bool:
    return any(name.endswith(suf) for suf in BUFFER_SUFFIXES)


def is_projection(name: str) -> bool:
    return bool(PROJECTION_RE.match(name))


def main() -> None:
    tensors: dict[str, torch.Tensor] = {}

    with safe_open(str(INPUT_PATH), framework="pt") as f:
        names = list(f.keys())
        for name in names:
            if is_buffer(name):
                continue
            t = f.get_tensor(name)
            if is_projection(name):
                t = t.to(torch.bfloat16)
            else:
                t = t.to(torch.float32)
            tensors[name] = t

    # ---- Required checks ----
    bf16_names = [n for n, t in tensors.items() if t.dtype == torch.bfloat16]
    if len(bf16_names) != 64:
        raise RuntimeError(f"expected exactly 64 bfloat16 tensors, got {len(bf16_names)}")

    qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tensors[qkv0].dtype != torch.bfloat16:
        raise RuntimeError(f"{qkv0} must be bfloat16, got {tensors[qkv0].dtype}")

    embed_in = "gpt_neox.embed_in.weight"
    if tensors[embed_in].dtype != torch.float32:
        raise RuntimeError(f"{embed_in} must be float32, got {tensors[embed_in].dtype}")

    if len(tensors) != 196:
        raise RuntimeError(f"expected exactly 196 output tensors, got {len(tensors)}")

    for suf in BUFFER_SUFFIXES:
        if any(n.endswith(suf) for n in tensors):
            raise RuntimeError(f"buffer suffix {suf} still present in output")

    # ---- Shard assignment ----
    # Deterministic order: as encountered in the input file.
    ordered_names = [n for n in names if n in tensors]

    def nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0

    for name in ordered_names:
        size = nbytes(tensors[name])
        if size > SHARD_LIMIT:
            # Oversized tensor gets its own shard.
            if current:
                shards.append(current)
                current = []
                current_size = 0
            shards.append([name])
            continue
        if current and current_size + size > SHARD_LIMIT:
            shards.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += size

    if current:
        shards.append(current)

    n_shards = len(shards)
    digits = max(5, len(str(n_shards)))
    shard_filenames = [
        f"model-{i + 1:0{digits}d}-of-{n_shards:0{digits}d}.safetensors"
        for i in range(n_shards)
    ]

    weight_map: dict[str, str] = {}
    total_size = 0
    for shard_names, shard_file in zip(shards, shard_filenames):
        shard_tensors = {name: tensors[name] for name in shard_names}
        save_file(shard_tensors, str(OUT_DIR / shard_file))
        for name in shard_names:
            weight_map[name] = shard_file
            total_size += nbytes(tensors[name])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"Wrote {len(tensors)} tensors across {n_shards} shards to {OUT_DIR}")


if __name__ == "__main__":
    main()
