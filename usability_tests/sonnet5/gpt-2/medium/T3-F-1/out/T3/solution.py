#!/usr/bin/env python
"""
T3: mixed-precision export with sharding for GPT-2 (124M).

- Cast the 48 projection matrices (attn.c_attn/c_proj, mlp.c_fc/c_proj weights,
  per layer) to bfloat16.
- Keep everything else (embeddings, layer norms, all biases) float32, values
  unchanged.
- Drop the 12 non-parameter causal-mask buffers `h.<i>.attn.bias`.
- Write a sharded safetensors checkpoint under out/T3/ with an index file,
  packing shards up to a 64 MiB tensor-data budget (a single oversized tensor
  gets its own shard).

Tools used: plain `torch` + `safetensors` (see F-allowed.md). No merge/adapter
tool applies here: this is precision casting + buffer pruning + resharding,
not a model merge or LoRA operation, so torch/safetensors is the most direct,
auditable route.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
IN_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUT_DIR = HERE.parent / "T3"
SHARD_BUDGET_BYTES = 64 * 1024 * 1024  # 64 MiB, tensor data only

# Exactly the 48 projection matrices to cast to bf16.
PROJ_RE = re.compile(
    r"^h\.\d+\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$"
)
# Non-parameter buffers to drop.
BUFFER_RE = re.compile(r"^h\.\d+\.attn\.bias$")


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    sd = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            sd[key] = f.get_tensor(key)
    return sd


def main() -> None:
    sd = load_state_dict(IN_PATH)
    n_in = len(sd)
    print(f"loaded {n_in} tensors from {IN_PATH}")

    out: dict[str, torch.Tensor] = {}
    cast_names = []
    dropped_names = []

    for name, tensor in sd.items():
        if BUFFER_RE.match(name):
            dropped_names.append(name)
            continue
        if PROJ_RE.match(name):
            out[name] = tensor.to(torch.bfloat16).contiguous()
            cast_names.append(name)
        else:
            # unchanged, float32, contiguous copy for safetensors
            out[name] = tensor.contiguous()

    print(f"cast to bf16: {len(cast_names)} tensors")
    print(f"dropped buffers: {len(dropped_names)} -> {dropped_names}")

    # ---- required checks: fail loudly before writing ----
    bf16_names = [n for n, t in out.items() if t.dtype == torch.bfloat16]
    assert len(bf16_names) == 48, (
        f"expected exactly 48 bfloat16 tensors, got {len(bf16_names)}: {bf16_names}"
    )
    assert "h.0.attn.c_attn.weight" in out and out["h.0.attn.c_attn.weight"].dtype == torch.bfloat16, (
        "h.0.attn.c_attn.weight must be bfloat16"
    )
    assert "wte.weight" in out and out["wte.weight"].dtype == torch.float32, (
        "wte.weight must be float32"
    )
    assert len(out) == 148, f"expected exactly 148 output tensors, got {len(out)}"
    for name in dropped_names:
        assert BUFFER_RE.match(name), f"attempted to drop non-buffer tensor {name}"
    assert len(dropped_names) == 12, f"expected exactly 12 dropped buffers, got {len(dropped_names)}"
    # sanity: no parameter accidentally dropped
    assert n_in - len(dropped_names) == len(out), "tensor count mismatch after drop"

    print("all required checks passed")

    # ---- pack into shards, <= SHARD_BUDGET_BYTES of tensor data each ----
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # Stable order: as encountered in the input state dict.
    names_in_order = [n for n in sd.keys() if n in out]

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names_in_order:
        size = tensor_nbytes(out[name])
        if size > SHARD_BUDGET_BYTES:
            # oversized tensor: gets its own shard
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

    for shard in shards:
        total = sum(tensor_nbytes(out[n]) for n in shard)
        assert total <= SHARD_BUDGET_BYTES or len(shard) == 1, (
            f"shard exceeds budget: {total} bytes, {len(shard)} tensors"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0

    for i, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {n: out[n] for n in shard_names}
        save_file(shard_tensors, str(OUT_DIR / shard_filename), metadata={"format": "pt"})
        for n in shard_names:
            weight_map[n] = shard_filename
            total_size += tensor_nbytes(out[n])
        print(f"wrote {shard_filename}: {len(shard_names)} tensors")

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path = OUT_DIR / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    print(f"wrote {index_path}")
    print(f"total tensors: {len(weight_map)}, shards: {n_shards}")


if __name__ == "__main__":
    main()
