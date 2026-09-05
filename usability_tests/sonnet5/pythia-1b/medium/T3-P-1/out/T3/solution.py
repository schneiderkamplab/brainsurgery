"""
T3: Mixed-precision export with sharding (Pythia-1B).

Casts the 64 large projection matrices to bfloat16, upcasts everything else
to float32, drops the 48 non-parameter buffers, and writes a sharded
safetensors checkpoint (<=256MiB of tensor data per shard, oversized tensors
alone in their own shard) with an index file.
"""

import json
import os
import re

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
INPUT_PATH = os.path.join(REPO_ROOT, "inputs", "base", "model.safetensors")
OUT_DIR = HERE

SHARD_BUDGET_BYTES = 256 * 1024 * 1024  # 256 MiB

# The 4 projection matrix name patterns, per layer.
PROJECTION_PATTERNS = [
    re.compile(r"^gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight$"),
    re.compile(r"^gpt_neox\.layers\.\d+\.attention\.dense\.weight$"),
    re.compile(r"^gpt_neox\.layers\.\d+\.mlp\.dense_h_to_4h\.weight$"),
    re.compile(r"^gpt_neox\.layers\.\d+\.mlp\.dense_4h_to_h\.weight$"),
]

# The 3 non-parameter buffer name patterns, per layer.
BUFFER_PATTERNS = [
    re.compile(r"^gpt_neox\.layers\.\d+\.attention\.bias$"),
    re.compile(r"^gpt_neox\.layers\.\d+\.attention\.masked_bias$"),
    re.compile(r"^gpt_neox\.layers\.\d+\.attention\.rotary_emb\.inv_freq$"),
]


def is_projection(name: str) -> bool:
    return any(p.match(name) for p in PROJECTION_PATTERNS)


def is_buffer(name: str) -> bool:
    return any(p.match(name) for p in BUFFER_PATTERNS)


def main() -> None:
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(INPUT_PATH, framework="pt") as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)

    total_input = len(tensors)
    print(f"Loaded {total_input} tensors from input.")

    output: dict[str, torch.Tensor] = {}
    dropped = []
    for name, tensor in tensors.items():
        if is_buffer(name):
            dropped.append(name)
            continue
        if is_projection(name):
            output[name] = tensor.to(torch.bfloat16).contiguous()
        else:
            output[name] = tensor.to(torch.float32).contiguous()

    # --- Required checks (fail loudly) ---
    if len(dropped) != 48:
        raise AssertionError(f"expected to drop 48 buffers, dropped {len(dropped)}: {dropped}")

    bf16_names = [n for n, t in output.items() if t.dtype == torch.bfloat16]
    if len(bf16_names) != 64:
        raise AssertionError(f"expected exactly 64 bfloat16 tensors, got {len(bf16_names)}")

    non_bf16 = [n for n, t in output.items() if t.dtype != torch.bfloat16 and t.dtype != torch.float32]
    if non_bf16:
        raise AssertionError(f"tensors with unexpected dtype (not bf16/fp32): {non_bf16}")

    check_name = "gpt_neox.layers.0.attention.query_key_value.weight"
    if output[check_name].dtype != torch.bfloat16:
        raise AssertionError(f"{check_name} is {output[check_name].dtype}, expected bfloat16")

    if output["gpt_neox.embed_in.weight"].dtype != torch.float32:
        raise AssertionError("gpt_neox.embed_in.weight is not float32")

    if len(output) != 196:
        raise AssertionError(f"expected 196 tensors in output, got {len(output)}")

    # No parameter (non-buffer) names were dropped: every dropped name must
    # match a buffer pattern (already enforced by is_buffer), and every kept
    # tensor from the input that is not a buffer must survive.
    expected_kept = {n for n in tensors if not is_buffer(n)}
    if set(output.keys()) != expected_kept:
        missing = expected_kept - set(output.keys())
        extra = set(output.keys()) - expected_kept
        raise AssertionError(f"key set mismatch; missing={missing} extra={extra}")

    # --- Sharding ---
    def tensor_nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # Stable order: as encountered in the input, minus dropped buffers.
    names_in_order = [n for n in tensors if n in output]

    shards: list[list[str]] = []
    current_shard: list[str] = []
    current_size = 0
    for name in names_in_order:
        size = tensor_nbytes(output[name])
        if size > SHARD_BUDGET_BYTES:
            # Oversized tensor gets its own shard.
            if current_shard:
                shards.append(current_shard)
                current_shard = []
                current_size = 0
            shards.append([name])
            continue
        if current_shard and current_size + size > SHARD_BUDGET_BYTES:
            shards.append(current_shard)
            current_shard = []
            current_size = 0
        current_shard.append(name)
        current_size += size
    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0

    os.makedirs(OUT_DIR, exist_ok=True)

    for idx, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
        shard_path = os.path.join(OUT_DIR, shard_filename)
        shard_tensors = {n: output[n] for n in shard_names}
        save_file(shard_tensors, shard_path)
        for n in shard_names:
            weight_map[n] = shard_filename
            total_size += tensor_nbytes(output[n])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path = os.path.join(OUT_DIR, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(f"Wrote {num_shards} shard(s) and index to {OUT_DIR}")
    print(f"Output tensor count: {len(output)}, bfloat16 tensors: {len(bf16_names)}")


if __name__ == "__main__":
    main()
