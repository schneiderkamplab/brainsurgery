#!/usr/bin/env python3
"""T3: mixed-precision export with sharding for Pythia-1B.

Plain script on top of `safetensors` + `torch` (both allowed under
condition F). No merge-toolkit or HF sharded-save helper does exactly this
mix of per-tensor dtype rules + buffer deletion + byte-budget sharding, so a
short script is the most direct and auditable route.

- The 64 large projection matrices (query_key_value, dense, dense_h_to_4h,
  dense_4h_to_h weights, all 16 layers) -> bfloat16.
- Every other surviving tensor (embeddings, layer norms, all biases) -> float32.
- The 48 non-parameter buffers (attention.bias, attention.masked_bias,
  attention.rotary_emb.inv_freq, all 16 layers) -> deleted.
- Output written as safetensors shards, each holding at most 256 MiB
  (268,435,456 bytes) of tensor data, plus model.safetensors.index.json.
  A tensor whose own size exceeds the budget (embed_in.weight and
  embed_out.weight, once upcast to float32) is placed alone in its shard.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
IN_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUT_DIR = HERE.parent / "T3"
SHARD_BUDGET_BYTES = 256 * 1024 * 1024  # 268,435,456

NUM_LAYERS = 16

PROJECTION_SUFFIXES = [
    "attention.query_key_value.weight",
    "attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
]
BUFFER_SUFFIXES = [
    "attention.bias",
    "attention.masked_bias",
    "attention.rotary_emb.inv_freq",
]
# Fixed per-layer key order used to build the deterministic output order.
LAYER_KEY_ORDER = [
    "input_layernorm.weight",
    "input_layernorm.bias",
    "attention.query_key_value.weight",
    "attention.query_key_value.bias",
    "attention.dense.weight",
    "attention.dense.bias",
    "post_attention_layernorm.weight",
    "post_attention_layernorm.bias",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    "mlp.dense_4h_to_h.bias",
]


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not IN_PATH.exists():
        fail(f"input checkpoint not found at {IN_PATH}")

    state_dict = load_file(str(IN_PATH))

    bf16_keys = {
        f"gpt_neox.layers.{i}.{suf}" for i in range(NUM_LAYERS) for suf in PROJECTION_SUFFIXES
    }
    delete_keys = {
        f"gpt_neox.layers.{i}.{suf}" for i in range(NUM_LAYERS) for suf in BUFFER_SUFFIXES
    }

    missing_bf16 = bf16_keys - state_dict.keys()
    if missing_bf16:
        fail(f"expected projection keys missing from input: {sorted(missing_bf16)}")
    missing_del = delete_keys - state_dict.keys()
    if missing_del:
        fail(f"expected buffer keys missing from input: {sorted(missing_del)}")

    out: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key in delete_keys:
            continue
        if key in bf16_keys:
            out[key] = tensor.to(torch.bfloat16).contiguous()
        else:
            out[key] = tensor.to(torch.float32).contiguous()

    # ---- required checks: fail loudly before writing anything ----
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 64:
        fail(f"expected exactly 64 bfloat16 tensors, got {n_bf16}")
    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    if out[probe].dtype != torch.bfloat16:
        fail(f"{probe} is {out[probe].dtype}, expected bfloat16")
    if out["gpt_neox.embed_in.weight"].dtype != torch.float32:
        fail(f"gpt_neox.embed_in.weight is {out['gpt_neox.embed_in.weight'].dtype}, expected float32")
    if len(out) != 196:
        fail(f"expected exactly 196 output tensors, got {len(out)}")
    for key in out:
        if key in delete_keys:
            fail(f"deleted buffer key {key!r} resurfaced in output")

    # ---- deterministic key order: embed_in, layers 0..15, final norm, embed_out ----
    ordered_keys = ["gpt_neox.embed_in.weight"]
    for i in range(NUM_LAYERS):
        for suf in LAYER_KEY_ORDER:
            ordered_keys.append(f"gpt_neox.layers.{i}.{suf}")
    ordered_keys += ["gpt_neox.final_layer_norm.weight", "gpt_neox.final_layer_norm.bias"]
    ordered_keys.append("embed_out.weight")

    if set(ordered_keys) != set(out.keys()):
        missing = set(out.keys()) - set(ordered_keys)
        extra = set(ordered_keys) - set(out.keys())
        fail(f"ordered key list does not match output tensors; missing={missing} extra={extra}")

    # ---- greedy byte-budget sharding ----
    def nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    def flush_current() -> None:
        nonlocal current, current_bytes
        if current:
            shards.append(current)
            current = {}
            current_bytes = 0

    for key in ordered_keys:
        tensor = out[key]
        size = nbytes(tensor)
        if size > SHARD_BUDGET_BYTES:
            flush_current()
            shards.append({key: tensor})
            continue
        if current_bytes + size > SHARD_BUDGET_BYTES:
            flush_current()
        current[key] = tensor
        current_bytes += size
    flush_current()

    for shard in shards:
        shard_total = sum(nbytes(t) for t in shard.values())
        if shard_total > SHARD_BUDGET_BYTES and len(shard) > 1:
            fail(f"shard with {len(shard)} tensors exceeds budget ({shard_total} bytes)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for idx, shard in enumerate(shards, start=1):
        filename = f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
        save_file(shard, str(OUT_DIR / filename), metadata={"format": "pt"})
        for key in shard:
            weight_map[key] = filename
            total_size += nbytes(shard[key])

    if set(weight_map.keys()) != set(out.keys()):
        fail("weight_map key set does not match output tensor set")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote {len(out)} tensors across {n_shards} shards to {OUT_DIR}")
    print(f"bfloat16 tensors: {n_bf16}, total tensor bytes: {total_size}")


if __name__ == "__main__":
    main()
