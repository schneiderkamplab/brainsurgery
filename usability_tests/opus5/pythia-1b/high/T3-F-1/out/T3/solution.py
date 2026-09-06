#!/usr/bin/env python
"""T3: mixed-precision export of Pythia-1B as a sharded safetensors checkpoint.

Projection matrices -> bfloat16, everything else -> float32, non-parameter
buffers dropped, output sharded at 256 MiB with an index file.

Targeting is done with an explicit, fully enumerated key list rather than a
regex, so that a pattern can neither miss a tensor nor spill onto embeddings,
layer norms or biases. Every expectation is asserted; the script raises before
anything is written if a check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import load_file, save_file

IN_PATH = Path("inputs/base/model.safetensors")
OUT_DIR = Path("out/T3")
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024  # 268,435,456

N_LAYERS = 16

# The four projection matrices per layer, with the shapes the task specifies.
PROJECTIONS = {
    "attention.query_key_value.weight": (6144, 2048),
    "attention.dense.weight": (2048, 2048),
    "mlp.dense_h_to_4h.weight": (8192, 2048),
    "mlp.dense_4h_to_h.weight": (2048, 8192),
}

# Non-parameter buffers to drop.
BUFFERS = (
    "attention.bias",
    "attention.masked_bias",
    "attention.rotary_emb.inv_freq",
)

BF16_KEYS = {
    f"gpt_neox.layers.{i}.{suffix}": shape
    for i in range(N_LAYERS)
    for suffix, shape in PROJECTIONS.items()
}
DROP_KEYS = {f"gpt_neox.layers.{i}.{name}" for i in range(N_LAYERS) for name in BUFFERS}

EXPECTED_BF16 = 64
EXPECTED_DROPPED = 48
EXPECTED_OUT_TENSORS = 196


def fail(msg: str) -> None:
    raise SystemExit(f"CHECK FAILED: {msg}")


def build_state_dict() -> dict[str, torch.Tensor]:
    with safe_open(IN_PATH, framework="pt") as f:
        in_keys = list(f.keys())

        # Pre-flight: the enumerated key sets must exist exactly as expected.
        key_set = set(in_keys)
        missing_proj = sorted(BF16_KEYS.keys() - key_set)
        if missing_proj:
            fail(f"projection matrices absent from input: {missing_proj}")
        missing_buf = sorted(DROP_KEYS - key_set)
        if missing_buf:
            fail(f"buffers absent from input: {missing_buf}")
        if len(BF16_KEYS) != EXPECTED_BF16:
            fail(f"expected {EXPECTED_BF16} projection keys, enumerated {len(BF16_KEYS)}")
        if len(DROP_KEYS) != EXPECTED_DROPPED:
            fail(f"expected {EXPECTED_DROPPED} buffer keys, enumerated {len(DROP_KEYS)}")
        if BF16_KEYS.keys() & DROP_KEYS:
            fail("a key is marked both bfloat16 and dropped")

        out: dict[str, torch.Tensor] = {}
        for key in in_keys:
            if key in DROP_KEYS:
                continue
            tensor = f.get_tensor(key)
            if key in BF16_KEYS:
                expected = BF16_KEYS[key]
                if tuple(tensor.shape) != expected:
                    fail(f"{key}: shape {tuple(tensor.shape)}, expected {expected}")
                # float16 -> float32 is exact, so this matches a direct cast.
                out[key] = tensor.to(torch.float32).to(torch.bfloat16).contiguous()
            else:
                out[key] = tensor.to(torch.float32).contiguous()
    return out


def required_checks(sd: dict[str, torch.Tensor]) -> None:
    """The four checks the task mandates, plus the invariants behind them."""
    bf16 = sorted(k for k, v in sd.items() if v.dtype is torch.bfloat16)
    if len(bf16) != EXPECTED_BF16:
        fail(f"expected exactly {EXPECTED_BF16} bfloat16 tensors, found {len(bf16)}")
    if set(bf16) != set(BF16_KEYS):
        fail("the bfloat16 tensors are not exactly the intended projection matrices")

    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    if sd[probe].dtype is not torch.bfloat16:
        fail(f"{probe} is {sd[probe].dtype}, expected bfloat16")

    if sd["gpt_neox.embed_in.weight"].dtype is not torch.float32:
        fail(f"gpt_neox.embed_in.weight is {sd['gpt_neox.embed_in.weight'].dtype}, expected float32")

    if len(sd) != EXPECTED_OUT_TENSORS:
        fail(f"expected exactly {EXPECTED_OUT_TENSORS} tensors, found {len(sd)}")

    # No buffer survived, and nothing but a buffer was dropped.
    leftover = sorted(DROP_KEYS & sd.keys())
    if leftover:
        fail(f"buffers still present: {leftover}")

    # Everything that is not bfloat16 must be float32.
    bad = sorted(
        k for k, v in sd.items() if v.dtype not in (torch.bfloat16, torch.float32)
    )
    if bad:
        fail(f"tensors that are neither bfloat16 nor float32: {bad}")
    non_bf16 = sorted(k for k, v in sd.items() if v.dtype is not torch.bfloat16)
    if len(non_bf16) != EXPECTED_OUT_TENSORS - EXPECTED_BF16:
        fail(f"expected {EXPECTED_OUT_TENSORS - EXPECTED_BF16} float32 tensors, found {len(non_bf16)}")


def write_sharded(sd: dict[str, torch.Tensor]) -> None:
    split = split_torch_state_dict_into_shards(sd, max_shard_size=MAX_SHARD_BYTES)

    # Clean any output from an earlier attempt so no stale shard is left behind.
    if OUT_DIR.exists():
        for stale in sorted(OUT_DIR.glob("*.safetensors")):
            stale.unlink()
        stale_index = OUT_DIR / INDEX_NAME
        if stale_index.exists():
            stale_index.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, keys in split.filename_to_tensors.items():
        shard = {k: sd[k] for k in keys}
        save_file(shard, OUT_DIR / filename, metadata={"format": "pt"})

    index = {
        "metadata": {"total_size": split.metadata["total_size"]},
        "weight_map": split.tensor_to_filename,
    }
    (OUT_DIR / INDEX_NAME).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def verify_on_disk(expected: dict[str, torch.Tensor]) -> None:
    """Re-read what was written and re-run every requirement against it."""
    index = json.loads((OUT_DIR / INDEX_NAME).read_text())
    weight_map: dict[str, str] = index["weight_map"]

    shard_files = sorted(p.name for p in OUT_DIR.glob("*.safetensors"))
    if set(weight_map.values()) != set(shard_files):
        fail(f"weight_map files {sorted(set(weight_map.values()))} != shards on disk {shard_files}")

    reloaded: dict[str, torch.Tensor] = {}
    for name in shard_files:
        shard = load_file(OUT_DIR / name)
        payload = sum(t.numel() * t.element_size() for t in shard.values())
        if payload > MAX_SHARD_BYTES and len(shard) != 1:
            fail(f"{name}: {payload} bytes over the {MAX_SHARD_BYTES} budget with {len(shard)} tensors")
        for k, v in shard.items():
            if k in reloaded:
                fail(f"{k} appears in more than one shard")
            if weight_map.get(k) != name:
                fail(f"weight_map sends {k} to {weight_map.get(k)}, it is in {name}")
            reloaded[k] = v

    if reloaded.keys() != expected.keys():
        fail("key set on disk differs from the intended key set")
    required_checks(reloaded)

    for k, want in expected.items():
        got = reloaded[k]
        if got.dtype is not want.dtype:
            fail(f"{k}: dtype {got.dtype} on disk, expected {want.dtype}")
        if got.shape != want.shape:
            fail(f"{k}: shape {tuple(got.shape)} on disk, expected {tuple(want.shape)}")
        if not torch.equal(got.view(torch.int16 if got.dtype is torch.bfloat16 else torch.int32),
                           want.view(torch.int16 if want.dtype is torch.bfloat16 else torch.int32)):
            fail(f"{k}: values on disk are not bit-identical")

    total = sum(t.numel() * t.element_size() for t in reloaded.values())
    if total != index["metadata"]["total_size"]:
        fail(f"index total_size {index['metadata']['total_size']} != actual {total}")

    print(f"verified {len(reloaded)} tensors across {len(shard_files)} shards, {total:,} bytes")


def main() -> int:
    if not IN_PATH.exists():
        fail(f"input not found: {IN_PATH}")
    sd = build_state_dict()
    required_checks(sd)  # fails loudly before anything is written
    write_sharded(sd)
    verify_on_disk(sd)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
