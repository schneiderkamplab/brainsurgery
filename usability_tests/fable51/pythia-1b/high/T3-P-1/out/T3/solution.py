"""T3: mixed-precision export of Pythia-1B with sharding.

- 64 projection matrices -> bfloat16
- everything else -> float32
- 48 attention buffers dropped
- sharded safetensors output (<= 256 MiB of tensor data per shard) + index
"""

import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "inputs", "base", "model.safetensors")
OUT_DIR = os.path.join(ROOT, "out", "T3")
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024  # 268,435,456

NUM_LAYERS = 16
PROJ_RE = re.compile(
    r"^gpt_neox\.layers\.(\d+)\."
    r"(attention\.query_key_value|attention\.dense|mlp\.dense_h_to_4h|mlp\.dense_4h_to_h)"
    r"\.weight$"
)
BUFFER_RE = re.compile(
    r"^gpt_neox\.layers\.(\d+)\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    # ---- load ------------------------------------------------------------
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(SRC, framework="pt", device="cpu") as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)
    print(f"loaded {len(tensors)} tensors from {SRC}")
    if len(tensors) != 244:
        fail(f"expected 244 input tensors, got {len(tensors)}")

    # ---- transform -------------------------------------------------------
    out: dict[str, torch.Tensor] = {}
    n_proj = 0
    n_dropped = 0
    for name, t in tensors.items():
        if BUFFER_RE.match(name):
            n_dropped += 1
            continue
        if PROJ_RE.match(name):
            if t.dim() != 2:
                fail(f"{name} matched projection pattern but has shape {tuple(t.shape)}")
            out[name] = t.to(torch.bfloat16).contiguous()
            n_proj += 1
        else:
            out[name] = t.to(torch.float32).contiguous()

    print(f"cast {n_proj} projection matrices to bfloat16, dropped {n_dropped} buffers")

    # ---- required checks (before writing) --------------------------------
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 64:
        fail(f"expected exactly 64 bfloat16 tensors, got {n_bf16}")
    if n_proj != 4 * NUM_LAYERS:
        fail(f"expected {4 * NUM_LAYERS} projection matches, got {n_proj}")
    if n_dropped != 3 * NUM_LAYERS:
        fail(f"expected {3 * NUM_LAYERS} dropped buffers, got {n_dropped}")
    qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    if qkv0 not in out or out[qkv0].dtype != torch.bfloat16:
        fail(f"{qkv0} is not bfloat16")
    emb = "gpt_neox.embed_in.weight"
    if emb not in out or out[emb].dtype != torch.float32:
        fail(f"{emb} is not float32")
    if len(out) != 196:
        fail(f"expected exactly 196 output tensors, got {len(out)}")
    # every non-bf16 tensor must be float32; no other dtype allowed
    bad = [n for n, t in out.items() if t.dtype not in (torch.bfloat16, torch.float32)]
    if bad:
        fail(f"unexpected dtypes: {bad[:5]}")
    # no parameter deleted: every input name that is not a buffer must survive
    missing = [n for n in tensors if not BUFFER_RE.match(n) and n not in out]
    if missing:
        fail(f"parameters missing from output: {missing[:5]}")
    # shapes unchanged
    for n, t in out.items():
        if tuple(t.shape) != tuple(tensors[n].shape):
            fail(f"shape changed for {n}: {tuple(tensors[n].shape)} -> {tuple(t.shape)}")

    # ---- shard -----------------------------------------------------------
    def nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name, t in out.items():
        size = nbytes(t)
        if size > MAX_SHARD_BYTES:
            # oversized tensor lives alone in its own shard
            if current:
                shards.append(current)
                current, current_size = [], 0
            shards.append([name])
            continue
        if current and current_size + size > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_size = [], 0
        current.append(name)
        current_size += size
    if current:
        shards.append(current)

    n_shards = len(shards)
    shard_names = [
        f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)
    ]

    # verify sharding invariants before writing
    for fname, names in zip(shard_names, shards):
        total = sum(nbytes(out[n]) for n in names)
        if total > MAX_SHARD_BYTES:
            if len(names) != 1:
                fail(f"shard {fname} holds {len(names)} tensors totalling {total} bytes")
        # if a single tensor is oversized, it must be alone
        if any(nbytes(out[n]) > MAX_SHARD_BYTES for n in names) and len(names) != 1:
            fail(f"oversized tensor shares shard {fname}")
    assigned = [n for names in shards for n in names]
    if sorted(assigned) != sorted(out.keys()) or len(assigned) != len(out):
        fail("shard assignment does not cover every tensor exactly once")

    # ---- write -----------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    # refuse to clobber an existing checkpoint silently
    stale = [
        f for f in os.listdir(OUT_DIR)
        if f.endswith(".safetensors") or f == INDEX_NAME
    ]
    if stale:
        print(f"removing stale output files: {stale}")
        for f in stale:
            os.remove(os.path.join(OUT_DIR, f))

    weight_map: dict[str, str] = {}
    total_size = 0
    for fname, names in zip(shard_names, shards):
        shard = {n: out[n] for n in names}
        shard_bytes = sum(nbytes(t) for t in shard.values())
        total_size += shard_bytes
        save_file(shard, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
        for n in names:
            weight_map[n] = fname
        print(f"wrote {fname}: {len(names)} tensors, {shard_bytes:,} bytes")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, INDEX_NAME), "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
        fh.write("\n")

    # ---- post-write verification ----------------------------------------
    seen = 0
    for fname in shard_names:
        with safe_open(os.path.join(OUT_DIR, fname), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            seen += len(keys)
            for k in keys:
                if weight_map[k] != fname:
                    fail(f"index mismatch for {k}")
                t = f.get_tensor(k)
                if t.dtype != out[k].dtype or tuple(t.shape) != tuple(out[k].shape):
                    fail(f"round-trip mismatch for {k}")
    if seen != 196 or len(weight_map) != 196:
        fail(f"expected 196 tensors on disk / in index, got {seen} / {len(weight_map)}")

    print(f"done: {n_shards} shards, {seen} tensors, total_size={total_size:,} bytes")
    print(f"index: {os.path.join(OUT_DIR, INDEX_NAME)}")


if __name__ == "__main__":
    main()
