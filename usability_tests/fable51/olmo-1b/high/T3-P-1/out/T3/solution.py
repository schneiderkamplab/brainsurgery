"""T3: mixed-precision export of OLMo-1B-0724-hf with sharding.

Casts exactly the 112 per-layer projection matrices to bfloat16, keeps every
other tensor float32 and untouched, and writes a sharded safetensors
checkpoint (<= 256 MiB of tensor data per shard, oversized tensors alone)
with a model.safetensors.index.json.
"""

import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.dirname(os.path.dirname(ROOT))
IN_DIR = os.path.join(SANDBOX, "inputs", "base")
OUT_DIR = os.path.join(SANDBOX, "out", "T3")

SHARD_BUDGET = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data
NUM_LAYERS = 16
PROJ_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
CAST_RE = re.compile(
    r"^model\.layers\.(\d+)\.(" + "|".join(re.escape(p) for p in PROJ_NAMES) + r")\.weight$"
)


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def load_input() -> dict[str, torch.Tensor]:
    with open(os.path.join(IN_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]
    tensors: dict[str, torch.Tensor] = {}
    handles = {}
    try:
        for name, shard in weight_map.items():
            if shard not in handles:
                handles[shard] = safe_open(os.path.join(IN_DIR, shard), framework="pt", device="cpu")
            tensors[name] = handles[shard].get_tensor(name)
    finally:
        for h in handles.values():
            if hasattr(h, "__exit__"):
                h.__exit__(None, None, None)
    return tensors


def main() -> None:
    src = load_input()
    if len(src) != 114:
        fail(f"expected 114 input tensors, found {len(src)}")

    expected_cast = {
        f"model.layers.{i}.{p}.weight" for i in range(NUM_LAYERS) for p in PROJ_NAMES
    }
    if len(expected_cast) != 112:
        fail(f"internal: expected 112 cast targets, built {len(expected_cast)}")
    missing = sorted(expected_cast - src.keys())
    if missing:
        fail(f"cast targets missing from input: {missing[:5]} ...")

    out: dict[str, torch.Tensor] = {}
    for name, t in src.items():
        m = CAST_RE.match(name)
        if m:
            if int(m.group(1)) >= NUM_LAYERS:
                fail(f"unexpected layer index in {name}")
            if t.dtype != torch.float32:
                fail(f"{name} is {t.dtype}, expected float32 before casting")
            out[name] = t.to(torch.bfloat16).contiguous()
        else:
            if t.dtype != torch.float32:
                fail(f"non-projection tensor {name} is {t.dtype}, expected float32")
            out[name] = t.contiguous()

    matched = {n for n in out if CAST_RE.match(n)}
    if matched != expected_cast:
        fail(
            f"cast pattern mismatch: extra={sorted(matched - expected_cast)[:5]} "
            f"missing={sorted(expected_cast - matched)[:5]}"
        )

    # Required checks (fail loudly before writing anything).
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 112:
        fail(f"expected exactly 112 bfloat16 tensors, found {n_bf16}")
    if out["model.layers.0.self_attn.q_proj.weight"].dtype != torch.bfloat16:
        fail("model.layers.0.self_attn.q_proj.weight is not bfloat16")
    if out["model.embed_tokens.weight"].dtype != torch.float32:
        fail("model.embed_tokens.weight is not float32")
    if len(out) != 114:
        fail(f"expected 114 output tensors, found {len(out)}")
    if set(out) != set(src):
        fail("output key set differs from input key set")
    for name, t in out.items():
        if t.shape != src[name].shape:
            fail(f"shape changed for {name}: {tuple(src[name].shape)} -> {tuple(t.shape)}")
        if t.dtype == torch.float32 and not torch.equal(t, src[name]):
            fail(f"float32 tensor {name} changed value")
    n_f32 = sum(1 for t in out.values() if t.dtype == torch.float32)
    if n_bf16 + n_f32 != 114:
        fail(f"unexpected dtypes: {n_bf16} bf16 + {n_f32} f32 != 114")

    # Greedy sharding in input order; oversized tensors go alone in their own shard.
    def nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name, t in out.items():
        size = nbytes(t)
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

    # Verify sharding invariants before writing.
    total_size = 0
    for names in shards:
        size = sum(nbytes(out[n]) for n in names)
        total_size += size
        if size > SHARD_BUDGET and len(names) != 1:
            fail(f"shard exceeds budget with {len(names)} tensors: {size} bytes")
    if sum(len(s) for s in shards) != 114:
        fail("sharding lost or duplicated tensors")

    os.makedirs(OUT_DIR, exist_ok=True)
    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    for idx, names in enumerate(shards, start=1):
        fname = f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            fail(f"refusing to overwrite existing shard {path}")
        save_file({n: out[n] for n in names}, path, metadata={"format": "pt"})
        for n in names:
            weight_map[n] = fname

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")

    # Post-write verification: reread and compare.
    reread = 0
    for idx, names in enumerate(shards, start=1):
        fname = f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
        with safe_open(os.path.join(OUT_DIR, fname), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if set(keys) != set(names):
                fail(f"shard {fname} key set mismatch")
            for k in keys:
                t = f.get_tensor(k)
                if t.dtype != out[k].dtype or not torch.equal(t, out[k]):
                    fail(f"reread mismatch for {k} in {fname}")
                reread += 1
    if reread != 114:
        fail(f"reread {reread} tensors, expected 114")

    print(f"OK: wrote {n_shards} shards, {reread} tensors, {n_bf16} bf16 / {n_f32} f32, "
          f"total_size={total_size}")
    for idx, names in enumerate(shards, start=1):
        print(f"  shard {idx}: {len(names)} tensors, {sum(nbytes(out[n]) for n in names)} bytes")


if __name__ == "__main__":
    main()
