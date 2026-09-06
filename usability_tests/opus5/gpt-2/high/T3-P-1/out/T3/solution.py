#!/usr/bin/env python3
"""T3: mixed-precision export of GPT-2 (124M) as a sharded safetensors checkpoint.

- the 48 projection matrices (attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj)
  are cast to bfloat16;
- everything else stays float32, bit-for-bit;
- the 12 `h.<i>.attn.bias` causal-mask buffers are dropped;
- the result is written to out/T3/ as shards of at most 64 MiB of tensor data
  plus a model.safetensors.index.json weight map.

Targeting is done with an explicit name list, not a regex: `.*weight` would
also match wte/wpe and the layer norms, and `attn.bias` is a buffer, not a bias
parameter.
"""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_PATH = "inputs/base/model.safetensors"
OUT_DIR = "out/T3"
INDEX_NAME = "model.safetensors.index.json"

N_LAYERS = 12
MAX_SHARD_BYTES = 64 * 1024 * 1024  # 67_108_864

# The projection matrices to store in bfloat16, named exactly.
CAST_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)
CAST_KEYS = {f"h.{i}.{s}" for i in range(N_LAYERS) for s in CAST_SUFFIXES}

# Non-parameter buffers to delete.
DROP_KEYS = {f"h.{i}.attn.bias" for i in range(N_LAYERS)}

EXPECTED_CAST = 48
EXPECTED_DROP = 12
EXPECTED_OUT_TENSORS = 148


def die(msg):
    raise SystemExit(f"FAILED CHECK: {msg}")


def main():
    # ---------------------------------------------------------------- load
    with safe_open(IN_PATH, framework="pt") as f:
        in_keys = list(f.keys())
        state = {k: f.get_tensor(k) for k in in_keys}

    if len(state) != 160:
        die(f"expected 160 input tensors, got {len(state)}")

    # The name lists must actually exist in the checkpoint; a silent no-match
    # would produce a plausible-looking but wrong output.
    missing = sorted((CAST_KEYS | DROP_KEYS) - set(state))
    if missing:
        die(f"{len(missing)} targeted names absent from the input: {missing[:5]}")
    if len(CAST_KEYS) != EXPECTED_CAST:
        die(f"cast list has {len(CAST_KEYS)} names, expected {EXPECTED_CAST}")
    if len(DROP_KEYS) != EXPECTED_DROP:
        die(f"drop list has {len(DROP_KEYS)} names, expected {EXPECTED_DROP}")

    for k, t in state.items():
        if t.dtype is not torch.float32:
            die(f"input tensor {k} is {t.dtype}, expected float32")

    # ------------------------------------------------------------ transform
    out = {}
    for k in sorted(state):  # deterministic order, independent of header layout
        if k in DROP_KEYS:
            continue
        t = state[k]
        out[k] = t.to(torch.bfloat16).contiguous() if k in CAST_KEYS else t.contiguous()

    # -------------------------------------------------- checks before writing
    n_bf16 = sum(1 for t in out.values() if t.dtype is torch.bfloat16)
    if n_bf16 != EXPECTED_CAST:
        die(f"{n_bf16} bfloat16 tensors, expected exactly {EXPECTED_CAST}")

    if out["h.0.attn.c_attn.weight"].dtype is not torch.bfloat16:
        die("h.0.attn.c_attn.weight is not bfloat16")

    if out["wte.weight"].dtype is not torch.float32:
        die("wte.weight is not float32")

    if len(out) != EXPECTED_OUT_TENSORS:
        die(f"{len(out)} output tensors, expected {EXPECTED_OUT_TENSORS}")

    # every non-cast tensor must still be float32 with unchanged values
    for k, t in out.items():
        if k in CAST_KEYS:
            continue
        if t.dtype is not torch.float32:
            die(f"{k} should have stayed float32, is {t.dtype}")
        if not torch.equal(t, state[k]):
            die(f"{k} value changed but should have been passed through")

    # only buffers were deleted
    deleted = set(state) - set(out)
    if deleted != DROP_KEYS:
        die(f"deleted key set is wrong: {sorted(deleted ^ DROP_KEYS)}")

    # ---------------------------------------------------------------- shard
    # Greedy packing in key order; a tensor larger than the budget gets a shard
    # of its own (wte.weight, 154 MB).
    shards, cur, cur_bytes = [], {}, 0
    for k, t in out.items():
        n = t.numel() * t.element_size()
        if n > MAX_SHARD_BYTES:
            if cur:
                shards.append(cur)
                cur, cur_bytes = {}, 0
            shards.append({k: t})
            continue
        if cur_bytes + n > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[k] = t
        cur_bytes += n
    if cur:
        shards.append(cur)

    n_shards = len(shards)
    names = [f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)]

    for name, shard in zip(names, shards):
        size = sum(t.numel() * t.element_size() for t in shard.values())
        if size > MAX_SHARD_BYTES and len(shard) != 1:
            die(f"shard {name} holds {size} bytes over budget across {len(shard)} tensors")

    weight_map, total_size = {}, 0
    for name, shard in zip(names, shards):
        for k, t in shard.items():
            weight_map[k] = name
            total_size += t.numel() * t.element_size()

    if len(weight_map) != EXPECTED_OUT_TENSORS:
        die(f"weight_map covers {len(weight_map)} tensors, expected {EXPECTED_OUT_TENSORS}")

    # ---------------------------------------------------------------- write
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, shard in zip(names, shards):
        save_file(shard, os.path.join(OUT_DIR, name), metadata={"format": "pt"})

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(OUT_DIR, INDEX_NAME), "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
        fh.write("\n")

    # ------------------------------------------------- verify what was written
    seen = {}
    for name in names:
        with safe_open(os.path.join(OUT_DIR, name), framework="pt") as f:
            for k in f.keys():
                if k in seen:
                    die(f"{k} written to more than one shard")
                seen[k] = f.get_tensor(k)

    if set(seen) != set(out):
        die("reloaded key set differs from the intended output")
    for k, t in seen.items():
        if t.dtype is not out[k].dtype or t.shape != out[k].shape:
            die(f"{k} reloaded as {t.dtype}{tuple(t.shape)}")
        if not torch.equal(t.view(torch.int16 if t.dtype is torch.bfloat16 else torch.int32),
                           out[k].view(torch.int16 if t.dtype is torch.bfloat16 else torch.int32)):
            die(f"{k} is not bit-identical after the roundtrip")

    print(f"wrote {len(out)} tensors ({n_bf16} bfloat16) to {OUT_DIR}/ "
          f"in {n_shards} shards, {total_size} bytes of tensor data")
    for name, shard in zip(names, shards):
        size = sum(t.numel() * t.element_size() for t in shard.values())
        print(f"  {name}: {len(shard):3d} tensors, {size:>10d} bytes")


if __name__ == "__main__":
    main()
