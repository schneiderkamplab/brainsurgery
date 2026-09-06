"""T3: mixed-precision sharded export of GPT-2 (124M).

- cast the 48 projection matrices (c_attn/c_proj/c_fc/mlp.c_proj weights) to bfloat16
- drop the 12 causal-mask buffers h.<i>.attn.bias
- keep everything else float32, values untouched
- write shards of at most 64 MiB tensor data plus model.safetensors.index.json
"""

import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SANDBOX = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(SANDBOX, "inputs", "base", "model.safetensors")
OUT_DIR = os.path.join(SANDBOX, "out", "T3")
MAX_SHARD_BYTES = 64 * 1024 * 1024  # 67,108,864

PROJ_RE = re.compile(r"^h\.(\d+)\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$")
BUFFER_RE = re.compile(r"^h\.(\d+)\.attn\.bias$")


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    # ---- load ------------------------------------------------------------
    state: dict[str, torch.Tensor] = {}
    with safe_open(SRC, framework="pt", device="cpu") as f:
        for name in f.keys():
            state[name] = f.get_tensor(name)
    print(f"loaded {len(state)} tensors from {SRC}")
    if len(state) != 160:
        fail(f"expected 160 input tensors, got {len(state)}")

    # ---- transform -------------------------------------------------------
    out: dict[str, torch.Tensor] = {}
    n_cast = n_dropped = 0
    for name, t in state.items():
        if BUFFER_RE.match(name):
            n_dropped += 1
            continue
        if PROJ_RE.match(name):
            if t.dtype != torch.float32:
                fail(f"{name}: expected float32 source, got {t.dtype}")
            out[name] = t.to(torch.bfloat16).contiguous()
            n_cast += 1
        else:
            if t.dtype != torch.float32:
                fail(f"{name}: expected float32, got {t.dtype}")
            out[name] = t.contiguous()
    print(f"cast {n_cast} matrices to bfloat16, dropped {n_dropped} buffers")

    # ---- required checks (before writing) --------------------------------
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 48:
        fail(f"expected exactly 48 bfloat16 tensors, got {n_bf16}")
    if n_cast != 48:
        fail(f"expected 48 casts, got {n_cast}")
    if n_dropped != 12:
        fail(f"expected 12 dropped buffers, got {n_dropped}")
    if out["h.0.attn.c_attn.weight"].dtype != torch.bfloat16:
        fail("h.0.attn.c_attn.weight is not bfloat16")
    if out["wte.weight"].dtype != torch.float32:
        fail("wte.weight is not float32")
    if len(out) != 148:
        fail(f"expected 148 output tensors, got {len(out)}")
    n_f32 = sum(1 for t in out.values() if t.dtype == torch.float32)
    if n_f32 + n_bf16 != len(out):
        fail("unexpected dtype present in output")
    # every non-cast tensor must be bit-identical to its source
    for name, t in out.items():
        if t.dtype == torch.float32 and not torch.equal(t, state[name]):
            fail(f"{name}: float32 values changed")
        if set(out) - set(state):
            fail("unexpected new tensor names")
    print("all pre-write checks passed")

    # ---- shard -----------------------------------------------------------
    def nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[dict[str, torch.Tensor]] = []
    cur: dict[str, torch.Tensor] = {}
    cur_size = 0
    for name, t in out.items():
        sz = nbytes(t)
        if cur and cur_size + sz > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_size = {}, 0
        cur[name] = t
        cur_size += sz
        if sz > MAX_SHARD_BYTES:
            # oversized tensor lives alone in its own shard
            shards.append(cur)
            cur, cur_size = {}, 0
    if cur:
        shards.append(cur)

    for shard in shards:
        total = sum(nbytes(t) for t in shard.values())
        if total > MAX_SHARD_BYTES and len(shard) != 1:
            fail(f"shard exceeds budget with {len(shard)} tensors ({total} bytes)")

    # ---- write -----------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            fail(f"destination already exists: {path}")
        save_file(shard, path, metadata={"format": "pt"})
        size = sum(nbytes(t) for t in shard.values())
        total_size += size
        for name in shard:
            weight_map[name] = fname
        print(f"wrote {fname}: {len(shard)} tensors, {size} bytes of tensor data")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    index_path = os.path.join(OUT_DIR, "model.safetensors.index.json")
    with open(index_path, "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
    print(f"wrote {index_path}: {len(weight_map)} entries, total_size={total_size}")

    # ---- post-write verification -----------------------------------------
    seen = 0
    for fname in sorted(set(weight_map.values())):
        with safe_open(os.path.join(OUT_DIR, fname), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            seen += len(keys)
            for k in keys:
                if weight_map[k] != fname:
                    fail(f"index mismatch for {k}")
                t = f.get_tensor(k)
                if t.dtype != out[k].dtype or not torch.equal(t, out[k]):
                    fail(f"{k}: written tensor differs from intended")
    if seen != 148 or len(weight_map) != 148:
        fail(f"post-write tensor count {seen}/{len(weight_map)} != 148")
    print("post-write verification passed: 148 tensors across", n, "shards")


if __name__ == "__main__":
    main()
