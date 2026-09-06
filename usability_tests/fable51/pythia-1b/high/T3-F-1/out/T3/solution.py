"""T3: mixed-precision sharded export of Pythia-1B.

Plain script on top of safetensors + torch. Exact tensor names are targeted
(no wildcard regex), so embeddings, norms, biases and buffers cannot be hit by
accident. The required checks are enforced before anything is written.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("inputs/base/model.safetensors")
OUT = Path("out/T3")
NUM_LAYERS = 16
SHARD_BUDGET = 256 * 1024 * 1024  # bytes of tensor data per shard, headers excluded

PROJ_RE = re.compile(
    r"^gpt_neox\.layers\.(\d+)\.(attention\.query_key_value|attention\.dense|"
    r"mlp\.dense_h_to_4h|mlp\.dense_4h_to_h)\.weight$"
)
BUFFER_RE = re.compile(
    r"^gpt_neox\.layers\.(\d+)\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    # Refuse to overwrite an existing output.
    existing = [p for p in OUT.glob("*.safetensors*")]
    if existing:
        fail(f"output already exists in {OUT}: {[p.name for p in existing]}")

    out: dict[str, torch.Tensor] = {}
    n_proj = n_buf = 0
    with safe_open(str(SRC), framework="pt", device="cpu") as f:
        for name in f.keys():
            if BUFFER_RE.match(name):
                n_buf += 1
                continue  # drop non-parameter buffer
            t = f.get_tensor(name)
            if PROJ_RE.match(name):
                n_proj += 1
                out[name] = t.to(torch.bfloat16)
            else:
                out[name] = t.to(torch.float32)

    # ---- Required checks (before writing) ----
    if n_proj != 4 * NUM_LAYERS:
        fail(f"matched {n_proj} projection matrices, expected {4 * NUM_LAYERS}")
    if n_buf != 3 * NUM_LAYERS:
        fail(f"matched {n_buf} buffers, expected {3 * NUM_LAYERS}")
    n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
    if n_bf16 != 64:
        fail(f"{n_bf16} bfloat16 tensors, expected 64")
    if out["gpt_neox.layers.0.attention.query_key_value.weight"].dtype != torch.bfloat16:
        fail("layers.0 query_key_value.weight is not bfloat16")
    if out["gpt_neox.embed_in.weight"].dtype != torch.float32:
        fail("embed_in.weight is not float32")
    if len(out) != 196:
        fail(f"output has {len(out)} tensors, expected 196")
    others = {k: t.dtype for k, t in out.items() if t.dtype != torch.bfloat16}
    if any(d != torch.float32 for d in others.values()):
        fail(f"non-float32 non-projection tensors: {others}")

    # ---- Greedy sharding in key order; oversized tensors go alone ----
    shards: list[dict[str, torch.Tensor]] = []
    cur: dict[str, torch.Tensor] = {}
    cur_size = 0
    for name, t in out.items():
        nbytes = t.numel() * t.element_size()
        if nbytes > SHARD_BUDGET:
            if cur:
                shards.append(cur)
                cur, cur_size = {}, 0
            shards.append({name: t})
            continue
        if cur and cur_size + nbytes > SHARD_BUDGET:
            shards.append(cur)
            cur, cur_size = {}, 0
        cur[name] = t
        cur_size += nbytes
    if cur:
        shards.append(cur)

    for shard in shards:
        size = sum(t.numel() * t.element_size() for t in shard.values())
        if size > SHARD_BUDGET and len(shard) != 1:
            fail(f"shard of {len(shard)} tensors exceeds budget: {size} bytes")

    # ---- Write ----
    OUT.mkdir(parents=True, exist_ok=True)
    n = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file({k: v.contiguous() for k, v in shard.items()}, str(OUT / fname), metadata={"format": "pt"})
        for k, v in shard.items():
            weight_map[k] = fname
            total_size += v.numel() * v.element_size()
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (OUT / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    # ---- Post-write verification ----
    seen = 0
    for i in range(1, n + 1):
        with safe_open(str(OUT / f"model-{i:05d}-of-{n:05d}.safetensors"), "pt") as f:
            for k in f.keys():
                seen += 1
                if weight_map[k] != f"model-{i:05d}-of-{n:05d}.safetensors":
                    fail(f"index mismatch for {k}")
    if seen != 196 or len(weight_map) != 196:
        fail(f"wrote {seen} tensors / index has {len(weight_map)}, expected 196")
    print(f"OK: {seen} tensors in {n} shards, {total_size} bytes of tensor data")


if __name__ == "__main__":
    main()
