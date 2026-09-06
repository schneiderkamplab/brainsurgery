"""T3: mixed-precision export with sharding for Pythia-1B."""

import json
import re
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_PATH = Path("inputs/base/model.safetensors")
OUT_DIR = Path("out/T3")
SHARD_LIMIT = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data

# Exactly the four projection matrices per layer; anchored so that biases,
# embeddings and layer norms cannot match.
BF16_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.(?:"
    r"attention\.query_key_value|attention\.dense|"
    r"mlp\.dense_h_to_4h|mlp\.dense_4h_to_h"
    r")\.weight$"
)

# Non-parameter buffers to drop.
DROP_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(?:bias|masked_bias|rotary_emb\.inv_freq)$"
)


def header_order(path):
    """Key order as stored in the file (== data layout order)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return [k for k in header if k != "__metadata__"]


def main():
    names = header_order(IN_PATH)
    tensors = {}
    with safe_open(IN_PATH, framework="pt") as fh:
        for name in names:
            if DROP_RE.match(name):
                continue
            t = fh.get_tensor(name)
            if BF16_RE.match(name):
                t = t.to(torch.float32).to(torch.bfloat16)
            else:
                t = t.to(torch.float32)
            tensors[name] = t.contiguous()

    order = [n for n in names if n in tensors]

    # --- required checks -------------------------------------------------
    n_bf16 = sum(1 for t in tensors.values() if t.dtype is torch.bfloat16)
    if n_bf16 != 64:
        raise SystemExit(f"CHECK FAILED: expected 64 bfloat16 tensors, got {n_bf16}")
    qkv = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tensors[qkv].dtype is not torch.bfloat16:
        raise SystemExit(f"CHECK FAILED: {qkv} is {tensors[qkv].dtype}, expected bfloat16")
    emb = "gpt_neox.embed_in.weight"
    if tensors[emb].dtype is not torch.float32:
        raise SystemExit(f"CHECK FAILED: {emb} is {tensors[emb].dtype}, expected float32")
    if len(tensors) != 196:
        raise SystemExit(f"CHECK FAILED: expected 196 output tensors, got {len(tensors)}")
    n_dropped = len(names) - len(tensors)
    if n_dropped != 48:
        raise SystemExit(f"CHECK FAILED: expected to drop 48 buffers, dropped {n_dropped}")
    non_f32 = [n for n, t in tensors.items()
               if t.dtype is not torch.float32 and not BF16_RE.match(n)]
    if non_f32:
        raise SystemExit(f"CHECK FAILED: non-float32 leftovers: {non_f32[:5]}")

    # --- greedy sharding in checkpoint order -----------------------------
    shards, cur, cur_bytes = [], [], 0
    for name in order:
        nbytes = tensors[name].numel() * tensors[name].element_size()
        if cur and cur_bytes + nbytes > SHARD_LIMIT:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(name)
        cur_bytes += nbytes
        if cur_bytes >= SHARD_LIMIT:  # full (or an oversized lone tensor): close it
            shards.append(cur)
            cur, cur_bytes = [], 0
    if cur:
        shards.append(cur)

    total = len(shards)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*.safetensors"):
        stale.unlink()

    weight_map, total_size = {}, 0
    for i, keys in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{total:05d}.safetensors"
        data_bytes = sum(tensors[k].numel() * tensors[k].element_size() for k in keys)
        if len(keys) > 1 and data_bytes > SHARD_LIMIT:
            raise SystemExit(f"CHECK FAILED: shard {fname} is {data_bytes} bytes > limit")
        save_file({k: tensors[k] for k in keys}, OUT_DIR / fname, metadata={"format": "pt"})
        for k in keys:
            weight_map[k] = fname
        total_size += data_bytes

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    print(f"wrote {len(weight_map)} tensors ({n_bf16} bfloat16) into {total} shards")
    for i, keys in enumerate(shards, start=1):
        b = sum(tensors[k].numel() * tensors[k].element_size() for k in keys)
        print(f"  shard {i:>2}: {len(keys):>3} tensors, {b/2**20:8.2f} MiB")


if __name__ == "__main__":
    main()
