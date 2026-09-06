"""T2: remove attention head 5 from every layer of Pythia-1B (GPT-NeoX layout).

Plain safetensors + torch script. Grading is bit-exact, so tensors are sliced
directly in their stored dtype and re-saved; nothing else is touched.
"""
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T2/model.safetensors")

N_LAYERS = 16
N_HEADS = 8
HEAD_DIM = 256
HIDDEN = 2048
PRUNE_HEAD = 5
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head: q | k | v interleaved per head

EXPECTED_TENSORS = 244


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"CHECK FAILED: {msg}")


def keep_index(n_blocks: int, block: int, drop: int) -> torch.Tensor:
    """Indices of all positions except block `drop` of width `block`, in order."""
    idx = torch.arange(n_blocks * block)
    mask = (idx // block) != drop
    return idx[mask]


def main() -> None:
    check(SRC.exists(), f"input missing: {SRC}")
    check(not DST.exists(), f"destination already exists: {DST}")

    qkv_keep = keep_index(N_HEADS, QKV_BLOCK, PRUNE_HEAD)  # rows 0..3839, 4608..6143
    dense_keep = keep_index(N_HEADS, HEAD_DIM, PRUNE_HEAD)  # cols 0..1279, 1536..2047
    check(qkv_keep.tolist() == list(range(0, 3840)) + list(range(4608, 6144)), "qkv index")
    check(dense_keep.tolist() == list(range(0, 1280)) + list(range(1536, 2048)), "dense index")

    out: dict[str, torch.Tensor] = {}
    touched = 0
    with safe_open(str(SRC), framework="pt") as f:
        keys = list(f.keys())
        check(len(keys) == EXPECTED_TENSORS, f"input has {len(keys)} tensors, expected {EXPECTED_TENSORS}")
        metadata = f.metadata()
        for k in keys:
            t = f.get_tensor(k)
            parts = k.split(".")
            is_layer = len(parts) > 3 and parts[0] == "gpt_neox" and parts[1] == "layers"
            suffix = ".".join(parts[3:]) if is_layer else None
            if suffix == "attention.query_key_value.weight":
                check(tuple(t.shape) == (N_HEADS * QKV_BLOCK, HIDDEN), f"{k} shape {tuple(t.shape)}")
                t = t[qkv_keep, :].contiguous()
                touched += 1
            elif suffix == "attention.query_key_value.bias":
                check(tuple(t.shape) == (N_HEADS * QKV_BLOCK,), f"{k} shape {tuple(t.shape)}")
                t = t[qkv_keep].contiguous()
                touched += 1
            elif suffix == "attention.dense.weight":
                check(tuple(t.shape) == (HIDDEN, N_HEADS * HEAD_DIM), f"{k} shape {tuple(t.shape)}")
                t = t[:, dense_keep].contiguous()
                touched += 1
            out[k] = t

    check(touched == 3 * N_LAYERS, f"touched {touched} tensors, expected {3 * N_LAYERS}")

    # Required checks (TASK.md), on layer 0 explicitly, then on every layer.
    check(tuple(out["gpt_neox.layers.0.attention.query_key_value.weight"].shape) == (5376, 2048),
          "layer 0 qkv weight shape")
    check(tuple(out["gpt_neox.layers.0.attention.query_key_value.bias"].shape) == (5376,),
          "layer 0 qkv bias shape")
    check(tuple(out["gpt_neox.layers.0.attention.dense.weight"].shape) == (2048, 1792),
          "layer 0 dense weight shape")
    for i in range(N_LAYERS):
        p = f"gpt_neox.layers.{i}.attention."
        check(tuple(out[p + "query_key_value.weight"].shape) == (5376, 2048), f"layer {i} qkv weight")
        check(tuple(out[p + "query_key_value.bias"].shape) == (5376,), f"layer {i} qkv bias")
        check(tuple(out[p + "dense.weight"].shape) == (2048, 1792), f"layer {i} dense weight")
    check(len(out) == EXPECTED_TENSORS, f"output has {len(out)} tensors, expected {EXPECTED_TENSORS}")

    # Spot-check values against the source for layer 0 to confirm block order.
    with safe_open(str(SRC), framework="pt") as f:
        w = f.get_tensor("gpt_neox.layers.0.attention.query_key_value.weight")
        d = f.get_tensor("gpt_neox.layers.0.attention.dense.weight")
    ow = out["gpt_neox.layers.0.attention.query_key_value.weight"]
    od = out["gpt_neox.layers.0.attention.dense.weight"]
    check(torch.equal(ow[:3840], w[:3840]) and torch.equal(ow[3840:], w[4608:]), "qkv row order")
    check(torch.equal(od[:, :1280], d[:, :1280]) and torch.equal(od[:, 1280:], d[:, 1536:]), "dense col order")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST), metadata=metadata)
    print(f"wrote {DST} with {len(out)} tensors; pruned head {PRUNE_HEAD} in {touched // 3} layers")


if __name__ == "__main__":
    sys.exit(main())
