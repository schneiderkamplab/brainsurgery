"""T2: remove attention head 5 from every layer of Pythia-1B (GPT-NeoX layout)."""

import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T2/model.safetensors"

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
HIDDEN = NUM_HEADS * HEAD_DIM  # 2048
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head in the fused projection
PRUNE_HEAD = 5
EXPECTED_TENSORS = 244


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def drop_block(t: torch.Tensor, dim: int, block: int, idx: int) -> torch.Tensor:
    """Remove the idx-th contiguous block of size `block` along `dim`."""
    n = t.shape[dim]
    if n % block != 0:
        fail(f"dim {dim} of size {n} is not a multiple of block {block}")
    lo, hi = block * idx, block * (idx + 1)
    keep = torch.cat([torch.arange(0, lo), torch.arange(hi, n)])
    return t.index_select(dim, keep).contiguous()


def main() -> None:
    state: dict[str, torch.Tensor] = {}
    with safe_open(SRC, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for k in f.keys():
            state[k] = f.get_tensor(k)

    if len(state) != EXPECTED_TENSORS:
        fail(f"input has {len(state)} tensors, expected {EXPECTED_TENSORS}")

    for i in range(NUM_LAYERS):
        p = f"gpt_neox.layers.{i}.attention."
        wk, bk, dk = p + "query_key_value.weight", p + "query_key_value.bias", p + "dense.weight"
        for k in (wk, bk, dk):
            if k not in state:
                fail(f"missing tensor {k}")

        w, b, d = state[wk], state[bk], state[dk]
        if tuple(w.shape) != (NUM_HEADS * QKV_BLOCK, HIDDEN):
            fail(f"{wk} has unexpected shape {tuple(w.shape)}")
        if tuple(b.shape) != (NUM_HEADS * QKV_BLOCK,):
            fail(f"{bk} has unexpected shape {tuple(b.shape)}")
        if tuple(d.shape) != (HIDDEN, HIDDEN):
            fail(f"{dk} has unexpected shape {tuple(d.shape)}")

        state[wk] = drop_block(w, 0, QKV_BLOCK, PRUNE_HEAD)  # rows: head block of q,k,v
        state[bk] = drop_block(b, 0, QKV_BLOCK, PRUNE_HEAD)
        state[dk] = drop_block(d, 1, HEAD_DIM, PRUNE_HEAD)  # columns: head input slice

        # Per-layer sanity: shapes and dtype preserved.
        exp = {
            wk: ((NUM_HEADS - 1) * QKV_BLOCK, HIDDEN),
            bk: ((NUM_HEADS - 1) * QKV_BLOCK,),
            dk: (HIDDEN, (NUM_HEADS - 1) * HEAD_DIM),
        }
        for k, shape in exp.items():
            if tuple(state[k].shape) != shape:
                fail(f"{k} has shape {tuple(state[k].shape)}, expected {shape}")
            if state[k].dtype != torch.float16:
                fail(f"{k} has dtype {state[k].dtype}, expected float16")

    # Required checks (TASK.md).
    checks = {
        "gpt_neox.layers.0.attention.query_key_value.weight": (5376, 2048),
        "gpt_neox.layers.0.attention.query_key_value.bias": (5376,),
        "gpt_neox.layers.0.attention.dense.weight": (2048, 1792),
    }
    for k, shape in checks.items():
        if tuple(state[k].shape) != shape:
            fail(f"required check: {k} has shape {tuple(state[k].shape)}, expected {shape}")
    if len(state) != EXPECTED_TENSORS:
        fail(f"required check: output has {len(state)} tensors, expected {EXPECTED_TENSORS}")

    # Verify the kept slices are bit-identical to the source ranges.
    with safe_open(SRC, framework="pt", device="cpu") as f:
        w0 = f.get_tensor("gpt_neox.layers.0.attention.query_key_value.weight")
        d0 = f.get_tensor("gpt_neox.layers.0.attention.dense.weight")
    ref_w = torch.cat([w0[:3840], w0[4608:]], dim=0)
    ref_d = torch.cat([d0[:, :1280], d0[:, 1536:]], dim=1)
    if not torch.equal(state["gpt_neox.layers.0.attention.query_key_value.weight"], ref_w):
        fail("qkv slice mismatch on layer 0")
    if not torch.equal(state["gpt_neox.layers.0.attention.dense.weight"], ref_d):
        fail("dense slice mismatch on layer 0")

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    save_file(state, DST, metadata=metadata)

    with safe_open(DST, framework="pt", device="cpu") as f:
        n = len(list(f.keys()))
    if n != EXPECTED_TENSORS:
        fail(f"written file has {n} tensors, expected {EXPECTED_TENSORS}")
    print(json.dumps({"output": DST, "tensors": n, "metadata": metadata}))


if __name__ == "__main__":
    main()
