#!/usr/bin/env python
"""T2: structured attention-head pruning of Pythia-1B.

Removes head 5 from every one of the 16 layers, at the checkpoint level:

  * attention.query_key_value.weight [6144, 2048] -> [5376, 2048]
    The fused projection is GPT-NeoX interleaved: head h owns the 768-row
    block 768*h .. 768*h+767, holding its q (256), k (256) and v (256) rows
    back to back.  Dropping a head is therefore dropping one whole 768-row
    block, which takes q, k and v with it.
  * attention.query_key_value.bias [6144] -> [5376], same row layout.
  * attention.dense.weight [2048, 2048] -> [2048, 1792]
    Output projection in nn.Linear [out, in] layout, so the heads live on the
    *column* (input) axis as 256-wide blocks.

Everything else, including attention.dense.bias and the attention buffers, is
head-independent and is copied through byte for byte.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("inputs/base/model.safetensors")
DST_DIR = Path("out/T2")
DST = DST_DIR / "model.safetensors"

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
HIDDEN = 2048
PRUNE_HEAD = 5

QKV_PER_HEAD = 3 * HEAD_DIM  # 768: q, k, v rows of one head, in that order
EXPECTED_TENSORS = 244


def fail(msg: str) -> None:
    """Abort loudly; no output file is written."""
    raise SystemExit(f"FAIL: {msg}")


def keep_indices(block: int, n_blocks: int, drop: int) -> torch.Tensor:
    """Indices of every block except `drop`, in ascending order."""
    lo, hi = drop * block, (drop + 1) * block
    return torch.cat(
        [torch.arange(0, lo, dtype=torch.long), torch.arange(hi, n_blocks * block, dtype=torch.long)]
    )


def main() -> None:
    if not SRC.is_file():
        fail(f"input checkpoint not found: {SRC}")

    # Index sets derived from the head geometry, then pinned against the literal
    # ranges the task specifies, so a wrong geometry can never reach the output.
    qkv_keep = keep_indices(QKV_PER_HEAD, NUM_HEADS, PRUNE_HEAD)
    dense_keep = keep_indices(HEAD_DIM, NUM_HEADS, PRUNE_HEAD)

    want_qkv = torch.cat([torch.arange(0, 3840), torch.arange(4608, 6144)])
    want_dense = torch.cat([torch.arange(0, 1280), torch.arange(1536, 2048)])
    if not torch.equal(qkv_keep, want_qkv):
        fail("derived qkv keep-indices do not match rows 0..3839 + 4608..6143")
    if not torch.equal(dense_keep, want_dense):
        fail("derived dense keep-indices do not match columns 0..1279 + 1536..2047")

    out: dict[str, torch.Tensor] = {}
    src_dtypes: dict[str, torch.dtype] = {}
    touched: set[str] = set()

    with safe_open(SRC, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        keys = list(f.keys())
        if len(keys) != EXPECTED_TENSORS:
            fail(f"input has {len(keys)} tensors, expected {EXPECTED_TENSORS}")

        qkv_w = {f"gpt_neox.layers.{i}.attention.query_key_value.weight" for i in range(NUM_LAYERS)}
        qkv_b = {f"gpt_neox.layers.{i}.attention.query_key_value.bias" for i in range(NUM_LAYERS)}
        dense_w = {f"gpt_neox.layers.{i}.attention.dense.weight" for i in range(NUM_LAYERS)}
        expected_targets = qkv_w | qkv_b | dense_w

        missing = sorted(expected_targets - set(keys))
        if missing:
            fail(f"input is missing {len(missing)} head-bearing tensor(s), first: {missing[0]}")

        for key in keys:
            t = f.get_tensor(key)
            src_dtypes[key] = t.dtype

            if key in qkv_w:
                if tuple(t.shape) != (NUM_HEADS * QKV_PER_HEAD, HIDDEN):
                    fail(f"{key}: expected [{NUM_HEADS * QKV_PER_HEAD}, {HIDDEN}], got {list(t.shape)}")
                t = t.index_select(0, qkv_keep).contiguous()
                touched.add(key)
            elif key in qkv_b:
                if tuple(t.shape) != (NUM_HEADS * QKV_PER_HEAD,):
                    fail(f"{key}: expected [{NUM_HEADS * QKV_PER_HEAD}], got {list(t.shape)}")
                t = t.index_select(0, qkv_keep).contiguous()
                touched.add(key)
            elif key in dense_w:
                if tuple(t.shape) != (HIDDEN, NUM_HEADS * HEAD_DIM):
                    fail(f"{key}: expected [{HIDDEN}, {NUM_HEADS * HEAD_DIM}], got {list(t.shape)}")
                # [out, in]: heads are column blocks on the input axis.
                t = t.index_select(1, dense_keep).contiguous()
                touched.add(key)

            out[key] = t

    if touched != expected_targets:
        fail(f"edited {len(touched)} tensors, expected {len(expected_targets)}")

    # ---- Required checks -------------------------------------------------
    n_keep_heads = NUM_HEADS - 1
    required = {
        "gpt_neox.layers.0.attention.query_key_value.weight": (n_keep_heads * QKV_PER_HEAD, HIDDEN),
        "gpt_neox.layers.0.attention.query_key_value.bias": (n_keep_heads * QKV_PER_HEAD,),
        "gpt_neox.layers.0.attention.dense.weight": (HIDDEN, n_keep_heads * HEAD_DIM),
    }
    for key, want in required.items():
        if key not in out:
            fail(f"required tensor missing from output: {key}")
        if tuple(out[key].shape) != want:
            fail(f"{key}: expected shape {list(want)}, got {list(out[key].shape)}")

    if len(out) != EXPECTED_TENSORS:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TENSORS}")

    # Same checks across every remaining layer, not just layer 0.
    for i in range(NUM_LAYERS):
        p = f"gpt_neox.layers.{i}.attention"
        for key, want in (
            (f"{p}.query_key_value.weight", (n_keep_heads * QKV_PER_HEAD, HIDDEN)),
            (f"{p}.query_key_value.bias", (n_keep_heads * QKV_PER_HEAD,)),
            (f"{p}.dense.weight", (HIDDEN, n_keep_heads * HEAD_DIM)),
        ):
            if tuple(out[key].shape) != want:
                fail(f"{key}: expected shape {list(want)}, got {list(out[key].shape)}")
        if tuple(out[f"{p}.dense.bias"].shape) != (HIDDEN,):
            fail(f"{p}.dense.bias must stay [{HIDDEN}]")

    for key, t in out.items():
        if t.dtype != src_dtypes[key]:
            fail(f"{key}: dtype changed from {src_dtypes[key]} to {t.dtype}")
    for key in sorted(expected_targets):
        if out[key].dtype != torch.float16:
            fail(f"{key}: expected float16 projection, got {out[key].dtype}")

    DST_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST), metadata=metadata)

    with safe_open(DST, framework="pt", device="cpu") as f:
        n = len(list(f.keys()))
    if n != EXPECTED_TENSORS:
        fail(f"written file has {n} tensors, expected {EXPECTED_TENSORS}")

    print(f"wrote {DST} ({n} tensors); pruned head {PRUNE_HEAD} from {NUM_LAYERS} layers")
    print(f"  metadata carried over: {metadata!r}")
    for key in required:
        print(f"  {key}: {list(out[key].shape)}")


if __name__ == "__main__":
    sys.exit(main())
