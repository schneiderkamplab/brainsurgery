"""T2: structured attention-head pruning of Pythia-1B (remove head 5 in every layer).

Condition F. Tools: safetensors (0.5.3) for checkpoint I/O, torch (2.14.0) for
tensor slicing. transformers has no prune_heads implementation for GPT-NeoX
(modeling_gpt_neox.py contains no prune code at all), and mergekit /
torch-state-bridge operate on whole tensors and key names, not on slices inside
a tensor, so none of them can do this. Slicing the stored fp16 rows/columns and
re-saving keeps values bit-exact -- no model instantiation, no dtype round-trip.

Layout (GPT-NeoX, nn.Linear [out, in]):
  attention.query_key_value.weight [6144, 2048] / .bias [6144]
      rows are grouped per head: head h owns rows 768*h .. 768*h+767,
      inside that block: 256 q rows, then 256 k rows, then 256 v rows.
  attention.dense.weight [2048, 2048]
      columns are grouped per head: head h owns columns 256*h .. 256*h+255.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T2/model.safetensors")

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
HIDDEN = 2048
PRUNE_HEAD = 5

QKV_PER_HEAD = 3 * HEAD_DIM  # 768 rows of the fused projection belong to one head
QKV_ROWS = NUM_HEADS * QKV_PER_HEAD  # 6144
KEPT_HEADS = NUM_HEADS - 1  # 7

EXPECT_QKV_W = (KEPT_HEADS * QKV_PER_HEAD, HIDDEN)  # [5376, 2048]
EXPECT_QKV_B = (KEPT_HEADS * QKV_PER_HEAD,)  # [5376]
EXPECT_DENSE_W = (HIDDEN, KEPT_HEADS * HEAD_DIM)  # [2048, 1792]
EXPECT_TENSOR_COUNT = 244


class CheckFailed(AssertionError):
    """A required check did not hold; nothing is written."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def keep_index(block: int, count: int, drop: int) -> torch.Tensor:
    """Indices of `count` blocks of size `block`, with block `drop` removed."""
    lo = torch.arange(0, drop * block, dtype=torch.long)
    hi = torch.arange((drop + 1) * block, count * block, dtype=torch.long)
    return torch.cat([lo, hi])


def main() -> int:
    require(SRC.is_file(), f"input checkpoint not found: {SRC}")

    qkv_keep = keep_index(QKV_PER_HEAD, NUM_HEADS, PRUNE_HEAD)  # rows 0..3839, 4608..6143
    dense_keep = keep_index(HEAD_DIM, NUM_HEADS, PRUNE_HEAD)  # cols 0..1279, 1536..2047

    # Sanity-check the index construction itself against the literal spec.
    require(
        qkv_keep.tolist() == list(range(0, 3840)) + list(range(4608, 6144)),
        "qkv keep-index is not rows 0..3839 followed by 4608..6143",
    )
    require(
        dense_keep.tolist() == list(range(0, 1280)) + list(range(1536, 2048)),
        "dense keep-index is not columns 0..1279 followed by 1536..2047",
    )

    edits = {}
    for i in range(NUM_LAYERS):
        base = f"gpt_neox.layers.{i}.attention."
        edits[base + "query_key_value.weight"] = (qkv_keep, 0, EXPECT_QKV_W)
        edits[base + "query_key_value.bias"] = (qkv_keep, 0, EXPECT_QKV_B)
        edits[base + "dense.weight"] = (dense_keep, 1, EXPECT_DENSE_W)

    out: dict[str, torch.Tensor] = {}
    touched: set[str] = set()
    with safe_open(str(SRC), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        keys = list(f.keys())
        require(
            len(keys) == EXPECT_TENSOR_COUNT,
            f"input has {len(keys)} tensors, expected {EXPECT_TENSOR_COUNT}",
        )
        missing = sorted(set(edits) - set(keys))
        require(not missing, f"head-bearing tensors missing from input: {missing[:3]}")

        for key in keys:
            tensor = f.get_tensor(key)
            if key in edits:
                index, dim, expected = edits[key]
                require(
                    tensor.shape[dim] == QKV_ROWS if dim == 0 else tensor.shape[dim] == HIDDEN,
                    f"{key}: unexpected input shape {tuple(tensor.shape)}",
                )
                tensor = tensor.index_select(dim, index).contiguous()
                require(
                    tuple(tensor.shape) == expected,
                    f"{key}: got shape {tuple(tensor.shape)}, expected {list(expected)}",
                )
                touched.add(key)
            out[key] = tensor

    # --- required checks, before writing -------------------------------------
    require(
        touched == set(edits),
        f"edited {len(touched)} head-bearing tensors, expected {len(edits)}",
    )
    require(
        tuple(out["gpt_neox.layers.0.attention.query_key_value.weight"].shape) == EXPECT_QKV_W,
        "layer 0 query_key_value.weight is not [5376, 2048]",
    )
    require(
        tuple(out["gpt_neox.layers.0.attention.query_key_value.bias"].shape) == EXPECT_QKV_B,
        "layer 0 query_key_value.bias is not [5376]",
    )
    require(
        tuple(out["gpt_neox.layers.0.attention.dense.weight"].shape) == EXPECT_DENSE_W,
        "layer 0 attention.dense.weight is not [2048, 1792]",
    )
    require(
        len(out) == EXPECT_TENSOR_COUNT,
        f"output has {len(out)} tensors, expected {EXPECT_TENSOR_COUNT}",
    )
    # Same check on every layer, not only layer 0.
    for i in range(NUM_LAYERS):
        base = f"gpt_neox.layers.{i}.attention."
        for key, expected in (
            (base + "query_key_value.weight", EXPECT_QKV_W),
            (base + "query_key_value.bias", EXPECT_QKV_B),
            (base + "dense.weight", EXPECT_DENSE_W),
        ):
            require(
                tuple(out[key].shape) == expected,
                f"{key}: got shape {tuple(out[key].shape)}, expected {list(expected)}",
            )
    require(
        all(t.dtype == torch.float16 for k, t in out.items() if not k.endswith("attention.bias")),
        "a non-buffer tensor is not float16",
    )

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST), metadata=metadata or None)
    print(f"wrote {DST} ({len(out)} tensors), pruned head {PRUNE_HEAD} in {NUM_LAYERS} layers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
