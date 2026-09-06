"""T2: structured attention-head pruning for Pythia-1B.

Removes head 5 of 8 from every layer, at the checkpoint level.

Layout facts this relies on (from TASK.md, verified by assertions below):
  - query_key_value.weight is [6144, 2048], nn.Linear [out, in]; the 6144 rows
    are GPT-NeoX interleaved per head: head h owns rows 768*h .. 768*h+767,
    holding its q (256), k (256) and v (256) in that order. So dropping a head
    is dropping one contiguous 768-row block.
  - query_key_value.bias is [6144] with the same row layout.
  - dense.weight is [2048, 2048] and consumes heads as 256-wide *column* blocks,
    so dropping head h drops columns 256*h .. 256*h+255.
"""

import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T2/model.safetensors")

NUM_HEADS = 8
HEAD_DIM = 256
HIDDEN = 2048
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head in the fused projection
PRUNE_HEAD = 5

QKV_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.(weight|bias)$")
DENSE_W_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.attention\.dense\.weight$")


def keep_index(total: int, block: int, drop: int) -> torch.Tensor:
    """Indices of `total` positions minus the `drop`-th block of size `block`."""
    assert total % block == 0, f"{total} not divisible by block {block}"
    lo, hi = drop * block, (drop + 1) * block
    return torch.cat([torch.arange(0, lo), torch.arange(hi, total)])


QKV_KEEP = keep_index(NUM_HEADS * QKV_BLOCK, QKV_BLOCK, PRUNE_HEAD)
DENSE_KEEP = keep_index(NUM_HEADS * HEAD_DIM, HEAD_DIM, PRUNE_HEAD)


def main() -> int:
    out: dict[str, torch.Tensor] = {}
    touched_qkv_w = touched_qkv_b = touched_dense = 0

    src_dtypes: dict[str, torch.dtype] = {}
    with safe_open(SRC, framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            src_dtypes[k] = t.dtype
            m = QKV_RE.match(k)
            if m:
                if m.group(2) == "weight":
                    assert t.shape == (NUM_HEADS * QKV_BLOCK, HIDDEN), (k, t.shape)
                    touched_qkv_w += 1
                else:
                    assert t.shape == (NUM_HEADS * QKV_BLOCK,), (k, t.shape)
                    touched_qkv_b += 1
                # heads live on dim 0 for both the weight rows and the bias
                out[k] = t.index_select(0, QKV_KEEP).contiguous()
                continue
            if DENSE_W_RE.match(k):
                assert t.shape == (HIDDEN, NUM_HEADS * HEAD_DIM), (k, t.shape)
                touched_dense += 1
                # heads live on dim 1 (input columns) of the output projection
                out[k] = t.index_select(1, DENSE_KEEP).contiguous()
                continue
            out[k] = t

    # --- Required checks: fail loudly before writing -------------------------
    assert touched_qkv_w == 16, f"expected 16 qkv weights, edited {touched_qkv_w}"
    assert touched_qkv_b == 16, f"expected 16 qkv biases, edited {touched_qkv_b}"
    assert touched_dense == 16, f"expected 16 dense weights, edited {touched_dense}"

    checks = {
        "gpt_neox.layers.0.attention.query_key_value.weight": (5376, 2048),
        "gpt_neox.layers.0.attention.query_key_value.bias": (5376,),
        "gpt_neox.layers.0.attention.dense.weight": (2048, 1792),
    }
    for name, want in checks.items():
        got = tuple(out[name].shape)
        assert got == want, f"{name}: shape {got}, expected {want}"

    assert len(out) == 244, f"output has {len(out)} tensors, expected 244"
    assert set(out) == set(keys), "tensor name set changed"

    # every layer, not just layer 0; and dtype preserved everywhere
    for i in range(16):
        p = f"gpt_neox.layers.{i}.attention."
        assert tuple(out[p + "query_key_value.weight"].shape) == (5376, 2048)
        assert tuple(out[p + "query_key_value.bias"].shape) == (5376,)
        assert tuple(out[p + "dense.weight"].shape) == (2048, 1792)
        assert tuple(out[p + "dense.bias"].shape) == (2048,), "dense.bias must be untouched"
    # dtypes must be preserved exactly (note: the attention mask buffers are
    # uint8/bool, not float16, so assume nothing -- compare against the source)
    for name, t in out.items():
        assert t.dtype == src_dtypes[name], f"{name}: dtype {t.dtype} != {src_dtypes[name]}"

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST))
    print(f"wrote {DST} with {len(out)} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
