"""
T2: Structured attention-head pruning (Pythia-1B).

Remove attention head 5 (0-indexed) from every layer of Pythia-1B.

Layout facts (from TASK.md):
- query_key_value.weight: [6144, 2048], rows grouped per head into 768-row
  blocks (head h -> rows 768*h .. 768*h+767), GPT-NeoX interleaved q/k/v
  inside each block (not used directly here since we drop whole head blocks).
- query_key_value.bias: [6144], same row layout.
- dense.weight: [2048, 2048], heads are 256-wide column blocks.
- dense.bias, MLP tensors, attention buffers: untouched.

Head 5 of 8:
- qkv rows to drop: 768*5 .. 768*5+767 = 3840..4607 -> keep 0..3839, 4608..6143
- dense cols to drop: 256*5 .. 256*5+255 = 1280..1535 -> keep 0..1279, 1536..2047
"""

import pathlib

import torch
from safetensors.torch import load_file, save_file

HERE = pathlib.Path(__file__).resolve().parent
IN_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUT_DIR = HERE
OUT_PATH = OUT_DIR / "model.safetensors"

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
QKV_BLOCK = 3 * HEAD_DIM  # 768
HIDDEN = NUM_HEADS * HEAD_DIM  # 2048
HEAD_TO_PRUNE = 5


def main() -> None:
    state = load_file(str(IN_PATH))
    assert len(state) == 244, f"expected 244 input tensors, got {len(state)}"

    qkv_row_lo = QKV_BLOCK * HEAD_TO_PRUNE
    qkv_row_hi = qkv_row_lo + QKV_BLOCK
    dense_col_lo = HEAD_DIM * HEAD_TO_PRUNE
    dense_col_hi = dense_col_lo + HEAD_DIM

    out = {}
    for name, tensor in state.items():
        if name.endswith("attention.query_key_value.weight"):
            assert tensor.shape == (6144, 2048), (name, tensor.shape)
            new_t = torch.cat([tensor[:qkv_row_lo], tensor[qkv_row_hi:]], dim=0)
            assert new_t.shape == (5376, 2048), (name, new_t.shape)
        elif name.endswith("attention.query_key_value.bias"):
            assert tensor.shape == (6144,), (name, tensor.shape)
            new_t = torch.cat([tensor[:qkv_row_lo], tensor[qkv_row_hi:]], dim=0)
            assert new_t.shape == (5376,), (name, new_t.shape)
        elif name.endswith("attention.dense.weight"):
            assert tensor.shape == (2048, 2048), (name, tensor.shape)
            new_t = torch.cat([tensor[:, :dense_col_lo], tensor[:, dense_col_hi:]], dim=1)
            assert new_t.shape == (2048, 1792), (name, new_t.shape)
        else:
            new_t = tensor
        out[name] = new_t.contiguous()

    assert len(out) == 244, f"expected 244 output tensors, got {len(out)}"

    for i in range(NUM_LAYERS):
        w = out[f"gpt_neox.layers.{i}.attention.query_key_value.weight"]
        b = out[f"gpt_neox.layers.{i}.attention.query_key_value.bias"]
        d = out[f"gpt_neox.layers.{i}.attention.dense.weight"]
        assert w.shape == (5376, 2048), (i, "qkv.weight", w.shape)
        assert b.shape == (5376,), (i, "qkv.bias", b.shape)
        assert d.shape == (2048, 1792), (i, "dense.weight", d.shape)

    # Required checks (explicit, from TASK.md)
    assert out["gpt_neox.layers.0.attention.query_key_value.weight"].shape == (5376, 2048)
    assert out["gpt_neox.layers.0.attention.query_key_value.bias"].shape == (5376,)
    assert out["gpt_neox.layers.0.attention.dense.weight"].shape == (2048, 1792)
    assert len(out) == 244

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"Wrote {OUT_PATH} with {len(out)} tensors")


if __name__ == "__main__":
    main()
