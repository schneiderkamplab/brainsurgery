"""T2: prune head 5 from every layer of Pythia-1B, checkpoint-level.

Plain script on top of `safetensors` and `torch` (both in F-allowed.md).
Direct slicing was chosen over transformers `prune_heads` because that API
prunes by permanently masking/removing heads from a live nn.Module and does
not guarantee the exact interleaved-row-block / column-block boundaries and
ordering the task spec pins down; a explicit slice on the raw state dict is
easier to verify bit-for-bit against the spec.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

IN_PATH = Path("inputs/base/model.safetensors")
OUT_DIR = Path("out/T2")
OUT_PATH = OUT_DIR / "model.safetensors"

NUM_LAYERS = 16
HEAD_TO_PRUNE = 5
QKV_BLOCK = 768  # rows per head in fused query_key_value
DENSE_BLOCK = 256  # columns per head in dense.weight


def rows_to_keep(head: int, block: int, total_rows: int) -> torch.Tensor:
    start = head * block
    end = start + block
    idx = torch.cat(
        [torch.arange(0, start), torch.arange(end, total_rows)]
    )
    return idx


def main() -> None:
    state = load_file(str(IN_PATH))
    assert len(state) == 244, f"expected 244 input tensors, got {len(state)}"

    qkv_keep = rows_to_keep(HEAD_TO_PRUNE, QKV_BLOCK, 6144)
    dense_keep = rows_to_keep(HEAD_TO_PRUNE, DENSE_BLOCK, 2048)

    out = {}
    for name, tensor in state.items():
        if name.endswith("attention.query_key_value.weight"):
            new_t = tensor.index_select(0, qkv_keep).contiguous()
        elif name.endswith("attention.query_key_value.bias"):
            new_t = tensor.index_select(0, qkv_keep).contiguous()
        elif name.endswith("attention.dense.weight"):
            new_t = tensor.index_select(1, dense_keep).contiguous()
        else:
            new_t = tensor
        out[name] = new_t

    # Required checks: fail loudly before writing if anything is off.
    for i in range(NUM_LAYERS):
        w = out[f"gpt_neox.layers.{i}.attention.query_key_value.weight"]
        b = out[f"gpt_neox.layers.{i}.attention.query_key_value.bias"]
        d = out[f"gpt_neox.layers.{i}.attention.dense.weight"]
        assert tuple(w.shape) == (5376, 2048), (i, "qkv.weight", w.shape)
        assert tuple(b.shape) == (5376,), (i, "qkv.bias", b.shape)
        assert tuple(d.shape) == (2048, 1792), (i, "dense.weight", d.shape)

    assert len(out) == 244, f"expected 244 output tensors, got {len(out)}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(out)} tensors")


if __name__ == "__main__":
    main()
