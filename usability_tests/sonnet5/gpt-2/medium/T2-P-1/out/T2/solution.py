"""Prune attention head 5 from every layer of GPT-2 (124M) at the checkpoint level.

For each layer i in 0..11:
  - h.<i>.attn.c_attn.weight [768, 2304] (fused q|k|v, heads as column blocks
    within each 768-wide segment) -> drop head 5's 64-wide column slice from
    each of the three segments -> [768, 2112].
  - h.<i>.attn.c_attn.bias [2304] -> same column layout -> [2112].
  - h.<i>.attn.c_proj.weight [768, 768] (heads as row blocks) -> drop head 5's
    64-wide row slice -> [704, 768].
  - Everything else is copied unchanged.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
INPUT_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUTPUT_PATH = HERE / "model.safetensors"

NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64
HIDDEN = NUM_HEADS * HEAD_DIM  # 768
PRUNE_HEAD = 5


def keep_indices_for_segment(seg_start: int) -> torch.Tensor:
    """Column/row indices to keep within one 768-wide segment (q, k, or v),
    offset by seg_start, dropping the PRUNE_HEAD-th 64-wide head block."""
    parts = []
    for h in range(NUM_HEADS):
        if h == PRUNE_HEAD:
            continue
        start = seg_start + h * HEAD_DIM
        parts.append(torch.arange(start, start + HEAD_DIM))
    return torch.cat(parts)


def main() -> None:
    state_dict = load_file(str(INPUT_PATH))
    assert len(state_dict) == 160, f"expected 160 input tensors, got {len(state_dict)}"

    # Indices to keep across the fused [q | k | v] axis (2304 -> 2112).
    qkv_keep = torch.cat(
        [keep_indices_for_segment(seg * HIDDEN) for seg in range(3)]
    )
    assert qkv_keep.shape == (2112,), qkv_keep.shape

    # Indices to keep along the single 768-wide head axis (768 -> 704).
    proj_keep = keep_indices_for_segment(0)
    assert proj_keep.shape == (704,), proj_keep.shape

    output: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if name.endswith(".attn.c_attn.weight"):
            assert tensor.shape == (HIDDEN, 3 * HIDDEN), (name, tensor.shape)
            output[name] = tensor[:, qkv_keep].contiguous()
        elif name.endswith(".attn.c_attn.bias"):
            assert tensor.shape == (3 * HIDDEN,), (name, tensor.shape)
            output[name] = tensor[qkv_keep].contiguous()
        elif name.endswith(".attn.c_proj.weight"):
            assert tensor.shape == (HIDDEN, HIDDEN), (name, tensor.shape)
            output[name] = tensor[proj_keep, :].contiguous()
        else:
            output[name] = tensor

    # Required checks.
    for i in range(NUM_LAYERS):
        w = output[f"h.{i}.attn.c_attn.weight"]
        b = output[f"h.{i}.attn.c_attn.bias"]
        p = output[f"h.{i}.attn.c_proj.weight"]
        assert w.shape == (768, 2112), f"h.{i}.attn.c_attn.weight shape {tuple(w.shape)}"
        assert b.shape == (2112,), f"h.{i}.attn.c_attn.bias shape {tuple(b.shape)}"
        assert p.shape == (704, 768), f"h.{i}.attn.c_proj.weight shape {tuple(p.shape)}"

    assert len(output) == 160, f"expected 160 output tensors, got {len(output)}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, str(OUTPUT_PATH))
    print(f"Wrote {len(output)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
