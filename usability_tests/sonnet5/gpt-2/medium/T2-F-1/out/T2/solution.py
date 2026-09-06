"""T2: structured attention-head pruning for GPT-2 (124M), condition F.

Removes head 5 (0-indexed) from every layer's attention block by slicing the
head-bearing tensors directly with safetensors + torch. No model class is
instantiated: `transformers.prune_heads` operates on nn.Linear-shaped weights
and GPT-2's Conv1D layout (`[in, out]`, the transpose of nn.Linear) does not
match what it expects, so a plain slice-and-concatenate script is the more
direct and auditable route here.

Layout recap (hidden size 768, 12 heads of 64 dims each):
- attn.c_attn.weight: [768, 2304] = [768, q(768) | k(768) | v(768)], heads are
  64-wide column blocks within each of the three 768-wide segments.
- attn.c_attn.bias: [2304], same column layout.
- attn.c_proj.weight: [768, 768], heads are 64-wide row blocks (single segment).
- attn.c_proj.bias [768] and attn.bias [1,1,1024,1024] are not per-head and
  are copied unchanged.
"""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HEAD_DIM = 64
NUM_HEADS = 12
HIDDEN = 768
PRUNE_HEAD = 5
NUM_LAYERS = 12

IN_PATH = Path("inputs/base/model.safetensors")
OUT_PATH = Path("out/T2/model.safetensors")


def drop_head_columns(t: torch.Tensor, head: int, num_segments: int) -> torch.Tensor:
    """Remove `head`'s 64-wide column block from each of `num_segments` equal
    HIDDEN-wide segments along the last dim, then re-concatenate in order."""
    seg_width = HIDDEN
    lo = head * HEAD_DIM
    hi = lo + HEAD_DIM
    pieces = []
    for s in range(num_segments):
        base = s * seg_width
        pieces.append(t[..., base : base + lo])
        pieces.append(t[..., base + hi : base + seg_width])
    return torch.cat(pieces, dim=-1)


def drop_head_rows(t: torch.Tensor, head: int) -> torch.Tensor:
    lo = head * HEAD_DIM
    hi = lo + HEAD_DIM
    return torch.cat([t[:lo], t[hi:]], dim=0)


def main() -> None:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(IN_PATH, framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            tensors[k] = f.get_tensor(k)

    assert len(tensors) == 160, f"expected 160 input tensors, got {len(tensors)}"

    out: dict[str, torch.Tensor] = {}
    for name, t in tensors.items():
        if name.endswith("attn.c_attn.weight"):
            assert t.shape == (HIDDEN, 3 * HIDDEN), (name, t.shape)
            new_t = drop_head_columns(t, PRUNE_HEAD, num_segments=3)
        elif name.endswith("attn.c_attn.bias"):
            assert t.shape == (3 * HIDDEN,), (name, t.shape)
            new_t = drop_head_columns(t, PRUNE_HEAD, num_segments=3)
        elif name.endswith("attn.c_proj.weight"):
            assert t.shape == (HIDDEN, HIDDEN), (name, t.shape)
            new_t = drop_head_rows(t, PRUNE_HEAD)
        else:
            new_t = t
        out[name] = new_t.contiguous()

    # Required checks: fail loudly before writing.
    assert out["h.0.attn.c_attn.weight"].shape == (768, 2112), out[
        "h.0.attn.c_attn.weight"
    ].shape
    assert out["h.0.attn.c_attn.bias"].shape == (2112,), out["h.0.attn.c_attn.bias"].shape
    assert out["h.0.attn.c_proj.weight"].shape == (704, 768), out[
        "h.0.attn.c_proj.weight"
    ].shape
    assert len(out) == 160, f"expected 160 output tensors, got {len(out)}"

    # Spot-check every layer, not just layer 0.
    for i in range(NUM_LAYERS):
        assert out[f"h.{i}.attn.c_attn.weight"].shape == (768, 2112)
        assert out[f"h.{i}.attn.c_attn.bias"].shape == (2112,)
        assert out[f"h.{i}.attn.c_proj.weight"].shape == (704, 768)
        assert out[f"h.{i}.attn.c_proj.bias"].shape == (768,)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, OUT_PATH)
    print(f"wrote {OUT_PATH} with {len(out)} tensors")


if __name__ == "__main__":
    main()
