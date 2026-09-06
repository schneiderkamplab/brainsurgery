"""T2: remove attention head 5 from every layer of GPT-2 (124M)."""
import os
import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "inputs", "base", "model.safetensors")
DST = os.path.join(HERE, "model.safetensors")

N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN = 12, 12, 64, 768
PRUNE_HEAD = 5


def keep_index(n_heads=N_HEADS, head_dim=HEAD_DIM, prune=PRUNE_HEAD):
    """Indices along one 768-wide head axis with head `prune` removed."""
    return torch.tensor(
        [j for h in range(n_heads) if h != prune for j in range(h * head_dim, (h + 1) * head_dim)]
    )


def main():
    sd = load_file(SRC)
    assert len(sd) == 160, f"expected 160 input tensors, got {len(sd)}"

    keep = keep_index()  # 704 indices within one q/k/v segment
    # Fused [q | k | v]: apply the same keep pattern to each 768-wide segment.
    keep_qkv = torch.cat([keep + s * HIDDEN for s in range(3)])
    assert keep_qkv.numel() == 3 * (HIDDEN - HEAD_DIM)

    for i in range(N_LAYERS):
        w = f"h.{i}.attn.c_attn.weight"
        b = f"h.{i}.attn.c_attn.bias"
        p = f"h.{i}.attn.c_proj.weight"
        assert tuple(sd[w].shape) == (HIDDEN, 3 * HIDDEN), (w, sd[w].shape)
        assert tuple(sd[b].shape) == (3 * HIDDEN,), (b, sd[b].shape)
        assert tuple(sd[p].shape) == (HIDDEN, HIDDEN), (p, sd[p].shape)
        sd[w] = sd[w].index_select(1, keep_qkv).contiguous()
        sd[b] = sd[b].index_select(0, keep_qkv).contiguous()
        sd[p] = sd[p].index_select(0, keep).contiguous()

    # Required checks (fail loudly before writing).
    assert tuple(sd["h.0.attn.c_attn.weight"].shape) == (768, 2112), sd["h.0.attn.c_attn.weight"].shape
    assert tuple(sd["h.0.attn.c_attn.bias"].shape) == (2112,), sd["h.0.attn.c_attn.bias"].shape
    assert tuple(sd["h.0.attn.c_proj.weight"].shape) == (704, 768), sd["h.0.attn.c_proj.weight"].shape
    assert len(sd) == 160, f"expected 160 output tensors, got {len(sd)}"

    # Extra sanity on every layer and on untouched tensors.
    for i in range(N_LAYERS):
        assert tuple(sd[f"h.{i}.attn.c_attn.weight"].shape) == (768, 2112)
        assert tuple(sd[f"h.{i}.attn.c_attn.bias"].shape) == (2112,)
        assert tuple(sd[f"h.{i}.attn.c_proj.weight"].shape) == (704, 768)
        assert tuple(sd[f"h.{i}.attn.c_proj.bias"].shape) == (768,)
        assert tuple(sd[f"h.{i}.attn.bias"].shape) == (1, 1, 1024, 1024)
        assert all(t.dtype == torch.float32 for t in sd.values())

    save_file(sd, DST)
    back = load_file(DST)
    assert len(back) == 160, f"wrote {len(back)} tensors"
    assert tuple(back["h.0.attn.c_attn.weight"].shape) == (768, 2112)
    print(f"wrote {DST} with {len(back)} tensors")


if __name__ == "__main__":
    main()
