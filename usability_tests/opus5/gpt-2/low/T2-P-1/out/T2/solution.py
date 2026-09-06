"""T2: remove attention head 5 from every layer of GPT-2 (124M)."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
DST = HERE / "model.safetensors"

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM
PRUNE = 5


def keep_indices(offset: int) -> list[int]:
    """Column/row indices to keep within one 768-wide segment starting at offset."""
    lo = offset + PRUNE * HEAD_DIM
    hi = lo + HEAD_DIM
    return [i for i in range(offset, offset + HIDDEN) if not (lo <= i < hi)]


def main() -> None:
    sd = load_file(str(SRC))
    n_in = len(sd)

    # c_attn is the fused [q | k | v] projection: three 768-wide segments.
    attn_keep = torch.tensor(
        keep_indices(0) + keep_indices(HIDDEN) + keep_indices(2 * HIDDEN),
        dtype=torch.long,
    )
    proj_keep = torch.tensor(keep_indices(0), dtype=torch.long)

    for i in range(N_LAYERS):
        w = f"h.{i}.attn.c_attn.weight"
        b = f"h.{i}.attn.c_attn.bias"
        p = f"h.{i}.attn.c_proj.weight"
        for k, want in ((w, (HIDDEN, 3 * HIDDEN)), (b, (3 * HIDDEN,)), (p, (HIDDEN, HIDDEN))):
            if k not in sd:
                raise SystemExit(f"missing tensor {k}")
            if tuple(sd[k].shape) != want:
                raise SystemExit(f"{k}: expected shape {want}, got {tuple(sd[k].shape)}")
        sd[w] = sd[w].index_select(1, attn_keep).contiguous()
        sd[b] = sd[b].index_select(0, attn_keep).contiguous()
        sd[p] = sd[p].index_select(0, proj_keep).contiguous()

    # Required checks: fail loudly before writing.
    checks = {
        "h.0.attn.c_attn.weight": (768, 2112),
        "h.0.attn.c_attn.bias": (2112,),
        "h.0.attn.c_proj.weight": (704, 768),
    }
    for k, want in checks.items():
        got = tuple(sd[k].shape)
        if got != want:
            raise SystemExit(f"check failed: {k} has shape {got}, expected {want}")
    if len(sd) != 160:
        raise SystemExit(f"check failed: output has {len(sd)} tensors, expected 160")
    if n_in != 160:
        raise SystemExit(f"check failed: input had {n_in} tensors, expected 160")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(DST))
    print(f"wrote {DST} with {len(sd)} tensors")


if __name__ == "__main__":
    main()
