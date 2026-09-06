"""T1: depth-prune GPT-2 (124M) from 12 to 9 transformer blocks.

Drops blocks 2, 5, 8 and renumbers the survivors so block indices are
contiguous 0..8, keeping every other part of each tensor name, and all
values, shapes and dtypes, unchanged.
"""

import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
DST = HERE / "model.safetensors"

DROP = {2, 5, 8}
N_BLOCKS_IN = 12
N_BLOCKS_OUT = 9
N_TENSORS_OUT = 121

BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def block_index(key: str) -> int | None:
    m = BLOCK_RE.match(key)
    return int(m.group(1)) if m else None


def main() -> None:
    src = load_file(str(SRC))

    # --- sanity on the input -------------------------------------------------
    in_blocks = sorted({i for i in map(block_index, src) if i is not None})
    if in_blocks != list(range(N_BLOCKS_IN)):
        fail(f"input block indices are {in_blocks}, expected 0..{N_BLOCKS_IN - 1}")
    if not DROP.issubset(set(in_blocks)):
        fail(f"blocks to drop {sorted(DROP)} not all present in input")

    # old index -> new index, survivors in original order, no gaps
    keep = [i for i in in_blocks if i not in DROP]
    if len(keep) != N_BLOCKS_OUT:
        fail(f"{len(keep)} blocks would survive, expected {N_BLOCKS_OUT}")
    remap = {old: new for new, old in enumerate(keep)}

    # --- build the new state dict -------------------------------------------
    # Writing into a fresh dict makes renumbering collision-free by
    # construction; a destination key must never already exist.
    out: dict[str, torch.Tensor] = {}
    for key, tensor in src.items():
        old = block_index(key)
        if old is None:
            new_key = key  # wte / wpe / ln_f, passed through untouched
        elif old in DROP:
            continue
        else:
            rest = BLOCK_RE.match(key).group(2)
            new_key = f"h.{remap[old]}.{rest}"
        if new_key in out:
            fail(f"destination key collision on {new_key!r} (from {key!r})")
        out[new_key] = tensor

    # --- required checks, all before anything is written --------------------
    out_blocks = sorted({i for i in map(block_index, out) if i is not None})

    stale = [i for i in out_blocks if i >= N_BLOCKS_OUT]
    if stale:
        fail(f"tensors of blocks {stale} still present; blocks 9, 10, 11 must be gone")

    if out_blocks != list(range(N_BLOCKS_OUT)):
        fail(f"output block indices are {out_blocks}, expected 0..{N_BLOCKS_OUT - 1}")

    n_attn = sum(1 for k in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", k))
    if n_attn != N_BLOCKS_OUT:
        fail(f"{n_attn} tensors match h.<i>.attn.c_attn.weight, expected {N_BLOCKS_OUT}")

    per_block: dict[int, int] = {}
    for k in out:
        i = block_index(k)
        if i is not None:
            per_block[i] = per_block.get(i, 0) + 1
    widths = set(per_block.values())
    if widths != {13}:
        fail(f"blocks have differing tensor counts {sorted(per_block.items())}, expected 13 each")

    if len(out) != N_TENSORS_OUT:
        fail(f"output has {len(out)} tensors, expected {N_TENSORS_OUT}")

    # every surviving tensor must be the exact object from the source
    for new_key, tensor in out.items():
        i = block_index(new_key)
        if i is None:
            old_key = new_key
        else:
            old_key = f"h.{keep[i]}.{BLOCK_RE.match(new_key).group(2)}"
        ref = src[old_key]
        if tensor.shape != ref.shape or tensor.dtype != ref.dtype:
            fail(f"{new_key!r} shape/dtype drifted from {old_key!r}")
        if not torch.equal(tensor, ref):
            fail(f"{new_key!r} values differ from {old_key!r}")

    # --- write ---------------------------------------------------------------
    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST), metadata={"format": "pt"})

    # --- verify what landed on disk -----------------------------------------
    back = load_file(str(DST))
    if len(back) != N_TENSORS_OUT:
        fail(f"written file has {len(back)} tensors, expected {N_TENSORS_OUT}")
    if set(back) != set(out):
        fail("written key set differs from the intended key set")
    for k, v in back.items():
        if v.dtype != out[k].dtype or v.shape != out[k].shape or not torch.equal(v, out[k]):
            fail(f"written tensor {k!r} does not round-trip")

    print(f"OK: {len(back)} tensors, blocks 0..{N_BLOCKS_OUT - 1}, dropped {sorted(DROP)}")
    print(f"    mapping: {', '.join(f'{o}->{n}' for o, n in remap.items())}")
    print(f"    wrote {DST}")


if __name__ == "__main__":
    main()
