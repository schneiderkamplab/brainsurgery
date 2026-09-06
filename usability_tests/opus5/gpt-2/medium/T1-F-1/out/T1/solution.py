#!/usr/bin/env python
"""T1: depth-prune GPT-2 (124M) from 12 to 9 blocks, renumbering the survivors.

Approach: safetensors for I/O, torch-state-bridge for the block renaming
(explicit per-key rules plus its collision detection), plus explicit checks
that must all pass before anything is written.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file
from torch_state_bridge import state_bridge

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"

DROP = (2, 5, 8)
N_BLOCKS_IN = 12
N_BLOCKS_OUT = 9
N_TENSORS_OUT = 121

BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def block_index(key: str) -> int | None:
    m = BLOCK_RE.match(key)
    return int(m.group(1)) if m else None


def main() -> None:
    with safe_open(SRC, framework="pt") as f:
        src = {k: f.get_tensor(k) for k in f.keys()}

    # --- input sanity -----------------------------------------------------
    src_blocks = sorted({i for i in map(block_index, src) if i is not None})
    if src_blocks != list(range(N_BLOCKS_IN)):
        fail(f"input blocks are {src_blocks}, expected 0..{N_BLOCKS_IN - 1}")
    if not set(DROP) <= set(src_blocks):
        fail(f"blocks to drop {DROP} not all present in input")

    # --- 1. drop blocks 2, 5, 8 ------------------------------------------
    kept = {k: v for k, v in src.items() if block_index(k) not in DROP}
    dropped = len(src) - len(kept)
    if dropped != len(DROP) * 13:
        fail(f"dropped {dropped} tensors, expected {len(DROP) * 13}")

    # --- 2. renumber survivors, in original order, to 0..8 ---------------
    survivors = [i for i in src_blocks if i not in DROP]
    mapping = {old: new for new, old in enumerate(survivors)}
    if mapping != {0: 0, 1: 1, 3: 2, 4: 3, 6: 4, 7: 5, 9: 6, 10: 7, 11: 8}:
        fail(f"renumbering map is {mapping}, not the one the task specifies")

    # One explicit literal rule per surviving key: no wildcard can reach a key
    # it was not meant to, and state_bridge raises on any collision. It applies
    # every rule in sequence to every key, so a destination that looks like
    # another rule's source gets rewritten twice (h.10 -> h.7 -> h.5). Renaming
    # through a temporary namespace that no rule can match avoids that.
    to_tmp, from_tmp = [], []
    for key in kept:
        i = block_index(key)
        if i is None:
            continue
        rest = BLOCK_RE.match(key).group(2)
        tmp = f"__NEW__{mapping[i]}__{rest}"
        to_tmp.append(f"h.{i}.{rest}, {tmp}")
        from_tmp.append(f"{tmp}, h.{mapping[i]}.{rest}")
    staged = state_bridge(kept, "\n".join(to_tmp), detect_collision=True)
    out = state_bridge(staged, "\n".join(from_tmp), detect_collision=True)

    # --- required checks, all before writing ------------------------------
    stale = sorted(k for k in out if block_index(k) in (9, 10, 11))
    if stale:
        fail(f"tensors of blocks 9/10/11 remain: {stale[:5]}")

    c_attn = sorted(k for k in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", k))
    if len(c_attn) != N_BLOCKS_OUT:
        fail(f"{len(c_attn)} blocks remain (by attn.c_attn.weight), expected {N_BLOCKS_OUT}")

    out_blocks = sorted({i for i in map(block_index, out) if i is not None})
    if out_blocks != list(range(N_BLOCKS_OUT)):
        fail(f"output block indices are {out_blocks}, expected contiguous 0..{N_BLOCKS_OUT - 1}")

    if len(out) != N_TENSORS_OUT:
        fail(f"output has {len(out)} tensors, expected {N_TENSORS_OUT}")

    # values/shapes/dtypes preserved, and each new key holds the right block
    inv = {new: old for old, new in mapping.items()}
    for key, tensor in out.items():
        i = block_index(key)
        if i is None:
            origin = key  # non-block tensor, must be untouched
        else:
            origin = f"h.{inv[i]}.{BLOCK_RE.match(key).group(2)}"
        if origin not in src:
            fail(f"output key {key} has no source key {origin}")
        ref = src[origin]
        if tensor.shape != ref.shape or tensor.dtype != ref.dtype:
            fail(f"{key}: shape/dtype changed ({tensor.shape}/{tensor.dtype} vs {ref.shape}/{ref.dtype})")
        if not tensor.equal(ref):
            fail(f"{key}: values differ from source {origin}")

    non_block = {k for k in out if block_index(k) is None}
    if non_block != {"wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"}:
        fail(f"non-block tensors are {sorted(non_block)}")

    # --- write only after every check passed ------------------------------
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(DST), suffix=".tmp")
    os.close(fd)
    try:
        save_file({k: v.contiguous() for k, v in out.items()}, tmp)
        os.replace(tmp, DST)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    print(f"wrote {DST}: {len(out)} tensors, {N_BLOCKS_OUT} blocks 0..{N_BLOCKS_OUT - 1}")


if __name__ == "__main__":
    main()
