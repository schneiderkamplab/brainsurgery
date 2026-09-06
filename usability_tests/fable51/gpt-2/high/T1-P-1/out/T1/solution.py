"""T1: depth-prune GPT-2 (124M) from 12 to 9 blocks, renumbering contiguously.

Removes blocks 2, 5, 8 and renumbers the survivors 0..8 in original order.
Builds a fresh dict (no in-place renames), so no collision is possible, and
runs every required check before anything is written to disk.
"""

import os
import re
import sys

from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "inputs", "base", "model.safetensors")
DST = os.path.join(HERE, "model.safetensors")

REMOVE = {2, 5, 8}
N_OLD = 12
N_NEW = N_OLD - len(REMOVE)  # 9
TENSORS_PER_BLOCK = 13
N_NONBLOCK = 4
EXPECTED_TOTAL = N_NEW * TENSORS_PER_BLOCK + N_NONBLOCK  # 121

BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    src = load_file(SRC)
    if len(src) != 160:
        fail(f"expected 160 input tensors, got {len(src)}")

    # old index -> new index, survivors in original order
    survivors = [i for i in range(N_OLD) if i not in REMOVE]
    remap = {old: new for new, old in enumerate(survivors)}
    if remap != {0: 0, 1: 1, 3: 2, 4: 3, 6: 4, 7: 5, 9: 6, 10: 7, 11: 8}:
        fail(f"unexpected remap {remap}")

    out = {}
    removed = 0
    for name, t in src.items():
        m = BLOCK_RE.match(name)
        if m is None:
            out[name] = t  # non-block tensor, unchanged
            continue
        old = int(m.group(1))
        if old >= N_OLD:
            fail(f"block index {old} out of range in {name}")
        if old in REMOVE:
            removed += 1
            continue
        new_name = f"h.{remap[old]}.{m.group(2)}"
        if new_name in out:
            fail(f"collision: {new_name} already present (from {name})")
        out[new_name] = t

    # ---- required checks (before writing) ----
    if removed != len(REMOVE) * TENSORS_PER_BLOCK:
        fail(f"removed {removed} tensors, expected {len(REMOVE) * TENSORS_PER_BLOCK}")

    block_ids = set()
    for name in out:
        m = BLOCK_RE.match(name)
        if m:
            block_ids.add(int(m.group(1)))
    stale = {i for i in block_ids if i >= N_NEW}
    if stale:
        fail(f"tensors of blocks {sorted(stale)} remain (only 0..{N_NEW - 1} allowed)")
    if block_ids != set(range(N_NEW)):
        fail(f"block indices are {sorted(block_ids)}, expected 0..{N_NEW - 1}")

    n_attn = sum(1 for n in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", n))
    if n_attn != N_NEW:
        fail(f"{n_attn} c_attn.weight tensors, expected {N_NEW}")
    for i in range(N_NEW):
        n_i = sum(1 for n in out if n.startswith(f"h.{i}."))
        if n_i != TENSORS_PER_BLOCK:
            fail(f"block {i} has {n_i} tensors, expected {TENSORS_PER_BLOCK}")

    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    # value/shape/dtype identity: each surviving tensor is the same object as the source
    for new_name, t in out.items():
        m = BLOCK_RE.match(new_name)
        if m:
            old = survivors[int(m.group(1))]
            src_name = f"h.{old}.{m.group(2)}"
        else:
            src_name = new_name
        s = src[src_name]
        if s.shape != t.shape or s.dtype != t.dtype or not s.equal(t):
            fail(f"{new_name} differs from source {src_name}")

    save_file({k: v.contiguous() for k, v in out.items()}, DST)

    # verify written file
    back = load_file(DST)
    if len(back) != EXPECTED_TOTAL or set(back) != set(out):
        fail("written file does not match the planned key set")
    print(f"OK: wrote {DST} with {len(back)} tensors, blocks 0..{N_NEW - 1}")


if __name__ == "__main__":
    main()
