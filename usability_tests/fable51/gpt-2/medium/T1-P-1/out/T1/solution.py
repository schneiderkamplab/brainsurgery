"""T1: depth-prune GPT-2 (124M) from 12 to 9 blocks, dropping blocks 2, 5, 8.

Loads inputs/base/model.safetensors, removes every tensor of the dropped
blocks, renumbers the survivors contiguously (in original order, ascending, so
no rename can collide with a not-yet-moved block), verifies the result and
writes out/T1/model.safetensors. Any failed check exits non-zero before the
output is written.
"""

import os
import re
import sys

from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "inputs", "base", "model.safetensors")
DST = os.path.join(HERE, "model.safetensors")

DROP = {2, 5, 8}
N_IN = 12
N_OUT = N_IN - len(DROP)
TENSORS_PER_BLOCK = 13
EXPECTED_OUT = N_OUT * TENSORS_PER_BLOCK + 4  # 121

BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if os.path.exists(DST):
        fail(f"destination already exists: {DST}")

    src = load_file(SRC)
    if len(src) != N_IN * TENSORS_PER_BLOCK + 4:
        fail(f"unexpected input tensor count {len(src)}")

    surviving = [i for i in range(N_IN) if i not in DROP]
    old_to_new = {old: new for new, old in enumerate(surviving)}

    out = {}
    per_block_in = {i: 0 for i in range(N_IN)}
    for name, tensor in src.items():
        m = BLOCK_RE.match(name)
        if m is None:
            out[name] = tensor  # wte/wpe/ln_f: unchanged
            continue
        old = int(m.group(1))
        if old not in per_block_in:
            fail(f"unexpected block index in input: {name}")
        per_block_in[old] += 1
        if old in DROP:
            continue
        new_name = f"h.{old_to_new[old]}.{m.group(2)}"
        if new_name in out:
            fail(f"rename collision: {name} -> {new_name}")
        out[new_name] = tensor

    for i, n in per_block_in.items():
        if n != TENSORS_PER_BLOCK:
            fail(f"input block {i} has {n} tensors, expected {TENSORS_PER_BLOCK}")

    # --- Required checks ---
    out_blocks = {}
    for name in out:
        m = BLOCK_RE.match(name)
        if m:
            out_blocks.setdefault(int(m.group(1)), []).append(name)

    stale = [i for i in out_blocks if i >= N_OUT]
    if stale:
        fail(f"tensors of blocks >= {N_OUT} remain: {sorted(stale)}")

    if sorted(out_blocks) != list(range(N_OUT)):
        fail(f"expected blocks 0..{N_OUT - 1}, got {sorted(out_blocks)}")
    n_attn = sum(1 for n in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", n))
    if n_attn != N_OUT:
        fail(f"expected {N_OUT} c_attn.weight tensors, found {n_attn}")
    for i, names in out_blocks.items():
        if len(names) != TENSORS_PER_BLOCK:
            fail(f"output block {i} has {len(names)} tensors")

    if len(out) != EXPECTED_OUT:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_OUT}")

    # Values/shapes/dtypes identical to the source block they came from.
    for old, new in old_to_new.items():
        for name in src:
            m = BLOCK_RE.match(name)
            if m and int(m.group(1)) == old:
                a, b = src[name], out[f"h.{new}.{m.group(2)}"]
                if a.shape != b.shape or a.dtype != b.dtype or not (a == b).all():
                    fail(f"mismatch after rename: {name}")
    for name in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"):
        if name not in out or not (out[name] == src[name]).all():
            fail(f"non-block tensor changed or missing: {name}")

    out = {k: v.contiguous() for k, v in out.items()}
    save_file(out, DST, metadata={"format": "pt"})

    # Re-read to verify the file on disk.
    back = load_file(DST)
    if len(back) != EXPECTED_OUT or set(back) != set(out):
        os.remove(DST)
        fail("written file does not match expected key set")
    print(f"OK: wrote {DST} with {len(back)} tensors, blocks {sorted(out_blocks)}")


if __name__ == "__main__":
    main()
