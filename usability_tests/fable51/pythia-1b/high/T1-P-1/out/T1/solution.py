"""T1: depth-prune Pythia-1B from 16 to 12 blocks, renumbering the survivors.

Removes blocks 2, 6, 10, 14 and shifts the remaining blocks down so that the
layer indices run 0..11. Fails loudly (non-zero exit, no output) if any check
in TASK.md does not hold.
"""

import os
import re
import sys

from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "inputs", "base", "model.safetensors")
DST = os.path.join(HERE, "model.safetensors")

N_OLD = 16
DROP = {2, 6, 10, 14}
N_NEW = N_OLD - len(DROP)  # 12
TENSORS_PER_BLOCK = 15
N_NONBLOCK = 4
EXPECTED_TOTAL = N_NEW * TENSORS_PER_BLOCK + N_NONBLOCK  # 184

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    # old index -> new index, surviving blocks in original order
    keep = [i for i in range(N_OLD) if i not in DROP]
    remap = {old: new for new, old in enumerate(keep)}
    assert list(remap.values()) == list(range(N_NEW))

    out: dict = {}
    n_in = 0
    dropped = 0
    with safe_open(SRC, framework="pt") as f:
        metadata = f.metadata()
        for name in f.keys():
            n_in += 1
            m = LAYER_RE.match(name)
            if m is None:
                # non-block tensor: pass through unchanged
                new_name = name
            else:
                old = int(m.group(1))
                if old >= N_OLD:
                    fail(f"unexpected block index {old} in {name}")
                if old in DROP:
                    dropped += 1
                    continue
                new_name = f"gpt_neox.layers.{remap[old]}.{m.group(2)}"
            if new_name in out:
                fail(f"collision: {new_name} produced twice (from {name})")
            out[new_name] = f.get_tensor(name)

    # ---- required checks ----
    if n_in != N_OLD * TENSORS_PER_BLOCK + N_NONBLOCK:
        fail(f"input has {n_in} tensors, expected {N_OLD * TENSORS_PER_BLOCK + N_NONBLOCK}")
    if dropped != len(DROP) * TENSORS_PER_BLOCK:
        fail(f"dropped {dropped} tensors, expected {len(DROP) * TENSORS_PER_BLOCK}")

    surviving_idx = set()
    for name in out:
        m = LAYER_RE.match(name)
        if m:
            surviving_idx.add(int(m.group(1)))
    stale = {i for i in surviving_idx if i >= N_NEW}
    if stale:
        fail(f"tensors of blocks {sorted(stale)} remain after renumbering")
    if surviving_idx != set(range(N_NEW)):
        fail(f"surviving block indices {sorted(surviving_idx)} != 0..{N_NEW - 1}")

    qkv = [n for n in out if re.fullmatch(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", n)]
    if len(qkv) != N_NEW:
        fail(f"{len(qkv)} query_key_value.weight tensors, expected {N_NEW}")
    for i in range(N_NEW):
        cnt = sum(1 for n in out if n.startswith(f"gpt_neox.layers.{i}."))
        if cnt != TENSORS_PER_BLOCK:
            fail(f"block {i} has {cnt} tensors, expected {TENSORS_PER_BLOCK}")

    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    # ---- write (only after all checks passed) ----
    os.makedirs(HERE, exist_ok=True)
    tmp = DST + ".tmp"
    save_file({k: v.contiguous() for k, v in out.items()}, tmp, metadata=metadata)
    os.replace(tmp, DST)

    # ---- post-write verification ----
    with safe_open(DST, framework="pt") as g:
        keys = list(g.keys())
        if len(keys) != EXPECTED_TOTAL:
            os.remove(DST)
            fail(f"written file has {len(keys)} tensors, expected {EXPECTED_TOTAL}")
        for k in keys:
            t = g.get_tensor(k)
            if t.shape != out[k].shape or t.dtype != out[k].dtype:
                os.remove(DST)
                fail(f"shape/dtype mismatch after write for {k}")
    print(f"OK: wrote {DST} with {len(keys)} tensors ({N_NEW} blocks)")


if __name__ == "__main__":
    main()
