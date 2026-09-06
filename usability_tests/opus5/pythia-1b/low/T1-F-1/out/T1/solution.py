#!/usr/bin/env python
"""T1: depth-prune Pythia-1B from 16 to 12 blocks, renumbering survivors."""
import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T1/model.safetensors")
DROP = {2, 6, 10, 14}
NUM_LAYERS = 16
BLOCK_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    keep = [i for i in range(NUM_LAYERS) if i not in DROP]
    remap = {old: new for new, old in enumerate(keep)}

    out = {}
    with safe_open(str(SRC), framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            m = BLOCK_RE.match(k)
            if m is None:
                out[k] = f.get_tensor(k)
                continue
            old = int(m.group(1))
            if old in DROP:
                continue
            new_key = f"gpt_neox.layers.{remap[old]}.{m.group(2)}"
            if new_key in out:
                fail(f"renumbering collision on {new_key}")
            out[new_key] = f.get_tensor(k)

    # Required checks, before anything is written.
    if len(keys) != 244:
        fail(f"input has {len(keys)} tensors, expected 244")
    for i in (12, 13, 14, 15):
        stale = [k for k in out if k.startswith(f"gpt_neox.layers.{i}.")]
        if stale:
            fail(f"{len(stale)} tensors of block {i} remain")
    heads = [k for k in out if re.fullmatch(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", k)]
    if len(heads) != 12:
        fail(f"{len(heads)} blocks remain, expected 12")
    idx = sorted(int(BLOCK_RE.match(k).group(1)) for k in heads)
    if idx != list(range(12)):
        fail(f"block indices are not contiguous 0..11: {idx}")
    if len(out) != 184:
        fail(f"output would have {len(out)} tensors, expected 184")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST))
    print(f"wrote {DST} with {len(out)} tensors, {len(heads)} blocks")


if __name__ == "__main__":
    main()
