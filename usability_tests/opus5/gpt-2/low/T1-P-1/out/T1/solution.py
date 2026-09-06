#!/usr/bin/env python
"""T1: depth-prune GPT-2 (124M) to 9 layers, renumbering surviving blocks."""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T1/model.safetensors")

DROP = {2, 5, 8}
N_OLD = 12
BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    src = load_file(str(SRC))
    if len(src) != 160:
        fail(f"expected 160 input tensors, got {len(src)}")

    keep = [i for i in range(N_OLD) if i not in DROP]
    remap = {old: new for new, old in enumerate(keep)}

    out: dict = {}
    for name, tensor in src.items():
        m = BLOCK_RE.match(name)
        if m is None:
            out[name] = tensor
            continue
        old = int(m.group(1))
        if old in DROP:
            continue
        new_name = f"h.{remap[old]}.{m.group(2)}"
        if new_name in out:
            fail(f"renumbering collision on {new_name}")
        out[new_name] = tensor

    # Required checks.
    indices = set()
    for name in out:
        m = BLOCK_RE.match(name)
        if m is not None:
            indices.add(int(m.group(1)))

    stale = indices & {9, 10, 11}
    if stale:
        fail(f"tensors of removed block indices remain: {sorted(stale)}")
    if indices != set(range(9)):
        fail(f"block indices are not 0..8: {sorted(indices)}")

    n_attn = sum(1 for k in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", k))
    if n_attn != 9:
        fail(f"expected 9 blocks, found {n_attn} h.<i>.attn.c_attn.weight tensors")

    if len(out) != 121:
        fail(f"expected 121 output tensors, got {len(out)}")

    for name in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"):
        if name not in out:
            fail(f"missing non-block tensor {name}")
        if out[name] is not src[name]:
            fail(f"non-block tensor {name} was modified")

    # Values/shapes/dtypes must be untouched.
    for old, new in remap.items():
        for key in src:
            m = BLOCK_RE.match(key)
            if m is None or int(m.group(1)) != old:
                continue
            tgt = out[f"h.{new}.{m.group(2)}"]
            if tgt.shape != src[key].shape or tgt.dtype != src[key].dtype:
                fail(f"shape/dtype changed for {key}")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, str(DST))
    print(f"wrote {DST} with {len(out)} tensors ({n_attn} blocks)")


if __name__ == "__main__":
    main()
