#!/usr/bin/env python
"""T1: depth-prune GPT-2 (124M) by dropping blocks 2, 5, 8 and renumbering
the survivors 0..8, contiguous, in original order.

Plain script on top of `safetensors` (loading with the torch framework so
tensors stay contiguous for saving). No mergekit / torch-state-bridge layer
here: the rename is a single regex substitution plus an explicit drop set,
which a config-driven merge tool would not express more safely than this.

Fails loudly (non-zero exit, nothing written to the final output path) if
any required check does not hold.
"""

import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")

DROP_BLOCKS = {2, 5, 8}
BLOCK_RE = re.compile(r"^h\.(\d+)\.(.*)$")


def main() -> None:
    if not INPUT.exists():
        sys.exit(f"input not found: {INPUT}")

    with safe_open(str(INPUT), framework="pt") as f:
        keys = list(f.keys())
        tensors = {k: f.get_tensor(k) for k in keys}

    if len(tensors) != 160:
        sys.exit(f"expected 160 input tensors, got {len(tensors)}")

    # Surviving old indices, in original order -> contiguous new indices.
    old_blocks = sorted(
        {int(m.group(1)) for k in tensors if (m := BLOCK_RE.match(k))}
    )
    surviving = [i for i in old_blocks if i not in DROP_BLOCKS]
    if surviving != [0, 1, 3, 4, 6, 7, 9, 10, 11]:
        sys.exit(f"unexpected surviving block set: {surviving}")
    remap = {old: new for new, old in enumerate(surviving)}

    out = {}
    for key, tensor in tensors.items():
        m = BLOCK_RE.match(key)
        if m is None:
            # Non-block tensor: unchanged.
            out[key] = tensor
            continue
        old_idx = int(m.group(1))
        if old_idx in DROP_BLOCKS:
            continue
        new_idx = remap[old_idx]
        out[f"h.{new_idx}.{m.group(2)}"] = tensor

    # --- Required checks: fail loudly, write nothing, on any violation. ---

    # No tensor of the dropped (originally last) blocks remains.
    for i in (9, 10, 11):
        leaked = [k for k in out if re.match(rf"^h\.{i}\.", k)]
        if leaked:
            sys.exit(f"block {i} leaked into output: {leaked}")

    # Exactly 9 blocks remain, e.g. exactly 9 c_attn.weight tensors.
    c_attn_matches = [k for k in out if re.match(r"^h\.\d+\.attn\.c_attn\.weight$", k)]
    if len(c_attn_matches) != 9:
        sys.exit(f"expected 9 blocks (c_attn.weight), got {len(c_attn_matches)}: {c_attn_matches}")
    new_block_indices = sorted(
        {int(m.group(1)) for k in out if (m := BLOCK_RE.match(k))}
    )
    if new_block_indices != list(range(9)):
        sys.exit(f"block indices not contiguous 0..8: {new_block_indices}")

    # Exactly 121 tensors total (9 * 13 + 4).
    if len(out) != 121:
        sys.exit(f"expected 121 output tensors, got {len(out)}")

    # Non-block tensors present and unchanged in count (sanity: 4 of them).
    non_block = [k for k in out if BLOCK_RE.match(k) is None]
    if len(non_block) != 4:
        sys.exit(f"expected 4 non-block tensors, got {len(non_block)}: {non_block}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".safetensors.tmp")
    save_file(out, str(tmp))
    tmp.rename(OUTPUT)
    print(f"wrote {OUTPUT} with {len(out)} tensors")


if __name__ == "__main__":
    main()
