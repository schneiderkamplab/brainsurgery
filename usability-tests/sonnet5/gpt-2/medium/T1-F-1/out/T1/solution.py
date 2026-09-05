#!/usr/bin/env python3
"""T1: depth-prune GPT-2 by dropping blocks {2, 5, 8} and renumbering the
survivors 0..8 in original order. Plain script on top of `safetensors`
(no mergekit/torch-state-bridge needed: the rename is a single regex over a
flat state dict, and doing it directly avoids pulling in a YAML merge config
or a rule-engine for a one-line renumbering).

Fails loudly (non-zero exit, no output written) if any required check does
not hold.
"""
import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
DST = HERE / "model.safetensors"

DROP_BLOCKS = {2, 5, 8}
BLOCK_RE = re.compile(r"^h\.(\d+)\.")

# Old block index -> new block index, in original order, skipping dropped
# blocks. Built programmatically so the mapping matches DROP_BLOCKS exactly.
SURVIVORS = [i for i in range(12) if i not in DROP_BLOCKS]
OLD_TO_NEW = {old: new for new, old in enumerate(SURVIVORS)}
assert OLD_TO_NEW == {0: 0, 1: 1, 3: 2, 4: 3, 6: 4, 7: 5, 9: 6, 10: 7, 11: 8}


def main() -> None:
    with safe_open(str(SRC), framework="pt") as f:
        src_keys = list(f.keys())
        tensors = {k: f.get_tensor(k) for k in src_keys}

    if len(src_keys) != 160:
        sys.exit(f"expected 160 input tensors, got {len(src_keys)}")

    out = {}
    seen_new_keys = set()
    for key, tensor in tensors.items():
        m = BLOCK_RE.match(key)
        if m is None:
            # non-block tensor: wte.weight, wpe.weight, ln_f.weight, ln_f.bias
            out[key] = tensor
            continue
        old_idx = int(m.group(1))
        if old_idx in DROP_BLOCKS:
            continue
        new_idx = OLD_TO_NEW[old_idx]
        new_key = f"h.{new_idx}." + key[m.end():]
        if new_key in seen_new_keys:
            sys.exit(f"collision: {new_key} already produced (from {key})")
        seen_new_keys.add(new_key)
        out[new_key] = tensor

    # --- Required checks: fail loudly, write nothing on failure ---

    # Every source tensor of old blocks 9, 10, 11 must have been renumbered
    # away (to h.6/h.7/h.8) or dropped; none may remain under its old name.
    for old_idx in (9, 10, 11):
        old_prefix = f"h.{old_idx}."
        stale = [k for k in out if k.startswith(old_prefix)]
        if stale:
            sys.exit(f"tensor of old block {old_idx} remains under its old name: {stale[0]}")

    block_count = sum(1 for k in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", k))
    if block_count != 9:
        sys.exit(f"expected exactly 9 surviving blocks, found {block_count}")
    if sorted(
        int(re.fullmatch(r"h\.(\d+)\.attn\.c_attn\.weight", k).group(1))
        for k in out
        if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", k)
    ) != list(range(9)):
        sys.exit("surviving block indices are not exactly 0..8")

    if len(out) != 121:
        sys.exit(f"expected exactly 121 output tensors, got {len(out)}")

    for name in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"):
        if name not in out:
            sys.exit(f"missing non-block tensor {name}")
        if not out[name].equal(tensors[name]):
            sys.exit(f"non-block tensor {name} was modified")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST))
    print(f"wrote {DST} with {len(out)} tensors")


if __name__ == "__main__":
    main()
