#!/usr/bin/env python3
"""T1: depth-prune Pythia-1B from 16 to 12 transformer blocks.

Removes blocks 2, 6, 10, 14 and renumbers the survivors in original order to
0..11 (old 0,1,3,4,5,7,8,9,11,12,13,15 -> new 0,1,2,3,4,5,6,7,8,9,10,11). Uses
plain safetensors I/O: this is a pure bulk-rename/delete on a flat state dict,
which a hand-written script does directly and unambiguously (no risk of a
layer-slicing tool's off-by-one or collision from renumbering in the wrong
order, since we build a brand-new dict keyed by the new names rather than
mutating in place).
"""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "inputs" / "base" / "model.safetensors"
OUTPUT_DIR = REPO_ROOT / "out" / "T1"
OUTPUT_PATH = OUTPUT_DIR / "model.safetensors"

DROP_BLOCKS = {2, 6, 10, 14}
NUM_ORIG_BLOCKS = 16
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.")


def main() -> None:
    state_dict = load_file(str(INPUT_PATH))

    surviving_old_indices = sorted(
        i for i in range(NUM_ORIG_BLOCKS) if i not in DROP_BLOCKS
    )
    old_to_new = {old: new for new, old in enumerate(surviving_old_indices)}

    out = {}
    for key, tensor in state_dict.items():
        m = LAYER_RE.match(key)
        if m is None:
            # non-block tensor: unchanged
            out[key] = tensor
            continue
        old_idx = int(m.group(1))
        if old_idx in DROP_BLOCKS:
            continue
        new_idx = old_to_new[old_idx]
        new_key = f"gpt_neox.layers.{new_idx}." + key[m.end() :]
        if new_key in out:
            raise RuntimeError(f"collision: {new_key} already produced")
        out[new_key] = tensor

    # --- Required checks: fail loudly, write nothing on failure ---
    for bad in (12, 13, 14, 15):
        prefix = f"gpt_neox.layers.{bad}."
        if any(k.startswith(prefix) for k in out):
            raise AssertionError(f"tensor of dropped/old block {bad} present in output")

    block_qkv_count = sum(
        1 for k in out if re.fullmatch(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", k)
    )
    if block_qkv_count != 12:
        raise AssertionError(f"expected 12 blocks, found {block_qkv_count}")

    if len(out) != 184:
        raise AssertionError(f"expected 184 tensors in output, found {len(out)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUTPUT_PATH))
    print(f"wrote {len(out)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # fail loudly, non-zero exit
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
