"""
T1: Depth pruning with layer renumbering (Pythia-1B).

Removes transformer blocks 2, 6, 10, 14 from a 16-layer Pythia-1B checkpoint
and renumbers the surviving blocks to 0..11 (in original order), leaving all
non-block tensors untouched.
"""

import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
IN_PATH = HERE.parents[1] / "inputs" / "base" / "model.safetensors"
OUT_DIR = HERE
OUT_PATH = OUT_DIR / "model.safetensors"

DROP_BLOCKS = {2, 6, 10, 14}
NUM_LAYERS_IN = 16
NUM_LAYERS_OUT = 12
TENSORS_PER_BLOCK = 15
NUM_NON_BLOCK = 4
EXPECTED_TOTAL_IN = NUM_LAYERS_IN * TENSORS_PER_BLOCK + NUM_NON_BLOCK
EXPECTED_TOTAL_OUT = NUM_LAYERS_OUT * TENSORS_PER_BLOCK + NUM_NON_BLOCK

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.")


def build_renumber_map():
    surviving_old = [i for i in range(NUM_LAYERS_IN) if i not in DROP_BLOCKS]
    if len(surviving_old) != NUM_LAYERS_OUT:
        raise RuntimeError(
            f"expected {NUM_LAYERS_OUT} surviving blocks, got {len(surviving_old)}"
        )
    mapping = {old: new for new, old in enumerate(surviving_old)}
    return mapping


def main():
    if not IN_PATH.exists():
        raise RuntimeError(f"input checkpoint not found: {IN_PATH}")

    old_to_new = build_renumber_map()

    with safe_open(str(IN_PATH), framework="pt") as f:
        keys = list(f.keys())
        if len(keys) != EXPECTED_TOTAL_IN:
            raise RuntimeError(
                f"expected {EXPECTED_TOTAL_IN} input tensors, found {len(keys)}"
            )

        out_tensors = {}
        for key in keys:
            m = LAYER_RE.match(key)
            if m is None:
                # non-block tensor: copy unchanged
                out_tensors[key] = f.get_tensor(key)
                continue

            old_idx = int(m.group(1))
            if old_idx in DROP_BLOCKS:
                continue  # drop this block's tensor

            new_idx = old_to_new[old_idx]
            new_key = f"gpt_neox.layers.{new_idx}." + key[m.end() :]
            if new_key in out_tensors:
                raise RuntimeError(f"collision writing {new_key} (from {key})")
            out_tensors[new_key] = f.get_tensor(key)

    # --- Required checks ---

    # no tensor of blocks 12, 13, 14, 15 remains
    for bad_idx in (12, 13, 14, 15):
        prefix = f"gpt_neox.layers.{bad_idx}."
        if any(k.startswith(prefix) for k in out_tensors):
            raise RuntimeError(f"tensor(s) from removed/out-of-range block {bad_idx} remain")

    # exactly 12 blocks remain
    qkv_count = sum(
        1
        for k in out_tensors
        if re.match(r"^gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight$", k)
    )
    if qkv_count != NUM_LAYERS_OUT:
        raise RuntimeError(f"expected {NUM_LAYERS_OUT} blocks, found {qkv_count}")

    seen_indices = set()
    for k in out_tensors:
        m = LAYER_RE.match(k)
        if m:
            seen_indices.add(int(m.group(1)))
    if seen_indices != set(range(NUM_LAYERS_OUT)):
        raise RuntimeError(f"surviving block indices not contiguous 0..11: {sorted(seen_indices)}")

    # exactly 184 tensors total
    if len(out_tensors) != EXPECTED_TOTAL_OUT:
        raise RuntimeError(f"expected {EXPECTED_TOTAL_OUT} output tensors, got {len(out_tensors)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(out_tensors)} tensors")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
