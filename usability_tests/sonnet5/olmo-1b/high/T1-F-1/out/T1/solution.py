#!/usr/bin/env python3
"""
T1: Depth pruning with layer renumbering (OLMo-1B-0724-hf).

Plain script on top of `safetensors` (one of the allowed packages in
F-allowed.md). Loads all shards of the input checkpoint, drops every tensor
belonging to blocks 2, 6, 10, 14, renumbers the surviving blocks to 0..11 in
original order, leaves the two non-block tensors untouched, and writes a
single-file safetensors checkpoint. Fails loudly (raises / non-zero exit,
no output written) if any required-check does not hold.
"""

import bisect
import json
import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE.parent.parent / "inputs" / "base"
OUTPUT_FILE = HERE / "model.safetensors"

REMOVE_BLOCKS = sorted([2, 6, 10, 14])
NUM_ORIGINAL_BLOCKS = 16
NUM_SURVIVING_BLOCKS = NUM_ORIGINAL_BLOCKS - len(REMOVE_BLOCKS)

BLOCK_RE = re.compile(r"^model\.layers\.(\d+)\.(.*)$")


def new_block_index(old_idx: int) -> int:
    """Shift old_idx down by the number of removed blocks below it."""
    shift = bisect.bisect_left(REMOVE_BLOCKS, old_idx)
    return old_idx - shift


def main() -> None:
    index_path = INPUT_DIR / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Load every shard exactly once.
    shard_names = sorted(set(weight_map.values()))
    shards = {name: load_file(str(INPUT_DIR / name)) for name in shard_names}

    def get_tensor(name: str):
        shard = weight_map[name]
        return shards[shard][name]

    out = {}
    seen_old_block_indices = set()

    for name in weight_map:
        m = BLOCK_RE.match(name)
        if m is None:
            # Non-block tensor: copy unchanged.
            out[name] = get_tensor(name)
            continue

        old_idx = int(m.group(1))
        rest = m.group(2)
        seen_old_block_indices.add(old_idx)

        if old_idx in REMOVE_BLOCKS:
            continue  # drop this tensor entirely

        new_idx = new_block_index(old_idx)
        new_name = f"model.layers.{new_idx}.{rest}"
        if new_name in out:
            raise RuntimeError(
                f"collision while renumbering: {name} -> {new_name} "
                f"already produced by another source tensor"
            )
        out[new_name] = get_tensor(name)

    if seen_old_block_indices != set(range(NUM_ORIGINAL_BLOCKS)):
        raise RuntimeError(
            f"expected blocks 0..{NUM_ORIGINAL_BLOCKS - 1} in input, "
            f"found {sorted(seen_old_block_indices)}"
        )

    # --- Required checks: fail loudly, write nothing on failure. ---

    # 1. No tensor of (new) blocks 12, 13, 14, 15 remains.
    for banned in (12, 13, 14, 15):
        prefix = f"model.layers.{banned}."
        if any(k.startswith(prefix) for k in out):
            raise RuntimeError(f"found tensor(s) of banned block {banned} in output")

    # 2. Exactly 12 blocks remain.
    q_proj_count = sum(
        1 for k in out if re.match(r"^model\.layers\.\d+\.self_attn\.q_proj\.weight$", k)
    )
    if q_proj_count != NUM_SURVIVING_BLOCKS:
        raise RuntimeError(
            f"expected {NUM_SURVIVING_BLOCKS} surviving blocks, "
            f"found {q_proj_count} q_proj tensors"
        )
    present_indices = sorted(
        int(re.match(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$", k).group(1))
        for k in out
        if re.match(r"^model\.layers\.\d+\.self_attn\.q_proj\.weight$", k)
    )
    if present_indices != list(range(NUM_SURVIVING_BLOCKS)):
        raise RuntimeError(f"surviving block indices not contiguous 0..11: {present_indices}")

    # 3. Output has exactly 86 tensors.
    expected_total = NUM_SURVIVING_BLOCKS * 7 + 2
    if len(out) != expected_total:
        raise RuntimeError(f"expected {expected_total} tensors in output, found {len(out)}")

    # Make sure every tensor is contiguous before saving.
    out = {k: v.contiguous() for k, v in out.items()}

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUTPUT_FILE))
    print(f"wrote {len(out)} tensors to {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
