#!/usr/bin/env python
"""
T1: Depth pruning with layer renumbering (OLMo-1B-0724-hf).

Plain script on top of `safetensors`: load the sharded input checkpoint,
drop every tensor belonging to blocks {2, 6, 10, 14}, renumber the
surviving blocks' indices to 0..11 (order-preserving), leave the 2
non-block tensors untouched, and write a single-file safetensors output.

Renumbering is done old->new via a dict built from the sorted list of
surviving old indices, and applied by building a brand-new dict (never
mutating in place), so there is no collision hazard from writing a new
index over a not-yet-moved old one.
"""

import json
import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

INPUT_DIR = Path("inputs/base")
OUTPUT_PATH = Path("out/T1/model.safetensors")
DROP_BLOCKS = {2, 6, 10, 14}
NUM_ORIGINAL_BLOCKS = 16
EXPECTED_SURVIVORS = NUM_ORIGINAL_BLOCKS - len(DROP_BLOCKS)  # 12
TENSORS_PER_BLOCK = 7
NON_BLOCK_TENSORS = {"model.embed_tokens.weight", "lm_head.weight"}
EXPECTED_TOTAL = EXPECTED_SURVIVORS * TENSORS_PER_BLOCK + len(NON_BLOCK_TENSORS)  # 86

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    index_path = INPUT_DIR / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Load every shard fully (small model; simplest correct approach).
    shard_files = sorted(set(weight_map.values()))
    state_dict = {}
    for shard in shard_files:
        state_dict.update(load_file(INPUT_DIR / shard))

    if set(state_dict.keys()) != set(weight_map.keys()):
        fail("loaded tensors do not match the index's key set")

    # Bucket keys by block index, and keep non-block keys as-is.
    old_indices = set()
    for key in state_dict:
        m = LAYER_RE.match(key)
        if m:
            old_indices.add(int(m.group(1)))

    if old_indices != set(range(NUM_ORIGINAL_BLOCKS)):
        fail(f"expected blocks 0..{NUM_ORIGINAL_BLOCKS - 1}, found {sorted(old_indices)}")

    surviving_old = sorted(old_indices - DROP_BLOCKS)
    if len(surviving_old) != EXPECTED_SURVIVORS:
        fail(f"expected {EXPECTED_SURVIVORS} surviving blocks, got {len(surviving_old)}")

    old_to_new = {old: new for new, old in enumerate(surviving_old)}

    new_state_dict = {}
    for key, tensor in state_dict.items():
        m = LAYER_RE.match(key)
        if m is None:
            if key in NON_BLOCK_TENSORS:
                new_state_dict[key] = tensor
            else:
                fail(f"unexpected non-block-pattern key: {key}")
            continue
        old_idx = int(m.group(1))
        if old_idx in DROP_BLOCKS:
            continue
        new_idx = old_to_new[old_idx]
        new_key = f"model.layers.{new_idx}." + key[m.end():]
        if new_key in new_state_dict:
            fail(f"collision: {new_key} already produced (from old block {old_idx})")
        new_state_dict[new_key] = tensor

    # --- Required checks: fail loudly, write nothing on failure. ---

    # No tensor of blocks 12, 13, 14, 15 remains under the NEW numbering
    # (these are the indices that would only exist if renumbering left
    # gaps or the old top blocks were mistakenly kept as-is).
    for forbidden in (12, 13, 14, 15):
        prefix = f"model.layers.{forbidden}."
        if any(k.startswith(prefix) for k in new_state_dict):
            fail(f"tensor of forbidden block {forbidden} remains in output")

    # Exactly 12 blocks remain.
    q_proj_count = sum(
        1 for k in new_state_dict if re.match(r"^model\.layers\.\d+\.self_attn\.q_proj\.weight$", k)
    )
    if q_proj_count != EXPECTED_SURVIVORS:
        fail(f"expected exactly {EXPECTED_SURVIVORS} blocks, found {q_proj_count}")
    surviving_new_indices = sorted(
        int(re.match(r"^model\.layers\.(\d+)\.", k).group(1))
        for k in new_state_dict
        if LAYER_RE.match(k)
    )
    if set(surviving_new_indices) != set(range(EXPECTED_SURVIVORS)):
        fail(f"surviving block indices are not contiguous 0..{EXPECTED_SURVIVORS - 1}: "
             f"{sorted(set(surviving_new_indices))}")

    # Exactly 86 tensors in total.
    if len(new_state_dict) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} tensors, got {len(new_state_dict)}")

    # Non-block tensors present and unchanged.
    for key in NON_BLOCK_TENSORS:
        if key not in new_state_dict:
            fail(f"missing non-block tensor: {key}")
        if not new_state_dict[key].equal(state_dict[key]):
            fail(f"non-block tensor {key} was modified")

    # Make tensors contiguous for safetensors (loaded tensors already are,
    # but be defensive since safetensors rejects non-contiguous storage).
    new_state_dict = {k: v.contiguous() for k, v in new_state_dict.items()}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(new_state_dict, str(OUTPUT_PATH))

    print(f"Wrote {len(new_state_dict)} tensors to {OUTPUT_PATH}")
    print(f"Surviving old block order -> new index: {old_to_new}")


if __name__ == "__main__":
    main()
