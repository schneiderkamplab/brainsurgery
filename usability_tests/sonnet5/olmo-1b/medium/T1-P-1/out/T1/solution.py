"""
T1: Depth pruning with layer renumbering (OLMo-1B-0724-hf).

Removes transformer blocks 2, 6, 10, 14 from a 16-layer checkpoint and
renumbers the remaining blocks to run contiguously 0..11, then writes a
single-file safetensors checkpoint with the surviving 86 tensors.
"""

import json
import os
import re
import sys

from safetensors import safe_open
from safetensors.torch import save_file

INPUT_DIR = "inputs/base"
INDEX_PATH = os.path.join(INPUT_DIR, "model.safetensors.index.json")
OUTPUT_DIR = "out/T1"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "model.safetensors")

REMOVE_BLOCKS = {2, 6, 10, 14}
NUM_ORIGINAL_BLOCKS = 16
NUM_SURVIVING_BLOCKS = NUM_ORIGINAL_BLOCKS - len(REMOVE_BLOCKS)  # 12
TENSORS_PER_BLOCK = 7
NUM_NON_BLOCK_TENSORS = 2
EXPECTED_TOTAL = NUM_SURVIVING_BLOCKS * TENSORS_PER_BLOCK + NUM_NON_BLOCK_TENSORS  # 86

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    with open(INDEX_PATH) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    surviving_old_indices = sorted(
        i for i in range(NUM_ORIGINAL_BLOCKS) if i not in REMOVE_BLOCKS
    )
    old_to_new = {old: new for new, old in enumerate(surviving_old_indices)}

    # Open all shard files referenced by the index.
    shard_files = sorted(set(weight_map.values()))
    handles = {
        shard: safe_open(os.path.join(INPUT_DIR, shard), framework="pt")
        for shard in shard_files
    }

    output_tensors = {}
    for old_key, shard in weight_map.items():
        m = LAYER_RE.match(old_key)
        if m is None:
            # Non-block tensor: keep name and value unchanged.
            new_key = old_key
        else:
            old_idx = int(m.group(1))
            if old_idx in REMOVE_BLOCKS:
                continue
            new_idx = old_to_new[old_idx]
            new_key = LAYER_RE.sub(f"model.layers.{new_idx}.", old_key)

        if new_key in output_tensors:
            fail(f"collision: two source tensors map to '{new_key}'")

        output_tensors[new_key] = handles[shard].get_tensor(old_key)

    # --- Required checks ---

    # No tensor of the four removed *old* blocks may remain under an index
    # equal to any of the removed indices' typical high range (12..15 would
    # only appear if a collision/mis-shift happened). Concretely: the new
    # block indices are 0..11 only.
    layer_indices_present = set()
    for key in output_tensors:
        m = LAYER_RE.match(key)
        if m is not None:
            layer_indices_present.add(int(m.group(1)))

    if layer_indices_present != set(range(NUM_SURVIVING_BLOCKS)):
        fail(
            "surviving block indices are not exactly 0..11: "
            f"got {sorted(layer_indices_present)}"
        )

    for forbidden in (12, 13, 14, 15):
        if forbidden in layer_indices_present:
            fail(f"tensor of block {forbidden} still present")

    q_proj_count = sum(
        1 for k in output_tensors if k.endswith("self_attn.q_proj.weight")
    )
    if q_proj_count != NUM_SURVIVING_BLOCKS:
        fail(f"expected exactly {NUM_SURVIVING_BLOCKS} q_proj tensors, got {q_proj_count}")

    if len(output_tensors) != EXPECTED_TOTAL:
        fail(f"expected exactly {EXPECTED_TOTAL} tensors, got {len(output_tensors)}")

    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        if name not in output_tensors:
            fail(f"missing non-block tensor '{name}'")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_file(output_tensors, OUTPUT_PATH)

    print(f"OK: wrote {len(output_tensors)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
