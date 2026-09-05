"""
Depth-prune Pythia-1B from 16 to 12 transformer blocks.

Removes blocks 2, 6, 10, 14 and renumbers the remaining blocks in their
original order so indices run 0..11 contiguously, then writes the result
to out/T1/model.safetensors.
"""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
IN_PATH = HERE.parents[1] / "inputs" / "base" / "model.safetensors"
OUT_PATH = HERE / "model.safetensors"

BLOCKS_TO_REMOVE = {2, 6, 10, 14}
NUM_BLOCKS_IN = 16
EXPECTED_TENSORS_PER_BLOCK = 15
EXPECTED_NON_BLOCK_TENSORS = 4
EXPECTED_OUTPUT_TENSORS = 184

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.")


def main():
    if not IN_PATH.is_file():
        print(f"input not found: {IN_PATH}", file=sys.stderr)
        sys.exit(1)

    tensors = load_file(str(IN_PATH))

    # Bucket tensor names by block index; keep non-block tensors separate.
    by_block = {i: {} for i in range(NUM_BLOCKS_IN)}
    non_block = {}
    for name, tensor in tensors.items():
        m = LAYER_RE.match(name)
        if m is None:
            non_block[name] = tensor
        else:
            idx = int(m.group(1))
            if idx not in by_block:
                raise AssertionError(f"unexpected block index {idx} in tensor name {name!r}")
            by_block[idx][name] = tensor

    # Sanity-check the input matches the documented shape of the checkpoint.
    for idx, block_tensors in by_block.items():
        if len(block_tensors) != EXPECTED_TENSORS_PER_BLOCK:
            raise AssertionError(
                f"block {idx} has {len(block_tensors)} tensors, "
                f"expected {EXPECTED_TENSORS_PER_BLOCK}"
            )
    if len(non_block) != EXPECTED_NON_BLOCK_TENSORS:
        raise AssertionError(
            f"expected {EXPECTED_NON_BLOCK_TENSORS} non-block tensors, got {len(non_block)}"
        )

    surviving_old_indices = [i for i in range(NUM_BLOCKS_IN) if i not in BLOCKS_TO_REMOVE]
    if len(surviving_old_indices) != 12:
        raise AssertionError(f"expected 12 surviving blocks, got {len(surviving_old_indices)}")

    # Renumber surviving blocks in original order, working from lowest new
    # index to highest so a block is never moved into a slot that still
    # holds a not-yet-moved survivor.
    output = dict(non_block)
    for new_idx, old_idx in enumerate(surviving_old_indices):
        for name, tensor in by_block[old_idx].items():
            new_name = name.replace(
                f"gpt_neox.layers.{old_idx}.", f"gpt_neox.layers.{new_idx}.", 1
            )
            if new_name in output:
                raise AssertionError(f"collision writing {new_name!r}")
            output[new_name] = tensor

    # --- Required checks ---
    for bad_idx in (12, 13, 14, 15):
        pattern = re.compile(rf"^gpt_neox\.layers\.{bad_idx}\.")
        if any(pattern.match(name) for name in output):
            raise AssertionError(f"tensor of removed/old-high block {bad_idx} present in output")

    qkv_pattern = re.compile(r"^gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.weight$")
    qkv_matches = [name for name in output if qkv_pattern.match(name)]
    if len(qkv_matches) != 12:
        raise AssertionError(f"expected 12 blocks (qkv weights), got {len(qkv_matches)}")

    if len(output) != EXPECTED_OUTPUT_TENSORS:
        raise AssertionError(
            f"expected {EXPECTED_OUTPUT_TENSORS} output tensors, got {len(output)}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {name: tensor.contiguous() for name, tensor in output.items()}
    save_file(output, str(OUT_PATH))
    print(f"wrote {len(output)} tensors to {OUT_PATH}")


if __name__ == "__main__":
    main()
