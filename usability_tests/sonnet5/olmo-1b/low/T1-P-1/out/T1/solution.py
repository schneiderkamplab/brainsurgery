"""
Depth-prune OLMo-1B-0724-hf: remove transformer blocks 2, 6, 10, 14 and
renumber the remaining blocks to 0..11 with no gaps.

Reads the sharded safetensors checkpoint under inputs/base/ and writes a
single-file checkpoint to out/T1/model.safetensors.
"""

import json
import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPO_ROOT / "inputs" / "base"
OUTPUT_DIR = REPO_ROOT / "out" / "T1"
OUTPUT_PATH = OUTPUT_DIR / "model.safetensors"

DROP_BLOCKS = {2, 6, 10, 14}
NUM_ORIGINAL_BLOCKS = 16
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    index_path = INPUT_DIR / "model.safetensors.index.json"
    if not index_path.exists():
        fail(f"missing index file {index_path}")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))

    # Load all tensors from all shards into one dict.
    all_tensors = {}
    for shard_name in shard_names:
        shard_path = INPUT_DIR / shard_name
        if not shard_path.exists():
            fail(f"missing shard {shard_path}")
        shard_tensors = load_file(str(shard_path))
        for key, tensor in shard_tensors.items():
            all_tensors[key] = tensor

    if set(all_tensors.keys()) != set(weight_map.keys()):
        fail("tensor keys loaded from shards do not match index weight_map")

    if len(all_tensors) != 114:
        fail(f"expected 114 input tensors, got {len(all_tensors)}")

    # Build the surviving-block renumbering map, in original order.
    surviving_blocks = [i for i in range(NUM_ORIGINAL_BLOCKS) if i not in DROP_BLOCKS]
    if len(surviving_blocks) != 12:
        fail(f"expected 12 surviving blocks, got {len(surviving_blocks)}")
    old_to_new = {old: new for new, old in enumerate(surviving_blocks)}

    output_tensors = {}
    for key, tensor in all_tensors.items():
        m = LAYER_RE.match(key)
        if m is None:
            # Non-block tensor: copy unchanged.
            output_tensors[key] = tensor
            continue
        old_idx = int(m.group(1))
        if old_idx in DROP_BLOCKS:
            continue
        new_idx = old_to_new[old_idx]
        new_key = f"model.layers.{new_idx}." + key[m.end():]
        if new_key in output_tensors:
            fail(f"collision writing {new_key} (from {key})")
        output_tensors[new_key] = tensor

    # --- Required checks ---

    # No tensor of the removed blocks (or now-invalid indices 12..15) remains.
    for i in (12, 13, 14, 15):
        pattern = re.compile(rf"^model\.layers\.{i}\.")
        if any(pattern.match(k) for k in output_tensors):
            fail(f"tensor of block {i} still present in output")

    # Exactly 12 blocks remain.
    q_proj_pattern = re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$")
    block_indices = {
        int(m.group(1))
        for k in output_tensors
        if (m := q_proj_pattern.match(k)) is not None
    }
    if block_indices != set(range(12)):
        fail(f"expected block indices 0..11, got {sorted(block_indices)}")

    # Non-block tensors unchanged.
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        if key not in output_tensors:
            fail(f"missing non-block tensor {key}")
        if not output_tensors[key].equal(all_tensors[key]):
            fail(f"non-block tensor {key} was modified")

    # Exactly 86 tensors in the output.
    if len(output_tensors) != 86:
        fail(f"expected 86 output tensors, got {len(output_tensors)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(output_tensors, str(OUTPUT_PATH))

    print(f"Wrote {len(output_tensors)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
