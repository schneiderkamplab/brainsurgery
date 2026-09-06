"""T1: depth pruning with layer renumbering for OLMo-1B-0724-hf.

Removes transformer blocks {2, 6, 10, 14} from a 16-layer checkpoint and
renumbers the surviving blocks to 0..11 in original order, writing a single
merged safetensors file. Uses only the `safetensors` package.

Fails loudly (raises / non-zero exit, no output written) if any required
check does not hold.
"""

import json
import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE.parents[1] / "inputs" / "base"
OUTPUT_PATH = HERE / "model.safetensors"

REMOVE_BLOCKS = {2, 6, 10, 14}
NUM_ORIG_BLOCKS = 16
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def build_renumber_map() -> dict[int, int]:
    survivors = [i for i in range(NUM_ORIG_BLOCKS) if i not in REMOVE_BLOCKS]
    return {old: new for new, old in enumerate(survivors)}


def main() -> None:
    index_path = INPUT_DIR / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    all_keys = sorted(weight_map.keys())

    renumber = build_renumber_map()

    # Load every tensor from its shard.
    shard_files = sorted(set(weight_map.values()))
    tensors_by_key = {}
    for shard_name in shard_files:
        shard_path = INPUT_DIR / shard_name
        with safe_open(str(shard_path), framework="pt") as f:
            for key in f.keys():
                tensors_by_key[key] = f.get_tensor(key)

    assert set(tensors_by_key.keys()) == set(all_keys), "shard contents do not match index"

    output = {}
    seen_new_keys = set()
    for key in all_keys:
        m = LAYER_RE.match(key)
        if m is None:
            # non-block tensor, unchanged
            new_key = key
        else:
            old_idx = int(m.group(1))
            if old_idx in REMOVE_BLOCKS:
                continue
            new_idx = renumber[old_idx]
            new_key = LAYER_RE.sub(f"model.layers.{new_idx}.", key)

        if new_key in seen_new_keys:
            raise RuntimeError(f"collision: {new_key!r} produced twice")
        seen_new_keys.add(new_key)
        output[new_key] = tensors_by_key[key]

    # --- Required checks ---
    removed_old_indices = {12, 13, 14, 15}
    for key in output:
        m = LAYER_RE.match(key)
        if m is None:
            continue
        # After renumbering, indices only run 0..11; check no leftover
        # tensor still carries an old index from a removed/renumbered-away
        # block by re-deriving from the layer count directly below.

    block_indices = set()
    for key in output:
        m = LAYER_RE.match(key)
        if m is not None:
            block_indices.add(int(m.group(1)))

    if block_indices != set(range(12)):
        raise RuntimeError(f"expected block indices 0..11, got {sorted(block_indices)}")

    q_proj_count = sum(1 for k in output if re.match(r"^model\.layers\.\d+\.self_attn\.q_proj\.weight$", k))
    if q_proj_count != 12:
        raise RuntimeError(f"expected exactly 12 blocks (q_proj count), got {q_proj_count}")

    if len(output) != 86:
        raise RuntimeError(f"expected exactly 86 tensors, got {len(output)}")

    # Non-block tensors unchanged in key and value.
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        if key not in output:
            raise RuntimeError(f"missing non-block tensor {key}")
        if not (output[key] == tensors_by_key[key]).all():
            raise RuntimeError(f"non-block tensor {key} was modified")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, str(OUTPUT_PATH))
    print(f"wrote {OUTPUT_PATH} with {len(output)} tensors")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        if OUTPUT_PATH.exists():
            OUTPUT_PATH.unlink()
        sys.exit(1)
