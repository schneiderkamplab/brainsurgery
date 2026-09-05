#!/usr/bin/env python3
"""Depth-prune OLMo-1B-0724-hf: drop blocks 2, 6, 10, 14 and renumber the
survivors to 0..11 contiguously. Plain script on top of `safetensors`
(chosen over mergekit's layer-slicing, which assumes a contiguous keep-range
rather than an arbitrary drop-list, and over torch-state-bridge, which would
just wrap the same regex rename).

Fails loudly (non-zero exit, no output written) if any required check does
not hold.
"""

import json
import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
IN_DIR = HERE.parent.parent / "inputs" / "base"
OUT_PATH = HERE / "model.safetensors"

DROP_BLOCKS = {2, 6, 10, 14}
N_OLD_BLOCKS = 16
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def main() -> None:
    index_path = IN_DIR / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]

    if len(weight_map) != 114:
        raise AssertionError(f"expected 114 tensors in input index, got {len(weight_map)}")

    # Build old-block -> new-block mapping for surviving blocks, in original order.
    surviving_old = [i for i in range(N_OLD_BLOCKS) if i not in DROP_BLOCKS]
    if len(surviving_old) != 12:
        raise AssertionError(f"expected 12 surviving blocks, got {len(surviving_old)}")
    old_to_new = {old: new for new, old in enumerate(surviving_old)}

    # Load every tensor exactly once, from whichever shard it lives in.
    shard_paths = {name: IN_DIR / name for name in set(weight_map.values())}
    open_shards = {name: safe_open(str(path), framework="pt") for name, path in shard_paths.items()}

    out_tensors = {}
    non_block_names = set()
    for name, shard_name in weight_map.items():
        m = LAYER_RE.match(name)
        if m is None:
            non_block_names.add(name)
            out_tensors[name] = open_shards[shard_name].get_tensor(name)
            continue
        old_idx = int(m.group(1))
        if old_idx in DROP_BLOCKS:
            continue
        new_idx = old_to_new[old_idx]
        new_name = name.replace(f"model.layers.{old_idx}.", f"model.layers.{new_idx}.", 1)
        if new_name in out_tensors:
            raise AssertionError(f"collision: {new_name} already produced (from old block {old_idx})")
        out_tensors[new_name] = open_shards[shard_name].get_tensor(name)

    if non_block_names != {"model.embed_tokens.weight", "lm_head.weight"}:
        raise AssertionError(f"unexpected non-block tensor set: {non_block_names}")

    # --- Required checks ---
    for bad in (12, 13, 14, 15):
        for name in out_tensors:
            m = LAYER_RE.match(name)
            if m and int(m.group(1)) == bad:
                raise AssertionError(f"tensor from removed/old-index block {bad} leaked into output: {name}")

    q_proj_count = sum(
        1 for name in out_tensors if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", name)
    )
    if q_proj_count != 12:
        raise AssertionError(f"expected exactly 12 q_proj tensors, got {q_proj_count}")

    if len(out_tensors) != 86:
        raise AssertionError(f"expected exactly 86 output tensors, got {len(out_tensors)}")

    # Make sure every surviving tensor is contiguous before saving.
    out_tensors = {k: v.contiguous() for k, v in out_tensors.items()}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, str(OUT_PATH))

    # Re-open and re-verify the written file to catch any save-time surprises.
    with safe_open(str(OUT_PATH), framework="pt") as f:
        keys = list(f.keys())
    if len(keys) != 86:
        raise AssertionError(f"output file has {len(keys)} tensors, expected 86")

    print(f"OK: wrote {len(keys)} tensors to {OUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # fail loudly, no partial output
        print(f"FAILED: {e}", file=sys.stderr)
        if OUT_PATH.exists():
            OUT_PATH.unlink()
        sys.exit(1)
