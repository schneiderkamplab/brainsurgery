#!/usr/bin/env python3
"""T1: depth-prune Pythia-1B by dropping blocks 2, 6, 10, 14 and renumbering
the 12 survivors to 0..11, keeping the 4 non-block tensors unchanged.

Plain script on top of `safetensors` (condition F allowed list): this task is
a fully-specified bulk rename with a fixed old->new index map, which does not
need mergekit's merge machinery or torch-state-bridge's regex engine -- a
direct rename is the smallest correct tool for the job.
"""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

DROP_BLOCKS = {2, 6, 10, 14}
NUM_LAYERS_IN = 16
NUM_LAYERS_OUT = 12

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def build_old_to_new_map() -> dict[int, int]:
    surviving = [i for i in range(NUM_LAYERS_IN) if i not in DROP_BLOCKS]
    if len(surviving) != NUM_LAYERS_OUT:
        raise AssertionError(
            f"expected {NUM_LAYERS_OUT} surviving blocks, got {len(surviving)}"
        )
    return {old: new for new, old in enumerate(surviving)}


def main() -> int:
    in_path = Path("inputs/base/model.safetensors")
    out_dir = Path("out/T1")
    out_path = out_dir / "model.safetensors"

    state_dict = load_file(str(in_path))

    old_to_new = build_old_to_new_map()

    new_state_dict: dict[str, "torch.Tensor"] = {}
    seen_block_indices: set[int] = set()

    for key, tensor in state_dict.items():
        m = LAYER_RE.match(key)
        if m is None:
            # Non-block tensor: carried over unchanged.
            new_state_dict[key] = tensor
            continue

        old_idx = int(m.group(1))
        rest = m.group(2)

        if old_idx in DROP_BLOCKS:
            continue

        if old_idx not in old_to_new:
            raise AssertionError(f"block index {old_idx} not in expected range 0..15")

        new_idx = old_to_new[old_idx]
        new_key = f"gpt_neox.layers.{new_idx}.{rest}"

        if new_key in new_state_dict:
            raise AssertionError(f"collision: {new_key} already produced (from block {old_idx})")

        new_state_dict[new_key] = tensor
        seen_block_indices.add(old_idx)

    # --- Required checks: fail loudly, write nothing, on any violation. ---

    # No tensor of blocks 12, 13, 14, 15 (the dropped set plus the tail that
    # would collide under a naive shift) may remain under those indices.
    for forbidden in (12, 13, 14, 15):
        bad = [k for k in new_state_dict if k.startswith(f"gpt_neox.layers.{forbidden}.")]
        if bad:
            raise AssertionError(f"found {len(bad)} tensor(s) still at forbidden index {forbidden}: {bad[:3]}")

    # Exactly 12 blocks remain.
    qkv_weight_keys = [
        k
        for k in new_state_dict
        if re.match(r"^gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight$", k)
    ]
    if len(qkv_weight_keys) != NUM_LAYERS_OUT:
        raise AssertionError(
            f"expected exactly {NUM_LAYERS_OUT} query_key_value.weight tensors, "
            f"found {len(qkv_weight_keys)}"
        )
    present_new_indices = set()
    for k in new_state_dict:
        m = LAYER_RE.match(k)
        if m:
            present_new_indices.add(int(m.group(1)))
    if present_new_indices != set(range(NUM_LAYERS_OUT)):
        raise AssertionError(f"surviving block indices are {sorted(present_new_indices)}, expected 0..11")

    # Exactly 184 tensors total in the output.
    expected_total = 4 + NUM_LAYERS_OUT * 15
    if len(new_state_dict) != expected_total:
        raise AssertionError(f"expected {expected_total} tensors, got {len(new_state_dict)}")
    if expected_total != 184:
        raise AssertionError(f"internal error: expected_total {expected_total} != 184")

    # The 4 non-block tensors are byte-identical to the input.
    non_block_keys = [k for k in state_dict if LAYER_RE.match(k) is None]
    if len(non_block_keys) != 4:
        raise AssertionError(f"expected 4 non-block tensors in input, found {len(non_block_keys)}")
    for k in non_block_keys:
        if k not in new_state_dict or not new_state_dict[k].equal(state_dict[k]):
            raise AssertionError(f"non-block tensor {k} was altered or dropped")

    # All surviving old blocks were actually consumed exactly once.
    expected_surviving = set(range(NUM_LAYERS_IN)) - DROP_BLOCKS
    if seen_block_indices != expected_surviving:
        raise AssertionError(
            f"consumed old block indices {sorted(seen_block_indices)}, expected {sorted(expected_surviving)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    new_state_dict = {k: v.contiguous() for k, v in new_state_dict.items()}
    save_file(new_state_dict, str(out_path))

    print(f"wrote {out_path} with {len(new_state_dict)} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
