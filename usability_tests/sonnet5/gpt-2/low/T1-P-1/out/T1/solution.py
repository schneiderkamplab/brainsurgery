"""
T1: Depth pruning with layer renumbering (GPT-2 124M).

Remove transformer blocks 2, 5, 8 and renumber the surviving blocks so
indices run 0..8 contiguously, preserving original relative order.
"""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
IN_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUT_DIR = HERE
OUT_PATH = OUT_DIR / "model.safetensors"

REMOVE_BLOCKS = {2, 5, 8}
NUM_ORIG_BLOCKS = 12
BLOCK_RE = re.compile(r"^h\.(\d+)\.")


def main() -> None:
    if not IN_PATH.exists():
        sys.exit(f"input checkpoint not found: {IN_PATH}")

    state_dict = load_file(str(IN_PATH))

    # Build old->new block index mapping for surviving blocks, in original order.
    surviving_old_indices = [i for i in range(NUM_ORIG_BLOCKS) if i not in REMOVE_BLOCKS]
    old_to_new = {old: new for new, old in enumerate(surviving_old_indices)}

    out_state_dict = {}
    block_tensor_count_per_old_index = {}

    for key, tensor in state_dict.items():
        m = BLOCK_RE.match(key)
        if m is None:
            # Non-block tensor: copy unchanged.
            out_state_dict[key] = tensor
            continue

        old_idx = int(m.group(1))
        block_tensor_count_per_old_index[old_idx] = (
            block_tensor_count_per_old_index.get(old_idx, 0) + 1
        )

        if old_idx in REMOVE_BLOCKS:
            continue

        new_idx = old_to_new[old_idx]
        new_key = f"h.{new_idx}." + key[m.end() :]
        if new_key in out_state_dict:
            sys.exit(f"collision writing renumbered key {new_key!r} (from {key!r})")
        out_state_dict[new_key] = tensor

    # --- Required checks: fail loudly, no output written, if anything is off. ---

    # No tensor of removed-and-shifted-away blocks 9, 10, 11 should remain
    # under their old names (they were renumbered, not literally absent from
    # the source, but the *output* must not contain h.9./h.10./h.11.).
    for bad_idx in (9, 10, 11):
        bad_prefix = f"h.{bad_idx}."
        if any(k.startswith(bad_prefix) for k in out_state_dict):
            sys.exit(f"output still contains tensors with prefix {bad_prefix!r}")

    # Exactly 9 blocks remain: check a representative tensor per block index.
    attn_weight_keys = [
        k for k in out_state_dict if re.match(r"^h\.\d+\.attn\.c_attn\.weight$", k)
    ]
    if len(attn_weight_keys) != 9:
        sys.exit(
            f"expected exactly 9 surviving blocks, found {len(attn_weight_keys)} "
            f"matching h.<i>.attn.c_attn.weight"
        )

    present_new_indices = sorted(
        int(re.match(r"^h\.(\d+)\.", k).group(1)) for k in attn_weight_keys
    )
    if present_new_indices != list(range(9)):
        sys.exit(f"surviving block indices are not contiguous 0..8: {present_new_indices}")

    # Output must have exactly 121 tensors.
    if len(out_state_dict) != 121:
        sys.exit(f"expected 121 output tensors, got {len(out_state_dict)}")

    # Sanity: every removed block actually existed with 13 tensors in the input.
    for removed_idx in REMOVE_BLOCKS:
        count = block_tensor_count_per_old_index.get(removed_idx, 0)
        if count != 13:
            sys.exit(
                f"expected block {removed_idx} to have 13 tensors in input, found {count}"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out_state_dict, str(OUT_PATH))
    print(f"wrote {len(out_state_dict)} tensors to {OUT_PATH}")


if __name__ == "__main__":
    main()
