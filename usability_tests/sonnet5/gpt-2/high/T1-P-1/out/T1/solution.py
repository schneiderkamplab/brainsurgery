"""
T1: Depth pruning with layer renumbering (GPT-2 124M).

Remove transformer blocks 2, 5, 8 from a 12-layer GPT-2 checkpoint and
renumber the surviving blocks so indices run 0..8 with no gaps:

    old: 0 1 3 4 6 7 9 10 11
    new: 0 1 2 3 4 5 6 7  8

Non-block tensors (wte, wpe, ln_f.*) pass through unchanged.
"""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
INPUT_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUTPUT_DIR = HERE
OUTPUT_PATH = OUTPUT_DIR / "model.safetensors"

REMOVE_BLOCKS = {2, 5, 8}
# Explicit old -> new mapping for surviving blocks, in original order.
BLOCK_MAP = {0: 0, 1: 1, 3: 2, 4: 3, 6: 4, 7: 5, 9: 6, 10: 7, 11: 8}

# Anchored, escaped-dot pattern: matches "h.<digits>." at the start of a key
# and nothing else (so h.1. never matches h.10. or h.11.).
BLOCK_KEY_RE = re.compile(r"^h\.(\d+)\.")


def block_index(key: str) -> int | None:
    m = BLOCK_KEY_RE.match(key)
    if m is None:
        return None
    return int(m.group(1))


def main() -> None:
    if not INPUT_PATH.is_file():
        sys.exit(f"input checkpoint not found: {INPUT_PATH}")

    state_dict = load_file(str(INPUT_PATH))

    if len(state_dict) != 160:
        sys.exit(f"expected 160 input tensors, found {len(state_dict)}")

    non_block_keys = [k for k in state_dict if block_index(k) is None]
    if len(non_block_keys) != 4:
        sys.exit(f"expected 4 non-block tensors, found {len(non_block_keys)}: {non_block_keys}")

    new_state_dict: dict[str, "torch.Tensor"] = {}

    # Non-block tensors pass through unchanged.
    for k in non_block_keys:
        new_state_dict[k] = state_dict[k]

    # Renumber surviving blocks. Iterate old indices in ascending order so
    # any accidental collision (two old keys mapping to the same new key)
    # is caught rather than silently overwritten.
    for old_idx, new_idx in sorted(BLOCK_MAP.items()):
        prefix_old = f"h.{old_idx}."
        matched_any = False
        for key, tensor in state_dict.items():
            idx = block_index(key)
            if idx != old_idx:
                continue
            matched_any = True
            rest = key[len(prefix_old):]
            new_key = f"h.{new_idx}.{rest}"
            if new_key in new_state_dict:
                sys.exit(f"collision: {new_key} already written (from old block {old_idx})")
            new_state_dict[new_key] = tensor.contiguous()
        if not matched_any:
            sys.exit(f"no tensors found for expected surviving block h.{old_idx}.*")

    # --- Required checks ---

    # Exactly the block indices 0..8 are present in the output, so old
    # blocks 9, 10, 11 (renumbered to 6, 7, 8) and removed blocks 2, 5, 8
    # cannot linger under a stale name, and no gap or duplicate slipped in.
    output_block_indices = set()
    for key in new_state_dict:
        idx = block_index(key)
        if idx is not None:
            output_block_indices.add(idx)

    if output_block_indices != set(range(9)):
        sys.exit(f"expected block indices 0..8 in output, found {sorted(output_block_indices)}")

    # Exactly 9 blocks remain, e.g. exactly 9 tensors match h.<i>.attn.c_attn.weight
    c_attn_weight_count = sum(
        1 for k in new_state_dict if re.match(r"^h\.\d+\.attn\.c_attn\.weight$", k)
    )
    if c_attn_weight_count != 9:
        sys.exit(f"expected 9 h.<i>.attn.c_attn.weight tensors, found {c_attn_weight_count}")

    if len(new_state_dict) != 121:
        sys.exit(f"expected 121 output tensors, found {len(new_state_dict)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(new_state_dict, str(OUTPUT_PATH))
    print(f"wrote {len(new_state_dict)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
