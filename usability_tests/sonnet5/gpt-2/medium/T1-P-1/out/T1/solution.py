"""
T1: Depth pruning with layer renumbering (GPT-2 124M).

Removes transformer blocks 2, 5, 8 from a 12-layer GPT-2 checkpoint and
renumbers the surviving blocks so indices run 0..8 without gaps:

    old 0 -> 0, old 1 -> 1, old 3 -> 2, old 4 -> 3, old 6 -> 4,
    old 7 -> 5, old 9 -> 6, old 10 -> 7, old 11 -> 8

Non-block tensors (wte, wpe, ln_f.*) pass through unchanged.
"""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
IN_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUT_DIR = HERE
OUT_PATH = OUT_DIR / "model.safetensors"

REMOVED_BLOCKS = {2, 5, 8}
NUM_ORIG_BLOCKS = 12
NUM_KEPT_BLOCKS = NUM_ORIG_BLOCKS - len(REMOVED_BLOCKS)
TENSORS_PER_BLOCK = 13
NUM_NON_BLOCK_TENSORS = 4
EXPECTED_INPUT_TENSORS = NUM_ORIG_BLOCKS * TENSORS_PER_BLOCK + NUM_NON_BLOCK_TENSORS
EXPECTED_OUTPUT_TENSORS = NUM_KEPT_BLOCKS * TENSORS_PER_BLOCK + NUM_NON_BLOCK_TENSORS

BLOCK_KEY_RE = re.compile(r"^h\.(\d+)\.")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not IN_PATH.exists():
        fail(f"input checkpoint not found at {IN_PATH}")

    state_dict = load_file(str(IN_PATH))

    if len(state_dict) != EXPECTED_INPUT_TENSORS:
        fail(
            f"expected {EXPECTED_INPUT_TENSORS} tensors in input, "
            f"found {len(state_dict)}"
        )

    # Build the old->new block index mapping for surviving blocks, in order.
    surviving_old_indices = [
        i for i in range(NUM_ORIG_BLOCKS) if i not in REMOVED_BLOCKS
    ]
    old_to_new = {old: new for new, old in enumerate(surviving_old_indices)}

    new_state_dict = {}
    seen_new_block_keys = set()

    for key, tensor in state_dict.items():
        m = BLOCK_KEY_RE.match(key)
        if m is None:
            # Non-block tensor: wte.weight, wpe.weight, ln_f.weight, ln_f.bias
            new_state_dict[key] = tensor
            continue

        old_idx = int(m.group(1))
        if old_idx in REMOVED_BLOCKS:
            continue  # drop this tensor entirely

        new_idx = old_to_new[old_idx]
        rest = key[m.end():]
        new_key = f"h.{new_idx}.{rest}"

        if new_key in new_state_dict:
            fail(f"collision while renumbering: {new_key} already exists")

        new_state_dict[new_key] = tensor
        seen_new_block_keys.add(new_key)

    # --- Required checks ---

    # No tensor of blocks 9, 10, 11 remains in the output: those old indices
    # were reassigned to 6, 7, 8 during renumbering, so plain block-9/10/11
    # keys must not appear in the result.
    for stale in (9, 10, 11):
        prefix = f"h.{stale}."
        if any(k.startswith(prefix) for k in new_state_dict):
            fail(f"tensor with stale block index {stale} present in output")

    # Exactly 9 blocks remain.
    c_attn_count = sum(
        1 for k in new_state_dict if re.match(r"^h\.\d+\.attn\.c_attn\.weight$", k)
    )
    if c_attn_count != NUM_KEPT_BLOCKS:
        fail(f"expected {NUM_KEPT_BLOCKS} surviving blocks, found {c_attn_count}")

    block_indices_present = sorted(
        {int(BLOCK_KEY_RE.match(k).group(1)) for k in new_state_dict if BLOCK_KEY_RE.match(k)}
    )
    if block_indices_present != list(range(NUM_KEPT_BLOCKS)):
        fail(f"block indices not contiguous 0..{NUM_KEPT_BLOCKS - 1}: {block_indices_present}")

    # Non-block tensors unchanged (same 4 keys, same tensors by identity from load).
    non_block_keys = {k for k in state_dict if BLOCK_KEY_RE.match(k) is None}
    if len(non_block_keys) != NUM_NON_BLOCK_TENSORS:
        fail(f"expected {NUM_NON_BLOCK_TENSORS} non-block tensors, found {len(non_block_keys)}")
    for k in non_block_keys:
        if k not in new_state_dict or new_state_dict[k].data_ptr() != state_dict[k].data_ptr():
            fail(f"non-block tensor {k} was not preserved unchanged")

    # Output has exactly 121 tensors.
    if len(new_state_dict) != EXPECTED_OUTPUT_TENSORS:
        fail(
            f"expected {EXPECTED_OUTPUT_TENSORS} tensors in output, "
            f"found {len(new_state_dict)}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Tensors from load_file share no cross-tensor storage in this checkpoint,
    # but call .contiguous() defensively since safetensors requires it.
    to_save = {k: v.contiguous() for k, v in new_state_dict.items()}
    save_file(to_save, str(OUT_PATH))

    print(f"Wrote {len(to_save)} tensors to {OUT_PATH}")


if __name__ == "__main__":
    main()
