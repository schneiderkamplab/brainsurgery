"""
T1: Depth pruning with layer renumbering (GPT-2 124M).

Remove transformer blocks 2, 5, 8 and renumber the surviving blocks so
indices run 0..8 without gaps, preserving relative order. Non-block
tensors (wte, wpe, ln_f.*) are copied unchanged.
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
NUM_ORIGINAL_BLOCKS = 12
TENSORS_PER_BLOCK = 13
NUM_NON_BLOCK_TENSORS = 4

BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def main() -> None:
    if not INPUT_PATH.is_file():
        print(f"error: input checkpoint not found at {INPUT_PATH}", file=sys.stderr)
        sys.exit(1)

    state_dict = load_file(str(INPUT_PATH))

    # Discover which block indices actually appear in the input, so we
    # renumber based on reality rather than an assumption of exactly 0..11.
    old_indices = set()
    for key in state_dict:
        m = BLOCK_RE.match(key)
        if m:
            old_indices.add(int(m.group(1)))

    expected_old_indices = set(range(NUM_ORIGINAL_BLOCKS))
    if old_indices != expected_old_indices:
        print(
            f"error: expected blocks {sorted(expected_old_indices)}, "
            f"found {sorted(old_indices)}",
            file=sys.stderr,
        )
        sys.exit(1)

    surviving_old = sorted(old_indices - REMOVE_BLOCKS)
    old_to_new = {old: new for new, old in enumerate(surviving_old)}

    output_state_dict = {}
    for key, tensor in state_dict.items():
        m = BLOCK_RE.match(key)
        if m:
            old_idx = int(m.group(1))
            rest = m.group(2)
            if old_idx in REMOVE_BLOCKS:
                continue
            new_key = f"h.{old_to_new[old_idx]}.{rest}"
            if new_key in output_state_dict:
                print(f"error: key collision writing {new_key}", file=sys.stderr)
                sys.exit(1)
            output_state_dict[new_key] = tensor
        else:
            # Non-block tensor: copy unchanged.
            if key in output_state_dict:
                print(f"error: key collision writing {key}", file=sys.stderr)
                sys.exit(1)
            output_state_dict[key] = tensor

    # --- Required checks ---

    # No tensor named "h.9.*", "h.10.*" or "h.11.*" remains in the output.
    # (Old blocks 9, 10, 11 are remapped to new indices 6, 7, 8 respectively,
    # so this check confirms the old, pre-renumbering names are gone.)
    for idx in (9, 10, 11):
        prefix = f"h.{idx}."
        if any(k.startswith(prefix) for k in output_state_dict):
            print(f"error: output still contains block {idx}", file=sys.stderr)
            sys.exit(1)

    # Exactly 9 blocks remain.
    attn_c_attn_keys = [
        k for k in output_state_dict if re.match(r"^h\.\d+\.attn\.c_attn\.weight$", k)
    ]
    if len(attn_c_attn_keys) != 9:
        print(
            f"error: expected 9 surviving blocks, found {len(attn_c_attn_keys)}",
            file=sys.stderr,
        )
        sys.exit(1)

    surviving_block_indices = sorted(
        int(re.match(r"^h\.(\d+)\.attn\.c_attn\.weight$", k).group(1))
        for k in attn_c_attn_keys
    )
    if surviving_block_indices != list(range(9)):
        print(
            f"error: surviving block indices are not contiguous 0..8: "
            f"{surviving_block_indices}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Exactly 121 tensors total.
    if len(output_state_dict) != 121:
        print(
            f"error: expected 121 tensors in output, found {len(output_state_dict)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Sanity: 9 blocks * 13 tensors/block + 4 non-block tensors == 121.
    expected_total = 9 * TENSORS_PER_BLOCK + NUM_NON_BLOCK_TENSORS
    assert expected_total == 121

    # Non-block tensors present and unchanged (identity check by name + equality).
    non_block_names = {"wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"}
    for name in non_block_names:
        if name not in output_state_dict:
            print(f"error: missing non-block tensor {name}", file=sys.stderr)
            sys.exit(1)
        if not torch_equal(output_state_dict[name], state_dict[name]):
            print(f"error: non-block tensor {name} was modified", file=sys.stderr)
            sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(output_state_dict, str(OUTPUT_PATH))
    print(f"wrote {OUTPUT_PATH} with {len(output_state_dict)} tensors")


def torch_equal(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool((a == b).all())


if __name__ == "__main__":
    main()
