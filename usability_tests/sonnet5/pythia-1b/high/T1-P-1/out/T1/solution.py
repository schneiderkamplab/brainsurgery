"""
T1: Depth pruning with layer renumbering (Pythia-1B).

Remove transformer blocks 2, 6, 10, 14 from the 16-layer checkpoint and
renumber the surviving blocks so indices run 0..11 contiguously, preserving
original relative order. Non-block tensors are copied unchanged.
"""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_DIR = Path("out/T1")
OUTPUT_PATH = OUTPUT_DIR / "model.safetensors"

DROP_LAYERS = {2, 6, 10, 14}
NUM_ORIGINAL_LAYERS = 16
NUM_SURVIVING_LAYERS = NUM_ORIGINAL_LAYERS - len(DROP_LAYERS)

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not INPUT_PATH.exists():
        fail(f"missing input checkpoint: {INPUT_PATH}")

    tensors = load_file(str(INPUT_PATH))

    # Build old-index -> new-index mapping for surviving layers, preserving
    # original order.
    surviving_old_indices = [i for i in range(NUM_ORIGINAL_LAYERS) if i not in DROP_LAYERS]
    if len(surviving_old_indices) != NUM_SURVIVING_LAYERS:
        fail(
            f"expected {NUM_SURVIVING_LAYERS} surviving layers, "
            f"got {len(surviving_old_indices)}"
        )
    old_to_new = {old: new for new, old in enumerate(surviving_old_indices)}

    output: dict[str, "object"] = {}
    seen_names = set()

    for name, tensor in tensors.items():
        m = LAYER_RE.match(name)
        if m is None:
            # Non-block tensor: copy unchanged.
            new_name = name
        else:
            old_idx = int(m.group(1))
            if old_idx in DROP_LAYERS:
                continue
            new_idx = old_to_new[old_idx]
            new_name = f"gpt_neox.layers.{new_idx}." + name[m.end() :]

        if new_name in seen_names:
            fail(f"name collision while renumbering: {new_name!r} produced twice")
        seen_names.add(new_name)
        output[new_name] = tensor.clone().contiguous()

    # --- Required checks ---

    # No tensor of blocks 12, 13, 14, 15 (old numbering, which now would
    # collide with the new 0..11 range at 12/13, and 14/15 must be fully
    # gone regardless) should remain unaccounted for: verify no leftover
    # names exceed the new max index and that dropped layers are absent.
    max_new_idx = NUM_SURVIVING_LAYERS - 1
    for name in output:
        m = LAYER_RE.match(name)
        if m is None:
            continue
        idx = int(m.group(1))
        if idx > max_new_idx:
            fail(f"tensor {name!r} has out-of-range layer index {idx} (max {max_new_idx})")

    # Exactly 12 blocks remain, checked via a representative per-layer tensor.
    qkv_weight_count = sum(
        1 for name in output if re.match(r"^gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight$", name)
    )
    if qkv_weight_count != NUM_SURVIVING_LAYERS:
        fail(
            f"expected {NUM_SURVIVING_LAYERS} query_key_value.weight tensors, "
            f"got {qkv_weight_count}"
        )

    # Confirm blocks 12, 13, 14, 15 (original indices beyond the pruned set,
    # per the task's required check) are fully absent: since all surviving
    # new indices are 0..11, this is equivalent to max_new_idx == 11.
    if max_new_idx != 11:
        fail(f"max surviving layer index is {max_new_idx}, expected 11")

    expected_total = 244 - 4 * 15
    if len(output) != expected_total:
        fail(f"expected {expected_total} tensors in output, got {len(output)}")
    if expected_total != 184:
        fail(f"internal error: expected_total {expected_total} != 184")

    # Sanity: the 4 non-block tensors are present and unchanged (by name).
    non_block_names = {
        "gpt_neox.embed_in.weight",
        "embed_out.weight",
        "gpt_neox.final_layer_norm.weight",
        "gpt_neox.final_layer_norm.bias",
    }
    missing_non_block = non_block_names - set(output)
    if missing_non_block:
        fail(f"missing non-block tensors: {sorted(missing_non_block)}")
    for name in non_block_names:
        if not tensors[name].equal(output[name]):
            fail(f"non-block tensor {name!r} was modified")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(output, str(OUTPUT_PATH))

    print(f"Wrote {len(output)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
