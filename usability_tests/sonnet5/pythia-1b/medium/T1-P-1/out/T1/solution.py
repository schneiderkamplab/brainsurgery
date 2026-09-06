"""
T1: Depth pruning with layer renumbering (Pythia-1B).

Remove transformer blocks 2, 6, 10, 14 from a 16-layer Pythia-1B checkpoint
and renumber the remaining 12 blocks to run 0..11 contiguously, in original
order. All other tensors (embeddings, final layer norm, and the internals of
each surviving block) are copied unchanged.
"""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
IN_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUT_DIR = HERE
OUT_PATH = OUT_DIR / "model.safetensors"

DROP_LAYERS = {2, 6, 10, 14}
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.")

TENSORS_PER_LAYER = 15
NUM_LAYERS_IN = 16
NUM_LAYERS_OUT = 12
NON_LAYER_TENSORS = 4
EXPECTED_OUT_COUNT = NUM_LAYERS_OUT * TENSORS_PER_LAYER + NON_LAYER_TENSORS  # 184


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not IN_PATH.exists():
        fail(f"input checkpoint not found at {IN_PATH}")

    tensors = load_file(str(IN_PATH))

    # Build the old-index -> new-index mapping for surviving layers, in
    # original order, contiguous starting at 0.
    surviving_old_indices = sorted(
        i for i in range(NUM_LAYERS_IN) if i not in DROP_LAYERS
    )
    if len(surviving_old_indices) != NUM_LAYERS_OUT:
        fail(
            f"expected {NUM_LAYERS_OUT} surviving layers, got "
            f"{len(surviving_old_indices)}"
        )
    old_to_new = {old: new for new, old in enumerate(surviving_old_indices)}

    out_tensors = {}
    for name, tensor in tensors.items():
        m = LAYER_RE.match(name)
        if m is None:
            # Non-block tensor: copy unchanged.
            out_tensors[name] = tensor
            continue

        old_idx = int(m.group(1))
        if old_idx in DROP_LAYERS:
            continue  # drop this tensor entirely

        new_idx = old_to_new[old_idx]
        rest = name[m.end() :]
        new_name = f"gpt_neox.layers.{new_idx}.{rest}"
        if new_name in out_tensors:
            fail(f"name collision while renumbering: {new_name!r}")
        out_tensors[new_name] = tensor

    # --- Required checks: fail loudly, no output written on failure ---

    for bad in (12, 13, 14, 15):
        pattern = re.compile(rf"^gpt_neox\.layers\.{bad}\.")
        if any(pattern.match(n) for n in out_tensors):
            fail(f"tensor of dropped/old block {bad} still present")

    qkv_pattern = re.compile(r"^gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.weight$")
    qkv_layers = {int(m.group(1)) for n in out_tensors if (m := qkv_pattern.match(n))}
    if len(qkv_layers) != NUM_LAYERS_OUT:
        fail(f"expected exactly {NUM_LAYERS_OUT} blocks, found {len(qkv_layers)}")
    if qkv_layers != set(range(NUM_LAYERS_OUT)):
        fail(f"surviving block indices are not contiguous 0..{NUM_LAYERS_OUT - 1}: {sorted(qkv_layers)}")

    if len(out_tensors) != EXPECTED_OUT_COUNT:
        fail(
            f"output has {len(out_tensors)} tensors, expected {EXPECTED_OUT_COUNT}"
        )

    # Sanity: verify a couple of renumbering mappings and that values are
    # bit-identical copies (no accidental transformation).
    for old, new in old_to_new.items():
        old_name = f"gpt_neox.layers.{old}.input_layernorm.weight"
        new_name = f"gpt_neox.layers.{new}.input_layernorm.weight"
        if not tensors[old_name].equal(out_tensors[new_name]):
            fail(f"value mismatch after renumbering {old_name} -> {new_name}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_tensors = {k: v.contiguous() for k, v in out_tensors.items()}
    save_file(out_tensors, str(OUT_PATH))

    # Verify the written file round-trips to the same tensor count.
    check = load_file(str(OUT_PATH))
    if len(check) != EXPECTED_OUT_COUNT:
        fail(f"written file has {len(check)} tensors, expected {EXPECTED_OUT_COUNT}")

    print(f"Wrote {len(out_tensors)} tensors to {OUT_PATH}")


if __name__ == "__main__":
    main()
