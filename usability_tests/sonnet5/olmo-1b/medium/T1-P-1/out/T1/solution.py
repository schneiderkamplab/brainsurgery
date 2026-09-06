"""
T1: Depth pruning with layer renumbering (OLMo-1B-0724-hf).

Removes transformer blocks 2, 6, 10, 14 from the 16-layer checkpoint and
renumbers the surviving blocks in original order so indices run 0..11
without gaps. Writes a single merged safetensors file.
"""

import json
import os
import re
import sys

from safetensors.torch import load_file, save_file

INPUT_DIR = "inputs/base"
OUTPUT_DIR = "out/T1"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "model.safetensors")

REMOVE_BLOCKS = {2, 6, 10, 14}
NUM_ORIGINAL_BLOCKS = 16
EXPECTED_SURVIVING_BLOCKS = NUM_ORIGINAL_BLOCKS - len(REMOVE_BLOCKS)  # 12
EXPECTED_TOTAL_TENSORS = 86

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def load_all_tensors(input_dir: str) -> dict:
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_files = sorted(set(weight_map.values()))
    tensors = {}
    for shard_file in shard_files:
        shard = load_file(os.path.join(input_dir, shard_file))
        tensors.update(shard)

    # sanity: every key the index claims should exist, and every key we
    # loaded should be accounted for by the index.
    if set(tensors.keys()) != set(weight_map.keys()):
        raise RuntimeError(
            "Loaded tensor keys do not match the index's weight_map; "
            f"missing={set(weight_map) - set(tensors)}, "
            f"extra={set(tensors) - set(weight_map)}"
        )
    return tensors


def main() -> None:
    tensors = load_all_tensors(INPUT_DIR)

    # Build the old->new block index mapping from the surviving blocks, in
    # ascending original order.
    surviving_old_indices = [
        i for i in range(NUM_ORIGINAL_BLOCKS) if i not in REMOVE_BLOCKS
    ]
    if len(surviving_old_indices) != EXPECTED_SURVIVING_BLOCKS:
        raise RuntimeError(
            f"Expected {EXPECTED_SURVIVING_BLOCKS} surviving blocks, "
            f"got {len(surviving_old_indices)}"
        )
    old_to_new = {old: new for new, old in enumerate(surviving_old_indices)}

    output_tensors = {}
    for name, tensor in tensors.items():
        m = LAYER_RE.match(name)
        if m is None:
            # Non-block tensor (embed_tokens, lm_head): carry over unchanged.
            output_tensors[name] = tensor
            continue

        old_idx = int(m.group(1))
        if old_idx in REMOVE_BLOCKS:
            # Drop this tensor entirely.
            continue

        new_idx = old_to_new[old_idx]
        new_name = f"model.layers.{new_idx}." + name[m.end() :]
        if new_name in output_tensors:
            raise RuntimeError(
                f"Collision: {new_name} already produced (from renaming "
                f"{name}). Renumbering is not injective."
            )
        output_tensors[new_name] = tensor

    # --- Required checks: fail loudly, no output written, if violated. ---

    # No tensor of removed (originally-highest) blocks 12, 13, 14, 15 remains.
    for forbidden in (12, 13, 14, 15):
        prefix = f"model.layers.{forbidden}."
        leaked = [k for k in output_tensors if k.startswith(prefix)]
        if leaked:
            raise AssertionError(
                f"Tensors of block {forbidden} leaked into output: {leaked}"
            )

    # Exactly 12 blocks remain.
    q_proj_keys = [
        k for k in output_tensors if re.match(r"^model\.layers\.\d+\.self_attn\.q_proj\.weight$", k)
    ]
    if len(q_proj_keys) != EXPECTED_SURVIVING_BLOCKS:
        raise AssertionError(
            f"Expected {EXPECTED_SURVIVING_BLOCKS} q_proj tensors (one per "
            f"surviving block), got {len(q_proj_keys)}: {sorted(q_proj_keys)}"
        )
    present_new_indices = sorted(
        int(re.match(r"^model\.layers\.(\d+)\.", k).group(1)) for k in q_proj_keys
    )
    if present_new_indices != list(range(EXPECTED_SURVIVING_BLOCKS)):
        raise AssertionError(
            f"Surviving block indices are not contiguous 0..{EXPECTED_SURVIVING_BLOCKS - 1}: "
            f"{present_new_indices}"
        )

    # Total tensor count.
    if len(output_tensors) != EXPECTED_TOTAL_TENSORS:
        raise AssertionError(
            f"Expected {EXPECTED_TOTAL_TENSORS} output tensors, got "
            f"{len(output_tensors)}"
        )

    # The 2 non-block tensors must be present and unchanged.
    for fixed_name in ("model.embed_tokens.weight", "lm_head.weight"):
        if fixed_name not in output_tensors:
            raise AssertionError(f"Missing expected non-block tensor: {fixed_name}")
        if not output_tensors[fixed_name].equal(tensors[fixed_name]):
            raise AssertionError(f"Non-block tensor {fixed_name} was modified")

    # Every surviving block must still have exactly 7 tensors, and every
    # value/shape/dtype for renamed tensors must exactly match its source.
    per_block_suffixes = {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
    for old_idx, new_idx in old_to_new.items():
        for suffix in per_block_suffixes:
            old_name = f"model.layers.{old_idx}.{suffix}"
            new_name = f"model.layers.{new_idx}.{suffix}"
            if new_name not in output_tensors:
                raise AssertionError(f"Missing {new_name} (renamed from {old_name})")
            old_t = tensors[old_name]
            new_t = output_tensors[new_name]
            if old_t.shape != new_t.shape or old_t.dtype != new_t.dtype:
                raise AssertionError(
                    f"Shape/dtype mismatch for {new_name} vs {old_name}: "
                    f"{new_t.shape}/{new_t.dtype} vs {old_t.shape}/{old_t.dtype}"
                )
            if not old_t.equal(new_t):
                raise AssertionError(f"Value mismatch for {new_name} vs {old_name}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Tensors loaded via load_file are already contiguous; make sure of it
    # anyway before saving (safetensors requires contiguous, non-shared
    # storage tensors).
    to_save = {k: v.contiguous() for k, v in output_tensors.items()}
    save_file(to_save, OUTPUT_PATH)

    print(f"Wrote {len(to_save)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
