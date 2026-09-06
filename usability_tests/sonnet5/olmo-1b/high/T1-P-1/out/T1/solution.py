"""
T1: Depth pruning with layer renumbering (OLMo-1B-0724-hf).

Removes transformer blocks 2, 6, 10, 14 from a 16-layer checkpoint and
renumbers the remaining 12 blocks to be contiguous (0..11), preserving
original relative order. Non-block tensors are copied unchanged.

Fails loudly (raises / non-zero exit, no output written) if any required
check does not hold.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # out/T1 -> out -> sandbox root
INPUT_DIR = REPO_ROOT / "inputs" / "base"
OUTPUT_PATH = HERE / "model.safetensors"

NUM_LAYERS_IN = 16
BLOCKS_TO_REMOVE = {2, 6, 10, 14}
TENSORS_PER_BLOCK = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
NON_BLOCK_TENSORS = ("model.embed_tokens.weight", "lm_head.weight")

LAYER_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def load_all_tensors() -> dict[str, torch.Tensor]:
    index_path = INPUT_DIR / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_files = sorted(set(weight_map.values()))
    tensors: dict[str, torch.Tensor] = {}
    for shard_file in shard_files:
        shard = load_file(str(INPUT_DIR / shard_file))
        tensors.update(shard)

    # Sanity: every key the index claims must actually be present, and the
    # loaded set must match the index exactly (no stray keys).
    assert set(tensors.keys()) == set(weight_map.keys()), (
        "loaded tensor keys do not match model.safetensors.index.json weight_map"
    )
    return tensors


def main() -> None:
    tensors = load_all_tensors()

    if len(tensors) != 114:
        raise AssertionError(f"expected 114 input tensors, got {len(tensors)}")

    # Discover the block indices actually present and validate the expected
    # shape of the checkpoint before doing any surgery.
    block_indices: set[int] = set()
    for key in tensors:
        m = LAYER_KEY_RE.match(key)
        if m:
            block_indices.add(int(m.group(1)))
    if block_indices != set(range(NUM_LAYERS_IN)):
        raise AssertionError(
            f"expected blocks 0..{NUM_LAYERS_IN - 1}, found {sorted(block_indices)}"
        )
    for i in range(NUM_LAYERS_IN):
        for rest in TENSORS_PER_BLOCK:
            key = f"model.layers.{i}.{rest}"
            if key not in tensors:
                raise AssertionError(f"missing expected tensor {key!r} in input")
    for key in NON_BLOCK_TENSORS:
        if key not in tensors:
            raise AssertionError(f"missing expected non-block tensor {key!r} in input")
    if len(tensors) != NUM_LAYERS_IN * len(TENSORS_PER_BLOCK) + len(NON_BLOCK_TENSORS):
        raise AssertionError("unexpected extra tensors in input checkpoint")

    for key, t in tensors.items():
        if t.dtype != torch.float32:
            raise AssertionError(f"expected float32, {key!r} has dtype {t.dtype}")

    # Build the old -> new block index mapping: surviving blocks keep their
    # original relative order and are renumbered contiguously from 0.
    surviving_old_indices = [i for i in range(NUM_LAYERS_IN) if i not in BLOCKS_TO_REMOVE]
    old_to_new = {old: new for new, old in enumerate(surviving_old_indices)}
    assert len(surviving_old_indices) == 12

    expected_mapping = {
        0: 0,
        1: 1,
        3: 2,
        4: 3,
        5: 4,
        7: 5,
        8: 6,
        9: 7,
        11: 8,
        12: 9,
        13: 10,
        15: 11,
    }
    if old_to_new != expected_mapping:
        raise AssertionError(f"renumbering mismatch: {old_to_new} != {expected_mapping}")

    # Build the output state dict fresh (never mutate the input dict's keys
    # in place), so there is no possibility of a rename collision regardless
    # of processing order.
    output: dict[str, torch.Tensor] = {}
    for key in NON_BLOCK_TENSORS:
        output[key] = tensors[key]

    for old_idx, new_idx in old_to_new.items():
        for rest in TENSORS_PER_BLOCK:
            src_key = f"model.layers.{old_idx}.{rest}"
            dst_key = f"model.layers.{new_idx}.{rest}"
            if dst_key in output:
                raise AssertionError(f"collision writing {dst_key!r}")
            output[dst_key] = tensors[src_key].contiguous()

    if len(output) != 86:
        raise AssertionError(f"expected 86 output tensors, got {len(output)}")

    # Required checks -----------------------------------------------------
    out_block_indices: set[int] = set()
    for key in output:
        m = LAYER_KEY_RE.match(key)
        if m:
            out_block_indices.add(int(m.group(1)))

    for removed in (12, 13, 14, 15):
        if removed in out_block_indices:
            raise AssertionError(f"tensor of block {removed} remains in output")

    q_proj_count = sum(1 for k in output if k.endswith("self_attn.q_proj.weight"))
    if q_proj_count != 12:
        raise AssertionError(f"expected exactly 12 blocks, found {q_proj_count}")

    if out_block_indices != set(range(12)):
        raise AssertionError(f"expected contiguous blocks 0..11, got {sorted(out_block_indices)}")

    if len(output) != 86:
        raise AssertionError(f"expected output to have exactly 86 tensors, got {len(output)}")

    # Value/shape/dtype fidelity spot-check against the sources.
    for old_idx, new_idx in old_to_new.items():
        for rest in TENSORS_PER_BLOCK:
            src = tensors[f"model.layers.{old_idx}.{rest}"]
            dst = output[f"model.layers.{new_idx}.{rest}"]
            if dst.shape != src.shape or dst.dtype != src.dtype:
                raise AssertionError(f"shape/dtype mismatch for old block {old_idx} -> {new_idx}")
            if not torch.equal(dst, src):
                raise AssertionError(f"value mismatch for old block {old_idx} -> {new_idx}")
    for key in NON_BLOCK_TENSORS:
        if not torch.equal(output[key], tensors[key]):
            raise AssertionError(f"non-block tensor {key!r} was modified")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, str(OUTPUT_PATH))
    print(f"Wrote {len(output)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # fail loudly, no partial output
        if OUTPUT_PATH.exists():
            OUTPUT_PATH.unlink()
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
