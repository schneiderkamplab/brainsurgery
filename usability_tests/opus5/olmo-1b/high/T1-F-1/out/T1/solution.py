#!/usr/bin/env python
"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 blocks, renumbering the survivors.

Route: safetensors for bit-exact tensor I/O, torch-state-bridge for the key
rewrite. The rename is expressed as an ordered rule list; the bridge applies
the rules with collision detection, which is the guard against the hazard the
task describes (a shifted block overwriting a surviving one).

Every required check runs on the in-memory result *before* anything is
written, so a failure exits non-zero with no output file.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file
from torch_state_bridge import state_bridge, state_bridge_preview

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent.parent
SRC_DIR = SANDBOX / "inputs" / "base"
OUT_FILE = HERE / "model.safetensors"

N_OLD_BLOCKS = 16
DROP = (2, 6, 10, 14)
N_NEW_BLOCKS = N_OLD_BLOCKS - len(DROP)
EXPECTED_IN_TENSORS = 114
EXPECTED_OUT_TENSORS = 86
NON_BLOCK_KEYS = {"model.embed_tokens.weight", "lm_head.weight"}

BLOCK_RE = re.compile(r"^model\.layers\.(\d+)\.")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_sharded(src_dir: Path) -> dict:
    """Load every tensor named by the safetensors index, unchanged."""
    index = json.loads((src_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    state = {}
    for shard in sorted(set(weight_map.values())):
        for key, tensor in load_file(src_dir / shard).items():
            if key in state:
                fail(f"duplicate key across shards: {key}")
            state[key] = tensor
    missing = set(weight_map) - set(state)
    extra = set(state) - set(weight_map)
    if missing or extra:
        fail(f"shard contents disagree with index (missing={missing}, extra={extra})")
    return state


def build_rules(survivors: list[int]) -> str:
    """One rename rule per surviving block, in ascending old-index order.

    The engine applies rules in sequence, so ascending order is what keeps a
    renamed key from being caught again by a later rule.
    """
    lines = ["# old block prefix, new block prefix"]
    for new_idx, old_idx in enumerate(survivors):
        if new_idx != old_idx:
            lines.append(f"model.layers.{old_idx}., model.layers.{new_idx}.")
    return "\n".join(lines)


def main() -> None:
    state = load_sharded(SRC_DIR)
    if len(state) != EXPECTED_IN_TENSORS:
        fail(f"input has {len(state)} tensors, expected {EXPECTED_IN_TENSORS}")

    survivors = [i for i in range(N_OLD_BLOCKS) if i not in DROP]
    if len(survivors) != N_NEW_BLOCKS:
        fail(f"{len(survivors)} survivors, expected {N_NEW_BLOCKS}")

    # 1. drop every tensor of the removed blocks
    kept = {}
    dropped = 0
    for key, tensor in state.items():
        m = BLOCK_RE.match(key)
        if m is None:
            if key not in NON_BLOCK_KEYS:
                fail(f"unrecognised non-block tensor: {key}")
            kept[key] = tensor
            continue
        if int(m.group(1)) in DROP:
            dropped += 1
            continue
        kept[key] = tensor
    if dropped != len(DROP) * 7:
        fail(f"dropped {dropped} tensors, expected {len(DROP) * 7}")

    # 2. renumber the survivors, refusing any collision
    rules = build_rules(survivors)
    mapping, _unchanged, collisions = state_bridge_preview(kept, rules)
    if collisions:
        fail(f"rename collisions: {sorted(collisions)}")
    out = state_bridge(kept, rules, detect_collision=True)

    # Independent cross-check of the rename against the spec's own table,
    # rather than trusting the rule list that produced it.
    expected_keys = set()
    for key in state:
        m = BLOCK_RE.match(key)
        if m is None:
            expected_keys.add(key)
            continue
        old_idx = int(m.group(1))
        if old_idx in DROP:
            continue
        rest = key[m.end():]
        expected_keys.add(f"model.layers.{survivors.index(old_idx)}.{rest}")
    if set(out) != expected_keys:
        fail(
            "renamed key set disagrees with the expected mapping: "
            f"unexpected={sorted(set(out) - expected_keys)}, "
            f"missing={sorted(expected_keys - set(out))}"
        )

    # Values must survive the rename untouched: every output tensor is the very
    # object that came out of the source shard for its pre-rename name.
    inverse = {new: old for old, new in mapping.items()}
    for new_key, tensor in out.items():
        src_key = inverse[new_key]
        if tensor is not kept[src_key]:
            fail(f"{new_key} is not the tensor loaded from {src_key}")
        ref = state[src_key]
        if tensor.shape != ref.shape or tensor.dtype != ref.dtype:
            fail(f"{new_key}: shape/dtype changed from {src_key}")

    # --- Required checks (all before any write) ---
    indices = sorted({int(BLOCK_RE.match(k).group(1)) for k in out if BLOCK_RE.match(k)})

    stale = [k for k in out if BLOCK_RE.match(k) and int(BLOCK_RE.match(k).group(1)) >= N_NEW_BLOCKS]
    if stale:
        fail(f"tensors of blocks >= {N_NEW_BLOCKS} remain: {sorted(stale)}")

    q_projs = [k for k in out if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", k)]
    if len(q_projs) != N_NEW_BLOCKS:
        fail(f"{len(q_projs)} blocks remain, expected exactly {N_NEW_BLOCKS}")
    if indices != list(range(N_NEW_BLOCKS)):
        fail(f"block indices are not contiguous 0..{N_NEW_BLOCKS - 1}: {indices}")

    if len(out) != EXPECTED_OUT_TENSORS:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_OUT_TENSORS}")

    for key in NON_BLOCK_KEYS:
        if key not in out or out[key] is not state[key]:
            fail(f"non-block tensor {key} was not carried through unchanged")

    # --- Write, then re-read and confirm what actually landed on disk ---
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_FILE), metadata={"format": "pt"})
    try:
        back = load_file(OUT_FILE)
        if len(back) != EXPECTED_OUT_TENSORS:
            fail(f"written file has {len(back)} tensors, expected {EXPECTED_OUT_TENSORS}")
        if set(back) != expected_keys:
            fail("written file key set differs from the verified result")
        for key, tensor in back.items():
            ref = out[key]
            if tensor.dtype != ref.dtype or tensor.shape != ref.shape:
                fail(f"{key}: shape/dtype changed on write")
            if not tensor.equal(ref):
                fail(f"{key}: values changed on write")
    except SystemExit:
        OUT_FILE.unlink(missing_ok=True)
        raise

    print(f"OK: wrote {OUT_FILE} with {len(out)} tensors, blocks 0..{N_NEW_BLOCKS - 1}")
    print(f"    dropped blocks {DROP}; renumbering: " + ", ".join(
        f"{o}->{n}" for n, o in enumerate(survivors) if n != o))


if __name__ == "__main__":
    main()
