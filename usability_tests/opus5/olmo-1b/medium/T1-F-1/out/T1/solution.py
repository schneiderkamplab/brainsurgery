#!/usr/bin/env python
"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 blocks with renumbering.

Tools: safetensors (I/O), torch-state-bridge (rule-based key rewriting with
collision detection), torch (tensors).

Strategy:
  1. Load the sharded input, verify it against its index.
  2. Drop every tensor of blocks 2, 6, 10, 14 (whole-block match, 7 each).
  3. Renumber survivors with torch-state-bridge, in two passes through a
     marker namespace (``model.layers.NEW<i>.``) so that no rewritten key can
     ever be re-matched by a later rule and no destination can collide with a
     not-yet-moved source. Collision detection stays on for both passes.
  4. Enforce the required checks. Only then write the output, via a temp file
     + atomic rename so a failure leaves no output behind.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch_state_bridge import state_bridge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # sandbox root
SRC_DIR = ROOT / "inputs" / "base"
OUT_PATH = HERE / "model.safetensors"

DROP = [2, 6, 10, 14]
N_IN_LAYERS = 16
N_OUT_LAYERS = 12
TENSORS_PER_BLOCK = 7
N_NON_BLOCK = 2
N_OUT_TENSORS = N_OUT_LAYERS * TENSORS_PER_BLOCK + N_NON_BLOCK   # 86

BLOCK_RE = re.compile(r"^model\.layers\.(\d+)\.")


def fail(msg: str) -> None:
    raise SystemExit(f"CHECK FAILED: {msg}")


def block_of(key: str) -> int | None:
    m = BLOCK_RE.match(key)
    return int(m.group(1)) if m else None


def load_input() -> dict[str, torch.Tensor]:
    index = json.loads((SRC_DIR / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    sd: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        part = load_file(SRC_DIR / shard)
        overlap = sd.keys() & part.keys()
        if overlap:
            fail(f"key present in more than one shard: {sorted(overlap)}")
        sd.update(part)
    if sd.keys() != weight_map.keys():
        fail("loaded key set does not match model.safetensors.index.json")
    if len(sd) != N_IN_LAYERS * TENSORS_PER_BLOCK + N_NON_BLOCK:
        fail(f"expected 114 input tensors, got {len(sd)}")
    return sd


def main() -> None:
    src = load_input()

    by_block: dict[int, list[str]] = defaultdict(list)
    non_block: list[str] = []
    for k in src:
        b = block_of(k)
        (by_block[b] if b is not None else non_block).append(k)

    if sorted(by_block) != list(range(N_IN_LAYERS)):
        fail(f"input blocks are not 0..15: {sorted(by_block)}")
    if len(non_block) != N_NON_BLOCK:
        fail(f"expected {N_NON_BLOCK} non-block tensors, got {sorted(non_block)}")
    for b, keys in by_block.items():
        if len(keys) != TENSORS_PER_BLOCK:
            fail(f"block {b} has {len(keys)} tensors, expected {TENSORS_PER_BLOCK}")

    # --- 1. drop whole blocks -------------------------------------------------
    kept = {k: v for k, v in src.items() if block_of(k) not in DROP}
    if len(kept) != len(src) - len(DROP) * TENSORS_PER_BLOCK:
        fail("dropping blocks removed the wrong number of tensors")
    if any(block_of(k) in DROP for k in kept):
        fail("a tensor of a dropped block survived")

    survivors = [b for b in range(N_IN_LAYERS) if b not in DROP]
    if len(survivors) != N_OUT_LAYERS:
        fail(f"{len(survivors)} blocks survive, expected {N_OUT_LAYERS}")
    mapping = {old: new for new, old in enumerate(survivors)}

    # --- 2. renumber via torch-state-bridge, through a marker namespace -------
    to_marker = "\n".join(
        f"model.layers.{old}., model.layers.NEW{new}." for old, new in mapping.items()
    )
    from_marker = "\n".join(
        f"model.layers.NEW{new}., model.layers.{new}." for new in mapping.values()
    )
    staged = state_bridge(kept, to_marker, detect_collision=True)
    out = state_bridge(staged, from_marker, detect_collision=True)

    if any("NEW" in k for k in out):
        fail("a marker key survived the second rewrite pass")

    # --- 3. required checks ---------------------------------------------------
    out_blocks = sorted({block_of(k) for k in out if block_of(k) is not None})
    if any(b >= N_OUT_LAYERS for b in out_blocks):
        fail(f"tensors of blocks >= {N_OUT_LAYERS} remain: {out_blocks}")
    if out_blocks != list(range(N_OUT_LAYERS)):
        fail(f"block indices are not contiguous 0..{N_OUT_LAYERS - 1}: {out_blocks}")
    n_q = sum(
        1
        for k in out
        if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", k)
    )
    if n_q != N_OUT_LAYERS:
        fail(f"{n_q} q_proj tensors, expected {N_OUT_LAYERS}")
    if len(out) != N_OUT_TENSORS:
        fail(f"output has {len(out)} tensors, expected {N_OUT_TENSORS}")

    # every output tensor must be the identical object from its expected source
    for old, new in mapping.items():
        for old_key in by_block[old]:
            new_key = old_key.replace(f"model.layers.{old}.", f"model.layers.{new}.", 1)
            if new_key not in out:
                fail(f"missing {new_key} (from {old_key})")
            if out[new_key] is not src[old_key]:
                fail(f"{new_key} does not carry the values of {old_key}")
            t, s = out[new_key], src[old_key]
            if t.shape != s.shape or t.dtype != s.dtype:
                fail(f"{new_key}: shape/dtype changed ({s.shape}/{s.dtype} -> {t.shape}/{t.dtype})")
    for k in non_block:
        if k not in out or out[k] is not src[k]:
            fail(f"non-block tensor {k} was altered or lost")

    # --- 4. write, atomically -------------------------------------------------
    tmp = OUT_PATH.with_suffix(".safetensors.tmp")
    save_file({k: v.contiguous() for k, v in out.items()}, tmp)
    os.replace(tmp, OUT_PATH)

    print(f"wrote {OUT_PATH} with {len(out)} tensors, blocks 0..{out_blocks[-1]}")
    print("mapping: " + ", ".join(f"{o}->{n}" for o, n in mapping.items()))


if __name__ == "__main__":
    sys.exit(main())
