"""T1: depth-prune Pythia-1B from 16 to 12 blocks, renumbering survivors.

Tools: safetensors (I/O), torch-state-bridge (rule-based key rewriting with
collision detection), torch (tensors). No brainsurgery.

Strategy: drop every tensor of blocks DROP, then renumber the survivors with
an explicit ordered rule list. Rules are emitted in ascending source order so
no rewritten key can be re-matched by a later rule; torch-state-bridge's
collision detector is the backstop that catches it if that reasoning is wrong.
All required checks run before anything is written; on failure the script
exits non-zero and leaves no output file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file
from torch_state_bridge import state_bridge, state_bridge_preview

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T1/model.safetensors")

N_LAYERS_IN = 16
DROP = [2, 6, 10, 14]
KEEP = [i for i in range(N_LAYERS_IN) if i not in DROP]  # 12 survivors
N_LAYERS_OUT = len(KEEP)
EXPECTED_TENSORS = 184

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.")


def die(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def block_index(key: str) -> int | None:
    m = LAYER_RE.match(key)
    return int(m.group(1)) if m else None


def main() -> None:
    sd = load_file(str(SRC))
    print(f"loaded {len(sd)} tensors from {SRC}")

    # --- sanity on the input, so a surprising input cannot be silently pruned ---
    present = sorted({i for i in map(block_index, sd) if i is not None})
    if present != list(range(N_LAYERS_IN)):
        die(f"input blocks are {present}, expected 0..{N_LAYERS_IN - 1}")

    # --- 1. drop whole blocks by pattern -----------------------------------
    pruned = {k: v for k, v in sd.items() if block_index(k) not in DROP}
    dropped = len(sd) - len(pruned)
    if dropped != len(DROP) * 15:
        die(f"dropped {dropped} tensors, expected {len(DROP) * 15}")
    print(f"dropped blocks {DROP}: {dropped} tensors, {len(pruned)} remain")

    # --- 2. renumber survivors ---------------------------------------------
    # One literal rule per block whose index changes. Ascending source order:
    # every destination index is strictly below every later source index, so a
    # key rewritten by rule i is never touched again by rule j > i.
    rules = "\n".join(
        f"gpt_neox.layers.{old}., gpt_neox.layers.{new}."
        for new, old in enumerate(KEEP)
        if new != old
    )
    print("renumber rules:\n" + rules)

    mapping, unchanged, collisions = state_bridge_preview(pruned, rules)
    if collisions:
        die(f"renumbering collisions: {sorted(collisions)}")

    # state_bridge raises KeyError on collision; keep it enabled as a backstop.
    out = state_bridge(pruned, rules, detect_collision=True)

    # The mapping must be exactly the one the task specifies, key for key.
    expected = {}
    for k, v in pruned.items():
        i = block_index(k)
        expected[k if i is None else LAYER_RE.sub(f"gpt_neox.layers.{KEEP.index(i)}.", k, count=1)] = v
    if set(out) != set(expected):
        die(f"renamed key set differs from expected; symmetric difference: "
            f"{sorted(set(out) ^ set(expected))[:10]}")
    for k in out:
        if out[k] is not expected[k]:
            die(f"tensor identity changed for {k}")
    print(f"renamed {len(mapping)} keys, {len(unchanged)} unchanged")

    # --- 3. required checks, all before writing ----------------------------
    stale = sorted(k for k in out if block_index(k) in range(N_LAYERS_OUT, N_LAYERS_IN))
    if stale:
        die(f"tensors of removed block indices 12..15 remain: {stale}")

    qkv = [k for k in out if re.fullmatch(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", k)]
    if len(qkv) != N_LAYERS_OUT:
        die(f"{len(qkv)} query_key_value.weight tensors, expected {N_LAYERS_OUT}")

    blocks = sorted({i for i in map(block_index, out) if i is not None})
    if blocks != list(range(N_LAYERS_OUT)):
        die(f"surviving block indices are {blocks}, expected 0..{N_LAYERS_OUT - 1}")

    per_block: dict[int, int] = {}
    for k in out:
        i = block_index(k)
        if i is not None:
            per_block[i] = per_block.get(i, 0) + 1
    if set(per_block.values()) != {15}:
        die(f"blocks do not all have 15 tensors: {sorted(per_block.items())}")

    non_block = sorted(k for k in out if block_index(k) is None)
    if non_block != ["embed_out.weight", "gpt_neox.embed_in.weight",
                     "gpt_neox.final_layer_norm.bias", "gpt_neox.final_layer_norm.weight"]:
        die(f"non-block tensors changed: {non_block}")
    for k in non_block:
        if out[k] is not sd[k]:
            die(f"non-block tensor {k} was not passed through unchanged")

    if len(out) != EXPECTED_TENSORS:
        die(f"output has {len(out)} tensors, expected {EXPECTED_TENSORS}")

    # values / shapes / dtypes must be untouched by the rename
    for new_k, v in out.items():
        i = block_index(new_k)
        old_k = new_k if i is None else LAYER_RE.sub(f"gpt_neox.layers.{KEEP[i]}.", new_k, count=1)
        src = sd[old_k]
        if v.shape != src.shape or v.dtype != src.dtype:
            die(f"{old_k} -> {new_k}: shape/dtype changed "
                f"({src.shape}/{src.dtype} -> {v.shape}/{v.dtype})")
    print("all pre-write checks passed")

    # --- 4. write ----------------------------------------------------------
    DST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DST.with_suffix(".safetensors.tmp")
    save_file({k: v.contiguous() for k, v in out.items()}, str(tmp))

    # verify what actually landed on disk, then publish
    try:
        back = load_file(str(tmp))
        if len(back) != EXPECTED_TENSORS:
            die(f"written file has {len(back)} tensors, expected {EXPECTED_TENSORS}")
        if set(back) != set(out):
            die("written key set differs from the checked key set")
        for k, v in back.items():
            if v.dtype != out[k].dtype or v.shape != out[k].shape or not v.equal(out[k]):
                die(f"written tensor {k} does not match")
    except SystemExit:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(DST)
    print(f"OK: wrote {DST} with {len(back)} tensors, {N_LAYERS_OUT} blocks")


if __name__ == "__main__":
    main()
