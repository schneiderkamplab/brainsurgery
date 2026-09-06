#!/usr/bin/env python
"""T1: depth-prune GPT-2 (124M) to 9 blocks and renumber the survivors.

Route: safetensors for I/O + torch-state-bridge for the key renumbering
(rule-based regex rewriting with built-in collision detection), plus explicit
checks. Every check is fatal: the script exits non-zero and leaves no output
file behind if any of them fails.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch_state_bridge import state_bridge, state_bridge_preview

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # sandbox root
SRC = ROOT / "inputs" / "base" / "model.safetensors"
OUT_DIR = ROOT / "out" / "T1"
OUT = OUT_DIR / "model.safetensors"
RULES = (HERE / "rename_rules.txt").read_text()

DROP = (2, 5, 8)                               # blocks to remove
KEEP = [0, 1, 3, 4, 6, 7, 9, 10, 11]           # survivors, in original order
N_OUT_BLOCKS = 9
TENSORS_PER_BLOCK = 13
N_NON_BLOCK = 4
N_OUT_TENSORS = N_OUT_BLOCKS * TENSORS_PER_BLOCK + N_NON_BLOCK   # 121

BLOCK_RE = re.compile(r"^h\.(\d+)\.")


class CheckFailed(Exception):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailed(msg)


def block_of(key: str) -> int | None:
    m = BLOCK_RE.match(key)
    return int(m.group(1)) if m else None


def main() -> int:
    # ---------------------------------------------------------------- load
    check(SRC.is_file(), f"input checkpoint not found: {SRC}")
    with safe_open(SRC, framework="pt") as f:
        src_meta = {k: (tuple(f.get_slice(k).get_shape()), f.get_slice(k).get_dtype())
                    for k in f.keys()}
    sd = load_file(SRC)
    check(len(sd) == 160, f"expected 160 input tensors, got {len(sd)}")

    src_blocks = sorted({b for b in map(block_of, sd) if b is not None})
    check(src_blocks == list(range(12)), f"expected input blocks 0..11, got {src_blocks}")
    check(sorted([*DROP, *KEEP]) == src_blocks,
          "drop/keep partition does not cover the input blocks exactly once")

    # ------------------------------------------------------- 1. drop blocks
    pruned = {k: v for k, v in sd.items() if block_of(k) not in DROP}
    dropped = len(sd) - len(pruned)
    check(dropped == len(DROP) * TENSORS_PER_BLOCK,
          f"dropping blocks {DROP} removed {dropped} tensors, expected "
          f"{len(DROP) * TENSORS_PER_BLOCK}")

    # -------------------------------------------------- 2. renumber (bridge)
    # Preview first so a bad rule set is reported before anything is built.
    mapping, _unchanged, collisions = state_bridge_preview(pruned, RULES)
    check(not collisions, f"rule set collides on: {sorted(collisions)}")
    expected_map = {old: new for old, new in zip(KEEP, range(N_OUT_BLOCKS))}
    for old_key, new_key in mapping.items():
        ob, nb = block_of(old_key), block_of(new_key)
        if ob is None:
            check(new_key == old_key, f"non-block tensor renamed: {old_key} -> {new_key}")
            continue
        check(nb == expected_map[ob],
              f"block {ob} renumbered to {nb}, expected {expected_map[ob]}")
        check(new_key == BLOCK_RE.sub(f"h.{nb}.", old_key, count=1),
              f"rename changed more than the block index: {old_key} -> {new_key}")

    # detect_collision=True makes the library itself refuse to overwrite a key.
    out_sd = state_bridge(pruned, RULES, detect_collision=True)

    # ------------------------------------------------ 3. required checks
    out_blocks = [b for b in map(block_of, out_sd) if b is not None]

    # (a) no tensor of blocks 9, 10, 11 remains
    stale = sorted(k for k in out_sd if block_of(k) in (9, 10, 11))
    check(not stale, f"tensors of blocks 9/10/11 still present: {stale}")

    # (b) exactly 9 blocks remain, indices contiguous 0..8, 13 tensors each
    c_attn = sorted(k for k in out_sd if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", k))
    check(len(c_attn) == N_OUT_BLOCKS,
          f"{len(c_attn)} tensors match h.<i>.attn.c_attn.weight, expected {N_OUT_BLOCKS}")
    check(sorted(set(out_blocks)) == list(range(N_OUT_BLOCKS)),
          f"block indices are {sorted(set(out_blocks))}, expected 0..{N_OUT_BLOCKS - 1}")
    for i in range(N_OUT_BLOCKS):
        n = out_blocks.count(i)
        check(n == TENSORS_PER_BLOCK,
              f"block {i} has {n} tensors, expected {TENSORS_PER_BLOCK}")

    # (c) the output has exactly 121 tensors
    check(len(out_sd) == N_OUT_TENSORS,
          f"output has {len(out_sd)} tensors, expected {N_OUT_TENSORS}")

    # ----------------------------------- 4. values/shapes/dtypes untouched
    inverse = {new: old for old, new in expected_map.items()}
    for k, v in out_sd.items():
        b = block_of(k)
        src_key = k if b is None else BLOCK_RE.sub(f"h.{inverse[b]}.", k, count=1)
        check(src_key in sd, f"output key {k} has no source key ({src_key})")
        ref = sd[src_key]
        shape, dtype = src_meta[src_key]
        check(tuple(v.shape) == shape, f"{k}: shape {tuple(v.shape)} != source {shape}")
        check(dtype == "F32", f"{src_key}: on-disk dtype is {dtype}, expected F32")
        check(v.dtype == ref.dtype, f"{k}: dtype {v.dtype} != source {ref.dtype}")
        check(torch.equal(v, ref), f"{k}: values differ from source {src_key}")

    # the 4 non-block tensors are unchanged
    non_block = sorted(k for k in out_sd if block_of(k) is None)
    check(non_block == ["ln_f.bias", "ln_f.weight", "wpe.weight", "wte.weight"],
          f"unexpected non-block tensor set: {non_block}")

    # ---------------------------- 5. it must load as a 9-layer GPT-2 config
    from transformers import AutoConfig
    from transformers.models.gpt2.modeling_gpt2 import GPT2Model

    cfg = AutoConfig.from_pretrained(SRC.parent)
    cfg.n_layer = N_OUT_BLOCKS
    with torch.device("meta"):
        expected = set(GPT2Model(cfg).state_dict().keys())
    missing = sorted(expected - set(out_sd))
    check(not missing, f"9-layer GPT2Model would be missing: {missing}")
    extra = set(out_sd) - expected
    allowed_extra = {f"h.{i}.attn.bias" for i in range(N_OUT_BLOCKS)}
    check(extra == allowed_extra,
          f"unexpected extra tensors: {sorted(extra - allowed_extra)}")

    # ------------------------------------------------------------- 6. write
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".safetensors.tmp")
    save_file({k: v.contiguous() for k, v in out_sd.items()}, tmp)

    # re-read what landed on disk, then publish only if it still checks out
    try:
        back = load_file(tmp)
        check(len(back) == N_OUT_TENSORS,
              f"written file has {len(back)} tensors, expected {N_OUT_TENSORS}")
        check(set(back) == set(out_sd), "written key set differs from the checked one")
        for k, v in back.items():
            check(v.dtype == out_sd[k].dtype, f"{k}: dtype changed on write")
            check(torch.equal(v, out_sd[k]), f"{k}: values changed on write")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, OUT)

    print(f"OK: wrote {OUT} with {len(out_sd)} tensors, blocks 0..{N_OUT_BLOCKS - 1}")
    print("     dropped blocks " + ", ".join(map(str, DROP)) + "; renumbering " +
          ", ".join(f"{o}->{n}" for o, n in expected_map.items() if o != n))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailed as e:
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)
