"""T1: depth-prune Pythia-1B from 16 to 12 transformer blocks.

Drops blocks 2, 6, 10, 14 and renumbers the survivors, in their original
order, onto contiguous indices 0..11.  The renaming is done by building a
fresh dict keyed by the new names, so a shift can never overwrite a block
that is still alive; every insertion is additionally checked for collision.

All required checks run against the in-memory result before anything is
written, so a failing run exits non-zero with no output file.
"""

import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(SANDBOX, "inputs", "base", "model.safetensors")
DST = os.path.join(SANDBOX, "out", "T1", "model.safetensors")

DROP = {2, 6, 10, 14}
N_OLD_BLOCKS = 16
N_NEW_BLOCKS = 12
TENSORS_PER_BLOCK = 15
N_NON_BLOCK = 4
EXPECTED_OUT_TENSORS = N_NEW_BLOCKS * TENSORS_PER_BLOCK + N_NON_BLOCK  # 184

BLOCK_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")
NON_BLOCK_KEYS = {
    "gpt_neox.embed_in.weight",
    "embed_out.weight",
    "gpt_neox.final_layer_norm.weight",
    "gpt_neox.final_layer_norm.bias",
}
QKV_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.weight$")


def die(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    # ---- load ------------------------------------------------------------
    with safe_open(SRC, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        src = {k: f.get_tensor(k) for k in f.keys()}

    print(f"loaded {len(src)} tensors from {SRC}")

    # ---- sanity on the input --------------------------------------------
    old_blocks = {}
    non_block = {}
    for key in src:
        m = BLOCK_RE.match(key)
        if m:
            old_blocks.setdefault(int(m.group(1)), {})[m.group(2)] = key
        elif key in NON_BLOCK_KEYS:
            non_block[key] = src[key]
        else:
            die(f"unrecognised tensor name in input: {key!r}")

    if sorted(old_blocks) != list(range(N_OLD_BLOCKS)):
        die(f"expected input blocks 0..{N_OLD_BLOCKS - 1}, found {sorted(old_blocks)}")
    if len(non_block) != N_NON_BLOCK:
        die(f"expected {N_NON_BLOCK} non-block tensors, found {sorted(non_block)}")

    suffixes = set(old_blocks[0])
    if len(suffixes) != TENSORS_PER_BLOCK:
        die(f"block 0 has {len(suffixes)} tensors, expected {TENSORS_PER_BLOCK}")
    for i, entries in sorted(old_blocks.items()):
        if set(entries) != suffixes:
            die(f"block {i} does not have the same tensor set as block 0")

    if not DROP <= set(old_blocks):
        die(f"blocks to drop {sorted(DROP)} are not all present in the input")

    # ---- build the renumbering map --------------------------------------
    keep = [i for i in range(N_OLD_BLOCKS) if i not in DROP]
    remap = {old: new for new, old in enumerate(keep)}
    print("renumbering: " + ", ".join(f"{o}->{n}" for o, n in sorted(remap.items())))

    # ---- rename into a fresh dict ---------------------------------------
    out = {}

    def put(key, tensor):
        if key in out:
            die(f"name collision while renumbering: {key!r} written twice")
        out[key] = tensor

    for old_idx in keep:
        new_idx = remap[old_idx]
        for suffix, old_key in old_blocks[old_idx].items():
            put(f"gpt_neox.layers.{new_idx}.{suffix}", src[old_key])
    for key, tensor in non_block.items():
        put(key, tensor)

    # ---- required checks (all before writing) ---------------------------
    remaining = sorted({int(BLOCK_RE.match(k).group(1)) for k in out if BLOCK_RE.match(k)})

    stale = [i for i in remaining if i >= N_NEW_BLOCKS]
    if stale:
        die(f"tensors of removed/out-of-range blocks remain: {stale}")

    if remaining != list(range(N_NEW_BLOCKS)):
        die(f"expected contiguous blocks 0..{N_NEW_BLOCKS - 1}, found {remaining}")

    n_qkv = sum(1 for k in out if QKV_RE.match(k))
    if n_qkv != N_NEW_BLOCKS:
        die(f"expected {N_NEW_BLOCKS} query_key_value.weight tensors, found {n_qkv}")

    if len(out) != EXPECTED_OUT_TENSORS:
        die(f"expected {EXPECTED_OUT_TENSORS} output tensors, found {len(out)}")

    # every dropped block is gone, every kept block is present intact
    for old_idx in sorted(DROP):
        for suffix in suffixes:
            leftover = [
                k
                for k in out
                if k.endswith("." + suffix) and out[k].data_ptr() == src[old_blocks[old_idx][suffix]].data_ptr()
            ]
            if leftover:
                die(f"tensor of dropped block {old_idx} survives as {leftover}")

    for old_idx in keep:
        new_idx = remap[old_idx]
        for suffix, old_key in old_blocks[old_idx].items():
            new_key = f"gpt_neox.layers.{new_idx}.{suffix}"
            a, b = out[new_key], src[old_key]
            if a.shape != b.shape or a.dtype != b.dtype:
                die(f"{new_key}: shape/dtype changed ({b.shape},{b.dtype} -> {a.shape},{a.dtype})")
            if not torch.equal(a, b):
                die(f"{new_key}: values differ from source {old_key}")

    for key in NON_BLOCK_KEYS:
        if key not in out:
            die(f"non-block tensor missing from output: {key}")
        if not torch.equal(out[key], src[key]):
            die(f"non-block tensor changed: {key}")

    print(f"all checks passed: {len(out)} tensors, blocks {remaining[0]}..{remaining[-1]}")

    # ---- write (contiguous copies; atomic rename) -----------------------
    out = {k: v.contiguous().clone() for k, v in out.items()}
    tmp = DST + ".tmp"
    save_file(out, tmp, metadata=metadata)
    os.replace(tmp, DST)
    print(f"wrote {DST} ({os.path.getsize(DST)} bytes)")


if __name__ == "__main__":
    main()
