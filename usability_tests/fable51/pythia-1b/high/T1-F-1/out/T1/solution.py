"""T1: depth-prune Pythia-1B from 16 to 12 blocks, renumbering contiguously.

Plain safetensors script: tensors are loaded once, block names rewritten via a
regex capture on the block index, dropped blocks skipped, and every required
check is asserted before anything is written. Output is written to a temp file
and renamed only after all checks pass, so a failure leaves no output.
"""
import os
import re
import sys

from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"
DROP = {2, 6, 10, 14}
N_OLD, N_NEW, PER_BLOCK, NON_BLOCK = 16, 12, 15, 4
BLOCK_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    if os.path.exists(DST):
        fail(f"destination already exists: {DST}")
    src = load_file(SRC)
    if len(src) != N_OLD * PER_BLOCK + NON_BLOCK:
        fail(f"unexpected input size {len(src)}")

    keep = [i for i in range(N_OLD) if i not in DROP]
    remap = {old: new for new, old in enumerate(keep)}  # order-preserving
    if len(remap) != N_NEW:
        fail(f"expected {N_NEW} surviving blocks, got {len(remap)}")

    out = {}
    for name, t in src.items():
        m = BLOCK_RE.match(name)
        if m is None:
            new_name = name
        else:
            old = int(m.group(1))
            if old not in remap:
                if old not in DROP:
                    fail(f"block {old} outside 0..{N_OLD - 1}: {name}")
                continue
            new_name = f"gpt_neox.layers.{remap[old]}.{m.group(2)}"
        if new_name in out:
            fail(f"collision: {name} -> {new_name} already present")
        out[new_name] = t

    # Required checks.
    for name in out:
        m = BLOCK_RE.match(name)
        if m and int(m.group(1)) >= N_NEW:
            fail(f"tensor of block >= {N_NEW} remains: {name}")
    qkv = sorted(
        int(BLOCK_RE.match(n).group(1))
        for n in out
        if n.endswith(".attention.query_key_value.weight") and BLOCK_RE.match(n)
    )
    if qkv != list(range(N_NEW)):
        fail(f"expected qkv weights for blocks 0..{N_NEW - 1}, got {qkv}")
    for i in range(N_NEW):
        cnt = sum(1 for n in out if n.startswith(f"gpt_neox.layers.{i}."))
        if cnt != PER_BLOCK:
            fail(f"block {i} has {cnt} tensors, expected {PER_BLOCK}")
    if len(out) != N_NEW * PER_BLOCK + NON_BLOCK:
        fail(f"output has {len(out)} tensors, expected {N_NEW * PER_BLOCK + NON_BLOCK}")
    # Verify each survivor is bit-identical to its source (same storage object).
    for old, new in remap.items():
        for name in src:
            m = BLOCK_RE.match(name)
            if m and int(m.group(1)) == old:
                if out[f"gpt_neox.layers.{new}.{m.group(2)}"] is not src[name]:
                    fail(f"value mismatch for {name}")
    for name in ("gpt_neox.embed_in.weight", "embed_out.weight",
                 "gpt_neox.final_layer_norm.weight", "gpt_neox.final_layer_norm.bias"):
        if out[name] is not src[name]:
            fail(f"non-block tensor changed: {name}")

    tmp = DST + ".tmp"
    save_file({k: v.contiguous() for k, v in out.items()}, tmp, metadata={"format": "pt"})
    os.replace(tmp, DST)
    print(f"wrote {DST}: {len(out)} tensors, {N_NEW} blocks (dropped {sorted(DROP)})")


if __name__ == "__main__":
    main()
