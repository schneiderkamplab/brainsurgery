"""T1: depth-prune Pythia-1B from 16 to 12 blocks, renumbering contiguously.

Plain safetensors + torch script. Builds an explicit old->new block map,
copies every tensor under its new name, enforces the required checks before
writing, and writes atomically so a failed check leaves no output behind.
"""
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"
DROP = {2, 6, 10, 14}
N_OLD, N_NEW = 16, 12
EXPECTED_TENSORS = 184
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    if os.path.exists(DST):
        fail(f"destination already exists: {DST}")
    src = load_file(SRC)
    if len(src) != N_OLD * 15 + 4:
        fail(f"unexpected input tensor count {len(src)}")

    keep = [i for i in range(N_OLD) if i not in DROP]
    remap = {old: new for new, old in enumerate(keep)}  # order-preserving
    assert len(remap) == N_NEW

    out = {}
    for name, t in src.items():
        m = LAYER_RE.match(name)
        if m is None:
            out[name] = t  # non-block tensor, unchanged
            continue
        old = int(m.group(1))
        if old in DROP:
            continue
        new_name = f"gpt_neox.layers.{remap[old]}.{m.group(2)}"
        if new_name in out:
            fail(f"collision: {new_name} already written (from old block {old})")
        out[new_name] = t

    # Required checks.
    for name in out:
        m = LAYER_RE.match(name)
        if m and int(m.group(1)) >= N_NEW:
            fail(f"block index >= {N_NEW} remains: {name}")
    qkv = [n for n in out if re.fullmatch(
        r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", n)]
    if len(qkv) != N_NEW:
        fail(f"expected {N_NEW} query_key_value.weight tensors, found {len(qkv)}")
    blocks = sorted({int(LAYER_RE.match(n).group(1)) for n in out if LAYER_RE.match(n)})
    if blocks != list(range(N_NEW)):
        fail(f"block indices not contiguous 0..{N_NEW - 1}: {blocks}")
    if len(out) != EXPECTED_TENSORS:
        fail(f"expected {EXPECTED_TENSORS} tensors, got {len(out)}")

    # Value-level check: each kept block is bit-identical to its source block.
    for old, new in remap.items():
        for rest in ("attention.query_key_value.weight", "attention.bias", "input_layernorm.bias"):
            a = src[f"gpt_neox.layers.{old}.{rest}"]
            b = out[f"gpt_neox.layers.{new}.{rest}"]
            if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                fail(f"mismatch old {old} -> new {new} for {rest}")

    tmp = DST + ".tmp"
    save_file({k: v.contiguous() for k, v in out.items()}, tmp, metadata={"format": "pt"})
    os.replace(tmp, DST)

    # Re-open the written file and verify.
    chk = load_file(DST)
    if len(chk) != EXPECTED_TENSORS or set(chk) != set(out):
        os.remove(DST)
        fail("written file does not match expected key set")
    print(f"OK: wrote {DST} with {len(chk)} tensors, blocks {blocks}")


if __name__ == "__main__":
    main()
