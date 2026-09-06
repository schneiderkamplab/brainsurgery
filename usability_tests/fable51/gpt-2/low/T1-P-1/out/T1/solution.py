"""T1: depth-prune GPT-2 (124M) from 12 to 9 blocks, renumbering contiguously."""
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"
REMOVE = {2, 5, 8}
N_OLD = 12
TENSORS_PER_BLOCK = 13
NON_BLOCK = 4
BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    sd = load_file(SRC)
    if len(sd) != N_OLD * TENSORS_PER_BLOCK + NON_BLOCK:
        fail(f"unexpected input tensor count {len(sd)}")

    keep = [i for i in range(N_OLD) if i not in REMOVE]
    remap = {old: new for new, old in enumerate(keep)}  # old -> new index
    n_new = len(keep)

    out: dict[str, torch.Tensor] = {}
    for name, t in sd.items():
        m = BLOCK_RE.match(name)
        if m is None:
            out[name] = t
            continue
        old = int(m.group(1))
        if old in REMOVE:
            continue
        if old not in remap:
            fail(f"block index {old} out of range in {name}")
        new_name = f"h.{remap[old]}.{m.group(2)}"
        if new_name in out:
            fail(f"collision: {new_name} already written (from {name})")
        out[new_name] = t

    # Required checks.
    for name in out:
        m = BLOCK_RE.match(name)
        if m and int(m.group(1)) >= n_new:
            fail(f"tensor of removed-range block remains: {name}")
    n_attn = sum(1 for n in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", n))
    if n_attn != n_new:
        fail(f"expected {n_new} blocks, found {n_attn} c_attn.weight tensors")
    block_ids = sorted({int(BLOCK_RE.match(n).group(1)) for n in out if BLOCK_RE.match(n)})
    if block_ids != list(range(n_new)):
        fail(f"block indices not contiguous: {block_ids}")
    expected_total = n_new * TENSORS_PER_BLOCK + NON_BLOCK
    if len(out) != expected_total:
        fail(f"expected {expected_total} tensors, got {len(out)}")
    # Value/shape/dtype preservation against the source.
    for old in keep:
        for name, t in sd.items():
            m = BLOCK_RE.match(name)
            if m and int(m.group(1)) == old:
                o = out[f"h.{remap[old]}.{m.group(2)}"]
                if o.shape != t.shape or o.dtype != t.dtype or not torch.equal(o, t):
                    fail(f"mismatch for {name}")

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, DST)

    # Verify the written file.
    back = load_file(DST)
    if set(back) != set(out) or len(back) != expected_total:
        fail("written file does not match in-memory result")
    print(f"OK: wrote {DST} with {len(back)} tensors, {n_new} blocks")


if __name__ == "__main__":
    main()
