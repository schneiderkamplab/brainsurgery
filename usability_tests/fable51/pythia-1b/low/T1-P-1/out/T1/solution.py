"""T1: depth-prune Pythia-1B from 16 to 12 blocks, renumbering contiguously."""
import os
import re
import sys

from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"
REMOVE = {2, 6, 10, 14}
N_OLD, N_NEW, PER_BLOCK, EXPECTED_TOTAL = 16, 12, 15, 184
PAT = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    sd = load_file(SRC)
    if len(sd) != N_OLD * PER_BLOCK + 4:
        fail(f"unexpected input tensor count {len(sd)}")

    keep = [i for i in range(N_OLD) if i not in REMOVE]
    remap = {old: new for new, old in enumerate(keep)}

    out = {}
    for name, t in sd.items():
        m = PAT.match(name)
        if m is None:
            out[name] = t
            continue
        old = int(m.group(1))
        if old in REMOVE:
            continue
        new_name = f"gpt_neox.layers.{remap[old]}.{m.group(2)}"
        if new_name in out:
            fail(f"collision on {new_name}")
        out[new_name] = t

    # Required checks.
    idxs = [int(PAT.match(k).group(1)) for k in out if PAT.match(k)]
    if any(i >= N_NEW for i in idxs):
        fail(f"tensors of blocks >= {N_NEW} remain: {sorted({i for i in idxs if i >= N_NEW})}")
    qkv = [k for k in out if PAT.match(k) and k.endswith(".attention.query_key_value.weight")]
    if len(qkv) != N_NEW:
        fail(f"expected {N_NEW} qkv weights, found {len(qkv)}")
    if sorted(set(idxs)) != list(range(N_NEW)):
        fail(f"block indices not contiguous 0..{N_NEW-1}: {sorted(set(idxs))}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} tensors, got {len(out)}")
    for i in range(N_NEW):
        n = sum(1 for j in idxs if j == i)
        if n != PER_BLOCK:
            fail(f"block {i} has {n} tensors, expected {PER_BLOCK}")

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, DST, metadata={"format": "pt"})
    print(f"wrote {DST} with {len(out)} tensors")


if __name__ == "__main__":
    main()
