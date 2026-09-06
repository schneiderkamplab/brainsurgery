"""T1: depth-prune GPT-2 (124M) from 12 to 9 blocks, renumbering contiguously.

Plain safetensors + regex. The mapping old->new is built once from the sorted
list of surviving block indices, so every tensor is written under its final
name in a fresh dict: there is no in-place shifting and hence no collision
hazard. All required checks are asserted before anything is written.
"""
import os
import re
import sys

from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"
N_LAYERS = 12
DROP = {2, 5, 8}
TENSORS_PER_BLOCK = 13
NON_BLOCK = {"wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"}

BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if os.path.exists(DST):
        fail(f"destination already exists: {DST}")

    sd = load_file(SRC)
    if len(sd) != N_LAYERS * TENSORS_PER_BLOCK + len(NON_BLOCK):
        fail(f"unexpected input tensor count {len(sd)}")

    keep = [i for i in range(N_LAYERS) if i not in DROP]
    remap = {old: new for new, old in enumerate(keep)}  # old 3 -> 2, old 11 -> 8, ...

    out = {}
    for name, t in sd.items():
        m = BLOCK_RE.match(name)
        if m is None:
            if name not in NON_BLOCK:
                fail(f"unexpected non-block tensor: {name}")
            out[name] = t
            continue
        old, rest = int(m.group(1)), m.group(2)
        if old in DROP:
            continue
        if old not in remap:
            fail(f"block index {old} out of range in {name}")
        new_name = f"h.{remap[old]}.{rest}"
        if new_name in out:
            fail(f"collision: {new_name} already written (from {name})")
        out[new_name] = t

    # ---- required checks (fail loudly, nothing written yet) ----
    n_keep = len(keep)
    for name in out:
        m = BLOCK_RE.match(name)
        if m and int(m.group(1)) >= n_keep:
            fail(f"tensor of removed index range remains: {name}")
    c_attn = sorted(int(BLOCK_RE.match(n).group(1)) for n in out
                    if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", n))
    if c_attn != list(range(n_keep)):
        fail(f"expected exactly {n_keep} contiguous blocks, got c_attn.weight indices {c_attn}")
    expected_total = n_keep * TENSORS_PER_BLOCK + len(NON_BLOCK)
    if len(out) != expected_total:
        fail(f"expected {expected_total} tensors, got {len(out)}")
    # each surviving block must have its full 13 tensors, identical to the source
    for old, new in remap.items():
        for rest in (BLOCK_RE.match(n).group(2) for n in sd if n.startswith(f"h.{old}.")):
            src_t, dst_t = sd[f"h.{old}.{rest}"], out[f"h.{new}.{rest}"]
            if src_t.shape != dst_t.shape or src_t.dtype != dst_t.dtype or not src_t.equal(dst_t):
                fail(f"mismatch h.{old}.{rest} -> h.{new}.{rest}")
    for name in NON_BLOCK:
        if not sd[name].equal(out[name]):
            fail(f"non-block tensor changed: {name}")

    save_file({k: v.contiguous() for k, v in out.items()}, DST, metadata={"format": "pt"})

    # ---- re-verify the written file ----
    written = load_file(DST)
    if len(written) != expected_total:
        os.remove(DST)
        fail(f"written file has {len(written)} tensors")
    print(f"OK: wrote {DST} with {len(written)} tensors, blocks 0..{n_keep - 1}")


if __name__ == "__main__":
    main()
