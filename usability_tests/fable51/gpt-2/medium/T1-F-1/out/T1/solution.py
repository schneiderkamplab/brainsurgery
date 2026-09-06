"""T1: depth-prune GPT-2 (124M) from 12 to 9 blocks, renumbering contiguously.

Plain script on top of safetensors. Fails loudly (non-zero exit, no output
written) if any required check does not hold; the output is written only
after every check on the in-memory result has passed.
"""
import os
import re
import sys

from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"
DROP = {2, 5, 8}
N_IN, N_OUT = 12, 9
EXPECTED_TENSORS = 121
BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if os.path.exists(DST):
        fail(f"destination already exists: {DST}")
    src = load_file(SRC)
    if len(src) != 160:
        fail(f"expected 160 input tensors, got {len(src)}")

    keep = [i for i in range(N_IN) if i not in DROP]
    if len(keep) != N_OUT:
        fail(f"expected {N_OUT} surviving blocks, got {len(keep)}")
    remap = {old: new for new, old in enumerate(keep)}  # old index -> new index

    out: dict = {}
    for name, tensor in src.items():
        m = BLOCK_RE.match(name)
        if m is None:
            out[name] = tensor  # non-block tensor, unchanged
            continue
        old = int(m.group(1))
        if old in DROP:
            continue
        if old not in remap:
            fail(f"unexpected block index {old} in {name}")
        new_name = f"h.{remap[old]}.{m.group(2)}"
        if new_name in out:
            fail(f"renumbering collision on {new_name} (from {name})")
        out[new_name] = tensor

    # Required checks on the in-memory result.
    for name in out:
        m = BLOCK_RE.match(name)
        if m and int(m.group(1)) >= N_OUT:
            fail(f"tensor of block >= {N_OUT} remains: {name}")
    c_attn = sorted(int(BLOCK_RE.match(n).group(1)) for n in out
                    if BLOCK_RE.match(n) and n.endswith(".attn.c_attn.weight"))
    if c_attn != list(range(N_OUT)):
        fail(f"expected exactly {N_OUT} blocks 0..{N_OUT-1}, got c_attn indices {c_attn}")
    if len(out) != EXPECTED_TENSORS:
        fail(f"expected {EXPECTED_TENSORS} tensors, got {len(out)}")
    # Every surviving block keeps all 13 tensors; every kept tensor is identical to its source.
    for new, old in enumerate(keep):
        new_keys = {n for n in out if n.startswith(f"h.{new}.")}
        old_keys = {n for n in src if n.startswith(f"h.{old}.")}
        if len(new_keys) != 13 or len(old_keys) != 13:
            fail(f"block {old}->{new}: expected 13 tensors, got {len(old_keys)}->{len(new_keys)}")
        for ok in old_keys:
            nk = f"h.{new}." + ok[len(f"h.{old}."):]
            if nk not in out or out[nk].data_ptr() != src[ok].data_ptr():
                fail(f"tensor {ok} not carried verbatim to {nk}")
    for name in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"):
        if name not in out or out[name].data_ptr() != src[name].data_ptr():
            fail(f"non-block tensor {name} missing or altered")

    save_file({k: v.contiguous() for k, v in out.items()}, DST, metadata={"format": "pt"})

    # Re-read and confirm the written file.
    chk = load_file(DST)
    if len(chk) != EXPECTED_TENSORS:
        os.remove(DST)
        fail(f"written file has {len(chk)} tensors, expected {EXPECTED_TENSORS}")
    for k, v in chk.items():
        if k not in out or v.shape != out[k].shape or v.dtype != out[k].dtype or not (v == out[k]).all():
            os.remove(DST)
            fail(f"written tensor {k} differs from in-memory result")
    print(f"OK: wrote {DST} with {len(chk)} tensors, blocks 0..{N_OUT-1}")


if __name__ == "__main__":
    main()
