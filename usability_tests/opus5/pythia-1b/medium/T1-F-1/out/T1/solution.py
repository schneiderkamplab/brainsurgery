"""T1: depth-prune Pythia-1B from 16 to 12 blocks, renumbering survivors.

Route: safetensors for I/O + torch-state-bridge for the block renumbering
(rule-based key rewriting with built-in collision detection).
"""

import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch_state_bridge import state_bridge

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"

DROP = [2, 6, 10, 14]
N_OLD, N_NEW = 16, 12
EXPECTED_TENSORS = 184
PER_BLOCK = 15

BLOCK_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    with safe_open(SRC, framework="pt") as f:
        src = {k: f.get_tensor(k) for k in f.keys()}

    keep = [i for i in range(N_OLD) if i not in DROP]
    if len(keep) != N_NEW:
        fail(f"expected {N_NEW} surviving blocks, got {len(keep)}")
    mapping = {old: new for new, old in enumerate(keep)}

    # Drop every tensor of the pruned blocks.
    pruned = {}
    dropped = 0
    for k, v in src.items():
        m = BLOCK_RE.match(k)
        if m and int(m.group(1)) in DROP:
            dropped += 1
            continue
        pruned[k] = v
    if dropped != len(DROP) * PER_BLOCK:
        fail(f"expected to drop {len(DROP) * PER_BLOCK} tensors, dropped {dropped}")

    # Renumber. Rules are emitted in increasing old-index order and every target
    # is smaller than its source, so no rule can re-fire on another rule's output.
    rules = "\n".join(
        f"gpt_neox.layers.{old}., gpt_neox.layers.{new}."
        for old, new in sorted(mapping.items())
        if old != new
    )
    out = state_bridge(pruned, rules, detect_collision=True)

    # ---- Required checks (all before anything is written) ----
    if len(out) != EXPECTED_TENSORS:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TENSORS}")

    idx = set()
    for k in out:
        m = BLOCK_RE.match(k)
        if m:
            idx.add(int(m.group(1)))
        elif k not in {
            "gpt_neox.embed_in.weight",
            "embed_out.weight",
            "gpt_neox.final_layer_norm.weight",
            "gpt_neox.final_layer_norm.bias",
        }:
            fail(f"unexpected non-block tensor {k!r}")

    stale = sorted(i for i in idx if i >= N_NEW)
    if stale:
        fail(f"tensors of blocks {stale} remain (expected indices 0..{N_NEW - 1})")
    if idx != set(range(N_NEW)):
        fail(f"block indices are {sorted(idx)}, expected 0..{N_NEW - 1}")

    qkv = [k for k in out if re.fullmatch(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", k)]
    if len(qkv) != N_NEW:
        fail(f"{len(qkv)} query_key_value.weight tensors, expected {N_NEW}")

    for i in range(N_NEW):
        n = sum(1 for k in out if BLOCK_RE.match(k) and int(BLOCK_RE.match(k).group(1)) == i)
        if n != PER_BLOCK:
            fail(f"block {i} has {n} tensors, expected {PER_BLOCK}")

    # Every output tensor must be bit-identical to its source under the mapping.
    for new_key, t in out.items():
        m = BLOCK_RE.match(new_key)
        if m:
            new_i, rest = int(m.group(1)), m.group(2)
            old_key = f"gpt_neox.layers.{keep[new_i]}.{rest}"
        else:
            old_key = new_key
        if old_key not in src:
            fail(f"output key {new_key!r} has no source {old_key!r}")
        s = src[old_key]
        if s.shape != t.shape or s.dtype != t.dtype or not torch.equal(s, t):
            fail(f"value/shape/dtype mismatch for {new_key!r} vs {old_key!r}")

    # ---- Write only after the checks pass; atomic move so no partial output ----
    tmp = DST + ".tmp"
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    save_file(out, tmp, metadata={"format": "pt"})

    with safe_open(tmp, framework="pt") as f:
        written = list(f.keys())
        if len(written) != EXPECTED_TENSORS or set(written) != set(out):
            os.remove(tmp)
            fail(f"written file has {len(written)} tensors, expected {EXPECTED_TENSORS}")
        for k in written:
            if not torch.equal(f.get_tensor(k), out[k]):
                os.remove(tmp)
                fail(f"written tensor {k!r} does not round-trip")

    os.replace(tmp, DST)
    print(f"OK: wrote {DST} with {len(out)} tensors, {N_NEW} blocks (dropped {DROP}).")


if __name__ == "__main__":
    main()
