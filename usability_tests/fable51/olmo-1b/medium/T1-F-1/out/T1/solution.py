"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 layers, renumbering contiguously.

Plain safetensors + torch. The output is built as a fresh dict from an explicit
old->new block map, so there is no in-place rename and no collision hazard.
All checks run before anything is written; the file is written to a temp path
and renamed into place only after the checks pass.
"""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, "..", "..", "inputs", "base")
OUT = os.path.join(HERE, "model.safetensors")

DROP = {2, 6, 10, 14}
N_IN, N_OUT = 16, 12
TENSORS_PER_BLOCK = 7
EXPECTED_TOTAL = 86

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    if os.path.exists(OUT):
        fail(f"output already exists: {OUT}")

    # Load all shards.
    with open(os.path.join(IN_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))
    src = {}
    for shard in shards:
        part = load_file(os.path.join(IN_DIR, shard))
        dup = set(part) & set(src)
        if dup:
            fail(f"duplicate keys across shards: {sorted(dup)[:5]}")
        src.update(part)
    if len(src) != 114:
        fail(f"expected 114 input tensors, got {len(src)}")

    # Explicit old -> new block mapping (surviving blocks in original order).
    keep = [i for i in range(N_IN) if i not in DROP]
    if len(keep) != N_OUT:
        fail(f"keep list has {len(keep)} blocks, expected {N_OUT}")
    remap = {old: new for new, old in enumerate(keep)}

    dst = {}
    dropped = 0
    for name, t in src.items():
        m = LAYER_RE.match(name)
        if m is None:
            if name not in ("model.embed_tokens.weight", "lm_head.weight"):
                fail(f"unexpected non-block tensor: {name}")
            new_name = name
        else:
            old = int(m.group(1))
            if old >= N_IN:
                fail(f"block index out of range: {name}")
            if old in DROP:
                dropped += 1
                continue
            new_name = f"model.layers.{remap[old]}.{m.group(2)}"
        if new_name in dst:
            fail(f"collision: {new_name} already assigned")
        dst[new_name] = t.contiguous()

    # ---- Required checks ----
    if dropped != len(DROP) * TENSORS_PER_BLOCK:
        fail(f"dropped {dropped} tensors, expected {len(DROP) * TENSORS_PER_BLOCK}")
    stale = [k for k in dst if (m := LAYER_RE.match(k)) and int(m.group(1)) >= N_OUT]
    if stale:
        fail(f"tensors of blocks >= {N_OUT} remain: {stale[:5]}")
    q_keys = [k for k in dst if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", k)]
    if len(q_keys) != N_OUT:
        fail(f"expected {N_OUT} q_proj tensors, got {len(q_keys)}")
    blocks = sorted({int(m.group(1)) for k in dst if (m := LAYER_RE.match(k))})
    if blocks != list(range(N_OUT)):
        fail(f"block indices not contiguous 0..{N_OUT - 1}: {blocks}")
    for b in range(N_OUT):
        n = sum(1 for k in dst if k.startswith(f"model.layers.{b}."))
        if n != TENSORS_PER_BLOCK:
            fail(f"block {b} has {n} tensors, expected {TENSORS_PER_BLOCK}")
    if len(dst) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} output tensors, got {len(dst)}")
    # Value/shape/dtype fidelity against the source under the mapping.
    for old, new in remap.items():
        for k in src:
            if k.startswith(f"model.layers.{old}."):
                nk = k.replace(f"model.layers.{old}.", f"model.layers.{new}.", 1)
                a, b_ = src[k], dst[nk]
                if a.shape != b_.shape or a.dtype != b_.dtype or not torch.equal(a, b_):
                    fail(f"mismatch {k} -> {nk}")
    for k in ("model.embed_tokens.weight", "lm_head.weight"):
        if not torch.equal(src[k], dst[k]):
            fail(f"non-block tensor changed: {k}")

    # ---- Write atomically ----
    tmp = OUT + ".tmp"
    try:
        save_file(dst, tmp, metadata={"format": "pt"})
        # Re-open and verify the written file.
        back = load_file(tmp)
        if set(back) != set(dst) or len(back) != EXPECTED_TOTAL:
            fail("written file key set does not match")
        os.replace(tmp, OUT)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print(f"OK: wrote {OUT} with {len(dst)} tensors; blocks {keep} -> 0..{N_OUT - 1}")


if __name__ == "__main__":
    main()
