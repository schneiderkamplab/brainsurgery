#!/usr/bin/env python
"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 blocks with renumbering.

Plain script on safetensors: load the sharded input, drop blocks {2,6,10,14},
renumber survivors 0..11, verify, then write out/T1/model.safetensors.
"""

import json
import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

IN_DIR = Path("inputs/base")
OUT_PATH = Path("out/T1/model.safetensors")
DROP = {2, 6, 10, 14}
N_OLD = 16
N_NEW = 12
N_TENSORS = 86
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def load_input() -> dict:
    index = json.loads((IN_DIR / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    state = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(IN_DIR / shard, framework="pt") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
    missing = set(weight_map) - set(state)
    if missing:
        fail(f"tensors listed in the index but absent from the shards: {sorted(missing)}")
    return state


def main() -> None:
    state = load_input()

    survivors = [i for i in range(N_OLD) if i not in DROP]
    if len(survivors) != N_NEW:
        fail(f"expected {N_NEW} surviving blocks, got {len(survivors)}")
    remap = {old: new for new, old in enumerate(survivors)}

    out: dict = {}
    for key, tensor in state.items():
        m = LAYER_RE.match(key)
        if m is None:
            new_key = key  # embeddings / lm_head, unchanged
        else:
            old = int(m.group(1))
            if old not in remap:
                continue  # pruned block
            new_key = f"model.layers.{remap[old]}.{m.group(2)}"
        if new_key in out:
            fail(f"renumbering collision: {new_key} written twice (from {key})")
        out[new_key] = tensor

    # Required checks.
    indices = set()
    for key in out:
        m = LAYER_RE.match(key)
        if m:
            indices.add(int(m.group(1)))
    stale = sorted(i for i in indices if i >= N_NEW)
    if stale:
        fail(f"tensors of blocks {stale} remain; indices must be 0..{N_NEW - 1}")
    q = [k for k in out if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", k)]
    if len(q) != N_NEW:
        fail(f"expected {N_NEW} blocks (q_proj matches), got {len(q)}")
    if indices != set(range(N_NEW)):
        fail(f"block indices are not contiguous 0..{N_NEW - 1}: {sorted(indices)}")
    if len(out) != N_TENSORS:
        fail(f"expected {N_TENSORS} tensors in the output, got {len(out)}")
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        if name not in out:
            fail(f"non-block tensor {name} missing from the output")
        if out[name].data_ptr() != state[name].data_ptr():
            fail(f"non-block tensor {name} was not passed through unchanged")

    # Values/shapes/dtypes must be identical to the source block.
    for old, new in remap.items():
        for key in state:
            m = LAYER_RE.match(key)
            if m and int(m.group(1)) == old:
                dst = out[f"model.layers.{new}.{m.group(2)}"]
                src = state[key]
                if dst.shape != src.shape or dst.dtype != src.dtype:
                    fail(f"shape/dtype changed for old block {old} -> {new} ({key})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(out)} tensors; blocks {sorted(indices)}")


if __name__ == "__main__":
    main()
