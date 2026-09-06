"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 layers, renumbering blocks.

Plain safetensors + torch. Reads the sharded input, drops blocks 2,6,10,14,
renumbers survivors to 0..11 via an explicit old->new map (built in a fresh
dict, so no in-place renaming collision is possible), verifies, then writes a
single model.safetensors. Any check failure raises before anything is written.
"""
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "inputs" / "base"
OUT = ROOT / "out" / "T1" / "model.safetensors"

REMOVE = {2, 6, 10, 14}
OLD_LAYERS = 16
NEW_LAYERS = OLD_LAYERS - len(REMOVE)
PER_BLOCK = 7
NON_BLOCK = 2
EXPECTED_TOTAL = NEW_LAYERS * PER_BLOCK + NON_BLOCK  # 86

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if OUT.exists():
        fail(f"output already exists: {OUT}")

    index = json.loads((SRC / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    src: dict[str, torch.Tensor] = {}
    for shard in shards:
        part = load_file(str(SRC / shard))
        dup = set(part) & set(src)
        if dup:
            fail(f"duplicate keys across shards: {sorted(dup)[:5]}")
        src.update(part)
    if len(src) != OLD_LAYERS * PER_BLOCK + NON_BLOCK:
        fail(f"unexpected input tensor count {len(src)}")

    survivors = [i for i in range(OLD_LAYERS) if i not in REMOVE]
    remap = {old: new for new, old in enumerate(survivors)}
    assert list(remap.values()) == list(range(NEW_LAYERS))

    dst: dict[str, torch.Tensor] = {}
    removed = 0
    for name, t in src.items():
        m = LAYER_RE.match(name)
        if m is None:
            new_name = name  # embed_tokens / lm_head
        else:
            idx = int(m.group(1))
            if idx in REMOVE:
                removed += 1
                continue
            if idx not in remap:
                fail(f"layer index {idx} outside 0..{OLD_LAYERS - 1}: {name}")
            new_name = f"model.layers.{remap[idx]}.{m.group(2)}"
        if new_name in dst:
            fail(f"collision: {name} -> {new_name} already present")
        dst[new_name] = t

    # ---- Required checks (all must hold before any write) ----
    if removed != len(REMOVE) * PER_BLOCK:
        fail(f"removed {removed} tensors, expected {len(REMOVE) * PER_BLOCK}")
    stale = [k for k in dst if (m := LAYER_RE.match(k)) and int(m.group(1)) >= NEW_LAYERS]
    if stale:
        fail(f"tensors of blocks >= {NEW_LAYERS} remain: {stale[:5]}")
    q = sorted(int(LAYER_RE.match(k).group(1)) for k in dst
               if k.endswith(".self_attn.q_proj.weight"))
    if q != list(range(NEW_LAYERS)):
        fail(f"expected exactly {NEW_LAYERS} contiguous blocks, q_proj indices = {q}")
    for i in range(NEW_LAYERS):
        n = sum(1 for k in dst if k.startswith(f"model.layers.{i}."))
        if n != PER_BLOCK:
            fail(f"block {i} has {n} tensors, expected {PER_BLOCK}")
    if len(dst) != EXPECTED_TOTAL:
        fail(f"output has {len(dst)} tensors, expected {EXPECTED_TOTAL}")
    # Value/shape/dtype identity against the source under the mapping.
    for old, new in remap.items():
        for k in src:
            m = LAYER_RE.match(k)
            if m and int(m.group(1)) == old:
                s, d = src[k], dst[f"model.layers.{new}.{m.group(2)}"]
                if s.shape != d.shape or s.dtype != d.dtype or not torch.equal(s, d):
                    fail(f"mismatch for {k}")
    for k in ("model.embed_tokens.weight", "lm_head.weight"):
        if k not in dst or not torch.equal(src[k], dst[k]):
            fail(f"non-block tensor changed or missing: {k}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in dst.items()}, str(OUT),
              metadata={"format": "pt"})
    print(f"OK: wrote {OUT} with {len(dst)} tensors, {NEW_LAYERS} blocks")


if __name__ == "__main__":
    main()
