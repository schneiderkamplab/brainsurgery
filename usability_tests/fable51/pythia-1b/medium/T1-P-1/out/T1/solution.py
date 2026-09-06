"""T1: depth-prune Pythia-1B from 16 to 12 layers, renumbering survivors."""
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"
REMOVE = {2, 6, 10, 14}
OLD_LAYERS = 16
NEW_LAYERS = OLD_LAYERS - len(REMOVE)
TENSORS_PER_BLOCK = 15
EXPECTED_TOTAL = 4 + NEW_LAYERS * TENSORS_PER_BLOCK  # 184

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    sd = load_file(SRC)
    if len(sd) != 4 + OLD_LAYERS * TENSORS_PER_BLOCK:
        fail(f"unexpected input tensor count {len(sd)}")

    survivors = [i for i in range(OLD_LAYERS) if i not in REMOVE]
    remap = {old: new for new, old in enumerate(survivors)}  # old index -> new index

    out = {}
    for name, t in sd.items():
        m = LAYER_RE.match(name)
        if m is None:
            out[name] = t  # non-block tensor, unchanged
            continue
        old = int(m.group(1))
        if old in REMOVE:
            continue
        if old not in remap:
            fail(f"unexpected layer index {old} in {name}")
        new_name = f"gpt_neox.layers.{remap[old]}.{m.group(2)}"
        if new_name in out:
            fail(f"collision: {new_name} already present")
        out[new_name] = t

    # Required checks.
    for name in out:
        m = LAYER_RE.match(name)
        if m and int(m.group(1)) >= NEW_LAYERS:
            fail(f"tensor of removed block range remains: {name}")
    qkv = [n for n in out if re.fullmatch(
        r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", n)]
    if len(qkv) != NEW_LAYERS:
        fail(f"expected {NEW_LAYERS} blocks, found {len(qkv)} qkv weights")
    block_ids = sorted({int(LAYER_RE.match(n).group(1)) for n in out if LAYER_RE.match(n)})
    if block_ids != list(range(NEW_LAYERS)):
        fail(f"block indices not contiguous: {block_ids}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} tensors, got {len(out)}")

    # Value preservation: every surviving block maps bit-exactly to its source.
    for new, old in enumerate(survivors):
        for rest in ("attention.query_key_value.weight", "input_layernorm.bias"):
            a = sd[f"gpt_neox.layers.{old}.{rest}"]
            b = out[f"gpt_neox.layers.{new}.{rest}"]
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                fail(f"value mismatch old {old} -> new {new} ({rest})")

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, DST, metadata={"format": "pt"})
    print(f"wrote {DST}: {len(out)} tensors, {NEW_LAYERS} blocks")


if __name__ == "__main__":
    main()
