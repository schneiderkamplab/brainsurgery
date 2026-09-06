"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 layers with contiguous renumbering.

Reads the sharded safetensors checkpoint under inputs/base, drops blocks
2, 6, 10, 14, renumbers the survivors 0..11 in their original order and
writes a single out/T1/model.safetensors. The output is built as a fresh
dict keyed by the *new* names (never renamed in place), so a shifted
block can never overwrite a surviving one. All checks run before the file
is written, and the file is written to a temp name and renamed only after
the written file has been re-read and verified.
"""

import json
import os
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
IN_DIR = ROOT / "inputs" / "base"
OUT_FILE = HERE / "model.safetensors"
TMP_FILE = HERE / "model.safetensors.tmp"

OLD_LAYERS = 16
REMOVE = {2, 6, 10, 14}
KEEP = [i for i in range(OLD_LAYERS) if i not in REMOVE]
NEW_LAYERS = len(KEEP)  # 12
OLD_TO_NEW = {old: new for new, old in enumerate(KEEP)}
TENSORS_PER_BLOCK = 7
NON_BLOCK = {"model.embed_tokens.weight", "lm_head.weight"}
EXPECTED_OUT = NEW_LAYERS * TENSORS_PER_BLOCK + len(NON_BLOCK)  # 86

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    if TMP_FILE.exists():
        TMP_FILE.unlink()
    sys.exit(1)


def check(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


def load_input() -> dict[str, torch.Tensor]:
    index = json.loads((IN_DIR / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(IN_DIR / shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                check(name not in tensors, f"duplicate tensor {name!r} across shards")
                check(weight_map.get(name) == shard, f"{name!r} not indexed to {shard}")
                tensors[name] = f.get_tensor(name)
    check(set(tensors) == set(weight_map), "shard contents differ from index weight_map")
    return tensors


def main() -> None:
    check(not OUT_FILE.exists(), f"output already exists: {OUT_FILE}")
    src = load_input()
    check(len(src) == OLD_LAYERS * TENSORS_PER_BLOCK + len(NON_BLOCK),
          f"expected 114 input tensors, found {len(src)}")

    # Sanity-check the input layout: every block 0..15 owns exactly 7 tensors.
    per_block: dict[int, int] = {}
    for name in src:
        m = LAYER_RE.match(name)
        if m:
            per_block[int(m.group(1))] = per_block.get(int(m.group(1)), 0) + 1
        else:
            check(name in NON_BLOCK, f"unexpected non-block tensor {name!r}")
    check(set(per_block) == set(range(OLD_LAYERS)), f"input blocks: {sorted(per_block)}")
    check(all(n == TENSORS_PER_BLOCK for n in per_block.values()),
          f"input blocks with unexpected tensor counts: {per_block}")

    # Build the output as a fresh dict keyed by the NEW name. Because the
    # old->new map is injective and we never mutate `src`, no collision is possible.
    dst: dict[str, torch.Tensor] = {}
    for name, t in src.items():
        m = LAYER_RE.match(name)
        if m is None:
            new_name = name
        else:
            old = int(m.group(1))
            if old in REMOVE:
                continue
            new_name = f"model.layers.{OLD_TO_NEW[old]}.{m.group(2)}"
        check(new_name not in dst, f"collision: {name!r} -> {new_name!r} already present")
        dst[new_name] = t

    verify(dst, src)

    # Write to a temp name; only rename to the final path after re-reading it.
    save_file({k: v.contiguous() for k, v in dst.items()}, str(TMP_FILE))
    with safe_open(TMP_FILE, framework="pt", device="cpu") as f:
        written = {k: f.get_tensor(k) for k in f.keys()}
    verify(written, src)
    os.replace(TMP_FILE, OUT_FILE)
    print(f"OK: wrote {OUT_FILE} with {len(written)} tensors, {NEW_LAYERS} blocks")


def verify(dst: dict[str, torch.Tensor], src: dict[str, torch.Tensor]) -> None:
    # Required check: no tensor of blocks 12..15 remains.
    stale = [k for k in dst if (m := LAYER_RE.match(k)) and int(m.group(1)) >= NEW_LAYERS]
    check(not stale, f"tensors of blocks >= {NEW_LAYERS} remain: {stale}")

    # Required check: exactly 12 blocks remain, indices contiguous 0..11.
    q = [k for k in dst if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", k)]
    check(len(q) == NEW_LAYERS, f"expected {NEW_LAYERS} q_proj tensors, found {len(q)}")
    blocks = sorted({int(m.group(1)) for k in dst if (m := LAYER_RE.match(k))})
    check(blocks == list(range(NEW_LAYERS)), f"block indices not contiguous 0..11: {blocks}")
    for b in blocks:
        n = sum(1 for k in dst if k.startswith(f"model.layers.{b}."))
        check(n == TENSORS_PER_BLOCK, f"block {b} has {n} tensors, expected {TENSORS_PER_BLOCK}")

    # Required check: exactly 86 tensors.
    check(len(dst) == EXPECTED_OUT, f"expected {EXPECTED_OUT} output tensors, found {len(dst)}")

    # Stronger: every output tensor is bit-identical to its source under the mapping.
    for k, t in dst.items():
        m = LAYER_RE.match(k)
        src_name = k if m is None else f"model.layers.{KEEP[int(m.group(1))]}.{m.group(2)}"
        s = src[src_name]
        check(t.shape == s.shape and t.dtype == s.dtype,
              f"{k}: shape/dtype {tuple(t.shape)}/{t.dtype} != {tuple(s.shape)}/{s.dtype}")
        check(t.dtype == torch.float32, f"{k}: dtype {t.dtype} is not float32")
        check(torch.equal(t, s), f"{k}: values differ from {src_name}")
    for k in NON_BLOCK:
        check(k in dst, f"missing non-block tensor {k!r}")


if __name__ == "__main__":
    main()
