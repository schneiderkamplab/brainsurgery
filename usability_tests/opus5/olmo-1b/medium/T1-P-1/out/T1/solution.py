"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 blocks with contiguous renumbering."""

import json
import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

SANDBOX = Path(__file__).resolve().parents[2]
SRC = SANDBOX / "inputs" / "base"
DST_DIR = SANDBOX / "out" / "T1"
DST = DST_DIR / "model.safetensors"

DROP = {2, 6, 10, 14}
N_OLD = 16
N_NEW = 12
N_EXPECTED_TENSORS = 86

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def die(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def load_source() -> dict:
    index_path = SRC / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards: dict[str, list[str]] = {}
        for name, shard in weight_map.items():
            shards.setdefault(shard, []).append(name)
    else:
        shards = {p.name: None for p in sorted(SRC.glob("*.safetensors"))}

    tensors: dict = {}
    for shard, names in shards.items():
        with safe_open(SRC / shard, framework="pt", device="cpu") as f:
            for name in (names if names is not None else f.keys()):
                if name in tensors:
                    die(f"duplicate tensor across shards: {name}")
                tensors[name] = f.get_tensor(name)
    return tensors


def main() -> None:
    src = load_source()
    print(f"loaded {len(src)} tensors from {SRC}")

    # old block index -> new block index, surviving blocks in original order
    survivors = [i for i in range(N_OLD) if i not in DROP]
    if len(survivors) != N_NEW:
        die(f"expected {N_NEW} surviving blocks, computed {len(survivors)}")
    remap = {old: new for new, old in enumerate(survivors)}

    out: dict = {}
    for name, tensor in src.items():
        m = LAYER_RE.match(name)
        if m is None:
            out[name] = tensor  # non-block tensor, passes through untouched
            continue
        old = int(m.group(1))
        if old not in remap:
            continue  # dropped block
        new_name = f"model.layers.{remap[old]}.{m.group(2)}"
        if new_name in out:
            die(f"renumbering collision: {name} -> {new_name} already present")
        out[new_name] = tensor

    # --- required checks ---------------------------------------------------
    stale = sorted(n for n in out if (m := LAYER_RE.match(n)) and int(m.group(1)) >= N_NEW)
    if stale:
        die(f"tensors of blocks >= {N_NEW} remain: {stale}")

    q_idx = sorted(
        int(m.group(1))
        for n in out
        if (m := LAYER_RE.match(n)) and m.group(2) == "self_attn.q_proj.weight"
    )
    if q_idx != list(range(N_NEW)):
        die(f"expected q_proj for blocks 0..{N_NEW - 1}, got {q_idx}")

    block_idx = sorted({int(m.group(1)) for n in out if (m := LAYER_RE.match(n))})
    if block_idx != list(range(N_NEW)):
        die(f"block indices are not contiguous 0..{N_NEW - 1}: {block_idx}")

    if len(out) != N_EXPECTED_TENSORS:
        die(f"expected {N_EXPECTED_TENSORS} tensors in output, got {len(out)}")

    # values/shapes/dtypes must be untouched
    for new_name, tensor in out.items():
        m = LAYER_RE.match(new_name)
        origin = f"model.layers.{survivors[int(m.group(1))]}.{m.group(2)}" if m else new_name
        ref = src[origin]
        if tensor.shape != ref.shape or tensor.dtype != ref.dtype:
            die(f"{new_name}: shape/dtype changed vs {origin}")

    DST_DIR.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, str(DST))
    print(f"wrote {len(out)} tensors to {DST}")


if __name__ == "__main__":
    main()
