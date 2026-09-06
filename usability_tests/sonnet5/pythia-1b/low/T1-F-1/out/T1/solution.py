"""T1: Depth pruning with layer renumbering (Pythia-1B).

Removes blocks 2, 6, 10, 14 and renumbers the surviving 12 blocks to 0..11,
preserving order. Uses safetensors for I/O and a plain regex-based rewrite
(the same rule-based approach torch-state-bridge offers) so the remap is a
single explicit table, not order-dependent renaming that could collide.
"""

import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

DROP_BLOCKS = {2, 6, 10, 14}
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.")

IN_PATH = Path(__file__).resolve().parents[2] / "inputs" / "base" / "model.safetensors"
OUT_PATH = Path(__file__).resolve().parent / "model.safetensors"


def build_remap(n_layers: int, drop: set[int]) -> dict[int, int]:
    surviving = [i for i in range(n_layers) if i not in drop]
    return {old: new for new, old in enumerate(surviving)}


def main() -> None:
    with safe_open(str(IN_PATH), framework="pt") as f:
        keys = list(f.keys())
        tensors = {k: f.get_tensor(k) for k in keys}

    remap = build_remap(16, DROP_BLOCKS)

    out = {}
    for key, tensor in tensors.items():
        m = LAYER_RE.match(key)
        if m is None:
            # non-block tensor: unchanged
            out[key] = tensor
            continue
        old_idx = int(m.group(1))
        if old_idx in DROP_BLOCKS:
            continue
        new_idx = remap[old_idx]
        new_key = LAYER_RE.sub(f"gpt_neox.layers.{new_idx}.", key)
        if new_key in out:
            raise RuntimeError(f"collision writing {new_key} (from {key})")
        out[new_key] = tensor

    # --- Required checks: fail loudly, write nothing on failure ---
    for bad in (12, 13, 14, 15):
        for k in out:
            m = LAYER_RE.match(k)
            if m and int(m.group(1)) == bad:
                raise RuntimeError(f"block {bad} tensor survived: {k}")

    qkv_count = sum(
        1 for k in out if re.fullmatch(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", k)
    )
    if qkv_count != 12:
        raise RuntimeError(f"expected 12 blocks, found {qkv_count}")

    if len(out) != 184:
        raise RuntimeError(f"expected 184 tensors, found {len(out)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(out)} tensors")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
