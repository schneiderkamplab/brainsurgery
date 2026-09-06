"""T1: depth pruning with layer renumbering for Pythia-1B."""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
DST = HERE / "model.safetensors"

DROP = {2, 6, 10, 14}
N_OLD = 16
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def main() -> None:
    src = load_file(str(SRC))

    keep = [i for i in range(N_OLD) if i not in DROP]
    remap = {old: new for new, old in enumerate(keep)}

    out = {}
    for name, tensor in src.items():
        m = LAYER_RE.match(name)
        if m is None:
            out[name] = tensor
            continue
        old = int(m.group(1))
        if old in DROP:
            continue
        new_name = f"gpt_neox.layers.{remap[old]}.{m.group(2)}"
        if new_name in out:
            raise SystemExit(f"renumbering collision on {new_name}")
        out[new_name] = tensor

    # Required checks.
    surviving = set()
    for name in out:
        m = LAYER_RE.match(name)
        if m is not None:
            surviving.add(int(m.group(1)))

    stale = sorted(i for i in surviving if i >= 12)
    if stale:
        raise SystemExit(f"tensors remain for out-of-range blocks {stale}")

    n_qkv = sum(
        1
        for name in out
        if re.fullmatch(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", name)
    )
    if n_qkv != 12:
        raise SystemExit(f"expected 12 blocks, found {n_qkv} query_key_value.weight tensors")
    if len(surviving) != 12 or surviving != set(range(12)):
        raise SystemExit(f"block indices are not exactly 0..11: {sorted(surviving)}")

    if len(out) != 184:
        raise SystemExit(f"expected 184 tensors, got {len(out)}")

    save_file(out, str(DST), metadata={"format": "pt"})
    print(f"wrote {DST} with {len(out)} tensors")


if __name__ == "__main__":
    sys.exit(main())
