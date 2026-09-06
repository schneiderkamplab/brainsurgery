"""
T1: Depth pruning with layer renumbering (GPT-2 124M).

Removes transformer blocks 2, 5, 8 and renumbers the surviving blocks so
indices run 0..8 contiguously, preserving original relative order:

  old: 0 1 [2] 3 4 [5] 6 7 [8] 9 10 11
  new: 0 1     2 3     4 5     6 7  8

Only tensor names are rewritten (the `h.<i>.` prefix); values, shapes and
dtypes are copied unchanged. Non-block tensors (wte, wpe, ln_f.*) pass
through untouched.

Plain script on top of `safetensors` (no merge-config DSL needed for a
same-architecture rename/drop/keep operation) -- see F-allowed.md.
"""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

DROP_BLOCKS = {2, 5, 8}
NUM_ORIGINAL_BLOCKS = 12
BLOCK_KEY_RE = re.compile(r"^h\.(\d+)\.")

IN_PATH = Path("inputs/base/model.safetensors")
OUT_PATH = Path("out/T1/model.safetensors")


def build_renumbering() -> dict[int, int]:
    """Map surviving old block indices to new contiguous indices, in order."""
    survivors = [i for i in range(NUM_ORIGINAL_BLOCKS) if i not in DROP_BLOCKS]
    return {old: new for new, old in enumerate(survivors)}


def main() -> None:
    tensors = load_file(str(IN_PATH))
    renumber = build_renumbering()

    out: dict[str, "torch.Tensor"] = {}
    seen_new_names: set[str] = set()

    for name, tensor in tensors.items():
        m = BLOCK_KEY_RE.match(name)
        if m is None:
            # Non-block tensor: wte.weight, wpe.weight, ln_f.weight, ln_f.bias.
            new_name = name
        else:
            old_idx = int(m.group(1))
            if old_idx not in renumber:
                # Dropped block (2, 5, 8): skip entirely.
                continue
            new_idx = renumber[old_idx]
            new_name = f"h.{new_idx}.{name[m.end():]}"

        if new_name in seen_new_names:
            print(f"FATAL: collision writing {new_name!r} (from {name!r})", file=sys.stderr)
            sys.exit(1)
        seen_new_names.add(new_name)
        out[new_name] = tensor.clone().contiguous()

    # --- Required checks: fail loudly, write nothing, on any violation. ---

    for bad in (9, 10, 11):
        if any(re.match(rf"^h\.{bad}\.", k) for k in out):
            print(f"FATAL: tensor of dropped/old block {bad} present in output", file=sys.stderr)
            sys.exit(1)

    block_attn_weights = [k for k in out if re.match(r"^h\.\d+\.attn\.c_attn\.weight$", k)]
    if len(block_attn_weights) != 9:
        print(
            f"FATAL: expected exactly 9 blocks, found {len(block_attn_weights)} "
            f"(matched {sorted(block_attn_weights)})",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(out) != 121:
        print(f"FATAL: expected exactly 121 tensors, got {len(out)}", file=sys.stderr)
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"OK: wrote {len(out)} tensors to {OUT_PATH}")


if __name__ == "__main__":
    main()
