"""
T1: Depth pruning with layer renumbering (GPT-2 124M).

Removes transformer blocks 2, 5, 8 and renumbers the surviving blocks so
that indices run 0..8 contiguously, in original order:
old 0->0, 1->1, 3->2, 4->3, 6->4, 7->5, 9->6, 10->7, 11->8.

Uses only `safetensors` (allowed under condition F). Fails loudly (raises /
non-zero exit, no output written) if any required check does not hold.
"""

import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
IN_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUT_PATH = HERE / "model.safetensors"

DROP_BLOCKS = {2, 5, 8}
N_ORIG_BLOCKS = 12
N_KEPT_BLOCKS = 9
TENSORS_PER_BLOCK = 13
N_NON_BLOCK_TENSORS = 4
N_EXPECTED_OUT_TENSORS = N_KEPT_BLOCKS * TENSORS_PER_BLOCK + N_NON_BLOCK_TENSORS  # 121

BLOCK_RE = re.compile(r"^h\.(\d+)\.")


def build_renumbering() -> dict[int, int]:
    survivors = [i for i in range(N_ORIG_BLOCKS) if i not in DROP_BLOCKS]
    assert len(survivors) == N_KEPT_BLOCKS
    return {old: new for new, old in enumerate(survivors)}


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"input checkpoint not found: {IN_PATH}")

    renumber = build_renumbering()

    with safe_open(str(IN_PATH), framework="pt") as f:
        in_keys = list(f.keys())
        if len(in_keys) != N_ORIG_BLOCKS * TENSORS_PER_BLOCK + N_NON_BLOCK_TENSORS:
            raise AssertionError(
                f"unexpected input tensor count: {len(in_keys)} "
                f"(expected {N_ORIG_BLOCKS * TENSORS_PER_BLOCK + N_NON_BLOCK_TENSORS})"
            )

        out_tensors = {}
        for key in in_keys:
            m = BLOCK_RE.match(key)
            if m is None:
                # non-block tensor (wte, wpe, ln_f.*): unchanged
                out_tensors[key] = f.get_tensor(key)
                continue

            old_idx = int(m.group(1))
            if old_idx in DROP_BLOCKS:
                continue

            new_idx = renumber[old_idx]
            new_key = f"h.{new_idx}.{key[m.end():]}"
            if new_key in out_tensors:
                raise AssertionError(f"collision writing {new_key} from {key}")
            out_tensors[new_key] = f.get_tensor(key)

    # --- Required checks: fail loudly, no output written on failure ---

    # 1. No tensor of blocks 9, 10, 11 (original numbering) may remain.
    # (By construction the output only ever contains indices 0..8, but we
    # check explicitly against the intent: original blocks 9/10/11 must be
    # gone, i.e. their surviving renamed forms exist under indices 6/7/8
    # and nothing is left referencing the old high indices as this data.)
    surviving_old = sorted(renumber.keys())
    if 9 not in surviving_old or 10 not in surviving_old or 11 not in surviving_old:
        raise AssertionError("expected original blocks 9, 10, 11 to survive under new indices")
    for forbidden in DROP_BLOCKS:
        if forbidden in (9, 10, 11):
            raise AssertionError("blocks 9, 10, 11 must not be dropped")

    # 2. Exactly 9 blocks remain.
    attn_c_attn_keys = [k for k in out_tensors if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", k)]
    if len(attn_c_attn_keys) != N_KEPT_BLOCKS:
        raise AssertionError(
            f"expected exactly {N_KEPT_BLOCKS} blocks, found {len(attn_c_attn_keys)}"
        )
    out_block_indices = sorted(int(BLOCK_RE.match(k).group(1)) for k in attn_c_attn_keys)
    if out_block_indices != list(range(N_KEPT_BLOCKS)):
        raise AssertionError(f"block indices not contiguous 0..8: {out_block_indices}")

    # 3. The 4 non-block tensors are unchanged (present, correct count).
    non_block_keys = [k for k in out_tensors if BLOCK_RE.match(k) is None]
    expected_non_block = {"wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"}
    if set(non_block_keys) != expected_non_block:
        raise AssertionError(f"non-block tensors mismatch: {sorted(non_block_keys)}")

    # 4. Output has exactly 121 tensors.
    if len(out_tensors) != N_EXPECTED_OUT_TENSORS:
        raise AssertionError(
            f"expected {N_EXPECTED_OUT_TENSORS} output tensors, got {len(out_tensors)}"
        )

    out_tensors = {k: v.contiguous() for k, v in out_tensors.items()}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(out_tensors)} tensors")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        if OUT_PATH.exists():
            OUT_PATH.unlink()
        sys.exit(1)
