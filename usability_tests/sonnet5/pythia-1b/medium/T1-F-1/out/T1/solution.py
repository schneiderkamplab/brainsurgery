#!/usr/bin/env python
"""T1: depth-prune Pythia-1B from 16 layers to 12 layers, renumbering blocks
so indices are contiguous again.

Approach: plain script on top of `safetensors` (load/save) and `torch`
(tensor handling). No mergekit / torch-state-bridge needed: this is a single,
fully-specified rename+drop over an in-memory dict, so building a *new* dict
keyed by the target names sidesteps the classic collision hazard entirely
(there is never a moment where two live tensors share a name, unlike an
in-place/sequential rename that can overwrite a not-yet-moved block).

Fails loudly (non-zero exit, no output file written) if any required check
does not hold.
"""

import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
IN_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUT_PATH = HERE / "model.safetensors"

DROP_LAYERS = {2, 6, 10, 14}
# old index -> new index for surviving layers, in original order
SURVIVORS = [i for i in range(16) if i not in DROP_LAYERS]
assert len(SURVIVORS) == 12
REMAP = {old: new for new, old in enumerate(SURVIVORS)}

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def main() -> None:
    state_dict = load_file(IN_PATH)
    assert len(state_dict) == 244, f"expected 244 input tensors, got {len(state_dict)}"

    new_state_dict: dict[str, torch.Tensor] = {}
    seen_targets: set[str] = set()

    for key, tensor in state_dict.items():
        m = LAYER_RE.match(key)
        if m is None:
            # non-block tensor: unchanged
            new_key = key
        else:
            old_idx = int(m.group(1))
            rest = m.group(2)
            if old_idx not in REMAP:
                continue  # dropped block
            new_idx = REMAP[old_idx]
            new_key = f"gpt_neox.layers.{new_idx}.{rest}"

        if new_key in seen_targets:
            raise RuntimeError(f"collision: {new_key} produced twice")
        seen_targets.add(new_key)
        new_state_dict[new_key] = tensor

    # --- required checks: fail loudly, write nothing on failure ---

    for bad_idx in (12, 13, 14, 15):
        for key in new_state_dict:
            m = LAYER_RE.match(key)
            if m and int(m.group(1)) == bad_idx:
                raise RuntimeError(f"tensor of dropped/out-of-range block {bad_idx} remains: {key}")

    qkv_weight_re = re.compile(r"^gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.weight$")
    block_indices = {
        int(qkv_weight_re.match(k).group(1))
        for k in new_state_dict
        if qkv_weight_re.match(k)
    }
    assert block_indices == set(range(12)), (
        f"expected exactly blocks 0..11, got {sorted(block_indices)}"
    )
    assert len(block_indices) == 12, f"expected exactly 12 blocks, got {len(block_indices)}"

    assert len(new_state_dict) == 184, f"expected 184 output tensors, got {len(new_state_dict)}"

    non_block_keys = {
        "gpt_neox.embed_in.weight",
        "embed_out.weight",
        "gpt_neox.final_layer_norm.weight",
        "gpt_neox.final_layer_norm.bias",
    }
    for key in non_block_keys:
        assert key in new_state_dict, f"missing non-block tensor: {key}"
        assert torch.equal(new_state_dict[key], state_dict[key]), f"non-block tensor changed: {key}"

    for tensor in new_state_dict.values():
        if not tensor.is_contiguous():
            raise RuntimeError("non-contiguous tensor would fail safetensors save")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(new_state_dict, OUT_PATH)
    print(f"wrote {len(new_state_dict)} tensors to {OUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # fail loudly, no partial output
        print(f"FAILED: {exc}", file=sys.stderr)
        if OUT_PATH.exists():
            OUT_PATH.unlink()
        sys.exit(1)
