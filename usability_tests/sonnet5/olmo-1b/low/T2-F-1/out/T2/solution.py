"""T2: structured attention-head pruning for OLMo-1B-0724-hf.

Removes head 5 (0-indexed) from every layer's q/k/v/o attention
projections, keeping all other tensors unchanged. Uses only `safetensors`
and `torch` directly (plain script, no framework needed for a pure
slice-and-reassemble op on a fixed, known layout).

Layout: nn.Linear weights are [out, in]; q/k/v pack heads as row blocks
of size head_dim=128 in [out]; o packs heads as column blocks of size 128
in [in]. Removing head h=5 means dropping rows/cols [h*128:(h+1)*128].
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

BASE = Path("inputs/base")
OUT = Path("out/T2/model.safetensors")

NUM_LAYERS = 16
NUM_HEADS = 16
HEAD_DIM = 128
HIDDEN = 2048
HEAD_TO_PRUNE = 5

EXPECTED_ROWS = 1920
EXPECTED_COLS = 1920


def head_slice_indices(num_heads, head_dim, prune_idx):
    """Row/col indices to KEEP, in order, after dropping one head block."""
    keep = []
    for h in range(num_heads):
        if h == prune_idx:
            continue
        keep.extend(range(h * head_dim, (h + 1) * head_dim))
    return torch.tensor(keep, dtype=torch.long)


def main():
    index = json.loads((BASE / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))

    tensors = {}
    for shard in shard_files:
        tensors.update(load_file(str(BASE / shard)))

    if len(tensors) != 114:
        sys.exit(f"expected 114 input tensors, got {len(tensors)}")

    keep_idx = head_slice_indices(NUM_HEADS, HEAD_DIM, HEAD_TO_PRUNE)
    assert keep_idx.numel() == EXPECTED_ROWS

    out = {}
    for name, tensor in tensors.items():
        is_q = name.endswith("self_attn.q_proj.weight")
        is_k = name.endswith("self_attn.k_proj.weight")
        is_v = name.endswith("self_attn.v_proj.weight")
        is_o = name.endswith("self_attn.o_proj.weight")

        if is_q or is_k or is_v:
            assert tensor.shape == (HIDDEN, HIDDEN), (name, tensor.shape)
            new_tensor = tensor.index_select(0, keep_idx).clone()
        elif is_o:
            assert tensor.shape == (HIDDEN, HIDDEN), (name, tensor.shape)
            new_tensor = tensor.index_select(1, keep_idx).clone()
        else:
            new_tensor = tensor

        out[name] = new_tensor

    # Required checks: fail loudly before writing if anything is off.
    checks = {
        "model.layers.0.self_attn.q_proj.weight": (EXPECTED_ROWS, HIDDEN),
        "model.layers.0.self_attn.k_proj.weight": (EXPECTED_ROWS, HIDDEN),
        "model.layers.0.self_attn.v_proj.weight": (EXPECTED_ROWS, HIDDEN),
        "model.layers.0.self_attn.o_proj.weight": (HIDDEN, EXPECTED_COLS),
    }
    for name, expected_shape in checks.items():
        actual = tuple(out[name].shape)
        if actual != expected_shape:
            sys.exit(f"check failed: {name} has shape {actual}, expected {expected_shape}")

    for i in range(NUM_LAYERS):
        for kind, expected_shape in (
            ("q_proj", (EXPECTED_ROWS, HIDDEN)),
            ("k_proj", (EXPECTED_ROWS, HIDDEN)),
            ("v_proj", (EXPECTED_ROWS, HIDDEN)),
            ("o_proj", (HIDDEN, EXPECTED_COLS)),
        ):
            name = f"model.layers.{i}.self_attn.{kind}.weight"
            actual = tuple(out[name].shape)
            if actual != expected_shape:
                sys.exit(f"check failed: {name} has shape {actual}, expected {expected_shape}")

    if len(out) != 114:
        sys.exit(f"check failed: output has {len(out)} tensors, expected 114")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata={"format": "pt"})
    print(f"wrote {OUT} with {len(out)} tensors")


if __name__ == "__main__":
    main()
