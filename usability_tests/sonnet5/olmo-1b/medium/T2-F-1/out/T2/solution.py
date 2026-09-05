#!/usr/bin/env python
"""
T2: structured attention-head pruning for OLMo-1B-0724-hf.

Removes head 5 (0-indexed) from every layer's self-attention projections:
  - q_proj, k_proj, v_proj: heads are row blocks -> drop rows 640..767 (5*128..6*128)
  - o_proj: heads are column blocks -> drop columns 640..767

All other tensors are copied through unchanged. Uses plain torch + safetensors
(both in F-allowed.md); no brainsurgery.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE.parent.parent / "inputs" / "base"
OUTPUT_PATH = HERE / "model.safetensors"

NUM_LAYERS = 16
HEAD_DIM = 128
NUM_HEADS = 16
HIDDEN = 2048
HEAD_TO_PRUNE = 5

assert HIDDEN == NUM_HEADS * HEAD_DIM

ROW_START = HEAD_TO_PRUNE * HEAD_DIM  # 640
ROW_END = ROW_START + HEAD_DIM  # 768


def load_all_tensors(input_dir: Path) -> dict[str, torch.Tensor]:
    index_path = input_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    tensors: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        shard_tensors = load_file(input_dir / shard)
        tensors.update(shard_tensors)
    # Sanity: every key in the index was actually loaded, and nothing extra.
    assert set(tensors.keys()) == set(weight_map.keys()), (
        f"loaded keys do not match index: "
        f"missing={set(weight_map) - set(tensors)}, extra={set(tensors) - set(weight_map)}"
    )
    return tensors


def prune_row_blocks(t: torch.Tensor) -> torch.Tensor:
    """Drop rows [ROW_START:ROW_END] (a head block), keep the rest, in order."""
    assert t.shape[0] == HIDDEN, f"expected {HIDDEN} rows, got {t.shape[0]}"
    return torch.cat([t[:ROW_START], t[ROW_END:]], dim=0).contiguous()


def prune_col_blocks(t: torch.Tensor) -> torch.Tensor:
    """Drop columns [ROW_START:ROW_END] (a head block), keep the rest, in order."""
    assert t.shape[1] == HIDDEN, f"expected {HIDDEN} cols, got {t.shape[1]}"
    return torch.cat([t[:, :ROW_START], t[:, ROW_END:]], dim=1).contiguous()


def main() -> None:
    tensors = load_all_tensors(INPUT_DIR)
    expected_total = len(tensors)
    assert expected_total == 114, f"expected 114 input tensors, found {expected_total}"

    out: dict[str, torch.Tensor] = {}

    for name, t in tensors.items():
        is_pruned = False
        for i in range(NUM_LAYERS):
            prefix = f"model.layers.{i}.self_attn."
            if name == prefix + "q_proj.weight":
                out[name] = prune_row_blocks(t)
                is_pruned = True
                break
            if name == prefix + "k_proj.weight":
                out[name] = prune_row_blocks(t)
                is_pruned = True
                break
            if name == prefix + "v_proj.weight":
                out[name] = prune_row_blocks(t)
                is_pruned = True
                break
            if name == prefix + "o_proj.weight":
                out[name] = prune_col_blocks(t)
                is_pruned = True
                break
        if not is_pruned:
            out[name] = t

    # --- Required checks: fail loudly before writing ---
    q0 = out["model.layers.0.self_attn.q_proj.weight"]
    k0 = out["model.layers.0.self_attn.k_proj.weight"]
    v0 = out["model.layers.0.self_attn.v_proj.weight"]
    o0 = out["model.layers.0.self_attn.o_proj.weight"]

    assert q0.shape == (1920, 2048), f"q_proj.0 shape {tuple(q0.shape)} != (1920, 2048)"
    assert k0.shape == (1920, 2048), f"k_proj.0 shape {tuple(k0.shape)} != (1920, 2048)"
    assert v0.shape == (1920, 2048), f"v_proj.0 shape {tuple(v0.shape)} != (1920, 2048)"
    assert o0.shape == (2048, 1920), f"o_proj.0 shape {tuple(o0.shape)} != (2048, 1920)"

    for i in range(NUM_LAYERS):
        prefix = f"model.layers.{i}.self_attn."
        for proj, expected_shape in (
            ("q_proj.weight", (1920, 2048)),
            ("k_proj.weight", (1920, 2048)),
            ("v_proj.weight", (1920, 2048)),
            ("o_proj.weight", (2048, 1920)),
        ):
            key = prefix + proj
            got = tuple(out[key].shape)
            assert got == expected_shape, f"{key} shape {got} != {expected_shape}"

    assert len(out) == 114, f"expected 114 output tensors, got {len(out)}"

    # Bit-exact spot check: pruned q_proj rows must equal the surviving
    # original rows, in the same order, with the head-5 block removed.
    orig_q0 = tensors["model.layers.0.self_attn.q_proj.weight"]
    expected_q0 = torch.cat([orig_q0[:640], orig_q0[768:]], dim=0)
    assert torch.equal(q0, expected_q0), "q_proj.0 content mismatch after pruning"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUTPUT_PATH))
    print(f"Wrote {len(out)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAILED CHECK: {e}", file=sys.stderr)
        sys.exit(1)
