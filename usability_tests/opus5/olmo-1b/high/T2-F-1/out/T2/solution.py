"""T2: structured attention-head pruning of OLMo-1B-0724-hf.

Removes head 5 (0-indexed) from every layer by slicing the head block out of
the q/k/v projections (heads are row blocks of the [out, in] nn.Linear weight)
and out of the o projection (heads are column blocks, since o_proj consumes the
concatenated head outputs along its input axis).

Reads the sharded input, writes a single unsharded out/T2/model.safetensors.
All required checks are asserted on the in-memory result BEFORE writing, and
re-verified against the written file afterwards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_DIR = Path("inputs/base")
OUT_DIR = Path("out/T2")
OUT_FILE = OUT_DIR / "model.safetensors"

N_LAYERS = 16
N_HEADS = 16
HEAD_DIM = 128
HIDDEN = N_HEADS * HEAD_DIM  # 2048
PRUNE_HEAD = 5
N_TENSORS = 114

# The head block being removed: rows/cols 640..767. Keep 0..639 then 768..2047.
LO = PRUNE_HEAD * HEAD_DIM
HI = LO + HEAD_DIM
KEPT = HIDDEN - HEAD_DIM  # 1920

ROW_SLICED = ("q_proj", "k_proj", "v_proj")
COL_SLICED = ("o_proj",)


def check(cond: bool, msg: str) -> None:
    """Hard failure; no silent fallback."""
    if not cond:
        raise AssertionError(msg)


def load_input() -> dict[str, torch.Tensor]:
    index = json.loads((IN_DIR / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    check(
        len(weight_map) == N_TENSORS,
        f"input index lists {len(weight_map)} tensors, expected {N_TENSORS}",
    )

    state: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(IN_DIR / shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                check(key not in state, f"duplicate tensor {key} across shards")
                state[key] = f.get_tensor(key)
    check(
        set(state) == set(weight_map),
        "tensors present in the shards differ from the index weight_map",
    )
    return state


def head_names(kind: str) -> list[str]:
    return [f"model.layers.{i}.self_attn.{kind}.weight" for i in range(N_LAYERS)]


def prune(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    touched: set[str] = set()

    for kind in ROW_SLICED + COL_SLICED:
        for name in head_names(kind):
            check(name in state, f"missing head-bearing tensor {name}")
            w = state[name]
            check(
                tuple(w.shape) == (HIDDEN, HIDDEN),
                f"{name}: expected shape [{HIDDEN}, {HIDDEN}], got {list(w.shape)}",
            )
            if kind in ROW_SLICED:
                pruned = torch.cat([w[:LO, :], w[HI:, :]], dim=0)
            else:
                pruned = torch.cat([w[:, :LO], w[:, HI:]], dim=1)
            out[name] = pruned.contiguous()
            touched.add(name)

    # Everything else passes through untouched, names unchanged.
    for name, w in state.items():
        if name not in touched:
            out[name] = w
    return out


def verify(out: dict[str, torch.Tensor], state: dict[str, torch.Tensor]) -> None:
    # --- required checks (layer 0) ---
    for kind in ROW_SLICED:
        name = f"model.layers.0.self_attn.{kind}.weight"
        check(
            tuple(out[name].shape) == (KEPT, HIDDEN),
            f"{name}: expected [{KEPT}, {HIDDEN}], got {list(out[name].shape)}",
        )
    name = "model.layers.0.self_attn.o_proj.weight"
    check(
        tuple(out[name].shape) == (HIDDEN, KEPT),
        f"{name}: expected [{HIDDEN}, {KEPT}], got {list(out[name].shape)}",
    )
    check(len(out) == N_TENSORS, f"output has {len(out)} tensors, expected {N_TENSORS}")

    # --- same checks generalised to every layer ---
    for i in range(N_LAYERS):
        for kind in ROW_SLICED:
            n = f"model.layers.{i}.self_attn.{kind}.weight"
            check(
                tuple(out[n].shape) == (KEPT, HIDDEN),
                f"{n}: expected [{KEPT}, {HIDDEN}], got {list(out[n].shape)}",
            )
        n = f"model.layers.{i}.self_attn.o_proj.weight"
        check(
            tuple(out[n].shape) == (HIDDEN, KEPT),
            f"{n}: expected [{HIDDEN}, {KEPT}], got {list(out[n].shape)}",
        )

    # --- names, dtypes, and untouched tensors ---
    check(set(out) == set(state), "output key set differs from input key set")
    for n in out:
        check(
            out[n].dtype == state[n].dtype,
            f"{n}: dtype changed {state[n].dtype} -> {out[n].dtype}",
        )
    head_bearing = {n for k in ROW_SLICED + COL_SLICED for n in head_names(k)}
    for n in sorted(set(out) - head_bearing):
        check(
            out[n].shape == state[n].shape and torch.equal(out[n], state[n]),
            f"{n} should have been left untouched but changed",
        )

    # --- kept slices are bit-exact and in the right order ---
    for i in range(N_LAYERS):
        for kind in ROW_SLICED:
            n = f"model.layers.{i}.self_attn.{kind}.weight"
            src, dst = state[n], out[n]
            check(torch.equal(dst[:LO, :], src[:LO, :]), f"{n}: rows 0..{LO - 1} altered")
            check(torch.equal(dst[LO:, :], src[HI:, :]), f"{n}: rows {HI}.. misplaced")
        n = f"model.layers.{i}.self_attn.o_proj.weight"
        src, dst = state[n], out[n]
        check(torch.equal(dst[:, :LO], src[:, :LO]), f"{n}: cols 0..{LO - 1} altered")
        check(torch.equal(dst[:, LO:], src[:, HI:]), f"{n}: cols {HI}.. misplaced")


def verify_written(out: dict[str, torch.Tensor]) -> None:
    with safe_open(OUT_FILE, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        check(
            len(keys) == N_TENSORS,
            f"written file has {len(keys)} tensors, expected {N_TENSORS}",
        )
        check(keys == set(out), "written key set differs from the verified result")
        for k in sorted(keys):
            t = f.get_tensor(k)
            check(t.shape == out[k].shape, f"{k}: shape changed on write")
            check(t.dtype == out[k].dtype, f"{k}: dtype changed on write")
            check(torch.equal(t, out[k]), f"{k}: values changed on write")


def main() -> int:
    state = load_input()
    print(f"loaded {len(state)} tensors from {IN_DIR}")
    out = prune(state)
    verify(out, state)
    print(f"checks passed: removed head {PRUNE_HEAD} (rows/cols {LO}..{HI - 1}) in all layers")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_FILE))
    verify_written(out)
    print(f"wrote {OUT_FILE} ({OUT_FILE.stat().st_size} bytes, {len(out)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
