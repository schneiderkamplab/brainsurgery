"""T2: structured attention-head pruning for OLMo-1B-0724-hf.

Removes head 5 from every layer at the checkpoint level:
  q/k/v_proj ([out, in], heads are row blocks)  -> drop rows   640..767
  o_proj     ([out, in], heads are column blocks) -> drop cols 640..767
Everything else is copied through unchanged. Result: one flat
out/T2/model.safetensors with 114 tensors.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent / "inputs" / "base"
OUT = HERE / "model.safetensors"

NUM_LAYERS = 16
NUM_HEADS = 16
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM  # 2048
PRUNE_HEAD = 5
KEPT = HIDDEN - HEAD_DIM  # 1920

LO = PRUNE_HEAD * HEAD_DIM  # 640
HI = LO + HEAD_DIM  # 768

ROW_PRUNED = {"q_proj", "k_proj", "v_proj"}
COL_PRUNED = {"o_proj"}


def load_sharded(base: Path) -> dict[str, torch.Tensor]:
    """Load every shard listed in the index into one flat state dict."""
    index = json.loads((base / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        shard_state = load_file(str(base / shard))
        for name, tensor in shard_state.items():
            if name in state:
                raise RuntimeError(f"tensor {name!r} appears in more than one shard")
            state[name] = tensor
    missing = set(weight_map) - set(state)
    if missing:
        raise RuntimeError(f"index lists tensors not found in the shards: {sorted(missing)}")
    extra = set(state) - set(weight_map)
    if extra:
        raise RuntimeError(f"shards contain tensors not in the index: {sorted(extra)}")
    return state


def drop_block(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Return `tensor` with the head-5 block removed along `dim`, order preserved."""
    if tensor.shape[dim] != HIDDEN:
        raise AssertionError(f"expected size {HIDDEN} on dim {dim}, got {tuple(tensor.shape)}")
    head = tensor.narrow(dim, 0, LO)
    tail = tensor.narrow(dim, HI, HIDDEN - HI)
    return torch.cat([head, tail], dim=dim).contiguous()


def main() -> None:
    state = load_sharded(BASE)
    print(f"loaded {len(state)} tensors from {BASE}")

    pruned_names = {
        f"model.layers.{i}.self_attn.{proj}.weight"
        for i in range(NUM_LAYERS)
        for proj in sorted(ROW_PRUNED | COL_PRUNED)
    }
    if not pruned_names <= set(state):
        raise AssertionError(f"input is missing: {sorted(pruned_names - set(state))}")

    out: dict[str, torch.Tensor] = {}
    touched = 0
    for name, tensor in state.items():
        parts = name.split(".")
        # model.layers.<i>.self_attn.<proj>.weight
        is_attn_weight = (
            len(parts) == 6
            and parts[0] == "model"
            and parts[1] == "layers"
            and parts[3] == "self_attn"
            and parts[5] == "weight"
        )
        if is_attn_weight and parts[4] in ROW_PRUNED:
            out[name] = drop_block(tensor, dim=0)
            touched += 1
        elif is_attn_weight and parts[4] in COL_PRUNED:
            out[name] = drop_block(tensor, dim=1)
            touched += 1
        else:
            out[name] = tensor.clone().contiguous()

    expected_touched = NUM_LAYERS * (len(ROW_PRUNED) + len(COL_PRUNED))
    if touched != expected_touched:
        raise AssertionError(f"pruned {touched} tensors, expected {expected_touched}")

    # dtypes and untouched tensors must survive the round trip
    for name, tensor in out.items():
        if tensor.dtype is not state[name].dtype:
            raise AssertionError(f"dtype changed for {name}")
    for name, tensor in out.items():
        if name not in pruned_names and tensor.shape != state[name].shape:
            raise AssertionError(f"shape changed for untouched tensor {name}")
        if name not in pruned_names and not torch.equal(tensor, state[name]):
            raise AssertionError(f"values changed for untouched tensor {name}")

    # Per-layer shape checks on every projection.
    for i in range(NUM_LAYERS):
        for proj in ("q_proj", "k_proj", "v_proj"):
            got = tuple(out[f"model.layers.{i}.self_attn.{proj}.weight"].shape)
            if got != (KEPT, HIDDEN):
                raise AssertionError(f"layer {i} {proj}: expected {(KEPT, HIDDEN)}, got {got}")
        got = tuple(out[f"model.layers.{i}.self_attn.o_proj.weight"].shape)
        if got != (HIDDEN, KEPT):
            raise AssertionError(f"layer {i} o_proj: expected {(HIDDEN, KEPT)}, got {got}")

    # Required checks, spelled out on layer 0 exactly as the task states them.
    required = {
        "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
    }
    for name, want in required.items():
        got = tuple(out[name].shape)
        if got != want:
            raise AssertionError(f"required check failed: {name} is {got}, expected {want}")
    if len(out) != 114:
        raise AssertionError(f"required check failed: output has {len(out)} tensors, expected 114")

    # Value spot check: the kept blocks must be the original rows/columns.
    ref = state["model.layers.0.self_attn.q_proj.weight"]
    new = out["model.layers.0.self_attn.q_proj.weight"]
    if not torch.equal(new[:LO], ref[:LO]) or not torch.equal(new[LO:], ref[HI:]):
        raise AssertionError("q_proj rows are not the original rows in the required order")
    ref_o = state["model.layers.0.self_attn.o_proj.weight"]
    new_o = out["model.layers.0.self_attn.o_proj.weight"]
    if not torch.equal(new_o[:, :LO], ref_o[:, :LO]) or not torch.equal(new_o[:, LO:], ref_o[:, HI:]):
        raise AssertionError("o_proj columns are not the original columns in the required order")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata={"format": "pt"})

    written = load_file(str(OUT))
    if len(written) != 114:
        raise AssertionError(f"written file has {len(written)} tensors, expected 114")
    if set(written) != set(state):
        raise AssertionError("written key set differs from the input key set")
    for name, tensor in written.items():
        if tensor.shape != out[name].shape or tensor.dtype != out[name].dtype:
            raise AssertionError(f"round trip changed {name}")
        if not torch.equal(tensor, out[name]):
            raise AssertionError(f"round trip changed values of {name}")
    print(f"wrote {OUT} with {len(written)} tensors; all checks passed")


if __name__ == "__main__":
    main()
