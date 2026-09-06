"""T2: remove attention head 5 from every layer of OLMo-1B-0724-hf."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

BASE = Path("inputs/base")
OUT = Path("out/T2/model.safetensors")

HEAD_DIM = 128
NUM_HEADS = 16
PRUNE = 5
HIDDEN = HEAD_DIM * NUM_HEADS

keep = torch.cat([
    torch.arange(0, PRUNE * HEAD_DIM),
    torch.arange((PRUNE + 1) * HEAD_DIM, HIDDEN),
])
assert keep.numel() == HIDDEN - HEAD_DIM

index = json.loads((BASE / "model.safetensors.index.json").read_text())
weight_map = index["weight_map"]

state = {}
for shard in sorted(set(weight_map.values())):
    state.update(load_file(BASE / shard))

missing = set(weight_map) - set(state)
if missing:
    raise SystemExit(f"tensors listed in the index but not loaded: {sorted(missing)}")
n_in = len(state)

ROW_PROJ = ("q_proj", "k_proj", "v_proj")

touched = 0
for name in list(state):
    parts = name.split(".")
    if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
        continue
    if parts[3] != "self_attn" or not name.endswith(".weight"):
        continue
    proj = parts[4]
    t = state[name]
    if proj in ROW_PROJ:
        if t.shape != (HIDDEN, HIDDEN):
            raise SystemExit(f"{name}: unexpected shape {tuple(t.shape)}")
        state[name] = t.index_select(0, keep).contiguous()
        touched += 1
    elif proj == "o_proj":
        if t.shape != (HIDDEN, HIDDEN):
            raise SystemExit(f"{name}: unexpected shape {tuple(t.shape)}")
        state[name] = t.index_select(1, keep).contiguous()
        touched += 1

expected_touched = 4 * 16
if touched != expected_touched:
    raise SystemExit(f"expected to prune {expected_touched} tensors, pruned {touched}")

# Required checks.
checks = {
    "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
}
for name, shape in checks.items():
    if name not in state:
        raise SystemExit(f"missing tensor {name}")
    got = tuple(state[name].shape)
    if got != shape:
        raise SystemExit(f"{name}: expected shape {shape}, got {got}")

if len(state) != 114:
    raise SystemExit(f"expected 114 tensors in the output, got {len(state)}")
if len(state) != n_in:
    raise SystemExit(f"tensor count changed: {n_in} -> {len(state)}")

OUT.parent.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous().clone() for k, v in state.items()}, str(OUT))
print(f"wrote {OUT} with {len(state)} tensors ({touched} pruned)")
