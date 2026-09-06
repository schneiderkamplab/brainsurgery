"""T2: remove attention head 5 from every layer of OLMo-1B-0724-hf.

Reads the sharded safetensors checkpoint under inputs/base, slices the head
out of q/k/v (row blocks) and o_proj (column blocks) for every layer, verifies
the required shapes and tensor count, then writes a single safetensors file.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "inputs" / "base"
DST = ROOT / "out" / "T2" / "model.safetensors"

NUM_LAYERS = 16
NUM_HEADS = 16
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
PRUNE_HEAD = 5
EXPECTED_TENSORS = 114

lo, hi = PRUNE_HEAD * HEAD_DIM, (PRUNE_HEAD + 1) * HEAD_DIM  # 640, 768
NEW_OUT = HIDDEN - HEAD_DIM  # 1920


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def keep(idx_lo: int, idx_hi: int) -> torch.Tensor:
    return torch.cat([torch.arange(0, idx_lo), torch.arange(idx_hi, HIDDEN)])


# ---- load all shards --------------------------------------------------------
index = json.loads((SRC / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
state: dict[str, torch.Tensor] = {}
for shard in shards:
    part = load_file(str(SRC / shard))
    dup = set(part) & set(state)
    if dup:
        fail(f"duplicate tensors across shards: {sorted(dup)[:5]}")
    state.update(part)

if len(state) != EXPECTED_TENSORS:
    fail(f"input has {len(state)} tensors, expected {EXPECTED_TENSORS}")
if set(state) != set(index["weight_map"]):
    fail("loaded tensor set does not match the index weight_map")

# ---- prune head 5 -----------------------------------------------------------
row_idx = keep(lo, hi)
for i in range(NUM_LAYERS):
    pre = f"model.layers.{i}.self_attn."
    for name in ("q_proj", "k_proj", "v_proj"):
        key = pre + name + ".weight"
        w = state[key]
        if tuple(w.shape) != (HIDDEN, HIDDEN):
            fail(f"{key}: unexpected input shape {tuple(w.shape)}")
        state[key] = w.index_select(0, row_idx).contiguous()
    key = pre + "o_proj.weight"
    w = state[key]
    if tuple(w.shape) != (HIDDEN, HIDDEN):
        fail(f"{key}: unexpected input shape {tuple(w.shape)}")
    state[key] = w.index_select(1, row_idx).contiguous()

# ---- required checks (before writing) ---------------------------------------
checks = {
    "model.layers.0.self_attn.q_proj.weight": (NEW_OUT, HIDDEN),
    "model.layers.0.self_attn.k_proj.weight": (NEW_OUT, HIDDEN),
    "model.layers.0.self_attn.v_proj.weight": (NEW_OUT, HIDDEN),
    "model.layers.0.self_attn.o_proj.weight": (HIDDEN, NEW_OUT),
}
for key, shape in checks.items():
    got = tuple(state[key].shape)
    if got != shape:
        fail(f"{key}: shape {got}, expected {shape}")
if len(state) != EXPECTED_TENSORS:
    fail(f"output would have {len(state)} tensors, expected {EXPECTED_TENSORS}")

# Extra sanity on every layer: shapes and that the kept blocks are bit-exact.
for i in range(NUM_LAYERS):
    pre = f"model.layers.{i}.self_attn."
    for name in ("q_proj", "k_proj", "v_proj"):
        if tuple(state[pre + name + ".weight"].shape) != (NEW_OUT, HIDDEN):
            fail(f"layer {i} {name} has wrong shape")
    if tuple(state[pre + "o_proj.weight"].shape) != (HIDDEN, NEW_OUT):
        fail(f"layer {i} o_proj has wrong shape")
for t in state.values():
    if t.dtype != torch.float32:
        fail("non-float32 tensor found")

# ---- write ------------------------------------------------------------------
DST.parent.mkdir(parents=True, exist_ok=True)
save_file(state, str(DST), metadata={"format": "pt"})

# Verify what landed on disk.
written = load_file(str(DST))
if len(written) != EXPECTED_TENSORS:
    fail(f"written file has {len(written)} tensors")
for key, shape in checks.items():
    if tuple(written[key].shape) != shape:
        fail(f"written {key} has shape {tuple(written[key].shape)}")
print(f"OK: wrote {DST} with {len(written)} tensors")
