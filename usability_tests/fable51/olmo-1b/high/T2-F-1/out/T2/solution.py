"""T2: remove attention head 5 from every layer of OLMo-1B-0724-hf.

Plain safetensors + torch. Reads the sharded input, slices the head block
out of q/k/v (rows) and o_proj (columns), checks shapes and tensor count,
then writes a single out/T2/model.safetensors.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "inputs" / "base"
DST = ROOT / "out" / "T2" / "model.safetensors"

N_LAYERS = 16
N_HEADS = 16
HEAD_DIM = 128
HIDDEN = N_HEADS * HEAD_DIM  # 2048
PRUNE_HEAD = 5
KEEP_OUT = HIDDEN - HEAD_DIM  # 1920

lo, hi = PRUNE_HEAD * HEAD_DIM, (PRUNE_HEAD + 1) * HEAD_DIM  # 640, 768
keep_idx = torch.cat([torch.arange(0, lo), torch.arange(hi, HIDDEN)])
assert keep_idx.numel() == KEEP_OUT


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


# ---- load every shard listed in the index -------------------------------
index = json.loads((SRC / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
state: dict[str, torch.Tensor] = {}
for shard in shards:
    part = load_file(str(SRC / shard))
    dup = set(part) & set(state)
    if dup:
        fail(f"duplicate keys across shards: {sorted(dup)[:5]}")
    state.update(part)

if len(state) != 114:
    fail(f"input has {len(state)} tensors, expected 114")

# ---- prune --------------------------------------------------------------
touched = 0
for i in range(N_LAYERS):
    p = f"model.layers.{i}.self_attn."
    for name in ("q_proj", "k_proj", "v_proj"):
        key = p + name + ".weight"
        w = state[key]
        if tuple(w.shape) != (HIDDEN, HIDDEN):
            fail(f"{key} has shape {tuple(w.shape)}, expected {(HIDDEN, HIDDEN)}")
        state[key] = w.index_select(0, keep_idx).contiguous()
        touched += 1
    key = p + "o_proj.weight"
    w = state[key]
    if tuple(w.shape) != (HIDDEN, HIDDEN):
        fail(f"{key} has shape {tuple(w.shape)}, expected {(HIDDEN, HIDDEN)}")
    state[key] = w.index_select(1, keep_idx).contiguous()
    touched += 1

if touched != N_LAYERS * 4:
    fail(f"touched {touched} tensors, expected {N_LAYERS * 4}")

# ---- required checks (fail loudly before writing) ------------------------
expected = {
    "model.layers.0.self_attn.q_proj.weight": (KEEP_OUT, HIDDEN),
    "model.layers.0.self_attn.k_proj.weight": (KEEP_OUT, HIDDEN),
    "model.layers.0.self_attn.v_proj.weight": (KEEP_OUT, HIDDEN),
    "model.layers.0.self_attn.o_proj.weight": (HIDDEN, KEEP_OUT),
}
# same check on every layer, not just layer 0
for i in range(N_LAYERS):
    p = f"model.layers.{i}.self_attn."
    for name in ("q_proj", "k_proj", "v_proj"):
        expected[p + name + ".weight"] = (KEEP_OUT, HIDDEN)
    expected[p + "o_proj.weight"] = (HIDDEN, KEEP_OUT)

for key, shape in expected.items():
    if key not in state:
        fail(f"missing tensor {key}")
    if tuple(state[key].shape) != shape:
        fail(f"{key} has shape {tuple(state[key].shape)}, expected {shape}")
    if state[key].dtype != torch.float32:
        fail(f"{key} has dtype {state[key].dtype}, expected float32")

if len(state) != 114:
    fail(f"output would have {len(state)} tensors, expected 114")

# ---- write --------------------------------------------------------------
if DST.exists():
    fail(f"destination already exists: {DST}")
DST.parent.mkdir(parents=True, exist_ok=True)
save_file(state, str(DST), metadata={"format": "pt"})

# ---- verify what landed on disk ----------------------------------------
from safetensors import safe_open

with safe_open(str(DST), framework="pt") as f:
    keys = list(f.keys())
    if len(keys) != 114:
        fail(f"written file has {len(keys)} tensors, expected 114")
    for key, shape in expected.items():
        if tuple(f.get_slice(key).get_shape()) != shape:
            fail(f"on-disk {key} has wrong shape")

print(f"OK: wrote {DST} with {len(keys)} tensors; pruned head {PRUNE_HEAD} in {N_LAYERS} layers")
