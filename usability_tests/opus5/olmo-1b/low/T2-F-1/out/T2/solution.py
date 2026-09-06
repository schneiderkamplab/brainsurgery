"""T2: remove attention head 5 from every layer of OLMo-1B-0724-hf.

Plain safetensors + torch. Rows (q/k/v) and columns (o) of the pruned head's
128-wide block are dropped; every other tensor is copied unchanged.
"""

import glob
import os
import re

import torch
from safetensors.torch import load_file, save_file

IN_DIR = "inputs/base"
OUT_FILE = "out/T2/model.safetensors"

HEAD_DIM = 128
PRUNE = 5
LO, HI = PRUNE * HEAD_DIM, (PRUNE + 1) * HEAD_DIM  # 640, 768
HIDDEN = 2048
KEEP = torch.cat([torch.arange(0, LO), torch.arange(HI, HIDDEN)])

ROW_RE = re.compile(r"^model\.layers\.\d+\.self_attn\.[qkv]_proj\.weight$")
COL_RE = re.compile(r"^model\.layers\.\d+\.self_attn\.o_proj\.weight$")

state = {}
shards = sorted(glob.glob(os.path.join(IN_DIR, "*.safetensors")))
assert shards, f"no safetensors shards under {IN_DIR}"
for shard in shards:
    for name, tensor in load_file(shard).items():
        assert name not in state, f"duplicate tensor {name}"
        state[name] = tensor

n_in = len(state)
assert n_in == 114, f"expected 114 input tensors, got {n_in}"

out = {}
n_row = n_col = 0
for name, tensor in state.items():
    if ROW_RE.match(name):
        assert tuple(tensor.shape) == (HIDDEN, HIDDEN), f"{name}: {tuple(tensor.shape)}"
        out[name] = tensor.index_select(0, KEEP).contiguous()
        n_row += 1
    elif COL_RE.match(name):
        assert tuple(tensor.shape) == (HIDDEN, HIDDEN), f"{name}: {tuple(tensor.shape)}"
        out[name] = tensor.index_select(1, KEEP).contiguous()
        n_col += 1
    else:
        out[name] = tensor

assert n_row == 16 * 3, f"expected 48 q/k/v tensors, matched {n_row}"
assert n_col == 16, f"expected 16 o_proj tensors, matched {n_col}"

# Required checks, before writing.
for proj in ("q_proj", "k_proj", "v_proj"):
    key = f"model.layers.0.self_attn.{proj}.weight"
    assert tuple(out[key].shape) == (1920, 2048), f"{key}: {tuple(out[key].shape)}"
o0 = "model.layers.0.self_attn.o_proj.weight"
assert tuple(out[o0].shape) == (2048, 1920), f"{o0}: {tuple(out[o0].shape)}"
assert len(out) == 114, f"expected 114 output tensors, got {len(out)}"
assert set(out) == set(state), "key set changed"

# Value spot-check: kept blocks must be bit-identical to the source.
src_q = state["model.layers.0.self_attn.q_proj.weight"]
dst_q = out["model.layers.0.self_attn.q_proj.weight"]
assert torch.equal(dst_q[:LO], src_q[:LO]) and torch.equal(dst_q[LO:], src_q[HI:])
src_o = state[o0]
dst_o = out[o0]
assert torch.equal(dst_o[:, :LO], src_o[:, :LO]) and torch.equal(dst_o[:, LO:], src_o[:, HI:])

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
save_file(out, OUT_FILE)
print(f"wrote {OUT_FILE}: {len(out)} tensors ({n_row} row-pruned, {n_col} col-pruned)")
