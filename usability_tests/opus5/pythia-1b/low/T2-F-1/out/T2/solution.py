"""T2: remove attention head 5 from every layer of Pythia-1B (checkpoint level)."""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN = Path("inputs/base/model.safetensors")
OUT = Path("out/T2/model.safetensors")

NUM_LAYERS = 16
HEAD_DIM = 256
PRUNE = 5
BLOCK = 3 * HEAD_DIM  # 768 fused q|k|v rows per head

row_keep = torch.cat([torch.arange(0, PRUNE * BLOCK), torch.arange((PRUNE + 1) * BLOCK, 8 * BLOCK)])
col_keep = torch.cat(
    [torch.arange(0, PRUNE * HEAD_DIM), torch.arange((PRUNE + 1) * HEAD_DIM, 8 * HEAD_DIM)]
)

state = {}
with safe_open(str(IN), framework="pt") as f:
    metadata = f.metadata()
    for key in f.keys():
        state[key] = f.get_tensor(key)

n_in = len(state)

for i in range(NUM_LAYERS):
    p = f"gpt_neox.layers.{i}.attention."
    for name, idx, dim in (
        (p + "query_key_value.weight", row_keep, 0),
        (p + "query_key_value.bias", row_keep, 0),
        (p + "dense.weight", col_keep, 1),
    ):
        if name not in state:
            raise KeyError(f"missing expected tensor {name}")
        state[name] = state[name].index_select(dim, idx).contiguous()

# Required checks: fail loudly before writing.
expected = {
    "gpt_neox.layers.0.attention.query_key_value.weight": (5376, 2048),
    "gpt_neox.layers.0.attention.query_key_value.bias": (5376,),
    "gpt_neox.layers.0.attention.dense.weight": (2048, 1792),
}
for name, shape in expected.items():
    got = tuple(state[name].shape)
    if got != shape:
        raise AssertionError(f"{name}: expected shape {shape}, got {got}")
if len(state) != 244:
    raise AssertionError(f"expected 244 tensors in the output, got {len(state)}")
if len(state) != n_in:
    raise AssertionError(f"tensor count changed: {n_in} -> {len(state)}")

OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(state, str(OUT), metadata=metadata)
print(f"wrote {OUT} with {len(state)} tensors")
