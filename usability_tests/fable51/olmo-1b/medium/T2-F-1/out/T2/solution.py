"""T2: remove attention head 5 from every layer of OLMo-1B-0724-hf (checkpoint level).

Plain safetensors + torch. Reads the sharded input, slices q/k/v rows and
o_proj columns, checks the required shapes and tensor count, then writes a
single out/T2/model.safetensors.
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "inputs" / "base"
OUT = ROOT / "out" / "T2" / "model.safetensors"

NUM_LAYERS, NUM_HEADS, HEAD_DIM, HIDDEN = 16, 16, 128, 2048
PRUNE_HEAD = 5
KEEP = [h for h in range(NUM_HEADS) if h != PRUNE_HEAD]
KEEP_IDX = torch.cat([torch.arange(h * HEAD_DIM, (h + 1) * HEAD_DIM) for h in KEEP])
assert KEEP_IDX.tolist() == list(range(0, 640)) + list(range(768, 2048))

index = json.loads((IN / "model.safetensors.index.json").read_text())
weight_map = index["weight_map"]
assert len(weight_map) == 114, len(weight_map)

tensors: dict[str, torch.Tensor] = {}
for shard in sorted(set(weight_map.values())):
    with safe_open(IN / shard, framework="pt", device="cpu") as f:
        for key in f.keys():
            assert key not in tensors, f"duplicate key {key}"
            tensors[key] = f.get_tensor(key)
assert set(tensors) == set(weight_map), "shard contents differ from index"

for i in range(NUM_LAYERS):
    p = f"model.layers.{i}.self_attn."
    for name in ("q_proj", "k_proj", "v_proj"):
        k = p + name + ".weight"
        w = tensors[k]
        assert tuple(w.shape) == (HIDDEN, HIDDEN), (k, w.shape)
        tensors[k] = w[KEEP_IDX, :].contiguous()
    k = p + "o_proj.weight"
    w = tensors[k]
    assert tuple(w.shape) == (HIDDEN, HIDDEN), (k, w.shape)
    tensors[k] = w[:, KEEP_IDX].contiguous()

# Required checks (fail loudly before writing).
pruned = (NUM_HEADS - 1) * HEAD_DIM
for name in ("q_proj", "k_proj", "v_proj"):
    s = tuple(tensors[f"model.layers.0.self_attn.{name}.weight"].shape)
    assert s == (pruned, HIDDEN), (name, s)
s = tuple(tensors["model.layers.0.self_attn.o_proj.weight"].shape)
assert s == (HIDDEN, pruned), ("o_proj", s)
assert len(tensors) == 114, len(tensors)
# Extra: every layer, all dtypes float32.
for i in range(NUM_LAYERS):
    for name in ("q_proj", "k_proj", "v_proj"):
        assert tuple(tensors[f"model.layers.{i}.self_attn.{name}.weight"].shape) == (pruned, HIDDEN)
    assert tuple(tensors[f"model.layers.{i}.self_attn.o_proj.weight"].shape) == (HIDDEN, pruned)
assert all(t.dtype == torch.float32 for t in tensors.values())

OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(OUT), metadata={"format": "pt"})
print(f"wrote {OUT} with {len(tensors)} tensors")
