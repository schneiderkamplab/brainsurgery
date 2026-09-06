"""Remove attention head 5 from every GPT-2 layer, at the checkpoint level."""

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN = "inputs/base/model.safetensors"
OUT = "out/T2/model.safetensors"

HIDDEN, N_HEADS, HEAD_DIM, PRUNE = 768, 12, 64, 5

# Indices kept inside one 768-wide segment (all heads except head 5).
seg_keep = [i for i in range(HIDDEN) if not (PRUNE * HEAD_DIM <= i < (PRUNE + 1) * HEAD_DIM)]
# c_attn is [q | k | v] fused: the same pattern repeated at each segment offset.
qkv_keep = torch.tensor([o + i for o in (0, HIDDEN, 2 * HIDDEN) for i in seg_keep])
seg_keep = torch.tensor(seg_keep)

tensors, metadata = {}, None
with safe_open(IN, framework="pt") as f:
    metadata = f.metadata()
    for key in f.keys():
        t = f.get_tensor(key)
        parts = key.split(".")
        if len(parts) == 5 and parts[0] == "h" and parts[2] == "attn":
            name = ".".join(parts[3:])
            if name == "c_attn.weight":      # [768, 2304], heads are columns
                t = t.index_select(1, qkv_keep)
            elif name == "c_attn.bias":      # [2304], same layout as those columns
                t = t.index_select(0, qkv_keep)
            elif name == "c_proj.weight":    # [768, 768], heads are rows
                t = t.index_select(0, seg_keep)
        tensors[key] = t.contiguous()

# Required checks: fail loudly before writing.
assert list(tensors["h.0.attn.c_attn.weight"].shape) == [768, 2112], tensors["h.0.attn.c_attn.weight"].shape
assert list(tensors["h.0.attn.c_attn.bias"].shape) == [2112], tensors["h.0.attn.c_attn.bias"].shape
assert list(tensors["h.0.attn.c_proj.weight"].shape) == [704, 768], tensors["h.0.attn.c_proj.weight"].shape
assert len(tensors) == 160, f"expected 160 tensors, got {len(tensors)}"
# Same checks on every layer, not just layer 0.
for i in range(12):
    assert list(tensors[f"h.{i}.attn.c_attn.weight"].shape) == [768, 2112]
    assert list(tensors[f"h.{i}.attn.c_attn.bias"].shape) == [2112]
    assert list(tensors[f"h.{i}.attn.c_proj.weight"].shape) == [704, 768]

save_file(tensors, OUT, metadata=metadata)
print(f"wrote {OUT}: {len(tensors)} tensors")
