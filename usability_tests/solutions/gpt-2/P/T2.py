"""T2 baseline for GPT-2 (124M): remove attention head 5 from every layer."""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from _ckpt import load_checkpoint

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T2")
N_LAYERS = 12
DROP_HEAD = 5
# name, axis holding heads, number of concatenated segments, segment width, head block width
SPECS = [
    ("attn.c_attn.weight", 1, 3, 768, 64),
    ("attn.c_attn.bias", 0, 3, 768, 64),
    ("attn.c_proj.weight", 0, 1, 768, 64),
]


def keep_index(segments: int, seg_size: int, block: int) -> torch.Tensor:
    keep = []
    for seg in range(segments):
        for h in range(seg_size // block):
            if h != DROP_HEAD:
                start = seg * seg_size + h * block
                keep.append(torch.arange(start, start + block))
    return torch.cat(keep)


sd = load_checkpoint("inputs/base/model.safetensors")
for layer in range(N_LAYERS):
    for rel, dim, segments, seg_size, block in SPECS:
        name = "h.{i}.".format(i=layer) + rel
        sd[name] = sd[name].index_select(dim, keep_index(segments, seg_size, block)).contiguous()

assert sd["h.0.attn.c_attn.weight"].shape == (768, 2112)
assert sd["h.0.attn.c_attn.bias"].shape == (2112,)
assert sd["h.0.attn.c_proj.weight"].shape == (704, 768)
assert len(sd) == 160

out_dir.mkdir(parents=True, exist_ok=True)
save_file(sd, str(out_dir / "model.safetensors"))
