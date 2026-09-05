"""T2 baseline for OLMo-1B-0724-hf: remove attention head 5 from every layer."""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from _ckpt import load_checkpoint

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T2")
N_LAYERS = 16
DROP_HEAD = 5
# name, axis holding heads, number of concatenated segments, segment width, head block width
SPECS = [
    ("self_attn.q_proj.weight", 0, 1, 2048, 128),
    ("self_attn.k_proj.weight", 0, 1, 2048, 128),
    ("self_attn.v_proj.weight", 0, 1, 2048, 128),
    ("self_attn.o_proj.weight", 1, 1, 2048, 128),
]


def keep_index(segments: int, seg_size: int, block: int) -> torch.Tensor:
    keep = []
    for seg in range(segments):
        for h in range(seg_size // block):
            if h != DROP_HEAD:
                start = seg * seg_size + h * block
                keep.append(torch.arange(start, start + block))
    return torch.cat(keep)


sd = load_checkpoint("inputs/base")
for layer in range(N_LAYERS):
    for rel, dim, segments, seg_size, block in SPECS:
        name = "model.layers.{i}.".format(i=layer) + rel
        sd[name] = sd[name].index_select(dim, keep_index(segments, seg_size, block)).contiguous()

assert sd["model.layers.0.self_attn.q_proj.weight"].shape == (1920, 2048)
assert sd["model.layers.0.self_attn.k_proj.weight"].shape == (1920, 2048)
assert sd["model.layers.0.self_attn.v_proj.weight"].shape == (1920, 2048)
assert sd["model.layers.0.self_attn.o_proj.weight"].shape == (2048, 1920)
assert len(sd) == 114

out_dir.mkdir(parents=True, exist_ok=True)
save_file(sd, str(out_dir / "model.safetensors"))
