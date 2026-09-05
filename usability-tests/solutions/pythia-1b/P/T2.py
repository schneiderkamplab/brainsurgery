"""T2 baseline for Pythia-1B: remove attention head 5 from every layer."""

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
    ("attention.query_key_value.weight", 0, 1, 6144, 768),
    ("attention.query_key_value.bias", 0, 1, 6144, 768),
    ("attention.dense.weight", 1, 1, 2048, 256),
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
        name = "gpt_neox.layers.{i}.".format(i=layer) + rel
        sd[name] = sd[name].index_select(dim, keep_index(segments, seg_size, block)).contiguous()

assert sd["gpt_neox.layers.0.attention.query_key_value.weight"].shape == (5376, 2048)
assert sd["gpt_neox.layers.0.attention.query_key_value.bias"].shape == (5376,)
assert sd["gpt_neox.layers.0.attention.dense.weight"].shape == (2048, 1792)
assert len(sd) == 244

out_dir.mkdir(parents=True, exist_ok=True)
save_file(sd, str(out_dir / "model.safetensors"))
