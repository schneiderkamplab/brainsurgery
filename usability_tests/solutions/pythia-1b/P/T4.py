"""T4 baseline for Pythia-1B: task-vector merge of two fine-tunes, lambda 0.4 each, MLP tensors only."""

import re
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from _ckpt import load_checkpoint

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T4")
LAMBDA = 0.4

base = load_checkpoint("inputs/base/model.safetensors")
ft1 = load_checkpoint("inputs/ft1/model.safetensors")
ft2 = load_checkpoint("inputs/ft2/model.safetensors")
mlp_re = re.compile(r"gpt_neox\.layers\.(\d+)\.(mlp\.dense_h_to_4h\.weight|mlp\.dense_h_to_4h\.bias|mlp\.dense_4h_to_h\.weight|mlp\.dense_4h_to_h\.bias)")

assert set(base) == set(ft1) == set(ft2)
for name in base:
    if not mlp_re.fullmatch(name):
        assert torch.equal(base[name], ft1[name]), f"ft1 differs on shared tensor {name}"
        assert torch.equal(base[name], ft2[name]), f"ft2 differs on shared tensor {name}"

out = dict(base)
merged = 0
for name in base:
    if mlp_re.fullmatch(name):
        b, f1, f2 = base[name].float(), ft1[name].float(), ft2[name].float()
        out[name] = (b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)).to(base[name].dtype)
        merged += 1
assert merged == 64
assert len(out) == 244

out_dir.mkdir(parents=True, exist_ok=True)
save_file(out, str(out_dir / "model.safetensors"))
