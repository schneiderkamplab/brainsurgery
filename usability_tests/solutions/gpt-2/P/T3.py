"""T3 baseline for GPT-2 (124M): bfloat16 projection matrices, float32 everything else, 64 MiB (67,108,864 bytes) shards."""

import re
import sys
from pathlib import Path

import torch

from _ckpt import load_checkpoint, save_sharded_safetensors

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T3")
MAX_SHARD = 67108864

sd = load_checkpoint("inputs/base/model.safetensors")
matrix_re = re.compile(r"h\.(\d+)\.(attn\.c_attn\.weight|attn\.c_proj\.weight|mlp\.c_fc\.weight|mlp\.c_proj\.weight)")
buffer_re = re.compile(r"h\.(\d+)\.(attn\.bias)")

out = {}
for name, tensor in sd.items():
    if buffer_re.fullmatch(name):
        continue
    out[name] = tensor.float().to(torch.bfloat16) if matrix_re.fullmatch(name) else tensor.float()

assert sum(t.dtype == torch.bfloat16 for t in out.values()) == 48
assert out["h.0.attn.c_attn.weight"].dtype == torch.bfloat16
assert out["wte.weight"].dtype == torch.float32
assert len(out) == 148, len(out)

save_sharded_safetensors(out, out_dir, MAX_SHARD)
