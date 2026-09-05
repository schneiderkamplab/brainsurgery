"""T3 baseline for OLMo-1B-0724-hf: bfloat16 projection matrices, float32 everything else, 256 MiB (268,435,456 bytes) shards."""

import re
import sys
from pathlib import Path

import torch

from _ckpt import load_checkpoint, save_sharded_safetensors

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T3")
MAX_SHARD = 268435456

sd = load_checkpoint("inputs/base")
matrix_re = re.compile(r"model\.layers\.(\d+)\.(self_attn\.q_proj\.weight|self_attn\.k_proj\.weight|self_attn\.v_proj\.weight|self_attn\.o_proj\.weight|mlp\.gate_proj\.weight|mlp\.up_proj\.weight|mlp\.down_proj\.weight)")
buffer_re = re.compile(r"model\.layers\.(\d+)\.((?!))")

out = {}
for name, tensor in sd.items():
    if buffer_re.fullmatch(name):
        continue
    out[name] = tensor.float().to(torch.bfloat16) if matrix_re.fullmatch(name) else tensor.float()

assert sum(t.dtype == torch.bfloat16 for t in out.values()) == 112
assert out["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
assert out["model.embed_tokens.weight"].dtype == torch.float32
assert len(out) == 114, len(out)

save_sharded_safetensors(out, out_dir, MAX_SHARD)
