"""T3 baseline for Pythia-1B: bfloat16 projection matrices, float32 everything else, 256 MiB (268,435,456 bytes) shards."""

import re
import sys
from pathlib import Path

import torch

from _ckpt import load_checkpoint, save_sharded_safetensors

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T3")
MAX_SHARD = 268435456

sd = load_checkpoint("inputs/base/model.safetensors")
matrix_re = re.compile(r"gpt_neox\.layers\.(\d+)\.(attention\.query_key_value\.weight|attention\.dense\.weight|mlp\.dense_h_to_4h\.weight|mlp\.dense_4h_to_h\.weight)")
buffer_re = re.compile(r"gpt_neox\.layers\.(\d+)\.(attention\.bias|attention\.masked_bias|attention\.rotary_emb\.inv_freq)")

out = {}
for name, tensor in sd.items():
    if buffer_re.fullmatch(name):
        continue
    out[name] = tensor.float().to(torch.bfloat16) if matrix_re.fullmatch(name) else tensor.float()

assert sum(t.dtype == torch.bfloat16 for t in out.values()) == 64
assert out["gpt_neox.layers.0.attention.query_key_value.weight"].dtype == torch.bfloat16
assert out["gpt_neox.embed_in.weight"].dtype == torch.float32
assert len(out) == 196, len(out)

save_sharded_safetensors(out, out_dir, MAX_SHARD)
