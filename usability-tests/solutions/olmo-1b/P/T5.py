"""T5 baseline for OLMo-1B-0724-hf: merge a PEFT-style LoRA adapter into the base weights, 512 MiB (536,870,912 bytes) shards."""

import json
import sys
from pathlib import Path

from safetensors.torch import load_file

from _ckpt import load_checkpoint, save_sharded_safetensors

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out/T5")
N_LAYERS = 16
MODULES = ["self_attn.q_proj", "self_attn.v_proj"]
MAX_SHARD = 536870912

sd = load_checkpoint("inputs/base")
adapter = load_file("inputs/lora/adapter_model.safetensors")
config = json.loads(Path("inputs/lora/adapter_config.json").read_text())
scale = config["lora_alpha"] / config["r"]
transpose = config["fan_in_fan_out"]

merged = 0
for layer in range(N_LAYERS):
    for module in MODULES:
        prefix = "base_model.model." + "model.layers.{i}.".format(i=layer) + module
        a = adapter[f"{prefix}.lora_A.weight"]  # [r, in]
        b = adapter[f"{prefix}.lora_B.weight"]  # [out, r]
        key = "model.layers.{i}.".format(i=layer) + module + ".weight"
        delta = b @ a
        if transpose:
            delta = delta.T
        sd[key] = (sd[key].float() + scale * delta).to(sd[key].dtype)
        merged += 1

assert merged == 32
assert not any("lora_" in name for name in sd)
assert sd["model.layers.0.self_attn.q_proj.weight"].shape == (2048, 2048)
assert len(sd) == 114

save_sharded_safetensors(sd, out_dir, MAX_SHARD)
