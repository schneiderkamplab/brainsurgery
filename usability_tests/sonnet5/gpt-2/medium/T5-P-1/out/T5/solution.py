"""Merge a LoRA adapter into a GPT-2 base checkpoint and write it sharded.

Standalone script: torch + safetensors only, no brainsurgery.
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root (out/T5/solution.py -> out -> root)
BASE_PATH = ROOT / "inputs" / "base" / "model.safetensors"
LORA_PATH = ROOT / "inputs" / "lora" / "adapter_model.safetensors"
LORA_CONFIG_PATH = ROOT / "inputs" / "lora" / "adapter_config.json"
OUT_DIR = HERE
SHARD_BUDGET_BYTES = 100 * 1024 * 1024  # 100 MiB of tensor data per shard

# --- load everything ---------------------------------------------------

with LORA_CONFIG_PATH.open() as f:
    lora_config = json.load(f)

r = lora_config["r"]
lora_alpha = lora_config["lora_alpha"]
fan_in_fan_out = lora_config["fan_in_fan_out"]
scale = lora_alpha / r

assert fan_in_fan_out is True, (
    f"expected fan_in_fan_out=true (Conv1D base layout), got {fan_in_fan_out!r}"
)

base_tensors = {}
with safe_open(str(BASE_PATH), framework="pt") as f:
    for key in f.keys():
        base_tensors[key] = f.get_tensor(key)

lora_tensors = {}
with safe_open(str(LORA_PATH), framework="pt") as f:
    for key in f.keys():
        lora_tensors[key] = f.get_tensor(key)

# --- map adapter names to base names and merge --------------------------

# adapter tensor names look like:
#   base_model.model.h.<i>.attn.c_attn.lora_A.weight
#   base_model.model.h.<i>.attn.c_attn.lora_B.weight
lora_a_re = re.compile(r"^base_model\.model\.(?P<base>.+)\.lora_A\.weight$")

pairs = []  # (base_name, A, B)
for name, tensor in lora_tensors.items():
    m = lora_a_re.match(name)
    if m is None:
        continue
    base_name = m.group("base")
    b_name = name.replace(".lora_A.weight", ".lora_B.weight")
    if b_name not in lora_tensors:
        raise RuntimeError(f"found lora_A without matching lora_B for {base_name}")
    a_tensor = tensor
    b_tensor = lora_tensors[b_name]
    pairs.append((base_name + ".weight", a_tensor, b_tensor))

assert len(pairs) == 12, f"expected exactly 12 adapter pairs, found {len(pairs)}"

merged_names = set()
for base_name, a_tensor, b_tensor in pairs:
    assert base_name in base_tensors, f"base tensor {base_name} not found"
    base_weight = base_tensors[base_name]

    a32 = a_tensor.to(torch.float32)
    b32 = b_tensor.to(torch.float32)
    base32 = base_weight.to(torch.float32)

    delta = scale * (b32 @ a32).T  # [out, in].T -> [in, out], matches Conv1D layout

    assert delta.shape == base32.shape, (
        f"shape mismatch merging into {base_name}: delta {tuple(delta.shape)} "
        f"vs base {tuple(base32.shape)}"
    )

    merged = (base32 + delta).contiguous()
    base_tensors[base_name] = merged.to(base_weight.dtype)
    merged_names.add(base_name)

assert len(merged_names) == 12, f"expected 12 distinct merged tensors, got {len(merged_names)}"

# --- required checks -----------------------------------------------------

for name in base_tensors:
    assert "lora_" not in name, f"adapter tensor leaked into output: {name}"

assert len(base_tensors) == 160, f"expected 160 tensors in output, got {len(base_tensors)}"

assert tuple(base_tensors["h.0.attn.c_attn.weight"].shape) == (768, 2304), (
    f"h.0.attn.c_attn.weight has unexpected shape "
    f"{tuple(base_tensors['h.0.attn.c_attn.weight'].shape)}"
)

# --- shard and write -------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.element_size() * t.nelement()


# Preserve base file's tensor order for deterministic, stable sharding.
ordered_names = list(base_tensors.keys())

shards: list[list[str]] = []
current_shard: list[str] = []
current_size = 0
for name in ordered_names:
    size = tensor_nbytes(base_tensors[name])
    if size > SHARD_BUDGET_BYTES:
        # oversized tensor gets its own shard
        if current_shard:
            shards.append(current_shard)
            current_shard = []
            current_size = 0
        shards.append([name])
        continue
    if current_shard and current_size + size > SHARD_BUDGET_BYTES:
        shards.append(current_shard)
        current_shard = []
        current_size = 0
    current_shard.append(name)
    current_size += size
if current_shard:
    shards.append(current_shard)

n_shards = len(shards)
weight_map = {}
total_size = 0
for idx, shard_names in enumerate(shards, start=1):
    shard_filename = f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
    shard_tensors = {name: base_tensors[name] for name in shard_names}
    save_file(shard_tensors, str(OUT_DIR / shard_filename), metadata={"format": "pt"})
    for name in shard_names:
        weight_map[name] = shard_filename
        total_size += tensor_nbytes(base_tensors[name])

index = {
    "metadata": {"total_size": total_size},
    "weight_map": weight_map,
}
with (OUT_DIR / "model.safetensors.index.json").open("w") as f:
    json.dump(index, f, indent=2, sort_keys=True)

print(f"Merged {len(merged_names)} LoRA pairs into base weights.")
print(f"Wrote {len(base_tensors)} tensors across {n_shards} shard(s) to {OUT_DIR}")
