"""
T3: Mixed-precision export with sharding (Pythia-1B).

Cast the 64 large projection matrices (query_key_value, dense,
dense_h_to_4h, dense_4h_to_h weights, per layer) to bfloat16. Upcast every
other tensor to float32. Drop the 48 non-parameter buffers (attention.bias,
attention.masked_bias, attention.rotary_emb.inv_freq per layer). Write the
result as a sharded safetensors checkpoint with a HF-style index file, each
shard capped at 256 MiB of tensor data (oversized tensors get their own
shard).

Tools: torch + safetensors for tensor I/O and dtype casts,
`transformers.modeling_utils.split_torch_state_dict_into_shards` (which
wraps `huggingface_hub.split_torch_state_dict_into_shards`) to compute the
shard layout and index -- this is the standard HF sharded-export routine
(F-allowed.md calls out "transformers ... sharded export" for this task).
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.modeling_utils import split_torch_state_dict_into_shards

HERE = Path(__file__).resolve().parent
IN_PATH = HERE.parents[1] / "inputs" / "base" / "model.safetensors"
OUT_DIR = HERE  # out/T3/

MAX_SHARD_BYTES = 256 * 1024 * 1024  # 256 MiB

# The 4 projection weight name patterns per layer that must become bfloat16.
BF16_PATTERNS = [
    re.compile(r"^gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight$"),
    re.compile(r"^gpt_neox\.layers\.\d+\.attention\.dense\.weight$"),
    re.compile(r"^gpt_neox\.layers\.\d+\.mlp\.dense_h_to_4h\.weight$"),
    re.compile(r"^gpt_neox\.layers\.\d+\.mlp\.dense_4h_to_h\.weight$"),
]

# The 3 non-parameter buffers per layer that must be dropped.
BUFFER_PATTERNS = [
    re.compile(r"^gpt_neox\.layers\.\d+\.attention\.bias$"),
    re.compile(r"^gpt_neox\.layers\.\d+\.attention\.masked_bias$"),
    re.compile(r"^gpt_neox\.layers\.\d+\.attention\.rotary_emb\.inv_freq$"),
]


def is_bf16_target(name: str) -> bool:
    return any(p.match(name) for p in BF16_PATTERNS)


def is_buffer(name: str) -> bool:
    return any(p.match(name) for p in BUFFER_PATTERNS)


def main() -> None:
    if not IN_PATH.exists():
        sys.exit(f"input checkpoint not found: {IN_PATH}")

    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(str(IN_PATH), framework="pt") as f:
        names = list(f.keys())
        for name in names:
            if is_buffer(name):
                continue
            tensor = f.get_tensor(name)
            if is_bf16_target(name):
                tensor = tensor.to(torch.bfloat16)
            else:
                tensor = tensor.to(torch.float32)
            state_dict[name] = tensor.contiguous()

    # --- Required checks: fail loudly before writing anything. ---
    bf16_names = [n for n, t in state_dict.items() if t.dtype == torch.bfloat16]
    if len(bf16_names) != 64:
        sys.exit(f"expected exactly 64 bfloat16 tensors, got {len(bf16_names)}: {bf16_names}")

    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    if state_dict[probe].dtype != torch.bfloat16:
        sys.exit(f"{probe} must be bfloat16, got {state_dict[probe].dtype}")

    if state_dict["gpt_neox.embed_in.weight"].dtype != torch.float32:
        sys.exit(
            "gpt_neox.embed_in.weight must be float32, got "
            f"{state_dict['gpt_neox.embed_in.weight'].dtype}"
        )

    if len(state_dict) != 196:
        sys.exit(f"expected exactly 196 tensors in output, got {len(state_dict)}")

    for name in state_dict:
        if is_buffer(name):
            sys.exit(f"buffer {name} leaked into output state_dict")
    for pat in BUFFER_PATTERNS:
        if any(pat.match(n) for n in state_dict):
            sys.exit(f"buffer pattern {pat.pattern} still present in output")

    non_bf16_non_f32 = [
        n for n, t in state_dict.items() if t.dtype not in (torch.bfloat16, torch.float32)
    ]
    if non_bf16_non_f32:
        sys.exit(f"tensors with unexpected dtype: {non_bf16_non_f32}")

    # --- Shard and write. ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split = split_torch_state_dict_into_shards(state_dict, max_shard_size=MAX_SHARD_BYTES)

    for filename, tensor_names in split.filename_to_tensors.items():
        shard = {name: state_dict[name] for name in tensor_names}
        save_file(shard, str(OUT_DIR / filename), metadata={"format": "pt"})

    if split.is_sharded:
        index = {
            "metadata": split.metadata,
            "weight_map": split.tensor_to_filename,
        }
        with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
    else:
        # split_torch_state_dict_into_shards only sets is_sharded=True when there is
        # more than one shard; the task requires an index file regardless.
        (filename,) = split.filename_to_tensors.keys()
        index = {
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state_dict.values())},
            "weight_map": {name: filename for name in state_dict},
        }
        with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

    print(f"wrote {len(state_dict)} tensors across {len(split.filename_to_tensors)} shard(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
