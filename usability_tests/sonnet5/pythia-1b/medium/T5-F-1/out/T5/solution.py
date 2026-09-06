"""
T5: LoRA adapter merge with sharded export (Pythia-1B), condition F.

Approach: read the base safetensors file and the PEFT adapter file directly
(no model instantiation), merge each LoRA pair into its base weight
(scale = lora_alpha / r, computed in float32, cast back to the base dtype),
and write a sharded safetensors checkpoint with an index, using
`huggingface_hub.split_torch_state_dict_into_shards` -- the same shard-packing
routine `transformers.PreTrainedModel.save_pretrained(..., safe_serialization=True)`
uses -- so the shard layout matches standard HF tooling.

Why this over `peft.merge_and_unload`: that path requires instantiating the
full GPTNeoXForCausalLM, loading the adapter onto it, merging, then saving.
It works here (config.json/tokenizer are present), but pulls in a transformers
forward-pass-capable model object just to move tensors around, and gives
implicit control over the float32 accumulation this task requires explicitly.
Operating directly on the state dicts is smaller, auditable, and matches the
task's own framing ("avoids instantiating the model").
"""

import json
import re
import sys
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import safe_open, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root (out/T5/solution.py -> sandbox)
INPUTS = ROOT / "inputs"
OUT_DIR = HERE  # out/T5/

BASE_PATH = INPUTS / "base" / "model.safetensors"
ADAPTER_PATH = INPUTS / "lora" / "adapter_model.safetensors"
ADAPTER_CONFIG_PATH = INPUTS / "lora" / "adapter_config.json"

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 512 MiB

LORA_A_RE = re.compile(
    r"^base_model\.model\.(?P<base_name>.+)\.lora_A\.weight$"
)


def load_all(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    adapter_config = json.loads(ADAPTER_CONFIG_PATH.read_text())
    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    fan_in_fan_out = adapter_config.get("fan_in_fan_out", False)
    if fan_in_fan_out:
        raise SystemExit(
            "fan_in_fan_out=true is not handled by this script "
            "(base weight layout would need transposing before add)"
        )
    scale = lora_alpha / r

    base = load_all(BASE_PATH)
    adapter = load_all(ADAPTER_PATH)

    # --- discover and validate adapter pairs -------------------------------
    pairs = []  # (base_name, A_key, B_key)
    for key in adapter:
        m = LORA_A_RE.match(key)
        if not m:
            continue
        base_name = m.group("base_name") + ".weight"
        a_key = key
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in adapter:
            raise SystemExit(f"found {a_key} with no matching {b_key}")
        pairs.append((base_name, a_key, b_key))

    if len(pairs) != 16:
        raise SystemExit(
            f"REQUIRED CHECK FAILED: expected exactly 16 adapter pairs, found {len(pairs)}"
        )

    merged_base_names = set()
    for base_name, a_key, b_key in pairs:
        if base_name not in base:
            raise SystemExit(f"adapter targets {base_name!r}, not present in base checkpoint")
        A = adapter[a_key]
        B = adapter[b_key]
        W = base[base_name]

        if A.shape[1] != W.shape[1] or B.shape[0] != W.shape[0] or A.shape[0] != B.shape[1]:
            raise SystemExit(
                f"REQUIRED CHECK FAILED: shape mismatch for {base_name}: "
                f"A={tuple(A.shape)} B={tuple(B.shape)} W={tuple(W.shape)}"
            )

        delta = scale * (B.to(torch.float32) @ A.to(torch.float32))
        merged = (W.to(torch.float32) + delta).to(W.dtype)

        if merged.shape != W.shape:
            raise SystemExit(
                f"REQUIRED CHECK FAILED: merged shape {tuple(merged.shape)} != "
                f"original shape {tuple(W.shape)} for {base_name}"
            )

        base[base_name] = merged.contiguous()
        merged_base_names.add(base_name)

    if len(merged_base_names) != 16:
        raise SystemExit(
            "REQUIRED CHECK FAILED: expected 16 distinct merged base tensors, "
            f"got {len(merged_base_names)}"
        )

    qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    if qkv0 not in base or tuple(base[qkv0].shape) != (6144, 2048):
        raise SystemExit(
            f"REQUIRED CHECK FAILED: {qkv0} has shape "
            f"{tuple(base.get(qkv0, torch.empty(0)).shape)}, expected (6144, 2048)"
        )

    if any("lora_" in name for name in base):
        raise SystemExit("REQUIRED CHECK FAILED: an adapter tensor leaked into the output")

    if len(base) != 244:
        raise SystemExit(f"REQUIRED CHECK FAILED: output has {len(base)} tensors, expected 244")

    # --- shard and write ----------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split = split_torch_state_dict_into_shards(base, max_shard_size=MAX_SHARD_BYTES)

    for filename, tensor_names in split.filename_to_tensors.items():
        shard = {name: base[name].contiguous() for name in tensor_names}
        save_file(shard, str(OUT_DIR / filename), metadata={"format": "pt"})

    if split.is_sharded:
        index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
        (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    else:
        raise SystemExit(
            "REQUIRED CHECK FAILED: checkpoint fit in a single shard; task requires "
            "a sharded output with an index file"
        )

    total_shard_bytes = sum(
        base[name].numel() * base[name].element_size() for name in base
    )
    print(f"Merged {len(pairs)} LoRA pairs into {len(merged_base_names)} base tensors.")
    print(f"Wrote {len(split.filename_to_tensors)} shards, {len(base)} tensors total, "
          f"{total_shard_bytes / (1024**2):.1f} MiB of tensor data, to {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
