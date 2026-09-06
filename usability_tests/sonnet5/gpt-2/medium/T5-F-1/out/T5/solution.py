"""
T5: LoRA adapter merge with sharded export (GPT-2 124M).

Loads the base checkpoint and the PEFT-style LoRA adapter directly as
safetensors state dicts (no model instantiation needed), merges the adapter
into the 12 targeted `attn.c_attn` weights, checks the required invariants,
and writes a sharded safetensors checkpoint with a HF-compatible index.

Tools: `safetensors` for I/O, `torch` for the merge arithmetic, and
`huggingface_hub.split_torch_state_dict_into_shards` for the sharding /
index-file logic (the same helper `transformers`' sharded `save_pretrained`
uses under the hood).
"""

import json
import re
import sys
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root
INPUTS = ROOT / "inputs"
OUT_DIR = ROOT / "out" / "T5"

MAX_SHARD_BYTES = 100 * 1024 * 1024  # 100 MiB, tensor data only

LORA_A_RE = re.compile(r"^base_model\.model\.(?P<base>.+)\.lora_A\.weight$")
LORA_B_RE = re.compile(r"^base_model\.model\.(?P<base>.+)\.lora_B\.weight$")


def main() -> None:
    base_sd = load_file(INPUTS / "base" / "model.safetensors")
    lora_sd = load_file(INPUTS / "lora" / "adapter_model.safetensors")
    adapter_config = json.loads((INPUTS / "lora" / "adapter_config.json").read_text())

    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    fan_in_fan_out = adapter_config["fan_in_fan_out"]
    scale = lora_alpha / r

    if not fan_in_fan_out:
        raise AssertionError(
            "expected fan_in_fan_out=true (Conv1D layout); merge math below assumes it"
        )

    # Map each adapted base tensor name -> (A, B) factor pair.
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in lora_sd.items():
        m_a = LORA_A_RE.match(name)
        m_b = LORA_B_RE.match(name)
        if m_a:
            pairs.setdefault(m_a.group("base") + ".weight", {})["A"] = tensor
        elif m_b:
            pairs.setdefault(m_b.group("base") + ".weight", {})["B"] = tensor
        else:
            raise AssertionError(f"unrecognized adapter tensor name: {name}")

    # --- Required check: exactly 12 adapter pairs found ---
    if len(pairs) != 12:
        raise AssertionError(f"expected exactly 12 adapter pairs, found {len(pairs)}")
    for base_name, factors in pairs.items():
        if set(factors) != {"A", "B"}:
            raise AssertionError(f"incomplete adapter pair for {base_name}: {sorted(factors)}")

    merged_sd = dict(base_sd)
    for base_name, factors in pairs.items():
        if base_name not in base_sd:
            raise AssertionError(f"adapter targets missing base tensor: {base_name}")
        weight = base_sd[base_name]
        a = factors["A"].to(torch.float32)  # [r, in]
        b = factors["B"].to(torch.float32)  # [out, r]
        delta = scale * (b @ a).T  # Conv1D layout [in, out]
        if delta.shape != weight.shape:
            raise AssertionError(
                f"shape mismatch for {base_name}: delta {tuple(delta.shape)} "
                f"vs weight {tuple(weight.shape)}"
            )
        merged = (weight.to(torch.float32) + delta).to(weight.dtype)
        merged_sd[base_name] = merged.contiguous()

    # --- Required check: no lora_* tensor leaks into the output ---
    if any("lora_" in k for k in merged_sd):
        raise AssertionError("adapter tensor name leaked into merged state dict")

    # --- Required check: h.0.attn.c_attn.weight keeps its shape ---
    probe = "h.0.attn.c_attn.weight"
    if merged_sd[probe].shape != torch.Size([768, 2304]):
        raise AssertionError(f"{probe} has unexpected shape {tuple(merged_sd[probe].shape)}")

    # --- Required check: exactly 160 tensors, same names as base ---
    if len(merged_sd) != 160:
        raise AssertionError(f"expected 160 tensors in output, found {len(merged_sd)}")
    if set(merged_sd) != set(base_sd):
        raise AssertionError("output tensor names differ from base tensor names")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split = split_torch_state_dict_into_shards(merged_sd, max_shard_size=MAX_SHARD_BYTES)
    for filename, tensor_names in split.filename_to_tensors.items():
        shard = {name: merged_sd[name] for name in tensor_names}
        save_file(shard, OUT_DIR / filename, metadata={"format": "pt"})

    if split.is_sharded:
        index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
        (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    else:
        # Only reachable if the whole checkpoint fit in one shard; the spec
        # still requires an index file mapping every tensor to its shard.
        (filename,) = split.filename_to_tensors.keys()
        index = {
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in merged_sd.values())},
            "weight_map": {name: filename for name in merged_sd},
        }
        (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    print(f"Merged 12 LoRA pairs, wrote {len(merged_sd)} tensors to {OUT_DIR}")
    print(f"Shards: {sorted(split.filename_to_tensors.keys())}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
