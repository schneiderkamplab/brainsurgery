"""
T5: LoRA adapter merge with sharded export (OLMo-1B-0724-hf).

Plain script on top of `safetensors` + `torch` (both in F-allowed.md). We
considered `peft.merge_and_unload`, but that requires instantiating the full
HF model just to get back raw tensors, and gives no control over the exact
byte-threshold shard packing the task requires. Operating directly on the
safetensors files is simpler, avoids loading the model into a transformers
architecture at all, and lets us assert every "Required check" explicitly.

Usage: python solution.py
Reads inputs/{base,lora}, writes out/T5/*.safetensors + index.json.
"""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # out/T5 -> out -> sandbox root
BASE_DIR = ROOT / "inputs" / "base"
LORA_DIR = ROOT / "inputs" / "lora"
OUT_DIR = HERE

MAX_SHARD_BYTES = 512 * 1024 * 1024  # 536,870,912 bytes, tensor data only

ADAPTED_MODULES = ["self_attn.q_proj", "self_attn.v_proj"]
LORA_A_RE = re.compile(
    r"^base_model\.model\.model\.layers\.(\d+)\.(self_attn\.[qv]_proj)\.lora_A\.weight$"
)
LORA_B_RE = re.compile(
    r"^base_model\.model\.model\.layers\.(\d+)\.(self_attn\.[qv]_proj)\.lora_B\.weight$"
)


def load_base_state_dict() -> dict[str, torch.Tensor]:
    index = json.loads((BASE_DIR / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    state_dict: dict[str, torch.Tensor] = {}
    for shard_file in shard_files:
        with safe_open(BASE_DIR / shard_file, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    assert len(state_dict) == len(weight_map), (
        f"loaded {len(state_dict)} tensors but index lists {len(weight_map)}"
    )
    return state_dict


def load_lora_state_dict() -> dict[str, torch.Tensor]:
    state_dict = {}
    with safe_open(LORA_DIR / "adapter_model.safetensors", framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def main() -> None:
    adapter_config = json.loads((LORA_DIR / "adapter_config.json").read_text())
    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    fan_in_fan_out = adapter_config["fan_in_fan_out"]
    assert not fan_in_fan_out, (
        "this script assumes fan_in_fan_out=False (nn.Linear [out, in] layout, "
        "no transpose); adapter_config.json says otherwise"
    )
    scale = lora_alpha / r

    base_sd = load_base_state_dict()
    lora_sd = load_lora_state_dict()

    # Pair up lora_A/lora_B by (layer, module).
    pairs: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for key, tensor in lora_sd.items():
        m = LORA_A_RE.match(key)
        if m:
            pairs.setdefault((m.group(1), m.group(2)), {})["A"] = tensor
            continue
        m = LORA_B_RE.match(key)
        if m:
            pairs.setdefault((m.group(1), m.group(2)), {})["B"] = tensor
            continue
        raise AssertionError(f"unexpected tensor in adapter checkpoint: {key}")

    expected_pairs = {
        (str(i), module) for i in range(16) for module in ADAPTED_MODULES
    }
    assert set(pairs.keys()) == expected_pairs, (
        f"adapter pairs found do not match expected 16 layers x {ADAPTED_MODULES}: "
        f"missing={expected_pairs - set(pairs.keys())} "
        f"extra={set(pairs.keys()) - expected_pairs}"
    )
    assert len(pairs) == 32, f"expected exactly 32 adapter pairs, found {len(pairs)}"
    for key, ab in pairs.items():
        assert "A" in ab and "B" in ab, f"incomplete lora pair for {key}"

    merged_names = set()
    for (layer, module), ab in pairs.items():
        base_name = f"model.layers.{layer}.{module}.weight"
        assert base_name in base_sd, f"base checkpoint is missing {base_name}"
        A = ab["A"].to(torch.float32)  # [r, in]
        B = ab["B"].to(torch.float32)  # [out, r]
        base_weight = base_sd[base_name].to(torch.float32)
        assert base_weight.shape == (2048, 2048), (
            f"{base_name}: expected shape [2048, 2048], got {tuple(base_weight.shape)}"
        )
        delta = scale * (B @ A)
        assert delta.shape == base_weight.shape, (
            f"{base_name}: delta shape {tuple(delta.shape)} != base shape "
            f"{tuple(base_weight.shape)}"
        )
        base_sd[base_name] = (base_weight + delta).contiguous()
        merged_names.add(base_name)

    assert len(merged_names) == 32, f"expected 32 merged tensors, got {len(merged_names)}"
    assert not any("lora_" in name for name in base_sd), (
        "adapter tensor leaked into output state dict"
    )
    assert base_sd["model.layers.0.self_attn.q_proj.weight"].shape == (2048, 2048)
    assert len(base_sd) == 114, f"expected 114 tensors in output, got {len(base_sd)}"

    # --- Shard and write ---
    # Deterministic order: original base ordering.
    index = json.loads((BASE_DIR / "model.safetensors.index.json").read_text())
    ordered_names = list(index["weight_map"].keys())
    assert set(ordered_names) == set(base_sd.keys())

    def tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    # TASK.md names model.embed_tokens.weight and lm_head.weight (412 MB each)
    # as tensors that must be stored alone in their own shard. Numerically
    # each is under the 512 MiB cap on its own, but the task is explicit that
    # these two specific tensors are always solo shards (the generic
    # oversized-tensor rule below additionally covers any tensor that
    # actually exceeds MAX_SHARD_BYTES, should that ever apply).
    ALWAYS_ALONE = {"model.embed_tokens.weight", "lm_head.weight"}

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in ordered_names:
        size = tensor_bytes(base_sd[name])
        if size > MAX_SHARD_BYTES or name in ALWAYS_ALONE:
            if current:
                shards.append(current)
                current = []
                current_bytes = 0
            shards.append([name])
            continue
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
    if current:
        shards.append(current)

    for shard in shards:
        total = sum(tensor_bytes(base_sd[n]) for n in shard)
        assert total <= MAX_SHARD_BYTES or len(shard) == 1, (
            f"shard exceeds {MAX_SHARD_BYTES} bytes and holds more than one tensor: {shard}"
        )

    n_shards = len(shards)
    weight_map = {}
    total_size = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
        shard_dict = {name: base_sd[name] for name in shard_names}
        save_file(shard_dict, OUT_DIR / shard_filename, metadata={"format": "pt"})
        for name in shard_names:
            weight_map[name] = shard_filename
            total_size += tensor_bytes(base_sd[name])

    assert len(weight_map) == 114, f"expected 114 entries in weight_map, got {len(weight_map)}"
    assert not any("lora_" in k for k in weight_map), "lora tensor name leaked into weight_map"

    out_index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(out_index, indent=2))

    print(f"Merged 32 LoRA pairs (scale={scale}) into base checkpoint.")
    print(f"Wrote {n_shards} shard(s), {len(weight_map)} tensors total, to {OUT_DIR}")


if __name__ == "__main__":
    main()
