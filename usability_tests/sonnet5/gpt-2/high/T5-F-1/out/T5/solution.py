#!/usr/bin/env python
"""T5: merge a PEFT LoRA adapter into the GPT-2 base checkpoint and write it
sharded, with no adapter runtime involved.

Approach: plain script directly on the safetensors files (no `peft` model
instantiation needed for a pure linear-algebra merge). We:

  1. Load the base state dict and the adapter state dict.
  2. Pair up `lora_A.weight` / `lora_B.weight` tensors by module path.
  3. For each pair, compute `scale * (B @ A).T` in float32 (scale =
     lora_alpha / r) and add it to the matching base tensor. The `.T` is
     required because `fan_in_fan_out=true`: GPT-2's Conv1D base weight is
     `[in, out]`, but the LoRA factors follow the `nn.Linear` convention
     (`B @ A` is `[out, in]`).
  4. Assert the required invariants (12 pairs, no leftover `lora_` names,
     `h.0.attn.c_attn.weight` shape, 160 total tensors).
  5. Shard the merged state dict greedily by tensor byte size (<=100 MiB of
     tensor data per shard; a single oversized tensor gets its own shard)
     and write a `model.safetensors.index.json` alongside the shards.

Why a plain script instead of `peft.merge_and_unload`: that API expects to
merge into a live `nn.Module` (building the model, wrapping it with
`PeftModel`, calling `merge_and_unload`, then re-exporting) which pulls in a
full model instantiation, tokenizer/config plumbing, and PEFT's own guess at
`fan_in_fan_out` handling, none of which we need for a checkpoint-only
merge. Doing the (already fully specified) linear algebra directly is
smaller, avoids extra dependencies on PEFT's module-name matching, and lets
us assert our own invariants before writing. `mergekit`'s task-arithmetic
merges assume same-shaped dense checkpoints, not low-rank adapters, so it
doesn't apply here.
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
LORA_PATH = HERE.parent.parent / "inputs" / "lora" / "adapter_model.safetensors"
LORA_CONFIG_PATH = HERE.parent.parent / "inputs" / "lora" / "adapter_config.json"
OUT_DIR = HERE

MAX_SHARD_BYTES = 100 * 1024 * 1024  # 100 MiB, tensor data only

ADAPTER_KEY_RE = re.compile(r"^base_model\.model\.(?P<module>.+)\.lora_(?P<which>[AB])\.weight$")


def main() -> None:
    base = load_file(str(BASE_PATH))
    adapter = load_file(str(LORA_PATH))
    lora_config = json.loads(LORA_CONFIG_PATH.read_text())

    r = lora_config["r"]
    alpha = lora_config["lora_alpha"]
    fan_in_fan_out = lora_config["fan_in_fan_out"]
    if not fan_in_fan_out:
        raise AssertionError("expected fan_in_fan_out=true for GPT-2 Conv1D adapted modules")
    scale = alpha / r

    # Pair up lora_A / lora_B tensors by module path.
    a_by_module: dict[str, str] = {}
    b_by_module: dict[str, str] = {}
    for key in adapter:
        m = ADAPTER_KEY_RE.match(key)
        if not m:
            raise AssertionError(f"unrecognized adapter tensor name: {key!r}")
        module = m.group("module")
        which = m.group("which")
        (a_by_module if which == "A" else b_by_module)[module] = key

    modules = sorted(set(a_by_module) & set(b_by_module))
    if set(a_by_module) != set(b_by_module):
        missing_a = set(b_by_module) - set(a_by_module)
        missing_b = set(a_by_module) - set(b_by_module)
        raise AssertionError(f"unpaired adapter tensors: missing_A={missing_a} missing_B={missing_b}")

    merged_count = 0
    for module in modules:
        a_key = a_by_module[module]
        b_key = b_by_module[module]
        base_key = f"{module}.weight"
        if base_key not in base:
            raise AssertionError(f"adapter targets {base_key!r} but it is not in the base checkpoint")

        A = adapter[a_key].to(torch.float32)  # [r, in]
        B = adapter[b_key].to(torch.float32)  # [out, r]
        delta = scale * (B @ A).T  # Conv1D layout: [in, out]

        base_tensor = base[base_key].to(torch.float32)
        if delta.shape != base_tensor.shape:
            raise AssertionError(
                f"shape mismatch for {base_key!r}: base {tuple(base_tensor.shape)} "
                f"vs delta {tuple(delta.shape)}"
            )
        base[base_key] = (base_tensor + delta).contiguous()
        merged_count += 1

    # --- Required checks: fail loudly before writing anything. ---
    if merged_count != 12:
        raise AssertionError(f"expected exactly 12 adapter pairs merged, got {merged_count}")

    if any("lora_" in name for name in base):
        raise AssertionError("an adapter tensor leaked into the output state dict")

    expected_shape = (768, 2304)
    if tuple(base["h.0.attn.c_attn.weight"].shape) != expected_shape:
        raise AssertionError(
            f"h.0.attn.c_attn.weight has shape {tuple(base['h.0.attn.c_attn.weight'].shape)}, "
            f"expected {expected_shape}"
        )

    if len(base) != 160:
        raise AssertionError(f"expected exactly 160 tensors in the output, got {len(base)}")

    for name, tensor in base.items():
        if tensor.dtype != torch.float32:
            raise AssertionError(f"{name!r} has dtype {tensor.dtype}, expected float32")

    # --- Greedy shard packing: close the current shard before a tensor
    # would push it over the byte budget; an empty shard always accepts
    # the next tensor even if that tensor alone exceeds the budget (this
    # is exactly the case for wte.weight, which lands alone). ---
    names = list(base.keys())
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_size = 0
    for name in names:
        tensor = base[name]
        size = tensor.numel() * tensor.element_size()
        if current and current_size + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = {}
            current_size = 0
        current[name] = tensor
        current_size += size
    if current:
        shards.append(current)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    digits = max(5, len(str(n_shards)))
    for i, shard in enumerate(shards, start=1):
        filename = f"model-{i:0{digits}d}-of-{n_shards:0{digits}d}.safetensors"
        save_file(shard, str(OUT_DIR / filename), metadata={"format": "pt"})
        for name, tensor in shard.items():
            weight_map[name] = filename
            total_size += tensor.numel() * tensor.element_size()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    print(f"merged {merged_count} adapter pairs into {len(base)} tensors, {n_shards} shard(s)")


if __name__ == "__main__":
    main()
