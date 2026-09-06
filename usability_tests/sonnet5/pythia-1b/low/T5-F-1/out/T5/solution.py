"""
T5: LoRA adapter merge with sharded export (Pythia-1B).

Plain script on top of `safetensors` + `torch` (no brainsurgery). peft's
`merge_and_unload` requires instantiating the actual model; here we work
directly on the checkpoint tensors as the task asks, which also makes the
required checks (adapter-pair count, no lora_ names, shape, tensor count)
trivial to assert before writing.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
LORA_PATH = HERE.parent.parent / "inputs" / "lora" / "adapter_model.safetensors"
ADAPTER_CONFIG_PATH = HERE.parent.parent / "inputs" / "lora" / "adapter_config.json"
OUT_DIR = HERE.parent / "T5"
MAX_SHARD_BYTES = 512 * 1024 * 1024

LORA_A_RE = re.compile(
    r"^base_model\.model\.gpt_neox\.layers\.(\d+)\.(.+)\.lora_A\.weight$"
)
LORA_B_RE = re.compile(
    r"^base_model\.model\.gpt_neox\.layers\.(\d+)\.(.+)\.lora_B\.weight$"
)


def main() -> None:
    adapter_config = json.loads(ADAPTER_CONFIG_PATH.read_text())
    assert adapter_config["fan_in_fan_out"] is False, (
        "solution assumes fan_in_fan_out=False (no transposition needed)"
    )
    scale = adapter_config["lora_alpha"] / adapter_config["r"]

    # --- load base tensors ---
    base_tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(BASE_PATH), framework="pt") as f:
        for key in f.keys():
            base_tensors[key] = f.get_tensor(key)

    # --- load adapter tensors and pair A/B by (layer, module) ---
    lora_a: dict[tuple[str, str], torch.Tensor] = {}
    lora_b: dict[tuple[str, str], torch.Tensor] = {}
    with safe_open(str(LORA_PATH), framework="pt") as f:
        for key in f.keys():
            m = LORA_A_RE.match(key)
            if m:
                lora_a[(m.group(1), m.group(2))] = f.get_tensor(key)
                continue
            m = LORA_B_RE.match(key)
            if m:
                lora_b[(m.group(1), m.group(2))] = f.get_tensor(key)
                continue
            raise AssertionError(f"unexpected adapter tensor name: {key}")

    pairs = sorted(set(lora_a) & set(lora_b))
    if set(lora_a) != set(lora_b):
        raise AssertionError("mismatched lora_A/lora_B keys in adapter checkpoint")

    # Required check: exactly 16 adapter pairs.
    assert len(pairs) == 16, f"expected 16 adapter pairs, found {len(pairs)}"

    for layer, module in pairs:
        base_key = f"gpt_neox.layers.{layer}.{module}.weight"
        assert base_key in base_tensors, f"missing base tensor {base_key}"

        base_w = base_tensors[base_key]
        base_dtype = base_w.dtype
        A = lora_a[(layer, module)].to(torch.float32)  # [r, in]
        B = lora_b[(layer, module)].to(torch.float32)  # [out, r]

        delta = scale * (B @ A)  # [out, in], float32
        assert delta.shape == base_w.shape, (
            f"shape mismatch for {base_key}: delta {delta.shape} vs base {base_w.shape}"
        )

        merged = (base_w.to(torch.float32) + delta).to(base_dtype)
        base_tensors[base_key] = merged

    # --- required checks before writing ---
    assert not any("lora_" in k for k in base_tensors), (
        "adapter tensor leaked into output"
    )
    qkv0 = base_tensors.get("gpt_neox.layers.0.attention.query_key_value.weight")
    assert qkv0 is not None and tuple(qkv0.shape) == (6144, 2048), (
        f"gpt_neox.layers.0.attention.query_key_value.weight has wrong shape: "
        f"{None if qkv0 is None else tuple(qkv0.shape)}"
    )
    assert len(base_tensors) == 244, f"expected 244 tensors, got {len(base_tensors)}"

    write_sharded(base_tensors)
    print(f"OK: wrote {len(base_tensors)} tensors to {OUT_DIR}")


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def write_sharded(tensors: dict[str, torch.Tensor]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Greedily bin-pack tensors into shards of at most MAX_SHARD_BYTES.
    # A tensor larger than the limit gets its own shard.
    items = list(tensors.items())
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    for name, t in items:
        nbytes = tensor_nbytes(t)
        if nbytes > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current = []
                current_bytes = 0
            shards.append([name])
            continue
        if current and current_bytes + nbytes > MAX_SHARD_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += nbytes
    if current:
        shards.append(current)

    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0

    for idx, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {name: tensors[name].contiguous() for name in shard_names}
        save_file(shard_tensors, str(OUT_DIR / shard_filename))
        for name in shard_names:
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(tensors[name])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (OUT_DIR / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAILED CHECK: {e}", file=sys.stderr)
        sys.exit(1)
