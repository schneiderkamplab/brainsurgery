"""
T5: LoRA adapter merge with sharded export (GPT-2 124M).

Standalone script: load base checkpoint + PEFT-style LoRA adapter, merge the
adapter into the base weights, and write the result as a sharded safetensors
checkpoint (<=100 MiB tensor data per shard, except a lone oversized tensor).
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent.parent / "inputs"
BASE_PATH = INPUTS / "base" / "model.safetensors"
ADAPTER_PATH = INPUTS / "lora" / "adapter_model.safetensors"
ADAPTER_CONFIG_PATH = INPUTS / "lora" / "adapter_config.json"
OUT_DIR = HERE
SHARD_MAX_BYTES = 100 * 1024 * 1024  # 100 MiB, tensor data only

LORA_A_RE = re.compile(r"^base_model\.model\.h\.(\d+)\.(.+)\.lora_A\.weight$")
LORA_B_RE = re.compile(r"^base_model\.model\.h\.(\d+)\.(.+)\.lora_B\.weight$")


def load_all(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    base = load_all(BASE_PATH)
    adapter = load_all(ADAPTER_PATH)
    config = json.loads(ADAPTER_CONFIG_PATH.read_text())

    if not config.get("fan_in_fan_out", False):
        raise RuntimeError("expected fan_in_fan_out=true; merge math assumes Conv1D layout")

    r = config["r"]
    lora_alpha = config["lora_alpha"]
    scale = lora_alpha / r

    # Pair up lora_A / lora_B tensors by (layer, module).
    a_tensors: dict[tuple[str, str], torch.Tensor] = {}
    b_tensors: dict[tuple[str, str], torch.Tensor] = {}
    for key, tensor in adapter.items():
        m = LORA_A_RE.match(key)
        if m:
            a_tensors[(m.group(1), m.group(2))] = tensor
            continue
        m = LORA_B_RE.match(key)
        if m:
            b_tensors[(m.group(1), m.group(2))] = tensor
            continue
        raise RuntimeError(f"unexpected adapter tensor name: {key}")

    pairs = sorted(set(a_tensors) & set(b_tensors))
    if set(a_tensors) != set(b_tensors):
        raise RuntimeError("mismatched lora_A/lora_B pairs in adapter checkpoint")
    if len(pairs) != 12:
        raise RuntimeError(f"expected exactly 12 adapter pairs, found {len(pairs)}")

    merged = dict(base)
    merged_count = 0
    for layer, module in pairs:
        base_name = f"h.{layer}.{module}.weight"
        if base_name not in merged:
            raise RuntimeError(f"base tensor {base_name!r} not found for adapter pair")

        A = a_tensors[(layer, module)].to(torch.float32)  # [r, in]
        B = b_tensors[(layer, module)].to(torch.float32)  # [out, r]
        base_weight = merged[base_name].to(torch.float32)

        delta = scale * (B @ A).T  # nn.Linear layout -> transpose to Conv1D [in, out]
        if delta.shape != base_weight.shape:
            raise RuntimeError(
                f"shape mismatch merging {base_name}: delta {tuple(delta.shape)} "
                f"vs base {tuple(base_weight.shape)}"
            )

        merged[base_name] = (base_weight + delta).contiguous()
        merged_count += 1

    if merged_count != 12:
        raise RuntimeError(f"expected to merge exactly 12 tensors, merged {merged_count}")

    for name in merged:
        if "lora_" in name:
            raise RuntimeError(f"adapter tensor leaked into output: {name}")

    if merged["h.0.attn.c_attn.weight"].shape != (768, 2304):
        raise RuntimeError("h.0.attn.c_attn.weight has unexpected shape after merge")

    if len(merged) != 160:
        raise RuntimeError(f"expected exactly 160 tensors in output, got {len(merged)}")

    write_sharded(merged, OUT_DIR)
    print(f"Merged {merged_count} adapter pairs into {len(merged)} tensors.")
    print(f"Wrote sharded checkpoint to {OUT_DIR}")


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def write_sharded(tensors: dict[str, torch.Tensor], out_dir: Path) -> None:
    # Greedily bin tensors into shards, each <= SHARD_MAX_BYTES of tensor data,
    # except a single tensor that alone exceeds the limit gets its own shard.
    names = list(tensors.keys())
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0

    for name in names:
        size = tensor_nbytes(tensors[name])
        if size > SHARD_MAX_BYTES:
            if current:
                shards.append(current)
                current = []
                current_size = 0
            shards.append([name])
            continue
        if current and current_size + size > SHARD_MAX_BYTES:
            shards.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += size

    if current:
        shards.append(current)

    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0

    for idx, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {name: tensors[name] for name in shard_names}
        save_file(shard_tensors, str(out_dir / shard_filename))
        for name in shard_names:
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(tensors[name])

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
