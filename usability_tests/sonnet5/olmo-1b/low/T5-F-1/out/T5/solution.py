"""T5: LoRA adapter merge with sharded export (OLMo-1B-0724-hf).

Plain script on top of `safetensors` + `torch`. No brainsurgery, no peft
model instantiation: we operate directly on the checkpoint tensors, which is
simpler and cheaper than loading the full HF model just to call
`merge_and_unload`.

Merge rule (fan_in_fan_out = false, base layout is nn.Linear [out, in]):
    W' = W + (alpha / r) * B @ A
for every layer i in 0..15 and module in {self_attn.q_proj, self_attn.v_proj}.

Output: out/T5/, sharded safetensors, each shard <= 512 MiB of tensor data
(a single oversize tensor gets its own shard), plus a model.safetensors.index.json.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
TASK_DIR = HERE.parent.parent
INPUTS = TASK_DIR / "inputs"
BASE_DIR = INPUTS / "base"
LORA_DIR = INPUTS / "lora"
OUT_DIR = TASK_DIR / "out" / "T5"

SHARD_LIMIT_BYTES = 512 * 1024 * 1024
LORA_NAME_RE = re.compile(
    r"^base_model\.model\.model\.layers\.(\d+)\.(self_attn\.(?:q|v)_proj)\.lora_(A|B)\.weight$"
)


def load_adapter_config() -> tuple[float, list[str]]:
    with open(LORA_DIR / "adapter_config.json") as f:
        cfg = json.load(f)
    if cfg.get("fan_in_fan_out", False):
        raise RuntimeError("fan_in_fan_out=true is not supported by this script")
    scale = cfg["lora_alpha"] / cfg["r"]
    return scale, cfg["target_modules"]


def load_lora_pairs() -> dict[tuple[int, str], dict[str, torch.Tensor]]:
    pairs: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    with safe_open(LORA_DIR / "adapter_model.safetensors", framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            m = LORA_NAME_RE.match(key)
            if m is None:
                raise RuntimeError(f"unexpected adapter tensor name: {key}")
            layer, module, ab = m.group(1), m.group(2), m.group(3)
            pairs.setdefault((int(layer), module), {})[ab] = f.get_tensor(key)
    for k, v in pairs.items():
        if set(v) != {"A", "B"}:
            raise RuntimeError(f"incomplete lora pair for {k}: has {sorted(v)}")
    return pairs


def load_base_index() -> dict[str, str]:
    with open(BASE_DIR / "model.safetensors.index.json") as f:
        index = json.load(f)
    return index["weight_map"]


def load_base_tensor(name: str, weight_map: dict[str, str]) -> torch.Tensor:
    shard = weight_map[name]
    with safe_open(BASE_DIR / shard, framework="pt") as f:
        return f.get_tensor(name)


def main() -> None:
    scale, target_modules = load_adapter_config()
    pairs = load_lora_pairs()

    expected_layers = 16
    expected_pairs = expected_layers * len(target_modules)
    if len(pairs) != 32 or len(pairs) != expected_pairs:
        raise RuntimeError(
            f"expected exactly 32 adapter pairs, found {len(pairs)} "
            f"(target_modules={target_modules})"
        )

    weight_map = load_base_index()
    base_names = list(weight_map.keys())
    if len(base_names) != 114:
        raise RuntimeError(f"expected 114 base tensors, found {len(base_names)}")

    merged: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in base_names:
        tensor = load_base_tensor(name, weight_map)
        m = re.match(r"^model\.layers\.(\d+)\.(self_attn\.(?:q|v)_proj)\.weight$", name)
        if m is not None:
            layer, module = int(m.group(1)), m.group(2)
            key = (layer, module)
            short_module = module.rsplit(".", 1)[-1]
            if key in pairs and short_module in target_modules:
                A = pairs[key]["A"].to(torch.float32)
                B = pairs[key]["B"].to(torch.float32)
                if tensor.dtype != torch.float32:
                    raise RuntimeError(f"expected float32 base tensor for {name}")
                delta = scale * (B @ A)
                if delta.shape != tensor.shape:
                    raise RuntimeError(
                        f"shape mismatch merging {name}: base {tuple(tensor.shape)} "
                        f"vs delta {tuple(delta.shape)}"
                    )
                tensor = tensor + delta
                merged_count += 1
        merged[name] = tensor

    if merged_count != 32:
        raise RuntimeError(f"expected to merge 32 weights, merged {merged_count}")

    if any("lora_" in name for name in merged):
        raise RuntimeError("adapter tensor leaked into output")
    if len(merged) != 114:
        raise RuntimeError(f"expected 114 output tensors, got {len(merged)}")
    q0 = merged["model.layers.0.self_attn.q_proj.weight"]
    if tuple(q0.shape) != (2048, 2048):
        raise RuntimeError(f"model.layers.0.self_attn.q_proj.weight has shape {tuple(q0.shape)}")

    write_sharded(merged)
    print(f"OK: merged {merged_count} weights, wrote {len(merged)} tensors to {OUT_DIR}")


def write_sharded(tensors: dict[str, torch.Tensor]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIR.glob("*.safetensors"):
        p.unlink()

    def nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name, t in tensors.items():
        size = nbytes(t)
        if size > SHARD_LIMIT_BYTES:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
            continue
        if current and current_bytes + size > SHARD_LIMIT_BYTES:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += size
    if current:
        shards.append(current)

    n = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for idx, names in enumerate(shards, start=1):
        shard_file = f"model-{idx:05d}-of-{n:05d}.safetensors"
        shard_tensors = {name: tensors[name].contiguous() for name in names}
        save_file(shard_tensors, OUT_DIR / shard_file, metadata={"format": "pt"})
        for name in names:
            weight_map[name] = shard_file
            total_size += nbytes(tensors[name])

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - fail loudly per task rules
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
