"""T5: LoRA adapter merge with sharded export (GPT-2, condition F).

Plain script on top of `safetensors` + `torch` (both in F-allowed.md). Does
not use peft's `merge_and_unload` because that requires instantiating the
HF model; merging directly on the checkpoint tensors is simpler and avoids
extra dependencies for something that is just a matmul + add per pair.

Merge rule (from TASK.md, fan_in_fan_out = true):
    h.<i>.<module>.weight += scale * (B @ A).T
    scale = lora_alpha / r
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # out/T5/solution.py -> out/ -> sandbox root
BASE_PATH = ROOT / "inputs" / "base" / "model.safetensors"
LORA_PATH = ROOT / "inputs" / "lora" / "adapter_model.safetensors"
LORA_CONFIG_PATH = ROOT / "inputs" / "lora" / "adapter_config.json"
OUT_DIR = ROOT / "out" / "T5"

MAX_SHARD_BYTES = 100 * 1024 * 1024  # 104,857,600

LORA_A_RE = re.compile(
    r"^base_model\.model\.h\.(\d+)\.(attn\.c_attn)\.lora_A\.weight$"
)


def main() -> None:
    config = json.loads(LORA_CONFIG_PATH.read_text())
    r = config["r"]
    alpha = config["lora_alpha"]
    fan_in_fan_out = config["fan_in_fan_out"]
    if not fan_in_fan_out:
        raise SystemExit("expected fan_in_fan_out=true for Conv1D base layers")
    scale = alpha / r

    base = load_file(str(BASE_PATH))
    lora = load_file(str(LORA_PATH))

    # Find every lora_A / lora_B pair and the base tensor it targets.
    pairs = {}
    for name in lora:
        m = LORA_A_RE.match(name)
        if not m:
            continue
        layer, module = m.group(1), m.group(2)
        b_name = name.replace("lora_A", "lora_B")
        if b_name not in lora:
            raise SystemExit(f"missing paired lora_B for {name}")
        base_name = f"h.{layer}.{module}.weight"
        if base_name not in base:
            raise SystemExit(f"base tensor {base_name} not found")
        pairs[base_name] = (name, b_name)

    if len(pairs) != 12:
        raise SystemExit(f"expected exactly 12 adapter pairs, found {len(pairs)}")

    merged = dict(base)
    for base_name, (a_name, b_name) in pairs.items():
        a = lora[a_name].to(torch.float32)
        b = lora[b_name].to(torch.float32)
        w = base[base_name].to(torch.float32)
        if w.shape != (768, 2304):
            raise SystemExit(f"{base_name} has unexpected shape {tuple(w.shape)}")
        delta = scale * (b @ a).T
        if delta.shape != w.shape:
            raise SystemExit(
                f"shape mismatch merging {base_name}: delta {tuple(delta.shape)} "
                f"vs base {tuple(w.shape)}"
            )
        merged[base_name] = (w + delta).contiguous()

    # Required checks before writing.
    if any("lora_" in name for name in merged):
        raise SystemExit("adapter tensor leaked into merged state dict")
    if merged["h.0.attn.c_attn.weight"].shape != (768, 2304):
        raise SystemExit("h.0.attn.c_attn.weight shape changed unexpectedly")
    if len(merged) != 160:
        raise SystemExit(f"expected exactly 160 tensors, got {len(merged)}")
    if merged.keys() != base.keys():
        raise SystemExit("merged tensor name set differs from base tensor name set")

    write_sharded(merged, OUT_DIR)
    print(f"wrote {len(merged)} tensors to {OUT_DIR}")


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def write_sharded(tensors: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bin-pack tensors into shards, each <= MAX_SHARD_BYTES of tensor data,
    # except a single oversized tensor gets its own shard.
    names = list(tensors.keys())
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        nbytes = tensor_nbytes(tensors[name])
        if nbytes > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
            continue
        if current and current_bytes + nbytes > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += nbytes
    if current:
        shards.append(current)

    n = len(shards)
    weight_map = {}
    total_size = 0
    digits = max(5, len(str(n)))
    for idx, shard_names in enumerate(shards, start=1):
        shard_filename = f"model-{idx:0{digits}d}-of-{n:0{digits}d}.safetensors"
        shard_tensors = {name: tensors[name] for name in shard_names}
        save_file(shard_tensors, str(out_dir / shard_filename))
        for name in shard_names:
            weight_map[name] = shard_filename
            total_size += tensor_nbytes(tensors[name])

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
