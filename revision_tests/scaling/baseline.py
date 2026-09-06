#!/usr/bin/env python3
"""Independent direct PyTorch/safetensors baseline for the scaling protocol."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-regex", required=True)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--shard-size-bytes", type=int, required=True)
    return parser.parse_args()


def checkpoint_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"not a safetensors checkpoint: {path}")
        return [path]
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid index: {index_path}")
        files = [path / name for name in sorted(set(weight_map.values()))]
    else:
        preferred = path / "model.safetensors"
        files = [preferred] if preferred.is_file() else sorted(path.glob("*.safetensors"))
    if not files or any(not item.is_file() for item in files):
        raise ValueError(f"cannot resolve safetensors checkpoint: {path}")
    return files


def load_state(path: Path) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for file_path in checkpoint_files(path):
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in state:
                    raise ValueError(f"duplicate tensor key: {name}")
                state[name] = handle.get_tensor(name)
    return dict(sorted(state.items()))


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def shard_state(
    state: dict[str, torch.Tensor], max_bytes: int
) -> list[dict[str, torch.Tensor]]:
    if max_bytes <= 0:
        raise ValueError("shard size must be positive")
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for name, tensor in state.items():
        size = tensor_nbytes(tensor)
        if size > max_bytes:
            if current:
                shards.append(current)
                current = {}
                current_bytes = 0
            shards.append({name: tensor})
        else:
            if current and current_bytes + size > max_bytes:
                shards.append(current)
                current = {}
                current_bytes = 0
            current[name] = tensor
            current_bytes += size
    if current:
        shards.append(current)
    return shards


def save_sharded(
    state: dict[str, torch.Tensor], output: Path, max_bytes: int
) -> dict[str, int]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    shards = shard_state(state, max_bytes)
    weight_map: dict[str, str] = {}
    for index, shard in enumerate(shards, start=1):
        name = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(output / name))
        weight_map.update({key: name for key in shard})
    index_doc = {
        "metadata": {"total_size": sum(tensor_nbytes(tensor) for tensor in state.values())},
        "weight_map": weight_map,
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"tensors": len(state), "shards": len(shards)}


def main() -> int:
    args = parse_args()
    pattern = re.compile(args.target_regex)
    state = load_state(args.input.resolve())
    selected = []
    for name, tensor in state.items():
        if pattern.fullmatch(name):
            if not tensor.is_floating_point():
                raise ValueError(f"matched non-floating tensor: {name} ({tensor.dtype})")
            state[name] = tensor * args.factor
            selected.append(name)
    if not selected:
        raise ValueError("operation matched no tensors")
    result = save_sharded(state, args.output.resolve(), args.shard_size_bytes)
    print(json.dumps(result | {"selected_tensors": len(selected)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
