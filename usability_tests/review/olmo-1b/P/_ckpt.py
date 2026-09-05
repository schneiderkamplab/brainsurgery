"""Shared helpers for the Python baselines: load a file or sharded directory, write shards."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def load_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.is_file():
        return load_file(str(path))
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        sd: dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            sd.update(load_file(str(path / shard)))
        return sd
    return load_file(str(path / "model.safetensors"))


def save_sharded_safetensors(sd: dict[str, torch.Tensor], out_dir: Path, max_bytes: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, torch.Tensor]] = []
    cur: dict[str, torch.Tensor] = {}
    cur_size = 0
    for name, tensor in sd.items():
        size = tensor.numel() * tensor.element_size()
        if cur and cur_size + size > max_bytes:
            shards.append(cur)
            cur, cur_size = {}, 0
        cur[name] = tensor.contiguous()
        cur_size += size
    if cur:
        shards.append(cur)
    weight_map: dict[str, str] = {}
    for idx, shard in enumerate(shards, start=1):
        shard_name = f"model-{idx:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(out_dir / shard_name))
        for name in shard:
            weight_map[name] = shard_name
    total = sum(t.numel() * t.element_size() for t in sd.values())
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2),
        encoding="utf-8",
    )
