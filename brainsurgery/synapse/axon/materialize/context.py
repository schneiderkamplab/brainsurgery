from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import safetensors
import torch


@dataclass(frozen=True)
class MaterializeContext:
    config: dict[str, object]
    state_keys: frozenset[str]
    checkpoint: str | None = None
    model_dir: Path | None = None


def load_json_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(payload).__name__}")
    return payload


def checkpoint_pragma_entries(pragmas: dict[str, object]) -> list[str]:
    raw = pragmas.get("checkpoints")
    if isinstance(raw, tuple):
        return [str(item) for item in raw]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str):
        return [raw]
    return []


def normalize_checkpoint_name(repo_id: str) -> str:
    name = repo_id.split("/")[-1]
    if name.endswith("-pt"):
        return name[: -len("-pt")]
    if name.endswith("-it"):
        return name[: -len("-it")]
    return name


def group_output_name(checkpoints: list[str]) -> str:
    names = [checkpoint.split("/")[-1] for checkpoint in checkpoints]
    unique_names = sorted(set(names), key=lambda name: (len(name), name))
    for candidate in unique_names:
        if all(name == candidate or name.startswith(candidate + "-") for name in names):
            return candidate
    normalized_names = {normalize_checkpoint_name(checkpoint) for checkpoint in checkpoints}
    if len(normalized_names) == 1:
        return next(iter(normalized_names))
    return unique_names[0]


def _index_weight_keys(model_dir: Path) -> set[str]:
    out: set[str] = set()
    for name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "model.bin.index.json",
    ):
        path = model_dir / name
        if not path.exists():
            continue
        payload = load_json_config(path)
        weight_map = payload.get("weight_map")
        if isinstance(weight_map, dict):
            out.update(str(key) for key in weight_map.keys())
    return out


def _safetensors_weight_keys(model_dir: Path) -> set[str]:
    out: set[str] = set()
    errors: list[tuple[Path, Exception]] = []
    for path in sorted(model_dir.glob("*.safetensors")):
        try:
            st = safetensors.safe_open(str(path), framework="pt")
        except Exception as exc:  # pragma: no cover - passthrough
            errors.append((path, exc))
            continue
        out.update(str(key) for key in st.keys())
    if not out and errors:
        first_path, first_exc = errors[0]
        raise type(first_exc)(f"{first_path}: {first_exc}")
    return out


def _torch_weight_keys(model_dir: Path) -> set[str]:
    candidates = sorted(model_dir.glob("*.bin")) + sorted(model_dir.glob("*.pt"))
    for path in candidates:
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict):
            state = loaded.get("state_dict") if "state_dict" in loaded else loaded
            if isinstance(state, dict):
                return {str(key) for key in state.keys()}
    return set()


def checkpoint_state_keys(model_dir: Path) -> set[str]:
    indexed = _index_weight_keys(model_dir)
    if indexed:
        return indexed
    safetensor_keys = _safetensors_weight_keys(model_dir)
    if safetensor_keys:
        return safetensor_keys
    torch_keys = _torch_weight_keys(model_dir)
    if torch_keys:
        return torch_keys
    raise FileNotFoundError(f"No checkpoint weights found in {model_dir}")


def load_materialize_context(*, checkpoint: str, models_root: Path) -> MaterializeContext:
    model_dir = models_root.resolve() / checkpoint
    return MaterializeContext(
        config=load_json_config(model_dir / "config.json"),
        state_keys=frozenset(checkpoint_state_keys(model_dir)),
        checkpoint=checkpoint,
        model_dir=model_dir,
    )


__all__ = [
    "MaterializeContext",
    "checkpoint_pragma_entries",
    "checkpoint_state_keys",
    "group_output_name",
    "load_json_config",
    "load_materialize_context",
    "normalize_checkpoint_name",
]
