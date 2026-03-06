from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, Literal, TypeVar

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

from .plan import OutputSpec
from .transform import StateDictLike

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    T = TypeVar("T")

    class _TqdmDummy:
        def __init__(self, iterable=None, total=None, **_):
            self.iterable = iterable

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def update(self, *_):
            pass

        def close(self):
            pass

    def tqdm(iterable=None, **kwargs):  # type: ignore
        return _TqdmDummy(iterable, **kwargs)


logger = logging.getLogger("brainsurgery")


def resolve_output_destination(
    output: OutputSpec,
    *,
    default_shard_size: str,
) -> tuple[Path, Literal["safetensors", "torch"], int | None]:
    path = output.path
    format_value = output.format
    shard_size = resolve_shard_size(output, default_shard_size=default_shard_size)

    if format_value is not None:
        if format_value == "safetensors":
            resolved_path, resolved_format = resolve_output_destination_for_explicit_safetensors(path)
        elif format_value == "torch":
            resolved_path, resolved_format = resolve_output_destination_for_explicit_torch(path)
        else:  # pragma: no cover
            raise RuntimeError(f"unsupported explicit output format: {format_value}")
    else:
        suffix = path.suffix.lower()
        if suffix == ".safetensors":
            resolved_path, resolved_format = path, "safetensors"
        elif suffix in {".pt", ".pth", ".bin"}:
            resolved_path, resolved_format = path, "torch"
        elif path.exists() and path.is_dir():
            resolved_path, resolved_format = path / "model.safetensors", "safetensors"
        elif suffix == "":
            resolved_path, resolved_format = path / "model.safetensors", "safetensors"
        else:
            raise RuntimeError(
                f"unsupported output format for {path}; use a directory, .safetensors, .pt, .pth, or .bin, "
                f"or specify output.format explicitly"
            )

    if shard_size is not None and resolved_format != "safetensors":
        raise RuntimeError("output.shard is only supported for safetensors output")

    return resolved_path, resolved_format, shard_size


def resolve_output_destination_for_explicit_safetensors(path: Path) -> tuple[Path, Literal["safetensors"]]:
    if path.exists() and path.is_dir():
        return path / "model.safetensors", "safetensors"
    if path.suffix == "":
        return path / "model.safetensors", "safetensors"
    if path.suffix.lower() != ".safetensors":
        raise RuntimeError(
            f"output.format='safetensors' is incompatible with file path {path}; "
            f"use a directory or a .safetensors file"
        )
    return path, "safetensors"


def resolve_output_destination_for_explicit_torch(path: Path) -> tuple[Path, Literal["torch"]]:
    if path.exists() and path.is_dir():
        raise RuntimeError("output.format='torch' requires a file path, not a directory")
    if path.suffix.lower() not in {".pt", ".pth", ".bin"}:
        raise RuntimeError(
            f"output.format='torch' requires a .pt, .pth, or .bin file path; got {path}"
        )
    return path, "torch"


def resolve_shard_size(output: OutputSpec, default_shard_size: str) -> int | None:
    raw = output.shard

    if raw is None:
        if is_directory_style_output(output):
            raw = default_shard_size
        else:
            return None

    return parse_shard_size(raw)


def is_directory_style_output(output: OutputSpec) -> bool:
    path = output.path

    if output.format == "torch":
        return False

    if output.format == "safetensors":
        if path.exists() and path.is_dir():
            return True
        return path.suffix == ""

    if path.exists() and path.is_dir():
        return True

    return path.suffix == ""


def parse_shard_size(raw: str | None) -> int | None:
    if raw is None or raw == "none":
        return None

    if not isinstance(raw, str) or not raw:
        raise RuntimeError("output.shard must be a non-empty string or 'none'")

    match = re.fullmatch(r"(?i)\s*(\d+)\s*(b|kb|mb|gb|tb)\s*", raw)
    if not match:
        raise RuntimeError(
            f"invalid output.shard value {raw!r}; expected values like 'none', '500MB', '5GB'"
        )

    value = int(match.group(1))
    unit = match.group(2).lower()

    multipliers = {
        "b": 1,
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
        "tb": 1024**4,
    }

    return value * multipliers[unit]


def resolve_sharded_output_directory(original_path: Path, resolved_path: Path) -> Path:
    if original_path.exists() and original_path.is_dir():
        return original_path
    if original_path.suffix == "":
        return original_path
    raise RuntimeError(
        "sharded safetensors output requires a directory-style output path, not a single file"
    )


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def shard_state_dict(
    state_dict: Dict[str, torch.Tensor],
    max_shard_size: int,
) -> list[dict[str, torch.Tensor]]:
    if max_shard_size <= 0:
        raise RuntimeError("max_shard_size must be positive")

    shards: list[dict[str, torch.Tensor]] = []
    current_shard: dict[str, torch.Tensor] = {}
    current_size = 0

    for key, tensor in state_dict.items():
        size = tensor_nbytes(tensor)

        if size > max_shard_size:
            if current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
            shards.append({key: tensor})
            continue

        if current_shard and current_size + size > max_shard_size:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += size

    if current_shard:
        shards.append(current_shard)

    return shards


def save_sharded_safetensors(
    state_dict: Dict[str, torch.Tensor],
    output_dir: Path,
    max_shard_size: int,
    *,
    max_io_workers: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    shards = shard_state_dict(state_dict, max_shard_size)
    total_size = sum(tensor_nbytes(tensor) for tensor in state_dict.values())
    total_shards = len(shards)

    logger.info(
        "Segmenting brain into %d safetensor shard(s) with max shard size %d bytes",
        total_shards,
        max_shard_size,
    )

    shard_infos: list[tuple[int, str, Path, dict[str, torch.Tensor]]] = []
    weight_map: dict[str, str] = {}

    for shard_index, shard in enumerate(shards, start=1):
        shard_name = f"model-{shard_index:05d}-of-{total_shards:05d}.safetensors"
        shard_path = output_dir / shard_name
        shard_infos.append((shard_index, shard_name, shard_path, shard))
        for key in shard:
            weight_map[key] = shard_name

    num_workers = choose_num_io_workers(total_shards, max_io_workers=max_io_workers)
    logger.info("Dispatching %d worker thread(s) for shard save", num_workers)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(save_safetensors_shard, shard_path, shard): (shard_index, shard_name)
            for shard_index, shard_name, shard_path, shard in shard_infos
        }

        progress = tqdm(total=total_shards, desc="Shard save", unit="shard", leave=False)
        try:
            for future in as_completed(futures):
                shard_index, shard_name = futures[future]
                future.result()
                logger.debug("Saved shard %d/%d to %s", shard_index, total_shards, shard_name)
                progress.update(1)
        finally:
            progress.close()

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path = output_dir / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index_path


def save_safetensors_shard(path: Path, shard: dict[str, torch.Tensor]) -> None:
    save_safetensors_file(shard, str(path))


def load_state_dict_from_path(path: Path, global_state_dict: StateDictLike, *, max_io_workers: int) -> None:
    if not path.exists():
        raise RuntimeError(f"checkpoint path does not exist: {path}")
    if path.is_dir():
        logger.info("CT scan shows a model directory at %s", path)
        return load_state_dict_from_directory(path, global_state_dict, max_io_workers=max_io_workers)
    logger.info("CT scan shows a single checkpoint file at %s", path)
    return load_state_dict_from_file(path, global_state_dict)

def load_state_dict_from_directory(path: Path, global_state_dict: StateDictLike, *, max_io_workers: int) -> Dict[str, torch.Tensor]:
    pt_files = sorted(path.glob("*.pt")) + sorted(path.glob("*.pth")) + sorted(path.glob("*.bin"))
    safetensor_files = sorted(path.glob("*.safetensors"))
    index_file = path / "model.safetensors.index.json"

    if pt_files and safetensor_files:
        raise RuntimeError(
            f"model directory contains both torch and safetensors files; refusing ambiguous load: {path}"
        )

    if safetensor_files:
        if index_file.exists():
            logger.info("Detected safetensors index at %s", index_file)
            files = resolve_safetensor_shards_from_index(index_file, path)
        else:
            logger.info("No safetensors index found; loading all safetensors shards")
            files = safetensor_files
    else:
        files = pt_files

    if not files:
        raise RuntimeError(f"no supported checkpoint files found in model directory: {path}")

    logger.info("Found %d checkpoint shard(s) in %s", len(files), path)

    num_workers = choose_num_io_workers(len(files), max_io_workers=max_io_workers)
    logger.info("Dispatching %d worker thread(s) for shard load", num_workers)

    merge_lock = Lock()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(load_state_dict_from_file, file_path, global_state_dict, merge_lock): file_path
            for file_path in files
        }

        progress = tqdm(total=len(files), desc=f"Open {path.name}", unit="file", leave=False)
        try:
            for future in as_completed(futures):
                future.result()  # propagate exceptions
                progress.update(1)
        finally:
            progress.close()

    logger.info("Cranial assembly complete for %s: %d tensor(s)", path, len(global_state_dict))


def choose_num_io_workers(num_items: int, max_io_workers: int) -> int:
    return max(1, min(max_io_workers, num_items))


def resolve_safetensor_shards_from_index(index_file: Path, base_dir: Path) -> list[Path]:
    try:
        index_data = json.loads(index_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to parse safetensors index: {index_file}") from exc

    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"invalid safetensors index: missing weight_map in {index_file}")

    shard_names = sorted(set(weight_map.values()))
    shard_paths: list[Path] = []

    for name in shard_names:
        shard_path = base_dir / name
        if not shard_path.exists():
            raise RuntimeError(
                f"safetensors index references missing shard {name!r} in {base_dir}"
            )
        shard_paths.append(shard_path)

    return shard_paths


def load_state_dict_from_file(path: Path, global_state_dict: StateDictLike, merge_lock: Lock) -> None:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        logger.info("Using safetensors instruments on %s", path)
        loaded = load_safetensors_file(str(path), device="cpu")
    else:
        logger.info("Using torch instruments on %s", path)
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            logger.info("Detected wrapped state_dict payload in %s", path)
            loaded = loaded["state_dict"]
    loaded = validate_state_dict_mapping(loaded, path)
    for key, tensor in loaded.items():
        with merge_lock:
            if key in global_state_dict:
                raise RuntimeError(f"duplicate tensor key {key!r} while loading file {path}")
            global_state_dict[key] = tensor


def validate_state_dict_mapping(loaded: object, path: Path) -> Dict[str, torch.Tensor]:
    if not isinstance(loaded, dict):
        raise RuntimeError(f"checkpoint at {path} is not a state_dict mapping")
    if not all(isinstance(k, str) and torch.is_tensor(v) for k, v in loaded.items()):
        raise RuntimeError(f"checkpoint at {path} is not a plain tensor state_dict")
    return dict(loaded)
