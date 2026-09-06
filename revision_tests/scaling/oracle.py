#!/usr/bin/env python3
"""Independent safetensors oracle and manifests for the scaling protocol."""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

PROTOCOL_ID = "eacl2027_scaling_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def discover_checkpoint(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"not a safetensors file: {path}")
        root = path.parent
        files = [path]
        index_path = None
        declared_map = None
        layout = "single"
    elif path.is_dir():
        root = path
        index_path = path / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            declared_map = index.get("weight_map")
            if not isinstance(declared_map, dict) or not declared_map:
                raise ValueError(f"invalid weight_map: {index_path}")
            if any(not isinstance(key, str) or not isinstance(value, str) for key, value in declared_map.items()):
                raise ValueError(f"non-string weight_map entry: {index_path}")
            files = [root / name for name in sorted(set(declared_map.values()))]
            stored_files = sorted(path.glob("*.safetensors"))
            if {item.name for item in stored_files} != {item.name for item in files}:
                raise ValueError("index does not cover exactly the stored safetensors files")
            layout = "sharded"
        else:
            preferred = path / "model.safetensors"
            files = [preferred] if preferred.is_file() else sorted(path.glob("*.safetensors"))
            declared_map = None
            layout = "single" if len(files) == 1 else "unindexed_multiple"
    else:
        raise ValueError(f"checkpoint does not exist: {path}")
    if not files or any(not item.is_file() for item in files):
        raise ValueError(f"checkpoint data files are missing under {path}")

    tensor_files: dict[str, Path] = {}
    tensor_metadata: dict[str, dict[str, Any]] = {}
    for file_path in files:
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in tensor_files:
                    raise ValueError(f"duplicate tensor across shards: {name}")
                tensor = handle.get_tensor(name)
                tensor_files[name] = file_path
                tensor_metadata[name] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "is_floating_point": tensor.is_floating_point(),
                    "numel": tensor.numel(),
                    "nbytes": tensor.numel() * tensor.element_size(),
                }
    if declared_map is not None:
        if set(declared_map) != set(tensor_files):
            raise ValueError("index keys do not equal stored tensor keys")
        wrong = [name for name, file_path in tensor_files.items() if declared_map[name] != file_path.name]
        if wrong:
            raise ValueError(f"index maps tensors to the wrong shard: {wrong[:3]}")
    return {
        "path": path,
        "root": root,
        "layout": layout,
        "data_files": files,
        "index_file": index_path if index_path and index_path.is_file() else None,
        "tensor_files": tensor_files,
        "tensor_metadata": tensor_metadata,
    }


def checkpoint_manifest(path: Path, *, tensor_hashes: bool) -> dict[str, Any]:
    layout = discover_checkpoint(path)
    files = list(layout["data_files"])
    if layout["index_file"] is not None:
        files.append(layout["index_file"])
    dtype_tensor_counts: dict[str, int] = {}
    dtype_logical_bytes: dict[str, int] = {}
    for item in layout["tensor_metadata"].values():
        dtype = item["dtype"]
        dtype_tensor_counts[dtype] = dtype_tensor_counts.get(dtype, 0) + 1
        dtype_logical_bytes[dtype] = dtype_logical_bytes.get(dtype, 0) + item["nbytes"]
    manifest: dict[str, Any] = {
        "layout": layout["layout"],
        "tensor_count": len(layout["tensor_files"]),
        "stored_tensor_element_count": sum(
            item["numel"] for item in layout["tensor_metadata"].values()
        ),
        "logical_tensor_bytes": sum(item["nbytes"] for item in layout["tensor_metadata"].values()),
        "checkpoint_file_bytes": sum(item.stat().st_size for item in files),
        "data_file_count": len(layout["data_files"]),
        "dtype_tensor_counts": dict(sorted(dtype_tensor_counts.items())),
        "dtype_logical_bytes": dict(sorted(dtype_logical_bytes.items())),
        "files": {
            item.name: {"bytes": item.stat().st_size, "sha256": sha256_file(item)}
            for item in files
        },
        "tensors": dict(sorted(layout["tensor_metadata"].items())),
    }
    if tensor_hashes:
        hashes: dict[str, str] = {}
        for file_path in layout["data_files"]:
            with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    hashes[name] = tensor_sha256(handle.get_tensor(name))
        for name, digest in hashes.items():
            manifest["tensors"][name]["sha256"] = digest
    return manifest


def validate_input_operation(path: Path, target_regex: str) -> dict[str, Any]:
    layout = discover_checkpoint(path)
    pattern = re.compile(target_regex)
    matched = sorted(name for name in layout["tensor_files"] if pattern.fullmatch(name))
    nonfloating = [
        name for name in matched if not layout["tensor_metadata"][name]["is_floating_point"]
    ]
    matched_dtype_counts: dict[str, int] = {}
    for name in matched:
        dtype = layout["tensor_metadata"][name]["dtype"]
        matched_dtype_counts[dtype] = matched_dtype_counts.get(dtype, 0) + 1
    return {
        "target_regex": target_regex,
        "matched_tensor_count": len(matched),
        "matched_tensors": matched,
        "matched_nonfloating_tensors": nonfloating,
        "matched_dtype_counts": dict(sorted(matched_dtype_counts.items())),
        "passed": bool(matched) and not nonfloating,
    }


def verify_huggingface_revision(path: Path, expected_revision: str) -> dict[str, Any]:
    """Verify huggingface_hub local-dir metadata for every checkpoint file."""
    layout = discover_checkpoint(path)
    files = list(layout["data_files"])
    if layout["index_file"] is not None:
        files.append(layout["index_file"])
    records = []
    for file_path in files:
        relative = file_path.relative_to(layout["root"])
        metadata_path = layout["root"] / ".cache" / "huggingface" / "download" / Path(
            relative.as_posix() + ".metadata"
        )
        lines = (
            metadata_path.read_text(encoding="utf-8").splitlines()
            if metadata_path.is_file()
            else []
        )
        observed = lines[0] if lines else None
        records.append(
            {
                "file": relative.as_posix(),
                "metadata_found": metadata_path.is_file(),
                "observed_revision": observed,
                "matches_expected": observed == expected_revision,
            }
        )
    return {
        "expected_revision": expected_revision,
        "files": records,
        "passed": bool(records) and all(record["matches_expected"] for record in records),
    }


def validate_shards(layout: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    shard_tensor_bytes: dict[str, int] = {}
    shard_tensor_counts: dict[str, int] = {}
    oversized_singletons: list[str] = []
    violations: list[str] = []
    for name, file_path in layout["tensor_files"].items():
        shard_tensor_bytes.setdefault(file_path.name, 0)
        shard_tensor_counts.setdefault(file_path.name, 0)
        shard_tensor_bytes[file_path.name] += layout["tensor_metadata"][name]["nbytes"]
        shard_tensor_counts[file_path.name] += 1
    for name, size in shard_tensor_bytes.items():
        if size > max_bytes:
            if shard_tensor_counts[name] == 1:
                oversized_singletons.append(name)
            else:
                violations.append(name)
    return {
        "max_shard_tensor_bytes": max(shard_tensor_bytes.values(), default=0),
        "shard_tensor_bytes": dict(sorted(shard_tensor_bytes.items())),
        "oversized_singleton_shards": sorted(oversized_singletons),
        "violating_shards": sorted(violations),
        "passed": not violations,
    }


def compare_output(
    source_path: Path,
    output_path: Path,
    *,
    target_regex: str,
    factor: float,
    shard_size_bytes: int,
) -> dict[str, Any]:
    source = discover_checkpoint(source_path)
    output = discover_checkpoint(output_path)
    pattern = re.compile(target_regex)
    source_names = set(source["tensor_files"])
    output_names = set(output["tensor_files"])
    missing = sorted(source_names - output_names)
    unexpected = sorted(output_names - source_names)
    checks: list[dict[str, Any]] = []
    matched = 0
    with ExitStack() as stack:
        source_handles = {
            path: stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            for path in source["data_files"]
        }
        output_handles = {
            path: stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            for path in output["data_files"]
        }
        for name in sorted(source_names & output_names):
            original = source_handles[source["tensor_files"][name]].get_tensor(name)
            actual = output_handles[output["tensor_files"][name]].get_tensor(name)
            selected = pattern.fullmatch(name) is not None
            if selected:
                matched += 1
            selectable = not selected or original.is_floating_point()
            expected = original * factor if selected and selectable else original
            shape_equal = tuple(actual.shape) == tuple(expected.shape)
            dtype_equal = actual.dtype == expected.dtype
            expected_hash = tensor_sha256(expected)
            actual_hash = tensor_sha256(actual)
            exact = shape_equal and dtype_equal and expected_hash == actual_hash
            checks.append(
                {
                    "name": name,
                    "selected": selected,
                    "selectable": selectable,
                    "shape_equal": shape_equal,
                    "dtype_equal": dtype_equal,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                    "byte_exact": exact,
                    "passed": selectable and exact,
                }
            )
    sharding = validate_shards(output, shard_size_bytes)
    output_manifest = checkpoint_manifest(output_path, tensor_hashes=False)
    passed_checks = sum(item["passed"] for item in checks)
    passed = (
        not missing
        and not unexpected
        and matched > 0
        and passed_checks == len(source_names) == len(output_names)
        and sharding["passed"]
        and output["layout"] == "sharded"
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "comparison": "byte_exact_independent_arithmetic_oracle",
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
        "matched_tensor_count": matched,
        "tensors_checked": len(checks),
        "tensors_passed": passed_checks,
        "sharding": sharding,
        "output_manifest": output_manifest,
        "tensor_checks": checks,
        "passed": passed,
    }


def files_unchanged(path: Path, before: dict[str, str]) -> dict[str, Any]:
    layout = discover_checkpoint(path)
    files = list(layout["data_files"])
    if layout["index_file"] is not None:
        files.append(layout["index_file"])
    after = {item.name: sha256_file(item) for item in files}
    return {
        "before": before,
        "after": after,
        "passed": after == before,
    }
