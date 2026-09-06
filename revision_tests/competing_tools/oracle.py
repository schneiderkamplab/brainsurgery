"""Independent fixtures and output oracle for competing-tool cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

PROTOCOL_ID = "eacl2027_competing_tools_v1"
ARITHMETIC_ATOL = 1e-6
ARITHMETIC_RTOL = 1e-6


def tiny_gpt2_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "transformer.wte.weight": (8, 4),
        "transformer.wpe.weight": (8, 4),
        "transformer.ln_f.weight": (4,),
        "transformer.ln_f.bias": (4,),
        "lm_head.weight": (8, 4),
    }
    per_layer = {
        "attn.c_attn.weight": (4, 12),
        "attn.c_attn.bias": (12,),
        "attn.c_proj.weight": (4, 4),
        "attn.c_proj.bias": (4,),
        "ln_1.weight": (4,),
        "ln_1.bias": (4,),
        "ln_2.weight": (4,),
        "ln_2.bias": (4,),
        "mlp.c_fc.weight": (4, 16),
        "mlp.c_fc.bias": (16,),
        "mlp.c_proj.weight": (16, 4),
        "mlp.c_proj.bias": (4,),
    }
    for layer in range(2):
        for suffix, shape in per_layer.items():
            shapes[f"transformer.h.{layer}.{suffix}"] = shape
    return shapes


def model_state(variant: str) -> dict[str, torch.Tensor]:
    offsets = {
        "base": 0.0,
        "model_a": 0.125,
        "model_b": -0.25,
        "finetune_1": 0.5,
        "finetune_2": -0.375,
    }
    if variant not in offsets:
        raise ValueError(f"unknown fixture model variant: {variant}")
    result = {}
    cursor = 0
    for name, shape in sorted(tiny_gpt2_shapes().items()):
        count = 1
        for size in shape:
            count *= size
        values = torch.arange(cursor, cursor + count, dtype=torch.float32)
        values = (values / 64.0 + offsets[variant]).reshape(shape)
        result[name] = values
        cursor += count
    return result


def rename_state() -> dict[str, torch.Tensor]:
    return {
        "layer.0.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "layer.0.bias": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        "layer.1.weight": torch.arange(8, dtype=torch.float16).reshape(2, 4),
        "layer.1.bias": torch.tensor([4, 5], dtype=torch.int64),
        "layer.2.weight": torch.tensor([[True, False], [False, True]], dtype=torch.bool),
        "layer.2.bias": torch.tensor([6.0, -6.0], dtype=torch.float64),
        "embedding.weight": torch.arange(20, dtype=torch.float32).reshape(5, 4),
        "metadata.version": torch.tensor(1, dtype=torch.int64),
    }


def expected_state(case_id: str, fixture_root: Path) -> dict[str, torch.Tensor]:
    if case_id == "R01":
        source = load_file(str(fixture_root / "rename" / "model.safetensors"), device="cpu")
        return {
            (name.replace("layer.", "block.", 1) if name.startswith("layer.") else name): tensor
            for name, tensor in source.items()
        }
    if case_id == "M01":
        a = load_model(fixture_root / "models" / "model_a")
        b = load_model(fixture_root / "models" / "model_b")
        return {name: 0.25 * a[name] + 0.75 * b[name] for name in a}
    if case_id == "M02":
        base = load_model(fixture_root / "models" / "base")
        first = load_model(fixture_root / "models" / "finetune_1")
        second = load_model(fixture_root / "models" / "finetune_2")
        return {
            name: base[name] + 0.5 * (first[name] - base[name]) + 0.25 * (second[name] - base[name])
            for name in base
        }
    raise ValueError(f"unknown case: {case_id}")


def load_model(path: Path) -> dict[str, torch.Tensor]:
    if path.is_file():
        return load_file(str(path), device="cpu")
    single = path / "model.safetensors"
    if single.is_file():
        return load_file(str(single), device="cpu")
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        candidates = sorted(path.glob("*.safetensors"))
        if len(candidates) == 1:
            return load_file(str(candidates[0]), device="cpu")
        raise ValueError(f"cannot identify safetensors output under {path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid safetensors index: {index_path}")
    result = {}
    for shard in sorted(set(weight_map.values())):
        shard_state = load_file(str(path / shard), device="cpu")
        overlap = result.keys() & shard_state.keys()
        if overlap:
            raise ValueError(f"duplicate tensors across shards: {sorted(overlap)}")
        result.update(shard_state)
    if set(result) != set(weight_map):
        raise ValueError("loaded shard keys do not match weight_map")
    return result


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def state_manifest(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "nbytes": tensor.numel() * tensor.element_size(),
            "sha256": tensor_sha256(tensor),
        }
        for name, tensor in sorted(state.items())
    }


def compare_output(case_id: str, actual_path: Path, fixture_root: Path) -> dict[str, Any]:
    expected = expected_state(case_id, fixture_root)
    actual = load_model(actual_path)
    expected_names = set(expected)
    actual_names = set(actual)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    tensor_results = []
    for name in sorted(expected_names & actual_names):
        want = expected[name]
        got = actual[name]
        shape_equal = want.shape == got.shape
        dtype_equal = want.dtype == got.dtype
        byte_exact = shape_equal and dtype_equal and tensor_sha256(want) == tensor_sha256(got)
        if shape_equal and dtype_equal and (want.is_floating_point() or want.is_complex()):
            difference = (want.to(torch.float64) - got.to(torch.float64)).abs()
            max_abs = float(difference.max().item()) if difference.numel() else 0.0
            mean_abs = float(difference.mean().item()) if difference.numel() else 0.0
            within_tolerance = bool(
                torch.allclose(got, want, atol=ARITHMETIC_ATOL, rtol=ARITHMETIC_RTOL)
            )
        else:
            max_abs = 0.0 if byte_exact else None
            mean_abs = 0.0 if byte_exact else None
            within_tolerance = byte_exact
        passed = (
            byte_exact if case_id == "R01" else shape_equal and dtype_equal and within_tolerance
        )
        tensor_results.append(
            {
                "name": name,
                "shape_equal": shape_equal,
                "dtype_equal": dtype_equal,
                "byte_exact": byte_exact,
                "within_tolerance": within_tolerance,
                "max_absolute_difference": max_abs,
                "mean_absolute_difference": mean_abs,
                "passed": passed,
            }
        )
    passed_count = sum(item["passed"] for item in tensor_results)
    return {
        "protocol_id": PROTOCOL_ID,
        "case_id": case_id,
        "comparison": "exact" if case_id == "R01" else "tolerance",
        "absolute_tolerance": 0.0 if case_id == "R01" else ARITHMETIC_ATOL,
        "relative_tolerance": 0.0 if case_id == "R01" else ARITHMETIC_RTOL,
        "expected_tensor_count": len(expected),
        "actual_tensor_count": len(actual),
        "tensors_checked": len(tensor_results),
        "tensors_passed": passed_count,
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
        "maximum_absolute_difference": max(
            (
                item["max_absolute_difference"]
                for item in tensor_results
                if item["max_absolute_difference"] is not None
            ),
            default=None,
        ),
        "passed": not missing and not unexpected and passed_count == len(expected) == len(actual),
        "tensor_results": tensor_results,
        "actual_manifest": state_manifest(actual),
    }


def validate_comparison_record(record: dict[str, Any], expected_case_id: str) -> None:
    if record.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("comparison record has incompatible protocol_id")
    if record.get("case_id") != expected_case_id:
        raise ValueError("comparison record has incompatible case_id")
    if not isinstance(record.get("passed"), bool):
        raise ValueError("comparison record lacks a boolean passed decision")
