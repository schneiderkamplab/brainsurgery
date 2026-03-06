from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml

from .transform import CompiledTransform, TransformError, get_transform


class PlanLoaderError(RuntimeError):
    pass


@dataclass(frozen=True)
class OutputSpec:
    path: Path
    format: Optional[Literal["safetensors", "torch"]] = None
    shard: Optional[str] = None


@dataclass(frozen=True)
class SurgeryPlan:
    inputs: Dict[str, Path]
    output: OutputSpec
    transforms: List[CompiledTransform]


def load_plan(path: str | Path) -> SurgeryPlan:
    plan_path = Path(path)

    try:
        raw = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PlanLoaderError(f"failed to read plan file: {plan_path}") from exc
    except yaml.YAMLError as exc:
        raise PlanLoaderError(f"failed to parse yaml from plan file: {plan_path}") from exc

    if not isinstance(raw, dict):
        raise PlanLoaderError("plan must be a YAML mapping")

    inputs = parse_inputs(raw.get("inputs"))
    output = parse_output(raw.get("output"))
    transforms = parse_transforms(raw.get("transforms"), inputs)

    return SurgeryPlan(inputs=inputs, output=output, transforms=transforms)


def parse_inputs(raw: Any) -> Dict[str, Path]:
    if not isinstance(raw, list) or not raw:
        raise PlanLoaderError("inputs must be a non-empty list")

    parsed: Dict[str, Path] = {}
    single_input = len(raw) == 1

    for item in raw:
        alias, path = parse_input_entry(item)

        if alias is None:
            if not single_input:
                raise PlanLoaderError(
                    f"input alias must not be empty when multiple inputs are provided: {item!r}"
                )
            alias = "model"

        if alias in parsed:
            raise PlanLoaderError(f"duplicate input alias: {alias!r}")
        parsed[alias] = path

    return parsed


def parse_input_entry(raw: Any) -> tuple[str, Path]:
    if not isinstance(raw, str) or not raw:
        raise PlanLoaderError("each inputs entry must be a non-empty string")

    if "::" in raw:
        alias, path_str = raw.split("::", 1)
        if not path_str:
            raise PlanLoaderError(f"input path must not be empty: {raw!r}")
        return alias, Path(path_str)

    # bare path → empty alias (resolved later if single input)
    return None, Path(raw)

def parse_output(raw: Any) -> OutputSpec:
    if isinstance(raw, str):
        if not raw:
            raise PlanLoaderError("output must be a non-empty string")
        return OutputSpec(path=Path(raw))

    if isinstance(raw, dict):
        return parse_output_mapping(raw)

    raise PlanLoaderError("output must be either a non-empty string or a mapping")


def parse_output_mapping(raw: Dict[str, Any]) -> OutputSpec:
    allowed_keys = {"path", "format", "shard"}

    unknown = set(raw) - allowed_keys
    if unknown:
        raise PlanLoaderError(f"output received unknown keys: {sorted(unknown)}")

    if "path" not in raw:
        raise PlanLoaderError("output.path is required")

    path_value = raw["path"]
    if not isinstance(path_value, str) or not path_value:
        raise PlanLoaderError("output.path must be a non-empty string")

    format_value = raw.get("format")
    if format_value is not None:
        if not isinstance(format_value, str) or not format_value:
            raise PlanLoaderError("output.format must be a non-empty string when provided")
        if format_value not in {"safetensors", "torch"}:
            raise PlanLoaderError("output.format must be one of: 'safetensors', 'torch'")

    shard_value = raw.get("shard")
    if shard_value is not None:
        if not isinstance(shard_value, str) or not shard_value:
            raise PlanLoaderError("output.shard must be a non-empty string when provided")

    return OutputSpec(
        path=Path(path_value),
        format=format_value,
        shard=shard_value,
    )


def parse_transforms(raw: Any, inputs: Dict[str, Path]) -> List[CompiledTransform]:
    if not isinstance(raw, list) or not raw:
        raise PlanLoaderError("transforms must be a non-empty list")

    default_model: Optional[str] = None
    if len(inputs) == 1:
        default_model = next(iter(inputs.keys()))

    parsed: List[CompiledTransform] = []
    for idx, item in enumerate(raw):
        parsed.append(parse_transform_entry(item, idx, inputs, default_model))

    return parsed


def parse_transform_entry(
    raw: Any,
    index: int,
    inputs: Dict[str, Path],
    default_model: Optional[str],
) -> CompiledTransform:
    if not isinstance(raw, dict) or len(raw) != 1:
        raise PlanLoaderError(f"transform #{index} must be a single-key mapping")

    op_name, payload = next(iter(raw.items()))

    try:
        transform = get_transform(op_name)
    except TransformError as exc:
        raise PlanLoaderError(f"transform #{index}: {exc}") from exc

    if not isinstance(payload, dict):
        raise PlanLoaderError(f"transform #{index}: payload must be a mapping")

    try:
        spec = transform.compile(payload, default_model)
    except TransformError as exc:
        raise PlanLoaderError(f"transform #{index}: {exc}") from exc

    validate_model_aliases(spec, inputs, index)

    return CompiledTransform(transform=transform, spec=spec)


def validate_model_aliases(spec: object, inputs: Dict[str, Path], index: int) -> None:
    for attr in ("from_ref", "to_ref", "target_ref", "ref", "left", "right"):
        validate_ref_model(getattr(spec, attr, None), inputs, index)

    expr = getattr(spec, "expr", None)
    if expr is not None:
        validate_expr_models(expr, inputs, index)


def validate_ref_model(ref: object, inputs: Dict[str, Path], index: int) -> None:
    if ref is None:
        return

    model = getattr(ref, "model", None)
    if model is not None and model not in inputs:
        raise PlanLoaderError(f"transform #{index}: unknown model alias: {model!r}")


def validate_expr_models(expr: object, inputs: Dict[str, Path], index: int) -> None:
    for attr in ("ref", "left", "right"):
        validate_ref_model(getattr(expr, attr, None), inputs, index)

    child = getattr(expr, "expr", None)
    if child is not None:
        validate_expr_models(child, inputs, index)

    children = getattr(expr, "exprs", None)
    if children is not None:
        for child_expr in children:
            validate_expr_models(child_expr, inputs, index)
