import copy
import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import torch
from omegaconf import OmegaConf

from brainsurgery.core import (
    BinaryMappingTransform,
    CompiledTransform,
    DestinationPolicy,
    IteratingTransform,
    TernaryMappingTransform,
    UnaryTransform,
    get_assert_expr_help,
    get_assert_expr_names,
    get_transform,
    list_transforms,
    match_expr_names,
    parse_model_expr,
    use_progress_callback,
)
from brainsurgery.engine import (
    SurgeryPlan,
    executed_plan_summary_yaml,
    format_preview_impact,
    get_runtime_flags,
    list_model_aliases,
    parse_summary_mode,
    preview_impact_for_transform,
    preview_requires_confirmation,
    use_output_emitter,
)

from .models import (
    ErrorInfoPayload,
    ErrorResponsePayload,
    RuntimeFlagsPayload,
    model_to_payload,
)

DISABLED_TRANSFORMS: set[str] = set()


def _normalize_assert_payload(payload: Any) -> Any:
    if not isinstance(payload, str):
        return payload
    text = payload.strip()
    if not text:
        raise ValueError("assert payload cannot be empty.")
    try:
        parsed = OmegaConf.create(text)
        return OmegaConf.to_container(parsed, resolve=True)
    except Exception as exc:
        raise ValueError(f"invalid assert YAML payload: {exc}") from exc


def _transform_items() -> list[dict[str, Any]]:
    specs = _transform_specs()
    items: list[dict[str, Any]] = []
    for name in list_transforms():
        if name in DISABLED_TRANSFORMS:
            continue
        spec = specs.get(name)
        items.append(
            {
                "name": name,
                "enabled": bool(spec and spec["enabled"]),
                "binary": bool(spec),
                "kind": spec["kind"] if spec else "other",
                "iterating": bool(spec and spec["iterating"]),
                "allowed_keys": spec["allowed_keys"] if spec else [],
                "required_keys": spec["required_keys"] if spec else [],
                "mode_key": spec["mode_key"] if spec else None,
                "default_mode": spec["default_mode"] if spec else "default",
                "modes": spec["modes"] if spec else ["default"],
                "mode_allowed_keys": spec["mode_allowed_keys"] if spec else {},
                "mode_required_keys": spec["mode_required_keys"] if spec else {},
                "reference_keys": spec["reference_keys"] if spec else [],
                "to_must_exist": bool(spec and spec["to_must_exist"]),
                "help_commands": spec["help_commands"] if spec else [],
                "help_subcommands": spec["help_subcommands"] if spec else {},
                "assert_expressions": spec["assert_expressions"] if spec else [],
                "assert_expression_meta": spec["assert_expression_meta"] if spec else {},
                "boolean_keys": spec["boolean_keys"] if spec else [],
            }
        )
    return items


def _transform_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for name in list_transforms():
        transform = get_transform(name)
        schema = transform.payload_schema()
        default_mode = str(schema.default_mode)
        mode_names = sorted(schema.modes())
        allowed = sorted(schema.allowed_keys_for_mode(default_mode))
        required = sorted(schema.required_keys_for_mode(default_mode))
        ref_keys = [key for key in transform.completion_reference_keys() if isinstance(key, str)]
        if allowed:
            ref_keys = [key for key in ref_keys if key in set(allowed)]
        if isinstance(transform, BinaryMappingTransform):
            kind = "binary"
        elif isinstance(transform, UnaryTransform):
            kind = "unary"
        elif isinstance(transform, TernaryMappingTransform):
            kind = "ternary"
        else:
            kind = "other"
        destination_policy = getattr(transform, "destination_policy", DestinationPolicy.ANY)
        specs[name] = {
            "enabled": name not in DISABLED_TRANSFORMS,
            "kind": kind,
            "iterating": isinstance(transform, IteratingTransform),
            "allowed_keys": allowed,
            "required_keys": required,
            "mode_key": schema.mode_key,
            "default_mode": default_mode,
            "modes": mode_names,
            "mode_allowed_keys": {
                mode: sorted(schema.allowed_keys_for_mode(mode)) for mode in mode_names
            },
            "mode_required_keys": {
                mode: sorted(schema.required_keys_for_mode(mode)) for mode in mode_names
            },
            "reference_keys": ref_keys,
            "to_must_exist": destination_policy is DestinationPolicy.MUST_EXIST,
            "help_commands": sorted(list_transforms()) if name == "help" else [],
            "help_subcommands": {"assert": sorted(get_assert_expr_names())}
            if name == "help"
            else {},
            "assert_expressions": sorted(get_assert_expr_names()) if name == "assert" else [],
            "assert_expression_meta": _assert_expression_meta() if name == "assert" else {},
            "boolean_keys": _boolean_keys_for_transform(transform, allowed),
        }
    return specs


def _is_boolean_annotation(annotation: Any) -> bool:
    if annotation is bool:
        return True
    origin = get_origin(annotation)
    if origin is None:
        return False
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    return bool(args) and all(arg is bool for arg in args)


def _boolean_keys_for_transform(transform: Any, allowed_keys: list[str]) -> list[str]:
    out: set[str] = set()
    allowed = set(allowed_keys)

    spec_type = getattr(transform, "spec_type", None)
    if spec_type is not None and is_dataclass(spec_type):
        try:
            type_hints = get_type_hints(spec_type)
        except Exception:
            type_hints = {}
        for field_name, annotation in type_hints.items():
            if _is_boolean_annotation(annotation):
                key = field_name.replace("_", "-")
                if key in allowed:
                    out.add(key)

    help_text = str(getattr(transform, "help_text", "") or "")
    for line in help_text.splitlines():
        text = line.strip()
        if not text.startswith("-") or "(boolean)" not in text.lower():
            continue
        key = text[1:].split("(", 1)[0].strip()
        if key in allowed:
            out.add(key)

    return sorted(out)


def _assert_expression_meta() -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for expr_name in sorted(get_assert_expr_names()):
        expr_help = get_assert_expr_help(expr_name)
        assert not isinstance(expr_help, dict)
        meta[expr_name] = {
            "payload_kind": expr_help.payload_kind,
            "allowed_keys": sorted(expr_help.allowed_keys or set()),
            "required_keys": sorted(expr_help.required_keys or set()),
            "description": expr_help.description or "",
        }
    return meta


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _assert_error_context(payload: Any) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if isinstance(payload, str):
        context["raw_payload"] = payload.strip()
        return context
    if not isinstance(payload, dict):
        return context
    if len(payload) == 1:
        expr_name = next(iter(payload))
        context["expression"] = expr_name
        expr_payload = payload.get(expr_name)
        if isinstance(expr_payload, dict):
            context["expression_keys"] = sorted(str(key) for key in expr_payload.keys())
    return context


def _api_error_payload(
    exc: Exception,
    *,
    endpoint: str,
    transform_name: str | None = None,
    payload: Any | None = None,
) -> dict[str, Any]:
    message = str(exc).strip() or exc.__class__.__name__
    code = "assert_error" if transform_name == "assert" else "request_error"
    location: dict[str, Any] | None = None
    context_payload: dict[str, Any] | None = None
    if transform_name == "assert":
        location = {
            "transform": "assert",
            "field": "payload",
        }
        context = _assert_error_context(payload)
        if context:
            context_payload = context

    return model_to_payload(
        ErrorResponsePayload(
            ok=False,
            error=message,
            error_info=ErrorInfoPayload(
                code=code,
                message=message,
                endpoint=endpoint,
                transform=transform_name,
                exception_type=type(exc).__name__,
                location=location,
                context=context_payload,
            ),
        )
    )


def _default_alias(provider: Any) -> str:
    aliases = set(list_model_aliases(provider))
    base = "model"
    if base not in aliases:
        return base
    index = 2
    while True:
        candidate = f"{base}_{index}"
        if candidate not in aliases:
            return candidate
        index += 1


def _parse_filter_expr(raw: Any, *, alias: str) -> str | list[Any]:
    source: Any
    if raw is None:
        source = ".*"
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            source = ".*"
        elif text.startswith("["):
            parsed = json.loads(text)
            source = parsed
        else:
            source = text
    elif isinstance(raw, list):
        source = raw
    else:
        raise ValueError("filter must be a string or JSON list.")

    ref = parse_model_expr(source, default_model=alias)
    return ref.expr


def _apply_transform(
    *,
    provider: Any,
    plan: SurgeryPlan,
    transform_name: str,
    payload: Any,
    default_model: str | None = None,
    record_payload: Any | None = None,
    progress_callback: Any | None = None,
) -> tuple[str, str | None]:
    if transform_name in DISABLED_TRANSFORMS:
        raise ValueError(f"transform {transform_name!r} is disabled in webui.")

    if transform_name == "assert":
        payload = _normalize_assert_payload(payload)

    if default_model is None:
        aliases = sorted(list_model_aliases(provider))
        default_model = aliases[0] if len(aliases) == 1 else None

    raw_transform = {transform_name: copy.deepcopy(payload)}
    recorded_transform = {
        transform_name: copy.deepcopy(payload if record_payload is None else record_payload),
    }

    return _execute_plan_transform(
        provider=provider,
        plan=plan,
        raw_transform=raw_transform,
        recorded_transform=recorded_transform,
        default_model=default_model,
        progress_callback=progress_callback,
    )


def _preview_transform(
    *,
    provider: Any,
    transform_name: str,
    payload: Any,
    default_model: str | None,
) -> tuple[str, bool, dict[str, Any]]:
    transform = get_transform(transform_name)
    compiled = CompiledTransform(
        transform=transform,
        spec=transform.compile(copy.deepcopy(payload), default_model),
    )
    impact = preview_impact_for_transform(compiled, provider)
    impact_payload = _preview_impact_payload(transform_name=transform_name, impact=impact)
    output = f"preview 1/1 {transform_name}: {format_preview_impact(impact)}"
    return output, preview_requires_confirmation(transform_name, impact), impact_payload


def _apply_load_transform(*, provider: Any, plan: SurgeryPlan, path: Path, alias: str) -> None:
    _execute_plan_transform(
        provider=provider,
        plan=plan,
        raw_transform={"load": {"path": str(path), "alias": alias}},
        recorded_transform={"load": {"path": str(path), "alias": alias}},
        default_model=None,
        progress_callback=None,
    )


def _execute_plan_transform(
    *,
    provider: Any,
    plan: SurgeryPlan,
    raw_transform: dict[str, Any],
    recorded_transform: dict[str, Any],
    default_model: str | None,
    progress_callback: Any | None = None,
) -> tuple[str, str | None]:
    step = plan.append_raw_transform(raw_transform)
    lines: list[str] = []
    with use_output_emitter(lines.append):
        with use_progress_callback(progress_callback):
            plan.compile_pending(
                extra_known_models=set(list_model_aliases(provider)),
                default_model=default_model,
            )
            plan.execute_pending(provider, interactive=False)

    step.raw = recorded_transform

    next_default_model = default_model
    try:
        if step.compiled is not None:
            output_alias = step.compiled.transform._infer_output_model(step.compiled.spec)
            if output_alias:
                next_default_model = output_alias
    except Exception:
        next_default_model = default_model
    return "\n".join(lines), next_default_model


def _serialize_runtime_flags() -> RuntimeFlagsPayload:
    flags = get_runtime_flags()
    return RuntimeFlagsPayload(
        dry_run=bool(flags.dry_run),
        preview=bool(flags.preview),
        verbose=bool(flags.verbose),
    )


def _render_dump_for_alias(
    *,
    provider: Any,
    alias: str,
    format_name: str,
    verbosity: str,
    target: str | list[Any],
) -> tuple[str, int, int]:
    names = provider.get_state_dict(alias).keys()
    matched = match_expr_names(
        expr=target,
        names=names,
        op_name="webui.dump",
        role="target",
    )
    total_count = len(list(names))
    matched_count = len(matched)
    if not matched:
        return "(no tensors matched filter)", 0, total_count

    dump_transform = get_transform("dump")
    spec = dump_transform.compile(
        {
            "target": target,
            "format": format_name,
            "verbosity": verbosity,
        },
        default_model=alias,
    )
    lines: list[str] = []
    with use_output_emitter(lines.append):
        dump_transform.apply(spec, provider)
    return "\n".join(lines), matched_count, total_count


def _serialize_models(provider: Any) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for alias in sorted(list_model_aliases(provider)):
        state_dict = provider.get_state_dict(alias)
        tensor_count = len(state_dict)
        models.append(
            {
                "alias": alias,
                "tensor_count": tensor_count,
                "matched_count": tensor_count,
                "total_count": tensor_count,
            }
        )
    return models


def _preview_impact_payload(*, transform_name: str, impact: Any) -> dict[str, Any]:
    changed = sorted(str(item) for item in getattr(impact, "changed", set()) or set())
    created = sorted(str(item) for item in getattr(impact, "created", set()) or set())
    deleted = sorted(str(item) for item in getattr(impact, "deleted", set()) or set())
    return {
        "transform": transform_name,
        "changed_count": len(changed),
        "created_count": len(created),
        "deleted_count": len(deleted),
        "changed_samples": changed[:5],
        "created_samples": created[:5],
        "deleted_samples": deleted[:5],
    }


def _module_prefix(name: str, *, depth: int = 2) -> str:
    parts = name.split(".")
    if len(parts) <= depth:
        return name
    return ".".join(parts[:depth])


def _coerce_stat_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_complex():
        return torch.abs(tensor.to(torch.complex128))
    if tensor.is_floating_point():
        return tensor
    return tensor.to(torch.float32)


def _module_health_for_alias(
    *,
    provider: Any,
    alias: str,
    target: str | list[Any],
    top_n: int = 10,
) -> dict[str, Any]:
    state_dict = provider.get_state_dict(alias)
    names = list(state_dict.keys())
    matched = match_expr_names(
        expr=target,
        names=names,
        op_name="webui.module_health",
        role="target",
    )
    if not matched:
        return {"summary": {"matched_count": 0, "total_count": len(names)}, "modules": []}

    buckets: dict[str, dict[str, Any]] = {}
    total_elements = 0
    total_tensors = 0

    for name in matched:
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            continue
        total_tensors += 1
        module = _module_prefix(name, depth=2)
        bucket = buckets.setdefault(
            module,
            {
                "module": module,
                "tensor_count": 0,
                "element_count": 0,
                "abs_mean_weighted_sum": 0.0,
                "max_abs": 0.0,
                "near_zero_count": 0,
            },
        )
        bucket["tensor_count"] += 1
        numel = int(tensor.numel())
        if numel <= 0:
            continue

        stat_tensor = _coerce_stat_tensor(tensor.detach())
        abs_tensor = torch.abs(stat_tensor)
        abs_mean = float(abs_tensor.mean().item())
        max_abs = float(abs_tensor.max().item())
        near_zero = int((abs_tensor <= 1e-12).sum().item())

        bucket["element_count"] += numel
        bucket["abs_mean_weighted_sum"] += abs_mean * numel
        bucket["max_abs"] = max(bucket["max_abs"], max_abs)
        bucket["near_zero_count"] += near_zero

        total_elements += numel

    modules: list[dict[str, Any]] = []
    for bucket in buckets.values():
        elements = int(bucket["element_count"])
        abs_mean = float(bucket["abs_mean_weighted_sum"]) / elements if elements > 0 else 0.0
        near_zero_ratio = float(bucket["near_zero_count"]) / elements if elements > 0 else 0.0
        modules.append(
            {
                "module": bucket["module"],
                "tensor_count": int(bucket["tensor_count"]),
                "element_count": elements,
                "abs_mean": abs_mean,
                "max_abs": float(bucket["max_abs"]),
                "near_zero_ratio": near_zero_ratio,
            }
        )

    modules.sort(key=lambda item: (item["abs_mean"], item["max_abs"]), reverse=True)
    return {
        "summary": {
            "matched_count": len(matched),
            "total_count": len(names),
            "tensor_count": total_tensors,
            "element_count": total_elements,
        },
        "modules": modules[:top_n],
    }


def _value_diff_max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.is_complex():
        diff = torch.abs(left.to(torch.complex128) - right.to(torch.complex128))
    else:
        diff = torch.abs(left.to(torch.float64) - right.to(torch.float64))
    return float(diff.max().item()) if diff.numel() else 0.0


def _model_diff_summary(
    *,
    provider: Any,
    left_alias: str,
    right_alias: str,
    left_target: str | list[Any],
    right_target: str | list[Any],
    eps: float | None = None,
    top_n: int = 20,
) -> dict[str, Any]:
    left_sd = provider.get_state_dict(left_alias)
    right_sd = provider.get_state_dict(right_alias)
    left_names = set(
        match_expr_names(
            expr=left_target,
            names=left_sd.keys(),
            op_name="webui.diff",
            role="left",
        )
    )
    right_names = set(
        match_expr_names(
            expr=right_target,
            names=right_sd.keys(),
            op_name="webui.diff",
            role="right",
        )
    )

    missing_left = sorted(right_names - left_names)
    missing_right = sorted(left_names - right_names)
    shared = sorted(left_names & right_names)

    changed = 0
    unchanged = 0
    by_kind = {"shape": 0, "dtype": 0, "device": 0, "values": 0}
    top_differences: list[dict[str, Any]] = []

    for name in shared:
        left = left_sd[name]
        right = right_sd[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            by_kind["values"] += 1
            changed += 1
            top_differences.append({"name": name, "kind": "values", "max_abs_diff": None})
            continue

        if left.shape != right.shape:
            by_kind["shape"] += 1
            changed += 1
            top_differences.append({"name": name, "kind": "shape", "max_abs_diff": None})
            continue
        if left.dtype != right.dtype:
            by_kind["dtype"] += 1
            changed += 1
            top_differences.append({"name": name, "kind": "dtype", "max_abs_diff": None})
            continue
        if left.device != right.device:
            by_kind["device"] += 1
            changed += 1
            top_differences.append({"name": name, "kind": "device", "max_abs_diff": None})
            continue

        max_abs_diff = _value_diff_max_abs(left, right)
        if eps is None:
            equal = bool(torch.equal(left, right))
        else:
            equal = max_abs_diff <= eps
        if equal:
            unchanged += 1
            continue

        by_kind["values"] += 1
        changed += 1
        top_differences.append(
            {"name": name, "kind": "values", "max_abs_diff": float(max_abs_diff)}
        )

    top_differences.sort(
        key=lambda item: (
            item["max_abs_diff"] if item["max_abs_diff"] is not None else -1.0,
            item["name"],
        ),
        reverse=True,
    )

    return {
        "summary": {
            "left_alias": left_alias,
            "right_alias": right_alias,
            "left_count": len(left_names),
            "right_count": len(right_names),
            "shared_count": len(shared),
            "missing_on_left": len(missing_left),
            "missing_on_right": len(missing_right),
            "changed": changed,
            "unchanged": unchanged,
            "by_kind": by_kind,
        },
        "missing_on_left_samples": missing_left[:10],
        "missing_on_right_samples": missing_right[:10],
        "top_differences": top_differences[:top_n],
    }


def _render_execution_summary(*, plan: SurgeryPlan, mode: str = "raw") -> str:
    summary_mode = parse_summary_mode(mode)
    return executed_plan_summary_yaml(plan, mode=summary_mode)
