import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import (
    CompiledTransform,
    TransformControl,
    TransformError,
    TransformPayloadSchema,
    TransformResult,
    TypedTransform,
    ensure_mapping_payload,
    get_transform,
    register_transform,
    validate_payload_schema,
)
from ..engine import (
    emit_verbose_event,
    execute_transform_pairs,
    list_model_aliases,
    normalize_raw_plan,
    normalize_transform_specs,
)


class ExecuteTransformError(TransformError):
    pass


@dataclass(frozen=True)
class ExecuteSpec:
    raw_transforms: list[dict[str, Any]]
    default_model_hint: str | None = None

    def collect_models(self) -> set[str]:
        return set()


class ExecuteTransform(TypedTransform[ExecuteSpec]):
    name = "execute"
    error_type = ExecuteTransformError
    spec_type = ExecuteSpec
    allowed_keys = {"transforms", "plan", "plan-yaml", "path"}
    help_text = (
        "Executes a batch of transforms from payload data.\n"
        "\n"
        "Sources (can be combined):\n"
        "  - transforms: inline transform or list of transforms\n"
        "  - plan: plan object/list (inputs are converted to load transforms)\n"
        "  - plan-yaml: YAML plan text\n"
        "  - path: filesystem path to a YAML plan\n"
        "\n"
        "Flags like dry-run/preview/verbose are respected for nested transforms.\n"
        "Any plan output target is ignored by execute.\n"
        "\n"
        "Examples:\n"
        "  execute: { transforms: [{ copy: { from: model::a, to: model::b } }] }\n"
        '  execute: { plan-yaml: "transforms:\\n  - dump: { target: model::.* }" }\n'
    )

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key=None,
            default_mode="default",
            common_required=set(),
            common_allowed=set(self.allowed_keys),
        )

    def compile(self, payload: Any, default_model: str | None) -> ExecuteSpec:
        del default_model
        try:
            payload = ensure_mapping_payload(payload, self.name)
            validate_payload_schema(
                payload,
                op_name=self.name,
                schema=self.payload_schema(),
                error_type=self.error_type,
            )
        except TransformError as exc:
            raise ExecuteTransformError(str(exc)) from exc

        raw_transforms: list[dict[str, Any]] = []
        default_model_hint: str | None = None
        if "transforms" in payload:
            try:
                raw_transforms.extend(normalize_transform_specs(payload["transforms"]))
            except Exception as exc:
                raise ExecuteTransformError(f"execute.transforms invalid: {exc}") from exc

        if "plan" in payload:
            transforms, hint = _extract_plan_transforms(payload["plan"])
            raw_transforms.extend(transforms)
            if default_model_hint is None and hint is not None:
                default_model_hint = hint

        if "plan-yaml" in payload:
            raw_yaml = payload["plan-yaml"]
            if not isinstance(raw_yaml, str):
                raise ExecuteTransformError("execute.plan-yaml must be a string")
            parsed = _yaml_module().safe_load(raw_yaml) if raw_yaml.strip() else {}
            transforms, hint = _extract_plan_transforms(parsed)
            raw_transforms.extend(transforms)
            if default_model_hint is None and hint is not None:
                default_model_hint = hint

        if "path" in payload:
            raw_path = payload["path"]
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ExecuteTransformError("execute.path must be a non-empty string")
            text = Path(raw_path).read_text(encoding="utf-8")
            parsed = _yaml_module().safe_load(text) if text.strip() else {}
            transforms, hint = _extract_plan_transforms(parsed)
            raw_transforms.extend(transforms)
            if default_model_hint is None and hint is not None:
                default_model_hint = hint

        if not raw_transforms:
            raise ExecuteTransformError(
                "execute payload must provide at least one transform source"
            )

        return ExecuteSpec(raw_transforms=raw_transforms, default_model_hint=default_model_hint)

    def apply(self, spec: object, provider: Any) -> TransformResult:
        typed = self.require_spec(spec)
        raw_transforms = typed.raw_transforms
        if not raw_transforms:
            return TransformResult(name=self.name, count=0)

        compiled_pairs: list[tuple[dict[str, Any], CompiledTransform]] = []
        default_model = typed.default_model_hint
        if default_model is None:
            aliases = sorted(list_model_aliases(provider))
            if len(aliases) == 1:
                default_model = aliases[0]

        for index, raw_transform in enumerate(raw_transforms):
            op_name, payload = next(iter(raw_transform.items()))
            if op_name == self.name:
                raise ExecuteTransformError("execute cannot recursively invoke execute")

            transform = get_transform(op_name)
            try:
                compiled_spec = transform.compile(payload, default_model)
            except TransformError as exc:
                raise ExecuteTransformError(f"execute transform #{index}: {exc}") from exc

            compiled_pairs.append(
                (
                    raw_transform,
                    CompiledTransform(transform=transform, spec=compiled_spec),
                )
            )

        should_continue, executed = execute_transform_pairs(
            compiled_pairs,
            provider,
            interactive=False,
        )
        control = TransformControl.CONTINUE if should_continue else TransformControl.EXIT
        emit_verbose_event(self.name, f"executed {len(executed)}/{len(raw_transforms)}")
        return TransformResult(name=self.name, count=len(executed), control=control)

    def _infer_output_model(self, spec: object) -> str:
        typed = self.require_spec(spec)
        if not typed.raw_transforms:
            raise ExecuteTransformError(
                "execute does not infer output model for empty transform lists"
            )
        # Keep conservative: execute composes multiple transforms and may touch many aliases.
        raise ExecuteTransformError("execute does not infer an output model")

    def contributes_output_model(self, spec: object) -> bool:
        del spec
        return False


def _extract_plan_transforms(raw: Any) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(raw, list):
        try:
            return normalize_transform_specs(raw), None
        except Exception as exc:
            raise ExecuteTransformError(f"execute plan list invalid: {exc}") from exc

    if not isinstance(raw, dict):
        raise ExecuteTransformError("execute plan must be a mapping or transform list")

    try:
        normalized_plan = normalize_raw_plan(raw)
    except Exception as exc:
        raise ExecuteTransformError(f"execute plan invalid: {exc}") from exc

    raw_inputs = normalized_plan.get("inputs", [])
    raw_transforms = list(normalized_plan.get("transforms", []))
    default_model_hint: str | None = None
    if raw_inputs:
        input_loads, default_model_hint = _inputs_to_load_transforms(raw_inputs)
        raw_transforms = input_loads + raw_transforms
    output_spec = normalized_plan.get("output")
    save_transform = _output_to_save_transform(output_spec)
    if save_transform is not None:
        raw_transforms.append(save_transform)
    return raw_transforms, default_model_hint


def _inputs_to_load_transforms(raw_inputs: Any) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(raw_inputs, list):
        raise ExecuteTransformError("execute plan.inputs must be a list")
    if not raw_inputs:
        return [], None

    transforms: list[dict[str, Any]] = []
    aliases_seen: list[str] = []
    many = len(raw_inputs) > 1
    for index, entry in enumerate(raw_inputs):
        if not isinstance(entry, str) or not entry:
            raise ExecuteTransformError("execute plan.inputs entries must be non-empty strings")
        if "::" in entry:
            alias, path = entry.split("::", 1)
            if not alias:
                alias = "model" if not many else f"model_{index + 1}"
            if not path:
                raise ExecuteTransformError(f"execute plan input path must not be empty: {entry!r}")
        else:
            if many:
                raise ExecuteTransformError(
                    "execute plan with multiple inputs requires explicit alias::path entries"
                )
            alias, path = "model", entry
        aliases_seen.append(alias)
        transforms.append({"load": {"path": path, "alias": alias}})
    unique_aliases = sorted(set(aliases_seen))
    default_model_hint = unique_aliases[0] if len(unique_aliases) == 1 else None
    return transforms, default_model_hint


def _yaml_module() -> Any:
    return importlib.import_module("yaml")


def _output_to_save_transform(raw_output: Any) -> dict[str, Any] | None:
    if raw_output is None:
        return None

    if isinstance(raw_output, str):
        text = raw_output.strip()
        if not text:
            return None
        alias: str | None = None
        path = text
        if "::" in text:
            alias_part, path_part = text.split("::", 1)
            if alias_part and path_part:
                alias = alias_part
                path = path_part
        payload: dict[str, Any] = {"path": path}
        if alias is not None:
            payload["alias"] = alias
        return {"save": payload}

    if isinstance(raw_output, dict):
        path_raw = raw_output.get("path")
        if not isinstance(path_raw, str) or not path_raw.strip():
            raise ExecuteTransformError("execute plan.output.path must be a non-empty string")
        output_alias: str | None = None
        path = path_raw.strip()
        if "::" in path:
            alias_part, path_part = path.split("::", 1)
            if alias_part and path_part:
                output_alias = alias_part
                path = path_part

        output_payload: dict[str, Any] = {"path": path}
        if output_alias is not None:
            output_payload["alias"] = output_alias
        for key in ("format", "shard"):
            value = raw_output.get(key)
            if value is not None:
                output_payload[key] = value
        return {"save": output_payload}

    raise ExecuteTransformError("execute plan.output must be either a string or a mapping")


register_transform(ExecuteTransform())
