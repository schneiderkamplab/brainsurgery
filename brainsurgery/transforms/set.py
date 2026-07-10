from dataclasses import dataclass
from typing import Any

from ..core import (
    StateDictProvider,
    TransformError,
    TransformPayloadSchema,
    TransformResult,
    TypedTransform,
    ensure_mapping_payload,
    register_transform,
    validate_payload_schema,
)
from ..engine import emit_line, emit_verbose_event, get_runtime_flags, set_runtime_flag


class SetTransformError(TransformError):
    pass


@dataclass(frozen=True)
class SetSpec:
    dry_run: bool | None = None
    preview: bool | None = None
    verbose: bool | None = None

    def collect_models(self) -> set[str]:
        return set()


class SetTransform(TypedTransform[SetSpec]):
    name = "set"
    error_type = SetTransformError
    spec_type = SetSpec
    allowed_keys = {"dry-run", "preview", "verbose"}
    help_text = (
        "Sets runtime flags used by the execution session.\n"
        "\n"
        "Supported flags:\n"
        "  - dry-run (boolean)\n"
        "  - preview (boolean)\n"
        "  - verbose (boolean)\n"
        "\n"
        "Boolean values may be: true/false, True/False, T/F.\n"
        "preview emits compact impact summaries before each transform apply.\n"
        "If no flags are provided, current values are printed.\n"
        "\n"
        "Examples:\n"
        "  set: {}\n"
        "  set: { dry-run: true }\n"
        "  set: { preview: true }\n"
        "  set: { verbose: T }\n"
        "  set: { dry-run: False, preview: true, verbose: true }"
    )

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key=None,
            default_mode="default",
            common_required=set(),
            common_allowed=set(self.allowed_keys),
        )

    def compile(self, payload: Any, default_model: str | None) -> SetSpec:
        del default_model

        try:
            payload = ensure_mapping_payload(payload, self.name)
        except TransformError as exc:
            raise SetTransformError(str(exc)) from exc
        try:
            validate_payload_schema(
                payload,
                op_name=self.name,
                schema=self.payload_schema(),
                error_type=self.error_type,
            )
        except TransformError as exc:
            raise SetTransformError(str(exc)) from exc
        dry_run: bool | None = None
        preview: bool | None = None
        verbose: bool | None = None
        if "dry-run" in payload:
            dry_run = _parse_bool(payload["dry-run"], field_name="dry-run")
        if "preview" in payload:
            preview = _parse_bool(payload["preview"], field_name="preview")
        if "verbose" in payload:
            verbose = _parse_bool(payload["verbose"], field_name="verbose")

        return SetSpec(
            dry_run=dry_run,
            preview=preview,
            verbose=verbose,
        )

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        del provider

        typed = self.require_spec(spec)
        changed = 0
        flags_before = get_runtime_flags()

        if typed.dry_run is not None:
            set_runtime_flag("dry_run", typed.dry_run)
            changed += 1
        if typed.preview is not None:
            set_runtime_flag("preview", typed.preview)
            changed += 1
        if typed.verbose is not None:
            set_runtime_flag("verbose", typed.verbose)
            changed += 1

        if changed == 0:
            emit_line(
                "set flags: "
                f"dry-run={flags_before.dry_run}, "
                f"preview={flags_before.preview}, "
                f"verbose={flags_before.verbose}"
            )

        emit_verbose_event(
            self.name,
            f"dry-run={typed.dry_run!r}, preview={typed.preview!r}, verbose={typed.verbose!r}",
        )
        return TransformResult(name=self.name, count=changed)

    def _infer_output_model(self, spec: object) -> str:
        del spec
        raise SetTransformError("set does not infer an output model")

    def contributes_output_model(self, spec: object) -> bool:
        del spec
        return False

    def completion_key_candidates(self, before_cursor: str, prefix_text: str) -> list[str] | None:
        del before_cursor
        candidates = [f"{name}: " for name in sorted(self.allowed_keys)]
        return [candidate for candidate in candidates if candidate.startswith(prefix_text)]

    def completion_value_candidates(
        self,
        value_key: str | None,
        prefix_text: str,
        model_aliases: list[str],
    ) -> list[str] | None:
        del model_aliases
        if value_key not in self.allowed_keys:
            return None
        candidates = ["false", "true", "F", "T", "False", "True"]
        return [candidate for candidate in candidates if candidate.startswith(prefix_text)]


def _parse_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"t", "true"}:
            return True
        if lowered in {"f", "false"}:
            return False

    raise SetTransformError(f"set.{field_name} must be a boolean (T, true, True, F, false, False)")


register_transform(SetTransform())
