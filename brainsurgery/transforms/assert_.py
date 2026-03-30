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
)
from ..engine import emit_verbose_event
from ..expressions import Expression, compile_assert_expr, get_assert_expr_names


@dataclass(frozen=True)
class AssertSpec:
    expr: Expression

    def collect_models(self) -> set[str]:
        return self.expr.collect_models()


class AssertTransform(TypedTransform[AssertSpec]):
    name = "assert"
    error_type = TransformError
    spec_type = AssertSpec
    help_text = (
        "Checks conditions on tensors using a single assert expression. "
        "The operation fails if the assertion does not hold and does not modify tensors.\n"
        "\n"
        "Expressions can match tensors by name or pattern and may include slicing "
        "(written after '::') and nesting.\n"
        "\n"
        "Examples:\n"
        "  assert: { equal: { left: a.weight, right: b.weight } }\n"
        "  assert: { not: { equal: { left: a.weight, right: b.weight } } }\n"
        "  assert: { dtype: { of: ln_f.weight, is: float32 } }\n"
        "  assert: { shape: { of: 'ln_f.weight::[:8]', is: [8] } }\n"
        "  assert: { reads: { of: 'model::.*', ge: 1 } }\n"
        "  assert: { writes: { of: ln_f.weight, lt: 2 } }\n"
        "  assert: { dimensions: { of: '.*weight', ge: 2 } }\n"
        "  assert: { all: [ { exists: '.*weight' }, { dimensions: { of: ln_f.weight, is: 1 } } ] }"
    )

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key=None,
            default_mode="default",
            common_required=set(),
            common_allowed=set(),
        )

    def completion_payload_start_candidates(self, prefix_text: str) -> list[str] | None:
        candidates = [name for name in get_assert_expr_names() if name.startswith(prefix_text)]
        if not prefix_text:
            return candidates + ["{ "]
        return candidates

    def completion_key_candidates(self, before_cursor: str, prefix_text: str) -> list[str] | None:
        del before_cursor
        return [f"{name}: " for name in get_assert_expr_names() if name.startswith(prefix_text)]

    def compile(self, payload: Any, default_model: str | None) -> AssertSpec:
        payload = ensure_mapping_payload(payload, self.name)
        expr = compile_assert_expr(payload, default_model)
        return AssertSpec(expr=expr)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        typed.expr.evaluate(provider)
        emit_verbose_event(self.name, "ok")
        return TransformResult(name=self.name, count=1)

    def _infer_output_model(self, spec: object) -> str:
        models = self.require_spec(spec).collect_models()
        if len(models) != 1:
            raise self.error_type(
                f"{self.name} must refer to exactly one model to infer output, got {sorted(models)}"
            )

        return next(iter(models))

    def contributes_output_model(self, spec: object) -> bool:
        del spec
        return False


register_transform(AssertTransform())
