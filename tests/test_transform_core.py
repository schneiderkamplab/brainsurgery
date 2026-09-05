from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from brainsurgery.core import (
    BaseTransform,
    CompiledTransform,
    TransformError,
    TransformResult,
    get_transform,
    list_transforms,
    register_transform,
)
from brainsurgery.core.runtime.transform import REGISTRY
from brainsurgery.core.specs.validation import (
    TransformPayloadSchema,
    ensure_mapping_payload,
    require_expr,
    require_nonempty_string,
    require_numeric,
    validate_payload_keys,
)
from brainsurgery.engine.output_model import _infer_output_model
from brainsurgery.engine.plan import PlanStep, SurgeryPlan
from brainsurgery.engine.state_dicts import _InMemoryStateDict
from brainsurgery.transforms.copy import CopyTransform
from brainsurgery.transforms.help import HelpTransform
from brainsurgery.transforms.save import SaveTransform


@dataclass(frozen=True)
class _Spec:
    model: str

    def collect_models(self) -> set[str]:
        return {self.model, "other"}


class _Transform(BaseTransform):
    name = "dummy"

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key=None,
            default_mode="default",
            common_required=set(),
            common_allowed=set(),
        )

    def compile(self, payload: dict, default_model: str | None) -> object:
        del payload, default_model
        return _Spec(model="model")

    def apply(self, spec: object, provider: object) -> TransformResult:
        del spec, provider
        return TransformResult(name=self.name, count=1)

    def _infer_output_model(self, spec: object) -> str:
        assert isinstance(spec, _Spec)
        return spec.model


class _FallbackTransform(_Transform):
    name = "fallback"

    def _infer_output_model(self, spec: object) -> str:
        del spec
        raise TransformError("needs provider")


class _Provider:
    def __init__(self) -> None:
        self.state_dicts = {
            "model": _InMemoryStateDict(),
            "other": _InMemoryStateDict(),
        }
        self.state_dicts["model"]["w"] = torch.ones(1)

    def get_state_dict(self, model: str) -> _InMemoryStateDict:
        return self.state_dicts[model]


class _SpecBadCollect:
    def collect_models(self) -> list[str]:
        return ["model"]


def _plan_with(compiled: list[CompiledTransform]) -> SurgeryPlan:
    return SurgeryPlan(
        inputs={},
        output=None,
        steps=[PlanStep(raw={}, compiled=item) for item in compiled],
    )


def test_transform_registry_registers_lists_and_rejects_duplicates() -> None:
    original_registry = dict(REGISTRY)
    REGISTRY.clear()
    try:
        transform = _Transform()
        register_transform(transform)

        assert get_transform("dummy") is transform
        assert list_transforms() == ["dummy"]

        with pytest.raises(TransformError, match="already registered"):
            register_transform(_Transform())
    finally:
        REGISTRY.clear()
        REGISTRY.update(original_registry)


def test_infer_output_model_uses_provider_fallback_when_needed() -> None:
    plan = _plan_with([CompiledTransform(_FallbackTransform(), _Spec("model"))])
    assert _infer_output_model(plan, _Provider()) == "model"


def test_infer_output_model_skips_non_contributing_transforms() -> None:
    plan = _plan_with(
        [
            CompiledTransform(HelpTransform(), HelpTransform().compile("copy", default_model=None)),
            CompiledTransform(
                SaveTransform(),
                SaveTransform().compile(
                    {"path": "/tmp/out.safetensors", "alias": "model"}, default_model=None
                ),
            ),
            CompiledTransform(
                CopyTransform(),
                CopyTransform().compile(
                    {"from": "model::w", "to": "model::w_copy"}, default_model=None
                ),
            ),
        ]
    )

    assert _infer_output_model(plan, _Provider()) == "model"


def test_infer_output_model_raises_when_no_destination_model() -> None:
    plan = _plan_with(
        [CompiledTransform(HelpTransform(), HelpTransform().compile({}, default_model=None))]
    )
    with pytest.raises(TransformError, match="cannot infer output model uniquely"):
        _infer_output_model(plan, _Provider())


def test_infer_output_model_raises_when_multiple_destination_models() -> None:
    provider = _Provider()
    provider.state_dicts["other"]["w"] = torch.ones(1)
    plan = _plan_with(
        [
            CompiledTransform(_Transform(), _Spec("model")),
            CompiledTransform(_Transform(), _Spec("other")),
        ]
    )
    with pytest.raises(TransformError, match="cannot infer output model uniquely"):
        _infer_output_model(plan, provider)


def test_infer_output_model_fallback_requires_provider() -> None:
    plan = _plan_with([CompiledTransform(_FallbackTransform(), _Spec("model"))])
    with pytest.raises(TransformError, match="needs provider"):
        _infer_output_model(plan, None)


def test_infer_output_model_fallback_rejects_non_set_collect_models() -> None:
    plan = _plan_with([CompiledTransform(_FallbackTransform(), _SpecBadCollect())])
    with pytest.raises(TransformError, match="needs provider"):
        _infer_output_model(plan, _Provider())


def test_validate_payload_helpers_cover_required_unknown_and_type_errors() -> None:
    validate_payload_keys(
        {"from": "a"}, op_name="copy", allowed_keys={"from"}, required_keys={"from"}
    )
    assert ensure_mapping_payload({"x": 1}, "copy") == {"x": 1}
    assert require_nonempty_string({"alias": "base"}, op_name="load", key="alias") == "base"
    assert require_expr({"from": ["a", "b"]}, op_name="copy", key="from") == ["a", "b"]
    assert require_numeric({"factor": "1.5"}, op_name="scale", key="factor") == 1.5

    with pytest.raises(TransformError, match="unknown keys"):
        validate_payload_keys({"extra": 1}, op_name="copy", allowed_keys={"from"})

    with pytest.raises(TransformError, match="copy.to is required"):
        validate_payload_keys(
            {"from": "a"},
            op_name="copy",
            allowed_keys={"from", "to"},
            required_keys={"from", "to"},
        )

    with pytest.raises(TransformError, match="must be a mapping"):
        ensure_mapping_payload([], "copy")

    with pytest.raises(TransformError, match="must be a non-empty string"):
        require_nonempty_string({"alias": ""}, op_name="load", key="alias")

    with pytest.raises(TransformError, match="non-empty string or a non-empty list"):
        require_expr({"from": []}, op_name="copy", key="from")

    with pytest.raises(TransformError, match="must be numeric"):
        require_numeric({"factor": "nope"}, op_name="scale", key="factor")
