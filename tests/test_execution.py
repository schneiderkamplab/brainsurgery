from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from brainsurgery.core import (
    BaseTransform,
    CompiledTransform,
    TensorRef,
    TransformControl,
    TransformError,
    TransformPayloadSchema,
    TransformResult,
)
from brainsurgery.engine import (
    RuntimeFlagLifecycleScope,
    reset_runtime_flags_for_scope,
    set_runtime_flag,
    use_output_emitter,
)
from brainsurgery.engine.execution import _execute_transform_pairs


class _Transform(BaseTransform):
    name = "dummy"

    def __init__(
        self, *, control: TransformControl = TransformControl.CONTINUE, fail: bool = False
    ):
        self.control = control
        self.fail = fail

    def compile(self, payload: dict, default_model: str | None) -> object:
        del payload, default_model
        return object()

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key=None,
            default_mode="default",
            common_required=set(),
            common_allowed=set(),
        )

    def apply(self, spec: object, provider: object) -> TransformResult:
        del spec, provider
        if self.fail:
            raise TransformError("boom")
        return TransformResult(name=self.name, count=1, control=self.control)

    def _infer_output_model(self, spec: object) -> str:
        del spec
        return "model"


def test_execute_transform_pairs_stops_on_exit_control() -> None:
    pairs = [
        ({"first": {}}, CompiledTransform(_Transform(), object())),
        ({"exit": {}}, CompiledTransform(_Transform(control=TransformControl.EXIT), object())),
    ]

    should_continue, executed = _execute_transform_pairs(pairs, object(), interactive=False)
    assert should_continue is False
    assert executed == [{"first": {}}, {"exit": {}}]


def test_execute_transform_pairs_returns_to_prompt_on_interactive_failure() -> None:
    pairs = [
        ({"ok": {}}, CompiledTransform(_Transform(), object())),
        ({"bad": {}}, CompiledTransform(_Transform(fail=True), object())),
    ]

    should_continue, executed = _execute_transform_pairs(pairs, object(), interactive=True)
    assert should_continue is True
    assert executed == [{"ok": {}}]


def test_execute_transform_pairs_raises_in_non_interactive_mode() -> None:
    with pytest.raises(TransformError, match="boom"):
        _execute_transform_pairs(
            [({"bad": {}}, CompiledTransform(_Transform(fail=True), object()))],
            object(),
            interactive=False,
        )


@dataclass(frozen=True)
class _BinarySpec:
    from_ref: TensorRef
    to_ref: TensorRef


class _NamedNoopTransform(BaseTransform):
    def __init__(self, name: str):
        self.name = name

    def compile(self, payload: dict, default_model: str | None) -> object:
        del payload, default_model
        return object()

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key=None,
            default_mode="default",
            common_required=set(),
            common_allowed=set(),
        )

    def apply(self, spec: object, provider: object) -> TransformResult:
        del spec, provider
        return TransformResult(name=self.name, count=1)

    def _infer_output_model(self, spec: object) -> str:
        del spec
        return "m"


class _Provider:
    def __init__(self) -> None:
        self._models = {"m": {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}}

    def get_state_dict(self, model: str) -> dict[str, torch.Tensor]:
        return self._models[model]


def test_execute_transform_pairs_preview_emits_transform_and_session_summaries() -> None:
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    set_runtime_flag("preview", True)
    provider = _Provider()
    lines: list[str] = []
    pairs = [
        (
            {"copy": {"from": "m::a", "to": "m::c"}},
            CompiledTransform(
                _NamedNoopTransform("copy"),
                _BinarySpec(
                    from_ref=TensorRef(model="m", expr="a"),
                    to_ref=TensorRef(model="m", expr="c"),
                ),
            ),
        ),
        (
            {"move": {"from": "m::b", "to": "m::d"}},
            CompiledTransform(
                _NamedNoopTransform("move"),
                _BinarySpec(
                    from_ref=TensorRef(model="m", expr="b"),
                    to_ref=TensorRef(model="m", expr="d"),
                ),
            ),
        ),
    ]
    with use_output_emitter(lines.append):
        should_continue, executed = _execute_transform_pairs(pairs, provider, interactive=False)
    assert should_continue is True
    assert executed == [pair[0] for pair in pairs]
    assert any("preview 1/2 copy: created[1] m::c" in line for line in lines)
    assert any("preview 2/2 move: created[1] m::d | deleted[1] m::b" in line for line in lines)
    assert any("preview session: changed[0], created[2], deleted[1]" in line for line in lines)
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)


def test_preview_bucket_renders_all_refs_without_abbreviation() -> None:
    from brainsurgery.engine.execution import _format_preview_bucket

    refs = {"m::z", "m::a", "m::x", "m::b"}
    rendered = _format_preview_bucket("changed", refs)
    assert rendered == "changed[4] m::a, m::b, m::x, m::z"


def test_execute_transform_pairs_preview_still_applies_set_transform() -> None:
    class _CountingSetTransform(_NamedNoopTransform):
        def __init__(self) -> None:
            super().__init__("set")
            self.calls = 0

        def apply(self, spec: object, provider: object) -> TransformResult:
            del spec, provider
            self.calls += 1
            return TransformResult(name=self.name, count=1)

    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    set_runtime_flag("preview", True)
    transform = _CountingSetTransform()
    pairs = [({"set": {"preview": True}}, CompiledTransform(transform, object()))]
    should_continue, executed = _execute_transform_pairs(pairs, object(), interactive=False)
    assert should_continue is True
    assert executed == [{"set": {"preview": True}}]
    assert transform.calls == 1
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)


def test_execute_transform_pairs_preview_interactive_no_go_skips_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CountingTransform(_NamedNoopTransform):
        def __init__(self) -> None:
            super().__init__("copy")
            self.calls = 0

        def apply(self, spec: object, provider: object) -> TransformResult:
            del spec, provider
            self.calls += 1
            return TransformResult(name=self.name, count=1)

    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    set_runtime_flag("preview", True)
    transform = _CountingTransform()
    monkeypatch.setattr("builtins.input", lambda _prompt: "no-go")
    pairs = [
        (
            {"copy": {"from": "m::a", "to": "m::c"}},
            CompiledTransform(
                transform,
                _BinarySpec(
                    from_ref=TensorRef(model="m", expr="a"),
                    to_ref=TensorRef(model="m", expr="c"),
                ),
            ),
        )
    ]
    should_continue, executed = _execute_transform_pairs(pairs, _Provider(), interactive=True)
    assert should_continue is True
    assert executed == []
    assert transform.calls == 0
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)


def test_execute_transform_pairs_preview_interactive_go_applies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CountingTransform(_NamedNoopTransform):
        def __init__(self) -> None:
            super().__init__("copy")
            self.calls = 0

        def apply(self, spec: object, provider: object) -> TransformResult:
            del spec, provider
            self.calls += 1
            return TransformResult(name=self.name, count=1)

    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    set_runtime_flag("preview", True)
    transform = _CountingTransform()
    monkeypatch.setattr("builtins.input", lambda _prompt: "go")
    pairs = [
        (
            {"copy": {"from": "m::a", "to": "m::c"}},
            CompiledTransform(
                transform,
                _BinarySpec(
                    from_ref=TensorRef(model="m", expr="a"),
                    to_ref=TensorRef(model="m", expr="c"),
                ),
            ),
        )
    ]
    should_continue, executed = _execute_transform_pairs(pairs, _Provider(), interactive=True)
    assert should_continue is True
    assert executed == [{"copy": {"from": "m::a", "to": "m::c"}}]
    assert transform.calls == 1
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)


def test_execute_transform_pairs_preview_non_interactive_with_dry_run_skips_apply() -> None:
    class _CountingTransform(_NamedNoopTransform):
        def __init__(self) -> None:
            super().__init__("copy")
            self.calls = 0

        def apply(self, spec: object, provider: object) -> TransformResult:
            del spec, provider
            self.calls += 1
            return TransformResult(name=self.name, count=1)

    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    set_runtime_flag("preview", True)
    set_runtime_flag("dry_run", True)
    transform = _CountingTransform()
    lines: list[str] = []
    pairs = [
        (
            {"copy": {"from": "m::a", "to": "m::c"}},
            CompiledTransform(
                transform,
                _BinarySpec(
                    from_ref=TensorRef(model="m", expr="a"),
                    to_ref=TensorRef(model="m", expr="c"),
                ),
            ),
        )
    ]
    with use_output_emitter(lines.append):
        should_continue, executed = _execute_transform_pairs(pairs, _Provider(), interactive=False)
    assert should_continue is True
    assert executed == []
    assert transform.calls == 0
    assert any("dry-run+preview, apply skipped" in line for line in lines)
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
