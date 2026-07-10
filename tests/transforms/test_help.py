import builtins
from importlib import import_module
from types import SimpleNamespace

import pytest

_module = import_module("brainsurgery.transforms.help")
HelpSpec = _module.HelpSpec
HelpTransform = _module.HelpTransform
HelpTransformError = _module.HelpTransformError
TransformPayloadSchema = _module.TransformPayloadSchema
TransformError = _module.TransformError


class _SchemaOnlyTransform:
    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key=None,
            default_mode="default",
            common_required=set(),
            common_allowed=set(),
        )


def test_help_compile_rejects_multi_key_mapping() -> None:
    try:
        HelpTransform().compile({"a": "x", "b": "y"}, default_model=None)
    except HelpTransformError as exc:
        assert "exactly one key" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected help mapping payload validation error")


def test_help_compile_string_payload() -> None:
    spec = HelpTransform().compile("copy", default_model=None)
    assert spec.command == "copy"
    assert spec.subcommand is None


def test_help_compile_mapping_with_subcommand() -> None:
    spec = HelpTransform().compile({"assert": "equal"}, default_model=None)
    assert spec.command == "assert"
    assert spec.subcommand == "equal"


def test_help_apply_rejects_subcommand_for_non_assert_command() -> None:
    transform = HelpTransform()
    with pytest.raises(HelpTransformError, match="does not support subcommand"):
        transform.apply(HelpSpec(command="copy", subcommand="from"), provider=object())


def test_help_print_command_help_emits_unavailable_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines: list[str] = []
    monkeypatch.setattr(_module, "get_transform", lambda _: _SchemaOnlyTransform())
    monkeypatch.setattr(_module, "emit_line", lines.append)

    HelpTransform()._print_command_help("noop")

    output = "\n".join(lines)
    assert "Help for noop" in output
    assert "Command: noop" in output
    assert "Required keys: none" in output
    assert "Optional keys: none" in output
    assert "All allowed keys: none" in output


def test_help_print_assert_expr_help_emits_required_optional_and_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines: list[str] = []
    monkeypatch.setattr(
        _module,
        "get_assert_expr_help",
        lambda _: SimpleNamespace(
            name="equal",
            payload_kind="mapping",
            description="desc",
            required_keys={"left"},
            allowed_keys={"left", "right"},
        ),
    )
    monkeypatch.setattr(_module, "emit_line", lines.append)

    HelpTransform()._print_assert_expr_help("equal")

    output = "\n".join(lines)
    assert "Help for assert.equal" in output
    assert "Assert expression: equal" in output
    assert "Payload: mapping" in output
    assert "Required keys:" in output
    assert "- left" in output
    assert "Optional keys:" in output
    assert "- right" in output


def test_help_compile_accepts_none_and_empty_mapping() -> None:
    transform = HelpTransform()
    assert transform.compile(None, default_model=None) == HelpSpec(command=None, subcommand=None)
    assert transform.compile({}, default_model=None) == HelpSpec(command=None, subcommand=None)


def test_help_compile_rejects_invalid_payload_types_and_subcommand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transform = HelpTransform()

    with pytest.raises(
        HelpTransformError, match="must be empty, a string, or a single-key mapping"
    ):
        transform.compile(123, default_model=None)

    with pytest.raises(HelpTransformError, match="subcommand must be a non-empty string"):
        transform.compile({"assert": ""}, default_model=None)

    with pytest.raises(HelpTransformError, match="help command must be a non-empty string"):
        transform.compile({1: "x"}, default_model=None)  # type: ignore[dict-item]


def test_help_infer_output_model_raises() -> None:
    with pytest.raises(HelpTransformError, match="does not infer an output model"):
        HelpTransform()._infer_output_model(HelpSpec())


def test_help_completion_helpers_cover_fallback_paths() -> None:
    transform = HelpTransform()
    assert transform.completion_key_candidates("", "z") == []
    assert transform.completion_value_candidates("unknown", "", []) is None
    assert transform.completion_committed_next_candidates("unknown") is None


def test_help_print_assert_help_without_assert_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr(
        _module,
        "get_transform",
        lambda name: (_ for _ in ()).throw(TransformError("missing assert")),
    )
    monkeypatch.setattr(_module, "get_assert_expr_names", lambda: ["equal"])
    monkeypatch.setattr(
        _module,
        "get_assert_expr_help",
        lambda _: SimpleNamespace(description=None),
    )
    monkeypatch.setattr(_module, "emit_line", lines.append)

    HelpTransform()._print_assert_help()
    output = "\n".join(lines)
    assert "Help for assert" in output
    assert "Command: assert" in output
    assert "equal" in output


def test_help_compile_rejects_blank_string_and_accepts_none_subpayload() -> None:
    transform = HelpTransform()
    with pytest.raises(HelpTransformError, match="non-empty string"):
        transform.compile("   ", default_model=None)
    assert transform.compile({"assert": None}, default_model=None) == HelpSpec(
        command="assert",
        subcommand=None,
    )


def test_help_apply_dispatch_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    transform = HelpTransform()
    monkeypatch.setattr(transform, "_print_all_commands", lambda: calls.append("all"))
    monkeypatch.setattr(transform, "_print_assert_help", lambda: calls.append("assert"))
    monkeypatch.setattr(
        transform, "_print_assert_expr_help", lambda name: calls.append(f"expr:{name}")
    )
    monkeypatch.setattr(transform, "_print_command_help", lambda name: calls.append(f"cmd:{name}"))

    transform.apply(HelpSpec(command=None), provider=object())
    transform.apply(HelpSpec(command="assert"), provider=object())
    transform.apply(HelpSpec(command="assert", subcommand="equal"), provider=object())
    transform.apply(HelpSpec(command="copy"), provider=object())

    assert calls == ["all", "assert", "expr:equal", "cmd:copy"]


def test_help_completion_helpers_more_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_module, "list_transforms", lambda: ["copy", "help", "exit"])
    transform = HelpTransform()
    assert transform.completion_payload_start_candidates("") == ["copy", "help", "exit", "{ "]
    assert transform.completion_payload_start_candidates("h") == ["help"]
    assert transform.completion_value_candidates("help", "c", []) == ["copy"]


def test_help_print_command_help_unknown_and_with_help_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(HelpTransformError, match="unknown command"):
        monkeypatch.setattr(
            _module, "get_transform", lambda _: (_ for _ in ()).throw(TransformError("x"))
        )
        HelpTransform()._print_command_help("missing")

    lines: list[str] = []
    monkeypatch.setattr(
        _module,
        "get_transform",
        lambda _: SimpleNamespace(
            help_text="help text",
            payload_schema=lambda: TransformPayloadSchema(
                mode_key=None,
                default_mode="default",
                common_required=set(),
                common_allowed=set(),
            ),
        ),
    )
    monkeypatch.setattr(_module, "emit_line", lines.append)
    HelpTransform()._print_command_help("copy")
    output = "\n".join(lines)
    assert "Help for copy" in output
    assert "help text" in output
    assert "Required keys: none" in output
    assert "Optional keys: none" in output
    assert "All allowed keys: none" in output


def test_help_emit_panel_fallback_and_emit_key_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr(_module, "emit_line", lines.append)

    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name in {"rich.console", "rich.panel"}:
            raise ImportError("no rich")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    transform = HelpTransform()
    transform._emit_help_panel("Fallback title", ["line 1", "line 2"])
    transform._emit_key_metadata(required_keys={"a"}, allowed_keys={"a", "b"})

    output = "\n".join(lines)
    assert "Fallback title" in output
    assert "line 1" in output
    assert "Required keys:" in output
    assert "- a" in output
