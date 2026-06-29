import pytest

import brainsurgery.transforms.prefixes as prefixes_module
from brainsurgery.transforms.prefixes import PrefixesSpec, PrefixesTransform, PrefixesTransformError


def test_prefixes_compile_accepts_none_and_empty_mapping() -> None:
    transform = PrefixesTransform()
    assert isinstance(transform.compile(None, default_model=None), PrefixesSpec)
    assert isinstance(transform.compile({}, default_model=None), PrefixesSpec)


def test_prefixes_compile_add_mode() -> None:
    spec = PrefixesTransform().compile(
        {"mode": "add", "alias": "scratch"},
        default_model=None,
    )
    assert spec.mode == "add"
    assert spec.alias == "scratch"


def test_prefixes_compile_rejects_invalid_mode() -> None:
    try:
        PrefixesTransform().compile({"mode": "explode"}, default_model=None)
    except PrefixesTransformError as exc:
        assert "must be one of" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected prefixes mode error")


def test_prefixes_list_aliases_from_provider_state() -> None:
    class _Provider:
        model_paths = {"base": object()}
        state_dicts = {"scratch": object()}

    assert prefixes_module._list_aliases(_Provider()) == {"base", "scratch"}


def test_prefixes_compile_and_apply_additional_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    transform = PrefixesTransform()
    assert transform.compile({"mode": "list"}, default_model=None) == PrefixesSpec(mode="list")

    with pytest.raises(PrefixesTransformError, match="non-empty string"):
        transform.compile({"mode": ""}, default_model=None)
    with pytest.raises(PrefixesTransformError, match="unknown keys"):
        transform.compile({"mode": "list", "alias": "x"}, default_model=None)

    class _Provider:
        model_paths = {}
        state_dicts = {}

    lines: list[str] = []
    monkeypatch.setattr(prefixes_module, "emit_line", lines.append)
    result = transform.apply(PrefixesSpec(mode="list"), _Provider())  # type: ignore[arg-type]
    assert result.count == 0
    assert lines == ["No model prefixes available."]

    with pytest.raises(PrefixesTransformError, match="unsupported prefixes mode"):
        transform.apply(PrefixesSpec(mode="list").__class__(mode="weird"), _Provider())  # type: ignore[arg-type]

    assert transform.contributes_output_model(PrefixesSpec(mode="add", alias="x")) is True
    assert (
        transform.contributes_output_model(
            PrefixesSpec(mode="rename", source_alias="a", dest_alias="b")
        )
        is True
    )
    assert transform.completion_key_candidates("prefixes: { ", "") == ["mode: "]
    assert transform.completion_key_candidates("prefixes: { mode: list, ", "") == ["}"]
    assert transform.completion_key_candidates("prefixes: { mode: ???, ", "") == ["}"]
    assert transform.completion_key_candidates("prefixes: { mode: add, ", "") == ["alias: "]
    assert transform.completion_value_candidates("unknown", "", []) is None

    assert prefixes_module._prefixes_mode("prefixes: {}") is None


def test_prefixes_alias_edit_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Provider:
        def __init__(self) -> None:
            self.model_paths = {"base": object()}
            self.state_dicts = {"scratch": {}}

    provider = _Provider()
    with pytest.raises(PrefixesTransformError, match="already exists"):
        prefixes_module._create_empty_alias(provider, "base")

    monkeypatch.setattr(
        prefixes_module,
        "get_or_create_alias_state_dict",
        lambda *args, **kwargs: {},
    )
    prefixes_module._create_empty_alias(provider, "new_alias")

    monkeypatch.setattr(
        prefixes_module,
        "get_or_create_alias_state_dict",
        lambda *args, **kwargs: (_ for _ in ()).throw(PrefixesTransformError("x")),
    )
    monkeypatch.setattr(prefixes_module, "_iter_alias_mappings", lambda p: [])
    with pytest.raises(PrefixesTransformError, match="supports editable aliases"):
        prefixes_module._create_empty_alias(provider, "x")

    with pytest.raises(PrefixesTransformError, match="must differ"):
        prefixes_module._rename_alias(provider, source="base", dest="base")
    with pytest.raises(PrefixesTransformError, match="already exists"):
        prefixes_module._rename_alias(provider, source="base", dest="scratch")


def test_prefixes_completion_unknown_mode_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    transform = PrefixesTransform()
    monkeypatch.setattr(prefixes_module, "_prefixes_mode", lambda before_cursor: "mystery")
    assert transform.completion_key_candidates("prefixes: { ", "z") == []
