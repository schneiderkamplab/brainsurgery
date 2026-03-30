from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

import brainsurgery.cli.complete as complete_module
import brainsurgery.cli.payload_scan as payload_scan_module
from brainsurgery.cli.complete import (
    _is_transform_payload_start,
)
from brainsurgery.cli.interactive import (
    _collect_completion_candidates,
    _collect_payload_candidates,
    _infer_active_transform,
    _is_top_level_completion_position,
    _list_model_aliases,
    _match_payload_candidates,
)
from brainsurgery.core import get_transform, list_transforms


@dataclass
class _MiniProvider:
    model_paths: dict[str, Path]
    state_dicts: dict[str, dict[str, object]]

    def list_model_aliases(self) -> set[str]:
        return set(self.state_dicts)


SINGLE_ALIAS_PROVIDER = _MiniProvider(
    model_paths={"model": Path("/tmp/model.safetensors")},
    state_dicts={
        "model": {
            "ln_f.weight": object(),
            "ln_f.bias": object(),
            "h.0.attn.c_attn.weight": object(),
            "h.0.attn.c_proj.weight": object(),
            "mlp.c_fc.weight": object(),
        }
    },
)

MULTI_ALIAS_PROVIDER = _MiniProvider(
    model_paths={
        "base": Path("/tmp/base.safetensors"),
        "scratch": Path("/tmp/scratch.safetensors"),
    },
    state_dicts={
        "base": {
            "ln_f.weight": object(),
            "ln_f.bias": object(),
            "h.0.attn.c_attn.weight": object(),
        },
        "scratch": {
            "ln_f.weight": object(),
            "new.weight": object(),
            "h.1.attn.c_proj.weight": object(),
        },
    },
)

_SPECIAL_TRANSFORMS = {"assert", "diff", "execute", "exit", "help", "prefixes", "set"}
_PREFERRED_KEYS = {
    "add": "from_a",
    "add_": "from",
    "assign": "from",
    "cast": "from",
    "cast_": "target",
    "clamp": "from",
    "clamp_": "target",
    "concat": "from",
    "copy": "from",
    "delete": "target",
    "dump": "target",
    "fill": "from",
    "fill_": "target",
    "infer": "input_ids",
    "load": "path",
    "matmul": "from_a",
    "move": "from",
    "multiply": "from_a",
    "ones": "target",
    "permute": "from",
    "phlora": "target",
    "phlora_": "target",
    "rand": "target",
    "reshape": "from",
    "reshape_": "target",
    "save": "target",
    "scale": "from",
    "scale_": "target",
    "split": "from",
    "subtract": "from_a",
    "subtract_": "from",
    "zeroes": "target",
}

_REFERENCE_TRANSFORMS = [
    name
    for name in list_transforms()
    if _PREFERRED_KEYS.get(name) in set(get_transform(name).completion_reference_keys())
]


def _top_level_candidate(transform_name: str) -> str:
    candidates = _collect_completion_candidates(None)
    with_payload = f"{transform_name}: "
    if with_payload in candidates:
        return with_payload
    return transform_name


def _shortest_unique_prefix(target: str, candidates: list[str]) -> str:
    for index in range(1, len(target) + 1):
        prefix = target[:index]
        matches = [candidate for candidate in candidates if candidate.startswith(prefix)]
        if matches == [target]:
            return prefix
    return target


def _shortest_unique_key_prefix(target_key_candidate: str, candidates: list[str]) -> str:
    bare_key = target_key_candidate[:-2]
    for index in range(1, len(bare_key) + 1):
        prefix = bare_key[:index]
        matches = [candidate for candidate in candidates if candidate.startswith(prefix)]
        if matches == [target_key_candidate]:
            return prefix
    return bare_key


def _current_token_bounds(buffer: str) -> tuple[int, int]:
    last_delim = max(buffer.rfind(" "), buffer.rfind("\t"), buffer.rfind("\n"))
    begidx = last_delim + 1
    endidx = len(buffer)
    return begidx, endidx


def _completion_matches(buffer: str, provider: object | None) -> list[str]:
    begidx, endidx = _current_token_bounds(buffer)
    text = buffer[begidx:endidx]

    if _is_top_level_completion_position(buffer, begidx):
        candidates = _collect_completion_candidates(provider)
        return [candidate for candidate in candidates if candidate.startswith(text)]

    active_transform = _infer_active_transform([], buffer)
    payload_candidates = _collect_payload_candidates(
        active_transform=active_transform,
        state_dict_provider=provider,
    )
    return _match_payload_candidates(
        text=text,
        line_buffer=buffer,
        begidx=begidx,
        endidx=endidx,
        payload_candidates=payload_candidates,
        active_transform=active_transform,
        model_aliases=sorted(_list_model_aliases(provider)),
    )


def _complete_unique(buffer: str, provider: object | None) -> str:
    matches = _completion_matches(buffer, provider)
    assert matches, f"expected completion matches for {buffer!r}"
    assert len(matches) == 1, f"expected unique completion for {buffer!r}, got {matches!r}"
    begidx, endidx = _current_token_bounds(buffer)
    return f"{buffer[:begidx]}{matches[0]}{buffer[endidx:]}"


def _transform_reference_keys(transform_name: str) -> list[str]:
    transform = get_transform(transform_name)
    return transform.completion_reference_keys()


def _next_reference_key(transform_name: str, current_key: str) -> str | None:
    ordered = _transform_reference_keys(transform_name)
    if current_key not in ordered:
        return None
    current_index = ordered.index(current_key)
    if current_index + 1 >= len(ordered):
        return None
    return ordered[current_index + 1]


def _first_base_tensor_match(matches: list[str]) -> str:
    for match in matches:
        if match.startswith("base::") and match != "base::":
            return match
    raise AssertionError(f"expected a base-qualified tensor match, got {matches!r}")


@pytest.mark.parametrize("transform_name", list_transforms(), ids=list_transforms())
def test_top_level_unique_completion_for_each_transform(transform_name: str) -> None:
    candidate = _top_level_candidate(transform_name)
    prefix = _shortest_unique_prefix(candidate, _collect_completion_candidates(None))
    assert _complete_unique(prefix, None) == candidate


@pytest.mark.parametrize(
    "transform_name",
    [name for name in list_transforms() if name not in _SPECIAL_TRANSFORMS],
    ids=[name for name in list_transforms() if name not in _SPECIAL_TRANSFORMS],
)
def test_single_alias_typical_completion_sequence_for_each_non_special_transform(
    transform_name: str,
) -> None:
    candidate = _top_level_candidate(transform_name)
    assert (
        _complete_unique(
            _shortest_unique_prefix(candidate, _collect_completion_candidates(None)),
            SINGLE_ALIAS_PROVIDER,
        )
        == candidate
    )

    assert _complete_unique(candidate, SINGLE_ALIAS_PROVIDER) == f"{candidate}{{ "

    preferred_key = _PREFERRED_KEYS[transform_name]
    mapping_start_matches = _completion_matches(f"{candidate}{{ ", SINGLE_ALIAS_PROVIDER)
    key_candidate = f"{preferred_key}: "
    assert key_candidate in mapping_start_matches

    key_prefix = _shortest_unique_key_prefix(key_candidate, mapping_start_matches)
    key_prefix_matches = [match for match in mapping_start_matches if match.startswith(key_prefix)]
    if key_prefix_matches == [key_candidate]:
        assert (
            _complete_unique(
                f"{candidate}{{ {key_prefix}",
                SINGLE_ALIAS_PROVIDER,
            )
            == f"{candidate}{{ {key_candidate}"
        )
    else:
        assert key_candidate in key_prefix_matches

    if preferred_key not in set(_transform_reference_keys(transform_name)):
        return

    reference_matches = _completion_matches(
        f"{candidate}{{ {key_candidate}",
        SINGLE_ALIAS_PROVIDER,
    )
    assert "model::" in reference_matches
    assert any(match.startswith("model::") and match != "model::" for match in reference_matches)
    assert any(
        "::" not in match and not match.endswith(": ") and match not in {"{ ", "}", "[ ", "]", ", "}
        for match in reference_matches
    )

    next_reference_key = _next_reference_key(transform_name, preferred_key)
    if next_reference_key is None:
        return

    continuation_matches = _completion_matches(
        f"{candidate}{{ {key_candidate}model::ln_f.weight",
        SINGLE_ALIAS_PROVIDER,
    )
    assert f"model::ln_f.weight, {next_reference_key}: " in continuation_matches


@pytest.mark.parametrize(
    "transform_name",
    _REFERENCE_TRANSFORMS,
    ids=_REFERENCE_TRANSFORMS,
)
def test_multi_alias_reference_completion_sequence_for_reference_transforms(
    transform_name: str,
) -> None:
    candidate = _top_level_candidate(transform_name)
    preferred_key = _PREFERRED_KEYS[transform_name]
    key_candidate = f"{preferred_key}: "

    alias_selection_matches = _completion_matches(
        f"{candidate}{{ {key_candidate}",
        MULTI_ALIAS_PROVIDER,
    )
    assert alias_selection_matches == ["base::", "scratch::"]

    base_matches = _completion_matches(
        f"{candidate}{{ {key_candidate}base::",
        MULTI_ALIAS_PROVIDER,
    )
    assert "base::" in base_matches
    assert any(match.startswith("base::") and match != "base::" for match in base_matches)
    assert all(not match.startswith("scratch::") for match in base_matches)

    next_reference_key = _next_reference_key(transform_name, preferred_key)
    if next_reference_key is None:
        return

    full_ref = _first_base_tensor_match(base_matches)
    continuation_matches = _completion_matches(
        f"{candidate}{{ {key_candidate}{full_ref}",
        MULTI_ALIAS_PROVIDER,
    )
    assert f"{full_ref}, {next_reference_key}: " in continuation_matches


def test_help_completion_sequence() -> None:
    candidate = _top_level_candidate("help")
    assert (
        _complete_unique(
            _shortest_unique_prefix(candidate, _collect_completion_candidates(None)),
            SINGLE_ALIAS_PROVIDER,
        )
        == candidate
    )

    payload_start_matches = _completion_matches(candidate, SINGLE_ALIAS_PROVIDER)
    assert "{ " in payload_start_matches
    assert "copy" in payload_start_matches
    assert "exit" in payload_start_matches

    mapping_start_matches = _completion_matches("help: { ", SINGLE_ALIAS_PROVIDER)
    assert mapping_start_matches == ["assert: "]

    assert _complete_unique("help: { assert: eq", SINGLE_ALIAS_PROVIDER) == "help: { assert: equal"


def test_assert_completion_sequence() -> None:
    candidate = _top_level_candidate("assert")
    assert (
        _complete_unique(
            _shortest_unique_prefix(candidate, _collect_completion_candidates(None)),
            SINGLE_ALIAS_PROVIDER,
        )
        == candidate
    )

    payload_start_matches = _completion_matches(candidate, SINGLE_ALIAS_PROVIDER)
    assert "{ " in payload_start_matches
    assert "equal" in payload_start_matches
    assert "exists" in payload_start_matches

    mapping_start_matches = _completion_matches("assert: { ", SINGLE_ALIAS_PROVIDER)
    assert "equal: " in mapping_start_matches
    assert "exists: " in mapping_start_matches

    assert "equal: " in mapping_start_matches
    assert "exists: " in mapping_start_matches


def test_prefixes_completion_sequence() -> None:
    candidate = _top_level_candidate("prefixes")
    assert (
        _complete_unique(
            _shortest_unique_prefix(candidate, _collect_completion_candidates(None)),
            MULTI_ALIAS_PROVIDER,
        )
        == candidate
    )

    assert _complete_unique("prefixes: { m", MULTI_ALIAS_PROVIDER) == "prefixes: { mode: "

    mode_matches = _completion_matches("prefixes: { mode: r", MULTI_ALIAS_PROVIDER)
    assert mode_matches == ["remove", "rename"]

    rename_value = _complete_unique("prefixes: { mode: ren", MULTI_ALIAS_PROVIDER)
    assert rename_value == "prefixes: { mode: rename"

    rename_key_matches = _completion_matches(
        "prefixes: { mode: rename, ",
        MULTI_ALIAS_PROVIDER,
    )
    assert rename_key_matches == ["from: ", "to: "]

    alias_matches = _completion_matches(
        "prefixes: { mode: rename, from: ",
        MULTI_ALIAS_PROVIDER,
    )
    assert alias_matches == ["base", "scratch"]


def test_diff_completion_sequence() -> None:
    candidate = _top_level_candidate("diff")
    assert (
        _complete_unique(
            _shortest_unique_prefix(candidate, _collect_completion_candidates(None)),
            MULTI_ALIAS_PROVIDER,
        )
        == candidate
    )

    assert _complete_unique(candidate, MULTI_ALIAS_PROVIDER) == f"{candidate}{{ "

    mapping_start_matches = _completion_matches(f"{candidate}{{ ", MULTI_ALIAS_PROVIDER)
    assert "mode: " in mapping_start_matches
    assert "left: " in mapping_start_matches

    mode_matches = _completion_matches(f"{candidate}{{ mode: a", MULTI_ALIAS_PROVIDER)
    assert mode_matches == ["aliases"]

    ref_matches = _completion_matches(f"{candidate}{{ left: ", MULTI_ALIAS_PROVIDER)
    assert ref_matches == ["base::", "scratch::"]

    alias_matches = _completion_matches(
        f"{candidate}{{ mode: aliases, left_alias: ",
        MULTI_ALIAS_PROVIDER,
    )
    assert alias_matches == ["base", "scratch"]


def test_exit_completion_sequence() -> None:
    candidate = _top_level_candidate("exit")
    assert (
        _complete_unique(
            _shortest_unique_prefix(candidate, _collect_completion_candidates(None)),
            None,
        )
        == candidate
    )


def test_list_model_aliases_handles_provider_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        complete_module,
        "list_model_aliases_from_provider",
        lambda provider: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert _list_model_aliases(object()) == set()


def test_list_loaded_tensor_names_handles_provider_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        complete_module,
        "list_loaded_tensor_names_from_provider",
        lambda provider: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert complete_module.list_loaded_tensor_names(object()) == {}


def test_extract_transform_name_handles_yaml_list_prefix_and_unknown_command() -> None:
    assert complete_module._extract_transform_name("- copy: { from: a, to: b }") == "copy"
    assert complete_module._extract_transform_name("unknown: {}") is None
    assert complete_module._extract_transform_name(": {}") is None


def test_is_top_level_completion_position_rejects_out_of_bounds_begidx() -> None:
    assert _is_top_level_completion_position("copy", 999) is False


def test_is_transform_payload_start_clamps_endidx_bounds() -> None:
    assert (
        _is_transform_payload_start(
            line_buffer="copy: ",
            begidx=0,
            endidx=999,
            active_transform="copy",
        )
        is True
    )
    assert (
        _is_transform_payload_start(
            line_buffer="copy: ",
            begidx=0,
            endidx=-2,
            active_transform="copy",
        )
        is False
    )


class _FakeReadline:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def parse_and_bind(self, command: str) -> None:
        self.commands.append(command)


def test_configure_readline_completion_bindings_accepts_none() -> None:
    complete_module._configure_readline_completion_bindings(None)


def test_configure_readline_completion_bindings_ignores_parse_failures() -> None:
    class _BrokenReadline:
        def parse_and_bind(self, command: str) -> None:
            del command
            raise RuntimeError("broken")

    complete_module._configure_readline_completion_bindings(_BrokenReadline())


def test_infer_active_transform_returns_none_when_unknown() -> None:
    assert _infer_active_transform(["something without colon"], "also invalid") is None


def test_collect_payload_candidates_handles_get_transform_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        complete_module,
        "get_transform",
        lambda name: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    candidates = _collect_payload_candidates(active_transform="copy", state_dict_provider=None)
    assert "{ " in candidates


def test_collect_payload_candidates_uses_mode_schema_hints_for_load() -> None:
    candidates = _collect_payload_candidates(
        active_transform="load",
        state_dict_provider=None,
        line_buffer="load: { backend: axon, ",
    )
    assert "weights: " in candidates
    assert "to: " not in candidates


def test_match_payload_candidates_covers_structural_prefix_and_any_context() -> None:
    matches = _match_payload_candidates(
        text="{",
        line_buffer="copy: {",
        begidx=len("copy: {"),
        payload_candidates=["{ ", "}", "from: "],
    )
    assert matches == ["from: "]

    any_ctx = _match_payload_candidates(
        text="",
        line_buffer="copy",
        begidx=len("copy"),
        payload_candidates=["x", "y"],
    )
    assert any_ctx == ["x", "y"]


def test_match_payload_candidates_handles_negative_endidx_and_close_brace_prefix() -> None:
    matches = _match_payload_candidates(
        text="}",
        line_buffer="copy: {",
        begidx=0,
        endidx=-1,
        payload_candidates=["{ ", "}", "from: "],
    )
    assert matches == ["from: "]


def test_match_payload_candidates_colon_prefix_key_context_spacing() -> None:
    matches = _match_payload_candidates(
        text="f",
        line_buffer="copy: { from",
        begidx=len("copy: { "),
        payload_candidates=["from: ", "to: "],
        active_transform="copy",
    )
    assert matches == ["from: "]

    comma_no_space = _match_payload_candidates(
        text="f",
        line_buffer="copy: { x,",
        begidx=len("copy: { x,"),
        payload_candidates=["from: ", "to: "],
    )
    assert comma_no_space == [" from: "]


def test_match_payload_candidates_with_custom_transform_key_candidates() -> None:
    class _T:
        def completion_key_candidates(self, before_cursor: str, prefix_text: str):
            del before_cursor, prefix_text
            return ["alpha: ", "}"]

        def completion_value_candidates(
            self, value_key: str | None, prefix_text: str, model_aliases: list[str]
        ):
            del value_key, prefix_text, model_aliases
            return None

        def completion_committed_next_candidates(self, value_key: str | None):
            del value_key
            return None

        def completion_reference_keys(self):
            return []

        def completion_payload_start_candidates(self, prefix_text: str):
            del prefix_text
            return None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(complete_module, "get_transform", lambda name: _T())
    try:
        start_with_brace = _match_payload_candidates(
            text="{",
            line_buffer="copy: {",
            begidx=len("copy: "),
            endidx=len("copy: {"),
            payload_candidates=["x"],
            active_transform="copy",
        )
        assert start_with_brace == ["{ alpha: "]

        after_comma = _match_payload_candidates(
            text=",",
            line_buffer="copy: { x,",
            begidx=len("copy: { x"),
            endidx=len("copy: { x,"),
            payload_candidates=["x"],
            active_transform="copy",
        )
        assert "}" in after_comma
    finally:
        monkeypatch.undo()


def test_match_payload_candidates_value_transform_override_and_reference_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _T:
        def completion_key_candidates(self, before_cursor: str, prefix_text: str):
            del before_cursor, prefix_text
            return None

        def completion_value_candidates(
            self, value_key: str | None, prefix_text: str, model_aliases: list[str]
        ):
            del model_aliases
            if value_key == "from":
                return []
            if value_key == "mode":
                return ["alpha", "beta"]
            return None

        def completion_committed_next_candidates(self, value_key: str | None):
            if value_key == "mode":
                return [", from: ", "}"]
            return None

        def completion_reference_keys(self):
            return ["from", "to"]

        def completion_payload_start_candidates(self, prefix_text: str):
            del prefix_text
            return None

    monkeypatch.setattr(complete_module, "get_transform", lambda name: _T())

    empty_override = _match_payload_candidates(
        text="x::",
        line_buffer="copy: { from: x::",
        begidx=len("copy: { from: x::"),
        payload_candidates=["x::name"],
        active_transform="copy",
    )
    assert empty_override == []

    committed_filtered = _match_payload_candidates(
        text="}",
        line_buffer="copy: { mode: alpha ",
        begidx=len("copy: { mode: alpha "),
        payload_candidates=["from: ", "to: "],
        active_transform="copy",
    )
    assert committed_filtered == []

    fallback_values = _match_payload_candidates(
        text="z",
        line_buffer="copy: { other: z",
        begidx=len("copy: { other: z"),
        payload_candidates=["zeta", "other: ", "x::y"],
        active_transform="copy",
    )
    assert "zeta" in fallback_values


def test_match_payload_candidates_additional_uncovered_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        complete_module,
        "get_transform",
        lambda name: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert _match_payload_candidates(
        text="x",
        line_buffer="copy: { from: x",
        begidx=len("copy: { from: x"),
        payload_candidates=["x::y"],
        active_transform="copy",
    ) == ["x::y"]

    class _FlakyTransform:
        def __init__(self) -> None:
            self.calls = 0

        def completion_key_candidates(self, before_cursor: str, prefix_text: str):
            del before_cursor, prefix_text
            return ["from: ", "}"]

        def completion_value_candidates(
            self, value_key: str | None, prefix_text: str, model_aliases: list[str]
        ):
            del value_key, prefix_text, model_aliases
            return None

        def completion_committed_next_candidates(self, value_key: str | None):
            del value_key
            return None

        def completion_reference_keys(self):
            self.calls += 1
            if self.calls == 1:
                return ["from"]
            return ["to"]

        def completion_payload_start_candidates(self, prefix_text: str):
            del prefix_text
            return None

    flaky = _FlakyTransform()
    monkeypatch.setattr(complete_module, "get_transform", lambda name: flaky)
    assert _match_payload_candidates(
        text="",
        line_buffer="copy: { from: ",
        begidx=len("copy: { from: "),
        payload_candidates=["base::", "base::", "scratch::", "scratch::"],
        active_transform="copy",
        model_aliases=["base", "scratch"],
    ) == ["base::", "scratch::"]

    matches = _match_payload_candidates(
        text="base::x",
        line_buffer="copy: { from: base::x",
        begidx=len("copy: { from: base::x"),
        payload_candidates=["base::", "base::x", "from: "],
        active_transform="copy",
    )
    assert matches == ["base::x"]

    assert _match_payload_candidates(
        text="{",
        line_buffer="copy: {",
        begidx=len("copy: "),
        endidx=len("copy: "),
        payload_candidates=["from: "],
        active_transform="copy",
    ) == ["{ "]

    assert _match_payload_candidates(
        text="f",
        line_buffer="copy: { x,",
        begidx=len("copy: { x,"),
        payload_candidates=["from: ", "}"],
        active_transform="copy",
    ) == [" from: ", "}"]

    assert _match_payload_candidates(
        text="",
        line_buffer="copy: { x,",
        begidx=len("copy: { x,"),
        payload_candidates=["from: ", "to: "],
    ) == [" from: ", " to: "]

    assert _match_payload_candidates(
        text="base::",
        line_buffer="copy: { from: base::",
        begidx=len("copy: { from: base::"),
        payload_candidates=["base::x", "from: "],
        active_transform="copy",
    ) == ["base::x"]

    assert _match_payload_candidates(
        text="f",
        line_buffer="copy: { x,",
        begidx=len("copy: { x"),
        endidx=len("copy: { x,"),
        payload_candidates=["from: "],
    ) == [" from: "]

    assert _match_payload_candidates(
        text="f",
        line_buffer="copy: { f",
        begidx=len("copy: { "),
        payload_candidates=["from: "],
    ) == ["from: "]

    assert _match_payload_candidates(
        text="f",
        line_buffer="copy: { f",
        begidx=len("copy: { f"),
        payload_candidates=["from: ", "to: "],
        active_transform=None,
    ) == ["from: "]

    assert (
        _match_payload_candidates(
            text="x",
            line_buffer="copy: { mode: alpha ",
            begidx=len("copy: { mode: alpha "),
            payload_candidates=["from: ", "to: ", "}"],
            active_transform=None,
        )
        == []
    )


def test_match_payload_candidates_hits_remaining_reference_and_key_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ChangingReferenceKeys:
        def __init__(self) -> None:
            self.calls = 0

        def completion_key_candidates(self, before_cursor: str, prefix_text: str):
            del before_cursor, prefix_text
            return None

        def completion_value_candidates(
            self, value_key: str | None, prefix_text: str, model_aliases: list[str]
        ):
            del value_key, prefix_text, model_aliases
            return None

        def completion_committed_next_candidates(self, value_key: str | None):
            del value_key
            return None

        def completion_reference_keys(self):
            self.calls += 1
            if self.calls == 1:
                return ["from"]
            return []

        def completion_payload_start_candidates(self, prefix_text: str):
            del prefix_text
            return None

    monkeypatch.setattr(complete_module, "get_transform", lambda name: _ChangingReferenceKeys())
    assert _match_payload_candidates(
        text="base::x",
        line_buffer="copy: { from: base::x",
        begidx=len("copy: { from: base::x"),
        payload_candidates=["base::", "base::x"],
        active_transform="copy",
        model_aliases=["base"],
    ) == ["base::x"]

    assert _match_payload_candidates(
        text="f",
        line_buffer="copy: {",
        begidx=len("copy: {"),
        endidx=len("copy: {"),
        payload_candidates=["from: ", "to: "],
        active_transform=None,
    ) == ["from: "]


def test_payload_cursor_state_handles_quoted_value_delimiters() -> None:
    assert complete_module._payload_context('copy: from: "a,b:c"') == "value"
    assert complete_module._current_value_key('copy: from: "a,b:c"') == "from"
    assert complete_module._current_value_fragment('copy: from: "a,b:c"') == ' "a,b:c"'


def test_match_payload_candidates_does_not_treat_nested_keys_as_used() -> None:
    matches = _match_payload_candidates(
        text="",
        line_buffer="copy: { meta: { from: a }, ",
        begidx=len("copy: { meta: { from: a }, "),
        payload_candidates=["from: ", "to: "],
    )
    assert matches == ["from: ", "to: "]


def test_load_path_completion_suggests_matching_filesystem_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "model.safetensors").write_text("", encoding="utf-8")
    (tmp_path / "meta.txt").write_text("", encoding="utf-8")

    matches = _match_payload_candidates(
        text="mo",
        line_buffer="load: { path: mo",
        begidx=len("load: { path: "),
        payload_candidates=[],
        active_transform="load",
    )
    assert "model.safetensors" in matches
    assert "models/" in matches
    assert "meta.txt" not in matches


def test_save_path_completion_supports_quoted_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out_dir").mkdir()
    (tmp_path / "out.pt").write_text("", encoding="utf-8")

    matches = _match_payload_candidates(
        text='"o',
        line_buffer='save: { path: "o',
        begidx=len("save: { path: "),
        payload_candidates=[],
        active_transform="save",
    )
    assert '"out_dir/' in matches
    assert '"out.pt' in matches


def test_cursor_helpers_cover_nested_quotes_and_invalid_keys() -> None:
    segment = '\'a:b\' "x\\"y:z" (u:v) [k:l] {m:n}: tail'
    colon_index = payload_scan_module._find_top_level_colon(segment)
    assert colon_index is not None
    assert segment[colon_index] == ":"
    assert payload_scan_module._parse_key_from_segment("from: x") == "from"
    assert payload_scan_module._parse_key_from_segment("1bad: x") is None


def test_split_top_level_segments_handles_quotes_and_closing_brace_boundary() -> None:
    completed, current = payload_scan_module._split_top_level_segments(
        '{ from: "x\\"y,z", note: \'a,b\', paren: (u,v), arr: [k,l] } trailing'
    )
    assert any("from:" in segment for segment in completed)
    assert any("note:" in segment for segment in completed)
    assert any("paren:" in segment for segment in completed)
    assert "arr:" in current


def test_match_payload_candidates_any_context_falls_back_to_prefix_filter() -> None:
    matches = _match_payload_candidates(
        text="ab",
        line_buffer="nonsense",
        begidx=0,
        payload_candidates=["abc", "zzz"],
    )
    assert matches == ["abc"]
