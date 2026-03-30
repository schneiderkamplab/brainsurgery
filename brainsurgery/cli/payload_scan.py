import re
from dataclasses import dataclass

_KEY_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_BARE_SCALAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_TRANSFORM_LINE_RE = re.compile(
    r"^\s*(?:-\s*)?[A-Za-z_][A-Za-z0-9_-]*\s*:(.*)$",
    re.DOTALL,
)


@dataclass(frozen=True)
class _PayloadCursorState:
    context: str
    current_key: str | None
    raw_value_fragment: str | None
    used_keys: set[str]


def _find_top_level_colon(segment: str) -> int | None:
    brace_depth = 0
    bracket_depth = 0
    paren_depth = 0
    in_single = False
    in_double = False
    escaped = False

    for index, ch in enumerate(segment):
        if in_single:
            if ch == "'":
                in_single = False
            continue
        if in_double:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_double = False
            continue
        if ch == "'":
            in_single = True
            continue
        if ch == '"':
            in_double = True
            continue
        if ch == "(":
            paren_depth += 1
            continue
        if ch == ")" and paren_depth > 0:
            paren_depth -= 1
            continue
        if ch == "[":
            bracket_depth += 1
            continue
        if ch == "]" and bracket_depth > 0:
            bracket_depth -= 1
            continue
        if ch == "{":
            brace_depth += 1
            continue
        if ch == "}" and brace_depth > 0:
            brace_depth -= 1
            continue
        if ch == ":" and brace_depth == 0 and bracket_depth == 0 and paren_depth == 0:
            return index
    return None


def _parse_key_from_segment(segment: str) -> str | None:
    colon_index = _find_top_level_colon(segment)
    if colon_index is None:
        return None
    key = segment[:colon_index].strip()
    if not _KEY_IDENT_RE.fullmatch(key):
        return None
    return key


def _split_top_level_segments(payload: str) -> tuple[list[str], str]:
    working = payload
    stripped = payload.lstrip()
    if stripped.startswith("{"):
        brace_index = payload.find("{")
        if brace_index >= 0:
            working = payload[brace_index + 1 :]

    segments: list[str] = []
    start = 0
    brace_depth = 0
    bracket_depth = 0
    paren_depth = 0
    in_single = False
    in_double = False
    escaped = False

    for index, ch in enumerate(working):
        if in_single:
            if ch == "'":
                in_single = False
            continue
        if in_double:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_double = False
            continue
        if ch == "'":
            in_single = True
            continue
        if ch == '"':
            in_double = True
            continue
        if ch == "(":
            paren_depth += 1
            continue
        if ch == ")" and paren_depth > 0:
            paren_depth -= 1
            continue
        if ch == "[":
            bracket_depth += 1
            continue
        if ch == "]" and bracket_depth > 0:
            bracket_depth -= 1
            continue
        if ch == "{":
            brace_depth += 1
            continue
        if ch == "}":
            if brace_depth == 0:
                segments.append(working[start:index])
                return segments[:-1], segments[-1]
            brace_depth -= 1
            continue
        if ch == "," and brace_depth == 0 and bracket_depth == 0 and paren_depth == 0:
            segments.append(working[start:index])
            start = index + 1

    segments.append(working[start:])
    return segments[:-1], segments[-1]


def _payload_cursor_state(before_cursor: str) -> _PayloadCursorState:
    transform_match = _TRANSFORM_LINE_RE.match(before_cursor)
    if transform_match is None:
        context = "key" if not before_cursor.strip() else "any"
        return _PayloadCursorState(
            context=context,
            current_key=None,
            raw_value_fragment=None,
            used_keys=set(),
        )

    payload = transform_match.group(1)
    if not payload.strip():
        return _PayloadCursorState(
            context="value",
            current_key=None,
            raw_value_fragment="",
            used_keys=set(),
        )
    completed_segments, current_segment = _split_top_level_segments(payload)

    used_keys = {
        key
        for key in (_parse_key_from_segment(segment) for segment in completed_segments)
        if key is not None
    }

    current_key = _parse_key_from_segment(current_segment)
    if current_key is not None:
        used_keys.add(current_key)
    colon_index = _find_top_level_colon(current_segment)
    raw_value_fragment: str | None = None
    if colon_index is not None:
        raw_value_fragment = current_segment[colon_index + 1 :]

    if not current_segment.strip():
        context = "key"
    elif colon_index is None:
        context = "key"
    else:
        context = "value"

    return _PayloadCursorState(
        context=context,
        current_key=current_key,
        raw_value_fragment=raw_value_fragment,
        used_keys=used_keys,
    )


def _parse_scalar_hint(raw_text: str) -> str | None:
    value = raw_text.strip()
    if not value:
        return None
    if _BARE_SCALAR_RE.fullmatch(value):
        return value
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return None


def _payload_value_hints(before_cursor: str) -> dict[str, str]:
    transform_match = _TRANSFORM_LINE_RE.match(before_cursor)
    if transform_match is None:
        return {}
    payload = transform_match.group(1)
    if not payload.strip():
        return {}
    completed_segments, current_segment = _split_top_level_segments(payload)
    segments = list(completed_segments)
    if current_segment.strip():
        segments.append(current_segment)

    hints: dict[str, str] = {}
    for segment in segments:
        key = _parse_key_from_segment(segment)
        if key is None:
            continue
        colon_index = _find_top_level_colon(segment)
        if colon_index is None:
            continue
        value_hint = _parse_scalar_hint(segment[colon_index + 1 :])
        if value_hint is not None:
            hints[key] = value_hint
    return hints


def _payload_context(before_cursor: str) -> str:
    return _payload_cursor_state(before_cursor).context


def _current_value_key(before_cursor: str) -> str | None:
    return _payload_cursor_state(before_cursor).current_key


def _current_value_fragment(before_cursor: str) -> str | None:
    return _payload_cursor_state(before_cursor).raw_value_fragment


__all__ = [
    "_PayloadCursorState",
    "_find_top_level_colon",
    "_parse_key_from_segment",
    "_split_top_level_segments",
    "_payload_cursor_state",
    "_payload_context",
    "_current_value_key",
    "_current_value_fragment",
    "_payload_value_hints",
]
