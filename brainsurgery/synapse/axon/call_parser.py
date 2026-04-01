from __future__ import annotations

import ast
import re
from typing import Any

_CALL_PAREN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_:.@]*)\((.*)\)$")
_ZERO_ARG_CALLS = {"list_init", "init", "Cache.init", "List.init", "_list_init"}
_INVALID_POSITIONAL_TOKENS = {"+", "-", "*", "/", "%", "==", "!=", "<", ">", "<=", ">="}


def split_top_level(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    seplen = len(sep)
    while i < len(text):
        ch = text[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if depth == 0 and text.startswith(sep, i):
            parts.append(text[start:i].strip())
            i += seplen
            start = i
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def split_csv(text: str) -> list[str]:
    return split_top_level(text, ",")


def parse_scalar(token: str) -> Any:
    value = token.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null":
        return None
    if value and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        return value[1:-1]
    if re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    if re.fullmatch(r"-?[0-9]+\.[0-9]+", value):
        return float(value)
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return value
        if isinstance(parsed, list):
            return parsed
    return value


def parse_call(expr: str) -> tuple[str, list[str], dict[str, Any]]:
    text = expr.strip()
    match = _CALL_PAREN_RE.match(text)
    if match is not None:
        callee = match.group(1).strip()
        raw_args = match.group(2).strip()
        tokens = split_csv(raw_args) if raw_args else []
    else:
        # Point-free calls require whitespace between callee and first argument.
        # Without this guard, tokens like `x+1` were misclassified as calls.
        callee_match = re.match(r"^([A-Za-z_][A-Za-z0-9_:.@]*)(?:\s+(.*))?$", text)
        if callee_match is None:
            raise ValueError(f"expected call expression, got: {expr!r}")
        callee = callee_match.group(1).strip()
        raw_rest = callee_match.group(2)
        rest = raw_rest.strip() if isinstance(raw_rest, str) else ""
        if not rest and "@" not in callee and "::" not in callee and callee not in _ZERO_ARG_CALLS:
            raise ValueError(f"expected call expression, got: {expr!r}")
        if not rest:
            tokens = []
        else:
            key_spans: list[tuple[int, int, str]] = []
            depth = 0
            i = 0
            while i < len(rest):
                ch = rest[i]
                if ch in "([":
                    depth += 1
                    i += 1
                    continue
                if ch in ")]":
                    depth -= 1
                    i += 1
                    continue
                if depth == 0 and (i == 0 or rest[i - 1].isspace()):
                    key_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", rest[i:])
                    if key_match is not None:
                        key = key_match.group(1)
                        key_end = i + key_match.end()
                        key_spans.append((i, key_end, key))
                        i = key_end
                        continue
                i += 1

            tokens = []
            if not key_spans:
                tokens.extend(part for part in split_top_level(rest, " ") if part)
            else:
                first_key_start = key_spans[0][0]
                pos_prefix = rest[:first_key_start].strip()
                if pos_prefix:
                    tokens.extend(part for part in split_top_level(pos_prefix, " ") if part)
                for idx, (_, key_end, key_name) in enumerate(key_spans):
                    next_start = key_spans[idx + 1][0] if idx + 1 < len(key_spans) else len(rest)
                    value_text = rest[key_end:next_start].strip()
                    tokens.append(f"{key_name}={value_text}")
    args: list[str] = []
    kwargs: dict[str, Any] = {}
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            kwargs[key.strip()] = parse_scalar(value)
        else:
            stripped = token.strip()
            if stripped in _INVALID_POSITIONAL_TOKENS:
                raise ValueError(f"expected call expression, got: {expr!r}")
            args.append(stripped)
    return callee, args, kwargs


def looks_like_call(expr: str) -> bool:
    try:
        parse_call(expr)
        return True
    except ValueError:
        return False


def strip_wrapping_parens(text: str) -> str:
    current = text.strip()
    while current.startswith("(") and current.endswith(")"):
        depth = 0
        valid = True
        for idx, ch in enumerate(current):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    valid = False
                    break
                if depth == 0 and idx != len(current) - 1:
                    valid = False
                    break
        if not valid or depth != 0:
            break
        current = current[1:-1].strip()
    return current


def render_call(callee: str, args: list[str], kwargs: dict[str, Any]) -> str:
    items = [*args, *[f"{k}={v}" for k, v in kwargs.items()]]
    return f"{callee}({', '.join(items)})"


__all__ = [
    "split_top_level",
    "split_csv",
    "parse_scalar",
    "parse_call",
    "looks_like_call",
    "strip_wrapping_parens",
    "render_call",
]
