from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .types import AxonExprPath


def parse_path_token(raw: str, *, op_name: str = "path") -> AxonExprPath:
    token = raw.strip()
    if not token.startswith("@"):
        raise ValueError(f"{op_name} expects Path key (expected @... or @@...)")
    absolute = token.startswith("@@")
    body = token[2:] if absolute else token[1:]
    if not body:
        raise ValueError(f"{op_name} expects one non-empty Path key")
    if len(body) >= 2 and body[0] == "'" and body[-1] == "'":
        quoted = body[1:-1].replace("\\'", "'").replace("\\\\", "\\")
        parts = tuple(part for part in quoted.split(".") if part)
    else:
        parts = tuple(part for part in body.split(".") if part)
    if not parts:
        raise ValueError(f"{op_name} expects one non-empty Path key")
    return AxonExprPath(absolute=absolute, parts=parts)


def path_expr_to_runtime_value(expr: AxonExprPath) -> dict[str, Any]:
    return {
        "_expr": "path",
        "absolute": expr.absolute,
        "parts": list(expr.parts),
    }


def runtime_value_to_path_expr(raw: object, *, op_name: str = "Path") -> AxonExprPath:
    if isinstance(raw, AxonExprPath):
        return raw
    if isinstance(raw, dict):
        kind = raw.get("_expr")
        if kind != "path":
            raise ValueError(f"{op_name} expects Path payload, got {raw!r}")
        absolute = raw.get("absolute")
        parts = raw.get("parts")
        if not isinstance(absolute, bool):
            raise ValueError(f"{op_name} path payload missing boolean 'absolute': {raw!r}")
        if (
            not isinstance(parts, list)
            or not parts
            or not all(isinstance(part, str) for part in parts)
        ):
            raise ValueError(f"{op_name} path payload missing string parts: {raw!r}")
        return AxonExprPath(absolute=absolute, parts=tuple(parts))
    if isinstance(raw, str):
        return parse_path_token(raw, op_name=op_name)
    raise ValueError(f"{op_name} expects one non-empty Path key")


def path_expr_template_text(expr: AxonExprPath) -> str:
    return ".".join(expr.parts)


def _resolve_template_text(text: str, env: Mapping[str, Any], op_name: str) -> str:
    if "{" not in text and "}" not in text:
        return text
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "}":
            raise ValueError(f"{op_name} key template has unmatched '}}': {text!r}")
        if ch != "{":
            out.append(ch)
            i += 1
            continue
        j = text.find("}", i + 1)
        if j < 0:
            raise ValueError(f"{op_name} key template has unmatched '{{': {text!r}")
        name = text[i + 1 : j].strip()
        if not name:
            raise ValueError(f"{op_name} key template has empty placeholder: {text!r}")
        if name not in env:
            raise ValueError(f"{op_name} key template placeholder {name!r} is not defined")
        value = env[name]
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(
                f"{op_name} key template placeholder {name!r} must resolve to scalar, got {type(value).__name__}"
            )
        out.append(str(value))
        i = j + 1
    return "".join(out)


def resolve_path_expr_to_key(
    raw: object,
    env: Mapping[str, Any],
    *,
    op_name: str = "Path",
) -> str:
    expr = runtime_value_to_path_expr(raw, op_name=op_name)
    resolved_parts: list[str] = []
    for part in expr.parts:
        resolved = _resolve_template_text(part, env, op_name)
        resolved_parts.extend(sub for sub in resolved.split(".") if sub)
    key = ".".join(resolved_parts)
    if not key:
        raise ValueError(f"{op_name} key must resolve to non-empty string")
    return key


def resolve_static_path_expr_to_key(raw: object, *, op_name: str = "Path") -> str:
    return resolve_path_expr_to_key(raw, {}, op_name=op_name)


__all__ = [
    "parse_path_token",
    "path_expr_template_text",
    "path_expr_to_runtime_value",
    "resolve_path_expr_to_key",
    "resolve_static_path_expr_to_key",
    "runtime_value_to_path_expr",
]
