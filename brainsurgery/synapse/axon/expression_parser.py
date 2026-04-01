from __future__ import annotations

import re

from .call_parser import looks_like_call, parse_call, parse_scalar, split_top_level
from .expressions import split_binary, split_if_then_else, split_ternary, tuple_items
from .types import (
    AxonExpr,
    AxonExprBinary,
    AxonExprBind,
    AxonExprCall,
    AxonExprIf,
    AxonExprLambda,
    AxonExprLiteral,
    AxonExprName,
    AxonExprParen,
    AxonExprPipe,
    AxonExprTernary,
    AxonExprTuple,
)

_NUMERIC_TOKEN_RE = re.compile(r"-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")


def _unwrap_one_paren(text: str) -> str | None:
    token = text.strip()
    if not token.startswith("(") or not token.endswith(")"):
        return None
    depth = 0
    for idx, ch in enumerate(token):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return None
            if depth == 0 and idx != len(token) - 1:
                return None
    if depth != 0:
        return None
    return token[1:-1].strip()


def _is_name(text: str) -> bool:
    token = text.strip()
    return (
        bool(token)
        and (token[0].isalpha() or token[0] == "_")
        and all(ch.isalnum() or ch == "_" for ch in token[1:])
    )


def _parse_bind_lambda(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    if not stripped.startswith("\\"):
        return None
    body = stripped[1:].strip()
    if "->" not in body:
        return None
    var_part, expr_part = body.split("->", 1)
    var_name = var_part.strip()
    expr = expr_part.strip()
    if not _is_name(var_name) or not expr:
        return None
    return var_name, expr


def _is_word_boundary(text: str, index: int) -> bool:
    if index < 0 or index >= len(text):
        return True
    ch = text[index]
    return not (ch.isalnum() or ch == "_")


def _split_keyword_binary(text: str, keyword: str) -> tuple[str, str] | None:
    depth = 0
    last = -1
    i = 0
    klen = len(keyword)
    limit = len(text) - klen
    while i <= limit:
        ch = text[i]
        if ch in "([":
            depth += 1
            i += 1
            continue
        if ch in ")]":
            depth -= 1
            i += 1
            continue
        if (
            depth == 0
            and text.startswith(keyword, i)
            and _is_word_boundary(text, i - 1)
            and _is_word_boundary(text, i + klen)
        ):
            last = i
            i += klen
            continue
        i += 1
    if last < 0:
        return None
    left = text[:last].strip()
    right = text[last + klen :].strip()
    if not left or not right:
        return None
    return left, right


def parse_expression(text: str) -> AxonExpr:
    raw = text.strip()
    if not raw:
        raise ValueError("empty expression")
    token = raw

    inner = _unwrap_one_paren(token)
    if inner is not None:
        return AxonExprParen(inner=parse_expression(inner))

    if_then_else = split_if_then_else(token)
    if if_then_else is not None:
        cond, true_expr, false_expr = if_then_else
        return AxonExprIf(
            cond=parse_expression(cond),
            true_expr=parse_expression(true_expr),
            false_expr=parse_expression(false_expr),
        )

    ternary = split_ternary(token)
    if ternary is not None:
        cond, true_expr, false_expr = ternary
        return AxonExprTernary(
            cond=parse_expression(cond),
            true_expr=parse_expression(true_expr),
            false_expr=parse_expression(false_expr),
        )

    bind_parts = split_top_level(token, ">>=")
    if len(bind_parts) > 1:
        current = parse_expression(bind_parts[0])
        for stage in bind_parts[1:]:
            parsed_lambda = _parse_bind_lambda(stage)
            if parsed_lambda is None:
                raise ValueError(f"expected lambda expression after >>=, got: {stage!r}")
            var_name, body = parsed_lambda
            current = AxonExprBind(value=current, var=var_name, body=parse_expression(body))
        return current

    pipe_parts = split_top_level(token, "|>")
    if len(pipe_parts) > 1:
        return AxonExprPipe(
            value=parse_expression(pipe_parts[0]),
            stages=tuple(parse_expression(part) for part in pipe_parts[1:]),
        )

    items = tuple_items(token)
    if len(items) > 1:
        return AxonExprTuple(items=tuple(parse_expression(item) for item in items))

    for op in ("==", "!=", "<=", ">=", "<", ">"):
        comparison = split_binary(token, op)
        if comparison is not None:
            left, right = comparison
            return AxonExprBinary(op=op, left=parse_expression(left), right=parse_expression(right))

    logical_or = _split_keyword_binary(token, "or")
    if logical_or is not None:
        left, right = logical_or
        return AxonExprBinary(op="or", left=parse_expression(left), right=parse_expression(right))

    logical_and = _split_keyword_binary(token, "and")
    if logical_and is not None:
        left, right = logical_and
        return AxonExprBinary(op="and", left=parse_expression(left), right=parse_expression(right))

    if looks_like_call(token):
        callee, args, kwargs = parse_call(token)
        parsed_kwargs: dict[str, AxonExpr | object] = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                parsed_kwargs[key] = parse_expression(value)
            else:
                parsed_kwargs[key] = value
        return AxonExprCall(
            callee=callee,
            args=tuple(parse_expression(arg) for arg in args),
            kwargs=parsed_kwargs,
        )

    plus = split_binary(token, "+")
    if plus is not None:
        left, right = plus
        return AxonExprBinary(op="+", left=parse_expression(left), right=parse_expression(right))

    minus = split_binary(token, "-")
    if minus is not None:
        left, right = minus
        return AxonExprBinary(op="-", left=parse_expression(left), right=parse_expression(right))

    mul = split_binary(token, "*")
    if mul is not None:
        left, right = mul
        return AxonExprBinary(op="*", left=parse_expression(left), right=parse_expression(right))

    div = split_binary(token, "/")
    if div is not None:
        left, right = div
        return AxonExprBinary(op="/", left=parse_expression(left), right=parse_expression(right))

    mod = split_binary(token, "%")
    if mod is not None:
        left, right = mod
        return AxonExprBinary(op="%", left=parse_expression(left), right=parse_expression(right))

    scalar = parse_scalar(token)
    if _NUMERIC_TOKEN_RE.fullmatch(token):
        return AxonExprLiteral(value=token)
    if scalar != token or token.lower() in {"true", "false", "null"}:
        return AxonExprLiteral(value=scalar)
    if _is_name(token):
        return AxonExprName(name=token)
    if token.startswith("\\"):
        parsed_lambda = _parse_bind_lambda(token)
        if parsed_lambda is not None:
            var_name, body = parsed_lambda
            return AxonExprLambda(var=var_name, body=parse_expression(body))
    raise ValueError(f"unsupported expression syntax: {raw!r}")


__all__ = ["parse_expression"]
