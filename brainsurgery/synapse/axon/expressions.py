from __future__ import annotations

import re

from .call_parser import split_csv


def tuple_items(expr: str) -> list[str]:
    text = expr.strip()
    if text.startswith("(") and text.endswith(")"):
        inner = text[1:-1].strip()
        return split_csv(inner)
    return split_csv(text)


def split_ternary(expr: str) -> tuple[str, str, str] | None:
    depth = 0
    qpos = -1
    cpos = -1
    for i, ch in enumerate(expr):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "?" and depth == 0 and qpos < 0:
            qpos = i
        elif ch == ":" and depth == 0 and qpos >= 0:
            if (i > 0 and expr[i - 1] == ":") or (i + 1 < len(expr) and expr[i + 1] == ":"):
                continue
            cpos = i
            break
    if qpos < 0 or cpos < 0:
        return None
    return expr[:qpos].strip(), expr[qpos + 1 : cpos].strip(), expr[cpos + 1 :].strip()


def _word_boundary(text: str, index: int) -> bool:
    if index < 0 or index >= len(text):
        return True
    return not (text[index].isalnum() or text[index] == "_")


def _find_top_level_keyword(text: str, keyword: str, *, start: int = 0) -> int:
    depth = 0
    i = start
    klen = len(keyword)
    while i < len(text):
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
            and _word_boundary(text, i - 1)
            and _word_boundary(text, i + klen)
        ):
            return i
        i += 1
    return -1


def split_if_then_else(expr: str) -> tuple[str, str, str] | None:
    text = expr.strip()
    if not text.startswith("if"):
        return None
    if not _word_boundary(text, 2):
        return None

    then_pos = _find_top_level_keyword(text, "then", start=2)
    if then_pos < 0:
        return None

    cond = text[2:then_pos].strip()
    if not cond:
        return None

    body_start = then_pos + len("then")
    depth = 0
    nested_if = 0
    i = body_start
    else_pos = -1
    while i < len(text):
        ch = text[i]
        if ch in "([":
            depth += 1
            i += 1
            continue
        if ch in ")]":
            depth -= 1
            i += 1
            continue
        if depth == 0:
            if (
                text.startswith("if", i)
                and _word_boundary(text, i - 1)
                and _word_boundary(text, i + 2)
            ):
                nested_if += 1
                i += 2
                continue
            if (
                text.startswith("else", i)
                and _word_boundary(text, i - 1)
                and _word_boundary(text, i + 4)
            ):
                if nested_if == 0:
                    else_pos = i
                    break
                nested_if -= 1
                i += 4
                continue
        i += 1

    if else_pos < 0:
        return None

    true_expr = text[body_start:else_pos].strip()
    false_expr = text[else_pos + len("else") :].strip()
    if not true_expr or not false_expr:
        return None
    return cond, true_expr, false_expr


def substitute_var(expr: str, name: str, value: str) -> str:
    return re.sub(rf"\b{re.escape(name)}\b", value, expr)


def split_binary(expr: str, operator: str) -> tuple[str, str] | None:
    if not operator:
        return None
    depth = 0
    op_len = len(operator)
    last_match = -1
    i = 0
    limit = len(expr) - op_len
    while i <= limit:
        ch = expr[i]
        if ch in ")]":
            depth -= 1
            i += 1
            continue
        if ch in "([":
            depth += 1
            i += 1
            continue
        if depth == 0 and expr.startswith(operator, i):
            # Keep scientific-notation exponents intact (e.g. 1e-05).
            if not (operator == "-" and i > 0 and expr[i - 1] in {"e", "E"}):
                last_match = i
            i += op_len
            continue
        i += 1
    if last_match >= 0:
        left = expr[:last_match].strip()
        right = expr[last_match + op_len :].strip()
        if left and right:
            return left, right
    return None


def is_name_token(expr: str) -> bool:
    token = expr.strip()
    return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token) is not None


__all__ = [
    "tuple_items",
    "split_ternary",
    "split_if_then_else",
    "substitute_var",
    "split_binary",
    "is_name_token",
]
