from __future__ import annotations

import re
from dataclasses import dataclass

from lark import Lark
from lark.exceptions import LarkError
from lark.tree import Tree

_GRAMMAR = r"""
for_stmt: "for" ["@" SCOPE] IDENT "<-" RANGE ["step" "=" STEP_EXPR] "do"
scope_bind_stmt: TARGETS "<-" "scope" ["@"] SCOPE "do"
scope_stmt: "scope" ["@"] SCOPE "do"
return_stmt: "return" EXPR
bind_stmt: TARGETS "<-" EXPR

SCOPE: /[A-Za-z_][A-Za-z0-9_.]*/
IDENT: /[A-Za-z_][A-Za-z0-9_]*/
RANGE: /[\[\(]\s*.+?\s*\.\.\s*.+?\s*[\]\)\[]/
STEP_EXPR: /.+?(?=\s+do$)/
TARGETS: /.+?(?=<-)/
EXPR: /.+/

%import common.WS
%ignore WS
"""

_RANGE_RE = re.compile(r"^([\[\(])\s*(.+?)\s*\.\.\s*(.+?)\s*([\]\)\[])$")


@dataclass(frozen=True)
class ParsedFor:
    name: str | None
    var: str
    start_delim: str
    start_expr: str
    end_expr: str
    end_delim: str
    step_expr: str | None


@dataclass(frozen=True)
class ParsedScopeBind:
    raw_targets: str
    prefix: str


@dataclass(frozen=True)
class ParsedScope:
    prefix: str


@dataclass(frozen=True)
class ParsedReturn:
    raw_values: str


@dataclass(frozen=True)
class ParsedBind:
    raw_targets: str
    expr: str


ParsedStatementHead = ParsedFor | ParsedScopeBind | ParsedScope | ParsedReturn | ParsedBind


_PARSER = Lark(
    _GRAMMAR,
    parser="lalr",
    start=["for_stmt", "scope_bind_stmt", "scope_stmt", "return_stmt", "bind_stmt"],
)


def _parse_range(raw: str) -> tuple[str, str, str, str]:
    match = _RANGE_RE.match(raw.strip())
    if match is None:
        raise ValueError(f"invalid for-range expression: {raw!r}")
    return match.group(1), match.group(2).strip(), match.group(3).strip(), match.group(4)


def _first_token_value(tree: Tree, token_type: str) -> str | None:
    for child in tree.children:
        if getattr(child, "type", None) == token_type:
            return str(child)
    return None


def parse_statement_head(line: str) -> ParsedStatementHead:
    text = line.strip()
    if not text:
        raise ValueError("empty Axon statement")
    for start in ("for_stmt", "scope_bind_stmt", "scope_stmt", "return_stmt", "bind_stmt"):
        try:
            tree = _PARSER.parse(text, start=start)
        except LarkError:
            continue
        if start == "for_stmt":
            raw_scope = _first_token_value(tree, "SCOPE")
            raw_var = _first_token_value(tree, "IDENT")
            raw_range = _first_token_value(tree, "RANGE")
            raw_step = _first_token_value(tree, "STEP_EXPR")
            if raw_var is None or raw_range is None:
                raise ValueError(f"invalid for statement: {line!r}")
            start_delim, start_expr, end_expr, end_delim = _parse_range(raw_range)
            return ParsedFor(
                name=raw_scope.strip() if raw_scope else None,
                var=raw_var.strip(),
                start_delim=start_delim,
                start_expr=start_expr,
                end_expr=end_expr,
                end_delim=end_delim,
                step_expr=raw_step.strip() if raw_step else None,
            )
        if start == "scope_bind_stmt":
            raw_targets = _first_token_value(tree, "TARGETS")
            raw_scope = _first_token_value(tree, "SCOPE")
            if raw_targets is None or raw_scope is None:
                raise ValueError(f"invalid scope bind statement: {line!r}")
            return ParsedScopeBind(raw_targets=raw_targets.strip(), prefix=raw_scope.strip())
        if start == "scope_stmt":
            raw_scope = _first_token_value(tree, "SCOPE")
            if raw_scope is None:
                raise ValueError(f"invalid scope statement: {line!r}")
            return ParsedScope(prefix=raw_scope.strip())
        if start == "return_stmt":
            raw_values = _first_token_value(tree, "EXPR")
            if raw_values is None:
                raise ValueError(f"invalid return statement: {line!r}")
            return ParsedReturn(raw_values=raw_values.strip())
        if start == "bind_stmt":
            raw_targets = _first_token_value(tree, "TARGETS")
            expr = _first_token_value(tree, "EXPR")
            if raw_targets is None or expr is None:
                raise ValueError(f"invalid bind statement: {line!r}")
            return ParsedBind(raw_targets=raw_targets.strip(), expr=expr.strip())
    raise ValueError(f"unsupported Axon statement: {line!r}")


__all__ = [
    "ParsedBind",
    "ParsedFor",
    "ParsedReturn",
    "ParsedScope",
    "ParsedScopeBind",
    "ParsedStatementHead",
    "parse_statement_head",
]
