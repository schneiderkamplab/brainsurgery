from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from .nodes import (
    AxonExpr,
    AxonExprAscribe,
    AxonExprBinary,
    AxonExprBind,
    AxonExprCall,
    AxonExprDo,
    AxonExprIf,
    AxonExprLambda,
    AxonExprList,
    AxonExprParen,
    AxonExprPipe,
    AxonExprTernary,
    AxonExprTuple,
)
from .source import AxonFile


def _strip_parens(expr: AxonExpr) -> AxonExpr:
    while isinstance(expr, AxonExprParen):
        expr = expr.inner
    return expr


def _normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, AxonExprParen):
        return _normalize(_strip_parens(value))
    if isinstance(value, AxonExprBinary):
        return (
            "AxonExprBinary",
            value.op,
            _normalize(_strip_parens(value.left)),
            _normalize(_strip_parens(value.right)),
        )
    if isinstance(value, AxonExprBind):
        return (
            "AxonExprBind",
            value.var,
            _normalize(_strip_parens(value.value)),
            _normalize(_strip_parens(value.body)),
        )
    if isinstance(value, AxonExprCall):
        return (
            "AxonExprCall",
            value.callee,
            tuple(_normalize(_strip_parens(arg)) for arg in value.args),
            tuple((key, _normalize(val)) for key, val in value.kwargs.items()),
        )
    if isinstance(value, AxonExprDo):
        return ("AxonExprDo", tuple(_normalize(stmt) for stmt in value.body), value.inline)
    if isinstance(value, AxonExprIf):
        return (
            "AxonExprIf",
            _normalize(_strip_parens(value.cond)),
            _normalize(_strip_parens(value.true_expr)),
            _normalize(_strip_parens(value.false_expr)),
        )
    if isinstance(value, AxonExprLambda):
        return ("AxonExprLambda", value.var, _normalize(_strip_parens(value.body)))
    if isinstance(value, AxonExprAscribe):
        return (
            "AxonExprAscribe",
            _normalize(_strip_parens(value.expr)),
            _normalize(value.type_expr),
        )
    if isinstance(value, AxonExprList):
        return ("AxonExprList", tuple(_normalize(_strip_parens(item)) for item in value.items))
    if isinstance(value, AxonExprPipe):
        return (
            "AxonExprPipe",
            _normalize(_strip_parens(value.value)),
            tuple(_normalize(_strip_parens(stage)) for stage in value.stages),
        )
    if isinstance(value, AxonExprTernary):
        return (
            "AxonExprTernary",
            _normalize(_strip_parens(value.cond)),
            _normalize(_strip_parens(value.true_expr)),
            _normalize(_strip_parens(value.false_expr)),
        )
    if isinstance(value, AxonExprTuple):
        return ("AxonExprTuple", tuple(_normalize(_strip_parens(item)) for item in value.items))
    if isinstance(value, AxonFile):
        return (
            "AxonFile",
            tuple(_normalize(module) for module in value.modules),
            value.imports,
            tuple((k, v) for k, v in value.imported_members.items()),
            value.exports,
            tuple((k, _normalize(v)) for k, v in value.pragmas.items()),
            tuple((k, _normalize(v)) for k, v in value.constants.items()),
            tuple((k, _normalize(v)) for k, v in value.type_aliases.items()),
        )
    if is_dataclass(value):
        return (
            type(value).__name__,
            tuple(
                (field.name, _normalize(getattr(value, field.name)))
                for field in fields(value)
                if not (type(value).__name__ == "AxonFile" and field.name == "origin_path")
            ),
        )
    if isinstance(value, tuple):
        return tuple(_normalize(item) for item in value)
    if isinstance(value, list):
        return tuple(_normalize(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _normalize(val)) for key, val in value.items())
    return value


def ast_equal(left: Any, right: Any) -> bool:
    return _normalize(left) == _normalize(right)


__all__ = ["ast_equal"]
