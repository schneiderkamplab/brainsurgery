from __future__ import annotations

from typing import Any

from .grammar import parse_expression_source
from .path_expr import path_expr_to_runtime_value
from .types import (
    AxonExpr,
    AxonExprBinary,
    AxonExprBool,
    AxonExprCall,
    AxonExprFloat,
    AxonExprIf,
    AxonExprInt,
    AxonExprList,
    AxonExprName,
    AxonExprNull,
    AxonExprParen,
    AxonExprPath,
    AxonExprString,
    AxonExprTernary,
    AxonExprTuple,
)


def axon_expr_to_runtime_value(expr: AxonExpr) -> Any:
    if isinstance(expr, AxonExprName):
        return {"_expr": "name", "id": expr.name}
    if isinstance(expr, AxonExprInt):
        return expr.value
    if isinstance(expr, AxonExprFloat):
        return expr.value
    if isinstance(expr, AxonExprBool):
        return expr.value
    if isinstance(expr, AxonExprNull):
        return None
    if isinstance(expr, AxonExprString):
        return {"_expr": "string", "value": expr.value}
    if isinstance(expr, AxonExprPath):
        return path_expr_to_runtime_value(expr)
    if isinstance(expr, AxonExprList):
        return [axon_expr_to_runtime_value(item) for item in expr.items]
    if isinstance(expr, AxonExprTuple):
        return {
            "_expr": "tuple",
            "items": [axon_expr_to_runtime_value(item) for item in expr.items],
        }
    if isinstance(expr, AxonExprParen):
        return axon_expr_to_runtime_value(expr.inner)
    if isinstance(expr, AxonExprCall):
        kw_out: dict[str, Any] = {}
        for key, value in expr.kwargs.items():
            if isinstance(value, AxonExpr):
                kw_out[key] = axon_expr_to_runtime_value(value)
            else:
                kw_out[key] = value
        return {
            "_expr": "call",
            "callee": expr.callee,
            "args": [axon_expr_to_runtime_value(arg) for arg in expr.args],
            "kwargs": kw_out,
        }
    if isinstance(expr, AxonExprBinary):
        return {
            "_expr": "binary",
            "op": expr.op,
            "left": axon_expr_to_runtime_value(expr.left),
            "right": axon_expr_to_runtime_value(expr.right),
        }
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return {
            "_expr": "if",
            "cond": axon_expr_to_runtime_value(expr.cond),
            "then": axon_expr_to_runtime_value(expr.true_expr),
            "else": axon_expr_to_runtime_value(expr.false_expr),
        }
    raise ValueError(
        f"expression form is not representable in runtime value form: {type(expr).__name__}"
    )


def parse_expression_to_runtime_value(source: str) -> Any:
    return axon_expr_to_runtime_value(parse_expression_source(source))


__all__ = ["axon_expr_to_runtime_value", "parse_expression_to_runtime_value"]
