from __future__ import annotations

from ..ast import (
    AxonBind,
    AxonCond,
    AxonExpr,
    AxonExprCall,
    AxonExprDo,
    AxonExprIf,
    AxonExprTernary,
    AxonExprTuple,
    AxonFile,
    AxonModule,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
    AxonYield,
    TypeTensor,
    TypeTuple,
)
from .flat import validate_flat_axon_file


def _validate_expr_typed(expr: AxonExpr, *, module: AxonModule) -> None:
    if expr.inferred_type is None:
        raise ValueError(
            f"Axon typed validation failed in module {module.name!r}: missing inferred type"
        )
    if expr.inferred_arity is None:
        raise ValueError(
            f"Axon typed validation failed in module {module.name!r}: missing inferred arity"
        )
    if isinstance(expr.inferred_type, TypeTensor) and expr.inferred_dims is None:
        raise ValueError(
            f"Axon typed validation failed in module {module.name!r}: tensor expr missing dims"
        )
    if isinstance(expr.inferred_type, TypeTuple) and expr.inferred_arity != len(
        expr.inferred_type.items
    ):
        raise ValueError(
            f"Axon typed validation failed in module {module.name!r}: tuple arity mismatch"
        )
    if isinstance(expr, AxonExprCall):
        for arg in expr.args:
            _validate_expr_typed(arg, module=module)
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                _validate_expr_typed(value, module=module)
    elif isinstance(expr, AxonExprTuple):
        for item in expr.items:
            _validate_expr_typed(item, module=module)
    elif isinstance(expr, AxonExprIf | AxonExprTernary):
        _validate_expr_typed(expr.cond, module=module)
        _validate_expr_typed(expr.true_expr, module=module)
        _validate_expr_typed(expr.false_expr, module=module)
    elif isinstance(expr, AxonExprDo):
        for stmt in expr.body:
            _validate_statement_typed(stmt, module=module)
    else:
        for attr in ("value", "inner", "expr", "left", "right", "body"):
            if hasattr(expr, attr):
                value = getattr(expr, attr)
                if isinstance(value, AxonExpr):
                    _validate_expr_typed(value, module=module)


def _validate_statement_typed(stmt: AxonStatement, *, module: AxonModule) -> None:
    if isinstance(stmt, AxonBind):
        _validate_expr_typed(stmt.expr, module=module)
    elif isinstance(stmt, AxonReturn | AxonYield):
        for value in stmt.values:
            _validate_expr_typed(value, module=module)
    elif isinstance(stmt, AxonCond):
        _validate_expr_typed(stmt.cond, module=module)
        for inner in stmt.true_body:
            _validate_statement_typed(inner, module=module)
        for inner in stmt.false_body:
            _validate_statement_typed(inner, module=module)
    elif isinstance(stmt, AxonRepeat):
        _validate_expr_typed(stmt.from_expr, module=module)
        _validate_expr_typed(stmt.to_expr, module=module)
        _validate_expr_typed(stmt.step_expr, module=module)
        for inner in stmt.body:
            _validate_statement_typed(inner, module=module)
    elif isinstance(stmt, AxonScopeBind):
        for raw_value in stmt.kwargs.values():
            if isinstance(raw_value, AxonExpr):
                _validate_expr_typed(raw_value, module=module)
        for inner in stmt.body:
            _validate_statement_typed(inner, module=module)


def validate_typed_axon_file(ast: AxonFile, *, main_module: str | None = None) -> None:
    validate_flat_axon_file(ast, main_module=main_module)
    for module in ast.modules:
        for stmt in module.statements:
            _validate_statement_typed(stmt, module=module)


__all__ = ["validate_typed_axon_file"]
