from __future__ import annotations

from ..ast import (
    AxonBind,
    AxonCond,
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
    AxonFile,
    AxonDefinition,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
    AxonYield,
)
from .normalized import validate_normalized_axon_file


def _validate_expr_elaborated(expr: AxonExpr, *, module: AxonDefinition) -> None:
    if isinstance(expr, AxonExprCall):
        if expr.kwargs:
            raise ValueError(
                f"Axon elaborated validation failed in module {module.name!r}: "
                f"call to {expr.callee!r} still has kwargs"
            )
        for arg in expr.args:
            _validate_expr_elaborated(arg, module=module)
        return
    if isinstance(expr, AxonExprPipe):
        _validate_expr_elaborated(expr.value, module=module)
        for stage in expr.stages:
            _validate_expr_elaborated(stage, module=module)
        return
    if isinstance(expr, AxonExprBind):
        _validate_expr_elaborated(expr.value, module=module)
        _validate_expr_elaborated(expr.body, module=module)
        return
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        _validate_expr_elaborated(expr.cond, module=module)
        _validate_expr_elaborated(expr.true_expr, module=module)
        _validate_expr_elaborated(expr.false_expr, module=module)
        return
    if isinstance(expr, AxonExprBinary):
        _validate_expr_elaborated(expr.left, module=module)
        _validate_expr_elaborated(expr.right, module=module)
        return
    if isinstance(expr, AxonExprLambda):
        _validate_expr_elaborated(expr.body, module=module)
        return
    if isinstance(expr, AxonExprParen):
        _validate_expr_elaborated(expr.inner, module=module)
        return
    if isinstance(expr, AxonExprAscribe):
        _validate_expr_elaborated(expr.expr, module=module)
        return
    if isinstance(expr, AxonExprList | AxonExprTuple):
        for item in expr.items:
            _validate_expr_elaborated(item, module=module)
        return
    if isinstance(expr, AxonExprDo):
        for stmt in expr.body:
            _validate_statement_elaborated(stmt, module=module)


def _validate_statement_elaborated(stmt: AxonStatement, *, module: AxonDefinition) -> None:
    if isinstance(stmt, AxonBind):
        _validate_expr_elaborated(stmt.expr, module=module)
        return
    if isinstance(stmt, AxonReturn | AxonYield):
        for value in stmt.values:
            _validate_expr_elaborated(value, module=module)
        return
    if isinstance(stmt, AxonCond):
        _validate_expr_elaborated(stmt.cond, module=module)
        for inner in stmt.true_body:
            _validate_statement_elaborated(inner, module=module)
        for inner in stmt.false_body:
            _validate_statement_elaborated(inner, module=module)
        return
    if isinstance(stmt, AxonRepeat):
        _validate_expr_elaborated(stmt.from_expr, module=module)
        _validate_expr_elaborated(stmt.to_expr, module=module)
        _validate_expr_elaborated(stmt.step_expr, module=module)
        for inner in stmt.body:
            _validate_statement_elaborated(inner, module=module)
        return
    if isinstance(stmt, AxonScopeBind):
        if stmt.kwargs:
            raise ValueError(
                f"Axon elaborated validation failed in module {module.name!r}: "
                "scope bind still has kwargs"
            )
        for inner in stmt.body:
            _validate_statement_elaborated(inner, module=module)


def validate_elaborated_axon_file(ast: AxonFile, *, main_module: str | None = None) -> None:
    validate_normalized_axon_file(ast, main_module=main_module)
    for module in ast.modules:
        for param in module.params:
            if param.default_expr is not None:
                raise ValueError(
                    f"Axon elaborated validation failed in module {module.name!r}: "
                    f"parameter {param.name!r} still has a default"
                )
        for stmt in module.statements:
            _validate_statement_elaborated(stmt, module=module)
        if module.body_expr is not None:
            _validate_expr_elaborated(module.body_expr, module=module)


__all__ = ["validate_elaborated_axon_file"]
