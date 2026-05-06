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
    TypePath,
)
from .closed import validate_closed_axon_file


def _split_callee_path_sugar(callee: str) -> tuple[str, tuple[str, ...]]:
    if "@" not in callee:
        return callee, ()
    parts = callee.split("@")
    return parts[0], tuple(parts[1:])


def _leading_path_param_count(module: AxonDefinition) -> int:
    count = 0
    for param in module.params:
        if isinstance(param.type_expr, TypePath):
            count += 1
            continue
        break
    return count


def _validate_expr_normalized(
    expr: AxonExpr, *, module: AxonDefinition, modules_by_name: dict[str, AxonDefinition]
) -> None:
    if isinstance(expr, AxonExprCall):
        base_callee, path_suffixes = _split_callee_path_sugar(expr.callee)
        if path_suffixes:
            raise ValueError(
                f"Axon normalized validation failed in module {module.name!r}: "
                "callee path sugar remains"
            )
        callee_module = modules_by_name.get(base_callee)
        if callee_module is not None:
            path_slot_count = len(callee_module.path_params) + _leading_path_param_count(
                callee_module
            )
            if len(expr.args) < path_slot_count:
                raise ValueError(
                    f"Axon normalized validation failed in module {module.name!r}: "
                    f"missing explicit path args for call to {base_callee!r}"
                )
        for arg in expr.args:
            _validate_expr_normalized(arg, module=module, modules_by_name=modules_by_name)
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                _validate_expr_normalized(value, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(expr, AxonExprPipe):
        raise ValueError(
            f"Axon normalized validation failed in module {module.name!r}: pipe remains"
        )
    if isinstance(expr, AxonExprBind):
        _validate_expr_normalized(expr.value, module=module, modules_by_name=modules_by_name)
        _validate_expr_normalized(expr.body, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        _validate_expr_normalized(expr.cond, module=module, modules_by_name=modules_by_name)
        _validate_expr_normalized(expr.true_expr, module=module, modules_by_name=modules_by_name)
        _validate_expr_normalized(expr.false_expr, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(expr, AxonExprBinary):
        _validate_expr_normalized(expr.left, module=module, modules_by_name=modules_by_name)
        _validate_expr_normalized(expr.right, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(expr, AxonExprLambda):
        _validate_expr_normalized(expr.body, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(expr, AxonExprParen):
        _validate_expr_normalized(expr.inner, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(expr, AxonExprAscribe):
        _validate_expr_normalized(expr.expr, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(expr, AxonExprList | AxonExprTuple):
        for item in expr.items:
            _validate_expr_normalized(item, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(expr, AxonExprDo):
        for stmt in expr.body:
            _validate_statement_normalized(
                stmt, module=module, modules_by_name=modules_by_name
            )


def _validate_repeat_normalized(stmt: AxonRepeat, *, module: AxonDefinition) -> None:
    if not stmt.body or not isinstance(stmt.body[-1], AxonYield):
        raise ValueError(
            f"Axon normalized validation failed in module {module.name!r}: "
            "repeat body must end with explicit yield"
        )
    if stmt.targets is not None:
        if stmt.carry is None:
            raise ValueError(
                f"Axon normalized validation failed in module {module.name!r}: "
                "targeted repeat must have explicit carry"
            )
        if len(stmt.targets) != len(stmt.carry):
            raise ValueError(
                f"Axon normalized validation failed in module {module.name!r}: "
                "repeat target/carry arity mismatch"
            )


def _validate_statement_normalized(
    stmt: AxonStatement, *, module: AxonDefinition, modules_by_name: dict[str, AxonDefinition]
) -> None:
    if isinstance(stmt, AxonBind):
        _validate_expr_normalized(stmt.expr, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(stmt, AxonReturn | AxonYield):
        for value in stmt.values:
            _validate_expr_normalized(value, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(stmt, AxonCond):
        _validate_expr_normalized(stmt.cond, module=module, modules_by_name=modules_by_name)
        for inner in stmt.true_body:
            _validate_statement_normalized(
                inner, module=module, modules_by_name=modules_by_name
            )
        for inner in stmt.false_body:
            _validate_statement_normalized(
                inner, module=module, modules_by_name=modules_by_name
            )
        return
    if isinstance(stmt, AxonRepeat):
        _validate_expr_normalized(stmt.from_expr, module=module, modules_by_name=modules_by_name)
        _validate_expr_normalized(stmt.to_expr, module=module, modules_by_name=modules_by_name)
        _validate_expr_normalized(stmt.step_expr, module=module, modules_by_name=modules_by_name)
        _validate_repeat_normalized(stmt, module=module)
        for inner in stmt.body:
            _validate_statement_normalized(
                inner, module=module, modules_by_name=modules_by_name
            )
        return
    if isinstance(stmt, AxonScopeBind):
        for raw_value in stmt.kwargs.values():
            if isinstance(raw_value, AxonExpr):
                _validate_expr_normalized(
                    raw_value, module=module, modules_by_name=modules_by_name
                )
        for inner in stmt.body:
            _validate_statement_normalized(
                inner, module=module, modules_by_name=modules_by_name
            )


def validate_normalized_axon_file(ast: AxonFile, *, main_module: str | None = None) -> None:
    validate_closed_axon_file(ast, main_module=main_module)
    modules_by_name = {module.name: module for module in ast.modules}
    for module in ast.modules:
        for param in module.params:
            if param.default_expr is not None:
                _validate_expr_normalized(
                    param.default_expr, module=module, modules_by_name=modules_by_name
                )
        for stmt in module.statements:
            _validate_statement_normalized(
                stmt, module=module, modules_by_name=modules_by_name
            )
        if module.body_expr is not None:
            _validate_expr_normalized(
                module.body_expr, module=module, modules_by_name=modules_by_name
            )


__all__ = ["validate_normalized_axon_file"]
