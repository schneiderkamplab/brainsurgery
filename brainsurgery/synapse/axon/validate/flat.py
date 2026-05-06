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
    AxonExprName,
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


def _validate_expr_flat(
    expr: AxonExpr, *, module: AxonDefinition, modules_by_name: dict[str, AxonDefinition]
) -> None:
    if isinstance(expr, AxonExprParen):
        raise ValueError(f"Axon flat validation failed in module {module.name!r}: parens remain")
    if isinstance(expr, AxonExprPipe):
        raise ValueError(f"Axon flat validation failed in module {module.name!r}: pipe remains")
    if isinstance(expr, AxonExprCall):
        base_callee, path_suffixes = _split_callee_path_sugar(expr.callee)
        if path_suffixes:
            raise ValueError(
                f"Axon flat validation failed in module {module.name!r}: callee path sugar remains"
            )
        callee_module = modules_by_name.get(base_callee)
        for arg in expr.args:
            _validate_expr_flat(arg, module=module, modules_by_name=modules_by_name)
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                _validate_expr_flat(value, module=module, modules_by_name=modules_by_name)
        if callee_module is not None:
            path_slot_count = len(callee_module.path_params) + _leading_path_param_count(
                callee_module
            )
            if len(expr.args) < path_slot_count:
                raise ValueError(
                    f"Axon flat validation failed in module {module.name!r}: explicit path args are missing"
                )
        return
    if isinstance(expr, AxonExprBinary):
        _validate_expr_flat(expr.left, module=module, modules_by_name=modules_by_name)
        _validate_expr_flat(expr.right, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(expr, AxonExprList | AxonExprTuple):
        for item in expr.items:
            _validate_expr_flat(item, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(expr, AxonExprAscribe):
        _validate_expr_flat(expr.expr, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(expr, AxonExprBind):
        raise ValueError(
            f"Axon flat validation failed in module {module.name!r}: bind-expr remains in flat AST"
        )
    if isinstance(expr, AxonExprIf):
        raise ValueError(
            f"Axon flat validation failed in module {module.name!r}: conditional expression remains"
        )
    if isinstance(expr, AxonExprTernary):
        _validate_expr_flat(expr.cond, module=module, modules_by_name=modules_by_name)
        _validate_expr_flat(expr.true_expr, module=module, modules_by_name=modules_by_name)
        _validate_expr_flat(expr.false_expr, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(expr, AxonExprLambda):
        _validate_expr_flat(expr.body, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(expr, AxonExprDo):
        raise ValueError(
            f"Axon flat validation failed in module {module.name!r}: do-expression remains"
        )


def _validate_statement_flat(
    stmt: AxonStatement, *, module: AxonDefinition, modules_by_name: dict[str, AxonDefinition]
) -> None:
    if isinstance(stmt, AxonBind):
        _validate_expr_flat(stmt.expr, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(stmt, AxonReturn | AxonYield):
        for value in stmt.values:
            _validate_expr_flat(value, module=module, modules_by_name=modules_by_name)
        return
    if isinstance(stmt, AxonCond):
        raise ValueError(
            f"Axon flat validation failed in module {module.name!r}: conditional statement remains"
        )
    if isinstance(stmt, AxonRepeat):
        raise ValueError(f"Axon flat validation failed in module {module.name!r}: repeat remains")
    if isinstance(stmt, AxonScopeBind):
        raise ValueError(
            f"Axon flat validation failed in module {module.name!r}: scope bind remains"
        )


def _validate_statements_flat(
    stmts: tuple[AxonStatement, ...], *, module: AxonDefinition, modules_by_name: dict[str, AxonDefinition]
) -> None:
    for stmt in stmts:
        _validate_statement_flat(stmt, module=module, modules_by_name=modules_by_name)


def validate_flat_axon_file(ast: AxonFile, *, main_module: str | None = None) -> None:
    validate_closed_axon_file(ast, main_module=main_module)
    if ast.type_aliases:
        raise ValueError("Axon flat validation failed: type aliases must be eliminated")
    modules_by_name = {module.name: module for module in ast.modules}
    for module in ast.modules:
        if module.type_aliases:
            raise ValueError(
                f"Axon flat validation failed in module {module.name!r}: type aliases must be eliminated"
            )
        if module.path_param is not None or module.path_params:
            raise ValueError(
                f"Axon flat validation failed in module {module.name!r}: path-sugar params must be eliminated"
            )
        if module.body_expr is not None:
            raise ValueError(
                f"Axon flat validation failed in module {module.name!r}: body_expr must be empty"
            )
        _validate_statements_flat(module.statements, module=module, modules_by_name=modules_by_name)


__all__ = ["validate_flat_axon_file"]
