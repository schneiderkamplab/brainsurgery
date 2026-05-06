from __future__ import annotations

from ..ast import (
    AxonBind,
    AxonCond,
    AxonFile,
    AxonDefinition,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
    AxonYield,
    TypeList,
    TypeOptional,
)
from .typed import validate_typed_axon_file


def _is_list_type(value: object) -> bool:
    while isinstance(value, TypeOptional):
        value = value.inner
    return isinstance(value, TypeList)


def _validate_statement_backend_required(stmt: AxonStatement, *, module: AxonDefinition) -> None:
    if isinstance(stmt, AxonBind):
        if len(stmt.targets) > 1 and _is_list_type(stmt.expr.inferred_type):
            raise ValueError(
                f"Axon backend-required validation failed in module {module.name!r}: "
                "list destructuring bind remains"
            )
        return
    if isinstance(stmt, AxonCond):
        for inner in stmt.true_body:
            _validate_statement_backend_required(inner, module=module)
        for inner in stmt.false_body:
            _validate_statement_backend_required(inner, module=module)
        return
    if isinstance(stmt, AxonRepeat):
        for inner in stmt.body:
            _validate_statement_backend_required(inner, module=module)
        return
    if isinstance(stmt, AxonScopeBind):
        for inner in stmt.body:
            _validate_statement_backend_required(inner, module=module)
        return
    if isinstance(stmt, AxonReturn | AxonYield):
        return


def validate_backend_required_flat_typed_axon_file(
    ast: AxonFile, *, main_module: str | None = None
) -> None:
    validate_typed_axon_file(ast, main_module=main_module)
    for module in ast.modules:
        for stmt in module.statements:
            _validate_statement_backend_required(stmt, module=module)


__all__ = ["validate_backend_required_flat_typed_axon_file"]
