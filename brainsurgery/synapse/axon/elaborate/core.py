from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from ..ast import (
    AxonBind,
    AxonCond,
    AxonDefinition,
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
    AxonExprNull,
    AxonExprParen,
    AxonExprPath,
    AxonExprPipe,
    AxonExprTernary,
    AxonExprTuple,
    AxonFile,
    AxonKwargValue,
    AxonParam,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
    AxonYield,
    TypePath,
)
from ..validate import validate_normalized_axon_file


def _leading_path_param_count(module: AxonDefinition) -> int:
    count = 0
    for param in module.params:
        if isinstance(param.type_expr, TypePath):
            count += 1
            continue
        break
    return count


def _path_slot_count(module: AxonDefinition) -> int:
    return len(module.path_params) + _leading_path_param_count(module)


def _scoped_default_expr(
    expr: AxonExpr,
    *,
    path_prefix: tuple[str, ...],
    default_base_path: AxonExprPath | None,
) -> AxonExpr:
    if isinstance(expr, AxonExprPath):
        if expr.absolute or default_base_path is None:
            return expr
        return AxonExprPath(
            absolute=default_base_path.absolute,
            parts=(*default_base_path.parts, *expr.parts),
        )
    if isinstance(expr, AxonExprCall):
        return replace(
            expr,
            args=tuple(
                _scoped_default_expr(
                    arg,
                    path_prefix=path_prefix,
                    default_base_path=default_base_path,
                )
                for arg in expr.args
            ),
            kwargs={
                key: _scoped_default_expr(
                    value,
                    path_prefix=path_prefix,
                    default_base_path=default_base_path,
                )
                if isinstance(value, AxonExpr)
                else deepcopy(value)
                for key, value in expr.kwargs.items()
            },
        )
    if isinstance(expr, AxonExprPipe):
        return replace(
            expr,
            value=_scoped_default_expr(
                expr.value,
                path_prefix=path_prefix,
                default_base_path=default_base_path,
            ),
            stages=tuple(
                _scoped_default_expr(
                    stage,
                    path_prefix=path_prefix,
                    default_base_path=default_base_path,
                )
                for stage in expr.stages
            ),
        )
    if isinstance(expr, AxonExprBind):
        return replace(
            expr,
            value=_scoped_default_expr(
                expr.value,
                path_prefix=path_prefix,
                default_base_path=default_base_path,
            ),
            body=_scoped_default_expr(
                expr.body,
                path_prefix=path_prefix,
                default_base_path=default_base_path,
            ),
        )
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return replace(
            expr,
            cond=_scoped_default_expr(
                expr.cond,
                path_prefix=path_prefix,
                default_base_path=default_base_path,
            ),
            true_expr=_scoped_default_expr(
                expr.true_expr,
                path_prefix=path_prefix,
                default_base_path=default_base_path,
            ),
            false_expr=_scoped_default_expr(
                expr.false_expr,
                path_prefix=path_prefix,
                default_base_path=default_base_path,
            ),
        )
    if isinstance(expr, AxonExprBinary):
        return replace(
            expr,
            left=_scoped_default_expr(
                expr.left,
                path_prefix=path_prefix,
                default_base_path=default_base_path,
            ),
            right=_scoped_default_expr(
                expr.right,
                path_prefix=path_prefix,
                default_base_path=default_base_path,
            ),
        )
    if isinstance(expr, AxonExprLambda):
        return replace(
            expr,
            body=_scoped_default_expr(
                expr.body,
                path_prefix=path_prefix,
                default_base_path=default_base_path,
            ),
        )
    if isinstance(expr, AxonExprParen):
        return replace(
            expr,
            inner=_scoped_default_expr(
                expr.inner,
                path_prefix=path_prefix,
                default_base_path=default_base_path,
            ),
        )
    if isinstance(expr, AxonExprAscribe):
        return replace(
            expr,
            expr=_scoped_default_expr(
                expr.expr,
                path_prefix=path_prefix,
                default_base_path=default_base_path,
            ),
        )
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return replace(
            expr,
            items=tuple(
                _scoped_default_expr(
                    item,
                    path_prefix=path_prefix,
                    default_base_path=default_base_path,
                )
                for item in expr.items
            ),
        )
    if isinstance(expr, AxonExprDo):
        return replace(
            expr,
            body=tuple(
                _scoped_default_statement(
                    stmt,
                    path_prefix=path_prefix,
                    default_base_path=default_base_path,
                )
                for stmt in expr.body
            ),
        )
    return expr


def _scoped_default_statement(
    stmt: AxonStatement,
    *,
    path_prefix: tuple[str, ...],
    default_base_path: AxonExprPath | None,
) -> AxonStatement:
    if isinstance(stmt, AxonBind):
        return replace(
            stmt,
            expr=_scoped_default_expr(
                stmt.expr,
                path_prefix=path_prefix,
                default_base_path=default_base_path,
            ),
        )
    if isinstance(stmt, AxonReturn | AxonYield):
        return replace(
            stmt,
            values=tuple(
                _scoped_default_expr(
                    value,
                    path_prefix=path_prefix,
                    default_base_path=default_base_path,
                )
                for value in stmt.values
            ),
        )
    if isinstance(stmt, AxonCond):
        return replace(
            stmt,
            cond=_scoped_default_expr(
                stmt.cond,
                path_prefix=path_prefix,
                default_base_path=default_base_path,
            ),
            true_body=tuple(
                _scoped_default_statement(
                    item,
                    path_prefix=path_prefix,
                    default_base_path=default_base_path,
                )
                for item in stmt.true_body
            ),
            false_body=tuple(
                _scoped_default_statement(
                    item,
                    path_prefix=path_prefix,
                    default_base_path=default_base_path,
                )
                for item in stmt.false_body
            ),
        )
    if isinstance(stmt, AxonRepeat):
        return replace(
            stmt,
            to_expr=_scoped_default_expr(
                stmt.to_expr,
                path_prefix=path_prefix,
                default_base_path=default_base_path,
            ),
            from_expr=_scoped_default_expr(
                stmt.from_expr,
                path_prefix=path_prefix,
                default_base_path=default_base_path,
            ),
            step_expr=_scoped_default_expr(
                stmt.step_expr,
                path_prefix=path_prefix,
                default_base_path=default_base_path,
            ),
            body=tuple(
                _scoped_default_statement(
                    item,
                    path_prefix=path_prefix,
                    default_base_path=default_base_path,
                )
                for item in stmt.body
            ),
        )
    if isinstance(stmt, AxonScopeBind):
        return replace(
            stmt,
            kwargs={
                key: _scoped_default_expr(
                    value,
                    path_prefix=path_prefix,
                    default_base_path=default_base_path,
                )
                if isinstance(value, AxonExpr)
                else deepcopy(value)
                for key, value in stmt.kwargs.items()
            },
            body=tuple(
                _scoped_default_statement(
                    item,
                    path_prefix=path_prefix,
                    default_base_path=default_base_path,
                )
                for item in stmt.body
            ),
        )
    return stmt


def _expr_has_defaults(expr: AxonExpr, *, modules_by_name: dict[str, AxonDefinition]) -> bool:
    if isinstance(expr, AxonExprCall):
        module = modules_by_name.get(expr.callee)
        if module is not None:
            path_slots = _path_slot_count(module)
            provided_params = set(expr.kwargs)
            positional_params = max(0, len(expr.args) - path_slots)
            if positional_params < len(module.params):
                for param in module.params[positional_params:]:
                    if param.name not in provided_params and (
                        param.default_expr is not None or param.optional
                    ):
                        return True
        return any(_expr_has_defaults(arg, modules_by_name=modules_by_name) for arg in expr.args) or any(
            _expr_has_defaults(value, modules_by_name=modules_by_name)
            for value in expr.kwargs.values()
            if isinstance(value, AxonExpr)
        )
    if isinstance(expr, AxonExprPipe):
        return _expr_has_defaults(expr.value, modules_by_name=modules_by_name) or any(
            _expr_has_defaults(stage, modules_by_name=modules_by_name) for stage in expr.stages
        )
    if isinstance(expr, AxonExprBind):
        return _expr_has_defaults(expr.value, modules_by_name=modules_by_name) or _expr_has_defaults(
            expr.body, modules_by_name=modules_by_name
        )
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return (
            _expr_has_defaults(expr.cond, modules_by_name=modules_by_name)
            or _expr_has_defaults(expr.true_expr, modules_by_name=modules_by_name)
            or _expr_has_defaults(expr.false_expr, modules_by_name=modules_by_name)
        )
    if isinstance(expr, AxonExprBinary):
        return _expr_has_defaults(expr.left, modules_by_name=modules_by_name) or _expr_has_defaults(
            expr.right, modules_by_name=modules_by_name
        )
    if isinstance(expr, AxonExprLambda):
        return _expr_has_defaults(expr.body, modules_by_name=modules_by_name)
    if isinstance(expr, AxonExprParen):
        return _expr_has_defaults(expr.inner, modules_by_name=modules_by_name)
    if isinstance(expr, AxonExprAscribe):
        return _expr_has_defaults(expr.expr, modules_by_name=modules_by_name)
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return any(_expr_has_defaults(item, modules_by_name=modules_by_name) for item in expr.items)
    if isinstance(expr, AxonExprDo):
        return _statements_have_defaults(expr.body, modules_by_name=modules_by_name)
    return False


def _statement_has_defaults(
    stmt: AxonStatement, *, modules_by_name: dict[str, AxonDefinition]
) -> bool:
    if isinstance(stmt, AxonBind):
        return _expr_has_defaults(stmt.expr, modules_by_name=modules_by_name)
    if isinstance(stmt, AxonReturn | AxonYield):
        return any(_expr_has_defaults(value, modules_by_name=modules_by_name) for value in stmt.values)
    if isinstance(stmt, AxonCond):
        return (
            _expr_has_defaults(stmt.cond, modules_by_name=modules_by_name)
            or _statements_have_defaults(stmt.true_body, modules_by_name=modules_by_name)
            or _statements_have_defaults(stmt.false_body, modules_by_name=modules_by_name)
        )
    if isinstance(stmt, AxonRepeat):
        return (
            _expr_has_defaults(stmt.to_expr, modules_by_name=modules_by_name)
            or _expr_has_defaults(stmt.from_expr, modules_by_name=modules_by_name)
            or _expr_has_defaults(stmt.step_expr, modules_by_name=modules_by_name)
            or _statements_have_defaults(stmt.body, modules_by_name=modules_by_name)
        )
    if isinstance(stmt, AxonScopeBind):
        return any(
            _expr_has_defaults(value, modules_by_name=modules_by_name)
            for value in stmt.kwargs.values()
            if isinstance(value, AxonExpr)
        ) or _statements_have_defaults(stmt.body, modules_by_name=modules_by_name)
    return False


def _statements_have_defaults(
    statements: tuple[AxonStatement, ...], *, modules_by_name: dict[str, AxonDefinition]
) -> bool:
    return any(_statement_has_defaults(stmt, modules_by_name=modules_by_name) for stmt in statements)


def _copy_default_expr(
    expr: AxonExpr,
    *,
    path_prefix: tuple[str, ...],
    path_names: set[str],
    default_base_path: AxonExprPath | None,
    modules_by_name: dict[str, AxonDefinition],
) -> AxonExpr:
    copied = deepcopy(expr)
    scoped = _scoped_default_expr(
        copied,
        path_prefix=path_prefix,
        default_base_path=default_base_path,
    )
    return _elaborate_expr(
        scoped,
        modules_by_name=modules_by_name,
        path_prefix=path_prefix,
        path_names=path_names,
    )


def _default_expr_for_param(
    param: AxonParam,
    *,
    path_prefix: tuple[str, ...],
    path_names: set[str],
    default_base_path: AxonExprPath | None,
    modules_by_name: dict[str, AxonDefinition],
) -> AxonExpr:
    if param.default_expr is None:
        return _copy_default_expr(
            AxonExprNull(),
            path_prefix=path_prefix,
            path_names=path_names,
            default_base_path=default_base_path,
            modules_by_name=modules_by_name,
        )
    return _copy_default_expr(
        param.default_expr,
        path_prefix=path_prefix,
        path_names=path_names,
        default_base_path=default_base_path,
        modules_by_name=modules_by_name,
    )


def _call_default_base_path(
    call: AxonExprCall,
    *,
    module: AxonDefinition,
    path_prefix: tuple[str, ...],
    path_names: set[str],
) -> AxonExprPath | None:
    path_slots = _path_slot_count(module)
    for arg in call.args[:path_slots]:
        if isinstance(arg, AxonExprPath):
            if not arg.absolute and len(arg.parts) == 1 and arg.parts[0] in path_names:
                return AxonExprPath(absolute=True, parts=(f"{{{arg.parts[0]}}}",))
            return arg
        if isinstance(arg, AxonExprName):
            return AxonExprPath(absolute=True, parts=(f"{{{arg.name}}}",))
    return None


def _elaborate_call(
    expr: AxonExprCall,
    *,
    modules_by_name: dict[str, AxonDefinition],
    path_prefix: tuple[str, ...],
    path_names: set[str],
) -> AxonExprCall:
    args = tuple(
        _elaborate_expr(
            arg, modules_by_name=modules_by_name, path_prefix=path_prefix, path_names=path_names
        )
        for arg in expr.args
    )
    kwargs: dict[str, AxonKwargValue] = {
        key: _elaborate_expr(
            value, modules_by_name=modules_by_name, path_prefix=path_prefix, path_names=path_names
        )
        if isinstance(value, AxonExpr)
        else deepcopy(value)
        for key, value in expr.kwargs.items()
    }
    module = modules_by_name.get(expr.callee)
    if module is None:
        return replace(expr, args=args, kwargs=kwargs)

    path_slots = _path_slot_count(module)
    positional_params = max(0, len(args) - path_slots)
    default_base_path = _call_default_base_path(
        replace(expr, args=args, kwargs=kwargs),
        module=module,
        path_prefix=path_prefix,
        path_names=path_names,
    )
    for param in module.params[positional_params:]:
        if param.name in kwargs:
            continue
        if param.default_expr is None and not param.optional:
            continue
        kwargs[param.name] = _default_expr_for_param(
            param,
            path_prefix=path_prefix,
            path_names=path_names,
            default_base_path=default_base_path,
            modules_by_name=modules_by_name,
        )
    return replace(expr, args=args, kwargs=kwargs)


def _elaborate_expr(
    expr: AxonExpr,
    *,
    modules_by_name: dict[str, AxonDefinition],
    path_prefix: tuple[str, ...],
    path_names: set[str],
) -> AxonExpr:
    if isinstance(expr, AxonExprCall):
        return _elaborate_call(
            expr,
            modules_by_name=modules_by_name,
            path_prefix=path_prefix,
            path_names=path_names,
        )
    if isinstance(expr, AxonExprPipe):
        return replace(
            expr,
            value=_elaborate_expr(
                expr.value,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
            stages=tuple(
                _elaborate_expr(
                    stage,
                    modules_by_name=modules_by_name,
                    path_prefix=path_prefix,
                    path_names=path_names,
                )
                for stage in expr.stages
            ),
        )
    if isinstance(expr, AxonExprBind):
        return replace(
            expr,
            value=_elaborate_expr(
                expr.value,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
            body=_elaborate_expr(
                expr.body,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
        )
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return replace(
            expr,
            cond=_elaborate_expr(
                expr.cond,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
            true_expr=_elaborate_expr(
                expr.true_expr,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
            false_expr=_elaborate_expr(
                expr.false_expr,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
        )
    if isinstance(expr, AxonExprBinary):
        return replace(
            expr,
            left=_elaborate_expr(
                expr.left,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
            right=_elaborate_expr(
                expr.right,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
        )
    if isinstance(expr, AxonExprLambda):
        return replace(
            expr,
            body=_elaborate_expr(
                expr.body,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
        )
    if isinstance(expr, AxonExprParen):
        return replace(
            expr,
            inner=_elaborate_expr(
                expr.inner,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
        )
    if isinstance(expr, AxonExprAscribe):
        return replace(
            expr,
            expr=_elaborate_expr(
                expr.expr,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
        )
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return replace(
            expr,
            items=tuple(
                _elaborate_expr(
                    item,
                    modules_by_name=modules_by_name,
                    path_prefix=path_prefix,
                    path_names=path_names,
                )
                for item in expr.items
            ),
        )
    if isinstance(expr, AxonExprDo):
        return replace(
            expr,
            body=_elaborate_statements(
                expr.body,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
        )
    return expr


def _elaborate_statement(
    stmt: AxonStatement,
    *,
    modules_by_name: dict[str, AxonDefinition],
    path_prefix: tuple[str, ...],
    path_names: set[str],
) -> AxonStatement:
    if isinstance(stmt, AxonBind):
        return replace(
            stmt,
            expr=_elaborate_expr(
                stmt.expr,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
        )
    if isinstance(stmt, AxonReturn | AxonYield):
        return replace(
            stmt,
            values=tuple(
                _elaborate_expr(
                    value,
                    modules_by_name=modules_by_name,
                    path_prefix=path_prefix,
                    path_names=path_names,
                )
                for value in stmt.values
            ),
        )
    if isinstance(stmt, AxonCond):
        return replace(
            stmt,
            cond=_elaborate_expr(
                stmt.cond,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
            true_body=_elaborate_statements(
                stmt.true_body,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
            false_body=_elaborate_statements(
                stmt.false_body,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
        )
    if isinstance(stmt, AxonRepeat):
        loop_prefix = (*path_prefix, *(_repeat_scope_parts(stmt) if stmt.name else ()))
        return replace(
            stmt,
            to_expr=_elaborate_expr(
                stmt.to_expr,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
            from_expr=_elaborate_expr(
                stmt.from_expr,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
            step_expr=_elaborate_expr(
                stmt.step_expr,
                modules_by_name=modules_by_name,
                path_prefix=path_prefix,
                path_names=path_names,
            ),
            body=_elaborate_statements(
                stmt.body,
                modules_by_name=modules_by_name,
                path_prefix=loop_prefix,
                path_names=path_names,
            ),
        )
    if isinstance(stmt, AxonScopeBind):
        return replace(
            stmt,
            kwargs={
                key: _elaborate_expr(
                    value,
                    modules_by_name=modules_by_name,
                    path_prefix=path_prefix,
                    path_names=path_names,
                )
                if isinstance(value, AxonExpr)
                else deepcopy(value)
                for key, value in stmt.kwargs.items()
            },
            body=_elaborate_statements(
                stmt.body,
                modules_by_name=modules_by_name,
                path_prefix=(*path_prefix, *stmt.prefix.parts),
                path_names=path_names,
            ),
        )
    return stmt


def _repeat_scope_parts(stmt: AxonRepeat) -> tuple[str, ...]:
    if stmt.name is None:
        return ()
    return (*tuple(part for part in stmt.name.split(".") if part), f"{{{stmt.var}}}")


def _elaborate_statements(
    statements: tuple[AxonStatement, ...],
    *,
    modules_by_name: dict[str, AxonDefinition],
    path_prefix: tuple[str, ...],
    path_names: set[str],
) -> tuple[AxonStatement, ...]:
    return tuple(
        _elaborate_statement(
            stmt,
            modules_by_name=modules_by_name,
            path_prefix=path_prefix,
            path_names=path_names,
        )
        for stmt in statements
    )


def _strip_param_defaults(params: tuple[AxonParam, ...]) -> tuple[AxonParam, ...]:
    return tuple(replace(param, default_expr=None) for param in params)


def _elaborate_module(
    module: AxonDefinition,
    *,
    modules_by_name: dict[str, AxonDefinition],
) -> AxonDefinition:
    path_names = {
        *module.path_params,
        *(param.name for param in module.params if isinstance(param.type_expr, TypePath)),
    }
    return replace(
        module,
        params=_strip_param_defaults(module.params),
        statements=_elaborate_statements(
            module.statements,
            modules_by_name=modules_by_name,
            path_prefix=(),
            path_names=path_names,
        ),
        body_expr=(
            _elaborate_expr(
                module.body_expr,
                modules_by_name=modules_by_name,
                path_prefix=(),
                path_names=path_names,
            )
            if module.body_expr is not None
            else None
        ),
    )


def elaborate_closed_axon_file(
    program: AxonFile, *, main_module: str | None = None
) -> AxonFile:
    validate_normalized_axon_file(program, main_module=main_module)
    modules_by_name = {module.name: module for module in program.modules}
    current = program
    while True:
        elaborated = replace(
            current,
            modules=tuple(
                _elaborate_module(module, modules_by_name=modules_by_name)
                for module in current.modules
            ),
        )
        modules_by_name = {module.name: module for module in elaborated.modules}
        if not any(
            param.default_expr is not None for module in elaborated.modules for param in module.params
        ) and not any(
            _statements_have_defaults(module.statements, modules_by_name=modules_by_name)
            or (
                module.body_expr is not None
                and _expr_has_defaults(module.body_expr, modules_by_name=modules_by_name)
            )
            for module in elaborated.modules
        ):
            validate_normalized_axon_file(elaborated, main_module=main_module)
            return elaborated
        current = elaborated


__all__ = ["elaborate_closed_axon_file"]
