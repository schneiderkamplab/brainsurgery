from __future__ import annotations

from dataclasses import replace

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
    AxonExprNull,
    AxonExprParen,
    AxonExprPath,
    AxonExprPipe,
    AxonExprTernary,
    AxonExprTuple,
    AxonFile,
    AxonKwargValue,
    AxonModule,
    AxonParam,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
    AxonYield,
    TypePath,
)
from ..validate import validate_closed_axon_file, validate_normalized_axon_file


def _split_callee_path_sugar(callee: str) -> tuple[str, tuple[AxonExprPath, ...]]:
    if "@" not in callee:
        return callee, ()
    parts = callee.split("@")
    base = parts[0]
    suffixes = parts[1:]
    path_args: list[AxonExprPath] = []
    for suffix in suffixes:
        if not suffix:
            raise ValueError(f"normalize failed: invalid callee path sugar {callee!r}")
        path_args.append(AxonExprPath(absolute=False, parts=tuple(suffix.split("."))))
    return base, tuple(path_args)


def _leading_path_param_count(module: AxonModule) -> int:
    count = 0
    for param in module.params:
        if isinstance(param.type_expr, TypePath):
            count += 1
            continue
        break
    return count


def _default_value_expr(module: AxonModule, param_name: str) -> AxonExpr:
    param = next((item for item in module.params if item.name == param_name), None)
    if param is None:
        raise ValueError(
            f"normalize failed: unknown param {param_name!r} for module {module.name!r}"
        )
    if param.default_expr is not None:
        return param.default_expr
    if param.optional:
        return AxonExprNull()
    raise ValueError(
        f"normalize failed: missing required arg {param_name!r} for call to {module.name!r}"
    )


def _expand_call_surface(
    expr: AxonExprCall, *, modules_by_name: dict[str, AxonModule]
) -> AxonExprCall:
    base_callee, sugared_path_args = _split_callee_path_sugar(expr.callee)
    explicit_args = [*sugared_path_args, *expr.args]
    module = modules_by_name.get(base_callee)
    if module is None:
        if base_callee == expr.callee:
            return replace(expr, args=tuple(explicit_args), kwargs=dict(expr.kwargs))
        return AxonExprCall(callee=base_callee, args=tuple(explicit_args), kwargs=dict(expr.kwargs))

    path_slot_count = len(module.path_params) + _leading_path_param_count(module)
    if len(sugared_path_args) > path_slot_count:
        raise ValueError(
            f"normalize failed: too many path args in call {expr.callee!r} for module {module.name!r}"
        )

    original_kwargs = dict(expr.kwargs)
    if len(explicit_args) < path_slot_count:
        path_param_names = list(module.path_params)
        path_param_names.extend(
            param.name for param in module.params[: _leading_path_param_count(module)]
        )
        for name in path_param_names[len(explicit_args) : path_slot_count]:
            value = original_kwargs.get(name)
            if not isinstance(value, AxonExpr):
                break
            explicit_args.append(value)
            original_kwargs.pop(name, None)

    if len(explicit_args) < path_slot_count:
        raise ValueError(
            f"normalize failed: missing path args in call {expr.callee!r} for module {module.name!r}"
        )

    kwargs: dict[str, AxonKwargValue] = {}
    provided_positional = max(0, len(explicit_args) - len(module.path_params))
    known_param_names = {param.name for param in module.params}
    for idx, param in enumerate(module.params):
        if idx < provided_positional:
            continue
        if param.name in original_kwargs:
            kwargs[param.name] = original_kwargs.pop(param.name)
            continue
        if param.optional or param.default_expr is not None:
            kwargs[param.name] = _default_value_expr(module, param.name)

    for key, value in original_kwargs.items():
        if key not in known_param_names:
            kwargs[key] = value

    return AxonExprCall(callee=base_callee, args=tuple(explicit_args), kwargs=kwargs)


def _pipe_stage_to_call(value: AxonExpr, stage: AxonExpr) -> AxonExpr:
    if isinstance(stage, AxonExprName):
        return AxonExprCall(callee=stage.name, args=(value,), kwargs={})
    if isinstance(stage, AxonExprCall):
        return AxonExprCall(
            callee=stage.callee, args=(value, *stage.args), kwargs=dict(stage.kwargs)
        )
    raise ValueError("normalize failed: pipeline stage must be a name or call")


def _normalize_expr(expr: AxonExpr, *, modules_by_name: dict[str, AxonModule]) -> AxonExpr:
    if isinstance(expr, AxonExprCall):
        args = tuple(_normalize_expr(arg, modules_by_name=modules_by_name) for arg in expr.args)
        kwargs = {
            key: _normalize_expr(value, modules_by_name=modules_by_name)
            if isinstance(value, AxonExpr)
            else value
            for key, value in expr.kwargs.items()
        }
        return _expand_call_surface(
            replace(expr, args=args, kwargs=kwargs), modules_by_name=modules_by_name
        )
    if isinstance(expr, AxonExprPipe):
        current = _normalize_expr(expr.value, modules_by_name=modules_by_name)
        for stage in expr.stages:
            current = _normalize_expr(
                _pipe_stage_to_call(current, stage), modules_by_name=modules_by_name
            )
        return current
    if isinstance(expr, AxonExprBind):
        return replace(
            expr,
            value=_normalize_expr(expr.value, modules_by_name=modules_by_name),
            body=_normalize_expr(expr.body, modules_by_name=modules_by_name),
        )
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return replace(
            expr,
            cond=_normalize_expr(expr.cond, modules_by_name=modules_by_name),
            true_expr=_normalize_expr(expr.true_expr, modules_by_name=modules_by_name),
            false_expr=_normalize_expr(expr.false_expr, modules_by_name=modules_by_name),
        )
    if isinstance(expr, AxonExprBinary):
        return replace(
            expr,
            left=_normalize_expr(expr.left, modules_by_name=modules_by_name),
            right=_normalize_expr(expr.right, modules_by_name=modules_by_name),
        )
    if isinstance(expr, AxonExprLambda):
        return replace(expr, body=_normalize_expr(expr.body, modules_by_name=modules_by_name))
    if isinstance(expr, AxonExprParen):
        return replace(expr, inner=_normalize_expr(expr.inner, modules_by_name=modules_by_name))
    if isinstance(expr, AxonExprAscribe):
        return replace(expr, expr=_normalize_expr(expr.expr, modules_by_name=modules_by_name))
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return replace(
            expr,
            items=tuple(_normalize_expr(item, modules_by_name=modules_by_name) for item in expr.items),
        )
    if isinstance(expr, AxonExprDo):
        return replace(
            expr,
            body=tuple(_normalize_statement(stmt, modules_by_name=modules_by_name) for stmt in expr.body),
        )
    return expr


def _normalize_repeat_yield(stmt: AxonRepeat) -> AxonRepeat:
    if stmt.body and isinstance(stmt.body[-1], AxonYield):
        if stmt.carry is not None:
            return stmt
        if stmt.targets is not None:
            return replace(stmt, carry=stmt.targets)
        return stmt
    if stmt.targets is None:
        return replace(stmt, body=(*stmt.body, AxonYield(values=(AxonExprNull(),))))
    normalized_body = tuple(
        [
            *stmt.body,
            AxonYield(values=tuple(AxonExprName(name=name) for name in stmt.targets)),
        ]
    )
    return replace(
        stmt,
        body=normalized_body,
        carry=stmt.targets if stmt.carry is None else stmt.carry,
    )


def _normalize_statement(
    stmt: AxonStatement, *, modules_by_name: dict[str, AxonModule]
) -> AxonStatement:
    if isinstance(stmt, AxonBind):
        return replace(stmt, expr=_normalize_expr(stmt.expr, modules_by_name=modules_by_name))
    if isinstance(stmt, AxonReturn | AxonYield):
        return replace(
            stmt,
            values=tuple(
                _normalize_expr(value, modules_by_name=modules_by_name) for value in stmt.values
            ),
        )
    if isinstance(stmt, AxonCond):
        return replace(
            stmt,
            cond=_normalize_expr(stmt.cond, modules_by_name=modules_by_name),
            true_body=tuple(
                _normalize_statement(item, modules_by_name=modules_by_name)
                for item in stmt.true_body
            ),
            false_body=tuple(
                _normalize_statement(item, modules_by_name=modules_by_name)
                for item in stmt.false_body
            ),
        )
    if isinstance(stmt, AxonRepeat):
        normalized = replace(
            stmt,
            from_expr=_normalize_expr(stmt.from_expr, modules_by_name=modules_by_name),
            to_expr=_normalize_expr(stmt.to_expr, modules_by_name=modules_by_name),
            step_expr=_normalize_expr(stmt.step_expr, modules_by_name=modules_by_name),
            body=tuple(
                _normalize_statement(item, modules_by_name=modules_by_name) for item in stmt.body
            ),
        )
        return _normalize_repeat_yield(normalized)
    if isinstance(stmt, AxonScopeBind):
        return replace(
            stmt,
            kwargs={
                key: _normalize_expr(value, modules_by_name=modules_by_name)
                if isinstance(value, AxonExpr)
                else value
                for key, value in stmt.kwargs.items()
            },
            body=tuple(
                _normalize_statement(item, modules_by_name=modules_by_name) for item in stmt.body
            ),
        )
    return stmt


def _normalize_module(module: AxonModule, *, modules_by_name: dict[str, AxonModule]) -> AxonModule:
    return replace(
        module,
        params=tuple(
            replace(
                param,
                default_expr=_normalize_expr(param.default_expr, modules_by_name=modules_by_name)
                if param.default_expr is not None
                else None,
            )
            for param in module.params
        ),
        statements=tuple(
            _normalize_statement(stmt, modules_by_name=modules_by_name)
            for stmt in module.statements
        ),
        body_expr=_normalize_expr(module.body_expr, modules_by_name=modules_by_name)
        if module.body_expr is not None
        else None,
    )


def normalize_closed_axon_file(program: AxonFile, *, main_module: str | None = None) -> AxonFile:
    validate_closed_axon_file(program, main_module=main_module)
    modules_by_name = {module.name: module for module in program.modules}
    normalized_modules = tuple(
        _normalize_module(module, modules_by_name=modules_by_name) for module in program.modules
    )
    normalized = replace(
        program,
        modules=normalized_modules,
        constants={
            name: _normalize_expr(expr, modules_by_name=modules_by_name)
            for name, expr in program.constants.items()
        },
    )
    validate_normalized_axon_file(normalized, main_module=main_module)
    return normalized


__all__ = ["normalize_closed_axon_file"]
