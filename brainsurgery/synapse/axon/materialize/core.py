from __future__ import annotations

import math
from dataclasses import replace
from typing import cast

from ..ast import (
    AxonBind,
    AxonExpr,
    AxonExprBinary,
    AxonExprBind,
    AxonExprBool,
    AxonExprCall,
    AxonExprDo,
    AxonExprFloat,
    AxonExprIf,
    AxonExprInt,
    AxonExprLambda,
    AxonExprList,
    AxonExprName,
    AxonExprNull,
    AxonExprParen,
    AxonExprPath,
    AxonExprPipe,
    AxonExprString,
    AxonExprTernary,
    AxonExprTuple,
    AxonFile,
    AxonKwargValue,
    AxonDefinition,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
    AxonYield,
    parse_path_token,
    resolve_path_expr_to_key,
)
from .context import MaterializeContext

_NOT_EVALUABLE = object()
_Number = int | float


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_number(value: object) -> _Number:
    if not _is_number(value):
        raise TypeError(f"Expected numeric value, got {type(value).__name__}")
    return cast(_Number, value)


def _expr_from_scalar(value: object) -> AxonExpr:
    if value is None:
        return AxonExprNull()
    if isinstance(value, bool):
        return AxonExprBool(value=value)
    if isinstance(value, int) and not isinstance(value, bool):
        return AxonExprInt(value=value)
    if isinstance(value, float):
        return AxonExprFloat(value=value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("@"):
            return parse_path_token(stripped, op_name="materialization path")
        return AxonExprString(value=value)
    if isinstance(value, list):
        return AxonExprList(items=tuple(_expr_from_scalar(item) for item in value))
    if isinstance(value, tuple):
        return AxonExprTuple(items=tuple(_expr_from_scalar(item) for item in value))
    raise ValueError(f"Unsupported literal value for materialization: {value!r}")


def _expr_config_lookup(config: dict[str, object], key: str) -> tuple[bool, object | None]:
    value: object = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _params_has_root(keys: frozenset[str], root: str) -> bool:
    if root == "":
        return True
    prefix = f"{root}."
    return any(key == root or key.startswith(prefix) for key in keys)


def _resolve_config_path_key(raw: object, env: dict[str, object], op_name: str) -> str:
    return resolve_path_expr_to_key(raw, env, op_name=op_name)


def _eval_expr(
    expr: AxonExpr,
    *,
    env: dict[str, object],
    ctx: MaterializeContext,
    resolve_names: bool = True,
) -> object:
    if isinstance(expr, AxonExprName):
        if expr.name in env:
            return env[expr.name]
        if "@" in expr.name:
            return _eval_expr(
                AxonExprCall(callee=expr.name, args=(), kwargs={}),
                env=env,
                ctx=ctx,
                resolve_names=resolve_names,
            )
        if not resolve_names:
            raise ValueError(f"name resolution disabled: {expr.name}")
        raise ValueError(f"unknown runtime name: {expr.name}")
    if isinstance(expr, AxonExprInt):
        return expr.value
    if isinstance(expr, AxonExprFloat):
        return expr.value
    if isinstance(expr, AxonExprBool):
        return expr.value
    if isinstance(expr, AxonExprNull):
        return None
    if isinstance(expr, AxonExprString):
        return expr.value
    if isinstance(expr, AxonExprPath):
        return expr.to_source()
    if isinstance(expr, AxonExprList):
        return [
            _eval_expr(item, env=env, ctx=ctx, resolve_names=resolve_names) for item in expr.items
        ]
    if isinstance(expr, AxonExprTuple):
        return tuple(
            _eval_expr(item, env=env, ctx=ctx, resolve_names=resolve_names) for item in expr.items
        )
    if isinstance(expr, AxonExprParen):
        return _eval_expr(expr.inner, env=env, ctx=ctx, resolve_names=resolve_names)
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        cond = _eval_expr(expr.cond, env=env, ctx=ctx, resolve_names=resolve_names)
        branch = expr.true_expr if bool(cond) else expr.false_expr
        return _eval_expr(branch, env=env, ctx=ctx, resolve_names=resolve_names)
    if isinstance(expr, AxonExprBinary):
        left = _eval_expr(expr.left, env=env, ctx=ctx, resolve_names=resolve_names)
        right = _eval_expr(expr.right, env=env, ctx=ctx, resolve_names=resolve_names)
        op = expr.op
        if op == "+":
            if isinstance(left, str) and isinstance(right, str):
                return left + right
            if _is_number(left) and _is_number(right):
                return _as_number(left) + _as_number(right)
            raise ValueError("binary '+' expects numeric or string operands")
        if op == "-":
            if not _is_number(left) or not _is_number(right):
                raise ValueError("binary '-' expects numeric operands")
            return _as_number(left) - _as_number(right)
        if op == "*":
            if not _is_number(left) or not _is_number(right):
                raise ValueError("binary '*' expects numeric operands")
            return _as_number(left) * _as_number(right)
        if op == "/":
            if not _is_number(left) or not _is_number(right):
                raise ValueError("binary '/' expects numeric operands")
            left_number = _as_number(left)
            right_number = _as_number(right)
            if isinstance(left_number, int) and isinstance(right_number, int):
                if right_number != 0 and left_number % right_number == 0:
                    return left_number // right_number
            return left_number / right_number
        if op == "%":
            if not _is_number(left) or not _is_number(right):
                raise ValueError("binary '%' expects numeric operands")
            return _as_number(left) % _as_number(right)
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return _as_number(left) < _as_number(right)
        if op == "<=":
            return _as_number(left) <= _as_number(right)
        if op == ">":
            return _as_number(left) > _as_number(right)
        if op == ">=":
            return _as_number(left) >= _as_number(right)
        if op == "and":
            return bool(left) and bool(right)
        if op == "or":
            return bool(left) or bool(right)
        raise ValueError(f"unsupported binary operator {op!r}")
    if isinstance(expr, AxonExprCall):
        raw_callee = expr.callee.strip()
        is_absolute_path = "@@" in raw_callee
        parse_callee = raw_callee.replace("@@", "@", 1) if is_absolute_path else raw_callee
        callee_parts = parse_callee.split("@")
        callee = callee_parts[0]
        callee_paths = callee_parts[1:]

        def _eval_kwarg_value(value: AxonKwargValue) -> object:
            if isinstance(value, AxonExpr):
                return _eval_expr(value, env=env, ctx=ctx, resolve_names=resolve_names)
            return value

        args = [_eval_expr(arg, env=env, ctx=ctx, resolve_names=resolve_names) for arg in expr.args]
        kwargs = {key: _eval_kwarg_value(value) for key, value in expr.kwargs.items()}

        def _path_token_from_suffix(token: str, *, absolute: bool) -> str:
            raw = token.strip()
            if not raw:
                raise ValueError("empty @path suffix is not allowed")
            return ("@@" if absolute else "@") + raw

        if callee.startswith("Config.") and callee_paths:
            if args:
                raise ValueError(
                    f"{callee} does not allow both positional key and @path suffix key"
                )
            if len(callee_paths) != 1:
                raise ValueError(f"{callee} expects exactly one @path suffix key")
            args = [_path_token_from_suffix(callee_paths[0], absolute=is_absolute_path)]

        if callee in {
            "sqrt",
            "Prelude.sqrt",
            "Math.sqrt",
            "log",
            "Prelude.log",
            "Math.log",
            "exp",
            "Prelude.exp",
            "Math.exp",
            "sin",
            "Prelude.sin",
            "Math.sin",
            "cos",
            "Prelude.cos",
            "Math.cos",
        }:
            if len(args) != 1 or not _is_number(args[0]):
                raise ValueError(f"{callee} expects one numeric argument")
            arg = float(_as_number(args[0]))
            fn_name = callee.split(".", 1)[-1]
            if fn_name == "sqrt":
                return math.sqrt(arg)
            if fn_name == "log":
                return math.log(arg)
            if fn_name == "exp":
                return math.exp(arg)
            if fn_name == "sin":
                return math.sin(arg)
            if fn_name == "cos":
                return math.cos(arg)
            raise ValueError(f"unsupported unary expression call: {callee!r}")
        if callee in {"pow", "Prelude.pow", "Math.pow"}:
            if len(args) != 2 or not _is_number(args[0]) or not _is_number(args[1]):
                raise ValueError("pow expects two numeric arguments")
            return math.pow(float(_as_number(args[0])), float(_as_number(args[1])))
        if callee in {"abs", "Prelude.abs"}:
            if len(args) != 1 or not _is_number(args[0]):
                raise ValueError("abs expects one numeric argument")
            return abs(_as_number(args[0]))
        if callee in {"min", "Prelude.min"}:
            if not args or any(not _is_number(arg) for arg in args):
                raise ValueError("min expects numeric arguments")
            return min(_as_number(arg) for arg in args)
        if callee in {"max", "Prelude.max"}:
            if not args or any(not _is_number(arg) for arg in args):
                raise ValueError("max expects numeric arguments")
            return max(_as_number(arg) for arg in args)

        if callee == "Config.value":
            if kwargs:
                raise ValueError("Config.value does not support kwargs")
            if len(args) < 1 or len(args) > 2:
                raise ValueError("Config.value expects positional arguments: key [, default]")
            key = _resolve_config_path_key(args[0], env, "Config.value")
            has_default = len(args) >= 2
            default_value = args[1] if has_default else None
            found, value = _expr_config_lookup(ctx.config, key)
            if not found:
                if not has_default:
                    raise KeyError(f"missing required config key: {key}")
                value = default_value
            return value

        if callee in {
            "Config.has_key",
            "Config.has_value",
            "Config.int",
            "Config.dim",
            "Config.float",
            "Config.str",
            "Config.bool",
            "Config.list",
        }:
            if len(args) != 1:
                raise ValueError(f"{callee} expects one non-empty Path key")
            if "root" in kwargs:
                raise ValueError(f"{callee} does not support root")
            key = _resolve_config_path_key(args[0], env, callee)
            found, value = _expr_config_lookup(ctx.config, key)
            if callee == "Config.has_key":
                if "default" in kwargs:
                    raise ValueError("Config.has_key does not support default")
                return bool(found)
            if callee == "Config.has_value":
                if "default" in kwargs:
                    raise ValueError("Config.has_value does not support default")
                return bool(found) and value is not None
            if not found:
                if "default" not in kwargs:
                    raise KeyError(f"missing required config key: {key}")
                value = kwargs["default"]
            if callee in {"Config.int", "Config.dim"}:
                if isinstance(value, bool):
                    raise ValueError(f"{callee} expected int")
                if isinstance(value, int):
                    return int(value)
                if isinstance(value, str):
                    raw = value.strip()
                    if raw and (raw.isdigit() or (raw[0] in {"+", "-"} and raw[1:].isdigit())):
                        return int(raw)
                raise ValueError(f"{callee} expected int")
            if callee == "Config.float":
                if isinstance(value, bool):
                    raise ValueError("Config.float expected float")
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str) and value.strip():
                    return float(value.strip())
                raise ValueError("Config.float expected float")
            if callee == "Config.str":
                if not isinstance(value, str):
                    raise ValueError("Config.str expected string")
                return value
            if callee == "Config.bool":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    raw = value.strip().lower()
                    if raw == "true":
                        return True
                    if raw == "false":
                        return False
                raise ValueError("Config.bool expected bool")
            if callee == "Config.list":
                if isinstance(value, list):
                    return value
                if isinstance(value, tuple):
                    return list(value)
                raise ValueError("Config.list expected list")
            raise ValueError(f"Unsupported config expression call: {callee}")

        if callee in {"Params.has_root", "Params.root"}:
            if len(args) != 1 or not isinstance(args[0], str):
                raise ValueError(f"{callee} expects one string root argument")
            root = args[0]
            if callee == "Params.has_root":
                return bool(_params_has_root(ctx.state_keys, root))
            default = kwargs.get("default", "")
            if not isinstance(default, str):
                raise ValueError("Params.root default must resolve to string")
            return root if _params_has_root(ctx.state_keys, root) else default

        raise ValueError(f"unsupported call expression: {callee!r}")
    raise ValueError(f"expression is not statically materializable: {type(expr).__name__}")


def _try_eval_expr(
    expr: AxonExpr,
    *,
    env: dict[str, object],
    ctx: MaterializeContext,
    resolve_names: bool = True,
) -> object:
    try:
        return _eval_expr(expr, env=env, ctx=ctx, resolve_names=resolve_names)
    except Exception:
        return _NOT_EVALUABLE


def _materialize_kwarg_value(
    value: AxonKwargValue,
    *,
    env: dict[str, object],
    ctx: MaterializeContext,
    resolve_names: bool = True,
) -> AxonKwargValue:
    if isinstance(value, AxonExpr):
        return _materialize_expr(value, env=env, ctx=ctx, resolve_names=resolve_names)
    return value


def _materialize_config_call(
    expr: AxonExprCall,
    *,
    env: dict[str, object],
    ctx: MaterializeContext,
    resolve_names: bool,
) -> AxonExpr | None:
    callee_base = expr.callee.split("@", 1)[0]
    if callee_base not in {
        "Config.value",
        "Config.has_key",
        "Config.has_value",
        "Config.int",
        "Config.dim",
        "Config.float",
        "Config.str",
        "Config.bool",
        "Config.list",
    }:
        return None
    materialized_call = AxonExprCall(
        callee=expr.callee,
        args=tuple(
            _materialize_expr(arg, env=env, ctx=ctx, resolve_names=resolve_names)
            for arg in expr.args
        ),
        kwargs={
            key: _materialize_kwarg_value(value, env=env, ctx=ctx, resolve_names=resolve_names)
            for key, value in expr.kwargs.items()
        },
    )
    evaluated = _try_eval_expr(
        materialized_call,
        env=env,
        ctx=ctx,
        resolve_names=resolve_names,
    )
    if evaluated is _NOT_EVALUABLE:
        return materialized_call
    return _expr_from_scalar(evaluated)


def _materialize_expr(
    expr: AxonExpr,
    *,
    env: dict[str, object],
    ctx: MaterializeContext,
    resolve_names: bool = True,
) -> AxonExpr:
    if isinstance(expr, AxonExprParen):
        return AxonExprParen(
            inner=_materialize_expr(expr.inner, env=env, ctx=ctx, resolve_names=resolve_names)
        )
    if isinstance(expr, AxonExprIf):
        return AxonExprIf(
            cond=_materialize_expr(expr.cond, env=env, ctx=ctx, resolve_names=resolve_names),
            true_expr=_materialize_expr(
                expr.true_expr, env=env, ctx=ctx, resolve_names=resolve_names
            ),
            false_expr=_materialize_expr(
                expr.false_expr, env=env, ctx=ctx, resolve_names=resolve_names
            ),
        )
    if isinstance(expr, AxonExprTernary):
        return AxonExprTernary(
            cond=_materialize_expr(expr.cond, env=env, ctx=ctx, resolve_names=resolve_names),
            true_expr=_materialize_expr(
                expr.true_expr, env=env, ctx=ctx, resolve_names=resolve_names
            ),
            false_expr=_materialize_expr(
                expr.false_expr, env=env, ctx=ctx, resolve_names=resolve_names
            ),
        )
    if isinstance(expr, AxonExprBinary):
        return AxonExprBinary(
            op=expr.op,
            left=_materialize_expr(expr.left, env=env, ctx=ctx, resolve_names=resolve_names),
            right=_materialize_expr(expr.right, env=env, ctx=ctx, resolve_names=resolve_names),
        )
    if isinstance(expr, AxonExprList):
        return AxonExprList(
            items=tuple(
                _materialize_expr(item, env=env, ctx=ctx, resolve_names=resolve_names)
                for item in expr.items
            )
        )
    if isinstance(expr, AxonExprTuple):
        return AxonExprTuple(
            items=tuple(
                _materialize_expr(item, env=env, ctx=ctx, resolve_names=resolve_names)
                for item in expr.items
            )
        )
    if isinstance(expr, AxonExprCall):
        config_call = _materialize_config_call(expr, env=env, ctx=ctx, resolve_names=resolve_names)
        if config_call is not None:
            return config_call
        return AxonExprCall(
            callee=expr.callee,
            args=tuple(
                _materialize_expr(arg, env=env, ctx=ctx, resolve_names=resolve_names)
                for arg in expr.args
            ),
            kwargs={
                key: _materialize_kwarg_value(value, env=env, ctx=ctx, resolve_names=resolve_names)
                for key, value in expr.kwargs.items()
            },
        )
    if isinstance(expr, AxonExprPipe):
        return AxonExprPipe(
            value=_materialize_expr(expr.value, env=env, ctx=ctx, resolve_names=resolve_names),
            stages=tuple(
                _materialize_expr(stage, env=env, ctx=ctx, resolve_names=resolve_names)
                for stage in expr.stages
            ),
        )
    if isinstance(expr, AxonExprBind):
        return replace(
            expr,
            value=_materialize_expr(expr.value, env=env, ctx=ctx, resolve_names=resolve_names),
            body=_materialize_expr(expr.body, env=env, ctx=ctx, resolve_names=resolve_names),
        )
    if isinstance(expr, AxonExprLambda):
        return replace(
            expr, body=_materialize_expr(expr.body, env=env, ctx=ctx, resolve_names=resolve_names)
        )
    if isinstance(expr, AxonExprDo):
        return replace(
            expr,
            body=tuple(
                _materialize_statement(stmt, env=env, ctx=ctx, resolve_names=resolve_names)
                for stmt in expr.body
            ),
        )
    if isinstance(expr, AxonExprName) and "@" in expr.name:
        evaluated = _try_eval_expr(expr, env=env, ctx=ctx, resolve_names=resolve_names)
        if evaluated is not _NOT_EVALUABLE:
            return _expr_from_scalar(evaluated)
    return expr


def _materialize_statement(
    stmt: AxonStatement,
    *,
    env: dict[str, object],
    ctx: MaterializeContext,
    resolve_names: bool = True,
) -> AxonStatement:
    if isinstance(stmt, AxonBind):
        return replace(
            stmt,
            expr=_materialize_expr(stmt.expr, env=env, ctx=ctx, resolve_names=resolve_names),
        )
    if isinstance(stmt, AxonReturn):
        return replace(
            stmt,
            values=tuple(
                _materialize_expr(value, env=env, ctx=ctx, resolve_names=resolve_names)
                for value in stmt.values
            ),
        )
    if isinstance(stmt, AxonYield):
        return replace(
            stmt,
            values=tuple(
                _materialize_expr(value, env=env, ctx=ctx, resolve_names=resolve_names)
                for value in stmt.values
            ),
        )
    if isinstance(stmt, AxonRepeat):
        return replace(
            stmt,
            to_expr=_materialize_expr(stmt.to_expr, env=env, ctx=ctx, resolve_names=resolve_names),
            from_expr=_materialize_expr(
                stmt.from_expr, env=env, ctx=ctx, resolve_names=resolve_names
            ),
            step_expr=_materialize_expr(
                stmt.step_expr, env=env, ctx=ctx, resolve_names=resolve_names
            ),
            body=tuple(
                _materialize_statement(item, env=env, ctx=ctx, resolve_names=resolve_names)
                for item in stmt.body
            ),
        )
    if isinstance(stmt, AxonScopeBind):
        return replace(
            stmt,
            kwargs={
                key: _materialize_kwarg_value(value, env=env, ctx=ctx, resolve_names=resolve_names)
                for key, value in stmt.kwargs.items()
            },
            body=tuple(
                _materialize_statement(item, env=env, ctx=ctx, resolve_names=resolve_names)
                for item in stmt.body
            ),
        )
    raise ValueError(f"Unsupported statement type: {type(stmt).__name__}")


def materialize_axon_file(ast: AxonFile, *, context: MaterializeContext) -> AxonFile:
    env: dict[str, object] = {}
    modules_out: list[AxonDefinition] = []
    for module in ast.modules:
        is_zero_arg_value = not module.params and module.body_expr is not None and not module.statements
        params = tuple(
            replace(
                param,
                default_expr=(
                    None
                    if param.default_expr is None
                    else _materialize_expr(
                        param.default_expr, env=env, ctx=context, resolve_names=True
                    )
                ),
            )
            for param in module.params
        )
        body_expr = (
            None
            if module.body_expr is None
            else _materialize_expr(
                module.body_expr,
                env=env,
                ctx=context,
                resolve_names=is_zero_arg_value,
            )
        )
        statements = tuple(
            _materialize_statement(stmt, env=env, ctx=context, resolve_names=False)
            for stmt in module.statements
        )
        modules_out.append(
            replace(
                module,
                params=params,
                body_expr=body_expr,
                statements=statements,
            )
        )
        if is_zero_arg_value and body_expr is not None:
            evaluated = _try_eval_expr(
                body_expr,
                env=env,
                ctx=context,
                resolve_names=True,
            )
            if evaluated is not _NOT_EVALUABLE:
                env[module.name] = evaluated
    return replace(
        ast,
        modules=tuple(modules_out),
    )


__all__ = ["materialize_axon_file"]
