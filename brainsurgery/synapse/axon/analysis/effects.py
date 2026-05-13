from __future__ import annotations

from enum import Enum
from typing import Mapping

from ..ast.nodes import (
    AxonBind,
    AxonCond,
    AxonDefinition,
    AxonExpr,
    AxonExprAscribe,
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
    AxonKwargValue,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
    AxonYield,
)


class PurityEffect(str, Enum):
    TOTAL_PURE = "total_pure"
    PARTIAL_PURE = "partial_pure"
    EFFECTFUL = "effectful"


_TOTAL_CORE_PREFIXES = ("core.binary.",)
_TOTAL_CORE_OPS = {
    "core.alias",
    "core.ascribe",
    "core.list",
    "core.select",
    "core.tuple",
}

_CONFIG_VALUE_OPS = {
    "config_bool",
    "config_dim",
    "config_float",
    "config_int",
    "config_list",
    "config_str",
    "config_value",
}

_CONFIG_TOTAL_OPS = {
    "config_has",
    "config_has_value",
}

_PARAM_PARTIAL_OPS = {
    "params_param",
}

_PARAM_TOTAL_OPS = {
    "params_has_root",
}

_PARTIAL_PURE_OPS = {
    "require",
    *_CONFIG_VALUE_OPS,
    *_PARAM_PARTIAL_OPS,
}

_TOTAL_PURE_PRIMITIVE_OPS = {
    "activation",
    "add",
    "and",
    "arange",
    "cast",
    "cast_like",
    "chunk",
    "clamp",
    "concat",
    "cos",
    "cumsum",
    "div",
    "dtype_value",
    "embedding",
    "eq",
    "exp",
    "expand",
    "expert_linear",
    "fill",
    "floor",
    "full",
    "gather",
    "index_add",
    "ir_alias",
    "ir_expr",
    "l2norm",
    "layernorm",
    "le",
    "linear",
    "list_append",
    "list_index",
    "list_init",
    "list_length",
    "log",
    "matmul",
    "mul",
    "permute",
    "pow",
    "repeat",
    "reshape",
    "rmsnorm",
    "scatter",
    "select",
    "shape",
    "sin",
    "slice",
    "softmax",
    "split",
    "sqrt",
    "sum",
    "tensor_like",
    "topk",
    "transpose",
    "unsqueeze",
    "where",
    "where_indices",
    "zeros",
    "zeros_like",
    *_CONFIG_TOTAL_OPS,
    *_PARAM_TOTAL_OPS,
}


def join_effect(left: PurityEffect, right: PurityEffect) -> PurityEffect:
    if PurityEffect.EFFECTFUL in {left, right}:
        return PurityEffect.EFFECTFUL
    if PurityEffect.PARTIAL_PURE in {left, right}:
        return PurityEffect.PARTIAL_PURE
    return PurityEffect.TOTAL_PURE


def _normalize_op_name(op_name: str) -> str:
    builtin_aliases = {
        "Config.bool": "config_bool",
        "Config.dim": "config_dim",
        "Config.float": "config_float",
        "Config.has_key": "config_has",
        "Config.has_value": "config_has_value",
        "Config.int": "config_int",
        "Config.list": "config_list",
        "Config.str": "config_str",
        "Config.value": "config_value",
        "Params.has_root": "params_has_root",
        "Params.param": "params_param",
    }
    if op_name in builtin_aliases:
        return builtin_aliases[op_name]
    if op_name.startswith("_activations_"):
        return op_name[1:]
    if op_name.startswith("_"):
        return op_name[1:]
    return op_name


def _is_null_default(value: object) -> bool:
    return value is None or isinstance(value, AxonExprNull)


def _has_non_null_default_value(value: object) -> bool:
    return value is not None and not _is_null_default(value)


def _has_non_null_default(attrs: Mapping[str, object] | None) -> bool:
    if not attrs or "default" not in attrs:
        return False
    return _has_non_null_default_value(attrs["default"])


def op_effect(op_name: str, *, attrs: Mapping[str, object] | None = None) -> PurityEffect:
    if op_name in _TOTAL_CORE_OPS or op_name.startswith(_TOTAL_CORE_PREFIXES):
        return PurityEffect.TOTAL_PURE
    if op_name.startswith("core."):
        return PurityEffect.PARTIAL_PURE

    normalized = _normalize_op_name(op_name)
    if normalized in _CONFIG_VALUE_OPS:
        return PurityEffect.TOTAL_PURE if _has_non_null_default(attrs) else PurityEffect.PARTIAL_PURE
    if normalized in _PARTIAL_PURE_OPS:
        return PurityEffect.PARTIAL_PURE
    if normalized.startswith("activations_"):
        return PurityEffect.TOTAL_PURE
    if normalized in _TOTAL_PURE_PRIMITIVE_OPS:
        return PurityEffect.TOTAL_PURE
    if normalized == "empty":
        return PurityEffect.EFFECTFUL
    return PurityEffect.PARTIAL_PURE


def _expr_non_null(
    expr: AxonExpr,
    *,
    env: Mapping[str, AxonExpr] | None = None,
    active: frozenset[str] = frozenset(),
) -> bool:
    if isinstance(expr, AxonExprNull):
        return False
    if isinstance(expr, AxonExprName) and env is not None and expr.name in env and expr.name not in active:
        return _expr_non_null(env[expr.name], env=env, active=active | {expr.name})
    return isinstance(
        expr,
        AxonExprInt
        | AxonExprFloat
        | AxonExprBool
        | AxonExprString
        | AxonExprPath
        | AxonExprList
        | AxonExprTuple,
    )


def _kwarg_non_null(
    value: AxonKwargValue,
    *,
    env: Mapping[str, AxonExpr] | None = None,
) -> bool:
    if isinstance(value, AxonExpr):
        return _expr_non_null(value, env=env)
    return _has_non_null_default_value(value)


def _op_call_effect(
    op_name: str,
    *,
    args: tuple[AxonExpr, ...] = (),
    kwargs: Mapping[str, AxonKwargValue] | None = None,
    env: Mapping[str, AxonExpr] | None = None,
) -> PurityEffect:
    normalized = _normalize_op_name(op_name)
    if normalized in _CONFIG_VALUE_OPS:
        if kwargs is not None and "default" in kwargs:
            return (
                PurityEffect.TOTAL_PURE
                if _kwarg_non_null(kwargs["default"], env=env)
                else PurityEffect.PARTIAL_PURE
            )
        if len(args) >= 2:
            return (
                PurityEffect.TOTAL_PURE
                if _expr_non_null(args[1], env=env)
                else PurityEffect.PARTIAL_PURE
            )
    return op_effect(op_name, attrs=kwargs)


def _kwarg_value_effect(
    value: AxonKwargValue,
    *,
    definition_effects: Mapping[str, PurityEffect] | None,
    definitions: Mapping[str, AxonDefinition] | None = None,
    env: Mapping[str, AxonExpr] | None = None,
) -> PurityEffect:
    if isinstance(value, AxonExpr):
        return axon_expr_effect(
            value,
            definition_effects=definition_effects,
            definitions=definitions,
            env=env,
        )
    return PurityEffect.TOTAL_PURE


def axon_expr_effect(
    expr: AxonExpr,
    *,
    definition_effects: Mapping[str, PurityEffect] | None = None,
    definitions: Mapping[str, AxonDefinition] | None = None,
    env: Mapping[str, AxonExpr] | None = None,
    active_calls: frozenset[str] = frozenset(),
    active_names: frozenset[str] = frozenset(),
) -> PurityEffect:
    if isinstance(expr, AxonExprName) and env is not None and expr.name in env and expr.name not in active_names:
        replacement = env[expr.name]
        if replacement == expr:
            return PurityEffect.TOTAL_PURE
        return axon_expr_effect(
            replacement,
            definition_effects=definition_effects,
            definitions=definitions,
            env=env,
            active_calls=active_calls,
            active_names=active_names | {expr.name},
        )
    if isinstance(expr, AxonExprName | AxonExprPath):
        return PurityEffect.TOTAL_PURE
    if isinstance(expr, AxonExprInt | AxonExprFloat | AxonExprBool | AxonExprNull | AxonExprString):
        return PurityEffect.TOTAL_PURE
    if isinstance(expr, AxonExprAscribe):
        return axon_expr_effect(expr.expr, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls, active_names=active_names)
    if isinstance(expr, AxonExprParen):
        return axon_expr_effect(expr.inner, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls, active_names=active_names)
    if isinstance(expr, AxonExprList | AxonExprTuple):
        effect = PurityEffect.TOTAL_PURE
        for item in expr.items:
            effect = join_effect(effect, axon_expr_effect(item, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls))
        return effect
    if isinstance(expr, AxonExprBinary):
        return join_effect(
            axon_expr_effect(expr.left, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls),
            axon_expr_effect(expr.right, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls),
        )
    if isinstance(expr, AxonExprTernary | AxonExprIf):
        effect = axon_expr_effect(expr.cond, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls)
        effect = join_effect(effect, axon_expr_effect(expr.true_expr, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls))
        return join_effect(effect, axon_expr_effect(expr.false_expr, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls))
    if isinstance(expr, AxonExprCall):
        definition = definitions.get(expr.callee) if definitions is not None else None
        if definition is not None and expr.callee not in active_calls:
            effect = _instantiated_definition_effect(
                definition,
                args=expr.args,
                kwargs=expr.kwargs,
                definition_effects=definition_effects,
                definitions=definitions,
                outer_env=env,
                active_calls=active_calls | {expr.callee},
            )
        else:
            effect = _op_call_effect(expr.callee, args=expr.args, kwargs=expr.kwargs, env=env)
        for arg in expr.args:
            effect = join_effect(effect, axon_expr_effect(arg, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls))
        for value in expr.kwargs.values():
            effect = join_effect(
                effect,
                _kwarg_value_effect(value, definition_effects=definition_effects, definitions=definitions, env=env),
            )
        return effect
    if isinstance(expr, AxonExprPipe):
        effect = axon_expr_effect(expr.value, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls)
        for stage in expr.stages:
            effect = join_effect(effect, axon_expr_effect(stage, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls))
        return effect
    if isinstance(expr, AxonExprBind):
        return join_effect(
            axon_expr_effect(expr.value, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls),
            axon_expr_effect(expr.body, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls),
        )
    if isinstance(expr, AxonExprLambda):
        return axon_expr_effect(expr.body, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls)
    if isinstance(expr, AxonExprDo):
        return axon_statements_effect(expr.body, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls)
    return PurityEffect.PARTIAL_PURE


def axon_statement_effect(
    statement: AxonStatement,
    *,
    definition_effects: Mapping[str, PurityEffect] | None = None,
    definitions: Mapping[str, AxonDefinition] | None = None,
    env: Mapping[str, AxonExpr] | None = None,
    active_calls: frozenset[str] = frozenset(),
) -> PurityEffect:
    if isinstance(statement, AxonBind):
        return axon_expr_effect(statement.expr, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls)
    if isinstance(statement, AxonReturn | AxonYield):
        effect = PurityEffect.TOTAL_PURE
        for value in statement.values:
            effect = join_effect(effect, axon_expr_effect(value, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls))
        return effect
    if isinstance(statement, AxonCond):
        effect = axon_expr_effect(statement.cond, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls)
        effect = join_effect(effect, axon_statements_effect(statement.true_body, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls))
        return join_effect(effect, axon_statements_effect(statement.false_body, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls))
    if isinstance(statement, AxonRepeat):
        effect = axon_expr_effect(statement.from_expr, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls)
        effect = join_effect(effect, axon_expr_effect(statement.to_expr, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls))
        effect = join_effect(effect, axon_expr_effect(statement.step_expr, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls))
        return join_effect(effect, axon_statements_effect(statement.body, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls))
    if isinstance(statement, AxonScopeBind):
        return axon_statements_effect(statement.body, definition_effects=definition_effects, definitions=definitions, env=env, active_calls=active_calls)
    return PurityEffect.PARTIAL_PURE


def axon_statements_effect(
    statements: tuple[AxonStatement, ...],
    *,
    definition_effects: Mapping[str, PurityEffect] | None = None,
    definitions: Mapping[str, AxonDefinition] | None = None,
    env: Mapping[str, AxonExpr] | None = None,
    active_calls: frozenset[str] = frozenset(),
) -> PurityEffect:
    effect = PurityEffect.TOTAL_PURE
    for statement in statements:
        effect = join_effect(
            effect,
            axon_statement_effect(
                statement,
                definition_effects=definition_effects,
                definitions=definitions,
                env=env,
                active_calls=active_calls,
            ),
        )
    return effect


def _instantiated_definition_effect(
    definition: AxonDefinition,
    *,
    args: tuple[AxonExpr, ...],
    kwargs: Mapping[str, AxonKwargValue],
    definition_effects: Mapping[str, PurityEffect] | None,
    definitions: Mapping[str, AxonDefinition] | None,
    outer_env: Mapping[str, AxonExpr] | None,
    active_calls: frozenset[str],
) -> PurityEffect:
    del outer_env
    env: dict[str, AxonExpr] = {}
    for param, arg in zip(definition.params, args, strict=False):
        env[param.name] = arg
    for param in definition.params[len(args) :]:
        if param.name in kwargs and isinstance(kwargs[param.name], AxonExpr):
            env[param.name] = kwargs[param.name]
        elif param.default_expr is not None:
            env[param.name] = param.default_expr
    if definition.body_expr is not None:
        return axon_expr_effect(
            definition.body_expr,
            definition_effects=definition_effects,
            definitions=definitions,
            env=env,
            active_calls=active_calls,
        )
    return axon_statements_effect(
        definition.statements,
        definition_effects=definition_effects,
        definitions=definitions,
        env=env,
        active_calls=active_calls,
    )


def axon_definition_effect(
    definition: AxonDefinition,
    *,
    definition_effects: Mapping[str, PurityEffect] | None = None,
    definitions: Mapping[str, AxonDefinition] | None = None,
) -> PurityEffect:
    if definition.body_expr is not None:
        return axon_expr_effect(definition.body_expr, definition_effects=definition_effects, definitions=definitions)
    return axon_statements_effect(definition.statements, definition_effects=definition_effects, definitions=definitions)


def infer_axon_definition_effects(
    definitions: tuple[AxonDefinition, ...],
    *,
    max_iterations: int = 16,
) -> dict[str, PurityEffect]:
    definitions_by_name = {definition.name: definition for definition in definitions}
    effects = {definition.name: PurityEffect.PARTIAL_PURE for definition in definitions}
    for _ in range(max_iterations):
        changed = False
        for definition in definitions:
            inferred = axon_definition_effect(
                definition,
                definition_effects=effects,
                definitions=definitions_by_name,
            )
            if effects[definition.name] != inferred:
                effects[definition.name] = inferred
                changed = True
        if not changed:
            break
    return effects


__all__ = [
    "PurityEffect",
    "axon_definition_effect",
    "axon_expr_effect",
    "axon_statement_effect",
    "axon_statements_effect",
    "infer_axon_definition_effects",
    "join_effect",
    "op_effect",
]
