from __future__ import annotations

from typing import Any

from ._params_common import (
    _compile_value_expr,
    _param_root_exists,
    _resolve_root_arg,
    _uses_node_path,
)

OP_NAME = "params_has_root"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    return _uses_node_path(emitter, node_spec)


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("params_has_root requires a single scalar output binding")


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    del node_path, scope
    out_name = model._require_name(node_spec.get("_bind"), field="params_has_root._bind")
    root = _resolve_root_arg(op_name=OP_NAME, node_spec=node_spec, env=env, symbols=symbols)
    env[out_name] = bool(_param_root_exists(model._state, root))


def compile(
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    *,
    node_path_var: str,
    scope_var: str,
    indent: str,
) -> list[str]:
    del node_path_var, scope_var
    lines: list[str] = []
    root_var = emitter._fresh("param_root")
    found_var = emitter._fresh("param_root_found")
    key_var = emitter._fresh("param_key")
    root_expr = _compile_value_expr(emitter=emitter, value=node_spec.get("_args"), env=env)
    lines.append(f"{indent}{root_var} = {root_expr}")
    lines.append(f"{indent}if not isinstance({root_var}, str):")
    lines.append(
        f"{indent}    raise ValueError('params_has_root root argument must resolve to string')"
    )
    lines.append(f"{indent}if {root_var} == '':")
    lines.append(f"{indent}    {found_var} = True")
    lines.append(f"{indent}else:")
    lines.append(f"{indent}    {found_var} = False")
    lines.append(f"{indent}    for {key_var} in self._state.keys():")
    lines.append(f"{indent}        if not isinstance({key_var}, str):")
    lines.append(f"{indent}            continue")
    lines.append(
        f"{indent}        if {key_var} == {root_var} or {key_var}.startswith({root_var} + '.'):"
    )
    lines.append(f"{indent}            {found_var} = True")
    lines.append(f"{indent}            break")
    out_name = str(node_spec.get("_bind"))
    out_var = emitter._assign_out_var(env, out_name)
    lines.append(f"{indent}{out_var} = bool({found_var})")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("String",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Bool",),
}

__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_validate_signature",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
