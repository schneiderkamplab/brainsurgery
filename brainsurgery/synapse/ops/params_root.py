from __future__ import annotations

from typing import Any

from ._params_common import (
    _compile_value_expr,
    _param_root_exists,
    _resolve_default,
    _resolve_root_arg,
    _uses_node_path,
)

OP_NAME = "params_root"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"default"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"default": "str"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    return _uses_node_path(emitter, node_spec)


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("params_root requires a single scalar output binding")


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
    out_name = model._require_name(node_spec.get("_bind"), field="params_root._bind")
    preferred = _resolve_root_arg(op_name=OP_NAME, node_spec=node_spec, env=env, symbols=symbols)
    has_default, default = _resolve_default(
        op_name=OP_NAME,
        node_spec=node_spec,
        env=env,
        symbols=symbols,
    )
    if _param_root_exists(model._state, preferred):
        env[out_name] = preferred
        return
    env[out_name] = default if has_default else ""


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
    preferred_var = emitter._fresh("preferred_root")
    found_var = emitter._fresh("preferred_found")
    key_var = emitter._fresh("param_key")
    resolved_var = emitter._fresh("resolved_root")
    preferred_expr = _compile_value_expr(emitter=emitter, value=node_spec.get("_args"), env=env)
    default_expr = (
        _compile_value_expr(emitter=emitter, value=node_spec.get("default"), env=env)
        if "default" in node_spec
        else repr("")
    )

    lines.append(f"{indent}{preferred_var} = {preferred_expr}")
    lines.append(f"{indent}if not isinstance({preferred_var}, str):")
    lines.append(
        f"{indent}    raise ValueError('params_root root argument must resolve to string')"
    )
    lines.append(f"{indent}{resolved_var} = {default_expr}")
    lines.append(f"{indent}if not isinstance({resolved_var}, str):")
    lines.append(f"{indent}    raise ValueError('params_root default must resolve to string')")
    lines.append(f"{indent}if {preferred_var} == '':")
    lines.append(f"{indent}    {found_var} = True")
    lines.append(f"{indent}else:")
    lines.append(f"{indent}    {found_var} = False")
    lines.append(f"{indent}    for {key_var} in self._state.keys():")
    lines.append(f"{indent}        if not isinstance({key_var}, str):")
    lines.append(f"{indent}            continue")
    lines.append(
        f"{indent}        if {key_var} == {preferred_var} or {key_var}.startswith({preferred_var} + '.'):"
    )
    lines.append(f"{indent}            {found_var} = True")
    lines.append(f"{indent}            break")
    lines.append(f"{indent}if {found_var}:")
    lines.append(f"{indent}    {resolved_var} = {preferred_var}")

    out_name = str(node_spec.get("_bind"))
    out_var = emitter._assign_out_var(env, out_name)
    lines.append(f"{indent}{out_var} = {resolved_var}")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("String",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("String",),
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
