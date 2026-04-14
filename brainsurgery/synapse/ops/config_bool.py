from __future__ import annotations

from typing import Any

from ._config_common import (
    _compile_default_lines,
    _compile_lookup_lines,
    _resolve_config_value,
    _uses_node_path,
)

OP_NAME = "config_bool"
LOWERING_ARITY = (1, 3)
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
        raise ValueError("config_bool requires a single scalar output binding")


def _coerce_bool(*, key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw == "true":
            return True
        if raw == "false":
            return False
    raise ValueError(f"{OP_NAME} key {key!r} expected bool, got {type(value).__name__}")


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
    out_name = model._require_name(node_spec.get("_bind"), field="config_bool._bind")
    key, _found, value = _resolve_config_value(
        op_name=OP_NAME,
        model=model,
        node_spec=node_spec,
        env=env,
        symbols=symbols,
    )
    env[out_name] = _coerce_bool(key=key, value=value)


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
    lines, key_var, found_var, value_var = _compile_lookup_lines(
        op_name=OP_NAME,
        emitter=emitter,
        node_spec=node_spec,
        env=env,
        indent=indent,
    )
    lines.extend(
        _compile_default_lines(
            op_name=OP_NAME,
            emitter=emitter,
            node_spec=node_spec,
            env=env,
            indent=indent,
            found_var=found_var,
            key_var=key_var,
            value_var=value_var,
        )
    )
    bool_from_value = emitter._fresh("bool_from_value")
    lines.append(f"{indent}if isinstance({value_var}, bool):")
    lines.append(f"{indent}    {bool_from_value} = {value_var}")
    lines.append(f"{indent}elif isinstance({value_var}, str):")
    lines.append(f"{indent}    _raw = {value_var}.strip().lower()")
    lines.append(f"{indent}    if _raw == 'true':")
    lines.append(f"{indent}        {bool_from_value} = True")
    lines.append(f"{indent}    elif _raw == 'false':")
    lines.append(f"{indent}        {bool_from_value} = False")
    lines.append(f"{indent}    else:")
    lines.append(
        f"{indent}        raise ValueError({OP_NAME!r} + ' key ' + repr({key_var}) + ' expected bool, got ' + type({value_var}).__name__)"
    )
    lines.append(f"{indent}else:")
    lines.append(
        f"{indent}    raise ValueError({OP_NAME!r} + ' key ' + repr({key_var}) + ' expected bool, got ' + type({value_var}).__name__)"
    )
    out_name = str(node_spec.get("_bind"))
    out_var = emitter._assign_out_var(env, out_name)
    lines.append(f"{indent}{out_var} = {bool_from_value}")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("String", "String", "Bool"),
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
