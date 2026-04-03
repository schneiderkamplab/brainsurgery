from __future__ import annotations

from typing import Any

from ._config_common import (
    _compile_default_lines,
    _compile_lookup_lines,
    _resolve_config_value,
    _uses_node_path,
)

OP_NAME = "config_value"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = {"default", "root"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {"root": "str"}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    return _uses_node_path(emitter, node_spec)


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("config_value requires a single output binding")


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
    out_name = model._require_name(node_spec.get("_bind"), field="config_value._bind")
    _, _found, value = _resolve_config_value(
        op_name=OP_NAME,
        model=model,
        node_spec=node_spec,
        env=env,
        symbols=symbols,
    )
    env[out_name] = value


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
    out_name = str(node_spec.get("_bind"))
    out_var = emitter._assign_out_var(env, out_name)
    lines.append(f"{indent}{out_var} = {value_var}")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("String",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Any",),
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
