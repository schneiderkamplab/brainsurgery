from __future__ import annotations

from typing import Any

from ._config_common import (
    _compile_lookup_lines,
    _config_lookup,
    _config_root,
    _resolve_key,
    _uses_node_path,
)

OP_NAME = "config_has"
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
        raise ValueError("config_has requires a single scalar output binding")


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
    out_name = model._require_name(node_spec.get("_bind"), field="config_has._bind")
    key = _resolve_key(op_name=OP_NAME, node_spec=node_spec, env=env, symbols=symbols)
    config = _config_root(model.spec)
    found, _ = _config_lookup(config, key)
    env[out_name] = bool(found)


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
    lines, _key_var, found_var, _value_var = _compile_lookup_lines(
        op_name=OP_NAME,
        emitter=emitter,
        node_spec=node_spec,
        env=env,
        indent=indent,
    )
    out_name = str(node_spec.get("_bind"))
    out_var = emitter._assign_out_var(env, out_name)
    lines.append(f"{indent}{out_var} = bool({found_var})")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Path",),
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
