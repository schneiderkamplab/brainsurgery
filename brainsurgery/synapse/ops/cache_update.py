from __future__ import annotations

from typing import Any

OP_NAME = "cache_update"
LOWERING_ARITY = (3, 3)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def _deprecated_message() -> str:
    return (
        "cache_update primitive is deprecated; use Cache.update from "
        "brainsurgery/synapse/builtins/Cache.axon"
    )


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_known_output_arity(*, kwargs: dict[str, Any]) -> int:
    del kwargs
    return 3


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, out, kwargs, ctx
    raise ValueError(_deprecated_message())


def lowering_infer_metadata(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> bool:
    del args, out, kwargs, ctx
    raise ValueError(_deprecated_message())


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    del model, node_spec, env, node_path, scope, symbols
    raise ValueError(_deprecated_message())


def compile(
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    *,
    node_path_var: str,
    scope_var: str,
    indent: str,
) -> list[str]:
    del emitter, node_spec, env, node_path_var, scope_var, indent
    raise ValueError(_deprecated_message())


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any", "Any"),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor", "Tensor", "Tensor"),
}

__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_known_output_arity",
    "lowering_validate_signature",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
