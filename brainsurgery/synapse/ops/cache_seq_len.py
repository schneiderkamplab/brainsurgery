from __future__ import annotations

from typing import Any

import torch

OP_NAME = "cache_seq_len"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, kwargs, ctx
    if not isinstance(out, str):
        raise ValueError("cache_seq_len requires a single scalar output binding")


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    ref = node_spec.get("_args")
    out_name = model._require_name(node_spec.get("_bind"), field="cache_seq_len._bind")
    if not isinstance(ref, str):
        raise ValueError("cache_seq_len.in must be a string")
    value = env.get(ref)
    if value is None:
        env[out_name] = 0
        return
    if isinstance(value, tuple) and len(value) >= 1 and torch.is_tensor(value[0]):
        env[out_name] = int(value[0].shape[-2])
        return
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, tuple) and len(first) >= 1 and torch.is_tensor(first[0]):
            env[out_name] = int(first[0].shape[-2])
            return
    raise ValueError("cache_seq_len expects kv tuple (k, v) or cache list of kv tuples")
    return


def compile(
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    *,
    node_path_var: str,
    scope_var: str,
    indent: str,
) -> list[str]:
    lines: list[str] = []

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    ref = node_spec.get("_args")
    if not isinstance(ref, str):
        raise ValueError("cache_seq_len.in must be string")
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    src = read(ref)
    lines.append(f"{indent}if {src} is None:")
    lines.append(f"{indent}    {out_var} = 0")
    lines.append(f"{indent}elif isinstance({src}, list):")
    lines.append(f"{indent}    {out_var} = int({src}[0][0].shape[-2]) if {src} else 0")
    lines.append(f"{indent}else:")
    lines.append(f"{indent}    {out_var} = int({src}[0].shape[-2])")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Int",),
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
