from __future__ import annotations

from typing import Any

OP_NAME = "_ir_alias"
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
    if isinstance(out, str):
        return
    if isinstance(out, list) and all(isinstance(name, str) for name in out):
        return
    raise ValueError("_ir_alias requires a scalar or list output binding")


def lowering_infer_metadata(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> bool:
    del kwargs
    if not isinstance(out, str) or not args:
        return False
    source_name = str(args[0]).strip()
    if source_name in ctx.tensor_last_dim:
        ctx.tensor_last_dim[out] = ctx.tensor_last_dim[source_name]
    if source_name in ctx.tensor_shape:
        ctx.tensor_shape[out] = ctx.tensor_shape[source_name]
    return True


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
    source = model._require_name(node_spec.get("_args"), field="_ir_alias._args")
    if source in env:
        source_value = env[source]
    elif source in symbols:
        source_value = symbols[source]
    else:
        raise ValueError(f"_ir_alias missing input {source!r}")
    bind_raw = node_spec.get("_bind")
    if isinstance(bind_raw, str):
        env[bind_raw] = source_value
        return
    if isinstance(bind_raw, list):
        value = source_value
        if not isinstance(value, list | tuple):
            raise ValueError("_ir_alias list-bind expects tuple/list source value")
        if len(value) != len(bind_raw):
            raise ValueError("_ir_alias list-bind arity mismatch")
        for idx, name in enumerate(bind_raw):
            if not isinstance(name, str):
                raise ValueError("_ir_alias list-bind expects string targets")
            if name == "_":
                continue
            env[name] = value[idx]
        return
    raise ValueError("_ir_alias._bind must be a name or list of names")
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
    del node_path_var, scope_var
    source = str(node_spec.get("_args"))
    if source in env:
        source_expr = emitter._read_env_var(env, source)
    else:
        source_expr = emitter._expr_code({"_expr": "name", "id": source}, env)
    bind_raw = node_spec.get("_bind")
    if isinstance(bind_raw, str):
        out_var = emitter._assign_out_var(env, bind_raw)
        return [f"{indent}{out_var} = {source_expr}"]
    if isinstance(bind_raw, list):
        tmp = f"_alias_unpack_{abs(hash((source, tuple(str(x) for x in bind_raw)))):x}"
        lines: list[str] = [f"{indent}{tmp} = {source_expr}"]
        for idx, raw_name in enumerate(bind_raw):
            if not isinstance(raw_name, str) or raw_name == "_":
                continue
            out_var = emitter._assign_out_var(env, raw_name)
            lines.append(f"{indent}{out_var} = {tmp}[{idx}]")
        return lines
    raise ValueError("_ir_alias._bind must be a name or list of names")


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": "dynamic",
}

__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "OP_NAME",
    "lowering_validate_signature",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
    "LOWERING_TYPE_SIGNATURE",
]
