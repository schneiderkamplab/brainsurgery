from __future__ import annotations

from typing import Any

OP_NAME = "_ir_expr"
LOWERING_ARITY = (0, 0)
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
    raise ValueError("_ir_expr requires a scalar or list output binding")


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
    value = node_spec.get("value")
    evaluated = model._eval_expr(value, env, symbols)
    bind_raw = node_spec.get("_bind")
    if isinstance(bind_raw, str):
        env[bind_raw] = evaluated
        return
    if isinstance(bind_raw, list):
        if not isinstance(evaluated, list | tuple):
            raise ValueError("_ir_expr list-bind expects tuple/list value")
        if len(evaluated) != len(bind_raw):
            raise ValueError("_ir_expr list-bind arity mismatch")
        for idx, name in enumerate(bind_raw):
            if not isinstance(name, str):
                raise ValueError("_ir_expr list-bind expects string targets")
            if name == "_":
                continue
            env[name] = evaluated[idx]
        return
    raise ValueError("_ir_expr._bind must be a name or list of names")
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
    value = node_spec.get("value")
    value_code = emitter._expr_code(value, env)
    bind_raw = node_spec.get("_bind")
    if isinstance(bind_raw, str):
        out_var = emitter._assign_out_var(env, bind_raw)
        return [f"{indent}{out_var} = {value_code}"]
    if isinstance(bind_raw, list):
        tmp = f"_expr_unpack_{abs(hash(tuple(str(x) for x in bind_raw))):x}"
        lines: list[str] = [f"{indent}{tmp} = {value_code}"]
        for idx, raw_name in enumerate(bind_raw):
            if not isinstance(raw_name, str) or raw_name == "_":
                continue
            out_var = emitter._assign_out_var(env, raw_name)
            lines.append(f"{indent}{out_var} = {tmp}[{idx}]")
        return lines
    raise ValueError("_ir_expr._bind must be a name or list of names")


LOWERING_TYPE_SIGNATURE = {
    "args": (),
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
