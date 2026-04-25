from __future__ import annotations

from typing import Any

OP_NAME = "select"
LOWERING_ARITY = (0, 0)
LOWERING_ALLOWED_KWARGS: set[str] = {"cond"}
LOWERING_REQUIRED_KWARGS: set[str] = {"cond"}
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def _as_name_list(value: Any, *, field: str) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
        return [str(item) for item in value]
    raise ValueError(f"select requires {field} to be a non-empty string or list of strings")


def _as_graph(value: Any, *, field: str) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    raise ValueError(f"select requires {field} to be a list graph")


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del args, out, kwargs, ctx


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int | float | bool],
) -> None:
    cond_value = bool(model._eval_expr(node_spec.get("cond"), env, symbols))
    out_names = _as_name_list(node_spec.get("_bind"), field="_bind")
    if cond_value:
        branch_graph = _as_graph(node_spec.get("_then"), field="_then")
        branch_names = _as_name_list(node_spec.get("_then_bind"), field="_then_bind")
    else:
        branch_graph = _as_graph(node_spec.get("_else"), field="_else")
        branch_names = _as_name_list(node_spec.get("_else_bind"), field="_else_bind")
    if len(out_names) != len(branch_names):
        raise ValueError(
            f"select output arity mismatch: _bind has {len(out_names)} names,"
            f" branch bind has {len(branch_names)} names"
        )
    model_spec = model.spec.get("model", {})
    blocks = model_spec.get("blocks", {})
    if not isinstance(blocks, dict):
        raise ValueError("spec.model.blocks must be a mapping when present")
    model._run_graph(branch_graph, env, scope=scope, symbols=symbols, blocks=blocks)
    for out_name, branch_name in zip(out_names, branch_names, strict=True):
        if branch_name not in env:
            raise ValueError(f"select branch did not bind expected value {branch_name!r}")
        env[out_name] = env[branch_name]


def compile(
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    *,
    node_path_var: str,
    scope_var: str,
    indent: str,
) -> list[str]:
    del node_path_var
    out_names = _as_name_list(node_spec.get("_bind"), field="_bind")
    then_graph = _as_graph(node_spec.get("_then"), field="_then")
    else_graph = _as_graph(node_spec.get("_else"), field="_else")
    then_binds = _as_name_list(node_spec.get("_then_bind"), field="_then_bind")
    else_binds = _as_name_list(node_spec.get("_else_bind"), field="_else_bind")
    if len(out_names) != len(then_binds) or len(out_names) != len(else_binds):
        raise ValueError("select arity mismatch between _bind, _then_bind, and _else_bind")

    cond_code = emitter._expr_code(node_spec.get("cond"), env)
    out_vars = {out_name: emitter._assign_out_var(env, out_name) for out_name in out_names}

    lines: list[str] = [f"{indent}if {cond_code}:"]
    then_env = dict(env)
    lines.extend(
        emitter._compile_graph_with_non_null(
            graph=then_graph,
            env=then_env,
            scope_var=scope_var,
            indent=indent + "    ",
            non_null_names=emitter._non_null_names_for_condition(
                node_spec.get("cond"), truthy=True
            ),
        )
    )
    for out_name, branch_name in zip(out_names, then_binds, strict=True):
        branch_var = emitter._read_env_var(then_env, branch_name)
        lines.append(f"{indent}    {out_vars[out_name]} = {branch_var}")

    lines.append(f"{indent}else:")
    else_env = dict(env)
    lines.extend(
        emitter._compile_graph_with_non_null(
            graph=else_graph,
            env=else_env,
            scope_var=scope_var,
            indent=indent + "    ",
            non_null_names=emitter._non_null_names_for_condition(
                node_spec.get("cond"), truthy=False
            ),
        )
    )
    for out_name, branch_name in zip(out_names, else_binds, strict=True):
        branch_var = emitter._read_env_var(else_env, branch_name)
        lines.append(f"{indent}    {out_vars[out_name]} = {branch_var}")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": (),
    "kwargs": {"cond": "Bool"},
    "returns": "dynamic",
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
