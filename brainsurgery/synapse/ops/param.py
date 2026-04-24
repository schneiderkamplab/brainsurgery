from __future__ import annotations

from typing import Any

import torch

OP_NAME = "params_param"
LOWERING_ARITY = (1, 1)
LOWERING_ALLOWED_KWARGS: set[str] = set()
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return True


def lowering_validate_signature(
    *, args: list[str], out: str | list[str], kwargs: dict[str, Any], ctx: Any
) -> None:
    del ctx
    if len(args) != 1:
        raise ValueError(f"params_param expects exactly 1 positional arg, got {len(args)}")
    unknown = set(kwargs.keys()) - LOWERING_ALLOWED_KWARGS
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValueError(f"params_param received unknown kwargs: {unknown_list}")
    if not isinstance(out, str):
        raise ValueError("params_param requires a single output binding")


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    del scope
    raw = node_spec.get("_args")
    path_value = model._eval_expr(raw, env, symbols)
    path_spec = dict(node_spec)
    path_spec["value"] = path_value
    resolved = model._infer_param_path(path_spec, node_path=node_path, param_name="value", env=env)
    out_name = model._require_name(node_spec.get("_bind"), field="params_param._bind")
    value = model._state.get(resolved)
    if not torch.is_tensor(value):
        raise ValueError(f"param path is not a tensor: {resolved}")
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
    del scope_var
    lines: list[str] = []
    path_expr = emitter._expr_code(node_spec.get("_args"), env)
    path_var = emitter._fresh("param_path_arg")
    resolved_var = emitter._fresh("param_path_resolved")
    path_spec_var = emitter._fresh("param_path_spec")
    out_name = str(node_spec.get("_bind"))
    out_var = emitter._assign_out_var(env, out_name)
    lines.append(f"{indent}{path_var} = {path_expr}")
    lines.append(f"{indent}{path_spec_var} = {{'_op': 'params_param', 'value': {path_var}}}")
    if "_scope" in node_spec:
        lines.append(f"{indent}{path_spec_var}['_scope'] = {node_spec.get('_scope')!r}")
    if "_abs_path" in node_spec:
        lines.append(f"{indent}{path_spec_var}['_abs_path'] = {node_spec.get('_abs_path')!r}")
    if "_param_root" in node_spec:
        lines.append(f"{indent}{path_spec_var}['_param_root'] = {node_spec.get('_param_root')!r}")
    lines.append(
        f"{indent}{resolved_var} = self._infer_param_path({path_spec_var}, node_path={node_path_var}, param_name='value', env={{**env, **locals()}})"
    )
    lines.append(f"{indent}{out_var} = emitter._param({resolved_var})")
    return lines


LOWERING_TYPE_SIGNATURE = {
    "args": ("Path",),
    "kwargs": dict(LOWERING_KWARG_KINDS),
    "returns": ("Tensor",),
}

__all__ = [
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "LOWERING_TYPE_SIGNATURE",
    "OP_NAME",
    "compile",
    "interpret",
    "lowering_validate_signature",
    "uses_node_path",
]
