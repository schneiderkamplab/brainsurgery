from __future__ import annotations

import ast
import re
from typing import Any


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return repr(value)


def _try_eval_numeric(text: str) -> int | float | bool | None:
    try:
        parsed = ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Not,
        ast.UnaryOp,
    )
    for node in ast.walk(parsed):
        if not isinstance(node, allowed):
            return None
        if isinstance(node, ast.Name):
            return None
    try:
        value = eval(compile(parsed, "<axon-render>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    return None


def _resolve_value(value: Any, symbols: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_value(v, symbols) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, symbols) for v in value]
    if not isinstance(value, str):
        return value

    token = value.strip()
    if token in symbols and isinstance(symbols[token], (int, float, bool)):
        return symbols[token]

    substituted = value
    for name, sym_val in sorted(symbols.items(), key=lambda kv: len(kv[0]), reverse=True):
        if not isinstance(sym_val, (int, float, bool)):
            continue
        substituted = re.sub(rf"\b{re.escape(name)}\b", repr(sym_val), substituted)
    evaluated = _try_eval_numeric(substituted)
    if evaluated is not None:
        return evaluated
    return substituted


def _axon_expr_from_node(node_spec: dict[str, Any], *, node_path: str | None = None) -> str:
    op = str(node_spec.get("_op"))
    in_value = node_spec.get("_args")
    in_args: list[str]
    if isinstance(in_value, list):
        in_args = [str(item) for item in in_value]
    elif isinstance(in_value, str):
        in_args = [in_value]
    else:
        in_args = []

    kwargs: list[str] = []
    for key, value in node_spec.items():
        if key in {"_op", "_args", "_bind", "_target", "_params"}:
            continue
        kwargs.append(f"{key}={_format_scalar(value)}")

    params = node_spec.get("_params")
    if isinstance(params, dict) and isinstance(params.get("weight"), str):
        weight = params["weight"]
        if weight.endswith(".weight"):
            path = weight[: -len(".weight")]
            callee = f"{op}@{path}"
        else:
            callee = op
    elif (
        node_path
        and op in {"linear", "embedding", "layernorm", "rmsnorm"}
        and not isinstance(node_spec.get("weight"), str)
    ):
        callee = f"{op}@{node_path}"
    elif op.startswith("activations_"):
        callee = f"_activations_{op[len('activations_') :]}"
    elif op == "cache_update":
        callee = "_cache_update"
    elif op == "cache_seq_len":
        callee = "_cache_seq_len"
    elif op == "split":
        callee = "split"
    else:
        callee = op

    if op == "add" and len(in_args) == 2 and not kwargs:
        return f"{in_args[0]} + {in_args[1]}"
    if op == "mul" and len(in_args) == 2 and not kwargs:
        return f"{in_args[0]} * {in_args[1]}"

    all_args = [*in_args, *kwargs]
    return f"{callee}({', '.join(all_args)})"


def _can_render_as_bind(node_spec: dict[str, Any]) -> bool:
    if node_spec.get("_op") == "call":
        return False
    if "graph" in node_spec and "_op" not in node_spec:
        return False
    out_ref = node_spec.get("_bind")
    if not isinstance(out_ref, (str, list)):
        return False
    return isinstance(node_spec.get("_op"), str)


def _render_module(
    *,
    module_name: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    graph: list[Any],
    symbols: dict[str, Any],
) -> list[str]:
    params: list[str] = []
    for name, input_spec in inputs.items():
        optional = isinstance(input_spec, dict) and bool(input_spec.get("optional", False))
        params.append(f"{name}?" if optional else str(name))

    return_names = list(outputs.keys()) if outputs else ["out"]
    arg_types = ["?Tensor" if p.endswith("?") else "Tensor" for p in params]
    if len(return_names) == 1:
        ret_type = "Tensor"
    else:
        ret_type = "(" + ", ".join("Tensor" for _ in return_names) + ")"
    sig = " -> ".join([*arg_types, ret_type]) if arg_types else ret_type
    def_params = [p[:-1] if p.endswith("?") else p for p in params]
    def_head = f"{module_name} {' '.join(def_params)}".rstrip()
    lines = [f"{module_name} :: {sig}", f"{def_head} = do"]

    def render_graph(items: list[Any], *, scope: str, indent: str) -> None:
        for item in items:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f"invalid graph item: {item!r}")
            node_name, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                raise ValueError(f"invalid node spec: {node_spec!r}")
            node_spec = _resolve_value(node_spec, symbols)

            node_path = f"{scope}.{node_name}" if scope else str(node_name)
            if node_spec.get("_op") == "for":
                scope_name = node_spec.get("_scope")
                if not isinstance(scope_name, str) or not scope_name:
                    raise ValueError("for node requires string _scope")
                var = str(node_spec.get("_var"))
                to_expr = _format_scalar(node_spec.get("_to"))
                start_expr = _format_scalar(node_spec.get("_from", 0))
                step_expr = _format_scalar(node_spec.get("_step", 1))
                body = node_spec.get("_body")
                if isinstance(body, list):
                    for_name = f"{scope}.{scope_name}" if scope else scope_name
                    step_suffix = "" if step_expr == "1" else f" step={step_expr}"
                    lines.append(
                        f"{indent}for@{for_name} {var} <- [{start_expr}..{to_expr}){step_suffix} do"
                    )
                    render_graph(body, scope=node_path, indent=indent + "  ")
                    continue

            if node_spec.get("_op") == "call":
                callee = str(node_spec.get("_target"))
                raw_args = node_spec.get("_args")
                bind = node_spec.get("_bind")
                args_values = (
                    raw_args
                    if isinstance(raw_args, list)
                    else ([raw_args] if raw_args is not None else [])
                )
                if isinstance(bind, list):
                    out_values = [str(v) for v in bind]
                elif isinstance(bind, str):
                    out_values = [bind]
                else:
                    raise ValueError(f"invalid call bind: {node_spec!r}")
                kwargs_parts: list[str] = []
                for key, value in node_spec.items():
                    if key.startswith("_") or key in {"when", "graph"}:
                        continue
                    kwargs_parts.append(f"{key}={value}")
                args_parts = [str(v) for v in args_values] + kwargs_parts
                lhs = ", ".join(out_values)
                args = ", ".join(args_parts)
                lines.append(f"{indent}{lhs} <- {callee}({args})")
                continue

            if "graph" in node_spec and "_op" not in node_spec:
                nested = node_spec.get("graph")
                if not isinstance(nested, list):
                    raise ValueError(f"invalid nested graph node: {node_spec!r}")
                render_graph(nested, scope=node_path, indent=indent)
                continue

            if _can_render_as_bind(node_spec):
                out_ref = node_spec.get("_bind")
                lhs = (
                    ", ".join(str(name) for name in out_ref)
                    if isinstance(out_ref, list)
                    else str(out_ref)
                )
                rhs = _axon_expr_from_node(node_spec, node_path=node_path)
                lines.append(f"{indent}{lhs} <- {rhs}")
                continue

            raise ValueError(f"node {node_path!r} cannot be rendered in strict Axon syntax")

    render_graph(graph, scope="", indent="  ")

    if outputs:
        ordered = [str(value) for value in outputs.values() if isinstance(value, str)]
        if len(ordered) == len(outputs):
            lines.append(f"  return {', '.join(ordered)}")
        else:
            lines.append("  return null")
    else:
        lines.append("  return null")

    return lines


def synapse_spec_to_axon_module_text(spec: dict[str, Any], *, module_name: str = "main") -> str:
    model = spec.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("spec.model must be a mapping")

    inputs = model.get("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError("model.inputs must be a mapping")
    outputs = model.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("model.outputs must be a mapping")
    graph = model.get("graph", [])
    if not isinstance(graph, list):
        raise ValueError("model.graph must be a list")

    lines: list[str] = []
    symbols = model.get("symbols", {})
    if not isinstance(symbols, dict):
        symbols = {}
    blocks = model.get("blocks")
    if isinstance(blocks, dict):
        for block_name, block_spec in blocks.items():
            if not isinstance(block_spec, dict):
                raise ValueError(f"invalid block spec: {block_name!r}")
            block_inputs = block_spec.get("inputs", {})
            block_outputs = block_spec.get("outputs", {})
            block_graph = block_spec.get("graph", [])
            if (
                not isinstance(block_inputs, dict)
                or not isinstance(block_outputs, dict)
                or not isinstance(block_graph, list)
            ):
                raise ValueError(f"invalid block structure: {block_name!r}")
            lines.extend(
                _render_module(
                    module_name=str(block_name),
                    inputs=block_inputs,
                    outputs=block_outputs,
                    graph=block_graph,
                    symbols=symbols,
                )
            )
            lines.append("")
    lines.extend(
        _render_module(
            module_name=module_name,
            inputs=inputs,
            outputs=outputs,
            graph=graph,
            symbols=symbols,
        )
    )
    return "\n".join(lines) + "\n"


__all__ = ["synapse_spec_to_axon_module_text"]
