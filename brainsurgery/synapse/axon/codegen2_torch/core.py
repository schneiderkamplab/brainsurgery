from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ...mxfp4 import materialize_mxfp4_aliases
from ..ast import (
    AxonExprPath,
    DimExprBinary,
    TypeDim,
    TypeList,
    TypeNamed,
    TypeOptional,
    TypeTensor,
    TypeTuple,
    parse_type_expr,
    render_type,
)
from ..ast.types import dim_token_names
from ..ast.path import path_expr_to_runtime_value, resolve_path_expr_to_key
from ..codegen2_common import (
    cache_past_length,
    compose_path,
    execute_common_primitive,
    is_null,
    lookup_config,
    normalize_primitive_op,
    path_parts,
)
from ..graph_ir import (
    GraphLiteral,
    GraphExpr,
    GraphModule,
    GraphOperand,
    GraphPath,
    GraphProgram,
    GraphValueRef,
    validate_graph_program,
)


def _graph_path_payload(path: GraphPath) -> dict[str, Any]:
    return path_expr_to_runtime_value(
        AxonExprPath(absolute=path.absolute, parts=path.parts)
    )


def _graph_path_token(path: GraphPath, env: Mapping[str, Any]) -> str:
    payload = _graph_path_payload(path)
    key = resolve_path_expr_to_key(payload, env, op_name="graph path")
    return ("@@" if path.absolute else "@") + key


def _operand_payload(operand: GraphOperand) -> Any:
    if isinstance(operand, GraphValueRef):
        return operand.name
    if isinstance(operand, GraphLiteral):
        return operand.value
    if isinstance(operand, GraphPath):
        return _graph_path_payload(operand)
    if isinstance(operand, GraphExpr):
        return _graph_expr_payload(operand)
    if isinstance(operand, tuple):
        return [_operand_payload(item) for item in operand]
    raise TypeError(f"unsupported graph operand: {operand!r}")


def _normalize_primitive_op(name: str) -> str:
    return normalize_primitive_op(name)


def _module_inputs_spec(module: GraphModule) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for value in module.inputs:
        type_expr = value.type_expr
        optional = value.optional or isinstance(type_expr, TypeOptional)
        out[value.name] = {
            "type": render_type(type_expr),
            "optional": optional,
        }
    return out


def _module_outputs_spec(module: GraphModule) -> dict[str, Any]:
    out: dict[str, Any] = {}
    names = module.output_names or _fallback_output_names(module)
    if not names:
        names = tuple(f"out{idx}" for idx in range(len(module.outputs)))
    for name, operand in zip(names, module.outputs, strict=False):
        out[name] = _operand_payload(operand)
    return out


def _fallback_output_names(module: GraphModule) -> tuple[str, ...]:
    names: list[str] = []
    for idx, operand in enumerate(module.outputs):
        if isinstance(operand, GraphValueRef):
            names.append(_display_graph_name(operand.name))
        else:
            names.append("logits" if len(module.outputs) == 1 and idx == 0 else f"out{idx}")
    return tuple(names)


def _display_graph_name(name: str) -> str:
    base, marker, suffix = name.rpartition("__g")
    if marker and base and suffix.isdigit():
        return base
    return name


def _type_dim_names(type_expr: Any) -> set[str]:
    if isinstance(type_expr, TypeTensor):
        names: set[str] = set()
        for dim in type_expr.dims:
            names.update(dim_token_names(dim))
        return names
    if isinstance(type_expr, TypeOptional):
        return _type_dim_names(type_expr.inner)
    if isinstance(type_expr, TypeList):
        return _type_dim_names(type_expr.item)
    if isinstance(type_expr, TypeTuple):
        names: set[str] = set()
        for item in type_expr.items:
            names.update(_type_dim_names(item))
        return names
    if isinstance(type_expr, TypeNamed):
        names: set[str] = set()
        for dim in type_expr.args:
            names.update(dim_token_names(dim))
        return names
    return set()


def _is_global_symbol_module(module: GraphModule) -> bool:
    return not module.inputs and len(module.outputs) == 1


def _global_symbol_module_names(program: GraphProgram) -> set[str]:
    return {
        module.name
        for module in program.modules
        if _is_global_symbol_module(module)
    }


def _free_dim_refs_in_operand(operand: GraphOperand, *, local: set[str], out: set[str]) -> None:
    if isinstance(operand, GraphValueRef):
        if operand.name not in local and isinstance(operand.type_expr, TypeDim):
            out.add(operand.name)
        return
    if isinstance(operand, GraphExpr):
        for item in operand.inputs:
            _free_dim_refs_in_operand(item, local=local, out=out)
        for item in operand.attrs.values():
            _free_dim_refs_in_operand(item, local=local, out=out)
        return
    if isinstance(operand, tuple):
        for item in operand:
            _free_dim_refs_in_operand(item, local=local, out=out)


def _module_free_dim_refs(module: GraphModule, *, global_names: set[str]) -> set[str]:
    local = {value.name for value in module.inputs}
    out: set[str] = set()
    for node in module.nodes:
        for operand in node.inputs:
            _free_dim_refs_in_operand(operand, local=local | global_names, out=out)
        for operand in node.attrs.values():
            _free_dim_refs_in_operand(operand, local=local | global_names, out=out)
        local.update(value.name for value in node.outputs)
    for operand in module.outputs:
        _free_dim_refs_in_operand(operand, local=local | global_names, out=out)
    return out


def _bind_dim_expr_runtime(dim: Any, actual: int, symbols: dict[str, Any]) -> None:
    if isinstance(dim, str):
        symbols.setdefault(dim, actual)
        return
    if not isinstance(dim, DimExprBinary):
        return
    left = dim.left
    right = dim.right
    if dim.op == "+":
        if isinstance(left, str) and right in symbols:
            symbols.setdefault(left, actual - int(symbols[right]))
        if isinstance(right, str) and left in symbols:
            symbols.setdefault(right, actual - int(symbols[left]))
    if dim.op == "-":
        if isinstance(left, str) and right in symbols:
            symbols.setdefault(left, actual + int(symbols[right]))
        if isinstance(right, str) and left in symbols:
            symbols.setdefault(right, int(symbols[left]) - actual)
    if dim.op == "*":
        if isinstance(left, str) and right in symbols and int(symbols[right]) != 0:
            symbols.setdefault(left, actual // int(symbols[right]))
        if isinstance(right, str) and left in symbols and int(symbols[left]) != 0:
            symbols.setdefault(right, actual // int(symbols[left]))
    if dim.op == "/":
        if isinstance(left, str) and right in symbols:
            symbols.setdefault(left, actual * int(symbols[right]))
        if isinstance(right, str) and left in symbols and actual != 0:
            symbols.setdefault(right, int(symbols[left]) // actual)


def _bind_nested_shape_symbols_runtime(type_expr: Any, value: Any, symbols: dict[str, Any]) -> None:
    if isinstance(type_expr, TypeOptional):
        if value is not None:
            _bind_nested_shape_symbols_runtime(type_expr.inner, value, symbols)
        return
    if isinstance(type_expr, TypeList):
        if isinstance(value, list | tuple) and value:
            _bind_nested_shape_symbols_runtime(type_expr.item, value[0], symbols)
        return
    if isinstance(type_expr, TypeTuple):
        if isinstance(value, list | tuple):
            for item_type, item_value in zip(type_expr.items, value, strict=False):
                _bind_nested_shape_symbols_runtime(item_type, item_value, symbols)
        return
    if isinstance(type_expr, TypeTensor) and torch.is_tensor(value):
        for idx, dim in enumerate(type_expr.dims):
            if idx < value.dim():
                _bind_dim_expr_runtime(dim, int(value.shape[idx]), symbols)


def _effective_graph_value_type(value: GraphValue) -> TypeExpr:
    if value.optional and not isinstance(value.type_expr, TypeOptional):
        return TypeOptional(value.type_expr)
    return value.type_expr


def _emit_bind_dim_expr(
    lines: list[str],
    *,
    add: Any,
    dim: Any,
    actual_expr: str,
    local: set[str],
    indent: int,
    guaranteed: bool,
) -> None:
    if isinstance(dim, str):
        if dim.isidentifier() and dim not in local:
            add(lines, indent, f"{dim} = {actual_expr}")
            local.add(dim)
        return
    if not isinstance(dim, DimExprBinary):
        return
    left = dim.left
    right = dim.right
    if dim.op == "+":
        if isinstance(left, str) and left.isidentifier() and left not in local and isinstance(right, str) and right in local:
            add(lines, indent, f"{left} = {actual_expr} - {right}")
            local.add(left)
        if isinstance(right, str) and right.isidentifier() and right not in local and isinstance(left, str) and left in local:
            add(lines, indent, f"{right} = {actual_expr} - {left}")
            local.add(right)
        return
    if dim.op == "-":
        if isinstance(left, str) and left.isidentifier() and left not in local and isinstance(right, str) and right in local:
            add(lines, indent, f"{left} = {actual_expr} + {right}")
            local.add(left)
        if isinstance(right, str) and right.isidentifier() and right not in local and isinstance(left, str) and left in local:
            add(lines, indent, f"{right} = {left} - {actual_expr}")
            local.add(right)
        return
    if dim.op == "*":
        if isinstance(left, str) and left.isidentifier() and left not in local and isinstance(right, str) and right in local:
            add(lines, indent, f"{left} = {actual_expr} // {right}")
            local.add(left)
        if isinstance(right, str) and right.isidentifier() and right not in local and isinstance(left, str) and left in local:
            add(lines, indent, f"{right} = {actual_expr} // {left}")
            local.add(right)
        return
    if dim.op == "/":
        if isinstance(left, str) and left.isidentifier() and left not in local and isinstance(right, str) and right in local:
            add(lines, indent, f"{left} = {actual_expr} * {right}")
            local.add(left)
        if isinstance(right, str) and right.isidentifier() and right not in local and isinstance(left, str) and left in local:
            add(lines, indent, f"{right} = {left} // {actual_expr}")
            local.add(right)


def _emit_bind_nested_shape_symbols(
    lines: list[str],
    *,
    add: Any,
    type_expr: Any,
    value_expr: str,
    local: set[str],
) -> None:
    if isinstance(type_expr, TypeOptional):
        add(lines, 8, f"if {value_expr} is not None:")
        add(lines, 12, "pass")
        _emit_bind_nested_shape_symbols_inner(
            lines, add=add, type_expr=type_expr.inner, value_expr=value_expr, local=local, indent=12, guaranteed=False
        )
        return
    _emit_bind_nested_shape_symbols_inner(
        lines, add=add, type_expr=type_expr, value_expr=value_expr, local=local, indent=8, guaranteed=True
    )


def _emit_bind_nested_shape_symbols_inner(
    lines: list[str],
    *,
    add: Any,
    type_expr: Any,
    value_expr: str,
    local: set[str],
    indent: int,
    guaranteed: bool,
) -> None:
    if isinstance(type_expr, TypeOptional):
        add(lines, indent, f"if {value_expr} is not None:")
        add(lines, indent + 4, "pass")
        _emit_bind_nested_shape_symbols_inner(
            lines,
            add=add,
            type_expr=type_expr.inner,
            value_expr=value_expr,
            local=local,
            indent=indent + 4,
            guaranteed=False,
        )
        return
    if isinstance(type_expr, TypeList):
        add(lines, indent, f"if isinstance({value_expr}, (list, tuple)) and {value_expr}:")
        add(lines, indent + 4, "pass")
        _emit_bind_nested_shape_symbols_inner(
            lines,
            add=add,
            type_expr=type_expr.item,
            value_expr=f"{value_expr}[0]",
            local=local,
            indent=indent + 4,
            guaranteed=False,
        )
        return
    if isinstance(type_expr, TypeTuple):
        for idx, item_type in enumerate(type_expr.items):
            _emit_bind_nested_shape_symbols_inner(
                lines,
                add=add,
                type_expr=item_type,
                value_expr=f"{value_expr}[{idx}]",
                local=local,
                indent=indent,
                guaranteed=guaranteed,
            )
        return
    if isinstance(type_expr, TypeTensor):
        for idx, dim in enumerate(type_expr.dims):
            _emit_bind_dim_expr(
                lines,
                add=add,
                dim=dim,
                actual_expr=f"{value_expr}.shape[{idx}]",
                local=local,
                indent=indent,
                guaranteed=guaranteed,
            )


class Codegen2GraphModel(nn.Module):
    """Execute the typed Axon graph IR without Synapse graph specs."""

    GRAPH: GraphProgram

    def __init__(
        self,
        graph: GraphProgram | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
        *,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        graph = self.GRAPH if graph is None else graph
        validate_graph_program(graph)
        self.graph = graph
        self.modules_by_name = {module.name: module for module in graph.modules}
        self.global_symbol_names = _global_symbol_module_names(graph)
        main = self.modules_by_name[graph.main_module]
        self.main_inputs_spec = _module_inputs_spec(main)
        self.main_outputs_spec = _module_outputs_spec(main)
        super().__init__()
        self.spec = {
            "synapse": 1,
            "model": {
                "inputs": self.main_inputs_spec,
                "outputs": self.main_outputs_spec,
                "graph": [],
                "blocks": {},
                "symbols": {},
                "config": model_config or {},
            },
        }
        self._state: dict[str, torch.Tensor] = {}
        self._symbols: dict[str, Any] = {}
        if state_dict is not None:
            self.load_state_dict_tensors(state_dict)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        graph: GraphProgram | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> "Codegen2GraphModel":
        return cls(graph=graph, state_dict=state_dict, model_config=model_config)

    def load_state_dict_tensors(self, state_dict: dict[str, torch.Tensor]) -> None:
        loaded = dict(state_dict)
        materialize_mxfp4_aliases(loaded, drop_packed=True)
        self._state = loaded

    def _prepare_env(
        self,
        *,
        input_ids: torch.Tensor | None,
        inputs: dict[str, Any],
        input_specs: dict[str, Any],
    ) -> dict[str, Any]:
        env = {"input_ids": input_ids, **inputs} if input_ids is not None else dict(inputs)
        for input_name, input_spec in input_specs.items():
            optional = isinstance(input_spec, dict) and bool(input_spec.get("optional", False))
            if input_name in env:
                continue
            if optional:
                env[input_name] = None
                continue
            raise ValueError(f"Missing required input: {input_name}")
        return env

    def _path_template_env(
        self, env: Mapping[str, Any], *, symbols: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        out = dict(symbols or {})
        out.update(env)
        return out

    def _state_key_alternatives(self, key: str, *, limit: int = 8) -> list[str]:
        if not isinstance(key, str) or not key:
            return []
        keys = [item for item in self._state.keys() if isinstance(item, str)]
        leaf = key.split(".")[-1]
        out: list[str] = []
        for existing in keys:
            if existing == leaf or existing.endswith(f".{leaf}"):
                out.append(existing)
                if len(out) >= limit:
                    return out
        return out

    def _state_tensor_from_resolved_path(self, path: str, *, field: str) -> torch.Tensor:
        resolved = path[2:] if isinstance(path, str) and path.startswith("@@") else path
        if resolved not in self._state:
            alternatives = self._state_key_alternatives(resolved, limit=8)
            alt_text = ", ".join(alternatives) if alternatives else "<none>"
            raise ValueError(
                f"{field} tensor not found at path: {resolved}. Alternatives: {alt_text}"
            )
        return self._state[resolved]

    def _bind_shape_symbols_from_types(
        self,
        *,
        env: dict[str, Any],
        input_types: dict[str, str],
        symbols: dict[str, Any],
    ) -> None:
        def bind_value(type_expr: Any, value: Any) -> None:
            if isinstance(type_expr, TypeOptional):
                if value is not None:
                    bind_value(type_expr.inner, value)
                return
            if isinstance(type_expr, TypeList):
                if isinstance(value, (list, tuple)) and value:
                    bind_value(type_expr.item, value[0])
                return
            if isinstance(type_expr, TypeTuple):
                if isinstance(value, (list, tuple)):
                    for item_type, item_value in zip(type_expr.items, value, strict=False):
                        bind_value(item_type, item_value)
                return
            if not isinstance(type_expr, TypeTensor) or not torch.is_tensor(value):
                return
            for axis, dim in enumerate(type_expr.dims):
                if not isinstance(dim, str) or axis >= value.ndim:
                    continue
                actual = int(value.shape[axis])
                current = symbols.get(dim)
                if isinstance(current, int) and current != actual:
                    continue
                symbols[dim] = actual

        for input_name, type_text in input_types.items():
            try:
                type_expr = parse_type_expr(type_text)
            except Exception:
                continue
            bind_value(type_expr, env.get(input_name))

    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:
        env = self._prepare_env(
            input_ids=input_ids,
            inputs=inputs,
            input_specs=self.main_inputs_spec,
        )
        symbols = self._evaluate_global_symbols(env)
        self._symbols = symbols
        self._bind_shape_symbols_from_types(
            env=env,
            input_types={
                value.name: render_type(value.type_expr)
                for value in self.modules_by_name[self.graph.main_module].inputs
            },
            symbols=symbols,
        )
        result = self._execute_module(self.graph.main_module, env, symbols)
        names = self.modules_by_name[self.graph.main_module].output_names
        if not names:
            names = _fallback_output_names(self.modules_by_name[self.graph.main_module])
        if len(result) == 1:
            return result[0]
        outputs = {name: value for name, value in zip(names, result, strict=False)}
        return outputs

    def _evaluate_global_symbols(self, env: dict[str, Any]) -> dict[str, Any]:
        symbols: dict[str, Any] = {}
        pending = sorted(self.global_symbol_names)
        last_errors: dict[str, Exception] = {}
        while pending:
            next_pending: list[str] = []
            progressed = False
            for name in pending:
                snapshot = dict(symbols)
                try:
                    result = self._execute_module(name, dict(env), symbols)
                except Exception as exc:
                    symbols.clear()
                    symbols.update(snapshot)
                    last_errors[name] = exc
                    next_pending.append(name)
                    continue
                symbols[name] = result[0] if len(result) == 1 else result
                progressed = True
            if not progressed:
                name, exc = next(iter(last_errors.items()))
                raise RuntimeError(
                    f"unable to evaluate graph global symbols; blocked at {name!r}"
                ) from exc
            pending = next_pending
        return symbols

    def _eval_graph_operand(
        self,
        operand: GraphOperand,
        *,
        env: Mapping[str, Any],
        symbols: Mapping[str, Any],
    ) -> Any:
        if isinstance(operand, GraphValueRef):
            if operand.name in env:
                value = env[operand.name]
                if (
                    operand.type_expr.__class__.__name__ == "TypeDim"
                    and isinstance(value, list | tuple)
                    and value
                ):
                    return value[-1]
                return value
            if operand.name in symbols:
                value = symbols[operand.name]
                if (
                    operand.type_expr.__class__.__name__ == "TypeDim"
                    and isinstance(value, list | tuple)
                    and value
                ):
                    return value[-1]
                return value
            raise ValueError(f"undefined graph value {operand.name!r}")
        if isinstance(operand, GraphLiteral):
            return operand.value
        if isinstance(operand, GraphPath):
            return _graph_path_token(
                operand,
                self._path_template_env({**symbols, **env}, symbols=symbols),
            )
        if isinstance(operand, GraphExpr):
            return self._eval_graph_expr(operand, env=env, symbols=symbols)
        if isinstance(operand, tuple):
            return [self._eval_graph_operand(item, env=env, symbols=symbols) for item in operand]
        raise TypeError(f"unsupported graph operand: {operand!r}")

    def _eval_graph_expr(
        self,
        expr: GraphExpr,
        *,
        env: Mapping[str, Any],
        symbols: dict[str, Any],
    ) -> Any:
        scratch = dict(env)
        out_name = "__expr"
        pseudo_node = type(
            "_PseudoGraphNode",
            (),
            {
                "op": expr.op,
                "inputs": expr.inputs,
                "attrs": expr.attrs,
                "outputs": [type("_PseudoGraphValue", (), {"name": out_name})()],
            },
        )()
        self._execute_node(pseudo_node, env=scratch, symbols=symbols)
        return scratch[out_name]

    def _eval_expr(self, expr: Any, env: dict[str, Any], symbols: dict[str, Any]) -> Any:
        if expr is None or isinstance(expr, (int, float, bool)):
            value = expr
        elif isinstance(expr, list):
            value = [self._eval_expr(item, env, symbols) for item in expr]
        elif isinstance(expr, tuple):
            value = tuple(self._eval_expr(item, env, symbols) for item in expr)
        elif isinstance(expr, dict):
            kind = expr.get("_expr")
            if kind == "name":
                ident = expr.get("id", expr.get("value"))
                if not isinstance(ident, str) or not ident:
                    raise ValueError(f"Invalid name expression payload: {expr!r}")
                if ident in env:
                    value = env[ident]
                elif ident in symbols:
                    value = self._eval_expr(symbols[ident], env, symbols)
                else:
                    raise ValueError(f"Unknown symbol in expression: {ident}")
            elif kind == "tuple":
                items = expr.get("items")
                if not isinstance(items, list):
                    raise ValueError(f"Invalid tuple expression payload: {expr!r}")
                value = tuple(self._eval_expr(item, env, symbols) for item in items)
            elif kind == "binary":
                op = expr.get("op")
                left = self._eval_expr(expr.get("left"), env, symbols)
                right = self._eval_expr(expr.get("right"), env, symbols)
                if op == "+":
                    value = left + right
                elif op == "-":
                    value = left - right
                elif op == "*":
                    value = left * right
                elif op == "/":
                    value = left / right
                elif op == "%":
                    value = left % right
                elif op == "==":
                    value = left == right
                elif op == "!=":
                    value = left != right
                elif op == "<":
                    value = left < right
                elif op == "<=":
                    value = left <= right
                elif op == ">":
                    value = left > right
                elif op == ">=":
                    value = left >= right
                elif op == "and":
                    value = bool(left) and bool(right)
                elif op == "or":
                    value = bool(left) or bool(right)
                else:
                    raise ValueError(f"Unsupported binary operator in expression: {op!r}")
            else:
                value = {key: self._eval_expr(item, env, symbols) for key, item in expr.items()}
        elif isinstance(expr, str):
            token = expr.strip()
            if token in env:
                value = env[token]
            elif token in symbols:
                value = self._eval_expr(symbols[token], env, symbols)
            elif token.lower() == "null":
                value = None
            elif token.lower() == "true":
                value = True
            elif token.lower() == "false":
                value = False
            else:
                try:
                    value = int(token)
                except ValueError:
                    try:
                        value = float(token)
                    except ValueError:
                        value = token
        else:
            value = expr
        if isinstance(expr, str) and isinstance(value, list | tuple) and value:
            return value[-1]
        if (
            isinstance(expr, dict)
            and expr.get("_expr") == "name"
            and isinstance(value, list | tuple)
            and value
        ):
            return value[-1]
        return value

    def _operand_arg(self, operand: GraphOperand) -> Any:
        return _operand_payload(operand)

    @staticmethod
    def _is_null(value: Any) -> bool:
        return is_null(value)

    @staticmethod
    def _cache_past_length(cache: Any) -> int:
        return cache_past_length(cache)

    @staticmethod
    def _path_parts(value: Any) -> tuple[bool, str]:
        return path_parts(value)

    def _compose_path(self, base: Any, leaf: Any) -> str:
        return compose_path(base, leaf)

    def _required_param(self, path: str, *, field: str) -> torch.Tensor:
        return self._state_tensor_from_resolved_path(path, field=field)

    def _optional_param(self, path: str) -> torch.Tensor | None:
        value = self._state.get(path)
        return value if torch.is_tensor(value) else None

    @staticmethod
    def _dtype_from_name(value: Any) -> torch.dtype:
        if value is None:
            raise ValueError("dtype name is null")
        token = str(value).strip().lower()
        if token in {"float32", "fp32"}:
            return torch.float32
        if token in {"float16", "fp16", "half"}:
            return torch.float16
        if token in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if token in {"int64", "long"}:
            return torch.int64
        if token in {"int32", "int"}:
            return torch.int32
        if token in {"bool", "boolean"}:
            return torch.bool
        raise ValueError(f"unsupported dtype name {value!r}")

    def _read_config_key(self, path: Any, *, env: Mapping[str, Any], symbols: Mapping[str, Any]) -> str:
        if isinstance(path, GraphPath):
            payload = _graph_path_payload(path)
        else:
            value = self._eval_graph_operand(path, env=env, symbols=symbols)
            payload = value
        key = resolve_path_expr_to_key(payload, {}, op_name="codegen2-torch config path")
        return key

    @staticmethod
    def _lookup_config(config: Any, key: str) -> tuple[bool, Any]:
        return lookup_config(config, key)

    def _execute_primitive_node(
        self,
        primitive: str,
        node: Any,
        *,
        env: dict[str, Any],
        symbols: dict[str, Any],
    ) -> bool:
        out_names = tuple(value.name for value in node.outputs)
        args = [self._eval_graph_operand(operand, env=env, symbols=symbols) for operand in node.inputs]
        kwargs = {
            key: self._eval_graph_operand(value, env=env, symbols=symbols)
            for key, value in node.attrs.items()
        }

        def out(value: Any) -> None:
            self._assign_outputs(out_names, value, env)

        handled, value = execute_common_primitive(
            primitive=primitive,
            args=args,
            kwargs=kwargs,
            config=self.spec.get("model", {}).get("config", {}),
            state_keys=lambda: self._state.keys(),
            require_param=lambda key: self._required_param(key, field=primitive),
        )
        if handled:
            out(value)
            return True

        if primitive == "embedding":
            if len(args) < 2:
                raise ValueError("embedding expects path and input")
            base, x = args[0], args[1]
            weight = self._required_param(self._compose_path(base, "@weight"), field="embedding.weight")
            if weight.device != x.device:
                weight = weight.to(device=x.device)
            out(F.embedding(x, weight))
            return True

        if primitive == "linear":
            if len(args) < 2:
                raise ValueError("linear expects path and input")
            base = args[0]
            x = args[1]
            bias_flag = bool(args[3]) if len(args) > 3 and not self._is_null(args[3]) else False
            transpose = bool(args[4]) if len(args) > 4 and not self._is_null(args[4]) else False
            expert = None if len(args) <= 5 or self._is_null(args[5]) else int(args[5])
            weight_leaf = args[6] if len(args) > 6 and not self._is_null(args[6]) else "@weight"
            bias_leaf = args[7] if len(args) > 7 and not self._is_null(args[7]) else "@bias"
            weight = self._required_param(self._compose_path(base, weight_leaf), field="linear.weight")
            if expert is not None:
                weight = weight[expert]
            bias = None
            if bias_flag:
                bias = self._optional_param(self._compose_path(base, bias_leaf))
                if bias is not None and expert is not None and bias.ndim >= 2:
                    bias = bias[expert]
            weight_run = weight.to(dtype=x.dtype) if x.is_floating_point() and weight.is_floating_point() and x.dtype != weight.dtype else weight
            bias_run = (
                bias.to(dtype=x.dtype)
                if bias is not None and x.is_floating_point() and bias.is_floating_point() and bias.dtype != x.dtype
                else bias
            )
            if transpose:
                out(torch.matmul(x, weight_run) + (bias_run if bias_run is not None else 0))
            else:
                out(F.linear(x, weight_run, bias_run))
            return True

        if primitive == "layernorm":
            if len(args) < 2:
                raise ValueError("layernorm expects path and input")
            base = args[0]
            x = args[1]
            eps = float(args[2]) if len(args) > 2 and not self._is_null(args[2]) else 1e-5
            weight_leaf = args[4] if len(args) > 4 and not self._is_null(args[4]) else "@weight"
            bias_flag = bool(args[5]) if len(args) > 5 and not self._is_null(args[5]) else True
            bias_leaf = args[6] if len(args) > 6 and not self._is_null(args[6]) else "@bias"
            weight = self._required_param(self._compose_path(base, weight_leaf), field="layernorm.weight")
            bias = self._optional_param(self._compose_path(base, bias_leaf)) if bias_flag else None
            out(F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=eps))
            return True

        if primitive == "rmsnorm":
            if len(args) < 1:
                raise ValueError("rmsnorm expects input")
            x = args[0]
            eps = float(args[1]) if len(args) > 1 and not self._is_null(args[1]) else 1e-6
            cast_float = bool(args[3]) if len(args) > 3 and not self._is_null(args[3]) else False
            x_calc = x.float() if cast_float else x
            y = x_calc * torch.rsqrt(torch.mean(x_calc * x_calc, dim=-1, keepdim=True) + eps)
            if cast_float:
                y = y.to(dtype=x.dtype)
            out(y)
            return True

        if primitive in {"activations_gelu", "activations_gelu_new", "activations_gelu_pytorch_tanh"}:
            x = args[0]
            if primitive == "activations_gelu":
                out(F.gelu(x))
            else:
                out(0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))))
            return True
        if primitive == "activations_tanh":
            out(torch.tanh(args[0]))
            return True
        if primitive == "activations_silu":
            out(F.silu(args[0]))
            return True
        if primitive == "activations_sigmoid":
            out(torch.sigmoid(args[0]))
            return True
        if primitive == "l2norm":
            x = args[0]
            eps = float(args[1]) if len(args) > 1 and not self._is_null(args[1]) else 1e-6
            x_float = x.float() if x.is_floating_point() else x
            mean_squared = torch.mean(x_float * x_float, dim=-1, keepdim=True) + eps
            out((x_float * torch.pow(mean_squared, -0.5)).to(dtype=x.dtype))
            return True
        if primitive == "activations_xielu":
            if len(args) != 5:
                raise ValueError("activations_xielu expects exactly 5 positional args")
            out(self._xielu(args[0], args[1], args[2], args[3], args[4]))
            return True

        if primitive == "reshape":
            out(torch.reshape(args[0], tuple(int(v) for v in args[1])))
            return True
        if primitive == "arange":
            if len(args) < 3:
                raise ValueError("arange expects reference, start, end")
            ref, start, end = args[:3]
            device = ref.device if torch.is_tensor(ref) else None
            out(torch.arange(int(start), int(end), device=device))
            return True
        if primitive == "slice":
            x, dim, start, end = args[:4]
            rank = x.dim()
            dim = int(dim)
            if dim < 0:
                dim += rank
            sl = [slice(None)] * rank
            sl[dim] = slice(int(start), int(end))
            out(x[tuple(sl)])
            return True
        if primitive == "concat":
            if "dim" in kwargs:
                items = args[0] if len(args) == 1 and isinstance(args[0], list | tuple) else args
                dim = kwargs["dim"]
            elif len(args) == 2 and isinstance(args[0], list | tuple):
                items = list(args[0])
                dim = args[1]
            else:
                *items, dim = args
            out(torch.cat(list(items), dim=int(dim)))
            return True
        if primitive == "repeat":
            x, repeats, dim = args[:3]
            repeats = int(repeats)
            dim = int(dim)
            if dim < 0:
                dim += x.dim()
            out(x if repeats == 1 else torch.repeat_interleave(x, repeats=repeats, dim=dim))
            return True
        if primitive == "expand":
            x, shape = args[:2]
            out(x.expand(tuple(int(v) for v in shape)))
            return True
        if primitive == "permute":
            x, dims = args[:2]
            out(torch.permute(x, tuple(int(v) for v in dims)))
            return True
        if primitive == "transpose":
            x, dim0, dim1 = args[:3]
            out(torch.transpose(x, int(dim0), int(dim1)))
            return True
        if primitive == "unsqueeze":
            out(torch.unsqueeze(args[0], int(args[1])))
            return True
        if primitive == "matmul":
            out(torch.matmul(args[0], args[1]))
            return True
        if primitive == "softmax":
            x = args[0]
            dim = int(args[1]) if len(args) > 1 and not self._is_null(args[1]) else -1
            dtype = torch.float32 if len(args) > 2 and args[2] == "float32" else None
            out(F.softmax(x, dim=dim, dtype=dtype))
            return True
        if primitive == "sum":
            x = args[0]
            dim = int(args[1]) if len(args) > 1 and not self._is_null(args[1]) else -1
            keepdim = bool(args[2]) if len(args) > 2 and not self._is_null(args[2]) else False
            out(torch.sum(x, dim=dim, keepdim=keepdim))
            return True
        if primitive == "activations_gegelu":
            x = args[0]
            if x.shape[-1] % 2 != 0:
                raise ValueError("gegelu requires even last dimension")
            x_gelu = x[..., ::2]
            x_linear = x[..., 1::2]
            limit = args[1] if len(args) > 1 and not self._is_null(args[1]) else None
            if limit is not None:
                limit = float(limit)
                x_gelu = torch.where(torch.isinf(x_gelu), x_gelu, x_gelu.clamp(max=limit))
                x_linear = torch.where(
                    torch.isinf(x_linear), x_linear, x_linear.clamp(min=-limit, max=limit)
                )
            out(x_gelu * torch.sigmoid(1.702 * x_gelu) * (x_linear + 1.0))
            return True
        if primitive == "where":
            out(torch.where(args[0], args[1], args[2]) if torch.is_tensor(args[0]) else (args[1] if args[0] else args[2]))
            return True
        if primitive == "gather":
            dim = int(args[2]) if len(args) > 2 and not self._is_null(args[2]) else -1
            out(torch.gather(args[0], dim=dim, index=args[1]))
            return True
        if primitive == "scatter":
            dim = int(args[3]) if len(args) > 3 and not self._is_null(args[3]) else -1
            if torch.is_tensor(args[2]):
                out(torch.scatter(args[0], dim=dim, index=args[1], src=args[2]))
            else:
                out(torch.scatter(args[0], dim=dim, index=args[1], value=args[2]))
            return True
        if primitive == "index_add":
            dim = int(args[3]) if len(args) > 3 and not self._is_null(args[3]) else 0
            out(torch.index_add(args[0], dim=dim, index=args[1], source=args[2]))
            return True
        if primitive == "clamp":
            x = args[0]
            min_value = args[1] if len(args) > 1 and not self._is_null(args[1]) else kwargs.get("min")
            max_value = args[2] if len(args) > 2 and not self._is_null(args[2]) else kwargs.get("max")
            min_value = None if self._is_null(min_value) else min_value
            max_value = None if self._is_null(max_value) else max_value
            out(torch.clamp(x, min=min_value, max=max_value))
            return True
        if primitive == "le":
            out(args[0] <= args[1])
            return True
        if primitive == "eq":
            left, right = args[0], args[1]
            out(torch.eq(left, right) if torch.is_tensor(left) or torch.is_tensor(right) else left == right)
            return True
        if primitive == "and":
            out(torch.logical_and(args[0], args[1]))
            return True
        if primitive == "add":
            out(args[0] + args[1])
            return True
        if primitive == "mul":
            out(args[0] * args[1])
            return True
        if primitive == "pow":
            out(torch.pow(args[0], args[1]) if torch.is_tensor(args[0]) else args[0] ** args[1])
            return True
        if primitive == "div":
            out(args[0] / args[1])
            return True
        if primitive == "floor":
            out(torch.floor(args[0]) if torch.is_tensor(args[0]) else int(args[0] // 1))
            return True
        if primitive == "sqrt":
            value = args[0]
            out(torch.sqrt(value) if torch.is_tensor(value) else math.sqrt(float(value)))
            return True
        if primitive == "sin":
            out(torch.sin(args[0]) if torch.is_tensor(args[0]) else math.sin(float(args[0])))
            return True
        if primitive == "cos":
            out(torch.cos(args[0]) if torch.is_tensor(args[0]) else math.cos(float(args[0])))
            return True
        if primitive == "exp":
            out(torch.exp(args[0]) if torch.is_tensor(args[0]) else math.exp(float(args[0])))
            return True
        if primitive == "log":
            out(torch.log(args[0]) if torch.is_tensor(args[0]) else math.log(float(args[0])))
            return True
        if primitive == "cast":
            x, dtype_name = args[:2]
            out(x.to(dtype=self._dtype_from_name(dtype_name)))
            return True
        if primitive == "cast_like":
            out(args[0].to(dtype=args[1].dtype))
            return True
        if primitive == "cumsum":
            x = args[0]
            dim = int(args[1]) if len(args) > 1 and not self._is_null(args[1]) else -1
            dtype = None if len(args) < 3 or self._is_null(args[2]) else self._dtype_from_name(args[2])
            out(torch.cumsum(x, dim=dim, dtype=dtype))
            return True
        if primitive == "dtype_value":
            info = torch.finfo(args[0].dtype)
            kind = str(args[1]).strip().lower()
            out({"min": info.min, "max": info.max, "eps": info.eps, "tiny": info.tiny, "inf": float("inf"), "-inf": float("-inf")}[kind])
            return True
        if primitive == "empty_like":
            dtype = None if len(args) < 2 or self._is_null(args[1]) else self._dtype_from_name(args[1])
            out(torch.empty_like(args[0], dtype=dtype) if dtype is not None else torch.empty_like(args[0]))
            return True
        if primitive == "fill":
            dtype = None if len(args) < 3 or self._is_null(args[2]) else self._dtype_from_name(args[2])
            out(torch.full_like(args[0], args[1], dtype=dtype if dtype is not None else args[0].dtype))
            return True
        if primitive == "empty":
            ref, shape = args[:2]
            dtype = None if len(args) < 3 or self._is_null(args[2]) else self._dtype_from_name(args[2])
            out(torch.empty(tuple(int(v) for v in shape), device=ref.device, dtype=dtype or ref.dtype))
            return True
        if primitive == "zeros":
            ref, shape = args[:2]
            dtype = None if len(args) < 3 or self._is_null(args[2]) else self._dtype_from_name(args[2])
            out(torch.zeros(tuple(int(v) for v in shape), device=ref.device, dtype=dtype or ref.dtype))
            return True
        if primitive == "full":
            ref, shape, value = args[:3]
            dtype = None if len(args) < 4 or self._is_null(args[3]) else self._dtype_from_name(args[3])
            out(torch.full(tuple(int(v) for v in shape), value, device=ref.device, dtype=dtype or ref.dtype))
            return True
        if primitive == "zeros_like":
            out(torch.zeros_like(args[0]))
            return True
        return False

    def _execute_module(
        self,
        module_name: str,
        env: dict[str, Any],
        symbols: dict[str, Any],
    ) -> tuple[Any, ...]:
        module = self.modules_by_name[module_name]
        local_env = dict(env)
        dim_names: set[str] = set()
        for value in module.inputs:
            dim_names.update(_type_dim_names(value.type_expr))
            for dim in value.dims or ():
                dim_names.update(dim_token_names(dim))
        local_symbols = dict(symbols)
        for name in dim_names:
            if name not in self.global_symbol_names:
                local_symbols.pop(name, None)
            value = local_env.get(name)
            if isinstance(value, int | float) and not isinstance(value, bool):
                local_symbols[name] = int(value)
        dim_params = [
            value.name
            for value in module.inputs
            if isinstance(value.type_expr, TypeDim) and value.name in local_env
        ]
        if len(dim_params) == 1:
            dim_value = local_env[dim_params[0]]
            for name in _module_free_dim_refs(module, global_names=self.global_symbol_names):
                local_symbols.setdefault(name, dim_value)
        self._bind_shape_symbols_from_types(
            env=local_env,
            input_types={
                value.name: render_type(_effective_graph_value_type(value))
                for value in module.inputs
            },
            symbols=local_symbols,
        )
        for value in module.inputs:
            _bind_nested_shape_symbols_runtime(
                _effective_graph_value_type(value), local_env.get(value.name), local_symbols
            )
        for name in dim_names:
            value = local_env.get(name)
            if isinstance(value, list | tuple) and value:
                local_env[name] = value[-1]
            value = local_symbols.get(name)
            if isinstance(value, list | tuple) and value:
                local_symbols[name] = value[-1]
        for node in module.nodes:
            try:
                self._execute_node(node, env=local_env, symbols=local_symbols)
            except Exception as exc:
                raise RuntimeError(
                    f"runtime2-torch graph node {node.id} ({node.op.name}) failed in {module_name}"
                ) from exc
        return tuple(
            self._eval_graph_operand(operand, env=local_env, symbols=local_symbols)
            for operand in module.outputs
        )

    def _execute_node(
        self,
        node: Any,
        *,
        env: dict[str, Any],
        symbols: dict[str, Any],
    ) -> None:
        op = node.op.name
        out_names = tuple(value.name for value in node.outputs)
        if op == "core.alias":
            values = [
                self._eval_graph_operand(operand, env=env, symbols=symbols)
                for operand in node.inputs
            ]
            self._assign_outputs(out_names, values[0] if len(values) == 1 else tuple(values), env)
            return
        if op == "core.tuple":
            self._assign_outputs(
                out_names,
                tuple(
                    self._eval_graph_operand(operand, env=env, symbols=symbols)
                    for operand in node.inputs
                ),
                env,
            )
            return
        if op == "core.list":
            self._assign_outputs(
                out_names,
                [
                    self._eval_graph_operand(operand, env=env, symbols=symbols)
                    for operand in node.inputs
                ],
                env,
            )
            return
        if op == "core.ascribe":
            if len(node.inputs) != 1:
                raise ValueError("core.ascribe expects one input")
            self._assign_outputs(
                out_names,
                self._eval_graph_operand(node.inputs[0], env=env, symbols=symbols),
                env,
            )
            return
        if op == "core.select":
            cond, true_value, false_value = node.inputs
            selected = true_value if bool(
                self._eval_graph_operand(cond, env=env, symbols=symbols)
            ) else false_value
            self._assign_outputs(
                out_names,
                self._eval_graph_operand(selected, env=env, symbols=symbols),
                env,
            )
            return
        if op.startswith("core.binary."):
            operator = op.removeprefix("core.binary.")
            left = self._eval_graph_operand(node.inputs[0], env=env, symbols=symbols)
            right = self._eval_graph_operand(node.inputs[1], env=env, symbols=symbols)
            self._assign_outputs(out_names, self._eval_binary(operator, left, right), env)
            return
        if op in {"Cache.past_length", "Cache.past_length_kv"}:
            if len(node.inputs) != 1:
                raise ValueError(f"{op} expects one argument")
            cache = self._eval_graph_operand(node.inputs[0], env=env, symbols=symbols)
            self._assign_outputs(out_names, self._cache_past_length(cache), env)
            return
        if op in self.modules_by_name:
            args = [
                self._eval_graph_operand(operand, env=env, symbols=symbols)
                for operand in node.inputs
            ]
            callee = self.modules_by_name[op]
            call_env = {
                value.name: arg for value, arg in zip(callee.inputs, args, strict=False)
            }
            for key, operand in node.attrs.items():
                if key in call_env:
                    raise ValueError(f"duplicate graph call argument {key!r} for {op!r}")
                call_env[key] = self._eval_graph_operand(operand, env=env, symbols=symbols)
            result = self._execute_module(op, call_env, symbols)
            self._assign_outputs(out_names, result[0] if len(result) == 1 else result, env)
            return

        if op == "_sqrt":
            value = self._eval_graph_operand(node.inputs[0], env=env, symbols=symbols)
            if isinstance(value, list | tuple) and value:
                value = value[-1]
            self._assign_outputs(out_names, torch.sqrt(torch.tensor(float(value))).item(), env)
            return
        if op == "_reshape":
            src = self._eval_graph_operand(node.inputs[0], env=env, symbols=symbols)
            shape = self._eval_graph_operand(node.inputs[1], env=env, symbols=symbols)
            if not isinstance(shape, list | tuple) or not shape:
                raise ValueError("reshape.shape must be a non-empty list")
            try:
                reshaped = torch.reshape(src, tuple(int(item) for item in shape))
            except Exception as exc:
                src_shape = tuple(src.shape) if torch.is_tensor(src) else type(src).__name__
                debug_symbols = {
                    key: symbols[key]
                    for key in ("B", "S", "P", "K", "PD", "NL")
                    if key in symbols
                }
                debug_env = {
                    key: env[key]
                    for key in ("B", "S", "P", "K", "PD", "NL")
                    if key in env
                }
                raise RuntimeError(
                    f"reshape from {src_shape} to {tuple(shape)!r} failed; "
                    f"env={debug_env!r}; symbols={debug_symbols!r}"
                ) from exc
            self._assign_outputs(out_names, reshaped, env)
            return

        primitive = _normalize_primitive_op(op)
        if self._execute_primitive_node(primitive, node, env=env, symbols=symbols):
            return
        raise NotImplementedError(f"codegen2-torch unsupported graph op {op!r}")

    def _assign_outputs(
        self,
        names: tuple[str, ...],
        value: Any,
        env: dict[str, Any],
    ) -> None:
        if len(names) == 1:
            env[names[0]] = value
            return
        if not isinstance(value, (tuple, list)) or len(value) != len(names):
            raise ValueError(f"cannot assign {value!r} to outputs {names!r}")
        for name, item in zip(names, value, strict=True):
            env[name] = item

    @staticmethod
    def _eval_binary(op: str, left: Any, right: Any) -> Any:
        if op == "+":
            if left is None:
                return right
            if right is None:
                return left
            return left + right
        if op == "-":
            if right is None:
                return left
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "%":
            return left % right
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "and":
            return bool(left) and bool(right)
        if op == "or":
            return bool(left) or bool(right)
        raise NotImplementedError(f"unsupported codegen2-torch binary op {op!r}")


def _py_ident(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if not out or out[0].isdigit():
        out = "_" + out
    return out


class _DirectTorchEmitter:
    def __init__(self, *, program: GraphProgram, class_name: str) -> None:
        self.program = program
        self.class_name = class_name
        self.modules_by_name = {module.name: module for module in program.modules}
        self.method_names = {name: f"_def_{_py_ident(name)}" for name in self.modules_by_name}
        self.global_symbol_names = _global_symbol_module_names(program)

    def emit(self) -> str:
        lines: list[str] = [f"class {self.class_name}(nn.Module):"]
        self._emit_common(lines)
        lines.append("")
        self._emit_eval_symbols(lines)
        for module in self.program.modules:
            lines.append("")
            self._emit_module(lines, module)
        lines.append("")
        self._emit_forward(lines)
        lines.append("")
        self._emit_generate(lines)
        return "\n".join(lines)

    @staticmethod
    def _add(lines: list[str], indent: int, text: str = "") -> None:
        lines.append(" " * indent + text)

    def _emit_common(self, lines: list[str]) -> None:
        add = self._add
        add(lines, 4, "def __init__(self, state_dict: dict[str, torch.Tensor], config: dict | None = None, param_devices=None):")
        add(lines, 8, "super().__init__()")
        add(lines, 8, "self.param_devices = self._normalize_param_devices(param_devices)")
        add(lines, 8, "self.state_dict_tensors = self._place_state_dict(dict(state_dict), self.param_devices)")
        add(lines, 8, "self.config = dict(({} if _MODEL_CONFIG is None else _MODEL_CONFIG) if config is None else config)")
        add(lines, 8, "self._symbols = self._eval_symbols()")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def from_state_dict(cls, state_dict, *, graph=None, model_config=None, param_devices=None):")
        add(lines, 8, "return cls(state_dict, config=_MODEL_CONFIG if model_config is None else model_config, param_devices=param_devices)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _normalize_param_devices(param_devices):")
        add(lines, 8, "if param_devices is None:")
        add(lines, 12, "return []")
        add(lines, 8, "devices = [torch.device(item) for item in param_devices]")
        add(lines, 8, "return devices")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _first_numeric_path_segment(key):")
        add(lines, 8, "for part in str(key).split('.'):")
        add(lines, 12, "if part.isdigit():")
        add(lines, 16, "return int(part)")
        add(lines, 8, "return None")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _param_placement_plan(cls, keys, devices):")
        add(lines, 8, "if not devices:")
        add(lines, 12, "return {}")
        add(lines, 8, "layer_ids = sorted({idx for key in keys if (idx := cls._first_numeric_path_segment(key)) is not None})")
        add(lines, 8, "if not layer_ids:")
        add(lines, 12, "return {str(key): devices[0] for key in keys}")
        add(lines, 8, "layer_to_device = {layer_id: devices[min(len(devices) - 1, pos * len(devices) // len(layer_ids))] for pos, layer_id in enumerate(layer_ids)}")
        add(lines, 8, "plan = {}")
        add(lines, 8, "for key in keys:")
        add(lines, 12, "idx = cls._first_numeric_path_segment(key)")
        add(lines, 12, "plan[str(key)] = layer_to_device[idx] if idx is not None else devices[0]")
        add(lines, 8, "return plan")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _place_state_dict(cls, state_dict, devices):")
        add(lines, 8, "plan = cls._param_placement_plan(state_dict.keys(), devices)")
        add(lines, 8, "if not plan:")
        add(lines, 12, "return state_dict")
        add(lines, 8, "placed = {}")
        add(lines, 8, "placement_cache = {}")
        add(lines, 8, "for key, value in state_dict.items():")
        add(lines, 12, "device = plan[str(key)]")
        add(lines, 12, "if not torch.is_tensor(value):")
        add(lines, 16, "placed[key] = value")
        add(lines, 16, "continue")
        add(lines, 12, "cache_key = (id(value), str(device))")
        add(lines, 12, "cached = placement_cache.get(cache_key)")
        add(lines, 12, "if cached is None:")
        add(lines, 16, "cached = value if value.device == device else value.to(device=device)")
        add(lines, 16, "placement_cache[cache_key] = cached")
        add(lines, 12, "placed[key] = cached")
        add(lines, 8, "return placed")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _move_to(cls, value, device):")
        add(lines, 8, "if torch.is_tensor(value):")
        add(lines, 12, "return value if value.device == device else value.to(device=device)")
        add(lines, 8, "if isinstance(value, tuple):")
        add(lines, 12, "return tuple(cls._move_to(item, device) for item in value)")
        add(lines, 8, "if isinstance(value, list):")
        add(lines, 12, "return [cls._move_to(item, device) for item in value]")
        add(lines, 8, "return value")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _align_pair(cls, left, right, *, prefer='right'):")
        add(lines, 8, "left_tensor = torch.is_tensor(left)")
        add(lines, 8, "right_tensor = torch.is_tensor(right)")
        add(lines, 8, "if not left_tensor or not right_tensor or left.device == right.device:")
        add(lines, 12, "return left, right")
        add(lines, 8, "device = right.device if prefer == 'right' else left.device")
        add(lines, 8, "return cls._move_to(left, device), cls._move_to(right, device)")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _binary_op(cls, op, left, right, *, prefer='right'):")
        add(lines, 8, "left, right = cls._align_pair(left, right, prefer=prefer)")
        add(lines, 8, "if op == '+':")
        add(lines, 12, "return left + right")
        add(lines, 8, "if op == '-':")
        add(lines, 12, "return left - right")
        add(lines, 8, "if op == '*':")
        add(lines, 12, "return left * right")
        add(lines, 8, "if op == '/':")
        add(lines, 12, "return left / right")
        add(lines, 8, "if op == '%':")
        add(lines, 12, "return left % right")
        add(lines, 8, "if op == '<=':")
        add(lines, 12, "return left <= right")
        add(lines, 8, "if op == '<':")
        add(lines, 12, "return left < right")
        add(lines, 8, "if op == '>=':")
        add(lines, 12, "return left >= right")
        add(lines, 8, "if op == '>':")
        add(lines, 12, "return left > right")
        add(lines, 8, "if op == '==':")
        add(lines, 12, "if left is None or right is None:")
        add(lines, 16, "return left is right")
        add(lines, 12, "return torch.eq(left, right) if torch.is_tensor(left) or torch.is_tensor(right) else left == right")
        add(lines, 8, "if op == '!=':")
        add(lines, 12, "if left is None or right is None:")
        add(lines, 16, "return left is not right")
        add(lines, 12, "return torch.ne(left, right) if torch.is_tensor(left) or torch.is_tensor(right) else left != right")
        add(lines, 8, "raise NotImplementedError(f'unsupported binary op {op!r}')")
        add(lines, 4, "")
        add(lines, 4, "_compose_path = staticmethod(_common_compose_path)")
        add(lines, 4, "_render_path = staticmethod(_common_render_path)")
        add(lines, 4, "_require_value = staticmethod(_common_require_value)")
        add(lines, 4, "")
        add(lines, 4, "def _param(self, path):")
        add(lines, 8, "return _common_required_state_value(self.state_dict_tensors, path)")
        add(lines, 4, "")
        add(lines, 4, "def _optional_param(self, path):")
        add(lines, 8, "return _common_optional_state_value(self.state_dict_tensors, path)")
        add(lines, 4, "")
        add(lines, 4, "def _linear(self, base, x, bias=False, transpose=False, expert=None, weight_leaf='weight', bias_leaf='bias'):")
        add(lines, 8, "weight = self._param(self._compose_path(base, weight_leaf))")
        add(lines, 8, "if expert is not None:")
        add(lines, 12, "weight = weight[int(expert)]")
        add(lines, 8, "x = self._move_to(x, weight.device)")
        add(lines, 8, "bias_value = self._optional_param(self._compose_path(base, bias_leaf)) if bias else None")
        add(lines, 8, "if bias_value is not None and expert is not None and bias_value.ndim >= 2:")
        add(lines, 12, "bias_value = bias_value[int(expert)]")
        add(lines, 8, "if bias_value is not None:")
        add(lines, 12, "bias_value = self._move_to(bias_value, weight.device)")
        add(lines, 8, "weight_run = weight.to(dtype=x.dtype) if x.is_floating_point() and weight.is_floating_point() and x.dtype != weight.dtype else weight")
        add(lines, 8, "bias_run = bias_value.to(dtype=x.dtype) if bias_value is not None and x.is_floating_point() and bias_value.is_floating_point() and x.dtype != bias_value.dtype else bias_value")
        add(lines, 8, "if x.numel() == 0:")
        add(lines, 12, "out_dim = int(weight_run.shape[-1] if transpose else weight_run.shape[-2])")
        add(lines, 12, "return x.new_empty((*x.shape[:-1], out_dim))")
        add(lines, 8, "if transpose:")
        add(lines, 12, "return torch.matmul(x, weight_run) + (bias_run if bias_run is not None else 0)")
        add(lines, 8, "return F.linear(x, weight_run, bias_run)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _gegelu(x, limit=None):")
        add(lines, 8, "if x.shape[-1] % 2 != 0:")
        add(lines, 12, "raise ValueError('gegelu requires even last dimension')")
        add(lines, 8, "x_gelu = x[..., ::2]")
        add(lines, 8, "x_linear = x[..., 1::2]")
        add(lines, 8, "if limit is not None:")
        add(lines, 12, "limit = float(limit)")
        add(lines, 12, "x_gelu = torch.where(torch.isinf(x_gelu), x_gelu, x_gelu.clamp(max=limit))")
        add(lines, 12, "x_linear = torch.where(torch.isinf(x_linear), x_linear, x_linear.clamp(min=-limit, max=limit))")
        add(lines, 8, "return x_gelu * torch.sigmoid(1.702 * x_gelu) * (x_linear + 1.0)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _xielu(x, alpha_p_raw, alpha_n_raw, beta_raw, eps_raw):")
        add(lines, 8, "target_dtype = x.dtype if x.is_floating_point() else torch.float32")
        add(lines, 8, "def value(raw):")
        add(lines, 12, "return raw.to(device=x.device, dtype=target_dtype) if torch.is_tensor(raw) else torch.tensor(raw, device=x.device, dtype=target_dtype)")
        add(lines, 8, "beta = value(beta_raw)")
        add(lines, 8, "alpha_p = F.softplus(value(alpha_p_raw))")
        add(lines, 8, "alpha_n = beta + F.softplus(value(alpha_n_raw))")
        add(lines, 8, "eps = value(eps_raw)")
        add(lines, 8, "return torch.where(x > 0, alpha_p * x * x + beta * x, (torch.expm1(torch.minimum(x, eps)) - x) * alpha_n + beta * x)")
        add(lines, 4, "")
        add(lines, 4, "def _config(self, path, default=None):")
        add(lines, 8, "return _common_config_value(self.config, path, default)")
        add(lines, 4, "")
        add(lines, 4, "def _has_config(self, path):")
        add(lines, 8, "return _common_has_config_value(self.config, path)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _dtype_from_name(value):")
        add(lines, 8, "if value is None:")
        add(lines, 12, "return None")
        add(lines, 8, "token = str(value).strip().lower()")
        add(lines, 8, "if token in ('', 'none', 'null', 'default'):")
        add(lines, 12, "return None")
        add(lines, 8, "if token in ('float32', 'fp32', 'single'):")
        add(lines, 12, "return torch.float32")
        add(lines, 8, "if token in ('float16', 'fp16', 'half'):")
        add(lines, 12, "return torch.float16")
        add(lines, 8, "if token in ('bfloat16', 'bf16'):")
        add(lines, 12, "return torch.bfloat16")
        add(lines, 8, "if token in ('int64', 'long'):")
        add(lines, 12, "return torch.int64")
        add(lines, 8, "if token in ('int32', 'int'):")
        add(lines, 12, "return torch.int32")
        add(lines, 8, "if token in ('bool', 'boolean'):")
        add(lines, 12, "return torch.bool")
        add(lines, 8, "raise ValueError(f'unsupported dtype name {value!r}')")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _binary_add(cls, left, right):")
        add(lines, 8, "if left is None:")
        add(lines, 12, "return right")
        add(lines, 8, "if right is None:")
        add(lines, 12, "return left")
        add(lines, 8, "return cls._binary_op('+', left, right)")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _binary_sub(cls, left, right):")
        add(lines, 8, "if right is None:")
        add(lines, 12, "return left")
        add(lines, 8, "return cls._binary_op('-', left, right)")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _eq(cls, left, right):")
        add(lines, 8, "return cls._binary_op('==', left, right)")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _where(cls, cond, yes, no):")
        add(lines, 8, "if not torch.is_tensor(cond):")
        add(lines, 12, "return yes if cond else no")
        add(lines, 8, "device = yes.device if torch.is_tensor(yes) else (no.device if torch.is_tensor(no) else cond.device)")
        add(lines, 8, "cond = cls._move_to(cond, device)")
        add(lines, 8, "yes = cls._move_to(yes, device) if torch.is_tensor(yes) else yes")
        add(lines, 8, "no = cls._move_to(no, device) if torch.is_tensor(no) else no")
        add(lines, 8, "return torch.where(cond, yes, no)")
        add(lines, 4, "")
        add(lines, 4, "_cache_past_length = staticmethod(_common_cache_past_length)")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _concat(cls, *args, dim=None):")
        add(lines, 8, "if dim is None:")
        add(lines, 12, "*items, dim = args")
        add(lines, 8, "else:")
        add(lines, 12, "items = args")
        add(lines, 8, "if len(items) == 1 and isinstance(items[0], (list, tuple)):")
        add(lines, 12, "items = tuple(items[0])")
        add(lines, 8, "device = next((item.device for item in items if torch.is_tensor(item)), None)")
        add(lines, 8, "if device is not None:")
        add(lines, 12, "items = tuple(cls._move_to(item, device) for item in items)")
        add(lines, 8, "return torch.cat(list(items), dim=int(dim))")

    def _emit_eval_symbols(self, lines: list[str]) -> None:
        add = self._add
        add(lines, 4, "def _eval_symbols(self):")
        add(lines, 8, "symbols = {}")
        add(lines, 8, "self._symbols = symbols")
        add(lines, 8, f"pending = {sorted(self.global_symbol_names)!r}")
        add(lines, 8, "last_errors = {}")
        add(lines, 8, "while pending:")
        add(lines, 12, "next_pending = []")
        add(lines, 12, "progressed = False")
        add(lines, 12, "for name in pending:")
        add(lines, 16, "try:")
        for name in sorted(self.global_symbol_names):
            method = self.method_names[name]
            add(lines, 20, f"if name == {name!r}:")
            add(lines, 24, f"result = self.{method}()")
            add(lines, 24, "symbols[name] = result[0] if len(result) == 1 else result")
            add(lines, 24, "progressed = True")
            add(lines, 24, "continue")
        add(lines, 20, "raise KeyError(name)")
        add(lines, 16, "except Exception as exc:")
        add(lines, 20, "last_errors[name] = exc")
        add(lines, 20, "next_pending.append(name)")
        add(lines, 12, "if not progressed:")
        add(lines, 16, "name, exc = next(iter(last_errors.items()))")
        add(lines, 16, "raise RuntimeError(f'unable to evaluate graph global symbols; blocked at {name!r}') from exc")
        add(lines, 12, "pending = next_pending")
        add(lines, 8, "return symbols")

    def _emit_module(self, lines: list[str], module: GraphModule) -> None:
        add = self._add
        # Axon optional parameters may precede required parameters. Python
        # cannot express that positionally, so generated internal helpers give
        # every parameter a default while public forward() still validates
        # required model inputs.
        params = ", ".join(f"{value.name}=None" for value in module.inputs)
        if params:
            params = ", " + params
        add(lines, 4, f"def {self.method_names[module.name]}(self{params}):")
        local = {value.name for value in module.inputs}
        for value in module.inputs:
            _emit_bind_nested_shape_symbols(
                lines,
                add=add,
                type_expr=_effective_graph_value_type(value),
                value_expr=value.name,
                local=local,
            )
        dim_params = [
            value.name
            for value in module.inputs
            if isinstance(value.type_expr, TypeDim)
        ]
        free_dim_refs = sorted(
            name
            for name in _module_free_dim_refs(module, global_names=self.global_symbol_names)
            if name not in local
        )
        if len(dim_params) == 1:
            source = _py_ident(dim_params[0])
            for name in free_dim_refs:
                target = _py_ident(name)
                add(lines, 8, f"{target} = {source}")
                local.add(name)
        for node in module.nodes:
            self._emit_node(lines, node, indent=8, local=local, symbols_dict="self._symbols")
            for output in node.outputs:
                local.add(output.name)
        outs = ", ".join(self._operand_expr(item, local=local, symbols_dict="self._symbols") for item in module.outputs)
        if len(module.outputs) == 1:
            add(lines, 8, f"return ({outs},)")
        else:
            add(lines, 8, f"return ({outs})")

    def _emit_forward(self, lines: list[str]) -> None:
        main = self.modules_by_name[self.program.main_module]
        add = self._add
        add(lines, 4, "def forward(self, input_ids=None, **inputs):")
        args: list[str] = []
        first_input = main.inputs[0].name if main.inputs else None
        for value in main.inputs:
            if value.name == "input_ids":
                add(lines, 8, "if input_ids is None:")
                add(lines, 12, "input_ids = inputs.get('input_ids')")
                add(lines, 8, "if input_ids is None:")
                add(lines, 12, "raise ValueError('Missing required input: input_ids')")
                args.append("input_ids")
            elif value.name == first_input:
                add(lines, 8, f"{value.name} = inputs.get({value.name!r}, input_ids)")
                if not (value.optional or isinstance(value.type_expr, TypeOptional)):
                    add(lines, 8, f"if {value.name} is None:")
                    add(lines, 12, f"raise ValueError('Missing required input: {value.name}')")
                args.append(value.name)
            else:
                default = "None" if value.optional or isinstance(value.type_expr, TypeOptional) else "_MISSING"
                if default == "_MISSING":
                    add(lines, 8, f"if {value.name!r} not in inputs:")
                    add(lines, 12, f"raise ValueError('Missing required input: {value.name}')")
                    add(lines, 8, f"{value.name} = inputs[{value.name!r}]")
                else:
                    add(lines, 8, f"{value.name} = inputs.get({value.name!r}, None)")
                args.append(value.name)
        add(lines, 8, f"result = self.{self.method_names[main.name]}({', '.join(args)})")
        names = main.output_names or _fallback_output_names(main)
        if len(names) == 1:
            add(lines, 8, "return result[0]")
        else:
            add(lines, 8, f"return {{{', '.join(f'{name!r}: result[{idx}]' for idx, name in enumerate(names))}}}")

    def _emit_generate(self, lines: list[str]) -> None:
        add = self._add
        main = self.modules_by_name[self.program.main_module]
        input_names = {value.name for value in main.inputs}
        output_names = set(main.output_names or _fallback_output_names(main))
        attention_name = "attn_mask" if "attn_mask" in input_names else (
            "attention_mask" if "attention_mask" in input_names else None
        )
        decoder_attention_name = (
            "decoder_attention_mask" if "decoder_attention_mask" in input_names else None
        )
        cache_name = "past_kv" if "past_kv" in input_names else (
            "past_cache" if "past_cache" in input_names else None
        )
        cache_output_name = "new_kv" if "new_kv" in output_names else (
            "past_kv" if "past_kv" in output_names else (
                "cache" if "cache" in output_names else None
            )
        )
        use_cache_name = "use_cache" if "use_cache" in input_names else None
        has_decoder_inputs = "decoder_input_ids" in input_names
        is_cached_decoder = (
            not has_decoder_inputs
            and cache_name is not None
            and cache_output_name is not None
        )
        is_decoder_only = not has_decoder_inputs
        add(lines, 4, "@torch.no_grad()")
        add(lines, 4, "def generate(self, input_ids, max_new_tokens=20, **kwargs):")
        add(lines, 8, "def _logits(result):")
        add(lines, 12, "return result.get('logits') if isinstance(result, dict) else result")
        add(lines, 8, "def _next_id(logits):")
        add(lines, 12, "return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)")
        add(lines, 8, "def _ones_like_ids(ids):")
        add(lines, 12, "return torch.ones(ids.shape, dtype=torch.long, device=ids.device)")
        add(lines, 8, "def _generation_limit(prompt_ids):")
        add(lines, 12, "requested = kwargs.pop('max_new_tokens', None)")
        add(lines, 12, "if requested is not None:")
        add(lines, 16, "return int(requested)")
        add(lines, 12, "max_len = kwargs.pop('max_len', None)")
        add(lines, 12, "if max_len is None:")
        add(lines, 16, "return int(max_new_tokens)")
        add(lines, 12, "return max(1, int(max_len) - int(prompt_ids.shape[1]))")
        add(lines, 8, "def _eos_state(batch_size, device):")
        add(lines, 12, "eos_token_id = kwargs.pop('eos_token_id', self.config.get('eos_token_id', None))")
        add(lines, 12, "pad_token_id = kwargs.pop('pad_token_id', eos_token_id)")
        add(lines, 12, "if eos_token_id is None:")
        add(lines, 16, "return None, None, None")
        add(lines, 12, "eos = torch.as_tensor(eos_token_id, dtype=torch.long, device=device).reshape(-1)")
        add(lines, 12, "pad = int(eos[0].item() if pad_token_id is None else pad_token_id)")
        add(lines, 12, "finished = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)")
        add(lines, 12, "return eos, pad, finished")
        add(lines, 8, "def _apply_eos(next_id, eos, pad, finished):")
        add(lines, 12, "if eos is None:")
        add(lines, 16, "return next_id, finished")
        add(lines, 12, "eos = eos.to(device=next_id.device)")
        add(lines, 12, "finished = finished.to(device=next_id.device)")
        add(lines, 12, "if torch.is_tensor(pad):")
        add(lines, 16, "pad = int(pad.to(device='cpu').reshape(-1)[0].item())")
        add(lines, 12, "raw_next = next_id")
        add(lines, 12, "next_id = torch.where(finished, torch.full_like(next_id, pad), next_id)")
        add(lines, 12, "hit = (raw_next == eos.view(1, -1)).any(dim=1, keepdim=True)")
        add(lines, 12, "finished = finished | hit")
        add(lines, 12, "return next_id, finished")
        if is_cached_decoder:
            add(lines, 8, "out = input_ids")
            add(lines, 8, "limit = _generation_limit(out)")
            add(lines, 8, "eos, pad, finished = _eos_state(out.shape[0], out.device)")
            add(lines, 8, f"cache = kwargs.pop({cache_name!r}, None)")
            if attention_name is not None:
                other = "attention_mask" if attention_name == "attn_mask" else "attn_mask"
                add(lines, 8, f"attention_mask = kwargs.pop({attention_name!r}, kwargs.pop({other!r}, None))")
                add(lines, 8, "if attention_mask is None:")
                add(lines, 12, "attention_mask = _ones_like_ids(out)")
            if use_cache_name is not None:
                add(lines, 8, f"kwargs.pop({use_cache_name!r}, None)")
            add(lines, 8, "for _ in range(limit):")
            add(lines, 12, "step_input = out[:, -1:] if cache is not None else out")
            add(lines, 12, "forward_kwargs = dict(kwargs)")
            add(lines, 12, f"forward_kwargs[{cache_name!r}] = cache")
            if use_cache_name is not None:
                add(lines, 12, f"forward_kwargs[{use_cache_name!r}] = True")
            if attention_name is not None:
                add(lines, 12, f"forward_kwargs[{attention_name!r}] = attention_mask")
            add(lines, 12, "result = self.forward(step_input, **forward_kwargs)")
            add(lines, 12, "logits = _logits(result)")
            add(lines, 12, "if isinstance(result, dict):")
            add(lines, 16, f"cache = result.get({cache_output_name!r}, cache)")
            add(lines, 12, "next_id = _next_id(logits)")
            add(lines, 12, "next_id, finished = _apply_eos(next_id, eos, pad, finished)")
            add(lines, 12, "out = self._move_to(out, next_id.device)")
            add(lines, 12, "out = torch.cat([out, next_id], dim=1)")
            if attention_name is not None:
                add(lines, 12, "attention_mask = self._move_to(attention_mask, next_id.device)")
                add(lines, 12, "attention_mask = torch.cat([attention_mask, _ones_like_ids(next_id)], dim=1)")
            add(lines, 12, "if finished is not None and bool(finished.all().item()):")
            add(lines, 16, "break")
            add(lines, 8, "return out")
            return
        if is_decoder_only:
            add(lines, 8, "out = input_ids")
            add(lines, 8, "limit = _generation_limit(out)")
            add(lines, 8, "eos, pad, finished = _eos_state(out.shape[0], out.device)")
            if attention_name is not None:
                other = "attention_mask" if attention_name == "attn_mask" else "attn_mask"
                add(lines, 8, f"attention_mask = kwargs.pop({attention_name!r}, kwargs.pop({other!r}, None))")
                add(lines, 8, "if attention_mask is None:")
                add(lines, 12, "attention_mask = _ones_like_ids(out)")
            add(lines, 8, "for _ in range(limit):")
            add(lines, 12, "forward_kwargs = dict(kwargs)")
            if attention_name is not None:
                add(lines, 12, f"forward_kwargs[{attention_name!r}] = attention_mask")
            add(lines, 12, "result = self.forward(out, **forward_kwargs)")
            add(lines, 12, "next_id = _next_id(_logits(result))")
            add(lines, 12, "next_id, finished = _apply_eos(next_id, eos, pad, finished)")
            add(lines, 12, "out = self._move_to(out, next_id.device)")
            add(lines, 12, "out = torch.cat([out, next_id], dim=1)")
            if attention_name is not None:
                add(lines, 12, "attention_mask = self._move_to(attention_mask, next_id.device)")
                add(lines, 12, "attention_mask = torch.cat([attention_mask, _ones_like_ids(next_id)], dim=1)")
            add(lines, 12, "if finished is not None and bool(finished.all().item()):")
            add(lines, 16, "break")
            add(lines, 8, "return out")
            return
        add(lines, 8, "decoder_input_ids = kwargs.pop('decoder_input_ids', None)")
        add(lines, 8, "if decoder_input_ids is None:")
        add(lines, 12, "start_id = kwargs.pop('decoder_start_token_id', self.config.get('decoder_start_token_id', self.config.get('pad_token_id', 0)))")
        add(lines, 12, "decoder_input_ids = torch.full((input_ids.shape[0], 1), int(start_id), dtype=input_ids.dtype, device=input_ids.device)")
        add(lines, 8, "limit = _generation_limit(input_ids)")
        add(lines, 8, "eos, pad, finished = _eos_state(decoder_input_ids.shape[0], decoder_input_ids.device)")
        if attention_name is not None:
            other = "attention_mask" if attention_name == "attn_mask" else "attn_mask"
            add(lines, 8, f"attention_mask = kwargs.pop({attention_name!r}, kwargs.pop({other!r}, None))")
            add(lines, 8, "if attention_mask is None:")
            add(lines, 12, "attention_mask = _ones_like_ids(input_ids)")
        if decoder_attention_name is not None:
            add(lines, 8, f"decoder_attention_mask = kwargs.pop({decoder_attention_name!r}, None)")
            add(lines, 8, "if decoder_attention_mask is None:")
            add(lines, 12, "decoder_attention_mask = _ones_like_ids(decoder_input_ids)")
        add(lines, 8, "for _ in range(limit):")
        add(lines, 12, "forward_kwargs = dict(kwargs)")
        add(lines, 12, "forward_kwargs['decoder_input_ids'] = decoder_input_ids")
        if attention_name is not None:
            add(lines, 12, f"forward_kwargs[{attention_name!r}] = attention_mask")
        if decoder_attention_name is not None:
            add(lines, 12, f"forward_kwargs[{decoder_attention_name!r}] = decoder_attention_mask")
        add(lines, 12, "result = self.forward(input_ids, **forward_kwargs)")
        add(lines, 12, "next_id = _next_id(_logits(result))")
        add(lines, 12, "next_id, finished = _apply_eos(next_id, eos, pad, finished)")
        add(lines, 12, "decoder_input_ids = self._move_to(decoder_input_ids, next_id.device)")
        add(lines, 12, "decoder_input_ids = torch.cat([decoder_input_ids, next_id], dim=1)")
        if decoder_attention_name is not None:
            add(lines, 12, "decoder_attention_mask = self._move_to(decoder_attention_mask, next_id.device)")
            add(lines, 12, "decoder_attention_mask = torch.cat([decoder_attention_mask, _ones_like_ids(next_id)], dim=1)")
        add(lines, 12, "if finished is not None and bool(finished.all().item()):")
        add(lines, 16, "break")
        add(lines, 8, "return decoder_input_ids")

    def _emit_node(self, lines: list[str], node: Any, *, indent: int, local: set[str], symbols_dict: str) -> None:
        add = self._add
        op = node.op.name
        targets = tuple(_py_ident(value.name) for value in node.outputs)
        expr = self._node_expr(node, local=local, symbols_dict=symbols_dict)
        if len(targets) == 1:
            add(lines, indent, f"{targets[0]} = {expr}")
        else:
            add(lines, indent, f"{', '.join(targets)} = {expr}")

    def _node_expr(self, node: Any, *, local: set[str], symbols_dict: str) -> str:
        op = node.op.name
        if op == "core.alias":
            return self._operand_expr(node.inputs[0], local=local, symbols_dict=symbols_dict)
        if op == "core.tuple":
            return "(" + ", ".join(self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs) + ")"
        if op == "core.list":
            return "[" + ", ".join(self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs) + "]"
        if op == "core.ascribe":
            return self._operand_expr(node.inputs[0], local=local, symbols_dict=symbols_dict)
        if op == "core.select":
            cond = self._operand_expr(node.inputs[0], local=local, symbols_dict=symbols_dict)
            yes = self._operand_expr(node.inputs[1], local=local, symbols_dict=symbols_dict)
            no = self._operand_expr(node.inputs[2], local=local, symbols_dict=symbols_dict)
            return f"({yes} if bool({cond}) else {no})"
        if op.startswith("core.binary."):
            binop = op.removeprefix("core.binary.")
            left = self._operand_expr(node.inputs[0], local=local, symbols_dict=symbols_dict)
            right = self._operand_expr(node.inputs[1], local=local, symbols_dict=symbols_dict)
            if binop == "+":
                return f"self._binary_add({left}, {right})"
            if binop == "-":
                return f"self._binary_sub({left}, {right})"
            if binop in {"*", "/", "%", "<", "<=", ">", ">=", "==", "!="}:
                return f"self._binary_op({binop!r}, {left}, {right})"
            pyop = {"and": "&", "or": "|"} .get(binop, binop)
            return f"({left} {pyop} {right})"
        if op in self.method_names:
            if op in {"Cache.past_length", "Cache.past_length_kv"}:
                args = [
                    self._operand_expr(x, local=local, symbols_dict=symbols_dict)
                    for x in node.inputs
                ]
                if len(args) != 1:
                    raise ValueError(f"{op} expects one argument")
                return f"self._cache_past_length({args[0]})"
            args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
            args.extend(f"{key}={self._operand_expr(value, local=local, symbols_dict=symbols_dict)}" for key, value in node.attrs.items())
            call = f"self.{self.method_names[op]}({', '.join(args)})"
            module = self.modules_by_name[op]
            return f"{call}[0]" if len(module.outputs) == 1 else call
        primitive = _normalize_primitive_op(op)
        return self._primitive_expr(primitive, node, local=local, symbols_dict=symbols_dict)

    def _primitive_expr(self, primitive: str, node: Any, *, local: set[str], symbols_dict: str) -> str:
        args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
        attrs = {k: self._operand_expr(v, local=local, symbols_dict=symbols_dict) for k, v in node.attrs.items()}
        if primitive == "params_param":
            return f"self._param({args[0]})"
        if primitive == "params_has_root":
            return f"any(k == {args[0]} or k.startswith(str({args[0]}) + '.') for k in self.state_dict_tensors)"
        if primitive.startswith("config_"):
            default = args[1] if len(args) > 1 else attrs.get("default", "None")
            if primitive == "config_has":
                return f"self._has_config({args[0]})"
            if primitive == "config_has_value":
                return f"self._has_config({args[0]})"
            value = f"self._config({args[0]}, {default})"
            if primitive in {"config_int", "config_dim"}:
                return f"int({value})"
            if primitive == "config_float":
                return f"float({value})"
            if primitive == "config_bool":
                return f"bool({value})"
            if primitive == "config_str":
                return f"str({value})"
            if primitive == "config_list":
                return f"list({value})"
            return value
        if primitive == "embedding":
            return f"(lambda _w: F.embedding(self._move_to({args[1]}, _w.device), _w))(self._param(self._compose_path({args[0]}, 'weight')))"
        if primitive == "linear":
            bias = args[3] if len(args) > 3 else "False"
            transpose = args[4] if len(args) > 4 else "False"
            expert = args[5] if len(args) > 5 else "None"
            weight_leaf = args[6] if len(args) > 6 else "'weight'"
            bias_leaf = args[7] if len(args) > 7 else "'bias'"
            return f"self._linear({args[0]}, {args[1]}, bias=bool({bias}), transpose=bool({transpose}), expert=({expert} if {expert} is not None else None), weight_leaf={weight_leaf}, bias_leaf={bias_leaf})"
        if primitive == "layernorm":
            eps = args[2] if len(args) > 2 else "1e-5"
            weight_leaf = args[4] if len(args) > 4 else "'weight'"
            bias = args[5] if len(args) > 5 else "True"
            bias_leaf = args[6] if len(args) > 6 else "'bias'"
            return f"(lambda _w, _b: F.layer_norm(self._move_to({args[1]}, _w.device), ({args[1]}.shape[-1],), weight=_w, bias=(self._move_to(_b, _w.device) if _b is not None else None), eps=float({eps})))(self._param(self._compose_path({args[0]}, {weight_leaf})), (self._optional_param(self._compose_path({args[0]}, {bias_leaf})) if {bias} else None))"
        if primitive == "rmsnorm":
            x = args[0]
            eps = args[1] if len(args) > 1 else "1e-6"
            cast_float = args[3] if len(args) > 3 else "False"
            x_float = f"{x}.float()"
            y_float = f"({x_float} * torch.rsqrt(torch.mean({x_float} * {x_float}, dim=-1, keepdim=True) + float({eps})))"
            y = f"({x} * torch.rsqrt(torch.mean({x} * {x}, dim=-1, keepdim=True) + float({eps})))"
            return f"({y_float}.to(dtype={x}.dtype) if {cast_float} else {y})"
        if primitive == "tensor_like":
            dtype = args[2] if len(args) > 2 else "None"
            target_dtype = f"({args[1]}.dtype if self._dtype_from_name({dtype}) is None else self._dtype_from_name({dtype}))"
            return f"({args[0]}.to(device={args[1]}.device, dtype={target_dtype}) if torch.is_tensor({args[0]}) else torch.tensor({args[0]}, device={args[1]}.device, dtype={target_dtype}))"
        if primitive == "where_indices":
            return f"torch.where({args[0]})"
        if primitive == "topk":
            return f"torch.topk({args[0]}, int({args[1]}), dim=int({args[2]}), largest=bool({args[3]}), sorted=bool({args[4]}))"
        if primitive == "concat":
            if "dim" in attrs:
                return f"self._concat({', '.join(args)}, dim={attrs['dim']})"
            if not args:
                raise ValueError("concat requires at least one argument")
            return f"self._concat({', '.join(args[:-1])}, dim={args[-1]})"
        if primitive == "clamp":
            min_value = args[1] if len(args) > 1 else attrs.get("min", "None")
            max_value = args[2] if len(args) > 2 else attrs.get("max", "None")
            return f"torch.clamp({args[0]}, min=({min_value} if {min_value} is not None else None), max=({max_value} if {max_value} is not None else None))"
        simple = {
            "reshape": lambda: f"torch.reshape({args[0]}, tuple(int(x) for x in {args[1]}))",
            "arange": lambda: f"torch.arange(int({args[1]}), int(({args[0]}.shape[-2] if {args[2]} is None and {args[0]}.ndim >= 2 else ({args[0]}.shape[-1] if {args[2]} is None else {args[2]}))), device={args[0]}.device, dtype=torch.long)",
            "slice": lambda: f"torch.narrow({args[0]}, int({args[1]}), int({args[2]}), int({args[3]}) - int({args[2]}))",
            "chunk": lambda: f"torch.chunk({args[0]}, int({args[2] if len(args) > 2 else attrs.get('parts', '1')}), dim=int({args[1] if len(args) > 1 else attrs.get('dim', '-1')}))",
            "split": lambda: f"torch.split({args[0]}, [int(x) for x in {args[2] if len(args) > 2 else attrs.get('sizes', '[]')}], dim=int({args[1] if len(args) > 1 else attrs.get('dim', '-1')}))",
            "sum": lambda: f"torch.sum({args[0]}, dim=int({args[1] if len(args) > 1 else '-1'}), keepdim=bool({args[2] if len(args) > 2 else 'False'}))",
            "expand": lambda: f"{args[0]}.expand(tuple(int(x) for x in {args[1]}))",
            "permute": lambda: f"torch.permute({args[0]}, tuple(int(x) for x in {args[1]}))",
            "transpose": lambda: f"torch.transpose({args[0]}, int({args[1]}), int({args[2]}))",
            "unsqueeze": lambda: f"torch.unsqueeze({args[0]}, int({args[1]}))",
            "repeat": lambda: f"({args[0]} if int({args[1]}) == 1 else torch.repeat_interleave({args[0]}, repeats=int({args[1]}), dim=(int({args[2]}) if int({args[2]}) >= 0 else int({args[2]}) + {args[0]}.dim())))",
            "matmul": lambda: f"(lambda _a, _b: torch.matmul(_a, _b))(*self._align_pair({args[0]}, {args[1]}, prefer='right'))",
            "softmax": lambda: f"F.softmax({args[0]}, dim=int({args[1] if len(args) > 1 else '-1'}))",
            "where": lambda: f"self._where({args[0]}, {args[1]}, {args[2]})",
            "require": lambda: f"self._require_value({args[0]})",
            "gather": lambda: f"torch.gather({args[0]}, dim=int({args[2] if len(args) > 2 else '-1'}), index=self._move_to({args[1]}, {args[0]}.device))",
            "scatter": lambda: f"(torch.scatter({args[0]}, dim=int({args[3] if len(args) > 3 else '-1'}), index=self._move_to({args[1]}, {args[0]}.device), src=self._move_to({args[2]}, {args[0]}.device)) if torch.is_tensor({args[2]}) else torch.scatter({args[0]}, dim=int({args[3] if len(args) > 3 else '-1'}), index=self._move_to({args[1]}, {args[0]}.device), value={args[2]}))",
            "index_add": lambda: f"torch.index_add({args[0]}, dim=int({args[3] if len(args) > 3 else '0'}), index=self._move_to({args[1]}, {args[0]}.device), source=self._move_to({args[2]}, {args[0]}.device))",
            "le": lambda: f"self._binary_op('<=', {args[0]}, {args[1]})",
            "eq": lambda: f"self._eq({args[0]}, {args[1]})",
            "and": lambda: f"(lambda _a, _b: torch.logical_and(_a, _b))(*self._align_pair({args[0]}, {args[1]}, prefer='right'))",
            "add": lambda: f"self._binary_add({args[0]}, {args[1]})",
            "mul": lambda: f"self._binary_op('*', {args[0]}, {args[1]})",
            "div": lambda: f"self._binary_op('/', {args[0]}, {args[1]})",
            "pow": lambda: f"(torch.pow({args[0]}, {args[1]}) if torch.is_tensor({args[0]}) else ({args[0]} ** {args[1]}))",
            "floor": lambda: f"torch.floor({args[0]}) if torch.is_tensor({args[0]}) else int({args[0]} // 1)",
            "sqrt": lambda: f"torch.sqrt({args[0]}) if torch.is_tensor({args[0]}) else ({args[0]} ** 0.5)",
            "sin": lambda: f"torch.sin({args[0]}) if torch.is_tensor({args[0]}) else __import__('math').sin(float({args[0]}))",
            "cos": lambda: f"torch.cos({args[0]}) if torch.is_tensor({args[0]}) else __import__('math').cos(float({args[0]}))",
            "exp": lambda: f"torch.exp({args[0]}) if torch.is_tensor({args[0]}) else __import__('math').exp(float({args[0]}))",
            "log": lambda: f"torch.log({args[0]}) if torch.is_tensor({args[0]}) else __import__('math').log(float({args[0]}))",
            "cast": lambda: f"{args[0]}.to(dtype=getattr(torch, str({args[1]})))",
            "cast_like": lambda: f"{args[0]}.to(device={args[1]}.device, dtype={args[1]}.dtype)",
            "dtype_value": lambda: f"{{'min': torch.finfo({args[0]}.dtype).min, 'max': torch.finfo({args[0]}.dtype).max, 'eps': torch.finfo({args[0]}.dtype).eps, 'tiny': torch.finfo({args[0]}.dtype).tiny, 'inf': float('inf'), '-inf': float('-inf')}}[str({args[1]})]",
            "cumsum": lambda: f"torch.cumsum({args[0]}, dim=int({args[1] if len(args) > 1 else '-1'}))",
            "empty_like": lambda: f"torch.empty_like({args[0]})",
            "fill": lambda: f"torch.full_like({args[0]}, {args[1]}, dtype=({args[0]}.dtype if self._dtype_from_name({args[2] if len(args) > 2 else 'None'}) is None else self._dtype_from_name({args[2] if len(args) > 2 else 'None'})))",
            "empty": lambda: f"torch.empty(tuple(int(x) for x in {args[1]}), device={args[0]}.device, dtype=((self._dtype_from_name({args[2] if len(args) > 2 else 'None'}) if {args[2] if len(args) > 2 else 'None'} is not None else None) or {args[0]}.dtype))",
            "zeros": lambda: f"torch.zeros(tuple(int(x) for x in {args[1]}), device={args[0]}.device, dtype=((self._dtype_from_name({args[2] if len(args) > 2 else 'None'}) if {args[2] if len(args) > 2 else 'None'} is not None else None) or {args[0]}.dtype))",
            "full": lambda: f"torch.full(tuple(int(x) for x in {args[1]}), {args[2]}, device={args[0]}.device, dtype=((self._dtype_from_name({args[3] if len(args) > 3 else 'None'}) if {args[3] if len(args) > 3 else 'None'} is not None else None) or {args[0]}.dtype))",
            "zeros_like": lambda: f"torch.zeros_like({args[0]})",
            "activations_tanh": lambda: f"torch.tanh({args[0]})",
            "activations_silu": lambda: f"F.silu({args[0]})",
            "activations_sigmoid": lambda: f"torch.sigmoid({args[0]})",
            "l2norm": lambda: f"(({args[0]}.float() * torch.pow(torch.mean({args[0]}.float() * {args[0]}.float(), dim=-1, keepdim=True) + float({args[1] if len(args) > 1 else '1e-6'}), -0.5)).to(dtype={args[0]}.dtype))",
            "activations_relu": lambda: f"F.relu({args[0]})",
            "activations_relu2": lambda: f"(F.relu({args[0]}) * F.relu({args[0]}))",
            "activations_gelu": lambda: f"F.gelu({args[0]})",
            "activations_gelu_new": lambda: f"(0.5 * {args[0]} * (1.0 + torch.tanh(0.7978845608028654 * ({args[0]} + 0.044715 * {args[0]} * {args[0]} * {args[0]}))))",
            "activations_gelu_pytorch_tanh": lambda: f"(0.5 * {args[0]} * (1.0 + torch.tanh(0.7978845608028654 * ({args[0]} + 0.044715 * {args[0]} * {args[0]} * {args[0]}))))",
            "activations_gegelu": lambda: f"self._gegelu({args[0]}, {args[1] if len(args) > 1 else 'None'})",
            "activations_xielu": lambda: f"self._xielu({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]})",
            "list_init": lambda: "[]",
            "list_append": lambda: f"([*({args[0]} or []), {args[1]}])",
            "list_index": lambda: f"{args[0]}[int({args[1]})]",
            "shape": lambda: f"list({args[0]}.shape)",
        }
        if primitive in simple:
            return simple[primitive]()
        raise NotImplementedError(f"direct codegen2-torch unsupported graph op {primitive!r}")

    def _operand_expr(self, operand: GraphOperand, *, local: set[str], symbols_dict: str) -> str:
        if isinstance(operand, GraphValueRef):
            name = _py_ident(operand.name)
            return name if operand.name in local else f"{symbols_dict}[{operand.name!r}]"
        if isinstance(operand, GraphLiteral):
            return repr(operand.value)
        if isinstance(operand, GraphPath):
            return self._path_expr(operand, local=local, symbols_dict=symbols_dict)
        if isinstance(operand, GraphExpr):
            if operand.op.name in {"Cache.past_length", "Cache.past_length_kv"}:
                args = [
                    self._operand_expr(item, local=local, symbols_dict=symbols_dict)
                    for item in operand.inputs
                ]
                if len(args) != 1:
                    raise ValueError(f"{operand.op.name} expects one argument")
                return f"self._cache_past_length({args[0]})"
            if operand.op.name in self.method_names:
                args = [
                    self._operand_expr(item, local=local, symbols_dict=symbols_dict)
                    for item in operand.inputs
                ]
                args.extend(
                    f"{key}={self._operand_expr(value, local=local, symbols_dict=symbols_dict)}"
                    for key, value in operand.attrs.items()
                )
                call = f"self.{self.method_names[operand.op.name]}({', '.join(args)})"
                module = self.modules_by_name[operand.op.name]
                return f"{call}[0]" if len(module.outputs) == 1 else call
            pseudo = type("_Node", (), {"op": operand.op, "inputs": operand.inputs, "attrs": operand.attrs, "outputs": ()})()
            return self._node_expr(pseudo, local=local, symbols_dict=symbols_dict)
        if isinstance(operand, tuple):
            return "[" + ", ".join(self._operand_expr(item, local=local, symbols_dict=symbols_dict) for item in operand) + "]"
        raise TypeError(f"unsupported graph operand: {operand!r}")

    def _path_expr(self, path: GraphPath, *, local: set[str], symbols_dict: str) -> str:
        prefix = "@@" if path.absolute else "@"
        part_exprs: list[str] = []
        has_dynamic = False
        for part in path.parts:
            names = re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", part)
            if not names:
                part_exprs.append(repr(part))
                continue
            has_dynamic = True
            pieces = []
            cursor = 0
            for match in re.finditer(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", part):
                pieces.append(part[cursor:match.start()].replace("{", "{{").replace("}", "}}"))
                name = match.group(1)
                pieces.append("{" + (_py_ident(name) if name in local else f"{symbols_dict}[{name!r}]") + "}")
                cursor = match.end()
            pieces.append(part[cursor:].replace("{", "{{").replace("}", "}}"))
            part_exprs.append("f" + repr("".join(pieces)))
        if not has_dynamic:
            return repr(prefix + ".".join(path.parts))
        return f"self._render_path({prefix!r}, [{', '.join(part_exprs)}])"


def emit_model_code_from_graph_ir(
    program: GraphProgram,
    *,
    class_name: str = "GeneratedAxonModel",
    model_config: dict[str, Any] | None = None,
) -> str:
    """Emit direct Python/PyTorch model code from graph IR."""
    validate_graph_program(program)
    emitter = _DirectTorchEmitter(program=program, class_name=class_name)
    body = emitter.emit()
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "import torch",
            "from torch import nn",
            "from torch.nn import functional as F",
            "from brainsurgery.synapse.axon.codegen2_common import (",
            "    cache_past_length as _common_cache_past_length,",
            "    compose_path as _common_compose_path,",
            "    config_value as _common_config_value,",
            "    has_config_value as _common_has_config_value,",
            "    optional_state_value as _common_optional_state_value,",
            "    render_path as _common_render_path,",
            "    required_state_value as _common_required_state_value,",
            "    require_value as _common_require_value,",
            ")",
            "",
            f"_MODEL_CONFIG = {model_config!r}",
            "",
            body,
        ]
    )


def _graph_expr_payload(expr: GraphExpr) -> Any:
    op = expr.op.name
    if op == "core.list":
        return [_operand_payload(item) for item in expr.inputs]
    if op == "core.tuple":
        return {
            "_expr": "tuple",
            "items": [_operand_payload(item) for item in expr.inputs],
        }
    if op == "core.ascribe":
        return _operand_payload(expr.inputs[0])
    if op == "core.select":
        return {
            "_expr": "if",
            "cond": _operand_payload(expr.inputs[0]),
            "then": _operand_payload(expr.inputs[1]),
            "else": _operand_payload(expr.inputs[2]),
        }
    if op.startswith("core.binary."):
        return {
            "_expr": "binary",
            "op": op.removeprefix("core.binary."),
            "left": _operand_payload(expr.inputs[0]),
            "right": _operand_payload(expr.inputs[1]),
        }
    return {
        "_expr": "call",
        "callee": op,
        "args": [_operand_payload(item) for item in expr.inputs],
        "kwargs": {key: _operand_payload(value) for key, value in expr.attrs.items()},
    }


def make_graph_model_class(
    program: GraphProgram,
    *,
    model_config: dict[str, Any] | None = None,
) -> type[Codegen2GraphModel]:
    validate_graph_program(program)

    class GeneratedCodegen2GraphModel(Codegen2GraphModel):
        GRAPH = program

        @classmethod
        def from_state_dict(
            cls,
            state_dict: dict[str, torch.Tensor],
            *,
            graph: GraphProgram | None = None,
            model_config: dict[str, Any] | None = None,
        ) -> "GeneratedCodegen2GraphModel":
            return cls(
                graph=program if graph is None else graph,
                state_dict=state_dict,
                model_config=model_config if model_config is not None else captured_model_config,
            )

    GeneratedCodegen2GraphModel.__name__ = "GeneratedCodegen2GraphModel"
    captured_model_config = model_config
    GeneratedCodegen2GraphModel._codegen2_model_config = captured_model_config
    return GeneratedCodegen2GraphModel


Runtime2GraphModel = Codegen2GraphModel


def make_runtime2_model_class(
    program: GraphProgram,
    *,
    model_config: dict[str, Any] | None = None,
) -> type[Codegen2GraphModel]:
    return make_graph_model_class(program, model_config=model_config)


__all__ = [
    "Codegen2GraphModel",
    "Runtime2GraphModel",
    "emit_model_code_from_graph_ir",
    "make_graph_model_class",
    "make_runtime2_model_class",
]
