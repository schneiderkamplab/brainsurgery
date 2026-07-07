from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ...mxfp4 import materialize_mxfp4_aliases
from ...ops import get_op_semantics
from ..ast import (
    AxonExprPath,
    DimExprBinary,
    TypeAny,
    TypeBool,
    TypeDim,
    TypeFloat,
    TypeInt,
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
    GraphNode,
    GraphOperand,
    GraphOp,
    GraphPackedParameter,
    GraphPath,
    GraphProgram,
    GraphValue,
    GraphValueRef,
    graph_operand_type,
    graph_path_template_names,
    validate_graph_program,
)
from ..graph_ir.effects import (
    GraphEffect,
    UsageClass,
    graph_node_effect,
    infer_graph_module_effects,
    infer_graph_module_usages,
)
from ..graph_ir.ownership import infer_graph_ownership
from ..graph_ir.substitute import substitute_graph_node_dims, substitute_graph_operand_dims


def _graph_path_payload(path: GraphPath) -> dict[str, Any]:
    return path_expr_to_runtime_value(
        AxonExprPath(absolute=path.absolute, parts=path.parts)
    )


def _graph_path_token(path: GraphPath, env: Mapping[str, Any]) -> str:
    payload = _graph_path_payload(path)
    key = resolve_path_expr_to_key(payload, env, op_name="graph path")
    return ("@@" if path.absolute else "@") + key


def _graph_path_pattern_key(path: GraphPath) -> str:
    return ".".join(part for part in path.parts if part)


def _graph_path_template_replacement(operand: GraphOperand) -> str | None:
    if isinstance(operand, GraphPath):
        return _graph_path_pattern_key(operand)
    if isinstance(operand, GraphLiteral) and isinstance(operand.value, str):
        return str(operand.value).lstrip("@").strip(".")
    return None


def _packed_parameter_spec_payload(packed: GraphPackedParameter) -> dict[str, Any]:
    return {
        "output": _graph_path_pattern_key(packed.output),
        "inputs": tuple(_graph_path_pattern_key(item) for item in packed.inputs),
        "dim": int(packed.dim),
        "mode": "cat",
        "remove_inputs": bool(packed.remove_inputs),
    }


def _path_pattern_regex(pattern: str) -> re.Pattern[str]:
    pieces: list[str] = []
    cursor = 0
    used: set[str] = set()
    for match in re.finditer(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", pattern):
        pieces.append(re.escape(pattern[cursor : match.start()]))
        name = match.group(1)
        if name in used:
            pieces.append(f"(?P={name})")
        else:
            pieces.append(f"(?P<{name}>[^.]+)")
            used.add(name)
        cursor = match.end()
    pieces.append(re.escape(pattern[cursor:]))
    return re.compile("^" + "".join(pieces) + "$")


def _format_path_pattern(pattern: str, values: Mapping[str, str]) -> str:
    out = pattern
    for key, value in values.items():
        out = out.replace("{" + key + "}", str(value))
    return out


def _materialize_joined_parameter(
    state: dict[str, torch.Tensor],
    output_key: str,
    input_keys: Sequence[str],
    *,
    dim: int,
    mode: str,
    remove_inputs: bool = True,
) -> torch.Tensor | None:
    existing = state.get(output_key)
    if torch.is_tensor(existing):
        return existing
    tensors = [state.get(key) for key in input_keys]
    if not tensors or not all(torch.is_tensor(item) for item in tensors):
        return None
    if mode == "cat":
        joined = torch.cat(tensors, dim=int(dim))  # type: ignore[arg-type]
    elif mode == "stack":
        joined = torch.stack(tensors, dim=int(dim))  # type: ignore[arg-type]
    else:
        raise ValueError(f"unknown parameter join mode {mode!r}")
    state[output_key] = joined
    if remove_inputs:
        for key in input_keys:
            state.pop(key, None)
    return joined


def _materialize_packed_parameters(
    state: dict[str, torch.Tensor],
    specs: tuple[dict[str, Any], ...],
    *,
    target_key: str | None = None,
) -> None:
    for spec in specs:
        output_pattern = str(spec["output"])
        regex = _path_pattern_regex(output_pattern)
        candidates: list[tuple[str, dict[str, str]]] = []
        if target_key is not None:
            match = regex.match(str(target_key))
            if match is not None:
                candidates.append((str(target_key), match.groupdict()))
        else:
            literal = "{" not in output_pattern
            if literal:
                candidates.append((output_pattern, {}))
            else:
                keys = list(state)
                for key in keys:
                    match = regex.match(str(key))
                    if match is not None:
                        candidates.append((str(key), match.groupdict()))
                input_patterns = tuple(str(item) for item in spec["inputs"])
                if input_patterns:
                    input_regex = _path_pattern_regex(input_patterns[0])
                    for key in keys:
                        match = input_regex.match(str(key))
                        if match is None:
                            continue
                        values = match.groupdict()
                        output_key = _format_path_pattern(output_pattern, values)
                        candidates.append((output_key, values))
        seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        for output_key, values in candidates:
            candidate_key = (output_key, tuple(sorted(values.items())))
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            if torch.is_tensor(state.get(output_key)):
                continue
            input_keys = [_format_path_pattern(str(item), values) for item in spec["inputs"]]
            _materialize_joined_parameter(
                state,
                output_key,
                input_keys,
                dim=int(spec["dim"]),
                mode=str(spec.get("mode", "cat")),
                remove_inputs=bool(spec.get("remove_inputs", True)),
            )


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


def _operand_uses_expert_linear(operand: GraphOperand) -> bool:
    if isinstance(operand, GraphExpr):
        if _normalize_primitive_op(operand.op.name) == "expert_linear" or operand.op.name in {
            "__torch_expert_swiglu_ffn",
            "__torch_expert_packed_swiglu_ffn",
            "__torch_selected_expert_swiglu_ffn",
            "__torch_selected_expert_packed_swiglu_ffn",
            "__torch_selected_expert_clamped_packed_swiglu_ffn",
            "__torch_selected_expert_packed_gegelu_ffn",
            "__torch_selected_expert_relu2_ffn",
        }:
            return True
        return any(_operand_uses_expert_linear(item) for item in operand.inputs) or any(
            _operand_uses_expert_linear(item) for item in operand.attrs.values()
        )
    return False


def _graph_uses_expert_linear(program: GraphProgram) -> bool:
    for module in program.modules:
        for node in module.nodes:
            if _normalize_primitive_op(node.op.name) == "expert_linear" or node.op.name in {
                "__torch_expert_swiglu_ffn",
                "__torch_expert_packed_swiglu_ffn",
                "__torch_selected_expert_swiglu_ffn",
                "__torch_selected_expert_packed_swiglu_ffn",
                "__torch_selected_expert_clamped_packed_swiglu_ffn",
                "__torch_selected_expert_packed_gegelu_ffn",
                "__torch_selected_expert_relu2_ffn",
            }:
                return True
            if any(_operand_uses_expert_linear(item) for item in node.inputs):
                return True
            if any(_operand_uses_expert_linear(item) for item in node.attrs.values()):
                return True
    return False


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


def _task_value(program: GraphProgram) -> str | None:
    raw = program.pragmas.get("TASK", program.pragmas.get("task"))
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], str):
        return raw[0]
    return None


def graph_main_output_names(program: GraphProgram, module: GraphModule) -> tuple[str, ...]:
    names = tuple(module.output_names or ()) or _fallback_output_names(module)
    task = _task_value(program)
    if task in {"causal_lm", "masked_lm", "seq2seq_lm"} and names and "logits" not in names:
        return ("logits", *names[1:])
    return names


def _main_output_names(program: GraphProgram, module: GraphModule) -> tuple[str, ...]:
    return graph_main_output_names(program, module)


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


def _tensor_dims_from_type(type_expr: Any) -> tuple[Any, ...] | None:
    if isinstance(type_expr, TypeOptional):
        return _tensor_dims_from_type(type_expr.inner)
    if isinstance(type_expr, TypeTensor):
        return tuple(type_expr.dims)
    return None


def _is_static_mask_type(type_expr: Any) -> bool:
    if isinstance(type_expr, TypeOptional):
        return _is_static_mask_type(type_expr.inner)
    if isinstance(type_expr, TypeNamed) and type_expr.name.endswith("StaticMask"):
        return True
    if (
        isinstance(type_expr, TypeTuple)
        and len(type_expr.items) == 2
        and isinstance(type_expr.items[0], TypeTensor)
        and len(type_expr.items[0].dims) == 2
        and isinstance(type_expr.items[1], TypeDim)
    ):
        return True
    return False


def _static_mask_capacity_dim(type_expr: Any) -> Any | None:
    if isinstance(type_expr, TypeOptional):
        return _static_mask_capacity_dim(type_expr.inner)
    if isinstance(type_expr, TypeNamed) and type_expr.name.endswith("StaticMask") and len(type_expr.args) >= 2:
        return type_expr.args[1]
    if (
        isinstance(type_expr, TypeTuple)
        and len(type_expr.items) == 2
        and isinstance(type_expr.items[0], TypeTensor)
        and len(type_expr.items[0].dims) == 2
    ):
        return type_expr.items[0].dims[1]
    return None


def _static_dim_runtime_expr(dim: Any, *, global_names: set[str]) -> str | None:
    if type(dim) is int:
        return repr(dim)
    if isinstance(dim, str):
        if dim in global_names:
            return f"self._symbols[{dim!r}]"
        return None
    if isinstance(dim, DimExprBinary):
        left = _static_dim_runtime_expr(dim.left, global_names=global_names)
        right = _static_dim_runtime_expr(dim.right, global_names=global_names)
        if left is None or right is None:
            return None
        op = "//" if dim.op == "/" else dim.op
        return f"(({left}) {op} ({right}))"
    return None


def _static_mask_capacity_expr(type_expr: Any, *, global_names: set[str]) -> str | None:
    dim = _static_mask_capacity_dim(type_expr)
    if dim is None:
        return None
    return _static_dim_runtime_expr(dim, global_names=global_names)


def _fresh_inline_dim(name: str, *, prefix: str) -> str:
    return f"{prefix}__dim_{_py_ident(name).lstrip('_')}"


def _bind_inline_dim_subst(
    formal: Any,
    actual: Any,
    subst: dict[str, Any],
    *,
    fresh_prefix: str,
    protected: set[str],
) -> None:
    if isinstance(formal, TypeTensor) and isinstance(actual, TypeTensor):
        # Tensor shape formals are runtime facts about the actual value. During
        # codegen inlining, binding them from the actual tensor's `.shape` is
        # safer than substituting symbolic annotations that may be less precise
        # or stale after earlier specialization. Freshen the formal names so an
        # inlined callee cannot capture same-named dimensions in its caller.
        for formal_dim in formal.dims:
            for name in dim_token_names(formal_dim):
                if (
                    isinstance(name, str)
                    and name.isidentifier()
                    and not name.startswith("..")
                    and name not in protected
                ):
                    subst.setdefault(name, _fresh_inline_dim(name, prefix=fresh_prefix))
        return
    if isinstance(formal, TypeOptional):
        _bind_inline_dim_subst(
            formal.inner,
            actual.inner if isinstance(actual, TypeOptional) else actual,
            subst,
            fresh_prefix=fresh_prefix,
            protected=protected,
        )
        return
    if isinstance(actual, TypeOptional):
        _bind_inline_dim_subst(
            formal,
            actual.inner,
            subst,
            fresh_prefix=fresh_prefix,
            protected=protected,
        )
        return
    if isinstance(formal, TypeList) and isinstance(actual, TypeList):
        _bind_inline_dim_subst(
            formal.item,
            actual.item,
            subst,
            fresh_prefix=fresh_prefix,
            protected=protected,
        )
        return
    if isinstance(formal, TypeTuple) and isinstance(actual, TypeTuple):
        for formal_item, actual_item in zip(formal.items, actual.items, strict=False):
            _bind_inline_dim_subst(
                formal_item,
                actual_item,
                subst,
                fresh_prefix=fresh_prefix,
                protected=protected,
            )
        return
    if isinstance(formal, TypeNamed) and isinstance(actual, TypeNamed) and len(formal.args) == len(actual.args):
        for formal_dim, actual_dim in zip(formal.args, actual.args, strict=True):
            if (
                isinstance(formal_dim, str)
                and formal_dim.isidentifier()
                and not formal_dim.startswith("..")
                and formal_dim not in protected
            ):
                subst.setdefault(formal_dim, actual_dim)


def _inline_dim_subst(
    params: tuple[GraphValue, ...],
    arg_operands: tuple[GraphOperand, ...],
    *,
    fresh_prefix: str,
    protected: set[str],
) -> dict[str, Any]:
    subst: dict[str, Any] = {}
    for param, operand in zip(params, arg_operands, strict=False):
        _bind_inline_dim_subst(
            param.type_expr,
            graph_operand_type(operand),
            subst,
            fresh_prefix=fresh_prefix,
            protected=protected,
        )
    return {name: dim for name, dim in subst.items() if dim != name}


def _is_int_dim_list_type(type_expr: Any) -> bool:
    return isinstance(type_expr, TypeList) and isinstance(type_expr.item, TypeDim | TypeInt)


def _tuple_literal_expr(items: list[str]) -> str:
    if not items:
        return "()"
    if len(items) == 1:
        return f"({items[0]},)"
    return "(" + ", ".join(items) + ")"


def _collapse_one_numeric_segment(key: str) -> tuple[str, int, int] | None:
    parts = str(key).split(".")
    for index, part in enumerate(parts):
        if part.isdigit():
            return ".".join(parts[:index] + parts[index + 1 :]), int(part), index
    return None


def _collapsed_numeric_segments(key: str) -> list[tuple[str, int, int]]:
    parts = str(key).split(".")
    return [
        (".".join(parts[:index] + parts[index + 1 :]), int(part), index)
        for index, part in enumerate(parts)
        if part.isdigit()
    ]


def _keys_for_collapsed_bank(state: Mapping[str, torch.Tensor], bank_key: str) -> list[str]:
    items: dict[int, str] = {}
    numeric_index: int | None = None
    for key in state:
        for collapsed_key, expert, index in _collapsed_numeric_segments(str(key)):
            if collapsed_key != bank_key:
                continue
            if numeric_index is None:
                numeric_index = index
            elif numeric_index != index:
                continue
            items[expert] = str(key)
            break
    if not items:
        return []
    ordered = [items[i] for i in range(len(items)) if i in items]
    return ordered if len(ordered) == len(items) else []


def _fused_gate_up_source_bank_keys(bank_key: str) -> tuple[str, str] | None:
    parts = bank_key.split(".")
    for index, part in enumerate(parts):
        if "gate_up" not in part:
            continue
        gate_parts = list(parts)
        up_parts = list(parts)
        gate_parts[index] = part.replace("gate_up", "gate", 1)
        up_parts[index] = part.replace("gate_up", "up", 1)
        return ".".join(gate_parts), ".".join(up_parts)
    return None


def _materialize_expert_bank_for_path(state: dict[str, torch.Tensor], bank_key: str) -> torch.Tensor | None:
    existing = state.get(bank_key)
    if torch.is_tensor(existing):
        return existing
    ordered_keys = _keys_for_collapsed_bank(state, bank_key)
    if ordered_keys:
        first = state[ordered_keys[0]]
        first_shape = tuple(first.shape)
        if all(torch.is_tensor(state[key]) and tuple(state[key].shape) == first_shape for key in ordered_keys):
            return _materialize_joined_parameter(
                state,
                bank_key,
                ordered_keys,
                dim=0,
                mode="stack",
                remove_inputs=True,
            )
    fused_sources = _fused_gate_up_source_bank_keys(bank_key)
    if fused_sources is None:
        return None
    gate_key, up_key = fused_sources
    gate = _materialize_expert_bank_for_path(state, gate_key)
    up = _materialize_expert_bank_for_path(state, up_key)
    if not torch.is_tensor(gate) or not torch.is_tensor(up):
        return None
    if gate.shape[:-2] != up.shape[:-2] or gate.shape[-1:] != up.shape[-1:]:
        return None
    concat_dim = -2 if gate.ndim >= 2 else -1
    return _materialize_joined_parameter(
        state,
        bank_key,
        (gate_key, up_key),
        dim=concat_dim,
        mode="cat",
        remove_inputs=True,
    )


def _expert_bank_lookup_from_state(state: dict[str, torch.Tensor], path: str) -> tuple[str, int] | None:
    for bank_key, expert, _ in _collapsed_numeric_segments(path):
        bank = state.get(bank_key)
        if torch.is_tensor(bank):
            return bank_key, expert
    return None


def _grouped_expert_linear_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    expert_idx: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    transpose: bool = False,
) -> torch.Tensor:
    expert_idx = expert_idx.to(device=weight.device, dtype=torch.long)
    x = x.to(device=weight.device)
    out_dim = int(weight.shape[-1] if transpose else weight.shape[-2])
    out = x.new_empty((*x.shape[:-1], out_dim))
    if x.numel() == 0:
        return out
    flat_x = x.reshape(-1, x.shape[-1])
    flat_idx = expert_idx.reshape(-1)
    if flat_idx.numel() != flat_x.shape[0]:
        raise ValueError(
            f"expert_idx shape {tuple(expert_idx.shape)} is incompatible with input shape {tuple(x.shape)}"
        )
    grouped_weight = weight if transpose else weight.transpose(-2, -1)
    expert_ids_g, perm = torch.sort(flat_idx)
    x_g = flat_x.index_select(0, perm)
    x_run = x_g.to(dtype=grouped_weight.dtype) if x_g.is_floating_point() and grouped_weight.is_floating_point() and x_g.dtype != grouped_weight.dtype else x_g
    histc_input = expert_ids_g.float() if weight.device.type in ("cpu", "mps") else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=weight.shape[0], min=0, max=weight.shape[0] - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
    if hasattr(torch.nn.functional, "grouped_mm") and weight.device.type == "cuda":
        y_g = torch.nn.functional.grouped_mm(x_run, grouped_weight, offs=offsets)
    elif hasattr(torch, "_grouped_mm") and weight.device.type == "cuda":
        y_g = torch._grouped_mm(x_run, grouped_weight, offs=offsets)
    else:
        y_g = x_run.new_empty((x_run.shape[0], out_dim))
        start = 0
        for expert, end in enumerate(offsets.tolist()):
            if start != end:
                torch.mm(x_run[start:end], grouped_weight[expert], out=y_g[start:end])
            start = end
    y_g = y_g.to(dtype=x.dtype) if x.is_floating_point() and y_g.is_floating_point() and y_g.dtype != x.dtype else y_g
    if bias is not None:
        bias_g = bias.to(device=weight.device).index_select(0, expert_ids_g)
        bias_g = bias_g.to(dtype=x.dtype) if x.is_floating_point() and bias_g.is_floating_point() and bias_g.dtype != x.dtype else bias_g
        y_g = y_g + bias_g
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)
    return y_g.index_select(0, inv_perm).reshape_as(out)


def _is_global_symbol_module(module: GraphModule) -> bool:
    return module.is_global_binding and not module.inputs and len(module.outputs) == 1


def _is_runtime_symbol_module(module: GraphModule) -> bool:
    return not module.inputs and len(module.outputs) == 1


def _referenced_symbol_names_in_operand(operand: GraphOperand, out: set[str]) -> None:
    if isinstance(operand, GraphValueRef):
        out.add(operand.name)
        return
    if isinstance(operand, GraphPath):
        out.update(graph_path_template_names(operand))
        return
    if isinstance(operand, GraphExpr):
        out.add(operand.op.name)
        for item in operand.inputs:
            _referenced_symbol_names_in_operand(item, out)
        for item in operand.attrs.values():
            _referenced_symbol_names_in_operand(item, out)
        return
    if isinstance(operand, tuple):
        for item in operand:
            _referenced_symbol_names_in_operand(item, out)


def _referenced_type_symbol_names_in_operand(operand: GraphOperand, out: set[str]) -> None:
    out.update(_type_dim_names(graph_operand_type(operand)))
    if isinstance(operand, GraphExpr):
        for item in operand.inputs:
            _referenced_type_symbol_names_in_operand(item, out)
        for item in operand.attrs.values():
            _referenced_type_symbol_names_in_operand(item, out)
    elif isinstance(operand, tuple):
        for item in operand:
            _referenced_type_symbol_names_in_operand(item, out)


def _referenced_runtime_symbol_module_names(program: GraphProgram) -> set[str]:
    modules_by_name = {module.name: module for module in program.modules}
    runtime_symbol_names = {
        name for name, module in modules_by_name.items() if _is_runtime_symbol_module(module)
    }
    referenced: set[str] = set()
    for module in program.modules:
        local = {value.name for value in module.inputs}
        for node in module.nodes:
            local.update(value.name for value in node.outputs)
        names: set[str] = set()
        for value in module.inputs:
            names.update(_type_dim_names(_effective_graph_value_type(value)))
        for operand in module.outputs:
            names.update(_type_dim_names(graph_operand_type(operand)))
        for node in module.nodes:
            for value in node.outputs:
                names.update(_type_dim_names(_effective_graph_value_type(value)))
            for operand in node.inputs:
                _referenced_symbol_names_in_operand(operand, names)
                _referenced_type_symbol_names_in_operand(operand, names)
            for operand in node.attrs.values():
                _referenced_symbol_names_in_operand(operand, names)
                _referenced_type_symbol_names_in_operand(operand, names)
        for operand in module.outputs:
            _referenced_symbol_names_in_operand(operand, names)
            _referenced_type_symbol_names_in_operand(operand, names)
        referenced.update((names - local) & runtime_symbol_names)
    return referenced


def _walk_graph_operands(operand: GraphOperand, out: list[GraphOperand]) -> None:
    out.append(operand)
    if isinstance(operand, GraphExpr):
        for item in operand.inputs:
            _walk_graph_operands(item, out)
        for item in operand.attrs.values():
            _walk_graph_operands(item, out)
    elif isinstance(operand, tuple):
        for item in operand:
            _walk_graph_operands(item, out)


def _state_key_filter_prefixes(program: GraphProgram) -> tuple[str, ...]:
    prefixes: set[str] = set()
    operands: list[GraphOperand] = []
    for module in program.modules:
        for node in module.nodes:
            for operand in node.inputs:
                _walk_graph_operands(operand, operands)
            for operand in node.attrs.values():
                _walk_graph_operands(operand, operands)
        for operand in module.outputs:
            _walk_graph_operands(operand, operands)
    for packed in program.packed_parameters:
        operands.extend(packed.inputs)
    for operand in operands:
        if not isinstance(operand, GraphPath):
            continue
        text = ".".join(part for part in operand.parts if part).strip(".")
        if not text:
            continue
        placeholder = text.find("{")
        prefix = text[:placeholder] if placeholder >= 0 else text
        prefix = prefix.strip(".")
        if prefix:
            prefixes.add(prefix)
            fused_sources = _fused_gate_up_source_bank_keys(prefix)
            if fused_sources is not None:
                prefixes.update(item for item in fused_sources if item)
    return tuple(sorted(prefixes))


def _global_symbol_module_names(
    program: GraphProgram,
    *,
    total_pure_only: bool = False,
) -> set[str]:
    effects = infer_graph_module_effects(program.modules) if total_pure_only else {}
    usages = infer_graph_module_usages(program.modules) if total_pure_only else {}
    return {
        module.name
        for module in program.modules
        if _is_global_symbol_module(module)
        and (
            not total_pure_only
            or (
                effects.get(module.name) == GraphEffect.TOTAL_PURE
                and usages.get(module.name) == UsageClass.UNRESTRICTED
            )
        )
    }


def _runtime_symbol_module_names(
    program: GraphProgram,
    *,
    total_pure_only: bool = False,
) -> set[str]:
    effects = infer_graph_module_effects(program.modules)
    usages = infer_graph_module_usages(program.modules)
    names = _global_symbol_module_names(program)
    names.update(
        name
        for name in _referenced_runtime_symbol_module_names(program)
        if effects.get(name) != GraphEffect.EFFECTFUL
        and usages.get(name) == UsageClass.UNRESTRICTED
    )
    if not total_pure_only:
        return names
    return {
        name
        for name in names
        if effects.get(name) == GraphEffect.TOTAL_PURE
        and usages.get(name) == UsageClass.UNRESTRICTED
    }


def _free_dim_refs_in_operand(operand: GraphOperand, *, local: set[str], out: set[str]) -> None:
    if isinstance(operand, GraphValueRef):
        if operand.name not in local and isinstance(operand.type_expr, TypeDim):
            out.add(operand.name)
        # Direct codegen for primitives and inlined modules may use symbolic
        # dimensions from tensor-typed value operands (for example reshape
        # shapes derived from TokenIds[B,S]). Treat those shape variables as
        # required when the value itself is used, otherwise generated code can
        # reference them before any output happens to bind them.
        out.update(name for name in _type_dim_names(operand.type_expr) if name not in local)
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


def _owned_inplace_primitive_ops(program: GraphProgram, inplace_node_ids: frozenset[str]) -> frozenset[str]:
    by_op: dict[str, set[str]] = {}
    for module in program.modules:
        for node in module.nodes:
            if node.op.name.startswith("core."):
                continue
            by_op.setdefault(node.op.name, set()).add(node.id)
    return frozenset(
        op_name
        for op_name, node_ids in by_op.items()
        if node_ids and node_ids <= inplace_node_ids
    )


def _collect_term_dim_ref_names(operand: GraphOperand, out: set[str]) -> None:
    if isinstance(operand, GraphValueRef):
        operand_type = operand.type_expr
        if isinstance(operand_type, TypeOptional):
            operand_type = operand_type.inner
        if isinstance(operand_type, TypeDim | TypeInt):
            out.add(operand.name)
        return
    if isinstance(operand, GraphExpr):
        for item in operand.inputs:
            _collect_term_dim_ref_names(item, out)
        for item in operand.attrs.values():
            _collect_term_dim_ref_names(item, out)
        return
    if isinstance(operand, tuple):
        for item in operand:
            _collect_term_dim_ref_names(item, out)


def _operand_may_be_none(operand: GraphOperand) -> bool:
    if isinstance(operand, GraphLiteral):
        return operand.value is None
    operand_type = graph_operand_type(operand)
    return isinstance(operand_type, TypeOptional)


def _module_value_required_shape_dim_names(
    module: GraphModule,
    *,
    global_names: set[str],
) -> dict[str, set[str]]:
    """Map each value to shape dim names required by later type annotations.

    Codegen only needs to bind symbolic dimensions from a value's runtime shape
    when those symbols are referenced later. Binding every name from every
    output type can turn stale or intentionally local type variables into Python
    locals, and can create bogus code for inlined helpers whose formal dim names
    are no longer in caller scope.
    """

    required_term_dims: set[str] = set()
    required: dict[str, set[str]] = {}
    for output in module.outputs:
        _collect_term_dim_ref_names(output, required_term_dims)
    for node in module.nodes:
        for operand in (*node.inputs, *node.attrs.values()):
            _collect_term_dim_ref_names(operand, required_term_dims)
    required_term_dims = {name for name in required_term_dims if name not in global_names}
    for node in module.nodes:
        for output in node.outputs:
            required[output.name] = _type_dim_names(output.type_expr) & required_term_dims
    for value in module.inputs:
        names = _type_dim_names(value.type_expr) & required_term_dims
        if names:
            required[value.name] = names
    return required


def _set_dim_symbol_runtime(
    symbols: dict[str, Any],
    name: str,
    value: int,
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        symbols[name] = value
    else:
        symbols.setdefault(name, value)


def _bind_dim_expr_runtime(
    dim: Any,
    actual: int,
    symbols: dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    if isinstance(dim, str):
        _set_dim_symbol_runtime(symbols, dim, actual, overwrite=overwrite)
        return
    if not isinstance(dim, DimExprBinary):
        return
    left = dim.left
    right = dim.right
    if dim.op == "+":
        if isinstance(left, str) and right in symbols:
            _set_dim_symbol_runtime(
                symbols, left, actual - int(symbols[right]), overwrite=overwrite
            )
        if isinstance(right, str) and left in symbols:
            _set_dim_symbol_runtime(
                symbols, right, actual - int(symbols[left]), overwrite=overwrite
            )
    if dim.op == "-":
        if isinstance(left, str) and right in symbols:
            _set_dim_symbol_runtime(
                symbols, left, actual + int(symbols[right]), overwrite=overwrite
            )
        if isinstance(right, str) and left in symbols:
            _set_dim_symbol_runtime(
                symbols, right, int(symbols[left]) - actual, overwrite=overwrite
            )
    if dim.op == "*":
        if isinstance(left, str) and right in symbols and int(symbols[right]) != 0:
            _set_dim_symbol_runtime(
                symbols, left, actual // int(symbols[right]), overwrite=overwrite
            )
        if isinstance(right, str) and left in symbols and int(symbols[left]) != 0:
            _set_dim_symbol_runtime(
                symbols, right, actual // int(symbols[left]), overwrite=overwrite
            )
    if dim.op == "/":
        if isinstance(left, str) and right in symbols:
            _set_dim_symbol_runtime(
                symbols, left, actual * int(symbols[right]), overwrite=overwrite
            )
        if isinstance(right, str) and left in symbols and actual != 0:
            _set_dim_symbol_runtime(
                symbols, right, int(symbols[left]) // actual, overwrite=overwrite
            )


def _bind_nested_shape_symbols_runtime(
    type_expr: Any,
    value: Any,
    symbols: dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    if isinstance(type_expr, TypeOptional):
        if value is not None:
            _bind_nested_shape_symbols_runtime(
                type_expr.inner, value, symbols, overwrite=overwrite
            )
        return
    if isinstance(type_expr, TypeList):
        if isinstance(value, list | tuple) and value:
            _bind_nested_shape_symbols_runtime(
                type_expr.item, value[0], symbols, overwrite=overwrite
            )
        return
    if isinstance(type_expr, TypeTuple):
        if isinstance(value, list | tuple):
            for item_type, item_value in zip(type_expr.items, value, strict=False):
                _bind_nested_shape_symbols_runtime(
                    item_type, item_value, symbols, overwrite=overwrite
                )
        return
    if isinstance(type_expr, TypeTensor) and torch.is_tensor(value):
        for idx, dim in enumerate(type_expr.dims):
            if idx < value.dim():
                _bind_dim_expr_runtime(
                    dim, int(value.shape[idx]), symbols, overwrite=overwrite
                )


def _effective_graph_value_type(value: GraphValue) -> TypeExpr:
    if value.optional and not isinstance(value.type_expr, TypeOptional):
        return TypeOptional(value.type_expr)
    return value.type_expr


def _direct_tensor_dim_names(type_expr: TypeExpr) -> set[str]:
    if isinstance(type_expr, TypeOptional):
        return _direct_tensor_dim_names(type_expr.inner)
    if not isinstance(type_expr, TypeTensor):
        return set()
    out: set[str] = set()
    for dim in type_expr.dims:
        out.update(dim_token_names(dim))
    return out


def _tensor_size_static_dim_names(module: GraphModule) -> set[str]:
    out: set[str] = set()
    for node in module.nodes:
        if _normalize_primitive_op(node.op.name) != "tensor_size" or len(node.inputs) < 2:
            continue
        tensor_type = getattr(node.inputs[0], "type_expr", None)
        if isinstance(tensor_type, TypeOptional):
            tensor_type = tensor_type.inner
        dim_operand = node.inputs[1]
        if not isinstance(tensor_type, TypeTensor) or not isinstance(dim_operand, GraphLiteral):
            continue
        if type(dim_operand.value) is not int or not tensor_type.dims:
            continue
        index = dim_operand.value
        if index < 0:
            index += len(tensor_type.dims)
        if 0 <= index < len(tensor_type.dims):
            out.update(dim_token_names(tensor_type.dims[index]))
    return out


def _emit_bind_dim_expr(
    lines: list[str],
    *,
    add: Any,
    dim: Any,
    actual_expr: str,
    local: set[str],
    protected: set[str],
    indent: int,
    guaranteed: bool,
    required_names: set[str] | None = None,
) -> None:
    if required_names is not None and not (dim_token_names(dim) & required_names):
        return
    if isinstance(dim, str):
        if (
            dim.isidentifier()
            and dim not in protected
            and dim not in local
            and (required_names is None or dim in required_names)
        ):
            add(lines, indent, f"{_dim_ident(dim)} = {actual_expr}")
            local.add(dim)
        return
    if not isinstance(dim, DimExprBinary):
        return
    left = dim.left
    right = dim.right
    if dim.op == "+":
        if (
            isinstance(left, str)
            and left.isidentifier()
            and left not in protected
            and left not in local
            and (required_names is None or left in required_names)
            and isinstance(right, str)
            and right in local
        ):
            add(lines, indent, f"{_dim_ident(left)} = {actual_expr} - {_local_dim_ref(right, local)}")
            local.add(left)
        if (
            isinstance(right, str)
            and right.isidentifier()
            and right not in protected
            and right not in local
            and (required_names is None or right in required_names)
            and isinstance(left, str)
            and left in local
        ):
            add(lines, indent, f"{_dim_ident(right)} = {actual_expr} - {_local_dim_ref(left, local)}")
            local.add(right)
        return
    if dim.op == "-":
        if (
            isinstance(left, str)
            and left.isidentifier()
            and left not in protected
            and left not in local
            and (required_names is None or left in required_names)
            and isinstance(right, str)
            and right in local
        ):
            add(lines, indent, f"{_dim_ident(left)} = {actual_expr} + {_local_dim_ref(right, local)}")
            local.add(left)
        if (
            isinstance(right, str)
            and right.isidentifier()
            and right not in protected
            and right not in local
            and (required_names is None or right in required_names)
            and isinstance(left, str)
            and left in local
        ):
            add(lines, indent, f"{_dim_ident(right)} = {_local_dim_ref(left, local)} - {actual_expr}")
            local.add(right)
        return
    if dim.op == "*":
        if (
            isinstance(left, str)
            and left.isidentifier()
            and left not in protected
            and left not in local
            and (required_names is None or left in required_names)
            and isinstance(right, str)
            and right in local
        ):
            add(lines, indent, f"{_dim_ident(left)} = {actual_expr} // {_local_dim_ref(right, local)}")
            local.add(left)
        if (
            isinstance(right, str)
            and right.isidentifier()
            and right not in protected
            and right not in local
            and (required_names is None or right in required_names)
            and isinstance(left, str)
            and left in local
        ):
            add(lines, indent, f"{_dim_ident(right)} = {actual_expr} // {_local_dim_ref(left, local)}")
            local.add(right)
        return
    if dim.op == "/":
        if (
            isinstance(left, str)
            and left.isidentifier()
            and left not in protected
            and left not in local
            and (required_names is None or left in required_names)
            and isinstance(right, str)
            and right in local
        ):
            add(lines, indent, f"{_dim_ident(left)} = {actual_expr} * {_local_dim_ref(right, local)}")
            local.add(left)
        if (
            isinstance(right, str)
            and right.isidentifier()
            and right not in protected
            and right not in local
            and (required_names is None or right in required_names)
            and isinstance(left, str)
            and left in local
        ):
            add(lines, indent, f"{_dim_ident(right)} = {_local_dim_ref(left, local)} // {actual_expr}")
            local.add(right)


def _emit_bind_nested_shape_symbols(
    lines: list[str],
    *,
    add: Any,
    type_expr: Any,
    value_expr: str,
    local: set[str],
    protected: set[str],
    required_names: set[str] | None = None,
    indent: int = 8,
) -> None:
    if isinstance(type_expr, TypeOptional):
        nested_lines: list[str] = []
        nested_local = set(local)
        _emit_bind_nested_shape_symbols_inner(
            nested_lines,
            add=add,
            type_expr=type_expr.inner,
            value_expr=value_expr,
            local=nested_local,
            protected=protected,
            required_names=required_names,
            indent=indent + 4,
            guaranteed=True,
        )
        if nested_lines:
            add(lines, indent, f"if {value_expr} is not None:")
            lines.extend(nested_lines)
        return
    _emit_bind_nested_shape_symbols_inner(
        lines,
        add=add,
        type_expr=type_expr,
        value_expr=value_expr,
        local=local,
        protected=protected,
        required_names=required_names,
        indent=indent,
        guaranteed=True,
    )


def _emit_bind_nested_shape_symbols_inner(
    lines: list[str],
    *,
    add: Any,
    type_expr: Any,
    value_expr: str,
    local: set[str],
    protected: set[str],
    required_names: set[str] | None,
    indent: int,
    guaranteed: bool,
) -> None:
    if isinstance(type_expr, TypeOptional):
        nested_lines: list[str] = []
        nested_local = set(local)
        _emit_bind_nested_shape_symbols_inner(
            nested_lines,
            add=add,
            type_expr=type_expr.inner,
            value_expr=value_expr,
            local=nested_local,
            protected=protected,
            required_names=required_names,
            indent=indent + 4,
            guaranteed=True,
        )
        if nested_lines:
            add(lines, indent, f"if {value_expr} is not None:")
            lines.extend(nested_lines)
        return
    if isinstance(type_expr, TypeList):
        nested_lines: list[str] = []
        nested_local = set(local)
        _emit_bind_nested_shape_symbols_inner(
            nested_lines,
            add=add,
            type_expr=type_expr.item,
            value_expr=f"{value_expr}[0]",
            local=nested_local,
            protected=protected,
            required_names=required_names,
            indent=indent + 4,
            guaranteed=False,
        )
        if nested_lines:
            add(lines, indent, f"if isinstance({value_expr}, (list, tuple)) and {value_expr}:")
            lines.extend(nested_lines)
        return
    if isinstance(type_expr, TypeTuple):
        nested_lines: list[str] = []
        nested_local = set(local)
        for idx, item_type in enumerate(type_expr.items):
            _emit_bind_nested_shape_symbols_inner(
                nested_lines,
                add=add,
                type_expr=item_type,
                value_expr=f"{value_expr}[{idx}]",
                local=nested_local,
                protected=protected,
                required_names=required_names,
                indent=indent + 4,
                guaranteed=guaranteed,
            )
        if nested_lines:
            add(lines, indent, f"if {value_expr} is not None:")
            lines.extend(nested_lines)
            local.update(nested_local)
        return
    if isinstance(type_expr, TypeTensor):
        for idx, dim in enumerate(type_expr.dims):
            _emit_bind_dim_expr(
                lines,
                add=add,
                dim=dim,
                actual_expr=f"{value_expr}.shape[{idx}]",
                local=local,
                protected=protected,
                required_names=required_names,
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
        self.needs_expert_banks = _graph_uses_expert_linear(graph)
        self.global_symbol_names = _runtime_symbol_module_names(graph)
        self.cached_global_symbol_names = _runtime_symbol_module_names(
            graph,
            total_pure_only=True,
        )
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
        self._cached_global_symbols: dict[str, Any] | None = None
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
        _materialize_packed_parameters(
            loaded,
            tuple(_packed_parameter_spec_payload(item) for item in self.graph.packed_parameters),
        )
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

    @staticmethod
    def _path_template_part(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("@@"):
            return value[2:].strip(".")
        if isinstance(value, str) and value.startswith("@"):
            return value[1:].strip(".")
        return value

    @staticmethod
    def _conv1d(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        stride: Any,
        padding_left: Any,
        padding_right: Any,
        dilation: Any,
        groups: Any,
    ) -> torch.Tensor:
        if x.device != weight.device:
            x = x.to(device=weight.device)
        if x.is_floating_point() and weight.is_floating_point() and x.dtype != weight.dtype:
            weight = weight.to(dtype=x.dtype)
        if bias is not None:
            if bias.device != weight.device:
                bias = bias.to(device=weight.device)
            if x.is_floating_point() and bias.is_floating_point() and x.dtype != bias.dtype:
                bias = bias.to(dtype=x.dtype)
        left = int(padding_left)
        right = int(padding_right)
        if left or right:
            x = F.pad(x, (left, right))
        return F.conv1d(
            x,
            weight,
            bias=bias,
            stride=int(stride),
            dilation=int(dilation),
            groups=int(groups),
        )

    @staticmethod
    def _assign_slice(
        x: torch.Tensor,
        src: torch.Tensor,
        dim: Any,
        start: Any,
        end: Any,
    ) -> torch.Tensor:
        out = x.clone()
        dim = int(dim)
        if dim < 0:
            dim += out.dim()
        start = int(start)
        end = int(end)
        sl = [slice(None)] * out.dim()
        sl[dim] = slice(start, end)
        if src.device != out.device:
            src = src.to(device=out.device)
        if out.is_floating_point() and src.is_floating_point() and out.dtype != src.dtype:
            src = src.to(dtype=out.dtype)
        out[tuple(sl)] = src
        return out

    def _scatter(self, x: torch.Tensor, index: torch.Tensor, src: Any, dim: Any) -> torch.Tensor:
        dim = int(dim)
        if torch.is_tensor(src):
            index = self._move_to(index, x.device)
            src = self._move_to(src, x.device)
            return torch.scatter(x, dim=dim, index=index, src=src)
        index = self._move_to(index, x.device)
        return torch.scatter(x, dim=dim, index=index, value=src)

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
        value = self._state.get(resolved)
        if torch.is_tensor(value):
            return value
        _materialize_expert_bank_for_path(self._state, resolved)
        value = self._state.get(resolved)
        if torch.is_tensor(value):
            return value
        bank = _expert_bank_lookup_from_state(self._state, resolved)
        if bank is not None:
            bank_key, expert = bank
            bank_value = self._state.get(bank_key)
            if torch.is_tensor(bank_value):
                return bank_value[expert]
        if resolved not in self._state:
            alternatives = self._state_key_alternatives(resolved, limit=8)
            alt_text = ", ".join(alternatives) if alternatives else "<none>"
            raise ValueError(
                f"{field} tensor not found at path: {resolved}. Alternatives: {alt_text}"
            )
        return self._state[resolved]

    def _linear_param(
        self,
        path: str,
        expert: int | None,
        *,
        optional: bool = False,
        field: str = "linear.weight",
    ) -> tuple[torch.Tensor | None, int | None]:
        resolved = path[2:] if isinstance(path, str) and path.startswith("@@") else path
        value = self._state.get(resolved)
        if torch.is_tensor(value):
            return value, expert
        _materialize_expert_bank_for_path(self._state, resolved)
        value = self._state.get(resolved)
        if torch.is_tensor(value):
            return value, expert
        bank = _expert_bank_lookup_from_state(self._state, resolved)
        if bank is not None:
            bank_key, path_expert = bank
            bank_value = self._state.get(bank_key)
            if torch.is_tensor(bank_value):
                if expert is None or expert == path_expert:
                    return bank_value[path_expert], None
                return bank_value, expert
        if optional:
            return None, expert
        return self._state_tensor_from_resolved_path(resolved, field=field), expert

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
        names = _main_output_names(self.graph, self.modules_by_name[self.graph.main_module])
        if len(result) == 1:
            return result[0]
        outputs = {name: value for name, value in zip(names, result, strict=False)}
        return outputs

    def _evaluate_global_symbols(self, env: dict[str, Any]) -> dict[str, Any]:
        if self._cached_global_symbols is None:
            self._cached_global_symbols = self._evaluate_global_symbol_subset(
                names=self.cached_global_symbol_names,
                env={},
                initial={},
            )
        symbols: dict[str, Any] = dict(self._cached_global_symbols)
        pending_names = self.global_symbol_names - set(symbols)
        if not pending_names:
            return symbols
        symbols.update(
            self._evaluate_global_symbol_subset(
                names=pending_names,
                env=env,
                initial=symbols,
            )
        )
        return symbols

    def _evaluate_global_symbol_subset(
        self,
        *,
        names: set[str],
        env: dict[str, Any],
        initial: dict[str, Any],
    ) -> dict[str, Any]:
        symbols: dict[str, Any] = dict(initial)
        pending = sorted(names)
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
        return {name: symbols[name] for name in names if name in symbols}

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
            if (
                operand.op.name in self.cached_global_symbol_names
                and not operand.inputs
                and not operand.attrs
                and operand.op.name in symbols
            ):
                return symbols[operand.op.name]
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
                "outputs": [
                    GraphValue(
                        name=out_name,
                        type_expr=expr.type_expr,
                        dims=expr.dims,
                    )
                ],
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
    def _path_parts(value: Any) -> tuple[bool, str]:
        return path_parts(value)

    def _compose_path(self, base: Any, leaf: Any) -> str:
        return compose_path(base, leaf)

    def _required_param(self, path: str, *, field: str) -> torch.Tensor:
        return self._state_tensor_from_resolved_path(path, field=field)

    def _optional_param(self, path: str) -> torch.Tensor | None:
        resolved = path[2:] if isinstance(path, str) and path.startswith("@@") else path
        value = self._state.get(resolved)
        if torch.is_tensor(value):
            return value
        bank = self._expert_bank_lookup(resolved)
        if bank is not None:
            bank_key, expert = bank
            bank_value = self._state.get(bank_key)
            if torch.is_tensor(bank_value):
                return bank_value[expert]
        return value if torch.is_tensor(value) else None

    @classmethod
    def _move_to(cls, value: Any, device: torch.device | str) -> Any:
        if torch.is_tensor(value):
            return value.to(device=device)
        if isinstance(value, tuple):
            return tuple(cls._move_to(item, device) for item in value)
        if isinstance(value, list):
            return [cls._move_to(item, device) for item in value]
        return value

    @classmethod
    def _align_pair(cls, left: Any, right: Any, *, prefer: str = "right") -> tuple[Any, Any]:
        left_tensor = torch.is_tensor(left)
        right_tensor = torch.is_tensor(right)
        if not left_tensor or not right_tensor or left.device == right.device:
            return left, right
        device = right.device if prefer == "right" else left.device
        return cls._move_to(left, device), cls._move_to(right, device)

    @classmethod
    def _rope_apply_factors(cls, x: Any, sin: Any, cos: Any, interleaved: bool = False) -> Any:
        if interleaved:
            raise NotImplementedError("__torch_rope_apply_factors only supports non-interleaved RoPE")
        if torch.is_tensor(x):
            sin = cls._move_to(sin, x.device)
            cos = cls._move_to(cos, x.device)
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return (x * cos) + (rotated * sin)

    @staticmethod
    def _gegelu(x: torch.Tensor, limit: Any = None, alpha: Any = 1.702) -> torch.Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError("gegelu requires even last dimension")
        x_gelu = x[..., ::2]
        x_linear = x[..., 1::2]
        if limit is not None:
            limit = float(limit)
            x_gelu = torch.where(torch.isinf(x_gelu), x_gelu, x_gelu.clamp(max=limit))
            x_linear = torch.where(torch.isinf(x_linear), x_linear, x_linear.clamp(min=-limit, max=limit))
        return x_gelu * torch.sigmoid(float(alpha) * x_gelu) * (x_linear + 1.0)

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

    @classmethod
    def _tensor_like(cls, value: Any, ref: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        target_dtype = dtype or ref.dtype
        if torch.is_tensor(value):
            return value.to(device=ref.device, dtype=target_dtype)
        if isinstance(value, (list, tuple)):
            return torch.as_tensor(value, device=ref.device, dtype=target_dtype)
        return torch.full_like(ref, value, dtype=target_dtype)

    @staticmethod
    def _fused_linear_pair_weight_path(gate_weight_path: Any, up_weight_path: Any) -> str:
        return "__fused_gate_up__." + str(gate_weight_path).lstrip("@") + "||" + str(up_weight_path).lstrip("@")

    @staticmethod
    def _fused_linear_pair_bias_path(gate_bias_path: Any, up_bias_path: Any) -> str:
        return "__fused_gate_up_bias__." + str(gate_bias_path).lstrip("@") + "||" + str(up_bias_path).lstrip("@")

    def _swiglu_ffn(
        self,
        x: Any,
        gate_weight_path: Any,
        up_weight_path: Any,
        down_weight_path: Any,
        gate_bias_path: Any = "bias",
        up_bias_path: Any = "bias",
        down_bias_path: Any = "bias",
    ) -> Any:
        del down_bias_path
        gate, up = self._primitive_gate_up_linear_pair(
            x,
            gate_weight_path,
            up_weight_path,
            gate_bias_path=gate_bias_path,
            up_bias_path=up_bias_path,
            bias=False,
            transpose=False,
        )
        hidden = F.silu(gate) * up
        down_weight = self._required_param(str(down_weight_path), field="swiglu_ffn.down.weight")
        hidden = self._move_to(hidden, down_weight.device)
        return F.linear(hidden, down_weight, None)

    def _expert_swiglu_ffn(
        self,
        x: Any,
        expert_idx: Any,
        gate_weight_path: Any,
        up_weight_path: Any,
        down_weight_path: Any,
    ) -> Any:
        gate_weight = self._required_param(str(gate_weight_path), field="expert_swiglu_ffn.gate.weight")
        up_weight = self._required_param(str(up_weight_path), field="expert_swiglu_ffn.up.weight")
        down_weight = self._required_param(str(down_weight_path), field="expert_swiglu_ffn.down.weight")
        gate = _grouped_expert_linear_torch(x, gate_weight, expert_idx, None, transpose=False)
        up = _grouped_expert_linear_torch(x, up_weight, expert_idx, None, transpose=False)
        hidden = F.silu(gate) * up
        return _grouped_expert_linear_torch(hidden, down_weight, expert_idx, None, transpose=False)

    def _expert_packed_swiglu_ffn(
        self,
        x: Any,
        expert_idx: Any,
        gate_up_weight_path: Any,
        down_weight_path: Any,
        transpose: bool = False,
    ) -> Any:
        gate_up_weight = self._required_param(str(gate_up_weight_path), field="expert_packed_swiglu_ffn.gate_up.weight")
        down_weight = self._required_param(str(down_weight_path), field="expert_packed_swiglu_ffn.down.weight")
        gate_up = _grouped_expert_linear_torch(x, gate_up_weight, expert_idx, None, transpose=transpose)
        gate, up = torch.chunk(gate_up, 2, dim=-1)
        hidden = F.silu(gate) * up
        return _grouped_expert_linear_torch(hidden, down_weight, expert_idx, None, transpose=transpose)

    def _selected_expert_packed_swiglu_ffn(
        self,
        x: Any,
        topk_scores: Any,
        topk_indices: Any,
        gate_up_weight_path: Any,
        down_weight_path: Any,
        transpose: bool = False,
    ) -> Any:
        topk_indices = topk_indices.to(dtype=torch.long)
        expanded = torch.unsqueeze(x, 2).expand((*topk_indices.shape, x.shape[-1]))
        values = self._expert_packed_swiglu_ffn(
            expanded,
            topk_indices,
            gate_up_weight_path,
            down_weight_path,
            transpose=transpose,
        )
        weights = torch.unsqueeze(topk_scores.to(device=values.device, dtype=values.dtype), -1)
        return torch.sum(values * weights, dim=2, keepdim=False)

    def _selected_expert_swiglu_ffn(
        self,
        x: Any,
        topk_scores: Any,
        topk_indices: Any,
        gate_weight_path: Any,
        up_weight_path: Any,
        down_weight_path: Any,
        transpose: bool = False,
    ) -> Any:
        topk_indices = topk_indices.to(dtype=torch.long)
        expanded = torch.unsqueeze(x, 2).expand((*topk_indices.shape, x.shape[-1]))
        gate_weight = self._required_param(str(gate_weight_path), field="selected_expert_swiglu_ffn.gate.weight")
        up_weight = self._required_param(str(up_weight_path), field="selected_expert_swiglu_ffn.up.weight")
        down_weight = self._required_param(str(down_weight_path), field="selected_expert_swiglu_ffn.down.weight")
        gate = _grouped_expert_linear_torch(expanded, gate_weight, topk_indices, None, transpose=transpose)
        up = _grouped_expert_linear_torch(expanded, up_weight, topk_indices, None, transpose=transpose)
        hidden = F.silu(gate) * up
        values = _grouped_expert_linear_torch(hidden, down_weight, topk_indices, None, transpose=transpose)
        weights = torch.unsqueeze(topk_scores.to(device=values.device, dtype=values.dtype), -1)
        return torch.sum(values * weights, dim=2, keepdim=False)

    def _selected_expert_packed_gegelu_ffn(
        self,
        x: Any,
        topk_scores: Any,
        topk_indices: Any,
        gate_up_weight_path: Any,
        gate_up_bias_path: Any,
        down_weight_path: Any,
        down_bias_path: Any,
        limit: Any,
        alpha: Any = 1.702,
        bias: bool = False,
        transpose: bool = False,
    ) -> Any:
        topk_indices = topk_indices.to(dtype=torch.long)
        expanded = torch.unsqueeze(x, 2).expand((*topk_indices.shape, x.shape[-1]))
        gate_up_weight = self._required_param(str(gate_up_weight_path), field="selected_expert_packed_gegelu_ffn.gate_up.weight")
        gate_up_bias = self._required_param(str(gate_up_bias_path), field="selected_expert_packed_gegelu_ffn.gate_up.bias") if bias else None
        down_weight = self._required_param(str(down_weight_path), field="selected_expert_packed_gegelu_ffn.down.weight")
        down_bias = self._required_param(str(down_bias_path), field="selected_expert_packed_gegelu_ffn.down.bias") if bias else None
        gate_up = _grouped_expert_linear_torch(expanded, gate_up_weight, topk_indices, gate_up_bias, transpose=transpose)
        hidden = self._gegelu(gate_up, limit, alpha)
        values = _grouped_expert_linear_torch(hidden, down_weight, topk_indices, down_bias, transpose=transpose)
        weights = torch.unsqueeze(topk_scores.to(device=values.device, dtype=values.dtype), -1)
        return torch.sum(values * weights, dim=2, keepdim=False)

    def _selected_expert_clamped_packed_swiglu_ffn(
        self,
        x: Any,
        topk_scores: Any,
        topk_indices: Any,
        gate_up_weight_path: Any,
        down_weight_path: Any,
        limit: Any,
        transpose: bool = False,
    ) -> Any:
        topk_indices = topk_indices.to(dtype=torch.long)
        expanded = torch.unsqueeze(x, 2).expand((*topk_indices.shape, x.shape[-1]))
        gate_up_weight = self._required_param(str(gate_up_weight_path), field="selected_expert_clamped_packed_swiglu_ffn.gate_up.weight")
        down_weight = self._required_param(str(down_weight_path), field="selected_expert_clamped_packed_swiglu_ffn.down.weight")
        gate_up = _grouped_expert_linear_torch(expanded, gate_up_weight, topk_indices, None, transpose=transpose)
        gate, up = torch.chunk(gate_up, 2, dim=-1)
        limit = float(limit)
        gate = torch.where(torch.isinf(gate), gate, gate.clamp(max=limit))
        up = torch.where(torch.isinf(up), up, up.clamp(min=-limit, max=limit))
        hidden = F.silu(gate) * up
        values = _grouped_expert_linear_torch(hidden, down_weight, topk_indices, None, transpose=transpose)
        weights = torch.unsqueeze(topk_scores.to(device=values.device, dtype=values.dtype), -1)
        return torch.sum(values * weights, dim=2, keepdim=False)

    def _selected_expert_relu2_ffn(
        self,
        x: Any,
        topk_scores: Any,
        topk_indices: Any,
        up_weight_path: Any,
        down_weight_path: Any,
        transpose: bool = False,
    ) -> Any:
        topk_indices = topk_indices.to(dtype=torch.long)
        expanded = torch.unsqueeze(x, 2).expand((*topk_indices.shape, x.shape[-1]))
        up_weight = self._required_param(str(up_weight_path), field="selected_expert_relu2_ffn.up.weight")
        down_weight = self._required_param(str(down_weight_path), field="selected_expert_relu2_ffn.down.weight")
        up = _grouped_expert_linear_torch(expanded, up_weight, topk_indices, None, transpose=transpose)
        hidden = F.relu(up) * F.relu(up)
        values = _grouped_expert_linear_torch(hidden, down_weight, topk_indices, None, transpose=transpose)
        weights = torch.unsqueeze(topk_scores.to(device=values.device, dtype=values.dtype), -1)
        return torch.sum(values * weights, dim=2, keepdim=False)

    def _primitive_gate_up_linear_pair(
        self,
        x: Any,
        gate_weight_path: Any,
        up_weight_path: Any,
        *,
        gate_bias_path: Any = "bias",
        up_bias_path: Any = "bias",
        bias: bool = False,
        transpose: bool = False,
    ) -> tuple[Any, Any]:
        fused_weight_path = self._fused_linear_pair_weight_path(gate_weight_path, up_weight_path)
        fused_bias_path = self._fused_linear_pair_bias_path(gate_bias_path, up_bias_path)
        fused_weight = self.state_dict_tensors.get(fused_weight_path)
        fused_bias = self.state_dict_tensors.get(fused_bias_path)
        if not torch.is_tensor(fused_weight):
            gate_key = str(gate_weight_path).lstrip("@")
            up_key = str(up_weight_path).lstrip("@")
            gate_weight = self.state_dict_tensors.pop(gate_key)
            up_weight = self.state_dict_tensors.pop(up_key)
            concat_dim = -1 if transpose else -2
            fused_weight = torch.cat([gate_weight, up_weight], dim=concat_dim)
            self.state_dict_tensors[fused_weight_path] = fused_weight
            fused_bias = None
            if bias:
                gate_bias = self.state_dict_tensors.pop(str(gate_bias_path).lstrip("@"), None)
                up_bias = self.state_dict_tensors.pop(str(up_bias_path).lstrip("@"), None)
                if torch.is_tensor(gate_bias) and torch.is_tensor(up_bias):
                    fused_bias = torch.cat([gate_bias, up_bias], dim=-1)
                    self.state_dict_tensors[fused_bias_path] = fused_bias
        x = self._move_to(x, fused_weight.device)
        weight_run = fused_weight.to(dtype=x.dtype) if x.is_floating_point() and fused_weight.is_floating_point() and x.dtype != fused_weight.dtype else fused_weight
        bias_run = fused_bias.to(dtype=x.dtype) if fused_bias is not None and x.is_floating_point() and fused_bias.is_floating_point() and x.dtype != fused_bias.dtype else fused_bias
        combined = torch.matmul(x, weight_run) + (bias_run if bias_run is not None else 0) if transpose else F.linear(x, weight_run, bias_run)
        return torch.chunk(combined, 2, dim=-1)

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
            self._assign_outputs(out_names, value, env, outputs=node.outputs, symbols=symbols)

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
                x = self._move_to(x, weight.device)
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
            weight, expert = self._linear_param(
                self._compose_path(base, weight_leaf),
                expert,
                field="linear.weight",
            )
            if expert is not None:
                weight = weight[expert]
            bias = None
            if bias_flag:
                bias, bias_expert = self._linear_param(
                    self._compose_path(base, bias_leaf),
                    expert,
                    optional=True,
                    field="linear.bias",
                )
                if bias is not None and bias_expert is not None and bias.ndim >= 2:
                    bias = bias[bias_expert]
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

        if primitive == "_torch_gate_up_linear_pair":
            if len(args) < 7:
                raise ValueError("__torch_gate_up_linear_pair expects input, gate/up weight paths, gate/up bias paths, bias, and transpose")
            out(
                self._primitive_gate_up_linear_pair(
                    args[0],
                    args[1],
                    args[2],
                    gate_bias_path=args[3],
                    up_bias_path=args[4],
                    bias=bool(args[5]),
                    transpose=bool(args[6]),
                )
            )
            return True

        if primitive == "_torch_swiglu_ffn":
            if len(args) < 7:
                raise ValueError("__torch_swiglu_ffn expects input, gate/up/down weight paths, and gate/up/down bias paths")
            out(self._swiglu_ffn(args[0], args[1], args[2], args[3], args[4], args[5], args[6]))
            return True

        if primitive == "_torch_expert_swiglu_ffn":
            if len(args) < 5:
                raise ValueError("__torch_expert_swiglu_ffn expects input, expert indices, and gate/up/down weight paths")
            out(self._expert_swiglu_ffn(args[0], args[1], args[2], args[3], args[4]))
            return True

        if primitive == "_torch_expert_packed_swiglu_ffn":
            if len(args) < 5:
                raise ValueError("__torch_expert_packed_swiglu_ffn expects input, expert indices, gate-up/down weight paths, and transpose")
            out(self._expert_packed_swiglu_ffn(args[0], args[1], args[2], args[3], transpose=bool(args[4])))
            return True

        if primitive == "_torch_selected_expert_packed_swiglu_ffn":
            if len(args) < 6:
                raise ValueError("__torch_selected_expert_packed_swiglu_ffn expects input, top-k scores/indices, gate-up/down weight paths, and transpose")
            out(
                self._selected_expert_packed_swiglu_ffn(
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4],
                    transpose=bool(args[5]),
                )
            )
            return True

        if primitive == "_torch_selected_expert_swiglu_ffn":
            if len(args) < 7:
                raise ValueError("__torch_selected_expert_swiglu_ffn expects input, top-k scores/indices, gate/up/down weight paths, and transpose")
            out(
                self._selected_expert_swiglu_ffn(
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4],
                    args[5],
                    transpose=bool(args[6]),
                )
            )
            return True

        if primitive == "_torch_selected_expert_packed_gegelu_ffn":
            if len(args) < 10:
                raise ValueError("__torch_selected_expert_packed_gegelu_ffn expects input, top-k scores/indices, gate-up/down weight/bias paths, limit, optional alpha, bias, and transpose")
            alpha = args[8] if len(args) >= 11 else 1.702
            bias = args[9] if len(args) >= 11 else args[8]
            transpose = args[10] if len(args) >= 11 else args[9]
            out(
                self._selected_expert_packed_gegelu_ffn(
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4],
                    args[5],
                    args[6],
                    args[7],
                    alpha=alpha,
                    bias=bool(bias),
                    transpose=bool(transpose),
                )
            )
            return True

        if primitive == "_torch_selected_expert_clamped_packed_swiglu_ffn":
            if len(args) < 7:
                raise ValueError("__torch_selected_expert_clamped_packed_swiglu_ffn expects input, top-k scores/indices, gate-up/down weight paths, limit, and transpose")
            out(
                self._selected_expert_clamped_packed_swiglu_ffn(
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4],
                    args[5],
                    transpose=bool(args[6]),
                )
            )
            return True

        if primitive == "_torch_selected_expert_relu2_ffn":
            if len(args) < 6:
                raise ValueError("__torch_selected_expert_relu2_ffn expects input, top-k scores/indices, up/down weight paths, and transpose")
            out(
                self._selected_expert_relu2_ffn(
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4],
                    transpose=bool(args[5]),
                )
            )
            return True

        if primitive == "_torch_weighted_topk_sum":
            if len(args) < 2:
                raise ValueError("__torch_weighted_topk_sum expects expert values and top-k scores")
            out(torch.sum(args[0] * torch.unsqueeze(args[1].to(device=args[0].device, dtype=args[0].dtype), -1), dim=2, keepdim=False))
            return True

        if primitive == "_torch_topk_normalize":
            if len(args) < 2:
                raise ValueError("__torch_topk_normalize expects top-k weights and a dtype reference")
            normalized = args[0] / torch.sum(args[0], dim=-1, keepdim=True)
            out(normalized.to(dtype=args[1].dtype))
            return True

        if primitive == "expert_linear":
            if len(args) < 3:
                raise ValueError("expert_linear expects path, input, and expert indices")
            base = args[0]
            x = args[1]
            expert_idx = args[2]
            bias_flag = bool(args[4]) if len(args) > 4 and not self._is_null(args[4]) else False
            transpose = bool(args[5]) if len(args) > 5 and not self._is_null(args[5]) else False
            weight_leaf = args[6] if len(args) > 6 and not self._is_null(args[6]) else "@weight"
            bias_leaf = args[7] if len(args) > 7 and not self._is_null(args[7]) else "@bias"
            weight = self._required_param(self._compose_path(base, weight_leaf), field="expert_linear.weight")
            bias = None
            if bias_flag:
                raw_bias = self._optional_param(self._compose_path(base, bias_leaf))
                if raw_bias is not None:
                    bias = self._move_to(raw_bias, weight.device)
            out(_grouped_expert_linear_torch(x, weight, expert_idx, bias, transpose=transpose))
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

        if primitive == "conv1d":
            if len(args) != 8:
                raise ValueError("conv1d expects x, weight, bias, stride, padding_left, padding_right, dilation, groups")
            out(self._conv1d(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]))
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
            out(torch.reshape(args[0], tuple(args[1])))
            return True
        if primitive == "arange":
            if len(args) < 3:
                raise ValueError("arange expects reference, start, end")
            ref, start, end = args[:3]
            device = ref.device if torch.is_tensor(ref) else None
            if end is None:
                end = ref.shape[-2] if torch.is_tensor(ref) and ref.ndim >= 3 else ref.shape[-1]
            out(torch.arange(int(start), int(end), device=device, dtype=torch.long))
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
            out(x.expand(tuple(shape)))
            return True
        if primitive == "permute":
            x, dims = args[:2]
            out(torch.permute(x, tuple(dims)))
            return True
        if primitive == "transpose":
            x, dim0, dim1 = args[:3]
            out(torch.transpose(x, int(dim0), int(dim1)))
            return True
        if primitive == "unsqueeze":
            out(torch.unsqueeze(args[0], int(args[1])))
            return True
        if primitive == "matmul":
            left, right = self._align_pair(args[0], args[1], prefer="right")
            out(torch.matmul(left, right))
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
            limit = args[1] if len(args) > 1 and not self._is_null(args[1]) else None
            out(self._gegelu(x, limit))
            return True
        if primitive == "where":
            if torch.is_tensor(args[0]):
                out(self._where(args[0], args[1], args[2]))
            else:
                out(args[1] if args[0] else args[2])
            return True
        if primitive == "gather":
            dim = int(args[2]) if len(args) > 2 and not self._is_null(args[2]) else -1
            out(torch.gather(args[0], dim=dim, index=args[1]))
            return True
        if primitive == "scatter":
            dim = int(args[3]) if len(args) > 3 and not self._is_null(args[3]) else -1
            out(self._scatter(args[0], args[1], args[2], dim))
            return True
        if primitive == "assign_slice":
            if len(args) < 5:
                raise ValueError("_assign_slice expects x, src, dim, start, end")
            out(self._assign_slice(args[0], args[1], args[2], args[3], args[4]))
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
            left, right = self._align_pair(args[0], args[1], prefer="right")
            out(left <= right)
            return True
        if primitive == "eq":
            left, right = args[0], args[1]
            if left is None or right is None:
                out(left is right)
            else:
                left, right = self._align_pair(left, right, prefer="right")
                out(torch.eq(left, right) if torch.is_tensor(left) or torch.is_tensor(right) else left == right)
            return True
        if primitive == "and":
            left, right = self._align_pair(args[0], args[1], prefer="right")
            out(torch.logical_and(left, right))
            return True
        if primitive == "add":
            left, right = self._align_pair(args[0], args[1], prefer="right")
            out(left + right)
            return True
        if primitive == "mul":
            left, right = self._align_pair(args[0], args[1], prefer="right")
            out(left * right)
            return True
        if primitive == "pow":
            left, right = self._align_pair(args[0], args[1], prefer="right")
            out(torch.pow(left, right) if torch.is_tensor(left) else left ** right)
            return True
        if primitive == "div":
            left, right = self._align_pair(args[0], args[1], prefer="right")
            out(left / right)
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
            out(torch.empty(tuple(shape), device=ref.device, dtype=dtype or ref.dtype))
            return True
        if primitive == "zeros":
            ref, shape = args[:2]
            dtype = None if len(args) < 3 or self._is_null(args[2]) else self._dtype_from_name(args[2])
            out(torch.zeros(tuple(shape), device=ref.device, dtype=dtype or ref.dtype))
            return True
        if primitive == "full":
            ref, shape, value = args[:3]
            dtype = None if len(args) < 4 or self._is_null(args[3]) else self._dtype_from_name(args[3])
            out(torch.full(tuple(shape), value, device=ref.device, dtype=dtype or ref.dtype))
            return True
        if primitive == "zeros_like":
            out(torch.zeros_like(args[0]))
            return True
        if primitive == "tensor_like":
            dtype = None if len(args) < 3 or self._is_null(args[2]) else self._dtype_from_name(args[2])
            out(self._tensor_like(args[0], args[1], dtype=dtype))
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
                _effective_graph_value_type(value),
                local_env.get(value.name),
                local_symbols,
                overwrite=True,
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

        def assign(value: Any) -> None:
            self._assign_outputs(out_names, value, env, outputs=node.outputs, symbols=symbols)

        if op == "core.alias":
            values = [
                self._eval_graph_operand(operand, env=env, symbols=symbols)
                for operand in node.inputs
            ]
            assign(values[0] if len(values) == 1 else tuple(values))
            return
        if op == "core.tuple":
            assign(
                tuple(
                    self._eval_graph_operand(operand, env=env, symbols=symbols)
                    for operand in node.inputs
                ),
            )
            return
        if op == "core.list":
            assign(
                [
                    self._eval_graph_operand(operand, env=env, symbols=symbols)
                    for operand in node.inputs
                ],
            )
            return
        if op == "core.ascribe":
            if len(node.inputs) != 1:
                raise ValueError("core.ascribe expects one input")
            assign(self._eval_graph_operand(node.inputs[0], env=env, symbols=symbols))
            return
        if op == "core.select":
            cond, true_value, false_value = node.inputs
            selected = true_value if bool(
                self._eval_graph_operand(cond, env=env, symbols=symbols)
            ) else false_value
            assign(self._eval_graph_operand(selected, env=env, symbols=symbols))
            return
        if op == "core.repeat":
            callee_attr = node.attrs.get("callee")
            arg_count_attr = node.attrs.get("arg_count")
            carry_count_attr = node.attrs.get("carry_count")
            if (
                not isinstance(callee_attr, GraphLiteral)
                or not isinstance(callee_attr.value, str)
                or not isinstance(arg_count_attr, GraphLiteral)
                or type(arg_count_attr.value) is not int
                or not isinstance(carry_count_attr, GraphLiteral)
                or type(carry_count_attr.value) is not int
            ):
                raise ValueError("core.repeat has invalid metadata")
            callee = callee_attr.value
            arg_count = arg_count_attr.value
            carry_count = carry_count_attr.value
            current = [
                self._eval_graph_operand(node.inputs[3 + index], env=env, symbols=symbols)
                for index in range(carry_count)
            ]
            for i in range(
                int(self._eval_graph_operand(node.inputs[0], env=env, symbols=symbols)),
                int(self._eval_graph_operand(node.inputs[1], env=env, symbols=symbols)),
                int(self._eval_graph_operand(node.inputs[2], env=env, symbols=symbols)),
            ):
                args: list[Any] = []
                for index in range(arg_count):
                    role_attr = node.attrs.get(f"arg_{index}")
                    if not isinstance(role_attr, GraphLiteral) or not isinstance(role_attr.value, str):
                        raise ValueError(f"core.repeat arg_{index} metadata is invalid")
                    role = role_attr.value
                    if role == "iter":
                        args.append(i)
                    elif role.startswith("carry:"):
                        args.append(current[int(role.removeprefix("carry:"))])
                    elif role.startswith("input:"):
                        args.append(
                            self._eval_graph_operand(
                                node.inputs[int(role.removeprefix("input:"))],
                                env=env,
                                symbols=symbols,
                            )
                        )
                    else:
                        raise ValueError(f"invalid core.repeat arg role {role!r}")
                callee_module = self.modules_by_name[callee]
                call_env = {
                    value.name: arg for value, arg in zip(callee_module.inputs, args, strict=False)
                }
                current = list(self._execute_module(callee, call_env, symbols))
            assign(current[0] if len(current) == 1 else tuple(current))
            return
        if op.startswith("core.binary."):
            operator = op.removeprefix("core.binary.")
            left = self._eval_graph_operand(node.inputs[0], env=env, symbols=symbols)
            right = self._eval_graph_operand(node.inputs[1], env=env, symbols=symbols)
            assign(self._eval_binary(operator, left, right))
            return
        if op == "__torch_sdpa":
            if len(node.inputs) < 6:
                raise ValueError("__torch_sdpa expects q, k, v, additive_mask, scale, enable_gqa")
            q = self._eval_graph_operand(node.inputs[0], env=env, symbols=symbols)
            k = self._eval_graph_operand(node.inputs[1], env=env, symbols=symbols)
            v = self._eval_graph_operand(node.inputs[2], env=env, symbols=symbols)
            additive_mask = self._eval_graph_operand(node.inputs[3], env=env, symbols=symbols)
            scale = self._eval_graph_operand(node.inputs[4], env=env, symbols=symbols)
            enable_gqa = bool(self._eval_graph_operand(node.inputs[5], env=env, symbols=symbols))
            if torch.is_tensor(q):
                k = self._move_to(k, q.device)
                v = self._move_to(v, q.device)
                additive_mask = self._move_to(additive_mask, q.device)
            assign(
                F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=additive_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=None if scale is None else float(scale),
                    enable_gqa=enable_gqa,
                )
            )
            return
        if op == "__torch_rope_apply_factors":
            if len(node.inputs) < 4:
                raise ValueError("__torch_rope_apply_factors expects x, sin, cos, interleaved")
            x = self._eval_graph_operand(node.inputs[0], env=env, symbols=symbols)
            sin = self._eval_graph_operand(node.inputs[1], env=env, symbols=symbols)
            cos = self._eval_graph_operand(node.inputs[2], env=env, symbols=symbols)
            interleaved = bool(self._eval_graph_operand(node.inputs[3], env=env, symbols=symbols))
            assign(self._rope_apply_factors(x, sin, cos, interleaved))
            return
        if op == "__torch_rope_pair_apply_factors":
            if len(node.inputs) < 5:
                raise ValueError("__torch_rope_pair_apply_factors expects q, k, sin, cos, interleaved")
            q = self._eval_graph_operand(node.inputs[0], env=env, symbols=symbols)
            k = self._eval_graph_operand(node.inputs[1], env=env, symbols=symbols)
            sin = self._eval_graph_operand(node.inputs[2], env=env, symbols=symbols)
            cos = self._eval_graph_operand(node.inputs[3], env=env, symbols=symbols)
            interleaved = bool(self._eval_graph_operand(node.inputs[4], env=env, symbols=symbols))
            assign(
                (
                    self._rope_apply_factors(q, sin, cos, interleaved),
                    self._rope_apply_factors(k, sin, cos, interleaved),
                )
            )
            return
        if op in self.modules_by_name:
            if (
                op in self.cached_global_symbol_names
                and not node.inputs
                and not node.attrs
                and op in symbols
            ):
                assign(symbols[op])
                return
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
            assign(result[0] if len(result) == 1 else result)
            return

        if op == "_sqrt":
            value = self._eval_graph_operand(node.inputs[0], env=env, symbols=symbols)
            if isinstance(value, list | tuple) and value:
                value = value[-1]
            assign(torch.sqrt(torch.tensor(float(value))).item())
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
            assign(reshaped)
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
        *,
        outputs: tuple[GraphValue, ...] | None = None,
        symbols: dict[str, Any] | None = None,
    ) -> None:
        if len(names) == 1:
            env[names[0]] = value
        else:
            if not isinstance(value, (tuple, list)) or len(value) != len(names):
                raise ValueError(f"cannot assign {value!r} to outputs {names!r}")
            for name, item in zip(names, value, strict=True):
                env[name] = item
        if outputs is None or symbols is None:
            return
        for output in outputs:
            _bind_nested_shape_symbols_runtime(
                _effective_graph_value_type(output),
                env.get(output.name),
                symbols,
                overwrite=True,
            )

    @classmethod
    def _eval_binary(cls, op: str, left: Any, right: Any) -> Any:
        if op == "+":
            if left is None:
                return right
            if right is None:
                return left
            left, right = cls._align_pair(left, right, prefer="right")
            return left + right
        if op == "-":
            if right is None:
                return left
            left, right = cls._align_pair(left, right, prefer="right")
            return left - right
        if op == "*":
            left, right = cls._align_pair(left, right, prefer="right")
            return left * right
        if op == "/":
            left, right = cls._align_pair(left, right, prefer="right")
            return left / right
        if op == "%":
            left, right = cls._align_pair(left, right, prefer="right")
            return left % right
        if op == "==":
            if left is None or right is None:
                return left is right
            left, right = cls._align_pair(left, right, prefer="right")
            return left == right
        if op == "!=":
            if left is None or right is None:
                return left is not right
            left, right = cls._align_pair(left, right, prefer="right")
            return left != right
        if op == "<":
            left, right = cls._align_pair(left, right, prefer="right")
            return left < right
        if op == "<=":
            left, right = cls._align_pair(left, right, prefer="right")
            return left <= right
        if op == ">":
            left, right = cls._align_pair(left, right, prefer="right")
            return left > right
        if op == ">=":
            left, right = cls._align_pair(left, right, prefer="right")
            return left >= right
        if op == "and":
            if torch.is_tensor(left) or torch.is_tensor(right):
                left, right = cls._align_pair(left, right, prefer="right")
                return torch.logical_and(left, right)
            return bool(left) and bool(right)
        if op == "or":
            if torch.is_tensor(left) or torch.is_tensor(right):
                left, right = cls._align_pair(left, right, prefer="right")
                return torch.logical_or(left, right)
            return bool(left) or bool(right)
        raise NotImplementedError(f"unsupported codegen2-torch binary op {op!r}")


def _py_ident(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if not out or out[0].isdigit():
        out = "_" + out
    if out.startswith("__"):
        out = "_" + out.lstrip("_")
    return out


def _dim_ident(name: str) -> str:
    ident = _py_ident(name)
    if re.fullmatch(r"_v\d+", ident) or ident.startswith("__"):
        return f"_dim_{ident.lstrip('_')}"
    return ident


def _local_dim_ref(name: str, local: set[str]) -> str:
    return _py_ident(name) if name in local else _dim_ident(name)


def _value_ident(value: GraphValueRef) -> str:
    if isinstance(value.type_expr, TypeDim):
        return _dim_ident(value.name)
    return _py_ident(value.name)


class _DirectTorchEmitter:
    def __init__(
        self,
        *,
        program: GraphProgram,
        class_name: str,
        profile: bool = False,
        align_devices: bool = False,
    ) -> None:
        self.program = program
        self.class_name = class_name
        self.profile = bool(profile)
        self.align_devices = bool(align_devices)
        self.modules_by_name = {module.name: module for module in program.modules}
        self.method_names = {name: f"_def_{_py_ident(name)}" for name in self.modules_by_name}
        self.global_symbol_names = _runtime_symbol_module_names(program)
        self.cached_global_symbol_names = _runtime_symbol_module_names(
            program,
            total_pure_only=True,
        )
        self.state_key_filter_prefixes = _state_key_filter_prefixes(program)
        self.emitted_module_names = self._emitted_module_names()
        self.needs_expert_banks = _graph_uses_expert_linear(program)
        self.ownership = infer_graph_ownership(program, assume_module_inputs_owned=True)
        self.inplace_owned_primitive_ops = _owned_inplace_primitive_ops(
            program,
            self.ownership.inplace_assign_slice_node_ids,
        )
        self._inline_stack: set[str] = set()
        self._emitted_defs_stack: list[dict[str, Any]] = []
        self._emitted_aliases_stack: list[dict[str, GraphOperand]] = []

    def emit(self) -> str:
        lines: list[str] = [f"class {self.class_name}(nn.Module):"]
        self._emit_common(lines)
        lines.append("")
        self._emit_eval_symbols(lines)
        for module in self.program.modules:
            if module.name not in self.emitted_module_names:
                continue
            lines.append("")
            self._emit_module(lines, module)
        lines.append("")
        self._emit_forward(lines)
        lines.append("")
        self._emit_generate(lines)
        return "\n".join(lines)

    def _repeat_callee_is_inlineable(self, node: Any) -> bool:
        if node.op.name != "core.repeat":
            return False
        try:
            callee = self._repeat_attr_string(node, "callee")
            arg_count = self._repeat_attr_int(node, "arg_count")
            carry_count = self._repeat_attr_int(node, "carry_count")
        except ValueError:
            return False
        callee_module = self.modules_by_name.get(callee)
        return (
            callee_module is not None
            and len(callee_module.inputs) == arg_count
            and len(callee_module.outputs) == carry_count
        )

    def _current_emitted_defs(self) -> dict[str, Any]:
        if not self._emitted_defs_stack:
            return {}
        return self._emitted_defs_stack[-1]

    def _record_emitted_node_defs(self, node: Any) -> None:
        if not self._emitted_defs_stack:
            return
        defs = self._emitted_defs_stack[-1]
        for output in getattr(node, "outputs", ()):
            defs[output.name] = node

    def _record_emitted_alias(self, name: str, operand: GraphOperand) -> None:
        if not self._emitted_aliases_stack:
            return
        self._emitted_aliases_stack[-1][name] = operand

    def _resolve_emitted_alias_operand(self, operand: GraphOperand, *, depth: int = 8) -> GraphOperand:
        current = operand
        defs = self._current_emitted_defs()
        for _ in range(depth):
            if not isinstance(current, GraphValueRef):
                return current
            if self._emitted_aliases_stack and current.name in self._emitted_aliases_stack[-1]:
                current = self._emitted_aliases_stack[-1][current.name]
                continue
            node = defs.get(current.name)
            if node is None or node.op.name not in {"core.alias", "core.ascribe"} or len(node.inputs) != 1:
                return current
            current = node.inputs[0]
        return current

    def _operand_ref_name(self, operand: GraphOperand) -> str | None:
        return operand.name if isinstance(operand, GraphValueRef) else None

    def _operand_uses_value_name(self, operand: GraphOperand, name: str) -> bool:
        if isinstance(operand, GraphValueRef):
            return operand.name == name
        if isinstance(operand, GraphExpr):
            return any(self._operand_uses_value_name(item, name) for item in operand.inputs) or any(
                self._operand_uses_value_name(item, name) for item in operand.attrs.values()
            )
        if isinstance(operand, tuple):
            return any(self._operand_uses_value_name(item, name) for item in operand)
        return False

    def _collect_inline_value_use_counts(self, module: GraphModule) -> Counter[str]:
        counts: Counter[str] = Counter()
        for node in module.nodes:
            for operand in (*node.inputs, *node.attrs.values()):
                for name in self._value_ref_names(operand):
                    counts[name] += 1
        for output in module.outputs:
            for name in self._value_ref_names(output):
                counts[name] += 1
        return counts

    def _value_ref_names(self, operand: GraphOperand) -> tuple[str, ...]:
        names: list[str] = []

        def visit(item: GraphOperand) -> None:
            if isinstance(item, GraphValueRef):
                names.append(item.name)
                return
            if isinstance(item, GraphExpr):
                for child in item.inputs:
                    visit(child)
                for child in item.attrs.values():
                    visit(child)
                return
            if isinstance(item, tuple):
                for child in item:
                    visit(child)

        visit(operand)
        return tuple(names)

    def _inline_body_node_as_expr(
        self,
        node: Any,
        *,
        original_output_names: tuple[str, ...],
        use_counts: Counter[str],
        module_effects: Mapping[str, GraphEffect],
    ) -> GraphExpr | None:
        if len(node.outputs) != 1 or len(original_output_names) != 1:
            return None
        original_name = original_output_names[0]
        if use_counts[original_name] != 1:
            return None
        if graph_node_effect(node, module_effects=dict(module_effects)) != GraphEffect.TOTAL_PURE:
            return None
        primitive = _normalize_primitive_op(node.op.name)
        if not (
            node.op.name in {"core.alias", "core.ascribe"}
            or node.op.name.startswith("core.binary.")
            or primitive in {
                "exp",
                "log",
                "matmul",
                "mul",
                "add",
                "sub",
                "unsqueeze",
                "reshape",
                "activations_silu",
            }
        ):
            return None
        return GraphExpr(
            op=node.op,
            inputs=node.inputs,
            attrs=node.attrs,
            type_expr=node.outputs[0].type_expr,
            dims=node.outputs[0].dims,
        )

    def _collect_codegen_module_refs(
        self,
        operand: GraphOperand,
        refs: set[str],
    ) -> None:
        if isinstance(operand, GraphExpr):
            if operand.op.name in self.modules_by_name:
                refs.add(operand.op.name)
            for item in operand.inputs:
                self._collect_codegen_module_refs(item, refs)
            for item in operand.attrs.values():
                self._collect_codegen_module_refs(item, refs)

    def _collect_emitted_module_refs_from_operand(
        self,
        operand: GraphOperand,
        refs: set[str],
    ) -> None:
        if isinstance(operand, GraphExpr):
            if operand.op.name in self.modules_by_name:
                refs.add(operand.op.name)
            for item in operand.inputs:
                self._collect_emitted_module_refs_from_operand(item, refs)
            for item in operand.attrs.values():
                self._collect_emitted_module_refs_from_operand(item, refs)

    def _collect_emitted_module_refs_from_module_body(
        self,
        module: GraphModule,
        refs: set[str],
        *,
        visiting_inline: set[str] | None = None,
    ) -> None:
        if visiting_inline is None:
            visiting_inline = set()
        if module.name in visiting_inline:
            return
        visiting_inline.add(module.name)
        try:
            for node in module.nodes:
                self._collect_emitted_module_refs_from_node(
                    node,
                    refs,
                    module_name=module.name,
                    visiting_inline=visiting_inline,
                )
            for output in module.outputs:
                self._collect_emitted_module_refs_from_operand(output, refs)
        finally:
            visiting_inline.remove(module.name)

    def _collect_emitted_module_refs_from_node(
        self,
        node: Any,
        refs: set[str],
        *,
        module_name: str,
        visiting_inline: set[str],
    ) -> None:
        if self._repeat_callee_is_inlineable(node):
            for operand in (*node.inputs, *node.attrs.values()):
                self._collect_emitted_module_refs_from_operand(operand, refs)
            callee = self._repeat_attr_string(node, "callee")
            callee_module = self.modules_by_name.get(callee)
            if callee_module is not None:
                self._collect_emitted_module_refs_from_module_body(
                    callee_module,
                    refs,
                    visiting_inline=visiting_inline,
                )
            return
        if node.op.name == "core.select" and len(node.inputs) == 3:
            self._collect_emitted_module_refs_from_operand(node.inputs[0], refs)
            for branch in node.inputs[1:]:
                if isinstance(branch, GraphExpr) and self._branch_benefits_from_control_inline(
                    branch,
                    module_name=module_name,
                ):
                    callee_module = self.modules_by_name.get(branch.op.name)
                    if callee_module is not None:
                        for arg in branch.inputs:
                            self._collect_emitted_module_refs_from_operand(arg, refs)
                        for arg in branch.attrs.values():
                            self._collect_emitted_module_refs_from_operand(arg, refs)
                        self._collect_emitted_module_refs_from_module_body(
                            callee_module,
                            refs,
                            visiting_inline=visiting_inline,
                        )
                    continue
                self._collect_emitted_module_refs_from_operand(branch, refs)
            return
        if self._can_inline_direct_module_call(
            node.op.name,
            module_name=module_name,
            attrs=node.attrs,
            visiting_inline=visiting_inline,
        ):
            refs.add(node.op.name)
            for operand in (*node.inputs, *node.attrs.values()):
                self._collect_emitted_module_refs_from_operand(operand, refs)
            callee_module = self.modules_by_name.get(node.op.name)
            if callee_module is not None:
                self._collect_emitted_module_refs_from_module_body(
                    callee_module,
                    refs,
                    visiting_inline=visiting_inline,
                )
            return
        if node.op.name in self.modules_by_name:
            refs.add(node.op.name)
        for operand in (*node.inputs, *node.attrs.values()):
            self._collect_emitted_module_refs_from_operand(operand, refs)

    def _can_inline_direct_module_call(
        self,
        callee: str,
        *,
        module_name: str,
        attrs: Mapping[str, GraphOperand],
        visiting_inline: set[str] | None = None,
    ) -> bool:
        return (
            callee in self.modules_by_name
            and callee != module_name
            and callee not in self.global_symbol_names
            and not attrs
            and (visiting_inline is None or callee not in visiting_inline)
        )

    def _emitted_module_names(self) -> set[str]:
        emitted: set[str] = {self.program.main_module} | set(self.global_symbol_names)
        pending = list(emitted)
        while pending:
            name = pending.pop()
            module = self.modules_by_name.get(name)
            if module is None:
                continue
            refs: set[str] = set()
            self._collect_emitted_module_refs_from_module_body(module, refs)
            for ref in refs:
                if ref not in emitted:
                    emitted.add(ref)
                    pending.append(ref)
        return emitted

    @staticmethod
    def _add(lines: list[str], indent: int, text: str = "") -> None:
        lines.append(" " * indent + text)

    def _emit_common(self, lines: list[str]) -> None:
        add = self._add
        add(lines, 4, "def __init__(self, state_dict: dict[str, torch.Tensor], config: dict | None = None, param_devices=None):")
        add(lines, 8, "super().__init__()")
        add(lines, 8, "self.param_devices = self._normalize_param_devices(param_devices)")
        add(lines, 8, "self.state_dict_tensors = {}")
        add(lines, 8, "self.config = dict(({} if _MODEL_CONFIG is None else _MODEL_CONFIG) if config is None else config)")
        add(lines, 8, "self._symbols = {}")
        if self.profile:
            add(lines, 8, "self._profile_cuda = True")
            add(lines, 8, "self._profile_records = {}")
        add(lines, 8, "self.load_state_dict(state_dict)")
        add(lines, 4, "")
        if self.profile:
            add(lines, 4, "def enable_profile(self, enabled=True, *, cuda=True, reset=True):")
            add(lines, 8, "del enabled")
            add(lines, 8, "self._profile_cuda = bool(cuda)")
            add(lines, 8, "if reset:")
            add(lines, 12, "self._profile_records = {}")
            add(lines, 8, "return self")

        if self.profile:
            add(lines, 4, "")
            add(lines, 4, "def _profile_call(self, name, fn, *args, **kwargs):")
            add(lines, 8, "use_cuda = bool(self._profile_cuda and torch.cuda.is_available())")
            add(lines, 8, "if use_cuda:")
            add(lines, 12, "torch.cuda.synchronize()")
            add(lines, 8, "start = time.perf_counter()")
            add(lines, 8, "try:")
            add(lines, 12, "return fn(*args, **kwargs)")
            add(lines, 8, "finally:")
            add(lines, 12, "if use_cuda:")
            add(lines, 16, "torch.cuda.synchronize()")
            add(lines, 12, "elapsed = time.perf_counter() - start")
            add(lines, 12, "count, total = self._profile_records.get(name, (0, 0.0))")
            add(lines, 12, "self._profile_records[name] = (count + 1, total + elapsed)")
            add(lines, 4, "")
            add(lines, 4, "def profile_summary(self, top_n=40):")
            add(lines, 8, "rows = []")
            add(lines, 8, "for name, (count, total) in self._profile_records.items():")
            add(lines, 12, "rows.append({'name': name, 'count': count, 'seconds': total, 'avg_seconds': total / max(1, count)})")
            add(lines, 8, "rows.sort(key=lambda row: row['seconds'], reverse=True)")
            add(lines, 8, "return rows[: int(top_n)]")
            add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def from_state_dict(cls, state_dict, *, graph=None, model_config=None, param_devices=None):")
        add(lines, 8, "return cls(state_dict, config=_MODEL_CONFIG if model_config is None else model_config, param_devices=param_devices)")
        add(lines, 4, "")
        add(lines, 4, "def load_state_dict(self, state_dict, strict=True):")
        add(lines, 8, "del strict")
        add(lines, 8, "state_dict = self._filter_state_dict(state_dict)")
        add(lines, 8, "state = dict(state_dict)")
        add(lines, 8, "_materialize_packed_parameters(state, self._PACKED_PARAMETER_SPECS)")
        add(lines, 8, "self.state_dict_tensors = self._place_state_dict(state, self.param_devices)")
        add(lines, 8, "self.setup()")
        add(lines, 8, "self._symbols = self._eval_symbols()")
        add(lines, 8, "return self")
        add(lines, 4, "")
        add(lines, 4, "def setup(self):")
        add(lines, 8, "pass")
        add(lines, 8, "return None")
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
        add(lines, 4, f"_STATE_KEY_FILTER_PREFIXES = {self.state_key_filter_prefixes!r}")
        add(lines, 4, f"_PACKED_PARAMETER_SPECS = {tuple(_packed_parameter_spec_payload(item) for item in self.program.packed_parameters)!r}")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _keep_state_key(cls, key):")
        add(lines, 8, "prefixes = cls._STATE_KEY_FILTER_PREFIXES")
        add(lines, 8, "if not prefixes:")
        add(lines, 12, "return True")
        add(lines, 8, "key = str(key)")
        add(lines, 8, "for prefix in prefixes:")
        add(lines, 12, "if key == prefix or key.startswith(prefix + '.'):")
        add(lines, 16, "return True")
        add(lines, 8, "return False")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _filter_state_dict(cls, state_dict):")
        add(lines, 8, "if not cls._STATE_KEY_FILTER_PREFIXES:")
        add(lines, 12, "return state_dict")
        add(lines, 8, "return {key: value for key, value in state_dict.items() if cls._keep_state_key(key)}")
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
        if self.align_devices:
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
        add(lines, 4, "def _rope_apply_factors(cls, x, sin, cos, interleaved=False):")
        add(lines, 8, "if interleaved:")
        add(lines, 12, "raise NotImplementedError('__torch_rope_apply_factors only supports non-interleaved RoPE')")
        add(lines, 8, "if torch.is_tensor(x):")
        add(lines, 12, "sin = cls._move_to(sin, x.device)")
        add(lines, 12, "cos = cls._move_to(cos, x.device)")
        add(lines, 8, "half = x.shape[-1] // 2")
        add(lines, 8, "rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)")
        add(lines, 8, "return (x * cos) + (rotated * sin)")
        add(lines, 4, "")
        add(lines, 4, "_compose_path = staticmethod(_common_compose_path)")
        add(lines, 4, "_render_path = staticmethod(_common_render_path)")
        add(lines, 4, "_require_value = staticmethod(_common_require_value)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _path_template_part(value):")
        add(lines, 8, "if isinstance(value, str) and value.startswith('@@'):")
        add(lines, 12, "return value[2:].strip('.')")
        add(lines, 8, "if isinstance(value, str) and value.startswith('@'):")
        add(lines, 12, "return value[1:].strip('.')")
        add(lines, 8, "return value")
        add(lines, 4, "")
        add(lines, 4, "def _conv1d(self, x, weight, bias, stride, padding_left, padding_right, dilation, groups):")
        add(lines, 8, "if x.device != weight.device:")
        add(lines, 12, "x = x.to(device=weight.device)")
        add(lines, 8, "if x.is_floating_point() and weight.is_floating_point() and x.dtype != weight.dtype:")
        add(lines, 12, "weight = weight.to(dtype=x.dtype)")
        add(lines, 8, "if bias is not None:")
        add(lines, 12, "if bias.device != weight.device:")
        add(lines, 16, "bias = bias.to(device=weight.device)")
        add(lines, 12, "if x.is_floating_point() and bias.is_floating_point() and x.dtype != bias.dtype:")
        add(lines, 16, "bias = bias.to(dtype=x.dtype)")
        add(lines, 8, "left = int(padding_left)")
        add(lines, 8, "right = int(padding_right)")
        add(lines, 8, "if left or right:")
        add(lines, 12, "x = F.pad(x, (left, right))")
        add(lines, 8, "return F.conv1d(x, weight, bias=bias, stride=int(stride), dilation=int(dilation), groups=int(groups))")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _assign_slice(x, src, dim, start, end):")
        add(lines, 8, "out = x.clone()")
        add(lines, 8, "dim = int(dim)")
        add(lines, 8, "if dim < 0:")
        add(lines, 12, "dim += out.dim()")
        add(lines, 8, "start = int(start)")
        add(lines, 8, "end = int(end)")
        add(lines, 8, "sl = [slice(None)] * out.dim()")
        add(lines, 8, "sl[dim] = slice(start, end)")
        add(lines, 8, "if src.device != out.device:")
        add(lines, 12, "src = src.to(device=out.device)")
        add(lines, 8, "if out.is_floating_point() and src.is_floating_point() and out.dtype != src.dtype:")
        add(lines, 12, "src = src.to(dtype=out.dtype)")
        add(lines, 8, "out[tuple(sl)] = src")
        add(lines, 8, "return out")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _assign_slice_inplace(x, src, dim, start, end):")
        add(lines, 8, "dim = int(dim)")
        add(lines, 8, "if dim < 0:")
        add(lines, 12, "dim += x.dim()")
        add(lines, 8, "start = int(start)")
        add(lines, 8, "end = int(end)")
        add(lines, 8, "sl = [slice(None)] * x.dim()")
        add(lines, 8, "sl[dim] = slice(start, end)")
        add(lines, 8, "if src.device != x.device:")
        add(lines, 12, "src = src.to(device=x.device)")
        add(lines, 8, "if x.is_floating_point() and src.is_floating_point() and x.dtype != src.dtype:")
        add(lines, 12, "src = src.to(dtype=x.dtype)")
        add(lines, 8, "x[tuple(sl)] = src")
        add(lines, 8, "return x")
        add(lines, 4, "")
        add(lines, 4, "def _scatter(self, x, index, src, dim):")
        add(lines, 8, "dim = int(dim)")
        add(lines, 8, "if torch.is_tensor(src):")
        add(lines, 12, "index = self._move_to(index, x.device)")
        add(lines, 12, "src = self._move_to(src, x.device)")
        add(lines, 12, "return torch.scatter(x, dim=dim, index=index, src=src)")
        add(lines, 8, "index = self._move_to(index, x.device)")
        add(lines, 8, "return torch.scatter(x, dim=dim, index=index, value=src)")
        add(lines, 4, "")
        add(lines, 4, "def _param(self, path):")
        add(lines, 8, "value, _ = self._linear_param(path, None, field='param')")
        add(lines, 8, "return value")
        add(lines, 4, "")
        add(lines, 4, "def _optional_param(self, path):")
        add(lines, 8, "value, _ = self._linear_param(path, None, optional=True, field='optional_param')")
        add(lines, 8, "return value")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _collapse_one_numeric_segment(key):")
        add(lines, 8, "parts = str(key).split('.')")
        add(lines, 8, "for index, part in enumerate(parts):")
        add(lines, 12, "if part.isdigit():")
        add(lines, 16, "return '.'.join(parts[:index] + parts[index + 1:]), int(part), index")
        add(lines, 8, "return None")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _collapsed_numeric_segments(key):")
        add(lines, 8, "parts = str(key).split('.')")
        add(lines, 8, "return [('.'.join(parts[:index] + parts[index + 1:]), int(part), index) for index, part in enumerate(parts) if part.isdigit()]")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _keys_for_collapsed_bank(cls, state, bank_key):")
        add(lines, 8, "items = {}")
        add(lines, 8, "numeric_index = None")
        add(lines, 8, "for key in state:")
        add(lines, 12, "for collapsed_key, expert, index in cls._collapsed_numeric_segments(str(key)):")
        add(lines, 16, "if collapsed_key != bank_key:")
        add(lines, 20, "continue")
        add(lines, 16, "if numeric_index is None:")
        add(lines, 20, "numeric_index = index")
        add(lines, 16, "elif numeric_index != index:")
        add(lines, 20, "continue")
        add(lines, 16, "items[expert] = str(key)")
        add(lines, 16, "break")
        add(lines, 8, "if not items:")
        add(lines, 12, "return []")
        add(lines, 8, "ordered = [items[i] for i in range(len(items)) if i in items]")
        add(lines, 8, "return ordered if len(ordered) == len(items) else []")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _fused_gate_up_source_bank_keys(bank_key):")
        add(lines, 8, "parts = str(bank_key).split('.')")
        add(lines, 8, "for index, part in enumerate(parts):")
        add(lines, 12, "if 'gate_up' not in part:")
        add(lines, 16, "continue")
        add(lines, 12, "gate_parts = list(parts)")
        add(lines, 12, "up_parts = list(parts)")
        add(lines, 12, "gate_parts[index] = part.replace('gate_up', 'gate', 1)")
        add(lines, 12, "up_parts[index] = part.replace('gate_up', 'up', 1)")
        add(lines, 12, "return '.'.join(gate_parts), '.'.join(up_parts)")
        add(lines, 8, "return None")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _materialize_expert_bank_for_path(cls, state, bank_key):")
        add(lines, 8, "existing = state.get(bank_key)")
        add(lines, 8, "if torch.is_tensor(existing):")
        add(lines, 12, "return existing")
        add(lines, 8, "ordered_keys = cls._keys_for_collapsed_bank(state, bank_key)")
        add(lines, 8, "if ordered_keys:")
        add(lines, 12, "first = state[ordered_keys[0]]")
        add(lines, 12, "first_shape = tuple(first.shape)")
        add(lines, 12, "if all(torch.is_tensor(state[key]) and tuple(state[key].shape) == first_shape for key in ordered_keys):")
        add(lines, 16, "return _materialize_joined_parameter(state, bank_key, ordered_keys, dim=0, mode='stack', remove_inputs=True)")
        add(lines, 8, "fused_sources = cls._fused_gate_up_source_bank_keys(bank_key)")
        add(lines, 8, "if fused_sources is None:")
        add(lines, 12, "return None")
        add(lines, 8, "gate_key, up_key = fused_sources")
        add(lines, 8, "gate = cls._materialize_expert_bank_for_path(state, gate_key)")
        add(lines, 8, "up = cls._materialize_expert_bank_for_path(state, up_key)")
        add(lines, 8, "if not torch.is_tensor(gate) or not torch.is_tensor(up):")
        add(lines, 12, "return None")
        add(lines, 8, "if gate.shape[:-2] != up.shape[:-2] or gate.shape[-1:] != up.shape[-1:]:")
        add(lines, 12, "return None")
        add(lines, 8, "concat_dim = -2 if gate.ndim >= 2 else -1")
        add(lines, 8, "return _materialize_joined_parameter(state, bank_key, (gate_key, up_key), dim=concat_dim, mode='cat', remove_inputs=True)")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _expert_bank_lookup(cls, state, path):")
        add(lines, 8, "for bank_key, expert, _ in cls._collapsed_numeric_segments(path):")
        add(lines, 12, "bank = state.get(bank_key)")
        add(lines, 12, "if torch.is_tensor(bank):")
        add(lines, 16, "return bank_key, expert")
        add(lines, 8, "return None")
        add(lines, 4, "")
        add(lines, 4, "def _linear_param(self, path, expert, *, optional=False, field='linear.weight'):")
        add(lines, 8, "resolved = path[2:] if isinstance(path, str) and path.startswith('@@') else path")
        add(lines, 8, "value = self.state_dict_tensors.get(resolved)")
        add(lines, 8, "if torch.is_tensor(value):")
        add(lines, 12, "return value, expert")
        add(lines, 8, "self._materialize_expert_bank_for_path(self.state_dict_tensors, resolved)")
        add(lines, 8, "value = self.state_dict_tensors.get(resolved)")
        add(lines, 8, "if torch.is_tensor(value):")
        add(lines, 12, "return value, expert")
        add(lines, 8, "bank = self._expert_bank_lookup(self.state_dict_tensors, resolved)")
        add(lines, 8, "if bank is not None:")
        add(lines, 12, "bank_key, path_expert = bank")
        add(lines, 12, "bank_value = self.state_dict_tensors.get(bank_key)")
        add(lines, 12, "if torch.is_tensor(bank_value):")
        add(lines, 16, "if expert is None or int(expert) == path_expert:")
        add(lines, 20, "return bank_value[path_expert], None")
        add(lines, 16, "return bank_value, expert")
        add(lines, 8, "if optional:")
        add(lines, 12, "return None, expert")
        add(lines, 8, "return _common_required_state_value(self.state_dict_tensors, resolved), expert")
        add(lines, 4, "")
        add(lines, 4, "def _linear(self, base, x, bias=False, transpose=False, expert=None, weight_leaf='weight', bias_leaf='bias'):")
        add(lines, 8, "weight, expert = self._linear_param(self._compose_path(base, weight_leaf), expert)")
        add(lines, 8, "if expert is not None:")
        add(lines, 12, "weight = weight[int(expert)]")
        add(lines, 8, "x = self._move_to(x, weight.device)")
        add(lines, 8, "bias_value, bias_expert = self._linear_param(self._compose_path(base, bias_leaf), expert, optional=True, field='linear.bias') if bias else (None, expert)")
        add(lines, 8, "if bias_value is not None and bias_expert is not None and bias_value.ndim >= 2:")
        add(lines, 12, "bias_value = bias_value[int(bias_expert)]")
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
        add(lines, 4, "def _fused_linear_pair_weight_path(gate_weight_path, up_weight_path):")
        add(lines, 8, "return '__fused_gate_up__.' + str(gate_weight_path).lstrip('@') + '||' + str(up_weight_path).lstrip('@')")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _fused_linear_pair_bias_path(gate_bias_path, up_bias_path):")
        add(lines, 8, "return '__fused_gate_up_bias__.' + str(gate_bias_path).lstrip('@') + '||' + str(up_bias_path).lstrip('@')")
        add(lines, 4, "")
        add(lines, 4, "def _gate_up_linear_pair(self, x, gate_weight_path, up_weight_path, gate_bias_path='bias', up_bias_path='bias', bias=False, transpose=False):")
        add(lines, 8, "fused_weight_path = self._fused_linear_pair_weight_path(gate_weight_path, up_weight_path)")
        add(lines, 8, "fused_bias_path = self._fused_linear_pair_bias_path(gate_bias_path, up_bias_path)")
        add(lines, 8, "fused_weight = self.state_dict_tensors.get(fused_weight_path)")
        add(lines, 8, "fused_bias = self.state_dict_tensors.get(fused_bias_path)")
        add(lines, 8, "if not torch.is_tensor(fused_weight):")
        add(lines, 12, "gate_key = str(gate_weight_path).lstrip('@')")
        add(lines, 12, "up_key = str(up_weight_path).lstrip('@')")
        add(lines, 12, "gate_weight = self.state_dict_tensors.pop(gate_key)")
        add(lines, 12, "up_weight = self.state_dict_tensors.pop(up_key)")
        add(lines, 12, "concat_dim = -1 if transpose else -2")
        add(lines, 12, "fused_weight = torch.cat([gate_weight, up_weight], dim=concat_dim)")
        add(lines, 12, "self.state_dict_tensors[fused_weight_path] = fused_weight")
        add(lines, 12, "if bias:")
        add(lines, 16, "gate_bias = self.state_dict_tensors.pop(str(gate_bias_path).lstrip('@'), None)")
        add(lines, 16, "up_bias = self.state_dict_tensors.pop(str(up_bias_path).lstrip('@'), None)")
        add(lines, 16, "if torch.is_tensor(gate_bias) and torch.is_tensor(up_bias):")
        add(lines, 20, "fused_bias = torch.cat([gate_bias, up_bias], dim=-1)")
        add(lines, 20, "self.state_dict_tensors[fused_bias_path] = fused_bias")
        add(lines, 8, "x = self._move_to(x, fused_weight.device)")
        add(lines, 8, "weight_run = fused_weight.to(dtype=x.dtype) if x.is_floating_point() and fused_weight.is_floating_point() and x.dtype != fused_weight.dtype else fused_weight")
        add(lines, 8, "bias_run = fused_bias.to(dtype=x.dtype) if fused_bias is not None and x.is_floating_point() and fused_bias.is_floating_point() and x.dtype != fused_bias.dtype else fused_bias")
        add(lines, 8, "combined = torch.matmul(x, weight_run) + (bias_run if bias_run is not None else 0) if transpose else F.linear(x, weight_run, bias_run)")
        add(lines, 8, "return torch.chunk(combined, 2, dim=-1)")
        add(lines, 4, "")
        add(lines, 4, "def _swiglu_ffn(self, x, gate_weight_path, up_weight_path, down_weight_path, gate_bias_path='bias', up_bias_path='bias', down_bias_path='bias'):")
        add(lines, 8, "gate, up = self._gate_up_linear_pair(x, gate_weight_path, up_weight_path, gate_bias_path=gate_bias_path, up_bias_path=up_bias_path, bias=False, transpose=False)")
        add(lines, 8, "hidden = F.silu(gate) * up")
        add(lines, 8, "down_weight = self._param(down_weight_path)")
        add(lines, 8, "hidden = self._move_to(hidden, down_weight.device)")
        add(lines, 8, "return F.linear(hidden, down_weight, None)")
        add(lines, 4, "")
        add(lines, 4, "def _expert_linear_weight(self, x, expert_idx, weight_path, bias_value=None, transpose=False):")
        add(lines, 8, "weight = self._param(weight_path)")
        add(lines, 8, "x = self._move_to(x, weight.device)")
        add(lines, 8, "expert_idx = self._move_to(expert_idx, weight.device).long()")
        add(lines, 8, "if bias_value is not None:")
        add(lines, 12, "bias_value = self._move_to(bias_value, weight.device)")
        add(lines, 8, "out_dim = int(weight.shape[-1] if transpose else weight.shape[-2])")
        add(lines, 8, "out = x.new_empty((*x.shape[:-1], out_dim))")
        add(lines, 8, "if x.numel() == 0:")
        add(lines, 12, "return out")
        add(lines, 8, "flat_x = x.reshape(-1, x.shape[-1])")
        add(lines, 8, "flat_idx = expert_idx.reshape(-1)")
        add(lines, 8, "if flat_idx.numel() != flat_x.shape[0]:")
        add(lines, 12, "raise ValueError(f'expert_idx shape {tuple(expert_idx.shape)} is incompatible with input shape {tuple(x.shape)}')")
        add(lines, 8, "grouped_weight = weight if transpose else weight.transpose(-2, -1)")
        add(lines, 8, "expert_ids_g, perm = torch.sort(flat_idx)")
        add(lines, 8, "x_g = flat_x.index_select(0, perm)")
        add(lines, 8, "x_run = x_g.to(dtype=grouped_weight.dtype) if x_g.is_floating_point() and grouped_weight.is_floating_point() and x_g.dtype != grouped_weight.dtype else x_g")
        add(lines, 8, "histc_input = expert_ids_g.float() if weight.device.type in ('cpu', 'mps') else expert_ids_g.int()")
        add(lines, 8, "tokens_per_expert = torch.histc(histc_input, bins=weight.shape[0], min=0, max=weight.shape[0] - 1)")
        add(lines, 8, "offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)")
        add(lines, 8, "if hasattr(torch.nn.functional, 'grouped_mm') and weight.device.type == 'cuda':")
        add(lines, 12, "y_g = torch.nn.functional.grouped_mm(x_run, grouped_weight, offs=offsets)")
        add(lines, 8, "elif hasattr(torch, '_grouped_mm') and weight.device.type == 'cuda':")
        add(lines, 12, "y_g = torch._grouped_mm(x_run, grouped_weight, offs=offsets)")
        add(lines, 8, "else:")
        add(lines, 12, "y_g = x_run.new_empty((x_run.shape[0], out_dim))")
        add(lines, 12, "start = 0")
        add(lines, 12, "for expert, end in enumerate(offsets.tolist()):")
        add(lines, 16, "if start != end:")
        add(lines, 20, "torch.mm(x_run[start:end], grouped_weight[expert], out=y_g[start:end])")
        add(lines, 16, "start = end")
        add(lines, 8, "y_g = y_g.to(dtype=x.dtype) if x.is_floating_point() and y_g.is_floating_point() and y_g.dtype != x.dtype else y_g")
        add(lines, 8, "if bias_value is not None:")
        add(lines, 12, "bias_g = bias_value.index_select(0, expert_ids_g)")
        add(lines, 12, "bias_g = bias_g.to(dtype=x.dtype) if x.is_floating_point() and bias_g.is_floating_point() and bias_g.dtype != x.dtype else bias_g")
        add(lines, 12, "y_g = y_g + bias_g")
        add(lines, 8, "inv_perm = torch.empty_like(perm)")
        add(lines, 8, "inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)")
        add(lines, 8, "return y_g.index_select(0, inv_perm).reshape_as(out)")
        add(lines, 4, "")
        add(lines, 4, "def _expert_swiglu_ffn(self, x, expert_idx, gate_weight_path, up_weight_path, down_weight_path):")
        add(lines, 8, "gate = self._expert_linear_weight(x, expert_idx, gate_weight_path)")
        add(lines, 8, "up = self._expert_linear_weight(x, expert_idx, up_weight_path)")
        add(lines, 8, "hidden = F.silu(gate) * up")
        add(lines, 8, "return self._expert_linear_weight(hidden, expert_idx, down_weight_path)")
        add(lines, 4, "")
        add(lines, 4, "def _expert_packed_swiglu_ffn(self, x, expert_idx, gate_up_weight_path, down_weight_path, transpose=False):")
        add(lines, 8, "gate_up = self._expert_linear_weight(x, expert_idx, gate_up_weight_path, transpose=transpose)")
        add(lines, 8, "gate, up = torch.chunk(gate_up, 2, dim=-1)")
        add(lines, 8, "hidden = F.silu(gate) * up")
        add(lines, 8, "return self._expert_linear_weight(hidden, expert_idx, down_weight_path, transpose=transpose)")
        add(lines, 4, "")
        add(lines, 4, "def _selected_expert_packed_swiglu_ffn(self, x, topk_scores, topk_indices, gate_up_weight_path, down_weight_path, transpose=False):")
        add(lines, 8, "topk_indices = topk_indices.long()")
        add(lines, 8, "expanded = torch.unsqueeze(x, 2).expand((*topk_indices.shape, x.shape[-1]))")
        add(lines, 8, "values = self._expert_packed_swiglu_ffn(expanded, topk_indices, gate_up_weight_path, down_weight_path, transpose=transpose)")
        add(lines, 8, "weights = torch.unsqueeze(topk_scores.to(device=values.device, dtype=values.dtype), -1)")
        add(lines, 8, "return torch.sum(values * weights, dim=2, keepdim=False)")
        add(lines, 4, "")
        add(lines, 4, "def _selected_expert_swiglu_ffn(self, x, topk_scores, topk_indices, gate_weight_path, up_weight_path, down_weight_path, transpose=False):")
        add(lines, 8, "topk_indices = topk_indices.long()")
        add(lines, 8, "expanded = torch.unsqueeze(x, 2).expand((*topk_indices.shape, x.shape[-1]))")
        add(lines, 8, "gate = self._expert_linear_weight(expanded, topk_indices, gate_weight_path, transpose=transpose)")
        add(lines, 8, "up = self._expert_linear_weight(expanded, topk_indices, up_weight_path, transpose=transpose)")
        add(lines, 8, "hidden = F.silu(gate) * up")
        add(lines, 8, "values = self._expert_linear_weight(hidden, topk_indices, down_weight_path, transpose=transpose)")
        add(lines, 8, "weights = torch.unsqueeze(topk_scores.to(device=values.device, dtype=values.dtype), -1)")
        add(lines, 8, "return torch.sum(values * weights, dim=2, keepdim=False)")
        add(lines, 4, "")
        add(lines, 4, "def _selected_expert_packed_gegelu_ffn(self, x, topk_scores, topk_indices, gate_up_weight_path, gate_up_bias_path, down_weight_path, down_bias_path, limit, alpha=1.702, bias=False, transpose=False):")
        add(lines, 8, "topk_indices = topk_indices.long()")
        add(lines, 8, "expanded = torch.unsqueeze(x, 2).expand((*topk_indices.shape, x.shape[-1]))")
        add(lines, 8, "gate_up_bias = self._param(gate_up_bias_path) if bias else None")
        add(lines, 8, "gate_up = self._expert_linear_weight(expanded, topk_indices, gate_up_weight_path, bias_value=gate_up_bias, transpose=transpose)")
        add(lines, 8, "hidden = self._gegelu(gate_up, limit, alpha)")
        add(lines, 8, "down_bias = self._param(down_bias_path) if bias else None")
        add(lines, 8, "values = self._expert_linear_weight(hidden, topk_indices, down_weight_path, bias_value=down_bias, transpose=transpose)")
        add(lines, 8, "weights = torch.unsqueeze(topk_scores.to(device=values.device, dtype=values.dtype), -1)")
        add(lines, 8, "return torch.sum(values * weights, dim=2, keepdim=False)")
        add(lines, 4, "")
        add(lines, 4, "def _selected_expert_clamped_packed_swiglu_ffn(self, x, topk_scores, topk_indices, gate_up_weight_path, down_weight_path, limit, transpose=False):")
        add(lines, 8, "topk_indices = topk_indices.long()")
        add(lines, 8, "expanded = torch.unsqueeze(x, 2).expand((*topk_indices.shape, x.shape[-1]))")
        add(lines, 8, "gate_up = self._expert_linear_weight(expanded, topk_indices, gate_up_weight_path, transpose=transpose)")
        add(lines, 8, "gate, up = torch.chunk(gate_up, 2, dim=-1)")
        add(lines, 8, "limit = float(limit)")
        add(lines, 8, "gate = torch.where(torch.isinf(gate), gate, gate.clamp(max=limit))")
        add(lines, 8, "up = torch.where(torch.isinf(up), up, up.clamp(min=-limit, max=limit))")
        add(lines, 8, "hidden = F.silu(gate) * up")
        add(lines, 8, "values = self._expert_linear_weight(hidden, topk_indices, down_weight_path, transpose=transpose)")
        add(lines, 8, "weights = torch.unsqueeze(topk_scores.to(device=values.device, dtype=values.dtype), -1)")
        add(lines, 8, "return torch.sum(values * weights, dim=2, keepdim=False)")
        add(lines, 4, "")
        add(lines, 4, "def _selected_expert_relu2_ffn(self, x, topk_scores, topk_indices, up_weight_path, down_weight_path, transpose=False):")
        add(lines, 8, "topk_indices = topk_indices.long()")
        add(lines, 8, "expanded = torch.unsqueeze(x, 2).expand((*topk_indices.shape, x.shape[-1]))")
        add(lines, 8, "up = self._expert_linear_weight(expanded, topk_indices, up_weight_path, transpose=transpose)")
        add(lines, 8, "hidden = F.relu(up) * F.relu(up)")
        add(lines, 8, "values = self._expert_linear_weight(hidden, topk_indices, down_weight_path, transpose=transpose)")
        add(lines, 8, "weights = torch.unsqueeze(topk_scores.to(device=values.device, dtype=values.dtype), -1)")
        add(lines, 8, "return torch.sum(values * weights, dim=2, keepdim=False)")
        add(lines, 4, "")
        add(lines, 4, "def _expert_linear(self, base, x, expert_idx, bias=False, transpose=False, weight_leaf='weight', bias_leaf='bias'):")
        add(lines, 8, "weight = self._param(self._compose_path(base, weight_leaf))")
        add(lines, 8, "x = self._move_to(x, weight.device)")
        add(lines, 8, "expert_idx = self._move_to(expert_idx, weight.device).long()")
        add(lines, 8, "bias_value = self._optional_param(self._compose_path(base, bias_leaf)) if bias else None")
        add(lines, 8, "if bias_value is not None:")
        add(lines, 12, "bias_value = self._move_to(bias_value, weight.device)")
        add(lines, 8, "out_dim = int(weight.shape[-1] if transpose else weight.shape[-2])")
        add(lines, 8, "out = x.new_empty((*x.shape[:-1], out_dim))")
        add(lines, 8, "if x.numel() == 0:")
        add(lines, 12, "return out")
        add(lines, 8, "flat_x = x.reshape(-1, x.shape[-1])")
        add(lines, 8, "flat_idx = expert_idx.reshape(-1)")
        add(lines, 8, "if flat_idx.numel() != flat_x.shape[0]:")
        add(lines, 12, "raise ValueError(f'expert_idx shape {tuple(expert_idx.shape)} is incompatible with input shape {tuple(x.shape)}')")
        add(lines, 8, "grouped_weight = weight if transpose else weight.transpose(-2, -1)")
        add(lines, 8, "expert_ids_g, perm = torch.sort(flat_idx)")
        add(lines, 8, "x_g = flat_x.index_select(0, perm)")
        add(lines, 8, "x_run = x_g.to(dtype=grouped_weight.dtype) if x_g.is_floating_point() and grouped_weight.is_floating_point() and x_g.dtype != grouped_weight.dtype else x_g")
        add(lines, 8, "histc_input = expert_ids_g.float() if weight.device.type in ('cpu', 'mps') else expert_ids_g.int()")
        add(lines, 8, "tokens_per_expert = torch.histc(histc_input, bins=weight.shape[0], min=0, max=weight.shape[0] - 1)")
        add(lines, 8, "offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)")
        add(lines, 8, "if hasattr(torch.nn.functional, 'grouped_mm') and weight.device.type == 'cuda':")
        add(lines, 12, "y_g = torch.nn.functional.grouped_mm(x_run, grouped_weight, offs=offsets)")
        add(lines, 8, "elif hasattr(torch, '_grouped_mm') and weight.device.type == 'cuda':")
        add(lines, 12, "y_g = torch._grouped_mm(x_run, grouped_weight, offs=offsets)")
        add(lines, 8, "else:")
        add(lines, 12, "y_g = x_run.new_empty((x_run.shape[0], out_dim))")
        add(lines, 12, "start = 0")
        add(lines, 12, "for expert, end in enumerate(offsets.tolist()):")
        add(lines, 16, "if start != end:")
        add(lines, 20, "torch.mm(x_run[start:end], grouped_weight[expert], out=y_g[start:end])")
        add(lines, 16, "start = end")
        add(lines, 8, "y_g = y_g.to(dtype=x.dtype) if x.is_floating_point() and y_g.is_floating_point() and y_g.dtype != x.dtype else y_g")
        add(lines, 8, "if bias_value is not None:")
        add(lines, 12, "bias_g = bias_value.index_select(0, expert_ids_g)")
        add(lines, 12, "bias_g = bias_g.to(dtype=x.dtype) if x.is_floating_point() and bias_g.is_floating_point() and bias_g.dtype != x.dtype else bias_g")
        add(lines, 12, "y_g = y_g + bias_g")
        add(lines, 8, "inv_perm = torch.empty_like(perm)")
        add(lines, 8, "inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)")
        add(lines, 8, "return y_g.index_select(0, inv_perm).reshape_as(out)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _gegelu(x, limit=None, alpha=1.702):")
        add(lines, 8, "if x.shape[-1] % 2 != 0:")
        add(lines, 12, "raise ValueError('gegelu requires even last dimension')")
        add(lines, 8, "x_gelu = x[..., ::2]")
        add(lines, 8, "x_linear = x[..., 1::2]")
        add(lines, 8, "if limit is not None:")
        add(lines, 12, "limit = float(limit)")
        add(lines, 12, "x_gelu = torch.where(torch.isinf(x_gelu), x_gelu, x_gelu.clamp(max=limit))")
        add(lines, 12, "x_linear = torch.where(torch.isinf(x_linear), x_linear, x_linear.clamp(min=-limit, max=limit))")
        add(lines, 8, "return x_gelu * torch.sigmoid(float(alpha) * x_gelu) * (x_linear + 1.0)")
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
        add(lines, 4, "def _tensor_like(cls, value, ref, dtype=None):")
        add(lines, 8, "target_dtype = dtype or ref.dtype")
        add(lines, 8, "if torch.is_tensor(value):")
        add(lines, 12, "return value.to(device=ref.device, dtype=target_dtype)")
        add(lines, 8, "if isinstance(value, (list, tuple)):")
        add(lines, 12, "return torch.as_tensor(value, device=ref.device, dtype=target_dtype)")
        add(lines, 8, "return torch.full_like(ref, value, dtype=target_dtype)")
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
        add(lines, 8, f"pending = {sorted(self.cached_global_symbol_names)!r}")
        add(lines, 8, "last_errors = {}")
        add(lines, 8, "while pending:")
        add(lines, 12, "next_pending = []")
        add(lines, 12, "progressed = False")
        add(lines, 12, "for name in pending:")
        add(lines, 16, "try:")
        for name in sorted(self.cached_global_symbol_names):
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
        add(lines, 16, "raise RuntimeError(f'unable to evaluate cached graph global symbols; blocked at {name!r}') from exc")
        add(lines, 12, "pending = next_pending")
        add(lines, 8, f"pending = {sorted(self.global_symbol_names - self.cached_global_symbol_names)!r}")
        add(lines, 8, "last_errors = {}")
        add(lines, 8, "while pending:")
        add(lines, 12, "next_pending = []")
        add(lines, 12, "progressed = False")
        add(lines, 12, "for name in pending:")
        add(lines, 16, "try:")
        for name in sorted(self.global_symbol_names - self.cached_global_symbol_names):
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
        params = ", ".join(f"{_py_ident(value.name)}=None" for value in module.inputs)
        if params:
            params = ", " + params
        method_name = self.method_names[module.name]
        arg_names = ", ".join(_py_ident(value.name) for value in module.inputs)
        body_name = method_name
        if self.profile:
            impl_name = f"{method_name}__impl"
            add(lines, 4, f"def {method_name}(self{params}):")
            call_args = f", {arg_names}" if arg_names else ""
            add(lines, 8, f"return self._profile_call({('module:' + module.name)!r}, self.{impl_name}{call_args})")
            add(lines, 4, "")
            body_name = impl_name
        add(lines, 4, f"def {body_name}(self{params}):")
        local = {value.name for value in module.inputs}
        required_dim_names = _module_free_dim_refs(module, global_names=self.global_symbol_names)
        required_dim_names.update(_tensor_size_static_dim_names(module))
        required_shape_dims_by_value = _module_value_required_shape_dim_names(
            module,
            global_names=self.global_symbol_names,
        )
        for value in module.inputs:
            _emit_bind_nested_shape_symbols(
                lines,
                add=add,
                type_expr=_effective_graph_value_type(value),
                value_expr=_py_ident(value.name),
                local=local,
                protected=self.global_symbol_names,
                required_names=required_dim_names | required_shape_dims_by_value.get(value.name, set()),
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
            source = _dim_ident(dim_params[0])
            for name in free_dim_refs:
                target = _dim_ident(name)
                add(lines, 8, f"{target} = {source}")
                local.add(name)
        self._emitted_defs_stack.append({})
        self._emitted_aliases_stack.append({})
        try:
            for node in module.nodes:
                self._emit_node(lines, node, module_name=module.name, indent=8, local=local, symbols_dict="self._symbols")
                self._record_emitted_node_defs(node)
                for output in node.outputs:
                    local.add(output.name)
                    output_type = _effective_graph_value_type(output)
                    # Node outputs can introduce local symbolic dimensions that are
                    # consumed later in the same module, e.g. Tensor[N] from
                    # _where_indices followed by Tensor[N,D] gathers. These symbols
                    # are not necessarily module-free dims, so bind all local
                    # output shape names. If type inference assigned a name that
                    # collides with a model-global constant, codegen must not
                    # create a Python local with that name: Python would shadow the
                    # global symbol expression throughout the function.
                    _emit_bind_nested_shape_symbols(
                        lines,
                        add=add,
                        type_expr=output_type,
                        value_expr=output.name,
                        local=local,
                        protected=self.global_symbol_names,
                        required_names=required_shape_dims_by_value.get(output.name, set()),
                    )
        finally:
            self._emitted_defs_stack.pop()
            self._emitted_aliases_stack.pop()
        outs = ", ".join(self._operand_expr(item, local=local, symbols_dict="self._symbols") for item in module.outputs)
        if len(module.outputs) == 1:
            add(lines, 8, f"return ({outs},)")
        else:
            add(lines, 8, f"return ({outs})")

    def _emit_forward(self, lines: list[str]) -> None:
        main = self.modules_by_name[self.program.main_module]
        add = self._add
        add(lines, 4, "def _forward(self, input_ids=None, **inputs):")
        args: list[str] = []
        first_input = main.inputs[0].name if main.inputs else None
        static_attention_inputs = {
            value.name
            for value in main.inputs
            if value.name in {"attn_mask", "attention_mask", "decoder_attention_mask"}
            and _is_static_mask_type(value.type_expr)
        }
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
                if value.name in static_attention_inputs:
                    capacity_expr = _static_mask_capacity_expr(value.type_expr, global_names=self.global_symbol_names)
                    add(lines, 8, f"if torch.is_tensor({value.name}):")
                    add(lines, 12, f"__static_len_{value.name} = int({value.name}.shape[1])")
                    if capacity_expr is None:
                        add(lines, 12, f"__static_declared_capacity_{value.name} = int(input_ids.shape[1]) if input_ids is not None else __static_len_{value.name}")
                    else:
                        add(lines, 12, f"__static_declared_capacity_{value.name} = int({capacity_expr})")
                    add(lines, 12, f"__static_capacity_{value.name} = max(__static_declared_capacity_{value.name}, __static_len_{value.name})")
                    add(lines, 12, f"__static_store_{value.name} = torch.zeros(({value.name}.shape[0], __static_capacity_{value.name}), dtype={value.name}.dtype, device={value.name}.device)")
                    add(lines, 12, f"__static_store_{value.name}[:, :__static_len_{value.name}] = {value.name}")
                    add(lines, 12, f"{value.name} = (__static_store_{value.name}, __static_len_{value.name})")
                args.append(value.name)
        add(lines, 8, f"result = self.{self.method_names[main.name]}({', '.join(args)})")
        names = _main_output_names(self.program, main)
        if len(names) == 1:
            add(lines, 8, "return result[0]")
        else:
            add(lines, 8, f"return {{{', '.join(f'{name!r}: result[{idx}]' for idx, name in enumerate(names))}}}")
        add(lines, 4, "")
        add(lines, 4, "def forward(self, input_ids=None, **inputs):")
        add(lines, 8, "return self._forward(input_ids, **inputs)")

    def _emit_generate(self, lines: list[str]) -> None:
        add = self._add
        main = self.modules_by_name[self.program.main_module]
        input_names = {value.name for value in main.inputs}
        output_names = set(_main_output_names(self.program, main))
        attention_name = "attn_mask" if "attn_mask" in input_names else (
            "attention_mask" if "attention_mask" in input_names else None
        )
        attention_value = next(
            (value for value in main.inputs if value.name == attention_name),
            None,
        )
        uses_static_attention_mask = (
            attention_value is not None and _is_static_mask_type(attention_value.type_expr)
        )
        static_attention_capacity_expr = (
            _static_mask_capacity_expr(attention_value.type_expr, global_names=self.global_symbol_names)
            if attention_value is not None
            else None
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
        add(lines, 8, "def _static_attention_mask(mask, prompt_ids, capacity):")
        add(lines, 12, "valid = _ones_like_ids(prompt_ids) if mask is None else mask.to(device=prompt_ids.device)")
        add(lines, 12, "store = torch.zeros((valid.shape[0], int(capacity)), dtype=valid.dtype, device=valid.device)")
        add(lines, 12, "length = int(valid.shape[1])")
        add(lines, 12, "store[:, :length] = valid")
        add(lines, 12, "return (store, length)")
        add(lines, 8, "def _append_static_attention_mask(mask, next_id):")
        add(lines, 12, "store, length = mask")
        add(lines, 12, "src = _ones_like_ids(next_id).to(dtype=store.dtype, device=store.device)")
        add(lines, 12, "store[:, length:length + 1] = src")
        add(lines, 12, "return (store, length + 1)")
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
                if uses_static_attention_mask:
                    if static_attention_capacity_expr is None:
                        add(lines, 8, "static_capacity = max(int(self.config.get('n_positions', self.config.get('max_position_embeddings', out.shape[1] + limit))), int(out.shape[1]) + int(limit))")
                    else:
                        add(lines, 8, f"static_capacity = max(int({static_attention_capacity_expr}), int(out.shape[1]) + int(limit))")
                    add(lines, 8, "attention_mask = _static_attention_mask(attention_mask, out, static_capacity)")
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
            add(lines, 12, "result = self._forward(step_input, **forward_kwargs)")
            add(lines, 12, "logits = _logits(result)")
            add(lines, 12, "if isinstance(result, dict):")
            add(lines, 16, f"cache = result.get({cache_output_name!r}, cache)")
            add(lines, 12, "next_id = _next_id(logits)")
            add(lines, 12, "next_id, finished = _apply_eos(next_id, eos, pad, finished)")
            if self.align_devices:
                add(lines, 12, "out = self._move_to(out, next_id.device)")
            add(lines, 12, "out = torch.cat([out, next_id], dim=1)")
            if attention_name is not None:
                if self.align_devices:
                    if uses_static_attention_mask:
                        add(lines, 12, "attention_mask = (self._move_to(attention_mask[0], next_id.device), attention_mask[1])")
                    else:
                        add(lines, 12, "attention_mask = self._move_to(attention_mask, next_id.device)")
                if uses_static_attention_mask:
                    add(lines, 12, "attention_mask = _append_static_attention_mask(attention_mask, next_id)")
                else:
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
            add(lines, 12, "result = self._forward(out, **forward_kwargs)")
            add(lines, 12, "next_id = _next_id(_logits(result))")
            add(lines, 12, "next_id, finished = _apply_eos(next_id, eos, pad, finished)")
            if self.align_devices:
                add(lines, 12, "out = self._move_to(out, next_id.device)")
            add(lines, 12, "out = torch.cat([out, next_id], dim=1)")
            if attention_name is not None:
                if self.align_devices:
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
        add(lines, 12, "result = self._forward(input_ids, **forward_kwargs)")
        add(lines, 12, "next_id = _next_id(_logits(result))")
        add(lines, 12, "next_id, finished = _apply_eos(next_id, eos, pad, finished)")
        if self.align_devices:
            add(lines, 12, "decoder_input_ids = self._move_to(decoder_input_ids, next_id.device)")
        add(lines, 12, "decoder_input_ids = torch.cat([decoder_input_ids, next_id], dim=1)")
        if decoder_attention_name is not None:
            if self.align_devices:
                add(lines, 12, "decoder_attention_mask = self._move_to(decoder_attention_mask, next_id.device)")
            add(lines, 12, "decoder_attention_mask = torch.cat([decoder_attention_mask, _ones_like_ids(next_id)], dim=1)")
        add(lines, 12, "if finished is not None and bool(finished.all().item()):")
        add(lines, 16, "break")
        add(lines, 8, "return decoder_input_ids")

    def _emit_node(self, lines: list[str], node: Any, *, module_name: str, indent: int, local: set[str], symbols_dict: str) -> None:
        add = self._add
        op = node.op.name
        if op == "core.repeat":
            self._emit_repeat_node(
                lines,
                node,
                module_name=module_name,
                indent=indent,
                local=local,
                symbols_dict=symbols_dict,
            )
            return
        targets = tuple(_value_ident(value) for value in node.outputs)
        target_names = tuple(value.name for value in node.outputs)
        if (
            self._emit_direct_module_call_node(
                lines,
                node,
                targets=targets,
                module_name=module_name,
                indent=indent,
                local=local,
                symbols_dict=symbols_dict,
            )
        ):
            return
        if (
            op == "core.select"
            and self._emit_select_node_as_control(
                lines,
                node,
                targets=targets,
                module_name=module_name,
                indent=indent,
                local=local,
                symbols_dict=symbols_dict,
            )
        ):
            return
        if (
            len(targets) == 1
            and _normalize_primitive_op(op) == "linear"
            and self._emit_linear_node(
                lines,
                node,
                target=targets[0],
                indent=indent,
                local=local,
                symbols_dict=symbols_dict,
            )
        ):
            return
        if (
            len(targets) == 1
            and _normalize_primitive_op(op) == "layernorm"
            and self._emit_layernorm_node(
                lines,
                node,
                target=targets[0],
                indent=indent,
                local=local,
                symbols_dict=symbols_dict,
            )
        ):
            return
        expr = self._node_expr(node, local=local, symbols_dict=symbols_dict)
        label = f"node:{module_name}:{','.join(target_names) or '_'}:{op}"
        if len(targets) == 1:
            if self.profile:
                add(lines, indent, f"{targets[0]} = self._profile_call({label!r}, lambda: {expr})")
            else:
                add(lines, indent, f"{targets[0]} = {expr}")
        else:
            joined = ", ".join(targets)
            if self.profile:
                add(lines, indent, f"{joined} = self._profile_call({label!r}, lambda: {expr})")
            else:
                add(lines, indent, f"{joined} = {expr}")

    def _literal_bool_arg(self, operand: GraphOperand) -> bool | None:
        if isinstance(operand, GraphLiteral) and type(operand.value) is bool:
            return operand.value
        return None

    def _literal_null_arg(self, operand: GraphOperand) -> bool:
        return isinstance(operand, GraphLiteral) and operand.value is None

    def _emit_optional_param_bind(
        self,
        lines: list[str],
        *,
        target: str,
        value_expr: str,
        flag_expr: str,
        flag_literal: bool | None,
        indent: int,
    ) -> None:
        if flag_literal is True:
            self._add(lines, indent, f"{target} = {value_expr}")
        elif flag_literal is False:
            self._add(lines, indent, f"{target} = None")
        else:
            self._add(lines, indent, f"{target} = ({value_expr} if {flag_expr} else None)")

    def _emit_select_node_as_control(
        self,
        lines: list[str],
        node: Any,
        *,
        targets: tuple[str, ...],
        module_name: str,
        indent: int,
        local: set[str],
        symbols_dict: str,
    ) -> bool:
        if len(node.inputs) != 3 or not targets:
            return False
        cond_operand, then_operand, else_operand = node.inputs
        if not (
            self._branch_benefits_from_control_inline(then_operand, module_name=module_name)
            or self._branch_benefits_from_control_inline(else_operand, module_name=module_name)
        ):
            return False
        cond = self._operand_expr(cond_operand, local=local, symbols_dict=symbols_dict)
        cond_expr = cond if self._operand_is_bool(cond_operand) else f"bool({cond})"
        self._add(lines, indent, f"if {cond_expr}:")
        self._emit_select_branch(
            lines,
            then_operand,
            targets=targets,
            target_outputs=node.outputs,
            module_name=module_name,
            indent=indent + 4,
            local=local,
            symbols_dict=symbols_dict,
            inline_prefix=f"__select_inline_{node.id.replace(':', '_')}_then",
        )
        self._add(lines, indent, "else:")
        self._emit_select_branch(
            lines,
            else_operand,
            targets=targets,
            target_outputs=node.outputs,
            module_name=module_name,
            indent=indent + 4,
            local=local,
            symbols_dict=symbols_dict,
            inline_prefix=f"__select_inline_{node.id.replace(':', '_')}_else",
        )
        return True

    def _branch_benefits_from_control_inline(self, operand: GraphOperand, *, module_name: str) -> bool:
        return (
            isinstance(operand, GraphExpr)
            and operand.op.name in self.modules_by_name
            and operand.op.name != module_name
            and operand.op.name not in self.global_symbol_names
            and not operand.attrs
        )

    def _emit_direct_module_call_node(
        self,
        lines: list[str],
        node: Any,
        *,
        targets: tuple[str, ...],
        module_name: str,
        indent: int,
        local: set[str],
        symbols_dict: str,
    ) -> bool:
        if (
            not targets
            or not self._can_inline_direct_module_call(
                node.op.name,
                module_name=module_name,
                attrs=node.attrs,
            )
        ):
            return False
        callee_module = self.modules_by_name.get(node.op.name)
        if callee_module is None:
            return False
        if self._emit_one_node_forwarder_call(
            lines,
            node,
            callee_module=callee_module,
            module_name=module_name,
            indent=indent,
            local=local,
            symbols_dict=symbols_dict,
        ):
            return True
        return self._emit_inline_module_body(
            lines,
            callee_module=callee_module,
            arg_operands=node.inputs,
            targets=targets,
            target_outputs=node.outputs,
            module_name=module_name,
            indent=indent,
            local=local,
            symbols_dict=symbols_dict,
            inline_prefix=f"__call_inline_{node.id.replace(':', '_')}",
        )

    def _emit_one_node_forwarder_call(
        self,
        lines: list[str],
        node: Any,
        *,
        callee_module: GraphModule,
        module_name: str,
        indent: int,
        local: set[str],
        symbols_dict: str,
    ) -> bool:
        if len(callee_module.nodes) != 1:
            return False
        inner_node = callee_module.nodes[0]
        if len(callee_module.outputs) != len(inner_node.outputs) or len(node.outputs) != len(inner_node.outputs):
            return False
        for module_output, inner_output in zip(callee_module.outputs, inner_node.outputs, strict=True):
            if not isinstance(module_output, GraphValueRef) or module_output.name != inner_output.name:
                return False
        param_subst: dict[str, GraphOperand] = {
            param.name: arg
            for param, arg in zip(callee_module.inputs, node.inputs, strict=True)
        }

        def rewrite_operand(operand: GraphOperand) -> GraphOperand:
            if isinstance(operand, GraphValueRef) and operand.name in param_subst:
                return param_subst[operand.name]
            if isinstance(operand, GraphPath):
                parts: list[str] = []
                for part in operand.parts:
                    rewritten_part = part
                    for name, replacement in param_subst.items():
                        if isinstance(replacement, GraphValueRef):
                            rewritten_part = rewritten_part.replace(
                                "{" + name + "}",
                                "{" + replacement.name + "}",
                            )
                        else:
                            replacement_text = _graph_path_template_replacement(replacement)
                            if replacement_text is not None:
                                rewritten_part = rewritten_part.replace(
                                    "{" + name + "}",
                                    replacement_text,
                                )
                    parts.append(rewritten_part)
                return replace(operand, parts=tuple(parts))
            if isinstance(operand, GraphExpr):
                return replace(
                    operand,
                    inputs=tuple(rewrite_operand(item) for item in operand.inputs),
                    attrs={key: rewrite_operand(value) for key, value in operand.attrs.items()},
                )
            return operand

        rewritten_node = replace(
            inner_node,
            id=f"{node.id}:forward:{inner_node.id}",
            inputs=tuple(rewrite_operand(item) for item in inner_node.inputs),
            attrs={key: rewrite_operand(value) for key, value in inner_node.attrs.items()},
            outputs=node.outputs,
            source_module=module_name,
            type_expr=node.type_expr,
            dims=node.dims,
        )
        self._emit_node(
            lines,
            rewritten_node,
            module_name=module_name,
            indent=indent,
            local=local,
            symbols_dict=symbols_dict,
        )
        return True

    def _emit_select_branch(
        self,
        lines: list[str],
        operand: GraphOperand,
        *,
        targets: tuple[str, ...],
        target_outputs: tuple[GraphValue, ...] | None,
        module_name: str,
        indent: int,
        local: set[str],
        symbols_dict: str,
        inline_prefix: str,
    ) -> None:
        if isinstance(operand, GraphExpr) and self._emit_inline_call_expr(
            lines,
            operand,
            targets=targets,
            target_outputs=target_outputs,
            module_name=module_name,
            indent=indent,
            local=local,
            symbols_dict=symbols_dict,
            inline_prefix=inline_prefix,
        ):
            return
        expr = self._operand_expr(operand, local=local, symbols_dict=symbols_dict)
        joined = ", ".join(targets)
        self._add(lines, indent, f"{joined} = {expr}")

    def _emit_inline_call_expr(
        self,
        lines: list[str],
        expr: GraphExpr,
        *,
        targets: tuple[str, ...],
        target_outputs: tuple[GraphValue, ...] | None,
        module_name: str,
        indent: int,
        local: set[str],
        symbols_dict: str,
        inline_prefix: str,
    ) -> bool:
        callee = expr.op.name
        callee_module = self.modules_by_name.get(callee)
        if (
            callee_module is None
            or callee == module_name
            or expr.attrs
            or len(callee_module.inputs) != len(expr.inputs)
        ):
            return False
        return self._emit_inline_module_body(
            lines,
            callee_module=callee_module,
            arg_operands=expr.inputs,
            targets=targets,
            target_outputs=target_outputs,
            module_name=module_name,
            indent=indent,
            local=local,
            symbols_dict=symbols_dict,
            inline_prefix=inline_prefix,
        )

    def _emit_inline_module_body(
        self,
        lines: list[str],
        *,
        callee_module: GraphModule,
        arg_operands: tuple[GraphOperand, ...],
        targets: tuple[str, ...],
        target_outputs: tuple[GraphValue, ...] | None,
        module_name: str,
        indent: int,
        local: set[str],
        symbols_dict: str,
        inline_prefix: str,
    ) -> bool:
        if len(callee_module.inputs) != len(arg_operands):
            return False
        if not self._inline_outputs_match_targets(callee_module, targets):
            return False
        if callee_module.name in self._inline_stack:
            return False
        self._inline_stack.add(callee_module.name)
        subst: dict[str, GraphOperand] = {}
        dim_name_subst: dict[str, Any] = {}
        inline_local = set(local)
        callee_required_dim_names = _module_free_dim_refs(
            callee_module,
            global_names=self.global_symbol_names,
        )
        callee_required_shape_dims_by_value = _module_value_required_shape_dim_names(
            callee_module,
            global_names=self.global_symbol_names,
        )
        callee_required_shape_dims_by_value = _module_value_required_shape_dim_names(
            callee_module,
            global_names=self.global_symbol_names,
        )
        try:
            safe_prefix = _py_ident(inline_prefix)
            callsite_output_by_callee_name: dict[str, GraphValue] = {}
            if target_outputs is not None and len(callee_module.outputs) == len(target_outputs):
                callsite_output_by_callee_name = {
                    output.name: target_output
                    for output, target_output in zip(callee_module.outputs, target_outputs, strict=True)
                    if isinstance(output, GraphValue | GraphValueRef)
                }
            dim_subst = _inline_dim_subst(
                callee_module.inputs,
                arg_operands,
                fresh_prefix=safe_prefix,
                protected=self.global_symbol_names,
            )
            for param, operand in zip(callee_module.inputs, arg_operands, strict=True):
                temp_name = f"{safe_prefix}_{_py_ident(param.name)}"
                expr = self._operand_expr(operand, local=local, symbols_dict=symbols_dict)
                self._add(lines, indent, f"{temp_name} = {expr}")
                self._record_emitted_alias(temp_name, operand)
                inline_local.add(temp_name)
                subst[param.name] = GraphValueRef(
                    name=temp_name,
                    type_expr=graph_operand_type(operand),
                    dims=operand.dims if isinstance(operand, GraphValueRef | GraphExpr) else None,
                )
                if isinstance(param.type_expr, TypeDim | TypeInt):
                    dim_name_subst[param.name] = temp_name
                _emit_bind_nested_shape_symbols(
                    lines,
                    add=self._add,
                    type_expr=graph_operand_type(operand),
                    value_expr=temp_name,
                    local=inline_local,
                    protected=self.global_symbol_names,
                    required_names=callee_required_dim_names,
                    indent=indent,
                )
                formal_ref = GraphValueRef(name=temp_name, type_expr=param.type_expr, dims=param.dims)
                if dim_subst:
                    formal_ref = substitute_graph_operand_dims(formal_ref, dim_subst)
                formal_type = formal_ref.type_expr
                if _operand_may_be_none(operand) and not isinstance(formal_type, TypeOptional):
                    formal_type = TypeOptional(formal_type)
                _emit_bind_nested_shape_symbols(
                    lines,
                    add=self._add,
                    type_expr=formal_type,
                    value_expr=temp_name,
                    local=inline_local,
                    protected=self.global_symbol_names,
                    required_names=None,
                    indent=indent,
                )

            def rewrite_operand(operand: GraphOperand) -> GraphOperand:
                if isinstance(operand, GraphValueRef):
                    if operand.name in subst:
                        return subst[operand.name]
                    if isinstance(operand.type_expr, TypeDim | TypeInt) and operand.name in dim_subst:
                        replacement = dim_subst[operand.name]
                        if isinstance(replacement, str):
                            return GraphValueRef(
                                name=replacement,
                                type_expr=operand.type_expr,
                                dims=operand.dims,
                            )
                    return operand
                if isinstance(operand, GraphPath):
                    parts: list[str] = []
                    for part in operand.parts:
                        rewritten_part = part
                        for name, replacement in subst.items():
                            if isinstance(replacement, GraphValueRef):
                                rewritten_part = rewritten_part.replace(
                                    "{" + name + "}",
                                    "{" + replacement.name + "}",
                                )
                            else:
                                replacement_text = _graph_path_template_replacement(replacement)
                                if replacement_text is not None:
                                    rewritten_part = rewritten_part.replace(
                                        "{" + name + "}",
                                        replacement_text,
                                    )
                        parts.append(rewritten_part)
                    return replace(operand, parts=tuple(parts))
                if isinstance(operand, GraphExpr):
                    return replace(
                        operand,
                        inputs=tuple(rewrite_operand(item) for item in operand.inputs),
                        attrs={key: rewrite_operand(value) for key, value in operand.attrs.items()},
                    )
                return operand

            inline_use_counts = self._collect_inline_value_use_counts(callee_module)
            module_effects = infer_graph_module_effects(self.program.modules)
            for inner_index, inner_node in enumerate(callee_module.nodes, start=1):
                rewritten_outputs: list[GraphValue] = []
                original_output_names = tuple(output.name for output in inner_node.outputs)
                pre_node_dim_subst = dict(dim_subst)
                pre_node_dim_subst.update(dim_name_subst)
                for output in inner_node.outputs:
                    name = f"{safe_prefix}_{inner_index}_{_py_ident(output.name)}"
                    output_type = output.type_expr
                    output_dims = output.dims
                    callsite_output = callsite_output_by_callee_name.get(output.name)
                    if callsite_output is not None:
                        output_type = callsite_output.type_expr
                        output_dims = callsite_output.dims
                    rewritten = GraphValue(
                        name=name,
                        type_expr=output_type,
                        dims=output_dims,
                        optional=output.optional,
                    )
                    if pre_node_dim_subst:
                        rewritten = substitute_graph_node_dims(
                            GraphNode(
                                id="__tmp",
                                op=GraphOp("core.alias"),
                                inputs=(),
                                attrs={},
                                outputs=(rewritten,),
                                source_module=module_name,
                                type_expr=rewritten.type_expr,
                                dims=rewritten.dims,
                            ),
                            pre_node_dim_subst,
                        ).outputs[0]
                    rewritten_outputs.append(rewritten)
                    subst[output.name] = GraphValueRef(
                        name=name,
                        type_expr=rewritten.type_expr,
                        dims=rewritten.dims,
                    )
                    if isinstance(output.type_expr, TypeDim | TypeInt):
                        dim_name_subst[output.name] = name
                rewritten_node = replace(
                    inner_node,
                    id=f"{safe_prefix}:inline:{inner_index}",
                    inputs=tuple(rewrite_operand(item) for item in inner_node.inputs),
                    attrs={key: rewrite_operand(value) for key, value in inner_node.attrs.items()},
                    outputs=tuple(rewritten_outputs),
                    source_module=module_name,
                )
                if pre_node_dim_subst:
                    rewritten_node = substitute_graph_node_dims(
                        rewritten_node,
                        pre_node_dim_subst,
                    )
                for original_name, rewritten_output in zip(
                    original_output_names,
                    rewritten_node.outputs,
                    strict=True,
                ):
                    subst[original_name] = GraphValueRef(
                        name=rewritten_output.name,
                        type_expr=rewritten_output.type_expr,
                        dims=rewritten_output.dims,
                    )
                    if isinstance(rewritten_output.type_expr, TypeDim | TypeInt):
                        dim_name_subst[original_name] = rewritten_output.name
                inline_expr = self._inline_body_node_as_expr(
                    rewritten_node,
                    original_output_names=original_output_names,
                    use_counts=inline_use_counts,
                    module_effects=module_effects,
                )
                if inline_expr is not None:
                    subst[original_output_names[0]] = inline_expr
                    self._record_emitted_alias(rewritten_node.outputs[0].name, inline_expr)
                    continue
                self._emit_node(
                    lines,
                    rewritten_node,
                    module_name=module_name,
                    indent=indent,
                    local=inline_local,
                    symbols_dict=symbols_dict,
                )
                self._record_emitted_node_defs(rewritten_node)
                for original_name, output in zip(
                    original_output_names,
                    rewritten_node.outputs,
                    strict=True,
                ):
                    inline_local.add(output.name)
                    output_type = _effective_graph_value_type(output)
                    _emit_bind_nested_shape_symbols(
                        lines,
                        add=self._add,
                        type_expr=output_type,
                        value_expr=output.name,
                        local=inline_local,
                        protected=self.global_symbol_names,
                        required_names=callee_required_shape_dims_by_value.get(original_name, set()),
                        indent=indent,
                    )
            output_exprs = [
                self._operand_expr(
                    substitute_graph_operand_dims(rewrite_operand(output), {**dim_subst, **dim_name_subst})
                    if dim_subst or dim_name_subst
                    else rewrite_operand(output),
                    local=inline_local,
                    symbols_dict=symbols_dict,
                )
                for output in callee_module.outputs
            ]
            if len(targets) == 1:
                rhs = output_exprs[0] if len(output_exprs) == 1 else f"({', '.join(output_exprs)})"
            elif len(output_exprs) == len(targets):
                rhs = ", ".join(output_exprs)
            elif (
                len(output_exprs) == 1
                and len(callee_module.outputs) == 1
                and isinstance(graph_operand_type(callee_module.outputs[0]), TypeTuple)
                and len(graph_operand_type(callee_module.outputs[0]).items) == len(targets)
            ):
                rhs = output_exprs[0]
            else:
                return False
            joined = ", ".join(targets)
            self._add(lines, indent, f"{joined} = {rhs}")
            if target_outputs is not None:
                for target, output in zip(targets, target_outputs, strict=True):
                    output_type = _effective_graph_value_type(output)
                    _emit_bind_nested_shape_symbols(
                        lines,
                        add=self._add,
                        type_expr=output_type,
                        value_expr=target,
                        local=inline_local,
                        protected=self.global_symbol_names,
                        required_names=set(),
                        indent=indent,
                    )
                local.update(inline_local)
            return True
        finally:
            self._inline_stack.remove(callee_module.name)

    @staticmethod
    def _inline_outputs_match_targets(callee_module: GraphModule, targets: tuple[str, ...]) -> bool:
        if not targets:
            return False
        if len(targets) == 1:
            return True
        if len(callee_module.outputs) == len(targets):
            return True
        if len(callee_module.outputs) != 1:
            return False
        output_type = graph_operand_type(callee_module.outputs[0])
        return isinstance(output_type, TypeTuple) and len(output_type.items) == len(targets)

    def _emit_linear_node(
        self,
        lines: list[str],
        node: Any,
        *,
        target: str,
        indent: int,
        local: set[str],
        symbols_dict: str,
    ) -> bool:
        if len(node.inputs) < 2:
            return False
        if len(node.inputs) > 5 and not self._literal_null_arg(node.inputs[5]):
            return False
        args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
        bias_expr = self._scalar_operand_expr(
            node.inputs[3],
            local=local,
            symbols_dict=symbols_dict,
            expected=(TypeBool,),
            cast="bool",
        ) if len(node.inputs) > 3 else "False"
        transpose_expr = self._scalar_operand_expr(
            node.inputs[4],
            local=local,
            symbols_dict=symbols_dict,
            expected=(TypeBool,),
            cast="bool",
        ) if len(node.inputs) > 4 else "False"
        bias_literal = self._literal_bool_arg(node.inputs[3]) if len(node.inputs) > 3 else False
        transpose_literal = self._literal_bool_arg(node.inputs[4]) if len(node.inputs) > 4 else False
        weight = self._param_expr_for_path(
            node.inputs[0],
            node.inputs[6] if len(node.inputs) > 6 else "weight",
            local=local,
            symbols_dict=symbols_dict,
        )
        bias_value = self._param_expr_for_path(
            node.inputs[0],
            node.inputs[7] if len(node.inputs) > 7 else "bias",
            optional=True,
            local=local,
            symbols_dict=symbols_dict,
        )
        weight_name = f"{target}__weight"
        bias_name = f"{target}__bias"
        x_name = f"{target}__x"
        self._add(lines, indent, f"{weight_name} = {weight}")
        if bias_literal is False:
            bias_arg = "None"
        else:
            self._emit_optional_param_bind(
                lines,
                target=bias_name,
                value_expr=bias_value,
                flag_expr=bias_expr,
                flag_literal=bias_literal,
                indent=indent,
            )
            if self.align_devices:
                self._add(lines, indent, f"{bias_name} = self._move_to({bias_name}, {weight_name}.device) if {bias_name} is not None else None")
            bias_arg = bias_name
        if self.align_devices:
            self._add(lines, indent, f"{x_name} = self._move_to({args[1]}, {weight_name}.device)")
        else:
            self._add(lines, indent, f"{x_name} = {args[1]}")
        if transpose_literal is True:
            if bias_literal is False:
                op_expr = f"torch.matmul({x_name}, {weight_name})"
            else:
                op_expr = f"torch.matmul({x_name}, {weight_name}) + ({bias_arg} if {bias_arg} is not None else 0)"
        elif transpose_literal is False:
            op_expr = f"F.linear({x_name}, {weight_name}, {bias_arg})"
        else:
            op_expr = f"(torch.matmul({x_name}, {weight_name}) + ({bias_arg} if {bias_arg} is not None else 0) if {transpose_expr} else F.linear({x_name}, {weight_name}, {bias_arg}))"
        if self.profile:
            self._add(lines, indent, f"{target} = self._profile_call({f'node:{target}:_linear'!r}, lambda: {op_expr})")
        else:
            self._add(lines, indent, f"{target} = {op_expr}")
        return True

    def _emit_layernorm_node(
        self,
        lines: list[str],
        node: Any,
        *,
        target: str,
        indent: int,
        local: set[str],
        symbols_dict: str,
    ) -> bool:
        if len(node.inputs) < 2:
            return False
        args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
        eps = self._scalar_operand_expr(
            node.inputs[2],
            local=local,
            symbols_dict=symbols_dict,
            expected=(TypeFloat, TypeInt, TypeDim),
            cast="float",
        ) if len(node.inputs) > 2 else "1e-5"
        bias = self._scalar_operand_expr(
            node.inputs[5],
            local=local,
            symbols_dict=symbols_dict,
            expected=(TypeBool,),
            cast="bool",
        ) if len(node.inputs) > 5 else "True"
        bias_literal = self._literal_bool_arg(node.inputs[5]) if len(node.inputs) > 5 else True
        weight = self._param_expr_for_path(
            node.inputs[0],
            node.inputs[4] if len(node.inputs) > 4 else "weight",
            local=local,
            symbols_dict=symbols_dict,
        )
        bias_value = self._param_expr_for_path(
            node.inputs[0],
            node.inputs[6] if len(node.inputs) > 6 else "bias",
            optional=True,
            local=local,
            symbols_dict=symbols_dict,
        )
        weight_name = f"{target}__weight"
        bias_name = f"{target}__bias"
        self._add(lines, indent, f"{weight_name} = {weight}")
        self._emit_optional_param_bind(
            lines,
            target=bias_name,
            value_expr=bias_value,
            flag_expr=bias,
            flag_literal=bias_literal,
            indent=indent,
        )
        x_expr = (
            f"self._move_to({args[1]}, {weight_name}.device)"
            if self.align_devices
            else args[1]
        )
        bias_expr = (
            f"(self._move_to({bias_name}, {weight_name}.device) if {bias_name} is not None else None)"
            if self.align_devices
            else bias_name
        )
        op_expr = (
            f"F.layer_norm({x_expr}, "
            f"({args[1]}.shape[-1],), weight={weight_name}, "
            f"bias={bias_expr}, "
            f"eps={eps})"
        )
        if self.profile:
            self._add(lines, indent, f"{target} = self._profile_call({f'node:{target}:_layernorm'!r}, lambda: {op_expr})")
        else:
            self._add(lines, indent, f"{target} = {op_expr}")
        return True

    def _repeat_attr_string(self, node: Any, key: str) -> str:
        value = node.attrs.get(key)
        if not isinstance(value, GraphLiteral) or not isinstance(value.value, str):
            raise ValueError(f"core.repeat attr {key!r} must be a string literal")
        return value.value

    def _repeat_attr_int(self, node: Any, key: str) -> int:
        value = node.attrs.get(key)
        if not isinstance(value, GraphLiteral) or type(value.value) is not int:
            raise ValueError(f"core.repeat attr {key!r} must be an int literal")
        return value.value

    def _emit_repeat_node(self, lines: list[str], node: Any, *, module_name: str, indent: int, local: set[str], symbols_dict: str) -> None:
        add = self._add
        callee = self._repeat_attr_string(node, "callee")
        arg_count = self._repeat_attr_int(node, "arg_count")
        carry_count = self._repeat_attr_int(node, "carry_count")
        if callee not in self.method_names:
            raise ValueError(f"core.repeat references unknown callee {callee!r}")
        if len(node.inputs) < 3 + carry_count:
            raise ValueError("core.repeat missing carry inputs")
        targets = tuple(_value_ident(value) for value in node.outputs)
        for index, target in enumerate(targets):
            init_expr = self._operand_expr(node.inputs[3 + index], local=local, symbols_dict=symbols_dict)
            add(lines, indent, f"{target} = {init_expr}")
        loop_var = f"__loop_i_{node.id.rsplit(':', 1)[-1].replace('-', '_')}"
        start = self._scalar_operand_expr(
            node.inputs[0],
            local=local,
            symbols_dict=symbols_dict,
            expected=(TypeInt, TypeDim),
            cast="int",
        )
        stop = self._scalar_operand_expr(
            node.inputs[1],
            local=local,
            symbols_dict=symbols_dict,
            expected=(TypeInt, TypeDim),
            cast="int",
        )
        step = self._scalar_operand_expr(
            node.inputs[2],
            local=local,
            symbols_dict=symbols_dict,
            expected=(TypeInt, TypeDim),
            cast="int",
        )
        add(lines, indent, f"for {loop_var} in range({start}, {stop}, {step}):")
        if self._emit_repeat_inline_body(
            lines,
            node,
            callee=callee,
            arg_count=arg_count,
            carry_count=carry_count,
            targets=targets,
            loop_var=loop_var,
            module_name=module_name,
            indent=indent + 4,
            local=local,
            symbols_dict=symbols_dict,
        ):
            return
        args: list[str] = []
        for index in range(arg_count):
            role = self._repeat_attr_string(node, f"arg_{index}")
            if role == "iter":
                args.append(loop_var)
            elif role.startswith("carry:"):
                carry_index = int(role.removeprefix("carry:"))
                args.append(targets[carry_index])
            elif role.startswith("input:"):
                input_index = int(role.removeprefix("input:"))
                args.append(self._operand_expr(node.inputs[input_index], local=local, symbols_dict=symbols_dict))
            else:
                raise ValueError(f"invalid core.repeat arg role {role!r}")
        call = f"self.{self.method_names[callee]}({', '.join(args)})"
        label = f"node:{module_name}:{','.join(value.name for value in node.outputs) or '_'}:core.repeat"
        joined = ", ".join(targets)
        rhs = f"{call}[0]" if len(targets) == 1 else call
        if self.profile:
            add(lines, indent + 4, f"{joined} = self._profile_call({label!r}, lambda: {rhs})")
        else:
            add(lines, indent + 4, f"{joined} = {rhs}")

    def _repeat_arg_exprs(
        self,
        node: Any,
        *,
        arg_count: int,
        targets: tuple[str, ...],
        loop_var: str,
        local: set[str],
        symbols_dict: str,
    ) -> list[str]:
        args: list[str] = []
        for index in range(arg_count):
            role = self._repeat_attr_string(node, f"arg_{index}")
            if role == "iter":
                args.append(loop_var)
            elif role.startswith("carry:"):
                carry_index = int(role.removeprefix("carry:"))
                args.append(targets[carry_index])
            elif role.startswith("input:"):
                input_index = int(role.removeprefix("input:"))
                args.append(self._operand_expr(node.inputs[input_index], local=local, symbols_dict=symbols_dict))
            else:
                raise ValueError(f"invalid core.repeat arg role {role!r}")
        return args

    def _emit_repeat_inline_body(
        self,
        lines: list[str],
        node: Any,
        *,
        callee: str,
        arg_count: int,
        carry_count: int,
        targets: tuple[str, ...],
        loop_var: str,
        module_name: str,
        indent: int,
        local: set[str],
        symbols_dict: str,
    ) -> bool:
        callee_module = self.modules_by_name.get(callee)
        if callee_module is None or len(callee_module.inputs) != arg_count or len(callee_module.outputs) != carry_count:
            return False
        arg_exprs = self._repeat_arg_exprs(
            node,
            arg_count=arg_count,
            targets=targets,
            loop_var=loop_var,
            local=local,
            symbols_dict=symbols_dict,
        )
        arg_operands: list[GraphOperand] = []
        for index in range(arg_count):
            role = self._repeat_attr_string(node, f"arg_{index}")
            if role == "iter":
                arg_operands.append(GraphValueRef(name=loop_var, type_expr=TypeInt(), dims=None))
            elif role.startswith("carry:"):
                carry_index = int(role.removeprefix("carry:"))
                arg_operands.append(
                    GraphValueRef(
                        name=targets[carry_index],
                        type_expr=node.outputs[carry_index].type_expr,
                        dims=node.outputs[carry_index].dims,
                    )
                )
            elif role.startswith("input:"):
                input_index = int(role.removeprefix("input:"))
                arg_operands.append(node.inputs[input_index])
            else:
                raise ValueError(f"invalid core.repeat arg role {role!r}")
        inline_prefix = _py_ident(f"__loop_inline_{node.id.replace(':', '_')}")
        subst: dict[str, GraphOperand] = {}
        dim_name_subst: dict[str, Any] = {}
        inline_local = set(local)
        dim_subst = _inline_dim_subst(
            callee_module.inputs,
            tuple(arg_operands),
            fresh_prefix=inline_prefix,
            protected=self.global_symbol_names,
        )
        callee_required_dim_names = _module_free_dim_refs(
            callee_module,
            global_names=self.global_symbol_names,
        )
        callee_required_shape_dims_by_value = _module_value_required_shape_dim_names(
            callee_module,
            global_names=self.global_symbol_names,
        )
        for param, expr, operand in zip(callee_module.inputs, arg_exprs, arg_operands, strict=True):
            temp_name = f"{inline_prefix}_{_py_ident(param.name)}"
            lines.append(" " * indent + f"{temp_name} = {expr}")
            self._record_emitted_alias(temp_name, operand)
            inline_local.add(temp_name)
            subst[param.name] = GraphValueRef(
                name=temp_name,
                type_expr=graph_operand_type(operand),
                dims=operand.dims if isinstance(operand, GraphValueRef | GraphExpr) else None,
            )
            if isinstance(param.type_expr, TypeDim | TypeInt):
                dim_name_subst[param.name] = temp_name
            _emit_bind_nested_shape_symbols(
                lines,
                add=self._add,
                type_expr=graph_operand_type(operand),
                value_expr=temp_name,
                local=inline_local,
                protected=self.global_symbol_names,
                required_names=callee_required_dim_names,
                indent=indent,
            )
            formal_ref = GraphValueRef(name=temp_name, type_expr=param.type_expr, dims=param.dims)
            if dim_subst:
                formal_ref = substitute_graph_operand_dims(formal_ref, dim_subst)
            formal_type = formal_ref.type_expr
            if _operand_may_be_none(operand) and not isinstance(formal_type, TypeOptional):
                formal_type = TypeOptional(formal_type)
            _emit_bind_nested_shape_symbols(
                lines,
                add=self._add,
                type_expr=formal_type,
                value_expr=temp_name,
                local=inline_local,
                protected=self.global_symbol_names,
                required_names=None,
                indent=indent,
            )

        def rewrite_operand(operand: GraphOperand) -> GraphOperand:
            if isinstance(operand, GraphValueRef):
                if operand.name in subst:
                    return subst[operand.name]
                if isinstance(operand.type_expr, TypeDim | TypeInt) and operand.name in dim_subst:
                    replacement = dim_subst[operand.name]
                    if isinstance(replacement, str):
                        return GraphValueRef(
                            name=replacement,
                            type_expr=operand.type_expr,
                            dims=operand.dims,
                        )
                return operand
            if isinstance(operand, GraphPath):
                parts: list[str] = []
                for part in operand.parts:
                    rewritten_part = part
                    for name, replacement in subst.items():
                        if isinstance(replacement, GraphValueRef):
                            rewritten_part = rewritten_part.replace(
                                "{" + name + "}",
                                "{" + replacement.name + "}",
                            )
                        else:
                            replacement_text = _graph_path_template_replacement(replacement)
                            if replacement_text is not None:
                                rewritten_part = rewritten_part.replace(
                                    "{" + name + "}",
                                    replacement_text,
                                )
                    parts.append(rewritten_part)
                return replace(operand, parts=tuple(parts))
            if isinstance(operand, GraphExpr):
                return replace(
                    operand,
                    inputs=tuple(rewrite_operand(item) for item in operand.inputs),
                    attrs={key: rewrite_operand(value) for key, value in operand.attrs.items()},
                )
            return operand

        inline_use_counts = self._collect_inline_value_use_counts(callee_module)
        module_effects = infer_graph_module_effects(self.program.modules)
        for inner_index, inner_node in enumerate(callee_module.nodes, start=1):
            rewritten_outputs: list[GraphValue] = []
            original_output_names = tuple(output.name for output in inner_node.outputs)
            pre_node_dim_subst = dict(dim_subst)
            pre_node_dim_subst.update(dim_name_subst)
            for output in inner_node.outputs:
                name = f"{inline_prefix}_{inner_index}_{_py_ident(output.name)}"
                rewritten = GraphValue(name=name, type_expr=output.type_expr, dims=output.dims, optional=output.optional)
                if pre_node_dim_subst:
                    rewritten = substitute_graph_node_dims(
                        GraphNode(
                            id="__tmp",
                            op=GraphOp("core.alias"),
                            inputs=(),
                            attrs={},
                            outputs=(rewritten,),
                            source_module=module_name,
                            type_expr=rewritten.type_expr,
                            dims=rewritten.dims,
                        ),
                        pre_node_dim_subst,
                    ).outputs[0]
                rewritten_outputs.append(rewritten)
                subst[output.name] = GraphValueRef(name=name, type_expr=output.type_expr, dims=output.dims)
                if isinstance(output.type_expr, TypeDim | TypeInt):
                    dim_name_subst[output.name] = name
            rewritten_node = replace(
                inner_node,
                id=f"{node.id}:inline:{inner_index}",
                inputs=tuple(rewrite_operand(item) for item in inner_node.inputs),
                attrs={key: rewrite_operand(value) for key, value in inner_node.attrs.items()},
                outputs=tuple(rewritten_outputs),
                source_module=module_name,
            )
            if pre_node_dim_subst:
                rewritten_node = substitute_graph_node_dims(
                    rewritten_node,
                    pre_node_dim_subst,
                )
            for original_name, rewritten_output in zip(
                original_output_names,
                rewritten_node.outputs,
                strict=True,
            ):
                subst[original_name] = GraphValueRef(
                    name=rewritten_output.name,
                    type_expr=rewritten_output.type_expr,
                    dims=rewritten_output.dims,
                )
                if isinstance(rewritten_output.type_expr, TypeDim | TypeInt):
                    dim_name_subst[original_name] = rewritten_output.name
            inline_expr = self._inline_body_node_as_expr(
                rewritten_node,
                original_output_names=original_output_names,
                use_counts=inline_use_counts,
                module_effects=module_effects,
            )
            if inline_expr is not None:
                subst[original_output_names[0]] = inline_expr
                self._record_emitted_alias(rewritten_node.outputs[0].name, inline_expr)
                continue
            self._emit_node(
                lines,
                rewritten_node,
                module_name=module_name,
                indent=indent,
                local=inline_local,
                symbols_dict=symbols_dict,
            )
            self._record_emitted_node_defs(rewritten_node)
            for original_name, output in zip(
                original_output_names,
                rewritten_node.outputs,
                strict=True,
            ):
                inline_local.add(output.name)
                output_type = _effective_graph_value_type(output)
                _emit_bind_nested_shape_symbols(
                    lines,
                    add=self._add,
                    type_expr=output_type,
                    value_expr=output.name,
                    local=inline_local,
                    protected=self.global_symbol_names,
                    required_names=callee_required_shape_dims_by_value.get(original_name, set()),
                    indent=indent,
                )
        output_exprs = [
            self._operand_expr(
                substitute_graph_operand_dims(rewrite_operand(output), {**dim_subst, **dim_name_subst})
                if dim_subst or dim_name_subst
                else rewrite_operand(output),
                local=inline_local,
                symbols_dict=symbols_dict,
            )
            for output in callee_module.outputs
        ]
        if len(output_exprs) != len(targets):
            return False
        joined = ", ".join(targets)
        rhs = output_exprs[0] if len(output_exprs) == 1 else f"({', '.join(output_exprs)})"
        lines.append(" " * indent + f"{joined} = {rhs}")
        rewritten_outputs = [
            substitute_graph_operand_dims(rewrite_operand(output), dim_subst)
            if dim_subst
            else rewrite_operand(output)
            for output in callee_module.outputs
        ]
        for target, rewritten_output in zip(targets, rewritten_outputs, strict=True):
            if isinstance(rewritten_output, GraphValueRef | GraphExpr):
                self._record_emitted_alias(target, rewritten_output)
        for target, output in zip(targets, node.outputs, strict=True):
            output_type = _effective_graph_value_type(output)
            _emit_bind_nested_shape_symbols(
                lines,
                add=self._add,
                type_expr=output_type,
                value_expr=target,
                local=inline_local,
                protected=self.global_symbol_names,
                required_names=set(),
                indent=indent,
            )
        local.update(inline_local)
        return True

    def _node_expr(self, node: Any, *, local: set[str], symbols_dict: str) -> str:
        op = node.op.name
        if op == "core.alias":
            return self._operand_expr(node.inputs[0], local=local, symbols_dict=symbols_dict)
        if op == "core.tuple":
            return "(" + ", ".join(self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs) + ")"
        if op == "core.list":
            items = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
            if _is_int_dim_list_type(getattr(node, "type_expr", None)):
                return _tuple_literal_expr(items)
            return "[" + ", ".join(items) + "]"
        if op == "core.ascribe":
            return self._operand_expr(node.inputs[0], local=local, symbols_dict=symbols_dict)
        if op == "core.select":
            cond = self._operand_expr(node.inputs[0], local=local, symbols_dict=symbols_dict)
            yes = self._operand_expr(node.inputs[1], local=local, symbols_dict=symbols_dict)
            no = self._operand_expr(node.inputs[2], local=local, symbols_dict=symbols_dict)
            cond_expr = cond if self._operand_is_bool(node.inputs[0]) else f"bool({cond})"
            return f"({yes} if {cond_expr} else {no})"
        if op.startswith("core.binary."):
            return self._binary_expr(
                op.removeprefix("core.binary."),
                node.inputs[0],
                node.inputs[1],
                result_type=getattr(node, "type_expr", None),
                local=local,
                symbols_dict=symbols_dict,
            )
        if op in self.method_names:
            if op in self.global_symbol_names and not node.inputs and not node.attrs:
                return f"{symbols_dict}[{op!r}]"
            args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
            args.extend(f"{key}={self._operand_expr(value, local=local, symbols_dict=symbols_dict)}" for key, value in node.attrs.items())
            call = f"self.{self.method_names[op]}({', '.join(args)})"
            module = self.modules_by_name[op]
            return f"{call}[0]" if len(module.outputs) == 1 else call
        primitive = _normalize_primitive_op(op)
        return self._primitive_expr(primitive, node, local=local, symbols_dict=symbols_dict)

    def _assign_slice_inplace_allowed(self, node: Any) -> bool:
        if node.id in self.ownership.inplace_assign_slice_node_ids:
            return True
        # Codegen may inline ordinary Axon definitions after ownership has run.
        # In that case the primitive node id is rewritten.  If all original
        # occurrences of the same primitive op were proven owned, synthesized
        # inline copies inherit the same lowering.
        if node.op.name in self.inplace_owned_primitive_ops:
            return True
        source_module = getattr(node, "source_module", None)
        if not source_module:
            return False
        source_node_id = f"{source_module}:1"
        return source_node_id in self.ownership.inplace_assign_slice_node_ids

    def _operand_may_be_tensor(self, operand: GraphOperand) -> bool:
        type_expr = getattr(operand, "type_expr", None)
        if isinstance(type_expr, TypeOptional):
            type_expr = type_expr.inner
        return isinstance(type_expr, TypeTensor | TypeAny)

    def _operand_is_bool(self, operand: GraphOperand) -> bool:
        type_expr = getattr(operand, "type_expr", None)
        if isinstance(type_expr, TypeOptional):
            type_expr = type_expr.inner
        return isinstance(type_expr, TypeBool) or (
            isinstance(operand, GraphLiteral) and type(operand.value) is bool
        )

    def _scalar_operand_expr(
        self,
        operand: GraphOperand,
        *,
        local: set[str],
        symbols_dict: str,
        expected: tuple[type, ...],
        cast: str,
    ) -> str:
        expr = self._operand_expr(operand, local=local, symbols_dict=symbols_dict)
        type_expr = getattr(operand, "type_expr", None)
        if isinstance(type_expr, TypeOptional):
            type_expr = type_expr.inner
        if isinstance(type_expr, expected):
            return expr
        if isinstance(operand, GraphLiteral):
            if TypeBool in expected and type(operand.value) is bool:
                return expr
            if (TypeInt in expected or TypeDim in expected) and type(operand.value) is int:
                return expr
            if TypeFloat in expected and type(operand.value) is float:
                return expr
        return f"{cast}({expr})"

    def _binary_expr(
        self,
        op: str,
        left_operand: GraphOperand,
        right_operand: GraphOperand,
        *,
        result_type: Any | None = None,
        local: set[str],
        symbols_dict: str,
    ) -> str:
        left = self._operand_expr(left_operand, local=local, symbols_dict=symbols_dict)
        right = self._operand_expr(right_operand, local=local, symbols_dict=symbols_dict)
        if isinstance(left_operand, GraphExpr):
            left = f"({left})"
        if isinstance(right_operand, GraphExpr):
            right = f"({right})"
        left_is_null = isinstance(left_operand, GraphLiteral) and left_operand.value is None
        right_is_null = isinstance(right_operand, GraphLiteral) and right_operand.value is None
        if op in {"==", "!="} and (left_is_null or right_is_null):
            other = right if left_is_null else left
            identity_op = "is" if op == "==" else "is not"
            return f"({other} {identity_op} None)"
        if op == "+" and left_is_null:
            return right
        if op == "+" and right_is_null:
            return left
        if op == "-" and right_is_null:
            return left
        pyop = {"and": "&", "or": "|"}.get(op, op)
        if op == "/" and isinstance(result_type, TypeDim | TypeInt):
            pyop = "//"
        if self.align_devices and (
            self._operand_may_be_tensor(left_operand) or self._operand_may_be_tensor(right_operand)
        ):
            return f"(lambda _a, _b: (_a {pyop} _b))(*self._align_pair({left}, {right}, prefer='right'))"
        return f"({left} {pyop} {right})"

    def _primitive_expr(self, primitive: str, node: Any, *, local: set[str], symbols_dict: str) -> str:
        args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
        attrs = {k: self._operand_expr(v, local=local, symbols_dict=symbols_dict) for k, v in node.attrs.items()}
        def tuple_int_arg(index: int, fallback: str | None = None) -> str:
            if index < len(node.inputs):
                folded = self._static_int_tuple_expr(
                    node.inputs[index],
                    local=local,
                    symbols_dict=symbols_dict,
                )
                if folded is not None:
                    return folded
                return args[index]
            if fallback is None:
                raise ValueError(f"{primitive} missing tuple/list argument {index}")
            return fallback

        def scalar_arg(
            index: int,
            fallback: str | None = None,
            *,
            expected: tuple[type, ...],
            cast: str,
        ) -> str:
            if index < len(node.inputs):
                return self._scalar_operand_expr(
                    node.inputs[index],
                    local=local,
                    symbols_dict=symbols_dict,
                    expected=expected,
                    cast=cast,
                )
            if fallback is None:
                raise ValueError(f"{primitive} missing scalar argument {index}")
            return fallback

        def int_arg(index: int, fallback: str | None = None) -> str:
            return scalar_arg(index, fallback, expected=(TypeInt, TypeDim), cast="int")

        def bool_arg(index: int, fallback: str | None = None) -> str:
            return scalar_arg(index, fallback, expected=(TypeBool,), cast="bool")

        def float_arg(index: int, fallback: str | None = None) -> str:
            return scalar_arg(index, fallback, expected=(TypeFloat, TypeInt, TypeDim), cast="float")

        def optional_input_expr(index: int, fallback: str = "None") -> str:
            if index >= len(node.inputs):
                return fallback
            operand = node.inputs[index]
            if isinstance(operand, GraphLiteral) and operand.value is None:
                return "None"
            return self._operand_expr(operand, local=local, symbols_dict=symbols_dict)

        def optional_int_arg(index: int, fallback: str) -> str:
            if index >= len(node.inputs):
                return fallback
            operand = node.inputs[index]
            if isinstance(operand, GraphLiteral) and operand.value is None:
                return fallback
            if _operand_may_be_none(operand) or isinstance(operand, GraphValueRef):
                raw = self._operand_expr(operand, local=local, symbols_dict=symbols_dict)
                return f"({fallback} if {raw} is None else int({raw}))"
            expr = int_arg(index)
            if (
                isinstance(operand, GraphValueRef)
                and operand.name not in local
                and operand.name not in self.global_symbol_names
            ):
                return fallback
            return expr

        def dtype_expr(index: int, ref_expr: str, fallback: str = "None") -> str:
            dtype_value = optional_input_expr(index, fallback)
            if dtype_value == "None":
                return f"{ref_expr}.dtype"
            return f"(self._dtype_from_name({dtype_value}) or {ref_expr}.dtype)"

        binary_primitives = {
            "le": "<=",
            "eq": "==",
            "add": "+",
            "mul": "*",
            "div": "/",
        }
        if primitive == "_torch_sdpa":
            if len(args) < 6:
                raise ValueError("__torch_sdpa expects q, k, v, additive_mask, scale, enable_gqa")
            scale = "None" if isinstance(node.inputs[4], GraphLiteral) and node.inputs[4].value is None else f"({args[4]} if {args[4]} is None else float({args[4]}))"
            return (
                f"(lambda _q, _k, _v, _mask: F.scaled_dot_product_attention("
                f"_q, self._move_to(_k, _q.device), self._move_to(_v, _q.device), "
                f"attn_mask=self._move_to(_mask, _q.device), dropout_p=0.0, is_causal=False, "
                f"scale={scale}, "
                f"enable_gqa={bool_arg(5)}))({args[0]}, {args[1]}, {args[2]}, {args[3]})"
            )
        if primitive == "_torch_rope_apply_factors":
            if len(args) < 4:
                raise ValueError("__torch_rope_apply_factors expects x, sin, cos, interleaved")
            return f"self._rope_apply_factors({args[0]}, {args[1]}, {args[2]}, {bool_arg(3)})"
        if primitive == "_torch_rope_pair_apply_factors":
            if len(args) < 5:
                raise ValueError("__torch_rope_pair_apply_factors expects q, k, sin, cos, interleaved")
            interleaved = bool_arg(4)
            return (
                f"(self._rope_apply_factors({args[0]}, {args[2]}, {args[3]}, {interleaved}), "
                f"self._rope_apply_factors({args[1]}, {args[2]}, {args[3]}, {interleaved}))"
            )
        if primitive == "assign_slice":
            if len(args) < 5:
                raise ValueError("_assign_slice expects x, src, dim, start, end")
            if self._assign_slice_inplace_allowed(node):
                return f"self._assign_slice_inplace({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]})"
            return f"self._assign_slice({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]})"
        if primitive == "_torch_gate_up_linear_pair":
            if len(args) < 7:
                raise ValueError("__torch_gate_up_linear_pair expects input, gate/up weight paths, gate/up bias paths, bias, and transpose")
            return (
                f"self._gate_up_linear_pair({args[0]}, {args[1]}, {args[2]}, "
                f"gate_bias_path={args[3]}, up_bias_path={args[4]}, "
                f"bias={bool_arg(5)}, transpose={bool_arg(6)})"
            )
        if primitive == "_torch_swiglu_ffn":
            if len(args) < 7:
                raise ValueError("__torch_swiglu_ffn expects input, gate/up/down weight paths, and gate/up/down bias paths")
            return (
                f"self._swiglu_ffn({args[0]}, {args[1]}, {args[2]}, {args[3]}, "
                f"gate_bias_path={args[4]}, up_bias_path={args[5]}, down_bias_path={args[6]})"
            )
        if primitive == "_torch_expert_swiglu_ffn":
            if len(args) < 5:
                raise ValueError("__torch_expert_swiglu_ffn expects input, expert indices, and gate/up/down weight paths")
            return f"self._expert_swiglu_ffn({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]})"
        if primitive == "_torch_expert_packed_swiglu_ffn":
            if len(args) < 5:
                raise ValueError("__torch_expert_packed_swiglu_ffn expects input, expert indices, gate-up/down weight paths, and transpose")
            return f"self._expert_packed_swiglu_ffn({args[0]}, {args[1]}, {args[2]}, {args[3]}, transpose={bool_arg(4)})"
        if primitive == "_torch_selected_expert_packed_swiglu_ffn":
            if len(args) < 6:
                raise ValueError("__torch_selected_expert_packed_swiglu_ffn expects input, top-k scores/indices, gate-up/down weight paths, and transpose")
            return (
                f"self._selected_expert_packed_swiglu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, transpose={bool_arg(5)})"
            )
        if primitive == "_torch_selected_expert_swiglu_ffn":
            if len(args) < 7:
                raise ValueError("__torch_selected_expert_swiglu_ffn expects input, top-k scores/indices, gate/up/down weight paths, and transpose")
            return (
                f"self._selected_expert_swiglu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, {args[5]}, transpose={bool_arg(6)})"
            )
        if primitive == "_torch_selected_expert_packed_gegelu_ffn":
            if len(args) < 10:
                raise ValueError("__torch_selected_expert_packed_gegelu_ffn expects input, top-k scores/indices, gate-up/down weight/bias paths, limit, optional alpha, bias, and transpose")
            alpha_arg = args[8] if len(args) >= 11 else "1.702"
            bias_idx = 9 if len(args) >= 11 else 8
            transpose_idx = 10 if len(args) >= 11 else 9
            return (
                f"self._selected_expert_packed_gegelu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, {args[5]}, {args[6]}, {args[7]}, "
                f"alpha={alpha_arg}, bias={bool_arg(bias_idx)}, transpose={bool_arg(transpose_idx)})"
            )
        if primitive == "_torch_selected_expert_clamped_packed_swiglu_ffn":
            if len(args) < 7:
                raise ValueError("__torch_selected_expert_clamped_packed_swiglu_ffn expects input, top-k scores/indices, gate-up/down weight paths, limit, and transpose")
            return (
                f"self._selected_expert_clamped_packed_swiglu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, {args[5]}, transpose={bool_arg(6)})"
            )
        if primitive == "_torch_selected_expert_relu2_ffn":
            if len(args) < 6:
                raise ValueError("__torch_selected_expert_relu2_ffn expects input, top-k scores/indices, up/down weight paths, and transpose")
            return (
                f"self._selected_expert_relu2_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, transpose={bool_arg(5)})"
            )
        if primitive == "_torch_weighted_topk_sum":
            if len(args) < 2:
                raise ValueError("__torch_weighted_topk_sum expects expert values and top-k scores")
            return f"torch.sum({args[0]} * torch.unsqueeze({args[1]}.to(device={args[0]}.device, dtype={args[0]}.dtype), -1), dim=2, keepdim=False)"
        if primitive == "_torch_topk_normalize":
            if len(args) < 2:
                raise ValueError("__torch_topk_normalize expects top-k weights and a dtype reference")
            normalized = f"({args[0]} / torch.sum({args[0]}, dim=-1, keepdim=True))"
            return f"{normalized}.to(device={args[1]}.device, dtype={args[1]}.dtype)"
        if primitive in binary_primitives and len(node.inputs) >= 2:
            return self._binary_expr(
                binary_primitives[primitive],
                node.inputs[0],
                node.inputs[1],
                result_type=getattr(node, "type_expr", None),
                local=local,
                symbols_dict=symbols_dict,
            )
        if primitive == "tensor_size":
            return f"{args[0]}.shape[{int_arg(1)}]"
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
            static_weight_key = self._static_param_key(node.inputs[0], "weight")
            if static_weight_key is not None:
                weight = f"self.state_dict_tensors[{static_weight_key!r}]"
                x = f"self._move_to({args[1]}, {weight}.device)" if self.align_devices else args[1]
                return f"F.embedding({x}, {weight})"
            weight = self._param_expr_for_path(node.inputs[0], "weight", local=local, symbols_dict=symbols_dict)
            if self.align_devices:
                return f"(lambda _w: F.embedding(self._move_to({args[1]}, _w.device), _w))({weight})"
            return f"F.embedding({args[1]}, {weight})"
        if primitive == "linear":
            bias = bool_arg(3, "False")
            transpose = bool_arg(4, "False")
            expert = args[5] if len(args) > 5 else "None"
            weight_leaf = args[6] if len(args) > 6 else "'weight'"
            bias_leaf = args[7] if len(args) > 7 else "'bias'"
            return f"self._linear({args[0]}, {args[1]}, bias={bias}, transpose={transpose}, expert=({expert} if {expert} is not None else None), weight_leaf={weight_leaf}, bias_leaf={bias_leaf})"
        if primitive == "expert_linear":
            bias = bool_arg(4, "False")
            transpose = bool_arg(5, "False")
            weight_leaf = args[6] if len(args) > 6 else "'weight'"
            bias_leaf = args[7] if len(args) > 7 else "'bias'"
            return f"self._expert_linear({args[0]}, {args[1]}, {args[2]}, bias={bias}, transpose={transpose}, weight_leaf={weight_leaf}, bias_leaf={bias_leaf})"
        if primitive == "layernorm":
            eps = float_arg(2, "1e-5")
            weight_leaf = args[4] if len(args) > 4 else "'weight'"
            bias = args[5] if len(args) > 5 else "True"
            bias_leaf = args[6] if len(args) > 6 else "'bias'"
            weight = self._param_expr_for_path(
                node.inputs[0],
                node.inputs[4] if len(node.inputs) > 4 else "weight",
                local=local,
                symbols_dict=symbols_dict,
            )
            bias_value = self._param_expr_for_path(
                node.inputs[0],
                node.inputs[6] if len(node.inputs) > 6 else "bias",
                optional=True,
                local=local,
                symbols_dict=symbols_dict,
            )
            bias_expr = f"({bias_value} if {bias} else None)"
            x = f"self._move_to({args[1]}, {weight}.device)" if self.align_devices else args[1]
            moved_bias = (
                f"(self._move_to({bias_expr}, {weight}.device) if {bias_expr} is not None else None)"
                if self.align_devices
                else bias_expr
            )
            return f"F.layer_norm({x}, ({args[1]}.shape[-1],), weight={weight}, bias={moved_bias}, eps={eps})"
        if primitive == "rmsnorm":
            x = args[0]
            eps = float_arg(1, "1e-6")
            cast_float = bool_arg(3, "False")
            x_float = f"{x}.float()"
            y_float = f"({x_float} * torch.rsqrt(torch.mean({x_float} * {x_float}, dim=-1, keepdim=True) + {eps}))"
            y = f"({x} * torch.rsqrt(torch.mean({x} * {x}, dim=-1, keepdim=True) + {eps}))"
            return f"({y_float}.to(dtype={x}.dtype) if {cast_float} else {y})"
        if primitive == "conv1d":
            return f"self._conv1d({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, {args[5]}, {args[6]}, {args[7]})"
        if primitive == "tensor_like":
            dtype = args[2] if len(args) > 2 else "None"
            target_dtype = f"({args[1]}.dtype if self._dtype_from_name({dtype}) is None else self._dtype_from_name({dtype}))"
            return f"self._tensor_like({args[0]}, {args[1]}, {target_dtype})"
        if primitive == "where_indices":
            return f"torch.where({args[0]})"
        if primitive == "topk":
            return f"torch.topk({args[0]}, {int_arg(1)}, dim={int_arg(2)}, largest={bool_arg(3)}, sorted={bool_arg(4)})"
        if primitive == "concat":
            if "dim" in attrs:
                if self.align_devices:
                    return f"self._concat({', '.join(args)}, dim={attrs['dim']})"
                return f"torch.cat([{', '.join(args)}], dim={attrs['dim']})"
            if not args:
                raise ValueError("concat requires at least one argument")
            if self.align_devices:
                return f"self._concat({', '.join(args[:-1])}, dim={args[-1]})"
            return f"torch.cat([{', '.join(args[:-1])}], dim={args[-1]})"
        if primitive == "clamp":
            min_value = optional_input_expr(1, attrs.get("min", "None"))
            max_value = optional_input_expr(2, attrs.get("max", "None"))
            return f"torch.clamp({args[0]}, min={min_value}, max={max_value})"
        simple = {
            "reshape": lambda: f"torch.reshape({args[0]}, {tuple_int_arg(1)})",
            "arange": lambda: f"torch.arange({int_arg(1)}, {optional_int_arg(2, f'({args[0]}.shape[-2] if {args[0]}.ndim >= 3 else {args[0]}.shape[-1])')}, device={args[0]}.device, dtype=torch.long)",
            "slice": lambda: f"torch.narrow({args[0]}, {int_arg(1)}, {int_arg(2)}, {int_arg(3)} - {int_arg(2)})",
            "chunk": lambda: f"torch.chunk({args[0]}, {int_arg(2, attrs.get('parts', '1'))}, dim={int_arg(1, attrs.get('dim', '-1'))})",
            "split": lambda: f"torch.split({args[0]}, {tuple_int_arg(2, attrs.get('sizes', '()'))}, dim={int_arg(1, attrs.get('dim', '-1'))})",
            "sum": lambda: f"torch.sum({args[0]}, dim={int_arg(1, '-1')}, keepdim={bool_arg(2, 'False')})",
            "expand": lambda: f"{args[0]}.expand({tuple_int_arg(1)})",
            "permute": lambda: f"torch.permute({args[0]}, {tuple_int_arg(1)})",
            "transpose": lambda: f"torch.transpose({args[0]}, {int_arg(1)}, {int_arg(2)})",
            "unsqueeze": lambda: f"torch.unsqueeze({args[0]}, {int_arg(1)})",
            "repeat": lambda: f"({args[0]} if {int_arg(1)} == 1 else torch.repeat_interleave({args[0]}, repeats={int_arg(1)}, dim=({int_arg(2)} if {int_arg(2)} >= 0 else {int_arg(2)} + {args[0]}.dim())))",
            "matmul": lambda: (
                f"(lambda _a, _b: torch.matmul(_a, _b))(*self._align_pair({args[0]}, {args[1]}, prefer='right'))"
                if self.align_devices
                else f"torch.matmul({args[0]}, {args[1]})"
            ),
            "softmax": lambda: f"F.softmax({args[0]}, dim={int_arg(1, '-1')})",
            "where": lambda: (
                f"self._where({args[0]}, {args[1]}, {args[2]})"
                if self.align_devices
                else f"torch.where({args[0]}, {args[1]}, {args[2]})"
            ),
            "require": lambda: f"self._require_value({args[0]})",
            "gather": lambda: f"torch.gather({args[0]}, dim={int_arg(2, '-1')}, index=self._move_to({args[1]}, {args[0]}.device))",
            "scatter": lambda: f"self._scatter({args[0]}, {args[1]}, {args[2]}, {int_arg(3, '-1')})",
            "assign_slice": lambda: f"self._assign_slice({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]})",
            "index_add": lambda: f"torch.index_add({args[0]}, dim={int_arg(3, '0')}, index=self._move_to({args[1]}, {args[0]}.device), source=self._move_to({args[2]}, {args[0]}.device))",
            "and": lambda: (
                f"(lambda _a, _b: torch.logical_and(_a, _b))(*self._align_pair({args[0]}, {args[1]}, prefer='right'))"
                if self.align_devices
                else f"torch.logical_and({args[0]}, {args[1]})"
            ),
            "pow": lambda: f"(torch.pow({args[0]}, {args[1]}) if torch.is_tensor({args[0]}) else ({args[0]} ** {args[1]}))",
            "floor": lambda: f"torch.floor({args[0]}) if torch.is_tensor({args[0]}) else int({args[0]} // 1)",
            "sqrt": lambda: f"torch.sqrt({args[0]}) if torch.is_tensor({args[0]}) else ({args[0]} ** 0.5)",
            "sin": lambda: f"torch.sin({args[0]}) if torch.is_tensor({args[0]}) else __import__('math').sin({float_arg(0)})",
            "cos": lambda: f"torch.cos({args[0]}) if torch.is_tensor({args[0]}) else __import__('math').cos({float_arg(0)})",
            "exp": lambda: f"torch.exp({args[0]}) if torch.is_tensor({args[0]}) else __import__('math').exp({float_arg(0)})",
            "log": lambda: f"(torch.log({args[0]}) if torch.is_tensor({args[0]}) else __import__('math').log(float({args[0]})))",
            "cast": lambda: f"{args[0]}.to(dtype=getattr(torch, str({args[1]})))",
            "cast_like": lambda: f"{args[0]}.to(device={args[1]}.device, dtype={args[1]}.dtype)",
            "dtype_value": lambda: f"{{'min': torch.finfo({args[0]}.dtype).min, 'max': torch.finfo({args[0]}.dtype).max, 'eps': torch.finfo({args[0]}.dtype).eps, 'tiny': torch.finfo({args[0]}.dtype).tiny, 'inf': float('inf'), '-inf': float('-inf')}}[str({args[1]})]",
            "cumsum": lambda: f"torch.cumsum({args[0]}, dim={int_arg(1, '-1')})",
            "empty_like": lambda: f"torch.empty_like({args[0]})",
            "fill": lambda: (
                f"torch.full_like({args[0]}, {args[1]})"
                if len(args) < 3 or (
                    isinstance(node.inputs[2], GraphLiteral)
                    and node.inputs[2].value is None
                )
                else f"torch.full_like({args[0]}, {args[1]}, dtype=self._dtype_from_name({args[2]}))"
            ),
            "empty": lambda: f"torch.empty({tuple_int_arg(1)}, device={args[0]}.device, dtype={dtype_expr(2, args[0])})",
            "zeros": lambda: f"torch.zeros({tuple_int_arg(1)}, device={args[0]}.device, dtype={dtype_expr(2, args[0])})",
            "full": lambda: f"torch.full({tuple_int_arg(1)}, {args[2]}, device={args[0]}.device, dtype={dtype_expr(3, args[0])})",
            "zeros_like": lambda: f"torch.zeros_like({args[0]})",
            "activations_tanh": lambda: f"torch.tanh({args[0]})",
            "activations_silu": lambda: f"F.silu({args[0]})",
            "activations_sigmoid": lambda: f"torch.sigmoid({args[0]})",
            "l2norm": lambda: f"(({args[0]}.float() * torch.pow(torch.mean({args[0]}.float() * {args[0]}.float(), dim=-1, keepdim=True) + {float_arg(1, '1e-6')}, -0.5)).to(dtype={args[0]}.dtype))",
            "activations_relu": lambda: f"F.relu({args[0]})",
            "activations_relu2": lambda: f"(F.relu({args[0]}) * F.relu({args[0]}))",
            "activations_gelu": lambda: f"F.gelu({args[0]})",
            "activations_gelu_new": lambda: f"(0.5 * {args[0]} * (1.0 + torch.tanh(0.7978845608028654 * ({args[0]} + 0.044715 * {args[0]} * {args[0]} * {args[0]}))))",
            "activations_gelu_pytorch_tanh": lambda: f"(0.5 * {args[0]} * (1.0 + torch.tanh(0.7978845608028654 * ({args[0]} + 0.044715 * {args[0]} * {args[0]} * {args[0]}))))",
            "activations_gegelu": lambda: f"self._gegelu({args[0]}, {args[1] if len(args) > 1 else 'None'})",
            "activations_xielu": lambda: f"self._xielu({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]})",
            "list_init": lambda: "[]",
            "list_append": lambda: f"([*({args[0]} or []), {args[1]}])",
            "list_index": lambda: f"{args[0]}[{int_arg(1)}]",
            "list_length": lambda: f"len({args[0]} or [])",
            "shape": lambda: f"list({args[0]}.shape)",
        }
        if primitive in simple:
            return simple[primitive]()
        raise NotImplementedError(f"direct codegen2-torch unsupported graph op {primitive!r}")

    def _static_int_tuple_expr(
        self,
        operand: GraphOperand,
        *,
        local: set[str],
        symbols_dict: str,
    ) -> str | None:
        if not (
            isinstance(operand, GraphExpr)
            and operand.op.name == "core.list"
            and not operand.attrs
            and _is_int_dim_list_type(operand.type_expr)
        ):
            return None
        items: list[str] = []
        for item in operand.inputs:
            expr: str | None = None
            if isinstance(item, GraphLiteral) and type(item.value) is int:
                expr = repr(item.value)
            elif isinstance(item, GraphValueRef) and isinstance(item.type_expr, TypeDim | TypeInt):
                expr = self._operand_expr(item, local=local, symbols_dict=symbols_dict)
            elif isinstance(item, GraphExpr) and isinstance(item.type_expr, TypeDim | TypeInt):
                expr = self._operand_expr(item, local=local, symbols_dict=symbols_dict)
            if expr is None:
                return None
            items.append(expr)
        return _tuple_literal_expr(items)

    def _static_path_key(self, operand: Any) -> str | None:
        if not isinstance(operand, GraphPath) or not operand.absolute:
            return None
        if any("{" in part or "}" in part for part in operand.parts):
            return None
        return ".".join(part for part in operand.parts if part)

    def _literal_leaf_key(self, operand: Any) -> str | None:
        if isinstance(operand, str):
            return operand
        if isinstance(operand, GraphLiteral) and isinstance(operand.value, str):
            return operand.value
        if isinstance(operand, GraphPath) and not any("{" in part or "}" in part for part in operand.parts):
            prefix = "@@" if operand.absolute else "@"
            return prefix + ".".join(part for part in operand.parts if part)
        return None

    def _static_param_key(self, base: Any, leaf: Any) -> str | None:
        leaf_key = self._literal_leaf_key(leaf)
        if leaf_key is None:
            return None
        if leaf_key.startswith("@@"):
            return leaf_key[2:].strip(".")
        base_key = self._static_path_key(base)
        if base_key is None:
            return None
        leaf_key = leaf_key[1:].strip(".") if leaf_key.startswith("@") else leaf_key.strip(".")
        if not leaf_key:
            return base_key
        if leaf_key == base_key or leaf_key.startswith(base_key + "."):
            return leaf_key
        return f"{base_key}.{leaf_key}"

    def _param_expr_for_path(
        self,
        base: Any,
        leaf: Any,
        *,
        optional: bool = False,
        local: set[str],
        symbols_dict: str,
    ) -> str:
        key = self._static_param_key(base, leaf)
        if key is not None:
            return f"self.state_dict_tensors.get({key!r})" if optional else f"self.state_dict_tensors[{key!r}]"
        key_expr = self._param_key_expr(base, leaf, local=local, symbols_dict=symbols_dict)
        if key_expr is not None:
            getter = "_optional_param" if optional else "_param"
            return f"self.{getter}({key_expr})"
        base_expr = self._operand_expr(base, local=local, symbols_dict=symbols_dict)
        leaf_expr = repr(leaf) if isinstance(leaf, str) else self._operand_expr(leaf, local=local, symbols_dict=symbols_dict)
        getter = "_optional_param" if optional else "_param"
        return f"self.{getter}(self._compose_path({base_expr}, {leaf_expr}))"

    def _param_key_expr(
        self,
        base: Any,
        leaf: Any,
        *,
        local: set[str],
        symbols_dict: str,
    ) -> str | None:
        if isinstance(leaf, GraphPath) and leaf.absolute:
            return self._path_key_expr(leaf, local=local, symbols_dict=symbols_dict)
        if isinstance(base, GraphPath) and base.absolute and isinstance(leaf, GraphPath) and not leaf.absolute:
            base_key = self._path_key_expr(base, local=local, symbols_dict=symbols_dict)
            leaf_key = self._path_key_expr(leaf, local=local, symbols_dict=symbols_dict)
            if base_key is not None and leaf_key is not None:
                return f"({base_key} + ('.' + {leaf_key} if {leaf_key} else ''))"
        return None

    def _dim_token_expr(self, dim: Any, *, local: set[str], symbols_dict: str) -> str | None:
        if isinstance(dim, bool):
            return repr(int(dim))
        if isinstance(dim, int):
            return repr(dim)
        if isinstance(dim, str):
            if dim.startswith(".."):
                return None
            name = _dim_ident(dim)
            if dim in local:
                return name
            if dim in self.global_symbol_names:
                return f"{symbols_dict}[{dim!r}]"
            return None
        if isinstance(dim, DimExprBinary):
            left = self._dim_token_expr(dim.left, local=local, symbols_dict=symbols_dict)
            right = self._dim_token_expr(dim.right, local=local, symbols_dict=symbols_dict)
            if left is None or right is None:
                return None
            op = "//" if dim.op == "/" else dim.op
            return f"({left} {op} {right})"
        return repr(dim)

    def _operand_expr(self, operand: GraphOperand, *, local: set[str], symbols_dict: str) -> str:
        if isinstance(operand, GraphValueRef):
            name = _dim_ident(operand.name) if isinstance(operand.type_expr, TypeDim) else _py_ident(operand.name)
            if operand.name in local:
                return name
            if operand.name in self.global_symbol_names:
                return f"{symbols_dict}[{operand.name!r}]"
            if isinstance(operand.type_expr, TypeDim):
                return name
            return f"{symbols_dict}[{operand.name!r}]"
        if isinstance(operand, GraphLiteral):
            return repr(operand.value)
        if isinstance(operand, GraphPath):
            return self._path_expr(operand, local=local, symbols_dict=symbols_dict)
        if isinstance(operand, GraphExpr):
            if (
                operand.op.name in self.global_symbol_names
                and not operand.inputs
                and not operand.attrs
            ):
                return f"{symbols_dict}[{operand.op.name!r}]"
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
            pseudo = type(
                "_Node",
                (),
                {
                    "op": operand.op,
                    "inputs": operand.inputs,
                    "attrs": operand.attrs,
                    "outputs": (),
                    "type_expr": operand.type_expr,
                },
            )()
            return self._node_expr(pseudo, local=local, symbols_dict=symbols_dict)
        if isinstance(operand, tuple):
            return "[" + ", ".join(self._operand_expr(item, local=local, symbols_dict=symbols_dict) for item in operand) + "]"
        raise TypeError(f"unsupported graph operand: {operand!r}")

    def _path_expr(self, path: GraphPath, *, local: set[str], symbols_dict: str) -> str:
        prefix = "@@" if path.absolute else "@"
        key_expr = self._path_key_expr(path, local=local, symbols_dict=symbols_dict)
        if key_expr is None:
            return repr(prefix)
        return f"({prefix!r} + {key_expr})"

    def _path_key_expr(self, path: GraphPath, *, local: set[str], symbols_dict: str) -> str | None:
        template = ".".join(part for part in path.parts if part)
        if "{" not in template and "}" not in template:
            return repr(template)
        pieces: list[str] = []
        cursor = 0
        for match in re.finditer(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", template):
            pieces.append(template[cursor:match.start()].replace("{", "{{").replace("}", "}}"))
            name = match.group(1)
            expr = _py_ident(name) if name in local else f"{symbols_dict}[{name!r}]"
            pieces.append("{self._path_template_part(" + expr + ")}")
            cursor = match.end()
        pieces.append(template[cursor:].replace("{", "{{").replace("}", "}}"))
        return "f" + repr("".join(pieces))


def emit_model_code_from_graph_ir(
    program: GraphProgram,
    *,
    class_name: str = "GeneratedAxonModel",
    model_config: dict[str, Any] | None = None,
    profile: bool = False,
    align_devices: bool = False,
) -> str:
    """Emit direct Python/PyTorch model code from graph IR."""
    validate_graph_program(program)
    emitter = _DirectTorchEmitter(
        program=program,
        class_name=class_name,
        profile=profile,
        align_devices=align_devices,
    )
    body = emitter.emit()
    header = [
        "from __future__ import annotations",
        "",
    ]
    if profile:
        header.append("import time")
    header.extend(
        [
            "import torch",
            "from torch import nn",
            "from torch.nn import functional as F",
            "from brainsurgery.synapse.axon.codegen2_torch.core import _materialize_joined_parameter, _materialize_packed_parameters",
            "from brainsurgery.synapse.axon.codegen2_common import (",
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
    return "\n".join(header)


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
    "graph_main_output_names",
    "make_graph_model_class",
    "make_runtime2_model_class",
]
