from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..ast import TypePath
from ..graph_ir.core import (
    GraphExpr,
    GraphLiteral,
    GraphModule,
    GraphNode,
    GraphOperand,
    GraphPath,
    GraphProgram,
    GraphValue,
    GraphValueRef,
)
from ..graph_ir.optimize import graph_provenance_facts
from ..graph_ir.provenance import (
    GraphProvenanceAnalysis,
    GraphSdpaGqaFact,
    infer_graph_provenance,
)


class VLLMLayerType(Enum):
    VOCAB_PARALLEL_EMBEDDING = "vocab_parallel_embedding"
    QKV_PARALLEL_LINEAR = "qkv_parallel_linear"
    MERGED_COLUMN_PARALLEL_LINEAR = "merged_column_parallel_linear"
    COLUMN_PARALLEL_LINEAR = "column_parallel_linear"
    ROW_PARALLEL_LINEAR = "row_parallel_linear"
    PARALLEL_LM_HEAD = "parallel_lm_head"
    RMSNORM = "rmsnorm"
    LAYERNORM = "layernorm"
    ATTENTION = "attention"
    MAMBA_MIXER = "mamba_mixer"
    DEFAULT = "default"


@dataclass(frozen=True)
class QKVGroup:
    q_node_id: str
    k_node_id: str
    v_node_id: str
    attention_node_id: str | None = None
    layout: str = "packed"


@dataclass(frozen=True)
class FFNGroup:
    gate_node_id: str | None = None
    up_node_id: str | None = None
    down_node_id: str | None = None
    gate_up_intrinsic_node_id: str | None = None


@dataclass
class VLLMLayerClassification:
    node_types: dict[str, VLLMLayerType] = field(default_factory=dict)
    qkv_groups: list[QKVGroup] = field(default_factory=list)
    ffn_groups: list[FFNGroup] = field(default_factory=list)
    attention_node_ids: set[str] = field(default_factory=set)
    embedding_node_ids: set[str] = field(default_factory=set)
    token_embedding_node_ids: set[str] = field(default_factory=set)
    position_embedding_node_ids: set[str] = field(default_factory=set)
    lm_head_node_id: str | None = None
    rmsnorm_node_ids: set[str] = field(default_factory=set)
    qk_norm_node_ids: set[str] = field(default_factory=set)
    q_norm_node_ids: set[str] = field(default_factory=set)
    k_norm_node_ids: set[str] = field(default_factory=set)
    repeated_module_names: set[str] = field(default_factory=set)
    loop_index_param: dict[str, str] = field(default_factory=dict)
    module_scope_parts: dict[str, tuple[str, ...]] = field(default_factory=dict)
    per_layer_scalar_node_id: str | None = None
    per_layer_scalar_has_residual_add: bool = False
    logit_softcap: float | None = None
    o_proj_node_ids: set[str] = field(default_factory=set)
    v_norm_node_ids: set[str] = field(default_factory=set)
    has_k_eq_v: bool = False
    pli_gate_node_id: str | None = None
    pli_proj_node_id: str | None = None
    pli_norm_node_id: str | None = None
    pli_embed_node_id: str | None = None
    pli_model_proj_node_id: str | None = None
    pli_proj_norm_node_id: str | None = None
    mamba_mixer_module_names: set[str] = field(default_factory=set)

    def layer_type(self, node: GraphNode) -> VLLMLayerType:
        return self.node_types.get(node.id, VLLMLayerType.DEFAULT)


def _discover_activation_primitives(program: GraphProgram) -> frozenset[str]:
    primitives: set[str] = set()
    for module in program.modules:
        for node in module.nodes:
            if node.op.name.startswith("_activations_"):
                primitives.add(node.op.name)
    return frozenset(primitives)


_SELECTED_EXPERT_INTRINSICS = frozenset(
    {
        "__vllm_selected_expert_clamped_packed_swiglu_ffn",
        "__vllm_selected_expert_packed_gegelu_ffn",
        "__vllm_selected_expert_packed_swiglu_ffn",
        "__vllm_selected_expert_relu2_ffn",
        "__vllm_selected_expert_swiglu_ffn",
    }
)

_TRIVIAL_TRANSFORM_OPS = frozenset({
    "Tensor.reshape",
    "Tensor.permute",
    "Tensor.transpose",
    "Tensor.cast",
    "Tensor.expand",
    "_reshape",
    "_permute",
    "_transpose",
    "_cast",
    "_expand",
    "_repeat",
    "Attention.reshape_heads",
    "Attention.flatten_heads",
})


def _value_name(operand: GraphOperand) -> str | None:
    if isinstance(operand, (GraphValueRef, GraphValue)):
        return operand.name
    return None


def _graph_path_key(path: GraphPath) -> str:
    return ".".join(path.parts)


def _literal_value(operand: GraphOperand, default: Any = None) -> Any:
    if isinstance(operand, GraphLiteral):
        return operand.value
    return default


def _node_output_name(node: GraphNode) -> str | None:
    if len(node.outputs) >= 1:
        return _value_name(node.outputs[0])
    return None


def _module_contains_primitive(
    module: GraphModule,
    prim_name: str,
    modules_by_name: dict[str, GraphModule] | None = None,
    *,
    recursive: bool = False,
) -> bool:
    for node in module.nodes:
        if node.op.name == prim_name:
            return True
        for operand in node.inputs:
            for expr in _iter_operand_exprs(operand):
                if expr.op.name == prim_name:
                    return True
        for attr in node.attrs.values():
            for expr in _iter_operand_exprs(attr):
                if expr.op.name == prim_name:
                    return True
        if (
            recursive
            and modules_by_name is not None
            and "." in node.op.name
            and node.op.name in modules_by_name
        ):
            callee = modules_by_name[node.op.name]
            if _module_contains_primitive(
                callee, prim_name, modules_by_name, recursive=True
            ):
                return True
    return False


_PARAM_PRIMITIVES = frozenset({
    "_linear", "_rmsnorm", "_embedding", "_params_param",
})


def _module_has_params(
    module: GraphModule,
    modules_by_name: dict[str, GraphModule],
    _visited: set[str] | None = None,
) -> bool:
    """Check if a module contains any parameterized nodes (recursively)."""
    if _visited is None:
        _visited = set()
    if module.name in _visited:
        return False
    _visited.add(module.name)
    for node in module.nodes:
        if node.op.name in _PARAM_PRIMITIVES:
            return True
        if node.op.name in modules_by_name:
            if _module_has_params(modules_by_name[node.op.name], modules_by_name, _visited):
                return True
        for operand in (*node.inputs, *node.attrs.values()):
            if _operand_has_params(operand, modules_by_name, _visited):
                return True
    return False


def _operand_has_params(
    operand: GraphOperand,
    modules_by_name: dict[str, GraphModule],
    visited: set[str],
) -> bool:
    if not isinstance(operand, GraphExpr):
        return False
    if operand.op.name in _PARAM_PRIMITIVES:
        return True
    if operand.op.name in modules_by_name and _module_has_params(
        modules_by_name[operand.op.name],
        modules_by_name,
        visited,
    ):
        return True
    return any(
        _operand_has_params(item, modules_by_name, visited)
        for item in (*operand.inputs, *operand.attrs.values())
    )


def _find_node_by_id(program: GraphProgram, node_id: str) -> GraphNode | None:
    for module in program.modules:
        for node in module.nodes:
            if node.id == node_id:
                return node
    return None


def _resolve_value_to_node(
    module: GraphModule,
    operand: GraphOperand,
) -> GraphNode | None:
    name = _value_name(operand)
    if name is None:
        return None
    for node in module.nodes:
        for out in node.outputs:
            if _value_name(out) == name:
                return node
    return None


def _output_index(node: GraphNode, name: str) -> int | None:
    for idx, out in enumerate(node.outputs):
        if _value_name(out) == name:
            return idx
    return None


def _core_list_literal_values(operand: GraphOperand) -> tuple[Any, ...] | None:
    if not isinstance(operand, GraphExpr):
        return None
    if operand.op.name != "core.list":
        return None
    values: list[Any] = []
    for item in operand.inputs:
        if isinstance(item, GraphLiteral):
            values.append(item.value)
        else:
            values.append(None)
    return tuple(values)


def _infer_qkv_layout_from_split_module(module: GraphModule) -> str | None:
    """Infer packed/interleaved QKV layout from primitive split structure.

    The relevant semantic evidence is the actual reshape/slice structure, not
    the definition name. A packed split reshapes to [B,T,3,H,HD]; an interleaved
    split reshapes to [B,T,H,3,HD]. vLLM's QKVParallelLinear stores packed
    [3,H,HD], so interleaved checkpoint tensors need a load-time conversion.
    """
    input_name = _value_name(module.inputs[0]) if module.inputs else None
    if input_name is None:
        return None
    for node in module.nodes:
        if node.op.name != "_reshape" or len(node.inputs) < 2:
            continue
        if _value_name(node.inputs[0]) != input_name:
            continue
        shape = _core_list_literal_values(node.inputs[1])
        if shape is None or len(shape) != 5:
            continue
        three_positions = [idx for idx, value in enumerate(shape) if value == 3]
        if len(three_positions) != 1:
            continue
        if three_positions[0] == 2:
            return "packed"
        if three_positions[0] == 3:
            return "interleaved"
    return None


def _find_qkv_split_module_for_operand(
    module: GraphModule,
    operand: GraphOperand,
    modules_by_name: dict[str, GraphModule],
    *,
    depth: int = 0,
    visited: set[str] | None = None,
) -> GraphModule | None:
    if depth > 12:
        return None
    if visited is None:
        visited = set()
    name = _value_name(operand)
    if name is None:
        if isinstance(operand, GraphExpr):
            for inp in operand.inputs:
                found = _find_qkv_split_module_for_operand(
                    module, inp, modules_by_name, depth=depth + 1, visited=visited
                )
                if found is not None:
                    return found
        return None
    if name in visited:
        return None
    visited.add(name)
    node = _resolve_value_to_node(module, operand)
    if node is None:
        return None
    if node.op.name in modules_by_name:
        callee = modules_by_name[node.op.name]
        if _infer_qkv_layout_from_split_module(callee) is not None:
            return callee
    if node.op.name == "core.alias" and node.inputs:
        return _find_qkv_split_module_for_operand(
            module, node.inputs[0], modules_by_name, depth=depth + 1, visited=visited
        )
    if node.op.name == "core.select" and len(node.inputs) >= 2:
        out_idx = _output_index(node, name)
        for branch in node.inputs[1:3]:
            if (
                out_idx is not None
                and isinstance(branch, GraphExpr)
                and branch.op.name == "core.tuple"
                and out_idx < len(branch.inputs)
            ):
                found = _find_qkv_split_module_for_operand(
                    module,
                    branch.inputs[out_idx],
                    modules_by_name,
                    depth=depth + 1,
                    visited=visited,
                )
            else:
                found = _find_qkv_split_module_for_operand(
                    module, branch, modules_by_name, depth=depth + 1, visited=visited
                )
            if found is not None:
                return found
    for inp in node.inputs:
        found = _find_qkv_split_module_for_operand(
            module, inp, modules_by_name, depth=depth + 1, visited=visited
        )
        if found is not None:
            return found
    return None


def _infer_qkv_layout_from_operands(
    module: GraphModule,
    q_actual: GraphOperand,
    k_actual: GraphOperand,
    v_actual: GraphOperand,
    modules_by_name: dict[str, GraphModule],
) -> str:
    layouts: set[str] = set()
    for operand in (q_actual, k_actual, v_actual):
        split_module = _find_qkv_split_module_for_operand(module, operand, modules_by_name)
        if split_module is None:
            continue
        layout = _infer_qkv_layout_from_split_module(split_module)
        if layout is not None:
            layouts.add(layout)
    if len(layouts) == 1:
        return next(iter(layouts))
    return "packed"


@dataclass(frozen=True)
class _GroupedQKVSlice:
    reshape_node_id: str
    axis_key: str
    start_key: str
    end_key: str


def _operand_key(operand: GraphOperand) -> str:
    if isinstance(operand, GraphLiteral):
        return f"literal:{operand.value!r}"
    name = _value_name(operand)
    if name is not None:
        return f"value:{name}"
    if isinstance(operand, GraphExpr):
        return "expr:" + operand.op.name + "(" + ",".join(
            _operand_key(item) for item in operand.inputs
        ) + ")"
    if isinstance(operand, GraphPath):
        return "path:" + _graph_path_key(operand)
    return repr(operand)


def _is_rank5_shape_operand(operand: GraphOperand) -> bool:
    return isinstance(operand, GraphExpr) and operand.op.name == "core.list" and len(operand.inputs) == 5


def _find_grouped_qkv_slice(
    module: GraphModule,
    operand: GraphOperand,
    shared_linear_id: str,
    *,
    depth: int = 0,
    visited: set[str] | None = None,
) -> _GroupedQKVSlice | None:
    if depth > 16:
        return None
    if visited is None:
        visited = set()
    name = _value_name(operand)
    if name is not None:
        if name in visited:
            return None
        visited.add(name)
    node = _resolve_value_to_node(module, operand)
    if node is None:
        if isinstance(operand, GraphExpr):
            for inp in operand.inputs:
                result = _find_grouped_qkv_slice(
                    module, inp, shared_linear_id, depth=depth + 1, visited=visited
                )
                if result is not None:
                    return result
        return None
    if node.op.name in {"Tensor.slice", "_slice"} and len(node.inputs) >= 4:
        reshape_node = _resolve_value_to_node(module, node.inputs[0])
        if (
            reshape_node is not None
            and reshape_node.op.name in {"Tensor.reshape", "_reshape"}
            and len(reshape_node.inputs) >= 2
            and _is_rank5_shape_operand(reshape_node.inputs[1])
            and _trace_back(module, reshape_node.inputs[0], {shared_linear_id}) == shared_linear_id
        ):
            return _GroupedQKVSlice(
                reshape_node_id=reshape_node.id,
                axis_key=_operand_key(node.inputs[1]),
                start_key=_operand_key(node.inputs[2]),
                end_key=_operand_key(node.inputs[3]),
            )
    for inp in node.inputs:
        result = _find_grouped_qkv_slice(
            module, inp, shared_linear_id, depth=depth + 1, visited=visited
        )
        if result is not None:
            return result
    return None


def _infer_grouped_qkv_layout_from_operands(
    module: GraphModule,
    q_actual: GraphOperand,
    k_actual: GraphOperand,
    v_actual: GraphOperand,
    shared_linear_id: str,
) -> str | None:
    q_slice = _find_grouped_qkv_slice(module, q_actual, shared_linear_id)
    k_slice = _find_grouped_qkv_slice(module, k_actual, shared_linear_id)
    v_slice = _find_grouped_qkv_slice(module, v_actual, shared_linear_id)
    if q_slice is None or k_slice is None or v_slice is None:
        return None
    if not (
        q_slice.reshape_node_id == k_slice.reshape_node_id == v_slice.reshape_node_id
        and q_slice.axis_key == k_slice.axis_key == v_slice.axis_key
    ):
        return None
    if q_slice.start_key != "literal:0":
        return None
    if q_slice.end_key != k_slice.start_key:
        return None
    if k_slice.end_key != v_slice.start_key:
        return None
    return "grouped"


def _find_packed_qkv_chunk_output(
    module: GraphModule,
    operand: GraphOperand,
    shared_linear_id: str,
    modules_by_name: dict[str, GraphModule],
    *,
    depth: int = 0,
    visited: set[str] | None = None,
) -> tuple[str, int] | None:
    if depth > 16:
        return None
    if visited is None:
        visited = set()
    name = _value_name(operand)
    if name is not None:
        key = f"{module.name}:{name}"
        if key in visited:
            return None
        visited.add(key)
    if isinstance(operand, GraphExpr):
        for inp in operand.inputs:
            result = _find_packed_qkv_chunk_output(
                module,
                inp,
                shared_linear_id,
                modules_by_name,
                depth=depth + 1,
                visited=visited,
            )
            if result is not None:
                return result
        return None
    node = _resolve_value_to_node(module, operand)
    if node is None:
        return None
    if node.op.name in {"Tensor.chunk", "_chunk"} and len(node.outputs) == 3 and len(node.inputs) >= 1:
        out_idx = _output_index(node, name) if name is not None else None
        if out_idx is not None and _trace_back(module, node.inputs[0], {shared_linear_id}) == shared_linear_id:
            return (node.id, out_idx)
    if node.op.name in modules_by_name:
        out_idx = _output_index(node, name) if name is not None else 0
        if out_idx is not None:
            for actual in _module_output_actual_candidates(
                modules_by_name[node.op.name],
                node.inputs,
                out_idx,
                modules_by_name,
            ):
                result = _find_packed_qkv_chunk_output(
                    module,
                    actual,
                    shared_linear_id,
                    modules_by_name,
                    depth=depth + 1,
                    visited=visited,
                )
                if result is not None:
                    return result
    if node.op.name == "core.select" and len(node.inputs) >= 3:
        out_idx = _output_index(node, name) if name is not None else None
        for branch in node.inputs[1:3]:
            if (
                out_idx is not None
                and isinstance(branch, GraphExpr)
                and branch.op.name == "core.tuple"
                and out_idx < len(branch.inputs)
            ):
                candidates = (branch.inputs[out_idx],)
            else:
                candidates = (branch,)
            for candidate in candidates:
                result = _find_packed_qkv_chunk_output(
                    module,
                    candidate,
                    shared_linear_id,
                    modules_by_name,
                    depth=depth + 1,
                    visited=visited,
                )
                if result is not None:
                    return result
        return None
    for inp in node.inputs:
        result = _find_packed_qkv_chunk_output(
            module,
            inp,
            shared_linear_id,
            modules_by_name,
            depth=depth + 1,
            visited=visited,
        )
        if result is not None:
            return result
    return None


def _infer_packed_chunk_qkv_layout_from_operands(
    module: GraphModule,
    q_actual: GraphOperand,
    k_actual: GraphOperand,
    v_actual: GraphOperand,
    shared_linear_id: str,
    modules_by_name: dict[str, GraphModule],
) -> str | None:
    q_chunk = _find_packed_qkv_chunk_output(module, q_actual, shared_linear_id, modules_by_name)
    k_chunk = _find_packed_qkv_chunk_output(module, k_actual, shared_linear_id, modules_by_name)
    v_chunk = _find_packed_qkv_chunk_output(module, v_actual, shared_linear_id, modules_by_name)
    if q_chunk is None or k_chunk is None or v_chunk is None:
        return None
    chunk_ids = {q_chunk[0], k_chunk[0], v_chunk[0]}
    output_indices = (q_chunk[1], k_chunk[1], v_chunk[1])
    if len(chunk_ids) == 1 and output_indices == (0, 1, 2):
        return "packed"
    return None


def _is_module_call(node: GraphNode, modules_by_name: dict[str, GraphModule]) -> bool:
    return node.op.name in modules_by_name


def _called_module(
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
) -> GraphModule | None:
    return modules_by_name.get(node.op.name)


def _is_embedding_call(
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
) -> bool:
    if node.op.name == "_embedding":
        return True
    mod = _called_module(node, modules_by_name)
    if mod is None:
        return False
    return _module_contains_primitive(
        mod, "_embedding", modules_by_name,
        recursive=True,
    )


def _iter_operand_exprs(operand: GraphOperand) -> Any:
    if isinstance(operand, GraphExpr):
        yield operand
        for item in operand.inputs:
            yield from _iter_operand_exprs(item)
        for item in operand.attrs.values():
            if isinstance(item, (GraphExpr, GraphValueRef, GraphValue, GraphLiteral, GraphPath)):
                yield from _iter_operand_exprs(item)


def _module_input_index(module: GraphModule, operand: GraphOperand) -> int | None:
    name = _value_name(operand)
    if name is None:
        return None
    for index, module_input in enumerate(module.inputs):
        if _value_name(module_input) == name:
            return index
    return None


def _module_callsite_operands(
    main_module: GraphModule,
    module: GraphModule,
    input_index: int,
) -> list[GraphOperand]:
    operands: list[GraphOperand] = []
    for node in main_module.nodes:
        if node.op.name == module.name and len(node.inputs) > input_index:
            operands.append(node.inputs[input_index])
        for operand in (*node.inputs, *node.attrs.values()):
            if not isinstance(operand, (GraphExpr, GraphValueRef, GraphValue, GraphLiteral, GraphPath)):
                continue
            for expr in _iter_operand_exprs(operand):
                if expr.op.name == module.name and len(expr.inputs) > input_index:
                    operands.append(expr.inputs[input_index])
    return operands


def _is_direct_main_input(operand: GraphOperand, main_input_names: set[str]) -> bool:
    name = _value_name(operand)
    return name is not None and name in main_input_names


def _embedding_index_is_direct_model_input(
    node: GraphNode,
    module: GraphModule,
    main_module: GraphModule | None,
) -> bool:
    if len(node.inputs) < 1:
        return False
    x_input = node.inputs[0] if len(node.inputs) < 2 else node.inputs[1]
    if main_module is None:
        return False
    main_input_names = {
        name
        for inp in main_module.inputs
        if (name := _value_name(inp)) is not None
    }
    if module.name == main_module.name:
        return _is_direct_main_input(x_input, main_input_names)
    input_index = _module_input_index(module, x_input)
    if input_index is None:
        return False
    callsite_operands = _module_callsite_operands(main_module, module, input_index)
    if not callsite_operands:
        return False
    return all(_is_direct_main_input(operand, main_input_names) for operand in callsite_operands)


def _is_linear_call(
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
) -> bool:
    mod = _called_module(node, modules_by_name)
    if mod is None:
        return False
    return _module_contains_primitive(
        mod, "_linear", modules_by_name, recursive=False,
    )


def _is_linear_expr(
    expr: GraphExpr,
    modules_by_name: dict[str, GraphModule],
) -> bool:
    mod = modules_by_name.get(expr.op.name)
    if mod is None:
        return False
    return _module_contains_primitive(
        mod, "_linear", modules_by_name, recursive=False,
    )


def _operand_contains_linear_call(
    operand: GraphOperand,
    modules_by_name: dict[str, GraphModule],
) -> bool:
    if isinstance(operand, GraphExpr):
        if _is_linear_expr(operand, modules_by_name):
            return True
        return any(
            _operand_contains_linear_call(item, modules_by_name)
            for item in operand.inputs
        ) or any(
            _operand_contains_linear_call(item, modules_by_name)
            for item in operand.attrs.values()
            if isinstance(item, (GraphExpr, GraphValueRef, GraphValue, GraphLiteral, GraphPath))
        )
    return False


def _is_layernorm_call(
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
) -> bool:
    mod = _called_module(node, modules_by_name)
    if mod is None:
        return False
    return _module_contains_primitive(
        mod, "_layernorm", modules_by_name,
        recursive="." in node.op.name,
    )


def _is_rmsnorm_call(
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
) -> bool:
    mod = _called_module(node, modules_by_name)
    if mod is None:
        return False
    return _module_contains_primitive(
        mod, "_rmsnorm", modules_by_name,
        recursive="." in node.op.name,
    )


def _is_activation_call(
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
    activation_primitives: frozenset[str] | None = None,
) -> bool:
    if _is_activation_operand(node, modules_by_name, activation_primitives):
        return True
    mod = _called_module(node, modules_by_name)
    if mod is None:
        return False
    if activation_primitives is None:
        return any(
            n.op.name.startswith("_activations_")
            or _is_activation_operand(n, modules_by_name, activation_primitives)
            for n in mod.nodes
        )
    return any(
        n.op.name in activation_primitives
        or _is_activation_operand(n, modules_by_name, activation_primitives)
        for n in mod.nodes
    )


def _is_activation_operand(
    operand: GraphOperand,
    modules_by_name: dict[str, GraphModule],
    activation_primitives: frozenset[str] | None = None,
    *,
    depth: int = 0,
) -> bool:
    if depth > 4:
        return False
    if isinstance(operand, GraphExpr):
        if operand.op.name.startswith("_activations_"):
            return activation_primitives is None or operand.op.name in activation_primitives
        if operand.op.name in modules_by_name:
            callee = modules_by_name[operand.op.name]
            return any(
                n.op.name.startswith("_activations_")
                and (activation_primitives is None or n.op.name in activation_primitives)
                for n in callee.nodes
            )
        if operand.op.name == "core.select" and len(operand.inputs) >= 3:
            return (
                _is_activation_operand(
                    operand.inputs[1],
                    modules_by_name,
                    activation_primitives,
                    depth=depth + 1,
                )
                and _is_activation_operand(
                    operand.inputs[2],
                    modules_by_name,
                    activation_primitives,
                    depth=depth + 1,
                )
            )
    if isinstance(operand, GraphNode) and operand.op.name == "core.select" and len(operand.inputs) >= 3:
        return (
            _is_activation_operand(
                operand.inputs[1],
                modules_by_name,
                activation_primitives,
                depth=depth + 1,
            )
            and _is_activation_operand(
                operand.inputs[2],
                modules_by_name,
                activation_primitives,
                depth=depth + 1,
            )
        )
    return False


def _activation_data_operand(
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
) -> GraphOperand | None:
    if node.inputs and node.op.name.startswith("_activations_"):
        return node.inputs[0]
    if node.op.name in modules_by_name:
        callee = modules_by_name[node.op.name]
        for inner in callee.nodes:
            if inner.op.name.startswith("_activations_") and inner.inputs:
                name = _value_name(inner.inputs[0])
                if name is None:
                    continue
                for idx, formal in enumerate(callee.inputs):
                    if _value_name(formal) == name and idx < len(node.inputs):
                        return node.inputs[idx]
    if node.op.name == "core.select" and len(node.inputs) >= 3:
        candidates: list[GraphOperand] = []
        for branch in node.inputs[1:3]:
            if isinstance(branch, GraphExpr) and branch.op.name.startswith("_activations_") and branch.inputs:
                candidates.append(branch.inputs[0])
            elif isinstance(branch, GraphExpr) and branch.op.name in modules_by_name:
                callee = modules_by_name[branch.op.name]
                for inner in callee.nodes:
                    if inner.op.name.startswith("_activations_") and inner.inputs:
                        name = _value_name(inner.inputs[0])
                        for idx, formal in enumerate(callee.inputs):
                            if _value_name(formal) == name and idx < len(branch.inputs):
                                candidates.append(branch.inputs[idx])
                                break
                        break
        names = {_value_name(candidate) for candidate in candidates}
        if len(candidates) == 2 and len(names) == 1:
            return candidates[0]
    return node.inputs[0] if node.inputs else None


def _activation_data_operand_from_operand(
    operand: GraphOperand,
    modules_by_name: dict[str, GraphModule],
) -> GraphOperand | None:
    if isinstance(operand, GraphExpr) and operand.op.name.startswith("_activations_"):
        return operand.inputs[0] if operand.inputs else None
    if isinstance(operand, GraphExpr) and operand.op.name in modules_by_name:
        callee = modules_by_name[operand.op.name]
        for inner in callee.nodes:
            if inner.op.name.startswith("_activations_") and inner.inputs:
                name = _value_name(inner.inputs[0])
                if name is None:
                    continue
                for idx, formal in enumerate(callee.inputs):
                    if _value_name(formal) == name and idx < len(operand.inputs):
                        return operand.inputs[idx]
    if isinstance(operand, GraphExpr) and operand.op.name == "core.select" and len(operand.inputs) >= 3:
        candidates = [
            _activation_data_operand_from_operand(branch, modules_by_name)
            for branch in operand.inputs[1:3]
        ]
        if all(candidate is not None for candidate in candidates):
            names = {_value_name(candidate) for candidate in candidates}
            if len(names) == 1:
                return candidates[0]
    return None


def _is_sdpa_intrinsic_node(node: GraphNode) -> bool:
    return node.op.name in {
        "__torch_sdpa",
        "__tinygrad_sdpa",
        "__vllm_paged_attention",
    }


def _linear_base_path_from_call(
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
) -> str | None:
    if len(node.inputs) < 1:
        return None
    base = node.inputs[0]
    if isinstance(base, GraphPath):
        return _graph_path_key(base)
    return None


def _is_main_input(
    operand: GraphOperand,
    main_module: GraphModule,
) -> bool:
    name = _value_name(operand)
    if name is None:
        return False
    main_input_names = {
        inp.name for inp in main_module.inputs
        if isinstance(inp, (GraphValueRef, GraphValue))
    }
    return name in main_input_names


def _trace_back(
    module: GraphModule,
    operand: GraphOperand,
    target_node_ids: set[str],
    depth: int = 0,
    visited: set[str] | None = None,
) -> str | None:
    if depth > 10:
        return None
    if visited is None:
        visited = set()
    producer = _resolve_value_to_node(module, operand)
    if producer is None:
        return None
    if producer.id in target_node_ids:
        return producer.id
    if producer.id in visited:
        return None
    visited.add(producer.id)
    if producer.op.name in _TRIVIAL_TRANSFORM_OPS and producer.inputs:
        return _trace_back(module, producer.inputs[0], target_node_ids, depth + 1, visited)
    return None


def _find_terminal_norm_call_for_value(
    module: GraphModule,
    operand: GraphOperand,
    modules_by_name: dict[str, GraphModule],
    depth: int = 0,
    visited: set[str] | None = None,
) -> GraphNode | None:
    if depth > 8:
        return None
    if visited is None:
        visited = set()
    if isinstance(operand, GraphExpr):
        if operand.op.name in modules_by_name:
            callee = modules_by_name[operand.op.name]
            if callee.outputs:
                result = _find_terminal_norm_call_for_value(
                    callee,
                    callee.outputs[0],
                    modules_by_name,
                    depth + 1,
                    set(),
                )
                if result is not None:
                    return result
            for candidate in _module_output_actual_candidates(
                callee,
                operand.inputs,
                0,
                modules_by_name,
            ):
                result = _find_terminal_norm_call_for_value(
                    module, candidate, modules_by_name, depth + 1, visited
                )
                if result is not None:
                    return result
        if operand.op.name in _TRIVIAL_TRANSFORM_OPS and operand.inputs:
            return _find_terminal_norm_call_for_value(
                module, operand.inputs[0], modules_by_name, depth + 1, visited
            )
        return None
    name = _value_name(operand)
    if name is None:
        return None
    if name in visited:
        return None
    visited.add(name)
    node = _resolve_value_to_node(module, operand)
    if node is None:
        return None
    if _is_rmsnorm_call(node, modules_by_name) or _is_layernorm_call(node, modules_by_name):
        return node
    if _is_linear_call(node, modules_by_name):
        return None
    if node.op.name in modules_by_name:
        out_idx = _output_index(node, name)
        if out_idx is not None:
            callee = modules_by_name[node.op.name]
            if out_idx < len(callee.outputs):
                result = _find_terminal_norm_call_for_value(
                    callee,
                    callee.outputs[out_idx],
                    modules_by_name,
                    depth + 1,
                    set(),
                )
                if result is not None:
                    return result
            for candidate in _module_output_actual_candidates(
                callee,
                node.inputs,
                out_idx,
                modules_by_name,
            ):
                result = _find_terminal_norm_call_for_value(
                    module, candidate, modules_by_name, depth + 1, visited
                )
                if result is not None:
                    return result
    if node.op.name == "core.select" and len(node.inputs) >= 2:
        out_idx = _output_index(node, name)
        if out_idx is not None:
            for branch in node.inputs[1:3]:
                candidates: tuple[GraphOperand, ...]
                if (
                    isinstance(branch, GraphExpr)
                    and branch.op.name == "core.tuple"
                    and out_idx < len(branch.inputs)
                ):
                    candidates = (branch.inputs[out_idx],)
                elif isinstance(branch, GraphExpr) and branch.op.name in modules_by_name:
                    callee = modules_by_name[branch.op.name]
                    if out_idx < len(callee.outputs):
                        result = _find_terminal_norm_call_for_value(
                            callee,
                            callee.outputs[out_idx],
                            modules_by_name,
                            depth + 1,
                            set(),
                        )
                        if result is not None:
                            return result
                    candidates = tuple(
                        _module_output_actual_candidates(
                            callee,
                            branch.inputs,
                            out_idx,
                            modules_by_name,
                        )
                    )
                elif isinstance(branch, GraphExpr):
                    candidates = (branch,)
                else:
                    candidates = (branch,)
                for candidate in candidates:
                    result = _find_terminal_norm_call_for_value(
                        module, candidate, modules_by_name, depth + 1, visited
                    )
                    if result is not None:
                        return result
        return None
    if node.op.name in {"core.alias", "core.ascribe"} and node.inputs:
        return _find_terminal_norm_call_for_value(
            module, node.inputs[0], modules_by_name, depth + 1, visited
        )
    if node.op.name in _TRIVIAL_TRANSFORM_OPS and node.inputs:
        return _find_terminal_norm_call_for_value(
            module, node.inputs[0], modules_by_name, depth + 1, visited
        )
    return None


def _record_qkv_norm_roles(
    module: GraphModule,
    q_actual: GraphOperand,
    k_actual: GraphOperand,
    v_actual: GraphOperand,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    q_norm = _find_terminal_norm_call_for_value(module, q_actual, modules_by_name)
    k_norm = _find_terminal_norm_call_for_value(module, k_actual, modules_by_name)
    v_norm = _find_terminal_norm_call_for_value(module, v_actual, modules_by_name)
    if q_norm is not None and q_norm.id:
        classification.q_norm_node_ids.add(q_norm.id)
        classification.qk_norm_node_ids.add(q_norm.id)
    if k_norm is not None and k_norm.id:
        classification.k_norm_node_ids.add(k_norm.id)
        classification.qk_norm_node_ids.add(k_norm.id)
    if v_norm is not None and v_norm.id:
        classification.v_norm_node_ids.add(v_norm.id)


def _classify_embeddings(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    main_module = modules_by_name.get(program.main_module)
    for module in program.modules:
        for node in module.nodes:
            if not _is_embedding_call(node, modules_by_name):
                continue
            if len(node.inputs) < 1:
                continue
            if _embedding_index_is_direct_model_input(node, module, main_module):
                classification.node_types[node.id] = VLLMLayerType.VOCAB_PARALLEL_EMBEDDING
                classification.embedding_node_ids.add(node.id)
                classification.token_embedding_node_ids.add(node.id)
            else:
                classification.node_types[node.id] = VLLMLayerType.VOCAB_PARALLEL_EMBEDDING
                classification.embedding_node_ids.add(node.id)
                classification.position_embedding_node_ids.add(node.id)


def _is_structural_attention_call(
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
) -> bool:
    mod = _called_module(node, modules_by_name)
    if mod is None:
        return False
    if len(node.inputs) < 3:
        return False
    has_matmul = any(
        _module_contains_primitive(mod, op, modules_by_name, recursive=True)
        for op in ("_matmul", "Tensor.matmul")
    )
    has_softmax = any(
        _module_contains_primitive(mod, op, modules_by_name, recursive=True)
        for op in ("_softmax", "Tensor.softmax")
    )
    return has_matmul and has_softmax


def _classify_qkv_producers_structural(
    *,
    module: GraphModule,
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    q_actual = node.inputs[0]
    k_actual = node.inputs[1]
    v_actual = node.inputs[2]
    q_linear = _find_linear_call_for_value_deep(module, q_actual, modules_by_name)
    k_linear = _find_linear_call_for_value_deep(module, k_actual, modules_by_name)
    v_linear = _find_linear_call_for_value_deep(module, v_actual, modules_by_name)
    if q_linear and k_linear and v_linear:
        ids = {q_linear.id, k_linear.id, v_linear.id}
        if len(ids) < 2:
            layout = _infer_grouped_qkv_layout_from_operands(
                module, q_actual, k_actual, v_actual, q_linear.id
            )
            if layout is None:
                layout = _infer_packed_chunk_qkv_layout_from_operands(
                    module, q_actual, k_actual, v_actual, q_linear.id, modules_by_name
                )
            if layout is None:
                return
        else:
            layout = _infer_qkv_layout_from_operands(
                module, q_actual, k_actual, v_actual, modules_by_name
            )
        group = QKVGroup(
            q_node_id=q_linear.id,
            k_node_id=k_linear.id,
            v_node_id=v_linear.id,
            attention_node_id=node.id,
            layout=layout,
        )
        classification.qkv_groups.append(group)
        _record_qkv_norm_roles(
            module, q_actual, k_actual, v_actual, modules_by_name, classification
        )
        for linear_node in [q_linear, k_linear, v_linear]:
            if linear_node.id not in classification.node_types:
                classification.node_types[linear_node.id] = VLLMLayerType.QKV_PARALLEL_LINEAR


def _classify_attention_and_qkv(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    provenance: GraphProvenanceAnalysis,
    classification: VLLMLayerClassification,
) -> None:
    for module in program.modules:
        for node in module.nodes:
            if _is_sdpa_intrinsic_node(node):
                classification.node_types[node.id] = VLLMLayerType.ATTENTION
                classification.attention_node_ids.add(node.id)
                _classify_qkv_producers_from_intrinsic(
                    module=module,
                    node=node,
                    modules_by_name=modules_by_name,
                    classification=classification,
                )
                continue
            sdpa_fact = _find_sdpa_fact(node, modules_by_name, provenance)
            if sdpa_fact is not None:
                classification.node_types[node.id] = VLLMLayerType.ATTENTION
                classification.attention_node_ids.add(node.id)
                _classify_qkv_producers(
                    program=program,
                    module=module,
                    node=node,
                    sdpa_fact=sdpa_fact,
                    modules_by_name=modules_by_name,
                    classification=classification,
                )
                continue
            if _is_structural_attention_call(node, modules_by_name):
                before = len(classification.qkv_groups)
                _classify_qkv_producers_structural(
                    module=module,
                    node=node,
                    modules_by_name=modules_by_name,
                    classification=classification,
                )
                if len(classification.qkv_groups) != before:
                    classification.node_types[node.id] = VLLMLayerType.ATTENTION
                    classification.attention_node_ids.add(node.id)
    _classify_packed_qkv(program, modules_by_name, classification)


def _find_sdpa_fact(
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
    provenance: GraphProvenanceAnalysis,
) -> GraphSdpaGqaFact | None:
    if node.op.name not in modules_by_name:
        return None
    callee = modules_by_name[node.op.name]
    if len(node.inputs) != len(callee.inputs):
        return None
    output_facts = tuple(
        graph_provenance_facts(item)
        for item in provenance.module_summary_provenance.get(callee.name, ())
    )
    if not output_facts:
        return None
    for fact in output_facts[0]:
        if fact.kind == "sdpa_gqa" and isinstance(fact.value, GraphSdpaGqaFact):
            return fact.value
    return None


def _classify_qkv_producers(
    *,
    program: GraphProgram,
    module: GraphModule,
    node: GraphNode,
    sdpa_fact: GraphSdpaGqaFact,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    callee = modules_by_name.get(node.op.name)
    if callee is None:
        return
    formal_to_actual = {
        formal.name: actual
        for formal, actual in zip(callee.inputs, node.inputs, strict=False)
    }
    q_actual = formal_to_actual.get(sdpa_fact.q)
    k_actual = formal_to_actual.get(sdpa_fact.k)
    v_actual = formal_to_actual.get(sdpa_fact.v)
    if q_actual is None or k_actual is None or v_actual is None:
        return
    q_linear = _find_linear_call_for_value_deep(module, q_actual, modules_by_name)
    k_linear = _find_linear_call_for_value_deep(module, k_actual, modules_by_name)
    v_linear = _find_linear_call_for_value_deep(module, v_actual, modules_by_name)
    if q_linear and k_linear and v_linear:
        ids = {q_linear.id, k_linear.id, v_linear.id}
        if len(ids) < 2:
            layout = _infer_grouped_qkv_layout_from_operands(
                module, q_actual, k_actual, v_actual, q_linear.id
            )
            if layout is None:
                layout = _infer_packed_chunk_qkv_layout_from_operands(
                    module, q_actual, k_actual, v_actual, q_linear.id, modules_by_name
                )
            if layout is None:
                layout = _infer_qkv_layout_from_operands(
                    module, q_actual, k_actual, v_actual, modules_by_name
                )
        else:
            layout = _infer_qkv_layout_from_operands(
                module, q_actual, k_actual, v_actual, modules_by_name
            )
        group = QKVGroup(
            q_node_id=q_linear.id,
            k_node_id=k_linear.id,
            v_node_id=v_linear.id,
            attention_node_id=node.id,
            layout=layout,
        )
        classification.qkv_groups.append(group)
        _record_qkv_norm_roles(
            module, q_actual, k_actual, v_actual, modules_by_name, classification
        )
        for linear_node in [q_linear, k_linear, v_linear]:
            if linear_node.id not in classification.node_types:
                classification.node_types[linear_node.id] = VLLMLayerType.QKV_PARALLEL_LINEAR


def _classify_qkv_producers_from_intrinsic(
    *,
    module: GraphModule,
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    if len(node.inputs) < 3:
        return
    q_actual = node.inputs[0]
    k_actual = node.inputs[1]
    v_actual = node.inputs[2]
    q_linear = _find_linear_call_for_value(module, q_actual, modules_by_name)
    k_linear = _find_linear_call_for_value(module, k_actual, modules_by_name)
    v_linear = _find_linear_call_for_value(module, v_actual, modules_by_name)
    if q_linear and k_linear and v_linear:
        ids = {q_linear.id, k_linear.id, v_linear.id}
        if len(ids) < 2:
            layout = _infer_grouped_qkv_layout_from_operands(
                module, q_actual, k_actual, v_actual, q_linear.id
            )
            if layout is None:
                layout = _infer_packed_chunk_qkv_layout_from_operands(
                    module, q_actual, k_actual, v_actual, q_linear.id, modules_by_name
                )
            if layout is None:
                layout = _infer_qkv_layout_from_operands(
                    module, q_actual, k_actual, v_actual, modules_by_name
                )
        else:
            layout = _infer_qkv_layout_from_operands(
                module, q_actual, k_actual, v_actual, modules_by_name
            )
        group = QKVGroup(
            q_node_id=q_linear.id,
            k_node_id=k_linear.id,
            v_node_id=v_linear.id,
            attention_node_id=node.id,
            layout=layout,
        )
        classification.qkv_groups.append(group)
        _record_qkv_norm_roles(
            module, q_actual, k_actual, v_actual, modules_by_name, classification
        )
        for linear_node in [q_linear, k_linear, v_linear]:
            if linear_node.id not in classification.node_types:
                classification.node_types[linear_node.id] = VLLMLayerType.QKV_PARALLEL_LINEAR


def _classify_packed_qkv(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    for module in program.modules:
        for node in module.nodes:
            if node.op.name != "_chunk":
                continue
            if len(node.outputs) != 3:
                continue
            if len(node.inputs) < 1:
                continue
            producer = _resolve_value_to_node(module, node.inputs[0])
            if producer is None or not _is_linear_call(producer, modules_by_name):
                continue
            if producer.id in classification.node_types:
                continue
            attn_ids_in_module = {
                aid for aid in classification.attention_node_ids
                if any(n.id == aid for n in module.nodes)
            }
            attn_id = next(iter(attn_ids_in_module), None)
            if attn_id is None:
                continue
            classification.node_types[producer.id] = VLLMLayerType.QKV_PARALLEL_LINEAR
            group = QKVGroup(
                q_node_id=producer.id,
                k_node_id=producer.id,
                v_node_id=producer.id,
                attention_node_id=attn_id,
                layout="packed",
            )
            classification.qkv_groups.append(group)
            _record_qkv_norm_roles(
                module,
                node.outputs[0],
                node.outputs[1],
                node.outputs[2],
                modules_by_name,
                classification,
            )


def _find_linear_call_for_value(
    module: GraphModule,
    operand: GraphOperand,
    modules_by_name: dict[str, GraphModule],
) -> GraphNode | None:
    name = _value_name(operand)
    if name is None:
        return None
    node = _resolve_value_to_node(module, operand)
    if node is not None and _is_linear_call(node, modules_by_name):
        return node
    return None


def _find_linear_call_for_value_deep(
    module: GraphModule,
    operand: GraphOperand,
    modules_by_name: dict[str, GraphModule],
    depth: int = 0,
    visited: set[str] | None = None,
) -> GraphNode | None:
    if depth > 12:
        return None
    if visited is None:
        visited = set()
    name = _value_name(operand)
    if name is None:
        if isinstance(operand, GraphExpr):
            for inp in operand.inputs:
                result = _find_linear_call_for_value_deep(
                    module, inp, modules_by_name, depth, visited
                )
                if result is not None:
                    return result
        return None
    if name in visited:
        return None
    visited.add(name)
    node = _resolve_value_to_node(module, operand)
    if node is None:
        return None
    if _is_linear_call(node, modules_by_name):
        return node
    if node.op.name == "core.select" and len(node.inputs) >= 2:
        out_idx: int | None = None
        for i, out in enumerate(node.outputs):
            if _value_name(out) == name:
                out_idx = i
                break
        if out_idx is not None:
            true_branch = node.inputs[1]
            if (
                isinstance(true_branch, GraphExpr)
                and true_branch.op.name == "core.tuple"
                and out_idx < len(true_branch.inputs)
            ):
                result = _find_linear_call_for_value_deep(
                    module, true_branch.inputs[out_idx],
                    modules_by_name, depth + 1, visited,
                )
                if result is not None:
                    return result
            if len(node.inputs) >= 3:
                false_branch = node.inputs[2]
                if isinstance(false_branch, GraphExpr):
                    for inp in false_branch.inputs:
                        result = _find_linear_call_for_value_deep(
                            module, inp, modules_by_name, depth + 1, visited,
                        )
                        if result is not None:
                            return result
            return None
    if node.op.name in modules_by_name and len(node.outputs) > 1:
        out_idx: int | None = None
        for i, out in enumerate(node.outputs):
            if _value_name(out) == name:
                out_idx = i
                break
        if out_idx is not None:
            for actual in _module_output_actual_candidates(
                modules_by_name[node.op.name],
                node.inputs,
                out_idx,
                modules_by_name,
            ):
                result = _find_linear_call_for_value_deep(
                    module, actual, modules_by_name, depth + 1, visited,
                )
                if result is not None:
                    return result
    for inp in node.inputs:
        result = _find_linear_call_for_value_deep(
            module, inp, modules_by_name, depth + 1, visited
        )
        if result is not None:
            return result
    return None


def _module_output_actual_candidates(
    callee: GraphModule,
    actuals: tuple[GraphOperand, ...],
    output_index: int,
    modules_by_name: dict[str, GraphModule],
) -> list[GraphOperand]:
    formal_to_actual = {
        formal.name: actual
        for formal, actual in zip(callee.inputs, actuals, strict=False)
        if _value_name(formal) is not None
    }
    if output_index < len(callee.outputs):
        output_operand: GraphOperand = callee.outputs[output_index]
        tuple_index: int | None = None
    elif len(callee.outputs) == 1:
        output_operand = callee.outputs[0]
        tuple_index = output_index
    else:
        return []
    deps = _formal_dependencies_for_operand(
        callee,
        output_operand,
        modules_by_name,
        tuple_index=tuple_index,
    )
    return [
        formal_to_actual[name]
        for name in sorted(deps)
        if name in formal_to_actual
    ]


def _formal_dependencies_for_operand(
    module: GraphModule,
    operand: GraphOperand,
    modules_by_name: dict[str, GraphModule],
    *,
    tuple_index: int | None = None,
    visited: set[str] | None = None,
) -> set[str]:
    if visited is None:
        visited = set()
    if isinstance(operand, GraphLiteral | GraphPath):
        return set()
    if isinstance(operand, GraphExpr):
        if operand.op.name == "core.tuple" and tuple_index is not None:
            if 0 <= tuple_index < len(operand.inputs):
                return _formal_dependencies_for_operand(
                    module, operand.inputs[tuple_index], modules_by_name, visited=visited
                )
            return set()
        if operand.op.name in modules_by_name:
            return _module_expr_output_formal_dependencies(
                operand, modules_by_name, tuple_index=tuple_index, visited=visited
            )
        deps: set[str] = set()
        for item in operand.inputs:
            deps.update(
                _formal_dependencies_for_operand(
                    module, item, modules_by_name, visited=visited
                )
            )
        return deps
    name = _value_name(operand)
    if name is None:
        return set()
    if name in {formal.name for formal in module.inputs}:
        return {name}
    if name in visited:
        return set()
    visited.add(name)
    producer = _resolve_value_to_node(module, operand)
    if producer is None:
        return set()
    if producer.op.name == "core.tuple":
        effective_tuple_index = tuple_index
        if effective_tuple_index is None:
            effective_tuple_index = _output_index(producer, name)
        if effective_tuple_index is not None and 0 <= effective_tuple_index < len(producer.inputs):
            return _formal_dependencies_for_operand(
                module,
                producer.inputs[effective_tuple_index],
                modules_by_name,
                visited=visited,
            )
        return set()
    if producer.op.name == "core.select" and len(producer.inputs) >= 3:
        effective_tuple_index = tuple_index
        if effective_tuple_index is None and len(producer.outputs) > 1:
            effective_tuple_index = _output_index(producer, name)
        deps: set[str] = set()
        for branch in producer.inputs[1:3]:
            deps.update(
                _formal_dependencies_for_operand(
                    module,
                    branch,
                    modules_by_name,
                    tuple_index=effective_tuple_index,
                    visited=set(visited),
                )
            )
        return deps
    if producer.op.name in modules_by_name:
        return set(
            name
            for actual in _module_output_actual_candidates(
                modules_by_name[producer.op.name],
                producer.inputs,
                tuple_index or 0,
                modules_by_name,
            )
            for name in _formal_dependencies_for_operand(
                module, actual, modules_by_name, visited=set(visited)
            )
        )
    deps: set[str] = set()
    for item in producer.inputs:
        deps.update(
            _formal_dependencies_for_operand(
                module, item, modules_by_name, visited=set(visited)
            )
        )
    return deps


def _module_expr_output_formal_dependencies(
    expr: GraphExpr,
    modules_by_name: dict[str, GraphModule],
    *,
    tuple_index: int | None,
    visited: set[str],
) -> set[str]:
    callee = modules_by_name.get(expr.op.name)
    if callee is None:
        return set()
    candidates = _module_output_actual_candidates(
        callee,
        expr.inputs,
        tuple_index or 0,
        modules_by_name,
    )
    deps: set[str] = set()
    synthetic = GraphModule(
        name="<expr>",
        inputs=tuple(
            GraphValue(name=f"arg{idx}", type_expr=arg.type_expr, dims=getattr(arg, "dims", None))
            for idx, arg in enumerate(expr.inputs)
            if isinstance(arg, GraphValueRef)
        ),
        outputs=(),
        output_names=(),
        nodes=(),
    )
    for actual in candidates:
        name = _value_name(actual)
        if name is not None:
            deps.add(name)
        else:
            deps.update(
                _formal_dependencies_for_operand(
                    synthetic, actual, modules_by_name, visited=set(visited)
                )
            )
    return deps


def _classify_ffn(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    provenance: GraphProvenanceAnalysis,
    classification: VLLMLayerClassification,
) -> None:
    _classify_selected_expert_ffn(program, classification)
    _classify_gated_ffn(program, modules_by_name, classification)
    _classify_simple_ffn(program, modules_by_name, classification)


def _classify_selected_expert_ffn(
    program: GraphProgram,
    classification: VLLMLayerClassification,
) -> None:
    """Classify already-proven selected-expert graph intrinsics as FFN blocks.

    The intrinsic itself is introduced by graph optimization from primitive
    provenance.  vLLM clean-forward scheduling must treat it as the FFN body;
    otherwise MoE blocks are silently skipped.
    """
    seen: set[str] = {g.gate_up_intrinsic_node_id for g in classification.ffn_groups if g.gate_up_intrinsic_node_id}
    for module in program.modules:
        for node in module.nodes:
            if node.op.name not in _SELECTED_EXPERT_INTRINSICS:
                continue
            if node.id in seen:
                continue
            classification.ffn_groups.append(FFNGroup(gate_up_intrinsic_node_id=node.id))
            seen.add(node.id)


def _classify_gated_ffn(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    for module in program.modules:
        nodes = list(module.nodes)
        for i, node in enumerate(nodes):
            act_output = _find_activation_followed_by_mul(module, node, nodes, i, modules_by_name)
            if act_output is None:
                continue
            act_node, mul_node, mul_output = act_output
            gate_input = _activation_data_operand(act_node, modules_by_name)
            up_input = None
            if mul_node and len(mul_node.inputs) >= 2:
                act_name = _node_output_name(act_node)
                in0_name = _value_name(mul_node.inputs[0])
                if in0_name == act_name:
                    up_input = mul_node.inputs[1]
                else:
                    up_input = mul_node.inputs[0]
            gate_linear = _find_linear_call_for_value(module, gate_input, modules_by_name) if gate_input else None
            up_linear = _find_linear_call_for_value(module, up_input, modules_by_name) if up_input else None
            if gate_linear is None and gate_input is not None:
                gate_linear = _find_linear_call_for_value_deep(module, gate_input, modules_by_name)
            if up_linear is None and up_input is not None:
                up_linear = _find_linear_call_for_value_deep(module, up_input, modules_by_name)
            down_linear = None
            for j in range(i + 1, len(nodes)):
                if _is_linear_call(nodes[j], modules_by_name) and len(nodes[j].inputs) >= 2:
                    down_input_name = _value_name(nodes[j].inputs[1])
                    if down_input_name == mul_output:
                        down_linear = nodes[j]
                        break
            if gate_linear and up_linear and down_linear:
                group = FFNGroup(
                    gate_node_id=gate_linear.id,
                    up_node_id=up_linear.id,
                    down_node_id=down_linear.id,
                )
                classification.ffn_groups.append(group)
                gate_up_type = (
                    VLLMLayerType.COLUMN_PARALLEL_LINEAR
                    if gate_linear.id == up_linear.id
                    else VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR
                )
                if gate_linear.id not in classification.node_types:
                    classification.node_types[gate_linear.id] = gate_up_type
                if up_linear.id not in classification.node_types:
                    classification.node_types[up_linear.id] = gate_up_type
                if down_linear.id not in classification.node_types:
                    classification.node_types[down_linear.id] = VLLMLayerType.ROW_PARALLEL_LINEAR


def _find_activation_followed_by_mul(
    module: GraphModule,
    node: GraphNode,
    nodes: list[GraphNode],
    index: int,
    modules_by_name: dict[str, GraphModule],
) -> tuple[GraphNode, GraphNode, str] | None:
    if node.op.name.startswith("_activations_"):
        pass
    elif not _is_activation_call(node, modules_by_name):
        return None
    else:
        mod = _called_module(node, modules_by_name)
        if mod is None or not any(n.op.name.startswith("_activations_") for n in mod.nodes):
            return None
    if len(node.inputs) < 1:
        return None
    act_output = _node_output_name(node)
    if act_output is None:
        return None
    for j in range(index + 1, len(nodes)):
        if nodes[j].op.name in {"_mul", "core.binary.*"} and len(nodes[j].inputs) >= 2:
            in0_name = _value_name(nodes[j].inputs[0])
            in1_name = _value_name(nodes[j].inputs[1])
            if in0_name == act_output or in1_name == act_output:
                mul_output = _node_output_name(nodes[j])
                if mul_output is None:
                    continue
                return node, nodes[j], mul_output
    return None


def _classify_simple_ffn(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    for module in program.modules:
        nodes = list(module.nodes)
        for i, node in enumerate(nodes):
            if not _is_activation_call(node, modules_by_name):
                continue
            up_input = _activation_data_operand(node, modules_by_name)
            up_linear = None
            if up_input is not None:
                up_linear = _find_linear_call_for_value(module, up_input, modules_by_name)
            if up_linear is None:
                continue
            if up_linear.id in classification.node_types:
                continue
            act_output = _node_output_name(node)
            if act_output is None:
                continue
            down_linear = None
            for j in range(i + 1, len(nodes)):
                if _is_linear_call(nodes[j], modules_by_name) and len(nodes[j].inputs) >= 2:
                    down_input_name = _value_name(nodes[j].inputs[1])
                    if down_input_name == act_output:
                        down_linear = nodes[j]
                        break
            if down_linear is None:
                continue
            if down_linear.id in classification.node_types:
                continue
            group = FFNGroup(
                up_node_id=up_linear.id,
                down_node_id=down_linear.id,
            )
            classification.ffn_groups.append(group)
            if up_linear.id not in classification.node_types:
                classification.node_types[up_linear.id] = VLLMLayerType.COLUMN_PARALLEL_LINEAR
            if down_linear.id not in classification.node_types:
                classification.node_types[down_linear.id] = VLLMLayerType.ROW_PARALLEL_LINEAR
        for node in nodes:
            if not _is_linear_call(node, modules_by_name) or len(node.inputs) < 2:
                continue
            if node.id in classification.node_types:
                continue
            up_input = _activation_data_operand_from_operand(node.inputs[1], modules_by_name)
            if up_input is None:
                continue
            up_linear = _find_linear_call_for_value(module, up_input, modules_by_name)
            if up_linear is None or up_linear.id in classification.node_types:
                continue
            group = FFNGroup(
                up_node_id=up_linear.id,
                down_node_id=node.id,
            )
            classification.ffn_groups.append(group)
            classification.node_types[up_linear.id] = VLLMLayerType.COLUMN_PARALLEL_LINEAR
            classification.node_types[node.id] = VLLMLayerType.ROW_PARALLEL_LINEAR


def _classify_output_projections(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    for module in program.modules:
        for node in module.nodes:
            if not _is_linear_call(node, modules_by_name):
                continue
            if node.id in classification.node_types:
                continue
            if len(node.inputs) < 2:
                continue
            found = _trace_back(module, node.inputs[1], classification.attention_node_ids)
            if found is not None:
                classification.node_types[node.id] = VLLMLayerType.ROW_PARALLEL_LINEAR


def _classify_lm_head(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    main_module = modules_by_name.get(program.main_module)
    if main_module is None:
        return
    all_output_names: set[str] = set()
    for out in main_module.outputs:
        if isinstance(out, (GraphValueRef, GraphValue)):
            all_output_names.add(out.name)

    def _reaches_output(name: str, visited: set[str] | None = None, depth: int = 0) -> bool:
        if name in all_output_names:
            return True
        if depth > 8:
            return False
        if visited is None:
            visited = set()
        if name in visited:
            return False
        visited.add(name)
        for node in main_module.nodes:
            for inp in node.inputs:
                found = False
                if _value_name(inp) == name:
                    found = True
                elif isinstance(inp, GraphExpr):
                    for sub in inp.inputs:
                        if _value_name(sub) == name:
                            found = True
                            break
                if found:
                    out_name = _node_output_name(node)
                    if out_name and _reaches_output(out_name, visited, depth + 1):
                        return True
        return False

    for node in reversed(main_module.nodes):
        node_output = _node_output_name(node)
        if not node_output or not _reaches_output(node_output):
            continue
        if _is_linear_call(node, modules_by_name) or (
            node.op.name == "core.select"
            and any(
                _operand_contains_linear_call(operand, modules_by_name)
                for operand in node.inputs[1:]
            )
        ):
            classification.node_types[node.id] = VLLMLayerType.PARALLEL_LM_HEAD
            classification.lm_head_node_id = node.id
            return


def _classify_norms(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    qkv_node_ids: set[str] = set()
    for g in classification.qkv_groups:
        qkv_node_ids.add(g.q_node_id)
        qkv_node_ids.add(g.k_node_id)
        qkv_node_ids.add(g.v_node_id)
    for module in program.modules:
        if "." in module.name and "__loop" not in module.name:
            continue
        for node in module.nodes:
            if node.op.name not in modules_by_name:
                continue
            if _is_layernorm_call(node, modules_by_name):
                classification.node_types[node.id] = VLLMLayerType.LAYERNORM
            elif _is_rmsnorm_call(node, modules_by_name):
                classification.node_types[node.id] = VLLMLayerType.RMSNORM
                classification.rmsnorm_node_ids.add(node.id)
                if len(node.inputs) >= 2 and qkv_node_ids:
                    data_inp = node.inputs[1] if isinstance(
                        node.inputs[0], GraphPath
                    ) else node.inputs[0]
                    src = _trace_back(module, data_inp, qkv_node_ids)
                    if src is not None:
                        classification.qk_norm_node_ids.add(node.id)


def _classify_qkv_norm_roles_from_attention_inputs(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    attention_ids = {
        group.attention_node_id
        for group in classification.qkv_groups
        if group.attention_node_id is not None
    }
    if not attention_ids:
        return
    for module in program.modules:
        for node in module.nodes:
            if node.id not in attention_ids or len(node.inputs) < 3:
                continue
            _record_qkv_norm_roles(
                module,
                node.inputs[0],
                node.inputs[1],
                node.inputs[2],
                modules_by_name,
                classification,
            )


def _classify_repeated_modules(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    for module in program.modules:
        if "__loop" not in module.name or "_step" not in module.name:
            continue
        loop_var_name: str | None = None
        for inp in module.inputs:
            name = _value_name(inp)
            if name is None or name == "__scope":
                continue
            if isinstance(inp.type_expr, TypePath):
                continue
            loop_var_name = name
            break
        if loop_var_name is None:
            continue
        found_called = False
        for node in module.nodes:
            if node.op.name not in modules_by_name:
                continue
            if "." in node.op.name:
                continue
            body_module = modules_by_name[node.op.name]
            if not node.inputs:
                continue
            scope_inp = node.inputs[0]
            scope_parts: tuple[str, ...] | None = None
            if isinstance(scope_inp, GraphPath):
                scope_parts = tuple(scope_inp.parts)
            for j, inp in enumerate(node.inputs):
                inp_name = _value_name(inp)
                if inp_name != loop_var_name:
                    continue
                if j < len(body_module.inputs):
                    body_param_name = _value_name(body_module.inputs[j])
                    if body_param_name is not None:
                        classification.repeated_module_names.add(node.op.name)
                        classification.loop_index_param[node.op.name] = body_param_name
                        if scope_parts is not None:
                            classification.module_scope_parts[node.op.name] = scope_parts
                break
            else:
                # Loop variable not passed as a direct value input; check if
                # it appears in the scope path (e.g. h.{i}).
                if scope_parts is not None:
                    loop_token = "{" + loop_var_name + "}"
                    if any(loop_token in p for p in scope_parts):
                        classification.repeated_module_names.add(node.op.name)
                        classification.module_scope_parts[node.op.name] = scope_parts
            found_called = True
        # If the block was inlined into the loop body (no called module found),
        # treat the loop body module itself as the repeated module.
        if not found_called:
            classification.repeated_module_names.add(module.name)
            classification.loop_index_param[module.name] = loop_var_name


def _classify_per_layer_called_modules(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    """Detect modules called from within repeated modules (e.g. gemma4_kv called from gemma4_e_block).

    Such modules' parameterized nodes (linear, norm) are per-layer and should
    be treated as ModuleList in the emitter.
    """
    if not classification.repeated_module_names:
        return

    def _module_path_formal_names(module_name: str) -> set[str]:
        module = modules_by_name.get(module_name)
        if module is None:
            return set()
        return {
            value.name
            for value in module.inputs
            if isinstance(value.type_expr, TypePath)
        }

    def _resolve_child_scope(
        *,
        parent_module_name: str,
        parent_scope: tuple[str, ...] | None,
        call_scope: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not call_scope:
            return parent_scope or ()
        first = call_scope[0]
        if first == "{__scope}":
            extra = call_scope[1:]
            return (tuple(parent_scope) + extra) if parent_scope is not None else call_scope
        if first.startswith("{") and first.endswith("}"):
            formal_name = first[1:-1]
            if formal_name in _module_path_formal_names(parent_module_name):
                extra = call_scope[1:]
                return (tuple(parent_scope) + extra) if parent_scope is not None else call_scope
        if parent_scope is not None:
            return tuple(parent_scope) + call_scope
        return call_scope

    def _mark_called_module(
        *,
        parent_module_name: str,
        called_name: str,
        call_inputs: tuple[GraphOperand, ...] | list[GraphOperand],
        parent_scope: tuple[str, ...] | None,
        parent_loop_index: str,
    ) -> bool:
        if "." in called_name:
            return False
        if called_name not in modules_by_name:
            return False
        called_mod = modules_by_name[called_name]
        if not _module_has_params(called_mod, modules_by_name):
            return False
        changed = called_name not in classification.repeated_module_names
        classification.repeated_module_names.add(called_name)
        classification.loop_index_param[called_name] = parent_loop_index
        if call_inputs and isinstance(call_inputs[0], GraphPath):
            resolved_scope = _resolve_child_scope(
                parent_module_name=parent_module_name,
                parent_scope=parent_scope,
                call_scope=tuple(call_inputs[0].parts),
            )
            if classification.module_scope_parts.get(called_name) != resolved_scope:
                classification.module_scope_parts[called_name] = resolved_scope
                changed = True
        elif call_inputs and parent_scope is not None:
            actual_name = _value_name(call_inputs[0])
            if actual_name is not None and actual_name in _module_path_formal_names(parent_module_name):
                resolved_scope = tuple(parent_scope)
                if classification.module_scope_parts.get(called_name) != resolved_scope:
                    classification.module_scope_parts[called_name] = resolved_scope
                    changed = True
        return changed

    def _mark_called_expr_modules(
        expr: GraphExpr,
        *,
        parent_module_name: str,
        parent_scope: tuple[str, ...] | None,
        parent_loop_index: str,
    ) -> None:
        _mark_called_module(
            parent_module_name=parent_module_name,
            called_name=expr.op.name,
            call_inputs=expr.inputs,
            parent_scope=parent_scope,
            parent_loop_index=parent_loop_index,
        )
        for operand in (*expr.inputs, *expr.attrs.values()):
            if isinstance(operand, GraphExpr):
                _mark_called_expr_modules(
                    operand,
                    parent_module_name=parent_module_name,
                    parent_scope=parent_scope,
                    parent_loop_index=parent_loop_index,
                )

    processed: set[str] = set()
    while True:
        pending = sorted(classification.repeated_module_names - processed)
        if not pending:
            break
        repeated_name = pending[0]
        processed.add(repeated_name)
        repeated_mod = modules_by_name.get(repeated_name)
        if repeated_mod is None:
            continue
        parent_scope = classification.module_scope_parts.get(repeated_name)
        parent_idx = classification.loop_index_param.get(repeated_name, "i")
        for node in repeated_mod.nodes:
            _mark_called_module(
                parent_module_name=repeated_name,
                called_name=node.op.name,
                call_inputs=node.inputs,
                parent_scope=parent_scope,
                parent_loop_index=parent_idx,
            )
            for operand in (*node.inputs, *node.attrs.values()):
                if isinstance(operand, GraphExpr):
                    _mark_called_expr_modules(
                        operand,
                        parent_module_name=repeated_name,
                        parent_scope=parent_scope,
                        parent_loop_index=parent_idx,
                    )


def _linear_path_leaf(node: GraphNode) -> str | None:
    """Extract the leaf name (e.g. 'q_proj') from a linear call's path."""
    for inp in node.inputs:
        if isinstance(inp, GraphPath) and inp.parts:
            for part in inp.parts:
                if part in ("q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"):
                    return part
    return None


def _classify_qkv_by_path(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    """Detect QKV by linear path names when structural detection fails."""
    if classification.qkv_groups:
        return
    if not classification.repeated_module_names:
        return
    repeated_mod_name = max(
        classification.repeated_module_names,
        key=lambda n: len(modules_by_name[n].nodes) if n in modules_by_name else 0,
    )
    repeated_mod = modules_by_name.get(repeated_mod_name)
    if repeated_mod is None:
        return

    q_node = k_node = v_node = o_node = attn_node = None

    for module in program.modules:
        for node in module.nodes:
            if _is_linear_call(node, modules_by_name):
                leaf = _linear_path_leaf(node)
                if leaf == "q_proj" and q_node is None:
                    q_node = node
                elif leaf == "k_proj" and k_node is None:
                    k_node = node
                elif leaf == "v_proj" and v_node is None:
                    v_node = node
                elif leaf == "o_proj" and o_node is None:
                    o_node = node
            elif _is_sdpa_intrinsic_node(node) and attn_node is None:
                attn_node = node

    if q_node and k_node and v_node and attn_node:
        group = QKVGroup(
            q_node_id=q_node.id,
            k_node_id=k_node.id,
            v_node_id=v_node.id,
            attention_node_id=attn_node.id,
        )
        classification.qkv_groups.append(group)
        for linear_node in [q_node, k_node, v_node]:
            if linear_node.id not in classification.node_types:
                classification.node_types[linear_node.id] = (
                    VLLMLayerType.QKV_PARALLEL_LINEAR
                )
        if attn_node:
            classification.node_types[attn_node.id] = VLLMLayerType.ATTENTION
            classification.attention_node_ids.add(attn_node.id)
        if o_node:
            classification.node_types[o_node.id] = VLLMLayerType.ROW_PARALLEL_LINEAR
            classification.o_proj_node_ids.add(o_node.id)


def _classify_per_layer_features(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    """Detect per-layer scalar, logit softcapping, and other features."""
    if not classification.repeated_module_names:
        return
    repeated_mod_name = max(
        classification.repeated_module_names,
        key=lambda n: len(modules_by_name[n].nodes) if n in modules_by_name else 0,
    )
    repeated_mod = modules_by_name.get(repeated_mod_name)
    if repeated_mod is None:
        return

    for node in repeated_mod.nodes:
        if node.op.name in ("Params.param", "Params.param_scale"):
            classification.per_layer_scalar_node_id = node.id
            if node.op.name == "Params.param_scale" and node.inputs:
                inp0 = node.inputs[0]
                if hasattr(inp0, "op") and inp0.op.name == "core.binary.+":
                    classification.per_layer_scalar_has_residual_add = True
            break

    main_module = modules_by_name.get(program.main_module)
    if main_module is not None:
        for node in main_module.nodes:
            if node.op.name == "Activations.tanh":
                tanh_input = node.inputs[0] if node.inputs else None
                if isinstance(tanh_input, GraphExpr) and tanh_input.op.name == "core.binary./":
                    for out_node in main_module.nodes:
                        if (
                            out_node.op.name == "core.binary.*"
                            and len(out_node.inputs) >= 2
                        ):
                            left, right = out_node.inputs[0], out_node.inputs[1]
                            left_name = _value_name(left)
                            out_name = _node_output_name(node)
                            if left_name and out_name and left_name == out_name:
                                cap = _literal_value(right, None)
                                if isinstance(cap, (int, float)):
                                    classification.logit_softcap = float(cap)
                                break

    # Detect PLI (per-layer inputs) modules
    _pli_leaf_names = {"per_layer_input_gate", "per_layer_projection",
                       "per_layer_model_projection", "embed_tokens_per_layer",
                       "per_layer_projection_norm", "post_per_layer_input_norm"}

    def _check_pli_path(node, modules_by_name):
        """Check node and its called module for PLI path leaves."""
        for inp in node.inputs:
            if isinstance(inp, GraphPath) and inp.parts:
                for part in inp.parts:
                    if part in _pli_leaf_names:
                        return part
        # Also check inside the called module's definition nodes
        called_mod = modules_by_name.get(node.op.name)
        if called_mod is not None:
            for inner_node in called_mod.nodes:
                for inp in inner_node.inputs:
                    if isinstance(inp, GraphPath) and inp.parts:
                        for part in inp.parts:
                            if part in _pli_leaf_names:
                                return part
        return None

    for module in program.modules:
        for node in module.nodes:
            matched_leaf = _check_pli_path(node, modules_by_name)
            if matched_leaf is None:
                continue
            if _is_linear_call(node, modules_by_name):
                if matched_leaf == "per_layer_input_gate":
                    classification.pli_gate_node_id = node.id
                elif matched_leaf == "per_layer_projection":
                    if repeated_mod is not None and node.id.startswith(repeated_mod_name + ":"):
                        classification.pli_proj_node_id = node.id
                elif matched_leaf == "per_layer_model_projection":
                    classification.pli_model_proj_node_id = node.id
            if _is_rmsnorm_call(node, modules_by_name):
                if matched_leaf == "post_per_layer_input_norm":
                    classification.pli_norm_node_id = node.id
                elif matched_leaf == "per_layer_projection_norm":
                    classification.pli_proj_norm_node_id = node.id
            if node.op.name == "NN.embedding" or node.op.name == "_embedding":
                if matched_leaf == "embed_tokens_per_layer":
                    classification.pli_embed_node_id = node.id


def _classify_v_norms(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    """Detect v_norms: norms that produce the 'v' output of a module with (k, v) outputs.

    Also detect k_eq_v: a KV module that has only one linear (k_proj) and no v_proj,
    meaning k and v come from the same projection.
    """
    for module in program.modules:
        out_names = [_value_name(o) for o in module.outputs]
        if "k" not in out_names or "v" not in out_names:
            continue
        v_out_idx = out_names.index("v")
        k_out_idx = out_names.index("k")
        v_out_name = _value_name(module.outputs[v_out_idx])
        k_out_name = _value_name(module.outputs[k_out_idx])
        if v_out_name is None or k_out_name is None:
            continue

        linear_count = 0
        has_v_proj = False
        for node in module.nodes:
            if _is_linear_call(node, modules_by_name):
                linear_count += 1
                leaf = _linear_path_leaf(node)
                if leaf == "v_proj":
                    has_v_proj = True
            if node.id in classification.rmsnorm_node_ids:
                node_out_name = _node_output_name(node)
                if node_out_name == v_out_name and node.id in classification.qk_norm_node_ids:
                    classification.qk_norm_node_ids.discard(node.id)
                    classification.v_norm_node_ids.add(node.id)

        if linear_count == 1 and not has_v_proj:
            classification.has_k_eq_v = True


def _classify_remaining_linears(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    """Classify unclassified linear calls as ColumnParallelLinear or RowParallelLinear.

    Heuristic: if the path leaf suggests projecting back to hidden_size
    (e.g. 'down_proj', 'per_layer_projection', 'o_proj'), classify as
    RowParallelLinear; otherwise ColumnParallelLinear.
    """
    row_hints = ("down_proj", "o_proj", "per_layer_projection")
    for module in program.modules:
        for node in module.nodes:
            if node.id in classification.node_types:
                continue
            if not _is_linear_call(node, modules_by_name):
                continue
            leaf = _linear_path_leaf(node)
            if leaf and any(leaf.endswith(h) for h in row_hints):
                classification.node_types[node.id] = VLLMLayerType.ROW_PARALLEL_LINEAR
            else:
                classification.node_types[node.id] = VLLMLayerType.COLUMN_PARALLEL_LINEAR


def _classify_qkv_deep_fallback(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    """Fallback: detect QKV by deep-tracing from attention intrinsics.

    Used when structural and path-based detection both fail (e.g. when
    q/k/v projections are separate and go through reshape/permute before
    reaching the attention node).
    """
    for module in program.modules:
        for node in module.nodes:
            if not _is_sdpa_intrinsic_node(node):
                continue
            if len(node.inputs) < 3:
                continue
            q_linear = _find_linear_call_for_value_deep(module, node.inputs[0], modules_by_name)
            k_linear = _find_linear_call_for_value_deep(module, node.inputs[1], modules_by_name)
            v_linear = _find_linear_call_for_value_deep(module, node.inputs[2], modules_by_name)
            if q_linear and k_linear and v_linear:
                ids = {q_linear.id, k_linear.id, v_linear.id}
                if len(ids) < 2:
                    layout = _infer_grouped_qkv_layout_from_operands(
                        module, node.inputs[0], node.inputs[1], node.inputs[2], q_linear.id
                    )
                    if layout is None:
                        layout = _infer_packed_chunk_qkv_layout_from_operands(
                            module,
                            node.inputs[0],
                            node.inputs[1],
                            node.inputs[2],
                            q_linear.id,
                            modules_by_name,
                        )
                    if layout is None:
                        layout = _infer_qkv_layout_from_operands(
                            module, node.inputs[0], node.inputs[1], node.inputs[2], modules_by_name
                        )
                else:
                    layout = _infer_qkv_layout_from_operands(
                        module, node.inputs[0], node.inputs[1], node.inputs[2], modules_by_name
                    )
                group = QKVGroup(
                    q_node_id=q_linear.id,
                    k_node_id=k_linear.id,
                    v_node_id=v_linear.id,
                    attention_node_id=node.id,
                    layout=layout,
                )
                classification.qkv_groups.append(group)
                _record_qkv_norm_roles(
                    module,
                    node.inputs[0],
                    node.inputs[1],
                    node.inputs[2],
                    modules_by_name,
                    classification,
                )
                for linear_node in [q_linear, k_linear, v_linear]:
                    if linear_node.id not in classification.node_types:
                        classification.node_types[linear_node.id] = VLLMLayerType.QKV_PARALLEL_LINEAR
                classification.node_types[node.id] = VLLMLayerType.ATTENTION
                classification.attention_node_ids.add(node.id)
                return


def _classify_ssm_mixers(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    """Detect SSM mixer modules by transitive presence of SSM primitive ops.

    A module is classified as MAMBA_MIXER if it is a repeated module that
    transitively calls modules containing SSM primitive operations
    (SSM.mamba_scan_step, SSM.causal_conv1d_full).  This is structural
    detection based on primitive operations, not definition names.
    """
    SSM_PRIMITIVES = {"SSM.mamba_scan_step", "SSM.causal_conv1d_full"}

    direct_ssm: set[str] = set()
    for name, mod in modules_by_name.items():
        for node in mod.nodes:
            if node.op.name in SSM_PRIMITIVES:
                direct_ssm.add(name)
                break
    if not direct_ssm:
        return

    calls: dict[str, set[str]] = {}
    for name, mod in modules_by_name.items():
        called: set[str] = set()
        for node in mod.nodes:
            if node.op.name in modules_by_name and node.op.name != name:
                called.add(node.op.name)
            for inp in node.inputs:
                if isinstance(inp, GraphExpr) and inp.op.name in modules_by_name:
                    called.add(inp.op.name)
        if "__loop_" in name and "_step_" in name:
            parent = name.split("__loop_")[0]
            if parent in modules_by_name:
                calls.setdefault(parent, set()).add(name)
        calls[name] = called

    def transitive(start: str, visited: set[str] | None = None) -> set[str]:
        if visited is None:
            visited = set()
        if start in visited:
            return set()
        visited.add(start)
        result: set[str] = set()
        for callee in calls.get(start, set()):
            result.add(callee)
            result |= transitive(callee, visited)
        return result

    for mod_name in classification.repeated_module_names:
        if mod_name in direct_ssm:
            classification.mamba_mixer_module_names.add(mod_name)
            continue
        reachable = transitive(mod_name)
        if reachable & direct_ssm:
            classification.mamba_mixer_module_names.add(mod_name)


def classify_graph_for_vllm(program: GraphProgram) -> VLLMLayerClassification:
    modules_by_name = {module.name: module for module in program.modules}
    provenance = infer_graph_provenance(program)
    classification = VLLMLayerClassification()
    _classify_embeddings(program, modules_by_name, classification)
    _classify_repeated_modules(program, modules_by_name, classification)
    _classify_per_layer_called_modules(program, modules_by_name, classification)
    _classify_attention_and_qkv(program, modules_by_name, provenance, classification)
    _classify_qkv_by_path(program, modules_by_name, classification)
    if not classification.qkv_groups:
        _classify_qkv_deep_fallback(program, modules_by_name, classification)
    _classify_ffn(program, modules_by_name, provenance, classification)
    _classify_output_projections(program, modules_by_name, classification)
    _classify_lm_head(program, modules_by_name, classification)
    _classify_norms(program, modules_by_name, classification)
    _classify_qkv_norm_roles_from_attention_inputs(program, modules_by_name, classification)
    _classify_remaining_linears(program, modules_by_name, classification)
    _classify_per_layer_features(program, modules_by_name, classification)
    _classify_v_norms(program, modules_by_name, classification)
    _classify_ssm_mixers(program, modules_by_name, classification)
    return classification


__all__ = [
    "VLLMLayerType",
    "QKVGroup",
    "FFNGroup",
    "VLLMLayerClassification",
    "classify_graph_for_vllm",
]
