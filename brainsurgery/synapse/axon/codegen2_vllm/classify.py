from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..graph_ir.core import (
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
    DEFAULT = "default"


@dataclass(frozen=True)
class QKVGroup:
    q_node_id: str
    k_node_id: str
    v_node_id: str
    attention_node_id: str | None = None


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
    lm_head_node_id: str | None = None
    rmsnorm_node_ids: set[str] = field(default_factory=set)

    def layer_type(self, node: GraphNode) -> VLLMLayerType:
        return self.node_types.get(node.id, VLLMLayerType.DEFAULT)


_ACTIVATION_PRIMITIVES = frozenset({
    "_activations_gelu_new",
    "_activations_gelu",
    "_activations_relu",
    "_activations_silu",
    "_activations_sigmoid",
})

_TRIVIAL_TRANSFORM_OPS = frozenset({
    "Tensor.reshape",
    "Tensor.permute",
    "Tensor.transpose",
    "Tensor.cast",
    "Tensor.expand",
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


def _module_contains_primitive(module: GraphModule, prim_name: str) -> bool:
    return any(node.op.name == prim_name for node in module.nodes)


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
    mod = _called_module(node, modules_by_name)
    if mod is None:
        return False
    return _module_contains_primitive(mod, "_embedding")


def _is_linear_call(
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
) -> bool:
    mod = _called_module(node, modules_by_name)
    if mod is None:
        return False
    return _module_contains_primitive(mod, "_linear")


def _is_layernorm_call(
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
) -> bool:
    mod = _called_module(node, modules_by_name)
    if mod is None:
        return False
    return _module_contains_primitive(mod, "_layernorm")


def _is_rmsnorm_call(
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
) -> bool:
    mod = _called_module(node, modules_by_name)
    if mod is None:
        return False
    return _module_contains_primitive(mod, "_rmsnorm")


def _is_activation_call(
    node: GraphNode,
    modules_by_name: dict[str, GraphModule],
) -> bool:
    mod = _called_module(node, modules_by_name)
    if mod is None:
        return False
    return any(n.op.name in _ACTIVATION_PRIMITIVES for n in mod.nodes)


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


def _classify_embeddings(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    main_module = next(
        (m for m in program.modules if m.name == program.main_module), None
    )
    if main_module is None:
        return
    for node in main_module.nodes:
        if not _is_embedding_call(node, modules_by_name):
            continue
        if len(node.inputs) < 2:
            continue
        x_input = node.inputs[1]
        if _is_main_input(x_input, main_module):
            classification.node_types[node.id] = VLLMLayerType.VOCAB_PARALLEL_EMBEDDING
            classification.embedding_node_ids.add(node.id)


def _classify_attention_and_qkv(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    provenance: GraphProvenanceAnalysis,
    classification: VLLMLayerClassification,
) -> None:
    for module in program.modules:
        for node in module.nodes:
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
            elif _is_sdpa_intrinsic_node(node):
                classification.node_types[node.id] = VLLMLayerType.ATTENTION
                classification.attention_node_ids.add(node.id)
                _classify_qkv_producers_from_intrinsic(
                    module=module,
                    node=node,
                    modules_by_name=modules_by_name,
                    classification=classification,
                )
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
    q_linear = _find_linear_call_for_value(module, q_actual, modules_by_name)
    k_linear = _find_linear_call_for_value(module, k_actual, modules_by_name)
    v_linear = _find_linear_call_for_value(module, v_actual, modules_by_name)
    if q_linear and k_linear and v_linear:
        group = QKVGroup(
            q_node_id=q_linear.id,
            k_node_id=k_linear.id,
            v_node_id=v_linear.id,
            attention_node_id=node.id,
        )
        classification.qkv_groups.append(group)
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
        group = QKVGroup(
            q_node_id=q_linear.id,
            k_node_id=k_linear.id,
            v_node_id=v_linear.id,
            attention_node_id=node.id,
        )
        classification.qkv_groups.append(group)
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
            classification.node_types[producer.id] = VLLMLayerType.QKV_PARALLEL_LINEAR
            attn_ids_in_module = {
                aid for aid in classification.attention_node_ids
                if any(n.id == aid for n in module.nodes)
            }
            attn_id = next(iter(attn_ids_in_module), None)
            group = QKVGroup(
                q_node_id=producer.id,
                k_node_id=producer.id,
                v_node_id=producer.id,
                attention_node_id=attn_id,
            )
            classification.qkv_groups.append(group)


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


def _classify_ffn(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    provenance: GraphProvenanceAnalysis,
    classification: VLLMLayerClassification,
) -> None:
    _classify_swiglu_ffn(program, modules_by_name, classification)
    _classify_simple_ffn(program, modules_by_name, classification)


def _classify_swiglu_ffn(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    for module in program.modules:
        nodes = list(module.nodes)
        for i, node in enumerate(nodes):
            silu_output = _find_silu_followed_by_mul(module, node, nodes, i, modules_by_name)
            if silu_output is None:
                continue
            silu_node, mul_node, mul_output = silu_output
            gate_input = silu_node.inputs[0] if silu_node.inputs else None
            up_input = None
            if mul_node and len(mul_node.inputs) >= 2:
                silu_name = _node_output_name(silu_node)
                in0_name = _value_name(mul_node.inputs[0])
                if in0_name == silu_name:
                    up_input = mul_node.inputs[1]
                else:
                    up_input = mul_node.inputs[0]
            gate_linear = _find_linear_call_for_value(module, gate_input, modules_by_name) if gate_input else None
            up_linear = _find_linear_call_for_value(module, up_input, modules_by_name) if up_input else None
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
                if gate_linear.id not in classification.node_types:
                    classification.node_types[gate_linear.id] = VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR
                if up_linear.id not in classification.node_types:
                    classification.node_types[up_linear.id] = VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR
                if down_linear.id not in classification.node_types:
                    classification.node_types[down_linear.id] = VLLMLayerType.ROW_PARALLEL_LINEAR


def _find_silu_followed_by_mul(
    module: GraphModule,
    node: GraphNode,
    nodes: list[GraphNode],
    index: int,
    modules_by_name: dict[str, GraphModule],
) -> tuple[GraphNode, GraphNode, str] | None:
    if node.op.name != "_activations_silu" and not _is_activation_call(node, modules_by_name):
        return None
    if _is_activation_call(node, modules_by_name):
        mod = _called_module(node, modules_by_name)
        if mod is None or not any(n.op.name == "_activations_silu" for n in mod.nodes):
            return None
    if len(node.inputs) < 1:
        return None
    silu_output = _node_output_name(node)
    if silu_output is None:
        return None
    for j in range(index + 1, len(nodes)):
        if nodes[j].op.name in {"_mul", "core.binary.*"} and len(nodes[j].inputs) >= 2:
            in0_name = _value_name(nodes[j].inputs[0])
            in1_name = _value_name(nodes[j].inputs[1])
            if in0_name == silu_output or in1_name == silu_output:
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
            if node.op.name == "_activations_silu":
                continue
            up_linear = None
            if node.inputs:
                up_linear = _find_linear_call_for_value(module, node.inputs[0], modules_by_name)
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
    main_module = next(
        (m for m in program.modules if m.name == program.main_module), None
    )
    if main_module is None:
        return
    output_names = {
        out.name for out in main_module.outputs
        if isinstance(out, (GraphValueRef, GraphValue))
    }
    for node in reversed(main_module.nodes):
        if not _is_linear_call(node, modules_by_name):
            continue
        if node.id in classification.node_types:
            continue
        node_output = _node_output_name(node)
        if node_output and node_output in output_names:
            classification.node_types[node.id] = VLLMLayerType.PARALLEL_LM_HEAD
            classification.lm_head_node_id = node.id
            return


def _classify_norms(
    program: GraphProgram,
    modules_by_name: dict[str, GraphModule],
    classification: VLLMLayerClassification,
) -> None:
    for module in program.modules:
        for node in module.nodes:
            if _is_layernorm_call(node, modules_by_name):
                classification.node_types[node.id] = VLLMLayerType.LAYERNORM
            elif _is_rmsnorm_call(node, modules_by_name):
                classification.node_types[node.id] = VLLMLayerType.RMSNORM
                classification.rmsnorm_node_ids.add(node.id)


def classify_graph_for_vllm(program: GraphProgram) -> VLLMLayerClassification:
    modules_by_name = {module.name: module for module in program.modules}
    provenance = infer_graph_provenance(program)
    classification = VLLMLayerClassification()
    _classify_embeddings(program, modules_by_name, classification)
    _classify_attention_and_qkv(program, modules_by_name, provenance, classification)
    _classify_ffn(program, modules_by_name, provenance, classification)
    _classify_output_projections(program, modules_by_name, classification)
    _classify_lm_head(program, modules_by_name, classification)
    _classify_norms(program, modules_by_name, classification)
    return classification


__all__ = [
    "VLLMLayerType",
    "QKVGroup",
    "FFNGroup",
    "VLLMLayerClassification",
    "classify_graph_for_vllm",
]
