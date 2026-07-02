from __future__ import annotations

import math
import hashlib
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from ...ops import get_op_semantics, get_op_type_rule
from ..ast import (
    Constraint,
    ConstraintAtom,
    ConstraintOperand,
    DimExprBinary,
    DimToken,
    TypeAny,
    TypeBool,
    TypeDim,
    TypeExpr,
    TypeFloat,
    TypeInt,
    TypeList,
    TypeNamed,
    TypeNull,
    TypeOptional,
    TypeTensor,
    TypeTuple,
    TypeString,
    TypeVar,
    dim_token_names,
)
from ..typecheck_shared import _PrimitiveTypeHelpers
from .core import (
    GraphExpr,
    GraphLiteral,
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
    graph_type_compatible,
    validate_graph_program,
    _value_ref_type,
)
from .domain import (
    GraphDomainFact,
    GraphDomainKind,
    infer_main_module_domain_facts,
    refine_graph_domain_facts_for_branch,
)
from .effects import (
    GraphEffect,
    UsageClass,
    graph_node_effect,
    graph_node_usage,
    graph_op_effect,
    graph_operand_effect,
    graph_operand_usage,
    infer_graph_module_effects,
    infer_graph_module_usages,
)
from .provenance import (
    GraphProvenance,
    GraphRopeApplyFactorsFact,
    GraphSdpaGqaFact,
    graph_provenance_facts,
    infer_graph_provenance,
)
from .substitute import (
    UnsupportedConstraintSubstitution,
    replace_constraint_refs,
    rename_operand,
    replace_operand_refs,
    substitute_dim_token,
    substitute_graph_module_dims,
    substitute_graph_node_dims,
    substitute_graph_operand_dims,
    substitute_type_expr,
)


@dataclass(frozen=True)
class GraphOptimizeConfig:
    prune_to_main: bool = True
    atomic_alias_cleanup: bool = True
    dead_temp_elimination: bool = True
    constant_folding: bool = True
    constant_dim_substitution: bool = True
    common_subexpression_elimination: bool = True
    specialize_definitions: str = "single-callsite"
    inline_safe: bool = True
    backend_intrinsics: str | None = None
    max_iterations: int = 64


_SPECIALIZE_MODES = {"off", "single-callsite", "monomorphize"}
_TORCH_BACKEND_INTRINSICS = frozenset(
    {
        "__torch_expert_packed_swiglu_ffn",
        "__torch_expert_swiglu_ffn",
        "__torch_rope_apply_factors",
        "__torch_rope_pair_apply_factors",
        "__torch_sdpa",
        "__torch_selected_expert_clamped_packed_swiglu_ffn",
        "__torch_selected_expert_packed_gegelu_ffn",
        "__torch_selected_expert_packed_swiglu_ffn",
        "__torch_selected_expert_relu2_ffn",
        "__torch_selected_expert_swiglu_ffn",
        "__torch_swiglu_ffn",
        "__torch_topk_normalize",
        "__torch_weighted_topk_sum",
    }
)
_TINYGRAD_BACKEND_INTRINSICS = frozenset(
    {
        "__tinygrad_sdpa",
    }
)
_TRITON_BACKEND_INTRINSICS = frozenset(
    {
        "__triton_rmsnorm_noscale",
        "__triton_rmsnorm_scaled",
        "__triton_rmsnorm_unit_offset_scaled",
        "__triton_geglu_tanh_activation",
        "__triton_selected_expert_packed_swiglu_ffn",
        "__triton_swiglu_activation",
    }
)
_VLLM_BACKEND_INTRINSICS = frozenset(
    {
        "__vllm_paged_attention",
    }
)
_BACKEND_INTRINSICS_BY_TARGET = {
    "codegen2-torch": _TORCH_BACKEND_INTRINSICS,
    "codegen2-tinygrad": _TINYGRAD_BACKEND_INTRINSICS,
    "codegen2-triton": _TRITON_BACKEND_INTRINSICS,
    "codegen2-vllm": _VLLM_BACKEND_INTRINSICS,
}
_BACKEND_INTRINSIC_PREFIX_BY_TARGET = {
    "codegen2-torch": "__torch_",
    "codegen2-tinygrad": "__tinygrad_",
    "codegen2-triton": "__triton_",
    "codegen2-vllm": "__vllm_",
}
_BACKEND_INTRINSIC_TARGETS = {None, *_BACKEND_INTRINSICS_BY_TARGET}
_SMALL_INLINE_NODE_LIMIT = 4


def _normalize_backend_intrinsic_name(name: str, *, target: str) -> str:
    token = name.strip()
    if not token:
        raise ValueError("empty backend intrinsic selector")
    if token in {"*", "all"}:
        return token
    allowed_intrinsics = _BACKEND_INTRINSICS_BY_TARGET[target]
    prefix = _BACKEND_INTRINSIC_PREFIX_BY_TARGET[target]
    single_prefix = prefix.removeprefix("_")
    if token.startswith(single_prefix):
        token = "_" + token
    elif not token.startswith(prefix):
        token = prefix + token
    if token not in allowed_intrinsics:
        allowed = ", ".join(sorted(allowed_intrinsics))
        raise ValueError(f"unknown {target} graph intrinsic {name!r}; expected one of: {allowed}")
    return token


def _parse_backend_intrinsics(value: str | None) -> tuple[str | None, frozenset[str]]:
    if value is None:
        return None, frozenset()
    raw = str(value).strip()
    if not raw:
        return None, frozenset()
    target, sep, selector_text = raw.partition(":")
    target = target.strip()
    if target not in _BACKEND_INTRINSIC_TARGETS or target is None:
        allowed = ", ".join(sorted(item for item in _BACKEND_INTRINSIC_TARGETS if item is not None))
        raise ValueError(
            f"unsupported graph backend intrinsics target {value!r}; expected {allowed!r} "
            "or target:intrinsic[,intrinsic...]"
        )
    allowed_intrinsics = _BACKEND_INTRINSICS_BY_TARGET[target]
    if not sep or selector_text.strip() in {"", "*", "all"}:
        return target, allowed_intrinsics
    selected = {_normalize_backend_intrinsic_name(item, target=target) for item in selector_text.split(",")}
    if "*" in selected or "all" in selected:
        return target, allowed_intrinsics
    return target, frozenset(selected)


def _backend_intrinsic_enabled(enabled: frozenset[str], name: str) -> bool:
    return name in enabled


def _is_atomic_operand(operand: GraphOperand) -> bool:
    return isinstance(operand, GraphValueRef | GraphLiteral | GraphPath)


def _path_has_template(path: GraphPath) -> bool:
    return any("{" in part or "}" in part for part in path.parts)


def _is_safe_specialization_operand(operand: GraphOperand) -> bool:
    if isinstance(operand, GraphPath):
        return True
    if isinstance(operand, GraphLiteral):
        return True
    return False


def _is_safe_callsite_specialization_operand(
    operand: GraphOperand,
    *,
    global_symbol_names: set[str],
) -> bool:
    return _is_safe_specialization_operand(operand) or (
        isinstance(operand, GraphValueRef) and operand.name in global_symbol_names
    ) or (
        isinstance(operand, GraphExpr)
        and operand.op.name in global_symbol_names
        and not operand.inputs
        and not operand.attrs
    )


def _is_safe_shared_specialization_operand(
    operand: GraphOperand,
    *,
    global_symbol_names: set[str],
) -> bool:
    if _is_safe_specialization_operand(operand):
        return True
    if isinstance(operand, GraphLiteral):
        return True
    if isinstance(operand, GraphValueRef) and operand.name in global_symbol_names:
        return True
    return (
        isinstance(operand, GraphExpr)
        and operand.op.name in global_symbol_names
        and not operand.inputs
        and not operand.attrs
    )


def _graph_value_ref_provenance(
    operand: GraphOperand,
    *,
    local_provenance: Mapping[str, GraphProvenance],
) -> GraphProvenance | None:
    if isinstance(operand, GraphValueRef):
        return local_provenance.get(operand.name)
    return None


def _has_additive_mask_from_keep_fact(
    provenance: GraphProvenance | None,
    keep_provenance: GraphProvenance | None,
) -> bool:
    if provenance is None or keep_provenance is None:
        return False
    return any(
        fact.kind == "additive_mask_from_keep" and fact.value == keep_provenance
        for fact in graph_provenance_facts(provenance)
    )


def _maybe_rewrite_node_to_backend_sdpa(
    node: GraphNode,
    *,
    module: GraphModule,
    modules_by_name: Mapping[str, GraphModule],
    provenance,
    op_name: str,
) -> GraphNode | None:
    if len(node.outputs) != 1 or node.attrs or node.op.name not in modules_by_name:
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
    sdpa_facts = [
        fact.value
        for fact in output_facts[0]
        if fact.kind == "sdpa_gqa" and isinstance(fact.value, GraphSdpaGqaFact)
    ]
    if not sdpa_facts:
        return None
    formal_to_actual = {
        formal.name: actual
        for formal, actual in zip(callee.inputs, node.inputs, strict=False)
    }
    local_provenance = provenance.module_local_provenance.get(module.name, {})
    for fact in sdpa_facts:
        try:
            q = formal_to_actual[fact.q]
            k = formal_to_actual[fact.k]
            v = formal_to_actual[fact.v]
            additive_mask = formal_to_actual[fact.additive_mask]
            keep = formal_to_actual[fact.keep]
        except KeyError:
            continue
        additive_prov = _graph_value_ref_provenance(
            additive_mask,
            local_provenance=local_provenance,
        )
        keep_prov = _graph_value_ref_provenance(keep, local_provenance=local_provenance)
        if fact.additive_mask != fact.keep and not _has_additive_mask_from_keep_fact(additive_prov, keep_prov):
            continue
        return replace(
            node,
            op=GraphOp(op_name),
            inputs=(
                q,
                k,
                v,
                additive_mask,
                GraphLiteral(value=None, type_expr=TypeNull()),
                GraphLiteral(value=True, type_expr=TypeBool()),
            ),
            attrs={},
        )
    return None


def _maybe_rewrite_node_to_torch_rope_apply_factors(
    node: GraphNode,
    *,
    module: GraphModule,
    modules_by_name: Mapping[str, GraphModule],
    provenance,
) -> GraphNode | None:
    call_rewrite = _maybe_rewrite_call_to_torch_rope_apply_factors(
        node,
        modules_by_name=modules_by_name,
        provenance=provenance,
    )
    if call_rewrite is not None:
        return call_rewrite
    if len(node.outputs) != 1:
        return None
    local_provenance = provenance.module_local_provenance.get(module.name, {})
    output_provenance = local_provenance.get(node.outputs[0].name)
    if output_provenance is None:
        return None
    if not any(
        fact.kind == "rope_apply_factors"
        and isinstance(fact.value, GraphRopeApplyFactorsFact)
        and not fact.value.interleaved
        for fact in graph_provenance_facts(output_provenance)
    ):
        return None
    operands = _match_rope_apply_factors_operands(node)
    if operands is None:
        return None
    x, sin, cos = operands
    return replace(
        node,
        op=GraphOp("__torch_rope_apply_factors"),
        inputs=(
            x,
            sin,
            cos,
            GraphLiteral(value=False, type_expr=TypeBool()),
        ),
        attrs={},
    )


def _maybe_rewrite_call_to_torch_rope_apply_factors(
    node: GraphNode,
    *,
    modules_by_name: Mapping[str, GraphModule],
    provenance,
) -> GraphNode | None:
    callee = modules_by_name.get(node.op.name)
    if callee is None or node.attrs or len(node.inputs) != len(callee.inputs):
        return None
    output_facts = [
        _single_rope_apply_fact(graph_provenance_facts(item))
        for item in provenance.module_summary_provenance.get(callee.name, ())
    ]
    if (
        len(node.outputs) == 1
        and len(output_facts) >= 1
        and output_facts[0] is not None
        and not isinstance(node.outputs[0].type_expr, TypeTuple)
    ):
        fact = output_facts[0]
        assert fact is not None
        actuals = _rope_fact_actuals(fact, callee=callee, node=node)
        if actuals is None:
            return None
        x, sin, cos = actuals
        return replace(
            node,
            op=GraphOp("__torch_rope_apply_factors"),
            inputs=(
                x,
                sin,
                cos,
                GraphLiteral(value=False, type_expr=TypeBool()),
            ),
            attrs={},
        )
    if len(node.outputs) == 2 and len(output_facts) >= 2:
        first, second = output_facts[:2]
        if first is None or second is None:
            return None
        first_actuals = _rope_fact_actuals(first, callee=callee, node=node)
        second_actuals = _rope_fact_actuals(second, callee=callee, node=node)
        if first_actuals is None or second_actuals is None:
            return None
        q, sin_a, cos_a = first_actuals
        k, sin_b, cos_b = second_actuals
        if sin_a != sin_b or cos_a != cos_b:
            return None
        return replace(
            node,
            op=GraphOp("__torch_rope_pair_apply_factors"),
            inputs=(
                q,
                k,
                sin_a,
                cos_a,
                GraphLiteral(value=False, type_expr=TypeBool()),
            ),
            attrs={},
        )
    return None


def _maybe_rewrite_node_to_assign_slice(
    node: GraphNode,
    *,
    module: GraphModule,
    provenance,
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
) -> GraphNode | None:
    if len(node.outputs) != 1:
        return None
    local_provenance = provenance.module_local_provenance.get(module.name, {})
    output_provenance = local_provenance.get(node.outputs[0].name)
    if (
        output_provenance is None
        or output_provenance.kind != "op"
        or output_provenance.op != "_scatter"
        or len(output_provenance.args) < 4
    ):
        return None
    base_prov, index_prov, src_prov, dim_prov = output_provenance.args[:4]
    if index_prov.kind != "op" or index_prov.op != "_fill" or len(index_prov.args) < 2:
        return None
    fill_base_prov, fill_value_prov = index_prov.args[:2]
    if fill_base_prov != src_prov:
        return None
    base = _provenance_to_graph_operand(base_prov, provenance_to_operand=provenance_to_operand)
    dim = _provenance_to_graph_operand(dim_prov, provenance_to_operand=provenance_to_operand)
    fill_value = _provenance_to_graph_operand(fill_value_prov, provenance_to_operand=provenance_to_operand)
    src = _provenance_to_graph_operand(src_prov, provenance_to_operand=provenance_to_operand)
    if base is None or dim is None or fill_value is None or src is None:
        return None
    base_type = getattr(base, "type_expr", None)
    if base_type is None or not graph_type_compatible(node.outputs[0].type_expr, base_type):
        return None
    src_type = getattr(src, "type_expr", None)
    if not isinstance(src_type, TypeTensor):
        return None
    end = GraphExpr(
        op=GraphOp("core.binary.+"),
        inputs=(
            fill_value,
            GraphLiteral(value=1, type_expr=TypeInt()),
        ),
        attrs={},
        type_expr=TypeInt(),
    )
    return replace(
        node,
        op=GraphOp("_assign_slice"),
        inputs=(base, src, dim, fill_value, end),
        attrs={},
    )


def _linear_provenance_args(
    output_provenance: GraphProvenance | None,
    *,
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
) -> tuple[GraphOperand, GraphOperand, GraphOperand, GraphOperand, GraphOperand, GraphOperand, GraphOperand, GraphOperand] | None:
    if output_provenance is None:
        return None
    if output_provenance.kind != "op" or output_provenance.op != "_linear":
        return None
    if len(output_provenance.args) < 8:
        return None
    converted = tuple(
        _provenance_to_graph_operand(item, provenance_to_operand=provenance_to_operand)
        for item in output_provenance.args[:8]
    )
    if any(item is None for item in converted):
        return None
    return converted  # type: ignore[return-value]


def _expert_linear_provenance_args(
    output_provenance: GraphProvenance | None,
    *,
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
) -> tuple[GraphOperand, GraphOperand, GraphOperand, GraphOperand, GraphOperand, GraphOperand, GraphOperand, GraphOperand] | None:
    if output_provenance is None:
        return None
    if output_provenance.kind != "op" or output_provenance.op != "_expert_linear":
        return None
    if len(output_provenance.args) < 8:
        return None
    converted = tuple(
        _provenance_to_graph_operand(item, provenance_to_operand=provenance_to_operand)
        for item in output_provenance.args[:8]
    )
    if any(item is None for item in converted):
        return None
    return converted  # type: ignore[return-value]


def _is_literal_provenance_value(provenance: GraphProvenance, value: object) -> bool:
    return provenance.kind == "literal" and provenance.value == value


def _graph_bool_literal_operand(operand: GraphOperand) -> GraphLiteral | None:
    if isinstance(operand, GraphLiteral) and isinstance(operand.value, bool):
        return operand
    if isinstance(operand, GraphLiteral) and operand.value in {0, 1}:
        return GraphLiteral(bool(operand.value), TypeBool())
    return None


def _is_literal_number_provenance_value(
    provenance: GraphProvenance,
    value: float,
    *,
    tolerance: float = 1e-12,
) -> bool:
    return provenance.kind == "literal" and isinstance(provenance.value, int | float) and math.isclose(
        float(provenance.value),
        value,
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def _provenance_binary_args(
    provenance: GraphProvenance,
    *op_names: str,
) -> tuple[GraphProvenance, GraphProvenance] | None:
    if provenance.kind != "op" or len(provenance.args) != 2:
        return None
    if provenance.op not in op_names:
        return None
    return provenance.args[0], provenance.args[1]


def _match_commutative_provenance_binary(
    provenance: GraphProvenance,
    op_name: str,
    *,
    left_predicate: Callable[[GraphProvenance], bool],
    right_predicate: Callable[[GraphProvenance], bool],
) -> tuple[GraphProvenance, GraphProvenance] | None:
    args = _provenance_binary_args(provenance, op_name)
    if args is None:
        return None
    left, right = args
    if left_predicate(left) and right_predicate(right):
        return left, right
    if left_predicate(right) and right_predicate(left):
        return right, left
    return None


def _path_parts_from_token(value: str) -> tuple[bool, tuple[str, ...]]:
    token = value.strip()
    if token.startswith("@@"):
        return True, tuple(part for part in token[2:].split(".") if part)
    if token.startswith("@"):
        return False, tuple(part for part in token[1:].split(".") if part)
    return False, tuple(part for part in token.split(".") if part)


def _compose_graph_path_operand(base: GraphOperand, leaf: GraphOperand) -> GraphOperand | None:
    if isinstance(leaf, GraphPath) and leaf.absolute:
        return leaf
    if isinstance(base, GraphPath):
        if isinstance(leaf, GraphPath):
            return GraphPath(base.absolute, base.parts + leaf.parts)
        if isinstance(leaf, GraphLiteral) and isinstance(leaf.value, str):
            absolute, parts = _path_parts_from_token(leaf.value)
            if absolute:
                return GraphPath(True, parts)
            return GraphPath(base.absolute, base.parts + parts)
    if isinstance(leaf, GraphLiteral) and isinstance(leaf.value, str):
        absolute, parts = _path_parts_from_token(leaf.value)
        if absolute:
            return GraphPath(True, parts)
    return None


def _graph_path_key(path: GraphPath) -> str:
    return ("@@" if path.absolute else "@") + ".".join(path.parts)


def _operand_local_provenance(
    operand: GraphOperand,
    *,
    local_provenance: Mapping[str, GraphProvenance],
) -> GraphProvenance | None:
    if isinstance(operand, GraphValueRef):
        return local_provenance.get(operand.name)
    if isinstance(operand, GraphLiteral):
        return GraphProvenance("literal", value=operand.value)
    if isinstance(operand, GraphPath):
        prefix = "@@" if operand.absolute else "@"
        return GraphProvenance("path", value=prefix + ".".join(operand.parts))
    if isinstance(operand, GraphExpr):
        args = tuple(
            _operand_local_provenance(item, local_provenance=local_provenance) or GraphProvenance("unknown")
            for item in (*operand.inputs, *operand.attrs.values())
        )
        return GraphProvenance("op", op=operand.op.name, args=args)
    return None


def _node_operand_for_provenance(
    node: GraphNode,
    provenance: GraphProvenance,
    *,
    local_provenance: Mapping[str, GraphProvenance],
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
) -> GraphOperand | None:
    matches: list[GraphOperand] = []
    for operand in (*node.inputs, *node.attrs.values()):
        operand_provenance = _operand_local_provenance(
            operand,
            local_provenance=local_provenance,
        )
        if operand_provenance == provenance:
            matches.append(operand)
    if len(matches) == 1:
        return matches[0]
    return _provenance_to_graph_operand(
        provenance,
        provenance_to_operand=provenance_to_operand,
    )


def _path_key_from_provenance(
    provenance: GraphProvenance,
    *,
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
) -> str | None:
    operand = _provenance_to_graph_operand(
        provenance,
        provenance_to_operand=provenance_to_operand,
    )
    if isinstance(operand, GraphPath):
        return _graph_path_key(operand)
    if isinstance(operand, GraphLiteral) and isinstance(operand.value, str):
        absolute, parts = _path_parts_from_token(operand.value)
        return ("@@" if absolute else "@") + ".".join(parts)
    return None


def _literal_bool_provenance_value(provenance: GraphProvenance, default: bool = False) -> bool:
    if provenance.kind == "literal":
        if provenance.value is None:
            return default
        return bool(provenance.value)
    return default


def _collect_parameter_path_keys_from_provenance(
    provenance: GraphProvenance,
    *,
    skip_provenances: frozenset[GraphProvenance],
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
    out: Counter[str],
    memo: dict[tuple[GraphProvenance, frozenset[GraphProvenance]], Counter[str]],
) -> None:
    if provenance in skip_provenances:
        return
    memo_key = (provenance, skip_provenances)
    cached = memo.get(memo_key)
    if cached is not None:
        out.update(cached)
        return
    local: Counter[str] = Counter()
    if provenance.kind == "op":
        if provenance.op is not None and provenance.op.startswith("__torch_gate_up_linear_pair") and len(provenance.args) >= 7:
            gate_key = _path_key_from_provenance(
                provenance.args[1],
                provenance_to_operand=provenance_to_operand,
            )
            up_key = _path_key_from_provenance(
                provenance.args[2],
                provenance_to_operand=provenance_to_operand,
            )
            if gate_key is not None:
                local[gate_key] += 1
            if up_key is not None:
                local[up_key] += 1
            if _literal_bool_provenance_value(provenance.args[5], default=False):
                gate_bias_key = _path_key_from_provenance(
                    provenance.args[3],
                    provenance_to_operand=provenance_to_operand,
                )
                up_bias_key = _path_key_from_provenance(
                    provenance.args[4],
                    provenance_to_operand=provenance_to_operand,
                )
                if gate_bias_key is not None:
                    local[gate_bias_key] += 1
                if up_bias_key is not None:
                    local[up_bias_key] += 1
            memo[memo_key] = local
            out.update(local)
            return
        if provenance.op is not None and provenance.op.startswith("__torch_swiglu_ffn") and len(provenance.args) >= 7:
            for index in (1, 2, 3):
                key = _path_key_from_provenance(
                    provenance.args[index],
                    provenance_to_operand=provenance_to_operand,
                )
                if key is not None:
                    local[key] += 1
            memo[memo_key] = local
            out.update(local)
            return
        if provenance.op == "_params_param" and provenance.args:
            key = _path_key_from_provenance(
                provenance.args[0],
                provenance_to_operand=provenance_to_operand,
            )
            if key is not None:
                local[key] += 1
            memo[memo_key] = local
            out.update(local)
            return
        if provenance.op == "_embedding" and provenance.args:
            base = _provenance_to_graph_operand(
                provenance.args[0],
                provenance_to_operand=provenance_to_operand,
            )
            weight_path = _compose_graph_path_operand(base, GraphPath(False, ("weight",)))
            if isinstance(weight_path, GraphPath):
                local[_graph_path_key(weight_path)] += 1
            memo[memo_key] = local
            out.update(local)
            return
        if provenance.op == "_linear" and len(provenance.args) >= 8:
            base = _provenance_to_graph_operand(
                provenance.args[0],
                provenance_to_operand=provenance_to_operand,
            )
            weight_leaf = _provenance_to_graph_operand(
                provenance.args[6],
                provenance_to_operand=provenance_to_operand,
            )
            weight_path = _compose_graph_path_operand(base, weight_leaf)
            if isinstance(weight_path, GraphPath):
                local[_graph_path_key(weight_path)] += 1
            if _literal_bool_provenance_value(provenance.args[3], default=False):
                bias_leaf = _provenance_to_graph_operand(
                    provenance.args[7],
                    provenance_to_operand=provenance_to_operand,
                )
                bias_path = _compose_graph_path_operand(base, bias_leaf)
                if isinstance(bias_path, GraphPath):
                    local[_graph_path_key(bias_path)] += 1
            memo[memo_key] = local
            out.update(local)
            return
        if provenance.op == "_expert_linear" and len(provenance.args) >= 8:
            base = _provenance_to_graph_operand(
                provenance.args[0],
                provenance_to_operand=provenance_to_operand,
            )
            weight_leaf = _provenance_to_graph_operand(
                provenance.args[6],
                provenance_to_operand=provenance_to_operand,
            )
            weight_path = _compose_graph_path_operand(base, weight_leaf)
            if isinstance(weight_path, GraphPath):
                local[_graph_path_key(weight_path)] += 1
            if _literal_bool_provenance_value(provenance.args[4], default=False):
                bias_leaf = _provenance_to_graph_operand(
                    provenance.args[7],
                    provenance_to_operand=provenance_to_operand,
                )
                bias_path = _compose_graph_path_operand(base, bias_leaf)
                if isinstance(bias_path, GraphPath):
                    local[_graph_path_key(bias_path)] += 1
            memo[memo_key] = local
            out.update(local)
            return
        if provenance.op == "_layernorm" and len(provenance.args) >= 7:
            base = _provenance_to_graph_operand(
                provenance.args[0],
                provenance_to_operand=provenance_to_operand,
            )
            weight_leaf = _provenance_to_graph_operand(
                provenance.args[4],
                provenance_to_operand=provenance_to_operand,
            )
            weight_path = _compose_graph_path_operand(base, weight_leaf)
            if isinstance(weight_path, GraphPath):
                local[_graph_path_key(weight_path)] += 1
            if _literal_bool_provenance_value(provenance.args[5], default=True):
                bias_leaf = _provenance_to_graph_operand(
                    provenance.args[6],
                    provenance_to_operand=provenance_to_operand,
                )
                bias_path = _compose_graph_path_operand(base, bias_leaf)
                if isinstance(bias_path, GraphPath):
                    local[_graph_path_key(bias_path)] += 1
            memo[memo_key] = local
            out.update(local)
            return
    for arg in provenance.args:
        _collect_parameter_path_keys_from_provenance(
            arg,
            skip_provenances=skip_provenances,
            provenance_to_operand=provenance_to_operand,
            out=local,
            memo=memo,
        )
    memo[memo_key] = local
    out.update(local)


def _semantic_parameter_path_counts(
    module: GraphModule,
    *,
    local_provenance: Mapping[str, GraphProvenance],
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    memo: dict[tuple[GraphProvenance, frozenset[GraphProvenance]], Counter[str]] = {}
    for node in module.nodes:
        if node.op.name == "__torch_gate_up_linear_pair" and len(node.inputs) >= 7:
            for operand in (node.inputs[1], node.inputs[2]):
                if isinstance(operand, GraphPath):
                    counts[_graph_path_key(operand)] += 1
            if isinstance(node.inputs[5], GraphLiteral) and bool(node.inputs[5].value):
                for operand in (node.inputs[3], node.inputs[4]):
                    if isinstance(operand, GraphPath):
                        counts[_graph_path_key(operand)] += 1
            continue
        if node.op.name == "__torch_swiglu_ffn" and len(node.inputs) >= 7:
            for operand in (node.inputs[1], node.inputs[2], node.inputs[3]):
                if isinstance(operand, GraphPath):
                    counts[_graph_path_key(operand)] += 1
            continue
        input_provenances = frozenset(
            item
            for operand in (*node.inputs, *node.attrs.values())
            if (item := _operand_local_provenance(operand, local_provenance=local_provenance)) is not None
        )
        for output in node.outputs:
            output_provenance = local_provenance.get(output.name)
            if output_provenance is None:
                continue
            _collect_parameter_path_keys_from_provenance(
                output_provenance,
                skip_provenances=input_provenances,
                provenance_to_operand=provenance_to_operand,
                out=counts,
                memo=memo,
            )
    return counts

def _module_provenance_to_operand_map(
    module: GraphModule,
    *,
    local_provenance: Mapping[str, GraphProvenance],
) -> dict[GraphProvenance, GraphOperand]:
    provenance_to_operand: dict[GraphProvenance, GraphOperand] = {}
    for value in module.inputs:
        value_provenance = local_provenance.get(value.name)
        if value_provenance is not None:
            provenance_to_operand.setdefault(
                value_provenance,
                GraphValueRef(value.name, value.type_expr, value.dims),
            )
    for node in module.nodes:
        for operand in (*node.inputs, *node.attrs.values()):
            _collect_literal_path_provenance_operands(operand, provenance_to_operand)
            operand_provenance = _operand_local_provenance(
                operand,
                local_provenance=local_provenance,
            )
            if operand_provenance is not None:
                provenance_to_operand.setdefault(operand_provenance, operand)
        for output in node.outputs:
            value_provenance = local_provenance.get(output.name)
            if value_provenance is not None:
                provenance_to_operand.setdefault(
                    value_provenance,
                    GraphValueRef(output.name, output.type_expr, output.dims),
                )
    return provenance_to_operand


def _linear_pair_is_dense_gate_up_candidate(
    first: GraphNode,
    second: GraphNode,
    *,
    parameter_path_counts: Mapping[str, int],
) -> tuple[GraphOperand, ...] | None:
    if first.op.name != "_linear" or second.op.name != "_linear":
        return None
    if len(first.outputs) != 1 or len(second.outputs) != 1:
        return None
    if len(first.inputs) < 8 or len(second.inputs) < 8:
        return None
    first_path, first_x, first_dim, first_bias, first_transpose, first_expert, first_weight_path, first_bias_path = first.inputs[:8]
    second_path, second_x, second_dim, second_bias, second_transpose, second_expert, second_weight_path, second_bias_path = second.inputs[:8]
    if first_x != second_x:
        return None
    if not (
        isinstance(first_bias, GraphLiteral)
        and first_bias.value is False
        and isinstance(second_bias, GraphLiteral)
        and second_bias.value is False
        and isinstance(first_transpose, GraphLiteral)
        and first_transpose.value is False
        and isinstance(second_transpose, GraphLiteral)
        and second_transpose.value is False
        and isinstance(first_expert, GraphLiteral)
        and first_expert.value is None
        and isinstance(second_expert, GraphLiteral)
        and second_expert.value is None
    ):
        return None
    if not graph_type_compatible(first.outputs[0].type_expr, second.outputs[0].type_expr):
        return None
    gate_weight_path = _compose_graph_path_operand(first_path, first_weight_path)
    up_weight_path = _compose_graph_path_operand(second_path, second_weight_path)
    gate_bias_path = _compose_graph_path_operand(first_path, first_bias_path)
    up_bias_path = _compose_graph_path_operand(second_path, second_bias_path)
    if (
        gate_weight_path is None
        or up_weight_path is None
        or gate_bias_path is None
        or up_bias_path is None
    ):
        return None
    if not isinstance(gate_weight_path, GraphPath) or not isinstance(up_weight_path, GraphPath):
        return None
    if parameter_path_counts.get(_graph_path_key(gate_weight_path), 0) != 1:
        return None
    if parameter_path_counts.get(_graph_path_key(up_weight_path), 0) != 1:
        return None
    del first_dim, second_dim
    return (
        first_x,
        gate_weight_path,
        up_weight_path,
        gate_bias_path,
        up_bias_path,
        first_bias,
        first_transpose,
    )


def _packed_gate_up_path(module_name: str, node: GraphNode, gate_path: GraphPath, up_path: GraphPath) -> GraphPath:
    base = "".join(ch if ch.isalnum() else "_" for ch in module_name).strip("_") or "module"
    node_id = "".join(ch if ch.isalnum() else "_" for ch in node.id).strip("_") or "node"
    return GraphPath(
        True,
        (
            "__packed",
            "linear_pair",
            base,
            node_id,
            "gate",
            *gate_path.parts,
            "up",
            *up_path.parts,
        ),
    )


def _double_last_tensor_dim(type_expr: TypeExpr) -> tuple[TypeExpr, tuple[DimToken, ...] | None] | None:
    if not isinstance(type_expr, TypeTensor) or not type_expr.dims:
        return None
    dims = tuple(type_expr.dims[:-1]) + (DimExprBinary("*", 2, type_expr.dims[-1]),)
    return TypeTensor(type_expr.base, dims), dims


def _sum_dim_tokens(tokens: tuple[DimToken, ...]) -> DimToken | None:
    if not tokens:
        return None
    total = tokens[0]
    for token in tokens[1:]:
        total = substitute_dim_token(DimExprBinary("+", total, token), {})
    return total


def _linear_pack_path(
    module_name: str,
    nodes: tuple[GraphNode, ...],
    paths: tuple[GraphPath, ...],
    *,
    leaf: str,
) -> GraphPath:
    base = "".join(ch if ch.isalnum() else "_" for ch in module_name).strip("_") or "module"
    node_ids = "_".join("".join(ch if ch.isalnum() else "_" for ch in node.id).strip("_") for node in nodes)
    path_parts: list[str] = ["__packed", "linear_pack", base, node_ids or "nodes"]
    for index, path in enumerate(paths, start=1):
        path_parts.append(f"p{index}")
        source_parts = path.parts[:-1] if path.parts and path.parts[-1] in {"weight", "bias"} else path.parts
        path_parts.extend(source_parts)
    path_parts.append(leaf)
    return GraphPath(True, tuple(path_parts))


def _linear_pack_leafs(path: GraphPath) -> tuple[GraphPath, GraphPath] | None:
    if not path.parts:
        return None
    return (
        GraphPath(path.absolute, path.parts[:-1]),
        GraphPath(False, (path.parts[-1],)),
    )


def _linear_pack_type(outputs: tuple[GraphValue, ...]) -> tuple[TypeExpr, tuple[DimToken, ...], DimToken] | None:
    if not outputs:
        return None
    output_types = tuple(output.type_expr for output in outputs)
    if not all(isinstance(tp, TypeTensor) and tp.dims for tp in output_types):
        return None
    tensor_types = tuple(tp for tp in output_types if isinstance(tp, TypeTensor))
    prefix = tensor_types[0].dims[:-1]
    if any(tp.base != tensor_types[0].base or tp.dims[:-1] != prefix for tp in tensor_types):
        return None
    last_dims = tuple(tp.dims[-1] for tp in tensor_types)
    total_dim = _sum_dim_tokens(last_dims)
    if total_dim is None:
        return None
    dims = (*prefix, total_dim)
    return TypeTensor(tensor_types[0].base, dims), dims, total_dim


def _linear_weight_pack_dim(transpose: GraphOperand) -> int:
    transpose_literal = _graph_bool_literal_operand(transpose)
    return -1 if transpose_literal is not None and bool(transpose_literal.value) else -2


def _linear_pack_candidate(
    module: GraphModule,
    start_index: int,
    *,
    parameter_path_counts: Mapping[str, int],
) -> tuple[tuple[int, ...], tuple[GraphOperand, GraphOperand, GraphOperand, GraphLiteral | None, tuple[GraphValue, ...], tuple[GraphPath, ...], tuple[GraphPath, ...] | None]] | None:
    first = module.nodes[start_index]
    if first.op.name != "_linear" or len(first.outputs) != 1 or len(first.inputs) < 8:
        return None
    first_path, first_x, _first_dim, first_bias, first_transpose, first_expert, first_weight_leaf, first_bias_leaf = first.inputs[:8]
    if not (
        isinstance(first_transpose, GraphLiteral)
        and first_transpose.value is False
        and isinstance(first_expert, GraphLiteral)
        and first_expert.value is None
    ):
        return None
    first_weight_path = _compose_graph_path_operand(first_path, first_weight_leaf)
    first_bias_path = _compose_graph_path_operand(first_path, first_bias_leaf)
    if not isinstance(first_weight_path, GraphPath):
        return None
    if parameter_path_counts.get(_graph_path_key(first_weight_path), 0) != 1:
        return None
    bias_enabled = isinstance(first_bias, GraphLiteral) and bool(first_bias.value)
    if bias_enabled:
        if not isinstance(first_bias_path, GraphPath):
            return None
        if parameter_path_counts.get(_graph_path_key(first_bias_path), 0) != 1:
            return None
    nodes: list[GraphNode] = [first]
    indexes: list[int] = [start_index]
    weight_paths: list[GraphPath] = [first_weight_path]
    bias_paths: list[GraphPath] = [first_bias_path] if bias_enabled and isinstance(first_bias_path, GraphPath) else []
    for index in range(start_index + 1, len(module.nodes)):
        node = module.nodes[index]
        if node.op.name != "_linear" or len(node.outputs) != 1 or len(node.inputs) < 8:
            continue
        path, x, _dim, bias, transpose, expert, weight_leaf, bias_leaf = node.inputs[:8]
        if x != first_x or bias != first_bias or transpose != first_transpose or expert != first_expert:
            continue
        if not (
            isinstance(transpose, GraphLiteral)
            and transpose.value is False
            and isinstance(expert, GraphLiteral)
            and expert.value is None
        ):
            continue
        weight_path = _compose_graph_path_operand(path, weight_leaf)
        bias_path = _compose_graph_path_operand(path, bias_leaf)
        if not isinstance(weight_path, GraphPath):
            continue
        if weight_path in weight_paths:
            continue
        if parameter_path_counts.get(_graph_path_key(weight_path), 0) != 1:
            # This is a projection in the same candidate run, but its parameter
            # tensor has another semantic read. Do not pack around it.
            break
        if bias_enabled:
            if not isinstance(bias_path, GraphPath):
                continue
            if bias_path in bias_paths:
                continue
            if parameter_path_counts.get(_graph_path_key(bias_path), 0) != 1:
                # Same as for weights: a reused bias blocks the whole pack.
                break
        nodes.append(node)
        indexes.append(index)
        weight_paths.append(weight_path)
        if bias_enabled and isinstance(bias_path, GraphPath):
            bias_paths.append(bias_path)
        if len(nodes) == 3:
            break
    if len(nodes) < 2:
        return None
    outputs = tuple(node.outputs[0] for node in nodes)
    if _linear_pack_type(outputs) is None:
        return None
    return (
        tuple(indexes),
        (
            first_x,
            first_bias,
            first_transpose,
            first_expert,
            outputs,
            tuple(weight_paths),
            tuple(bias_paths) if bias_enabled else None,
        ),
    )


def _collect_direct_graph_path_keys(operand: GraphOperand, counts: Counter[str]) -> None:
    if isinstance(operand, GraphPath):
        counts[_graph_path_key(operand)] += 1
        return
    if isinstance(operand, GraphExpr):
        for item in operand.inputs:
            _collect_direct_graph_path_keys(item, counts)
        for item in operand.attrs.values():
            _collect_direct_graph_path_keys(item, counts)


def _direct_parameter_path_counts(graph: GraphProgram) -> Counter[str]:
    counts: Counter[str] = Counter()
    for module in graph.modules:
        for node in module.nodes:
            for operand in (*node.inputs, *node.attrs.values()):
                _collect_direct_graph_path_keys(operand, counts)
            if node.op.name == "_linear" and len(node.inputs) >= 8:
                base, _x, _dim, bias, _transpose, _expert, weight_leaf, bias_leaf = node.inputs[:8]
                weight_path = _compose_graph_path_operand(base, weight_leaf)
                if isinstance(weight_path, GraphPath):
                    counts[_graph_path_key(weight_path)] += 1
                if isinstance(bias, GraphLiteral) and bool(bias.value):
                    bias_path = _compose_graph_path_operand(base, bias_leaf)
                    if isinstance(bias_path, GraphPath):
                        counts[_graph_path_key(bias_path)] += 1
            elif node.op.name == "_layernorm" and len(node.inputs) >= 7:
                base, _x, _eps, _dim, weight_leaf, bias, bias_leaf = node.inputs[:7]
                weight_path = _compose_graph_path_operand(base, weight_leaf)
                if isinstance(weight_path, GraphPath):
                    counts[_graph_path_key(weight_path)] += 1
                if isinstance(bias, GraphLiteral) and bool(bias.value):
                    bias_path = _compose_graph_path_operand(base, bias_leaf)
                    if isinstance(bias_path, GraphPath):
                        counts[_graph_path_key(bias_path)] += 1
    return counts


def _rewrite_linear_projection_packs(graph: GraphProgram) -> GraphProgram:
    parameter_path_counts = _direct_parameter_path_counts(graph)
    changed = False
    new_modules: list[GraphModule] = []
    packed_parameters: list[GraphPackedParameter] = []
    for module in graph.modules:
        replacement_by_start: dict[int, tuple[tuple[int, ...], tuple[GraphNode, GraphNode], tuple[GraphPackedParameter, ...]]] = {}
        consumed: set[int] = set()
        for index in range(len(module.nodes)):
            if index in consumed:
                continue
            candidate = _linear_pack_candidate(
                module,
                index,
                parameter_path_counts=parameter_path_counts,
            )
            if candidate is None:
                continue
            indexes, (x, bias, transpose, expert, outputs, weight_paths, bias_paths) = candidate
            pack_type = _linear_pack_type(outputs)
            if pack_type is None:
                continue
            combined_type_expr, combined_dims, total_dim = pack_type
            packed_weight_path = _linear_pack_path(
                module.name,
                tuple(module.nodes[i] for i in indexes),
                weight_paths,
                leaf="weight",
            )
            leafs = _linear_pack_leafs(packed_weight_path)
            if leafs is None:
                continue
            packed_base_path, packed_weight_leaf = leafs
            packed_bias_leaf: GraphOperand = GraphLiteral(value=None, type_expr=TypeNull())
            specs = [
                GraphPackedParameter(
                    output=packed_weight_path,
                    inputs=weight_paths,
                    dim=_linear_weight_pack_dim(transpose),
                    remove_inputs=True,
                )
            ]
            if bias_paths is not None:
                packed_bias_path = _linear_pack_path(
                    module.name,
                    tuple(module.nodes[i] for i in indexes),
                    bias_paths,
                    leaf="bias",
                )
                bias_leafs = _linear_pack_leafs(packed_bias_path)
                if bias_leafs is None:
                    continue
                bias_base, packed_bias_leaf = bias_leafs
                if bias_base != packed_base_path:
                    continue
                specs.append(
                    GraphPackedParameter(
                        output=packed_bias_path,
                        inputs=bias_paths,
                        dim=-1,
                        remove_inputs=True,
                    )
                )
            combined_output = GraphValue(
                name=f"{outputs[0].name}__linear_pack",
                type_expr=combined_type_expr,
                dims=combined_dims,
            )
            linear_node = replace(
                module.nodes[indexes[0]],
                op=GraphOp("_linear"),
                inputs=(
                    packed_base_path,
                    x,
                    _dim_token_operand(total_dim, TypeDim()),
                    bias,
                    transpose,
                    expert,
                    packed_weight_leaf,
                    packed_bias_leaf,
                ),
                attrs={},
                outputs=(combined_output,),
                type_expr=combined_type_expr,
                dims=combined_dims,
            )
            sizes = GraphExpr(
                op=GraphOp("core.list"),
                inputs=tuple(_dim_token_operand(output.dims[-1], TypeDim()) for output in outputs if output.dims),
                attrs={},
                type_expr=TypeList(TypeDim()),
            )
            if len(sizes.inputs) != len(outputs):
                continue
            split_node = replace(
                module.nodes[indexes[0]],
                id=f"{module.nodes[indexes[0]].id}:linear_pack_split",
                op=GraphOp("_split"),
                inputs=(
                    GraphValueRef(combined_output.name, combined_output.type_expr, combined_output.dims),
                    GraphLiteral(value=-1, type_expr=TypeInt()),
                    sizes,
                ),
                attrs={},
                outputs=outputs,
                type_expr=TypeTuple(tuple(output.type_expr for output in outputs)),
                dims=None,
            )
            replacement_by_start[indexes[0]] = (indexes, (linear_node, split_node), tuple(specs))
            consumed.update(indexes[1:])
        if not replacement_by_start:
            new_modules.append(module)
            continue
        changed = True
        nodes: list[GraphNode] = []
        for index, node in enumerate(module.nodes):
            replacement = replacement_by_start.get(index)
            if replacement is not None:
                _indexes, replacement_nodes, specs = replacement
                nodes.extend(replacement_nodes)
                packed_parameters.extend(specs)
                continue
            if index in consumed:
                continue
            nodes.append(node)
        new_modules.append(replace(module, nodes=tuple(nodes)))
    if not changed:
        return graph
    return replace(
        graph,
        modules=tuple(new_modules),
        packed_parameters=tuple((*graph.packed_parameters, *packed_parameters)),
    )


def _rewrite_dense_gate_up_linear_pairs(graph: GraphProgram) -> GraphProgram:
    parameter_path_counts = _direct_parameter_path_counts(graph)
    changed = False
    new_modules: list[GraphModule] = []
    packed_parameters: list[GraphPackedParameter] = []
    for module in graph.modules:
        new_nodes: list[GraphNode] = []
        index = 0
        while index < len(module.nodes):
            node = module.nodes[index]
            if index + 1 >= len(module.nodes):
                new_nodes.append(node)
                index += 1
                continue
            next_node = module.nodes[index + 1]
            inputs = _linear_pair_is_dense_gate_up_candidate(
                node,
                next_node,
                parameter_path_counts=parameter_path_counts,
            )
            if inputs is None:
                new_nodes.append(node)
                index += 1
                continue
            x, gate_weight_path, up_weight_path, gate_bias_path, up_bias_path, bias, transpose = inputs
            del gate_bias_path, up_bias_path
            if not isinstance(gate_weight_path, GraphPath) or not isinstance(up_weight_path, GraphPath):
                new_nodes.append(node)
                index += 1
                continue
            combined_type = _double_last_tensor_dim(node.outputs[0].type_expr)
            if combined_type is None:
                new_nodes.append(node)
                index += 1
                continue
            combined_type_expr, combined_dims = combined_type
            combined_dim_arg = (
                _dim_token_operand(combined_dims[-1], TypeDim())
                if combined_dims
                else GraphLiteral(value=None, type_expr=TypeNull())
            )
            packed_weight_path = _packed_gate_up_path(module.name, node, gate_weight_path, up_weight_path)
            combined_output = GraphValue(
                name=f"{node.outputs[0].name}__gate_up",
                type_expr=combined_type_expr,
                dims=combined_dims,
            )
            if not packed_weight_path.parts:
                new_nodes.append(node)
                index += 1
                continue
            packed_base_path = GraphPath(
                packed_weight_path.absolute,
                packed_weight_path.parts[:-1],
            )
            packed_weight_leaf = GraphPath(False, (packed_weight_path.parts[-1],))
            chunk_type = TypeTuple((node.outputs[0].type_expr, next_node.outputs[0].type_expr))
            changed = True
            packed_parameters.append(
                GraphPackedParameter(
                    output=packed_weight_path,
                    inputs=(gate_weight_path, up_weight_path),
                    dim=_linear_weight_pack_dim(transpose),
                    remove_inputs=True,
                )
            )
            new_nodes.append(
                replace(
                    node,
                    op=GraphOp("_linear"),
                    inputs=(
                        packed_base_path,
                        x,
                        combined_dim_arg,
                        bias,
                        transpose,
                        GraphLiteral(value=None, type_expr=TypeNull()),
                        packed_weight_leaf,
                        GraphLiteral(value=None, type_expr=TypeNull()),
                    ),
                    attrs={},
                    outputs=(combined_output,),
                    type_expr=combined_type_expr,
                    dims=combined_dims,
                )
            )
            new_nodes.append(
                replace(
                    next_node,
                    op=GraphOp("_chunk"),
                    inputs=(
                        GraphValueRef(combined_output.name, combined_output.type_expr, combined_output.dims),
                        GraphLiteral(value=-1, type_expr=TypeInt()),
                        GraphLiteral(value=2, type_expr=TypeDim()),
                    ),
                    attrs={},
                    outputs=(node.outputs[0], next_node.outputs[0]),
                    type_expr=chunk_type,
                    dims=None,
                )
            )
            index += 2
        new_modules.append(replace(module, nodes=tuple(new_nodes)))
    if not changed:
        return graph
    return replace(
        graph,
        modules=tuple(new_modules),
        packed_parameters=tuple((*graph.packed_parameters, *packed_parameters)),
    )


def _value_ref_name(operand: GraphOperand) -> str | None:
    return operand.name if isinstance(operand, GraphValueRef) else None


def _module_value_ref_counts(module: GraphModule) -> Counter[str]:
    counts: Counter[str] = Counter()
    path_template_refs: set[str] = set()
    for node in module.nodes:
        for operand in (*node.inputs, *node.attrs.values()):
            _collect_value_ref_counts(
                operand,
                counts=counts,
                path_template_refs=path_template_refs,
            )
    for output in module.outputs:
        _collect_value_ref_counts(
            output,
            counts=counts,
            path_template_refs=path_template_refs,
        )
    return counts


def _torch_swiglu_ffn_candidate(
    gate_up_node: GraphNode,
    silu_node: GraphNode,
    mul_node: GraphNode,
    down_node: GraphNode,
    *,
    value_ref_counts: Mapping[str, int],
    parameter_path_counts: Mapping[str, int],
) -> tuple[GraphOperand, ...] | None:
    if gate_up_node.op.name != "__torch_gate_up_linear_pair":
        return None
    if silu_node.op.name != "_activations_silu":
        return None
    if mul_node.op.name not in {"_mul", "core.binary.*"}:
        return None
    if down_node.op.name != "_linear":
        return None
    if len(gate_up_node.outputs) != 2 or len(silu_node.outputs) != 1 or len(mul_node.outputs) != 1 or len(down_node.outputs) != 1:
        return None
    if len(silu_node.inputs) != 1 or len(mul_node.inputs) != 2 or len(down_node.inputs) < 8:
        return None
    gate_name = gate_up_node.outputs[0].name
    up_name = gate_up_node.outputs[1].name
    silu_name = silu_node.outputs[0].name
    mul_name = mul_node.outputs[0].name
    if value_ref_counts.get(gate_name, 0) != 1:
        return None
    if value_ref_counts.get(up_name, 0) != 1:
        return None
    if value_ref_counts.get(silu_name, 0) != 1:
        return None
    if value_ref_counts.get(mul_name, 0) != 1:
        return None
    gate_ref = GraphValueRef(gate_name, gate_up_node.outputs[0].type_expr, gate_up_node.outputs[0].dims)
    up_ref = GraphValueRef(up_name, gate_up_node.outputs[1].type_expr, gate_up_node.outputs[1].dims)
    silu_ref = GraphValueRef(silu_name, silu_node.outputs[0].type_expr, silu_node.outputs[0].dims)
    if silu_node.inputs[0] != gate_ref:
        return None
    if not ((mul_node.inputs[0] == silu_ref and mul_node.inputs[1] == up_ref) or (mul_node.inputs[1] == silu_ref and mul_node.inputs[0] == up_ref)):
        return None
    mul_ref = GraphValueRef(mul_name, mul_node.outputs[0].type_expr, mul_node.outputs[0].dims)
    down_base, down_x, _down_dim, down_bias, down_transpose, down_expert, down_weight_leaf, down_bias_leaf = down_node.inputs[:8]
    if down_x != mul_ref:
        return None
    if not (
        isinstance(down_bias, GraphLiteral)
        and down_bias.value is False
        and isinstance(down_transpose, GraphLiteral)
        and down_transpose.value is False
        and isinstance(down_expert, GraphLiteral)
        and down_expert.value is None
    ):
        return None
    if len(gate_up_node.inputs) < 7:
        return None
    gate_weight_path = gate_up_node.inputs[1]
    up_weight_path = gate_up_node.inputs[2]
    if not isinstance(gate_weight_path, GraphPath) or not isinstance(up_weight_path, GraphPath):
        return None
    if parameter_path_counts.get(_graph_path_key(gate_weight_path), 0) != 1:
        return None
    if parameter_path_counts.get(_graph_path_key(up_weight_path), 0) != 1:
        return None
    down_weight_path = _compose_graph_path_operand(down_base, down_weight_leaf)
    down_bias_path = _compose_graph_path_operand(down_base, down_bias_leaf)
    if not isinstance(down_weight_path, GraphPath) or not isinstance(down_bias_path, GraphPath):
        return None
    del down_bias, down_transpose, down_expert
    return (
        gate_up_node.inputs[0],
        gate_weight_path,
        up_weight_path,
        down_weight_path,
        gate_up_node.inputs[3],
        gate_up_node.inputs[4],
        down_bias_path,
    )


def _rewrite_torch_swiglu_ffn_intrinsics(graph: GraphProgram) -> GraphProgram:
    parameter_path_counts = _direct_parameter_path_counts(graph)
    changed = False
    new_modules: list[GraphModule] = []
    for module in graph.modules:
        value_ref_counts = _module_value_ref_counts(module)
        new_nodes: list[GraphNode] = []
        index = 0
        while index < len(module.nodes):
            if index + 3 >= len(module.nodes):
                new_nodes.append(module.nodes[index])
                index += 1
                continue
            inputs = _torch_swiglu_ffn_candidate(
                module.nodes[index],
                module.nodes[index + 1],
                module.nodes[index + 2],
                module.nodes[index + 3],
                value_ref_counts=value_ref_counts,
                parameter_path_counts=parameter_path_counts,
            )
            if inputs is None:
                new_nodes.append(module.nodes[index])
                index += 1
                continue
            down_node = module.nodes[index + 3]
            changed = True
            new_nodes.append(
                replace(
                    down_node,
                    op=GraphOp("__torch_swiglu_ffn"),
                    inputs=inputs,
                    attrs={},
                )
            )
            index += 4
        new_modules.append(replace(module, nodes=tuple(new_nodes)))
    return replace(graph, modules=tuple(new_modules)) if changed else graph


def _triton_swiglu_activation_candidate(
    mul_node: GraphNode,
    *,
    output_provenance: GraphProvenance | None,
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
) -> tuple[GraphOperand, GraphOperand] | None:
    if mul_node.op.name not in {"_mul", "core.binary.*"}:
        return None
    if len(mul_node.outputs) != 1:
        return None
    if output_provenance is None or output_provenance.kind != "op":
        return None
    if output_provenance.op not in {"_mul", "core.binary.*"} or len(output_provenance.args) != 2:
        return None
    left, right = output_provenance.args
    if left.kind == "op" and left.op == "_activations_silu" and len(left.args) == 1:
        gate = _provenance_to_graph_operand(left.args[0], provenance_to_operand=provenance_to_operand)
        up = _provenance_to_graph_operand(right, provenance_to_operand=provenance_to_operand)
        return (gate, up) if gate is not None and up is not None else None
    if right.kind == "op" and right.op == "_activations_silu" and len(right.args) == 1:
        gate = _provenance_to_graph_operand(right.args[0], provenance_to_operand=provenance_to_operand)
        up = _provenance_to_graph_operand(left, provenance_to_operand=provenance_to_operand)
        return (gate, up) if gate is not None and up is not None else None
    return None


def _triton_geglu_tanh_activation_candidate(
    mul_node: GraphNode,
    *,
    output_provenance: GraphProvenance | None,
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
) -> tuple[GraphOperand, GraphOperand] | None:
    if mul_node.op.name not in {"_mul", "core.binary.*"}:
        return None
    if len(mul_node.outputs) != 1:
        return None
    if output_provenance is None or output_provenance.kind != "op":
        return None
    if output_provenance.op not in {"_mul", "core.binary.*"} or len(output_provenance.args) != 2:
        return None

    def match_order(
        activated: GraphProvenance,
        up_provenance: GraphProvenance,
    ) -> tuple[GraphOperand, GraphOperand] | None:
        if (
            activated.kind != "op"
            or activated.op not in {"_activations_gelu_new", "_activations_gelu_pytorch_tanh"}
            or len(activated.args) != 1
        ):
            return None
        gate = _provenance_to_graph_operand(activated.args[0], provenance_to_operand=provenance_to_operand)
        up = _provenance_to_graph_operand(up_provenance, provenance_to_operand=provenance_to_operand)
        if gate is None or up is None:
            return None
        if not isinstance(graph_operand_type(gate), TypeTensor):
            return None
        if not isinstance(graph_operand_type(up), TypeTensor):
            return None
        return gate, up

    left, right = output_provenance.args
    return match_order(left, right) or match_order(right, left)


def _triton_rmsnorm_noscale_candidate(
    node: GraphNode,
    *,
    output_provenance: GraphProvenance | None,
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
) -> tuple[GraphOperand, GraphOperand, GraphOperand, GraphOperand] | None:
    if len(node.outputs) != 1 or len(node.inputs) < 4:
        return None
    if output_provenance is None or output_provenance.kind != "op":
        return None
    if output_provenance.op != "_rmsnorm" or len(output_provenance.args) < 4:
        return None
    x = _provenance_to_graph_operand(output_provenance.args[0], provenance_to_operand=provenance_to_operand)
    eps = _provenance_to_graph_operand(output_provenance.args[1], provenance_to_operand=provenance_to_operand)
    dim = _provenance_to_graph_operand(output_provenance.args[2], provenance_to_operand=provenance_to_operand)
    cast_float = _provenance_to_graph_operand(output_provenance.args[3], provenance_to_operand=provenance_to_operand)
    if x is None or eps is None or dim is None or cast_float is None:
        return None
    if not (
        isinstance(dim, GraphLiteral)
        and (dim.value is None or dim.value == -1)
    ):
        return None
    return x, eps, dim, cast_float


def _triton_rmsnorm_scaled_candidate(
    node: GraphNode,
    *,
    output_provenance: GraphProvenance | None,
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
) -> tuple[GraphOperand, GraphOperand, GraphOperand, GraphOperand, GraphOperand] | None:
    if len(node.outputs) != 1:
        return None
    if output_provenance is None or output_provenance.kind != "op":
        return None
    if output_provenance.op not in {"_mul", "core.binary.*"} or len(output_provenance.args) != 2:
        return None

    def match_order(
        rmsnorm: GraphProvenance,
        scale: GraphProvenance,
    ) -> tuple[GraphOperand, GraphOperand, GraphOperand, GraphOperand, GraphOperand] | None:
        if (
            rmsnorm.kind != "op"
            or rmsnorm.op not in {"_rmsnorm", "__triton_rmsnorm_noscale"}
            or len(rmsnorm.args) < 4
        ):
            return None
        if scale.kind != "op" or scale.op != "_params_param" or len(scale.args) < 1:
            return None
        x = _provenance_to_graph_operand(rmsnorm.args[0], provenance_to_operand=provenance_to_operand)
        eps = _provenance_to_graph_operand(rmsnorm.args[1], provenance_to_operand=provenance_to_operand)
        dim = _provenance_to_graph_operand(rmsnorm.args[2], provenance_to_operand=provenance_to_operand)
        cast_float = _provenance_to_graph_operand(rmsnorm.args[3], provenance_to_operand=provenance_to_operand)
        scale_operand = _provenance_to_graph_operand(scale, provenance_to_operand=provenance_to_operand)
        if scale_operand is None:
            scale_operand = _provenance_to_graph_operand(scale.args[0], provenance_to_operand=provenance_to_operand)
            if scale_operand is not None and isinstance(graph_operand_type(scale_operand), TypeTensor):
                scale_operand = None
        if x is None or eps is None or dim is None or cast_float is None or scale_operand is None:
            return None
        if not isinstance(graph_operand_type(x), TypeTensor):
            return None
        if not (
            isinstance(dim, GraphLiteral)
            and (dim.value is None or dim.value == -1)
        ):
            return None
        return x, scale_operand, eps, dim, cast_float

    left, right = output_provenance.args
    return match_order(left, right) or match_order(right, left)


def _rewrite_triton_rmsnorm_scaled_intrinsics(graph: GraphProgram) -> GraphProgram:
    provenance = infer_graph_provenance(graph)
    changed = False
    new_modules: list[GraphModule] = []
    for module in graph.modules:
        local_provenance = provenance.module_local_provenance.get(module.name, {})
        provenance_to_operand = _module_provenance_to_operand_map(
            module,
            local_provenance=local_provenance,
        )
        new_nodes: list[GraphNode] = []
        for node in module.nodes:
            output_provenance = local_provenance.get(node.outputs[0].name) if len(node.outputs) == 1 else None
            inputs = _triton_rmsnorm_scaled_candidate(
                node,
                output_provenance=output_provenance,
                provenance_to_operand=provenance_to_operand,
            )
            if inputs is None:
                new_nodes.append(node)
                continue
            changed = True
            new_nodes.append(
                replace(
                    node,
                    op=GraphOp("__triton_rmsnorm_scaled"),
                    inputs=inputs,
                    attrs={},
                )
            )
        new_modules.append(replace(module, nodes=tuple(new_nodes)))
    return replace(graph, modules=tuple(new_modules)) if changed else graph


def _triton_rmsnorm_unit_offset_scaled_candidate(
    node: GraphNode,
    *,
    output_provenance: GraphProvenance | None,
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
) -> tuple[GraphOperand, GraphOperand, GraphOperand, GraphOperand, GraphOperand] | None:
    if len(node.outputs) != 1:
        return None
    if output_provenance is None or output_provenance.kind != "op":
        return None
    if output_provenance.op not in {"_add", "core.binary.+"} or len(output_provenance.args) != 2:
        return None

    def parse_scaled(
        item: GraphProvenance,
    ) -> tuple[GraphOperand, GraphOperand, GraphOperand, GraphOperand, GraphOperand] | None:
        if item.kind != "op":
            return None
        if item.op == "__triton_rmsnorm_scaled" and len(item.args) >= 5:
            x = _provenance_to_graph_operand(item.args[0], provenance_to_operand=provenance_to_operand)
            scale = _provenance_to_graph_operand(item.args[1], provenance_to_operand=provenance_to_operand)
            eps = _provenance_to_graph_operand(item.args[2], provenance_to_operand=provenance_to_operand)
            dim = _provenance_to_graph_operand(item.args[3], provenance_to_operand=provenance_to_operand)
            cast_float = _provenance_to_graph_operand(item.args[4], provenance_to_operand=provenance_to_operand)
            if x is None or scale is None or eps is None or dim is None or cast_float is None:
                return None
            return x, scale, eps, dim, cast_float
        if item.op not in {"_mul", "core.binary.*"} or len(item.args) != 2:
            return None

        def match_raw(
            rmsnorm: GraphProvenance,
            scale: GraphProvenance,
        ) -> tuple[GraphOperand, GraphOperand, GraphOperand, GraphOperand, GraphOperand] | None:
            if (
                rmsnorm.kind != "op"
                or rmsnorm.op not in {"_rmsnorm", "__triton_rmsnorm_noscale"}
                or len(rmsnorm.args) < 4
            ):
                return None
            if scale.kind != "op" or scale.op != "_params_param" or len(scale.args) < 1:
                return None
            x = _provenance_to_graph_operand(rmsnorm.args[0], provenance_to_operand=provenance_to_operand)
            scale_operand = _provenance_to_graph_operand(scale, provenance_to_operand=provenance_to_operand)
            if scale_operand is None:
                scale_operand = _provenance_to_graph_operand(scale.args[0], provenance_to_operand=provenance_to_operand)
                if scale_operand is not None and isinstance(graph_operand_type(scale_operand), TypeTensor):
                    scale_operand = None
            eps = _provenance_to_graph_operand(rmsnorm.args[1], provenance_to_operand=provenance_to_operand)
            dim = _provenance_to_graph_operand(rmsnorm.args[2], provenance_to_operand=provenance_to_operand)
            cast_float = _provenance_to_graph_operand(rmsnorm.args[3], provenance_to_operand=provenance_to_operand)
            if x is None or scale_operand is None or eps is None or dim is None or cast_float is None:
                return None
            if not isinstance(graph_operand_type(x), TypeTensor):
                return None
            return x, scale_operand, eps, dim, cast_float

        left, right = item.args
        return match_raw(left, right) or match_raw(right, left)

    def parse_noscale(
        item: GraphProvenance,
    ) -> tuple[GraphOperand, GraphOperand, GraphOperand, GraphOperand] | None:
        if (
            item.kind != "op"
            or item.op not in {"_rmsnorm", "__triton_rmsnorm_noscale"}
            or len(item.args) < 4
        ):
            return None
        x = _provenance_to_graph_operand(item.args[0], provenance_to_operand=provenance_to_operand)
        eps = _provenance_to_graph_operand(item.args[1], provenance_to_operand=provenance_to_operand)
        dim = _provenance_to_graph_operand(item.args[2], provenance_to_operand=provenance_to_operand)
        cast_float = _provenance_to_graph_operand(item.args[3], provenance_to_operand=provenance_to_operand)
        if x is None or eps is None or dim is None or cast_float is None:
            return None
        return x, eps, dim, cast_float

    def match_order(
        scaled: GraphProvenance,
        noscale: GraphProvenance,
    ) -> tuple[GraphOperand, GraphOperand, GraphOperand, GraphOperand, GraphOperand] | None:
        scaled_parts = parse_scaled(scaled)
        noscale_parts = parse_noscale(noscale)
        if scaled_parts is None or noscale_parts is None:
            return None
        x, scale, eps, dim, cast_float = scaled_parts
        noscale_x, noscale_eps, noscale_dim, noscale_cast_float = noscale_parts
        if (x, eps, dim, cast_float) != (noscale_x, noscale_eps, noscale_dim, noscale_cast_float):
            return None
        if not (
            isinstance(dim, GraphLiteral)
            and (dim.value is None or dim.value == -1)
        ):
            return None
        return x, scale, eps, dim, cast_float

    left, right = output_provenance.args
    return match_order(left, right) or match_order(right, left)


def _rewrite_triton_rmsnorm_unit_offset_scaled_intrinsics(graph: GraphProgram) -> GraphProgram:
    provenance = infer_graph_provenance(graph)
    changed = False
    new_modules: list[GraphModule] = []
    for module in graph.modules:
        local_provenance = provenance.module_local_provenance.get(module.name, {})
        provenance_to_operand = _module_provenance_to_operand_map(
            module,
            local_provenance=local_provenance,
        )
        new_nodes: list[GraphNode] = []
        for node in module.nodes:
            output_provenance = local_provenance.get(node.outputs[0].name) if len(node.outputs) == 1 else None
            inputs = _triton_rmsnorm_unit_offset_scaled_candidate(
                node,
                output_provenance=output_provenance,
                provenance_to_operand=provenance_to_operand,
            )
            if inputs is None:
                new_nodes.append(node)
                continue
            changed = True
            new_nodes.append(
                replace(
                    node,
                    op=GraphOp("__triton_rmsnorm_unit_offset_scaled"),
                    inputs=inputs,
                    attrs={},
                )
            )
        new_modules.append(replace(module, nodes=tuple(new_nodes)))
    return replace(graph, modules=tuple(new_modules)) if changed else graph


def _rewrite_triton_rmsnorm_noscale_intrinsics(graph: GraphProgram) -> GraphProgram:
    provenance = infer_graph_provenance(graph)
    changed = False
    new_modules: list[GraphModule] = []
    for module in graph.modules:
        local_provenance = provenance.module_local_provenance.get(module.name, {})
        provenance_to_operand = _module_provenance_to_operand_map(
            module,
            local_provenance=local_provenance,
        )
        new_nodes: list[GraphNode] = []
        for node in module.nodes:
            output_provenance = local_provenance.get(node.outputs[0].name) if len(node.outputs) == 1 else None
            inputs = _triton_rmsnorm_noscale_candidate(
                node,
                output_provenance=output_provenance,
                provenance_to_operand=provenance_to_operand,
            )
            if inputs is None:
                new_nodes.append(node)
                continue
            changed = True
            new_nodes.append(
                replace(
                    node,
                    op=GraphOp("__triton_rmsnorm_noscale"),
                    inputs=inputs,
                    attrs={},
                )
            )
        new_modules.append(replace(module, nodes=tuple(new_nodes)))
    return replace(graph, modules=tuple(new_modules)) if changed else graph


def _rewrite_triton_swiglu_activation_intrinsics(graph: GraphProgram) -> GraphProgram:
    provenance = infer_graph_provenance(graph)
    changed = False
    new_modules: list[GraphModule] = []
    for module in graph.modules:
        local_provenance = provenance.module_local_provenance.get(module.name, {})
        provenance_to_operand = _module_provenance_to_operand_map(
            module,
            local_provenance=local_provenance,
        )
        new_nodes: list[GraphNode] = []
        index = 0
        while index < len(module.nodes):
            node = module.nodes[index]
            output_provenance = local_provenance.get(node.outputs[0].name) if len(node.outputs) == 1 else None
            inputs = _triton_swiglu_activation_candidate(
                node,
                output_provenance=output_provenance,
                provenance_to_operand=provenance_to_operand,
            )
            if inputs is None:
                new_nodes.append(node)
                index += 1
                continue
            mul_node = node
            changed = True
            new_nodes.append(
                replace(
                    mul_node,
                    op=GraphOp("__triton_swiglu_activation"),
                    inputs=inputs,
                    attrs={},
                )
            )
            index += 1
        new_modules.append(replace(module, nodes=tuple(new_nodes)))
    return replace(graph, modules=tuple(new_modules)) if changed else graph


def _rewrite_triton_geglu_tanh_activation_intrinsics(graph: GraphProgram) -> GraphProgram:
    provenance = infer_graph_provenance(graph)
    changed = False
    new_modules: list[GraphModule] = []
    for module in graph.modules:
        local_provenance = provenance.module_local_provenance.get(module.name, {})
        provenance_to_operand = _module_provenance_to_operand_map(
            module,
            local_provenance=local_provenance,
        )
        new_nodes: list[GraphNode] = []
        for node in module.nodes:
            output_provenance = local_provenance.get(node.outputs[0].name) if len(node.outputs) == 1 else None
            inputs = _triton_geglu_tanh_activation_candidate(
                node,
                output_provenance=output_provenance,
                provenance_to_operand=provenance_to_operand,
            )
            if inputs is None:
                new_nodes.append(node)
                continue
            changed = True
            new_nodes.append(
                replace(
                    node,
                    op=GraphOp("__triton_geglu_tanh_activation"),
                    inputs=inputs,
                    attrs={},
                )
            )
        new_modules.append(replace(module, nodes=tuple(new_nodes)))
    return replace(graph, modules=tuple(new_modules)) if changed else graph


def _torch_expert_swiglu_ffn_candidate(
    gate_node: GraphNode,
    up_node: GraphNode,
    silu_node: GraphNode,
    mul_node: GraphNode,
    down_node: GraphNode,
    *,
    local_provenance: Mapping[str, GraphProvenance],
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
    value_ref_counts: Mapping[str, int],
) -> tuple[GraphOperand, ...] | None:
    if (
        len(gate_node.outputs) != 1
        or len(up_node.outputs) != 1
        or len(silu_node.outputs) != 1
        or len(mul_node.outputs) != 1
        or len(down_node.outputs) != 1
    ):
        return None
    gate_name = gate_node.outputs[0].name
    up_name = up_node.outputs[0].name
    silu_name = silu_node.outputs[0].name
    mul_name = mul_node.outputs[0].name
    if value_ref_counts.get(gate_name, 0) != 1:
        return None
    if value_ref_counts.get(up_name, 0) != 1:
        return None
    if value_ref_counts.get(silu_name, 0) != 1:
        return None
    if value_ref_counts.get(mul_name, 0) != 1:
        return None
    gate_prov = local_provenance.get(gate_name)
    up_prov = local_provenance.get(up_name)
    gate_args = _expert_linear_provenance_args(
        gate_prov,
        provenance_to_operand=provenance_to_operand,
    )
    up_args = _expert_linear_provenance_args(
        up_prov,
        provenance_to_operand=provenance_to_operand,
    )
    if gate_args is None or up_args is None or gate_prov is None or up_prov is None:
        return None
    gate_base, gate_x, gate_expert_idx, _gate_dim, gate_bias, gate_transpose, gate_weight_leaf, _gate_bias_leaf = gate_args
    up_base, up_x, up_expert_idx, _up_dim, up_bias, up_transpose, up_weight_leaf, _up_bias_leaf = up_args
    gate_x_hint = _node_operand_for_provenance(
        gate_node,
        gate_prov.args[1],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    gate_expert_idx_hint = _node_operand_for_provenance(
        gate_node,
        gate_prov.args[2],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    gate_base_hint = _node_operand_for_provenance(
        gate_node,
        gate_prov.args[0],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    gate_weight_leaf_hint = _node_operand_for_provenance(
        gate_node,
        gate_prov.args[6],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    up_base_hint = _node_operand_for_provenance(
        up_node,
        up_prov.args[0],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    up_x_hint = _node_operand_for_provenance(
        up_node,
        up_prov.args[1],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    up_expert_idx_hint = _node_operand_for_provenance(
        up_node,
        up_prov.args[2],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    up_weight_leaf_hint = _node_operand_for_provenance(
        up_node,
        up_prov.args[6],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    gate_x = gate_x_hint if gate_x_hint is not None else gate_x
    gate_expert_idx = gate_expert_idx_hint if gate_expert_idx_hint is not None else gate_expert_idx
    gate_base = gate_base_hint if gate_base_hint is not None else gate_base
    gate_weight_leaf = gate_weight_leaf_hint if gate_weight_leaf_hint is not None else gate_weight_leaf
    up_base = up_base_hint if up_base_hint is not None else up_base
    up_x = up_x_hint if up_x_hint is not None else up_x
    up_expert_idx = up_expert_idx_hint if up_expert_idx_hint is not None else up_expert_idx
    up_weight_leaf = up_weight_leaf_hint if up_weight_leaf_hint is not None else up_weight_leaf
    if gate_x != up_x or gate_expert_idx != up_expert_idx:
        return None
    gate_prov_args = gate_prov.args[:8]
    up_prov_args = up_prov.args[:8]
    if not (
        _is_literal_provenance_value(gate_prov_args[4], False)
        and _is_literal_provenance_value(up_prov_args[4], False)
        and _is_literal_provenance_value(gate_prov_args[5], False)
        and _is_literal_provenance_value(up_prov_args[5], False)
    ):
        return None
    if not graph_type_compatible(gate_node.outputs[0].type_expr, up_node.outputs[0].type_expr):
        return None
    silu_prov = local_provenance.get(silu_name)
    if (
        silu_prov is None
        or silu_prov.kind != "op"
        or silu_prov.op != "_activations_silu"
        or len(silu_prov.args) != 1
        or silu_prov.args[0] != gate_prov
    ):
        return None
    mul_prov = local_provenance.get(mul_name)
    if mul_prov is None or mul_prov.kind != "op" or mul_prov.op not in {"_mul", "core.binary.*"} or len(mul_prov.args) != 2:
        return None
    if not (
        (mul_prov.args[0] == silu_prov and mul_prov.args[1] == up_prov)
        or (mul_prov.args[1] == silu_prov and mul_prov.args[0] == up_prov)
    ):
        return None
    down_prov = local_provenance.get(down_node.outputs[0].name)
    down_args = _expert_linear_provenance_args(
        down_prov,
        provenance_to_operand=provenance_to_operand,
    )
    if down_args is None or down_prov is None:
        return None
    down_base, _down_x, down_expert_idx, _down_dim, down_bias, down_transpose, down_weight_leaf, _down_bias_leaf = down_args
    down_base_hint = _node_operand_for_provenance(
        down_node,
        down_prov.args[0],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    down_weight_leaf_hint = _node_operand_for_provenance(
        down_node,
        down_prov.args[6],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    down_expert_idx_hint = _node_operand_for_provenance(
        down_node,
        down_prov.args[2],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    down_base = down_base_hint if down_base_hint is not None else down_base
    down_weight_leaf = down_weight_leaf_hint if down_weight_leaf_hint is not None else down_weight_leaf
    down_expert_idx = down_expert_idx_hint if down_expert_idx_hint is not None else down_expert_idx
    if len(down_prov.args) < 3 or down_prov.args[1] != mul_prov or down_prov.args[2] != gate_prov.args[2]:
        return None
    if down_expert_idx != gate_expert_idx:
        return None
    down_prov_args = down_prov.args[:8]
    if not (
        _is_literal_provenance_value(down_prov_args[4], False)
        and _is_literal_provenance_value(down_prov_args[5], False)
    ):
        return None
    gate_weight_path = _compose_graph_path_operand(gate_base, gate_weight_leaf)
    up_weight_path = _compose_graph_path_operand(up_base, up_weight_leaf)
    down_weight_path = _compose_graph_path_operand(down_base, down_weight_leaf)
    if (
        not isinstance(gate_weight_path, GraphPath)
        or not isinstance(up_weight_path, GraphPath)
        or not isinstance(down_weight_path, GraphPath)
    ):
        return None
    del gate_bias, gate_transpose, up_bias, up_transpose, down_bias, down_transpose
    return (
        gate_x,
        gate_expert_idx,
        gate_weight_path,
        up_weight_path,
        down_weight_path,
    )


def _rewrite_torch_expert_swiglu_ffn_intrinsics(graph: GraphProgram) -> GraphProgram:
    if not any(node.op.name == "_expert_linear" for module in graph.modules for node in module.nodes):
        return graph
    provenance = infer_graph_provenance(graph)
    module_provenance_to_operand: dict[str, dict[GraphProvenance, GraphOperand]] = {}
    for module in graph.modules:
        local_provenance = provenance.module_local_provenance.get(module.name, {})
        module_provenance_to_operand[module.name] = _module_provenance_to_operand_map(
            module,
            local_provenance=local_provenance,
        )
    changed = False
    new_modules: list[GraphModule] = []
    for module in graph.modules:
        local_provenance = provenance.module_local_provenance.get(module.name, {})
        provenance_to_operand = module_provenance_to_operand.get(module.name, {})
        value_ref_counts = _module_value_ref_counts(module)
        new_nodes: list[GraphNode] = []
        index = 0
        while index < len(module.nodes):
            if index + 4 >= len(module.nodes):
                new_nodes.append(module.nodes[index])
                index += 1
                continue
            inputs = _torch_expert_swiglu_ffn_candidate(
                module.nodes[index],
                module.nodes[index + 1],
                module.nodes[index + 2],
                module.nodes[index + 3],
                module.nodes[index + 4],
                local_provenance=local_provenance,
                provenance_to_operand=provenance_to_operand,
                value_ref_counts=value_ref_counts,
            )
            if inputs is None:
                new_nodes.append(module.nodes[index])
                index += 1
                continue
            down_node = module.nodes[index + 4]
            changed = True
            new_nodes.append(
                replace(
                    down_node,
                    op=GraphOp("__torch_expert_swiglu_ffn"),
                    inputs=inputs,
                    attrs={},
                )
            )
            index += 5
        new_modules.append(replace(module, nodes=tuple(new_nodes)))
    return replace(graph, modules=tuple(new_modules)) if changed else graph


def _match_chunk_output_pair(
    gate: GraphProvenance,
    up: GraphProvenance,
) -> tuple[GraphProvenance, GraphProvenance] | None:
    if gate.kind != "op" or up.kind != "op":
        return None
    if gate.op == "_split[0]" and up.op == "_split[1]":
        if len(gate.args) < 3 or len(up.args) < 3:
            return None
        if gate.args != up.args:
            return None
        source, dim, parts = gate.args[:3]
        if not _is_literal_provenance_value(dim, -1):
            return None
        if parts.kind != "op" or parts.op != "core.list" or len(parts.args) != 2:
            return None
        if parts.args[0] != parts.args[1]:
            return None
        return source, parts
    if gate.op != "_chunk[0]" or up.op != "_chunk[1]":
        return None
    if len(gate.args) < 3 or len(up.args) < 3:
        return None
    if gate.args != up.args:
        return None
    source, dim, parts = gate.args[:3]
    if not _is_literal_provenance_value(dim, -1):
        return None
    if not _is_literal_provenance_value(parts, 2):
        return None
    return source, parts


def _torch_expert_packed_swiglu_ffn_candidate(
    gate_up_node: GraphNode,
    chunk_node: GraphNode,
    silu_node: GraphNode,
    mul_node: GraphNode,
    down_node: GraphNode,
    *,
    local_provenance: Mapping[str, GraphProvenance],
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
    value_ref_counts: Mapping[str, int],
) -> tuple[GraphOperand, ...] | None:
    if (
        len(gate_up_node.outputs) != 1
        or len(chunk_node.outputs) != 2
        or len(silu_node.outputs) != 1
        or len(mul_node.outputs) != 1
        or len(down_node.outputs) != 1
    ):
        return None
    gate_up_name = gate_up_node.outputs[0].name
    gate_name = chunk_node.outputs[0].name
    up_name = chunk_node.outputs[1].name
    silu_name = silu_node.outputs[0].name
    mul_name = mul_node.outputs[0].name
    if value_ref_counts.get(gate_up_name, 0) != 1:
        return None
    if value_ref_counts.get(gate_name, 0) != 1:
        return None
    if value_ref_counts.get(up_name, 0) != 1:
        return None
    if value_ref_counts.get(silu_name, 0) != 1:
        return None
    if value_ref_counts.get(mul_name, 0) != 1:
        return None
    gate_up_prov = local_provenance.get(gate_up_name)
    gate_up_args = _expert_linear_provenance_args(
        gate_up_prov,
        provenance_to_operand=provenance_to_operand,
    )
    if gate_up_args is None or gate_up_prov is None:
        return None
    gate_up_base, gate_up_x, gate_up_expert_idx, _gate_up_dim, gate_up_bias, gate_up_transpose, gate_up_weight_leaf, _gate_up_bias_leaf = gate_up_args
    gate_up_prov_args = gate_up_prov.args[:8]
    if not _is_literal_provenance_value(gate_up_prov_args[4], False):
        return None
    if not (
        _is_literal_provenance_value(gate_up_prov_args[5], False)
        or _is_literal_provenance_value(gate_up_prov_args[5], True)
    ):
        return None
    gate_up_x_hint = _node_operand_for_provenance(
        gate_up_node,
        gate_up_prov.args[1],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    gate_up_expert_idx_hint = _node_operand_for_provenance(
        gate_up_node,
        gate_up_prov.args[2],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    gate_up_base_hint = _node_operand_for_provenance(
        gate_up_node,
        gate_up_prov.args[0],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    gate_up_weight_leaf_hint = _node_operand_for_provenance(
        gate_up_node,
        gate_up_prov.args[6],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    gate_up_transpose_hint = _node_operand_for_provenance(
        gate_up_node,
        gate_up_prov.args[5],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    gate_up_x = gate_up_x_hint if gate_up_x_hint is not None else gate_up_x
    gate_up_expert_idx = gate_up_expert_idx_hint if gate_up_expert_idx_hint is not None else gate_up_expert_idx
    gate_up_base = gate_up_base_hint if gate_up_base_hint is not None else gate_up_base
    gate_up_weight_leaf = gate_up_weight_leaf_hint if gate_up_weight_leaf_hint is not None else gate_up_weight_leaf
    gate_up_transpose = gate_up_transpose_hint if gate_up_transpose_hint is not None else gate_up_transpose
    gate_prov = local_provenance.get(gate_name)
    up_prov = local_provenance.get(up_name)
    if gate_prov is None or up_prov is None:
        return None
    chunk_match = _match_chunk_output_pair(gate_prov, up_prov)
    if chunk_match is None:
        return None
    chunk_source, _parts = chunk_match
    if chunk_source != gate_up_prov:
        return None
    silu_prov = local_provenance.get(silu_name)
    if (
        silu_prov is None
        or silu_prov.kind != "op"
        or silu_prov.op != "_activations_silu"
        or len(silu_prov.args) != 1
        or silu_prov.args[0] != gate_prov
    ):
        return None
    mul_prov = local_provenance.get(mul_name)
    if mul_prov is None or mul_prov.kind != "op" or mul_prov.op not in {"_mul", "core.binary.*"} or len(mul_prov.args) != 2:
        return None
    if not (
        (mul_prov.args[0] == silu_prov and mul_prov.args[1] == up_prov)
        or (mul_prov.args[1] == silu_prov and mul_prov.args[0] == up_prov)
    ):
        return None
    down_prov = local_provenance.get(down_node.outputs[0].name)
    down_args = _expert_linear_provenance_args(
        down_prov,
        provenance_to_operand=provenance_to_operand,
    )
    if down_args is None or down_prov is None:
        return None
    down_base, _down_x, down_expert_idx, _down_dim, down_bias, down_transpose, down_weight_leaf, _down_bias_leaf = down_args
    if len(down_prov.args) < 3 or down_prov.args[1] != mul_prov or down_prov.args[2] != gate_up_prov.args[2]:
        return None
    down_prov_args = down_prov.args[:8]
    if not _is_literal_provenance_value(down_prov_args[4], False):
        return None
    if down_prov_args[5] != gate_up_prov_args[5]:
        return None
    down_base_hint = _node_operand_for_provenance(
        down_node,
        down_prov.args[0],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    down_expert_idx_hint = _node_operand_for_provenance(
        down_node,
        down_prov.args[2],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    down_weight_leaf_hint = _node_operand_for_provenance(
        down_node,
        down_prov.args[6],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    down_base = down_base_hint if down_base_hint is not None else down_base
    down_expert_idx = down_expert_idx_hint if down_expert_idx_hint is not None else down_expert_idx
    down_weight_leaf = down_weight_leaf_hint if down_weight_leaf_hint is not None else down_weight_leaf
    if down_expert_idx != gate_up_expert_idx:
        return None
    gate_up_weight_path = _compose_graph_path_operand(gate_up_base, gate_up_weight_leaf)
    down_weight_path = _compose_graph_path_operand(down_base, down_weight_leaf)
    if not isinstance(gate_up_weight_path, GraphPath) or not isinstance(down_weight_path, GraphPath):
        return None
    del gate_up_bias, down_bias, down_transpose
    return (
        gate_up_x,
        gate_up_expert_idx,
        gate_up_weight_path,
        down_weight_path,
        gate_up_transpose,
    )


def _match_expand_unsqueeze_hidden_provenance(
    provenance: GraphProvenance,
) -> GraphProvenance | None:
    if provenance.kind != "op" or provenance.op != "_expand" or len(provenance.args) < 1:
        return None
    unsqueezed = provenance.args[0]
    if unsqueezed.kind != "op" or unsqueezed.op != "_unsqueeze" or len(unsqueezed.args) < 2:
        return None
    if not _is_literal_provenance_value(unsqueezed.args[1], 2):
        return None
    return unsqueezed.args[0]


def _tensor_has_rank_or_variadic(tensor_type: TypeTensor, rank: int) -> bool:
    if len(tensor_type.dims) == rank:
        return True
    return len(tensor_type.dims) == 1 and isinstance(tensor_type.dims[0], str) and tensor_type.dims[0].startswith("..")


def _tensor_prefix_dims_match(left: TypeTensor, right: TypeTensor, count: int) -> bool:
    if len(left.dims) < count or len(right.dims) < count:
        return True
    return tuple(left.dims[:count]) == tuple(right.dims[:count])


def _torch_direct_selected_expert_packed_swiglu_candidate(
    sum_node: GraphNode,
    *,
    local_provenance: Mapping[str, GraphProvenance],
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
    producer_by_output: Mapping[str, GraphNode] | None = None,
) -> tuple[GraphOperand, ...] | None:
    if sum_node.op.name == "__torch_weighted_topk_sum" and len(sum_node.inputs) >= 2:
        values_operand, topk_scores = sum_node.inputs[:2]
        if isinstance(values_operand, GraphValueRef) and producer_by_output is not None:
            values_node = producer_by_output.get(values_operand.name)
            if (
                values_node is not None
                and values_node.op.name == "__torch_expert_packed_swiglu_ffn"
                and len(values_node.inputs) >= 5
            ):
                x_sel, topk_indices, gate_up_weight, down_weight, transpose = values_node.inputs[:5]
                if isinstance(x_sel, GraphValueRef):
                    expand_node = producer_by_output.get(x_sel.name)
                else:
                    expand_node = None
                if (
                    expand_node is not None
                    and expand_node.op.name == "_expand"
                    and expand_node.inputs
                    and isinstance(expand_node.inputs[0], GraphValueRef)
                ):
                    unsqueeze_node = producer_by_output.get(expand_node.inputs[0].name)
                else:
                    unsqueeze_node = None
                if (
                    unsqueeze_node is not None
                    and unsqueeze_node.op.name == "_unsqueeze"
                    and len(unsqueeze_node.inputs) >= 2
                    and _is_literal_value(unsqueeze_node.inputs[1], 2)
                    and isinstance(gate_up_weight, GraphPath)
                    and isinstance(down_weight, GraphPath)
                    and isinstance(transpose, GraphLiteral)
                    and isinstance(transpose.value, bool)
                ):
                    hidden = unsqueeze_node.inputs[0]
                    hidden_type = graph_operand_type(hidden)
                    scores_type = graph_operand_type(topk_scores)
                    indices_type = graph_operand_type(topk_indices)
                    output_type = sum_node.outputs[0].type_expr if sum_node.outputs else TypeAny()
                    if (
                        isinstance(hidden_type, TypeTensor)
                        and isinstance(scores_type, TypeTensor)
                        and isinstance(indices_type, TypeTensor)
                        and isinstance(output_type, TypeTensor)
                        and _tensor_has_rank_or_variadic(hidden_type, 3)
                        and _tensor_has_rank_or_variadic(scores_type, 3)
                        and _tensor_has_rank_or_variadic(indices_type, 3)
                        and _tensor_has_rank_or_variadic(output_type, 3)
                        and (tuple(scores_type.dims) == tuple(indices_type.dims) or len(scores_type.dims) == 1 or len(indices_type.dims) == 1)
                        and _tensor_prefix_dims_match(scores_type, hidden_type, 2)
                        and _tensor_prefix_dims_match(output_type, hidden_type, 2)
                    ):
                        return (
                            hidden,
                            topk_scores,
                            topk_indices,
                            gate_up_weight,
                            down_weight,
                            transpose,
                        )
        values_prov = _operand_local_provenance(values_operand, local_provenance=local_provenance)
        if (
            values_prov is None
            or values_prov.kind != "op"
            or values_prov.op != "__torch_expert_packed_swiglu_ffn"
            or len(values_prov.args) < 5
        ):
            return None
        hidden_prov = _match_expand_unsqueeze_hidden_provenance(values_prov.args[0])
        if hidden_prov is None:
            return None
        hidden = _provenance_to_graph_operand(
            hidden_prov,
            provenance_to_operand=provenance_to_operand,
        )
        topk_indices = _provenance_to_graph_operand(
            values_prov.args[1],
            provenance_to_operand=provenance_to_operand,
        )
        gate_up_weight = _provenance_to_graph_operand(
            values_prov.args[2],
            provenance_to_operand=provenance_to_operand,
        )
        down_weight = _provenance_to_graph_operand(
            values_prov.args[3],
            provenance_to_operand=provenance_to_operand,
        )
        transpose = _provenance_to_graph_operand(
            values_prov.args[4],
            provenance_to_operand=provenance_to_operand,
        )
        if not (
            hidden is not None
            and topk_indices is not None
            and isinstance(gate_up_weight, GraphPath)
            and isinstance(down_weight, GraphPath)
            and isinstance(transpose, GraphLiteral)
            and isinstance(transpose.value, bool)
        ):
            return None
        hidden_type = graph_operand_type(hidden)
        scores_type = graph_operand_type(topk_scores)
        indices_type = graph_operand_type(topk_indices)
        output_type = sum_node.outputs[0].type_expr if sum_node.outputs else TypeAny()
        if not (
            isinstance(hidden_type, TypeTensor)
            and isinstance(scores_type, TypeTensor)
            and isinstance(indices_type, TypeTensor)
            and isinstance(output_type, TypeTensor)
        ):
            return None
        if len(hidden_type.dims) != 3 or len(scores_type.dims) != 3 or len(indices_type.dims) != 3 or len(output_type.dims) != 3:
            return None
        if tuple(scores_type.dims) != tuple(indices_type.dims):
            return None
        if tuple(output_type.dims) != tuple(hidden_type.dims):
            return None
        if tuple(scores_type.dims[:2]) != tuple(hidden_type.dims[:2]):
            return None
        return (
            hidden,
            topk_scores,
            topk_indices,
            gate_up_weight,
            down_weight,
            transpose,
        )

    if len(sum_node.outputs) != 1:
        return None
    sum_prov = local_provenance.get(sum_node.outputs[0].name)
    if sum_prov is None or sum_prov.kind != "op" or sum_prov.op != "_sum" or len(sum_prov.args) < 3:
        return None
    if not _is_literal_provenance_value(sum_prov.args[1], 2):
        return None
    if not _is_literal_provenance_value(sum_prov.args[2], False):
        return None
    weighted_prov = sum_prov.args[0]
    if weighted_prov.kind != "op" or weighted_prov.op not in {"_mul", "core.binary.*"} or len(weighted_prov.args) != 2:
        return None
    left, right = weighted_prov.args
    if left.kind == "op" and left.op == "_unsqueeze" and len(left.args) >= 2 and _is_literal_provenance_value(left.args[1], -1):
        scale_prov = left
        values_prov = right
    elif right.kind == "op" and right.op == "_unsqueeze" and len(right.args) >= 2 and _is_literal_provenance_value(right.args[1], -1):
        scale_prov = right
        values_prov = left
    else:
        return None
    topk_scores = _provenance_to_graph_operand(
        scale_prov.args[0],
        provenance_to_operand=provenance_to_operand,
    )
    if topk_scores is None:
        return None

    if len(values_prov.args) < 8:
        return None
    if not _is_literal_provenance_value(values_prov.args[4], False):
        return None
    down_base = _provenance_to_graph_operand(values_prov.args[0], provenance_to_operand=provenance_to_operand)
    down_expert_idx = _provenance_to_graph_operand(values_prov.args[2], provenance_to_operand=provenance_to_operand)
    down_transpose = _provenance_to_graph_operand(values_prov.args[5], provenance_to_operand=provenance_to_operand)
    down_weight_leaf = _provenance_to_graph_operand(values_prov.args[6], provenance_to_operand=provenance_to_operand)
    if down_base is None or down_expert_idx is None or down_transpose is None or down_weight_leaf is None:
        return None
    ff_prov = values_prov.args[1]
    topk_indices_prov = values_prov.args[2]

    if ff_prov.kind != "op" or ff_prov.op not in {"_mul", "core.binary.*"} or len(ff_prov.args) != 2:
        return None
    ff_left, ff_right = ff_prov.args
    if ff_left.kind == "op" and ff_left.op == "_activations_silu" and len(ff_left.args) == 1:
        silu_prov = ff_left
        up_prov = ff_right
    elif ff_right.kind == "op" and ff_right.op == "_activations_silu" and len(ff_right.args) == 1:
        silu_prov = ff_right
        up_prov = ff_left
    else:
        return None
    gate_prov = silu_prov.args[0]
    chunk_match = _match_chunk_output_pair(gate_prov, up_prov)
    if chunk_match is None:
        return None
    gate_up_prov, _parts = chunk_match
    if len(gate_up_prov.args) < 8:
        return None
    if not _is_literal_provenance_value(gate_up_prov.args[4], False):
        return None
    gate_up_base = _provenance_to_graph_operand(gate_up_prov.args[0], provenance_to_operand=provenance_to_operand)
    gate_up_expert_idx = _provenance_to_graph_operand(gate_up_prov.args[2], provenance_to_operand=provenance_to_operand)
    gate_up_transpose = _provenance_to_graph_operand(gate_up_prov.args[5], provenance_to_operand=provenance_to_operand)
    gate_up_weight_leaf = _provenance_to_graph_operand(gate_up_prov.args[6], provenance_to_operand=provenance_to_operand)
    if gate_up_base is None or gate_up_expert_idx is None or gate_up_transpose is None or gate_up_weight_leaf is None:
        return None
    if gate_up_prov.args[2] != topk_indices_prov or values_prov.args[2] != topk_indices_prov:
        return None
    if gate_up_prov.args[5] != values_prov.args[5]:
        return None
    hidden_prov = _match_expand_unsqueeze_hidden_provenance(gate_up_prov.args[1])
    if hidden_prov is None:
        return None

    hidden = _provenance_to_graph_operand(
        hidden_prov,
        provenance_to_operand=provenance_to_operand,
    )
    topk_indices = _provenance_to_graph_operand(
        topk_indices_prov,
        provenance_to_operand=provenance_to_operand,
    )
    if hidden is None or topk_indices is None:
        return None
    if topk_indices != gate_up_expert_idx or topk_indices != down_expert_idx:
        return None
    gate_up_weight_path = _compose_graph_path_operand(gate_up_base, gate_up_weight_leaf)
    down_weight_path = _compose_graph_path_operand(down_base, down_weight_leaf)
    if not isinstance(gate_up_weight_path, GraphPath) or not isinstance(down_weight_path, GraphPath):
        return None
    gate_up_transpose_bool = _graph_bool_literal_operand(gate_up_transpose)
    down_transpose_bool = _graph_bool_literal_operand(down_transpose)
    if gate_up_transpose_bool is None or down_transpose_bool is None:
        return None
    if gate_up_transpose_bool.value != down_transpose_bool.value:
        return None

    hidden_type = graph_operand_type(hidden)
    scores_type = graph_operand_type(topk_scores)
    indices_type = graph_operand_type(topk_indices)
    output_type = sum_node.outputs[0].type_expr
    if not (
        isinstance(hidden_type, TypeTensor)
        and isinstance(scores_type, TypeTensor)
        and isinstance(indices_type, TypeTensor)
        and isinstance(output_type, TypeTensor)
    ):
        return None
    if not (
        _tensor_has_rank_or_variadic(hidden_type, 3)
        and _tensor_has_rank_or_variadic(scores_type, 3)
        and _tensor_has_rank_or_variadic(indices_type, 3)
        and _tensor_has_rank_or_variadic(output_type, 3)
    ):
        return None
    if tuple(scores_type.dims) != tuple(indices_type.dims) and len(scores_type.dims) != 1 and len(indices_type.dims) != 1:
        return None
    if not _tensor_prefix_dims_match(scores_type, hidden_type, 2):
        return None
    if not _tensor_prefix_dims_match(output_type, hidden_type, 2):
        return None

    return (
        hidden,
        topk_scores,
        topk_indices,
        gate_up_weight_path,
        down_weight_path,
        gate_up_transpose_bool,
    )


def _torch_direct_selected_expert_swiglu_candidate(
    sum_node: GraphNode,
    *,
    local_provenance: Mapping[str, GraphProvenance],
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
) -> tuple[GraphOperand, ...] | None:
    if len(sum_node.outputs) != 1:
        return None
    sum_prov = local_provenance.get(sum_node.outputs[0].name)
    if sum_prov is None or sum_prov.kind != "op" or sum_prov.op != "_sum" or len(sum_prov.args) < 3:
        return None
    if not _is_literal_provenance_value(sum_prov.args[1], 2) or not _is_literal_provenance_value(sum_prov.args[2], False):
        return None
    weighted_prov = sum_prov.args[0]
    if weighted_prov.kind != "op" or weighted_prov.op not in {"_mul", "core.binary.*"} or len(weighted_prov.args) != 2:
        return None
    left, right = weighted_prov.args
    if left.kind == "op" and left.op == "_unsqueeze" and len(left.args) >= 2 and _is_literal_provenance_value(left.args[1], -1):
        scale_prov = left
        values_prov = right
    elif right.kind == "op" and right.op == "_unsqueeze" and len(right.args) >= 2 and _is_literal_provenance_value(right.args[1], -1):
        scale_prov = right
        values_prov = left
    else:
        return None
    topk_scores = _provenance_to_graph_operand(scale_prov.args[0], provenance_to_operand=provenance_to_operand)
    if topk_scores is None or len(values_prov.args) < 8:
        return None
    if not _is_literal_provenance_value(values_prov.args[4], False):
        return None
    down_base = _provenance_to_graph_operand(values_prov.args[0], provenance_to_operand=provenance_to_operand)
    down_expert_idx = _provenance_to_graph_operand(values_prov.args[2], provenance_to_operand=provenance_to_operand)
    down_transpose = _provenance_to_graph_operand(values_prov.args[5], provenance_to_operand=provenance_to_operand)
    down_weight_leaf = _provenance_to_graph_operand(values_prov.args[6], provenance_to_operand=provenance_to_operand)
    if down_base is None or down_expert_idx is None or down_transpose is None or down_weight_leaf is None:
        return None
    ff_prov = values_prov.args[1]
    topk_indices_prov = values_prov.args[2]
    if ff_prov.kind != "op" or ff_prov.op not in {"_mul", "core.binary.*"} or len(ff_prov.args) != 2:
        return None
    ff_left, ff_right = ff_prov.args
    if ff_left.kind == "op" and ff_left.op == "_activations_silu" and len(ff_left.args) == 1:
        silu_prov = ff_left
        up_prov = ff_right
    elif ff_right.kind == "op" and ff_right.op == "_activations_silu" and len(ff_right.args) == 1:
        silu_prov = ff_right
        up_prov = ff_left
    else:
        return None
    gate_prov = silu_prov.args[0]
    if len(gate_prov.args) < 8 or len(up_prov.args) < 8:
        return None
    gate_base = _provenance_to_graph_operand(gate_prov.args[0], provenance_to_operand=provenance_to_operand)
    gate_expert_idx = _provenance_to_graph_operand(gate_prov.args[2], provenance_to_operand=provenance_to_operand)
    gate_transpose = _provenance_to_graph_operand(gate_prov.args[5], provenance_to_operand=provenance_to_operand)
    gate_weight_leaf = _provenance_to_graph_operand(gate_prov.args[6], provenance_to_operand=provenance_to_operand)
    up_base = _provenance_to_graph_operand(up_prov.args[0], provenance_to_operand=provenance_to_operand)
    up_expert_idx = _provenance_to_graph_operand(up_prov.args[2], provenance_to_operand=provenance_to_operand)
    up_transpose = _provenance_to_graph_operand(up_prov.args[5], provenance_to_operand=provenance_to_operand)
    up_weight_leaf = _provenance_to_graph_operand(up_prov.args[6], provenance_to_operand=provenance_to_operand)
    if (
        gate_base is None
        or gate_expert_idx is None
        or gate_transpose is None
        or gate_weight_leaf is None
        or up_base is None
        or up_expert_idx is None
        or up_transpose is None
        or up_weight_leaf is None
    ):
        return None
    if not (
        _is_literal_provenance_value(gate_prov.args[4], False)
        and _is_literal_provenance_value(up_prov.args[4], False)
        and gate_prov.args[1] == up_prov.args[1]
        and gate_prov.args[2] == topk_indices_prov
        and up_prov.args[2] == topk_indices_prov
        and values_prov.args[2] == topk_indices_prov
        and gate_prov.args[5] == up_prov.args[5]
        and gate_prov.args[5] == values_prov.args[5]
    ):
        return None
    hidden_prov = _match_expand_unsqueeze_hidden_provenance(gate_prov.args[1])
    if hidden_prov is None:
        return None
    hidden = _provenance_to_graph_operand(hidden_prov, provenance_to_operand=provenance_to_operand)
    topk_indices = _provenance_to_graph_operand(topk_indices_prov, provenance_to_operand=provenance_to_operand)
    if hidden is None or topk_indices is None:
        return None
    if topk_indices != gate_expert_idx or topk_indices != up_expert_idx or topk_indices != down_expert_idx:
        return None
    gate_weight_path = _compose_graph_path_operand(gate_base, gate_weight_leaf)
    up_weight_path = _compose_graph_path_operand(up_base, up_weight_leaf)
    down_weight_path = _compose_graph_path_operand(down_base, down_weight_leaf)
    if (
        not isinstance(gate_weight_path, GraphPath)
        or not isinstance(up_weight_path, GraphPath)
        or not isinstance(down_weight_path, GraphPath)
    ):
        return None
    gate_transpose_bool = _graph_bool_literal_operand(gate_transpose)
    up_transpose_bool = _graph_bool_literal_operand(up_transpose)
    down_transpose_bool = _graph_bool_literal_operand(down_transpose)
    if (
        gate_transpose_bool is None
        or up_transpose_bool is None
        or down_transpose_bool is None
        or gate_transpose_bool.value != up_transpose_bool.value
        or gate_transpose_bool.value != down_transpose_bool.value
    ):
        return None
    hidden_type = graph_operand_type(hidden)
    scores_type = graph_operand_type(topk_scores)
    indices_type = graph_operand_type(topk_indices)
    output_type = sum_node.outputs[0].type_expr
    if not (
        isinstance(hidden_type, TypeTensor)
        and isinstance(scores_type, TypeTensor)
        and isinstance(indices_type, TypeTensor)
        and isinstance(output_type, TypeTensor)
        and _tensor_has_rank_or_variadic(hidden_type, 3)
        and _tensor_has_rank_or_variadic(scores_type, 3)
        and _tensor_has_rank_or_variadic(indices_type, 3)
        and _tensor_has_rank_or_variadic(output_type, 3)
        and (tuple(scores_type.dims) == tuple(indices_type.dims) or len(scores_type.dims) == 1 or len(indices_type.dims) == 1)
        and _tensor_prefix_dims_match(scores_type, hidden_type, 2)
        and _tensor_prefix_dims_match(output_type, hidden_type, 2)
    ):
        return None
    return (
        hidden,
        topk_scores,
        topk_indices,
        gate_weight_path,
        up_weight_path,
        down_weight_path,
        gate_transpose_bool,
    )


def _rewrite_torch_expert_packed_swiglu_ffn_intrinsics(graph: GraphProgram) -> GraphProgram:
    if not any(node.op.name == "_expert_linear" for module in graph.modules for node in module.nodes):
        return graph
    provenance = infer_graph_provenance(graph)
    module_provenance_to_operand: dict[str, dict[GraphProvenance, GraphOperand]] = {}
    for module in graph.modules:
        local_provenance = provenance.module_local_provenance.get(module.name, {})
        module_provenance_to_operand[module.name] = _module_provenance_to_operand_map(
            module,
            local_provenance=local_provenance,
        )
    changed = False
    new_modules: list[GraphModule] = []
    for module in graph.modules:
        local_provenance = provenance.module_local_provenance.get(module.name, {})
        provenance_to_operand = module_provenance_to_operand.get(module.name, {})
        value_ref_counts = _module_value_ref_counts(module)
        new_nodes: list[GraphNode] = []
        index = 0
        while index < len(module.nodes):
            if index + 4 >= len(module.nodes):
                new_nodes.append(module.nodes[index])
                index += 1
                continue
            inputs = _torch_expert_packed_swiglu_ffn_candidate(
                module.nodes[index],
                module.nodes[index + 1],
                module.nodes[index + 2],
                module.nodes[index + 3],
                module.nodes[index + 4],
                local_provenance=local_provenance,
                provenance_to_operand=provenance_to_operand,
                value_ref_counts=value_ref_counts,
            )
            if inputs is None:
                new_nodes.append(module.nodes[index])
                index += 1
                continue
            down_node = module.nodes[index + 4]
            changed = True
            new_nodes.append(
                replace(
                    down_node,
                    op=GraphOp("__torch_expert_packed_swiglu_ffn"),
                    inputs=inputs,
                    attrs={},
                )
            )
            index += 5
        new_modules.append(replace(module, nodes=tuple(new_nodes)))
    return replace(graph, modules=tuple(new_modules)) if changed else graph


def _torch_weighted_topk_sum_candidate(
    scale_node: GraphNode,
    mul_node: GraphNode,
    sum_node: GraphNode,
    *,
    local_provenance: Mapping[str, GraphProvenance],
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
    value_ref_counts: Mapping[str, int],
) -> tuple[GraphOperand, ...] | None:
    if len(scale_node.outputs) != 1 or len(mul_node.outputs) != 1 or len(sum_node.outputs) != 1:
        return None
    if scale_node.op.name == "_unsqueeze" and mul_node.op.name in {"_mul", "core.binary.*"} and sum_node.op.name == "_sum":
        if len(scale_node.inputs) >= 2 and len(mul_node.inputs) == 2 and len(sum_node.inputs) >= 3:
            scale_ref = GraphValueRef(
                scale_node.outputs[0].name,
                scale_node.outputs[0].type_expr,
                scale_node.outputs[0].dims,
            )
            mul_ref = GraphValueRef(
                mul_node.outputs[0].name,
                mul_node.outputs[0].type_expr,
                mul_node.outputs[0].dims,
            )
            if (
                _is_literal_value(scale_node.inputs[1], -1)
                and sum_node.inputs[0] == mul_ref
                and _is_literal_value(sum_node.inputs[1], 2)
                and _is_literal_value(sum_node.inputs[2], False)
                and value_ref_counts.get(scale_node.outputs[0].name, 0) == 1
                and value_ref_counts.get(mul_node.outputs[0].name, 0) == 1
            ):
                if mul_node.inputs[0] == scale_ref:
                    values_operand = mul_node.inputs[1]
                elif mul_node.inputs[1] == scale_ref:
                    values_operand = mul_node.inputs[0]
                else:
                    values_operand = None
                scores_operand = scale_node.inputs[0]
                values_type = graph_operand_type(values_operand) if values_operand is not None else None
                scores_type = graph_operand_type(scores_operand)
                output_type = sum_node.outputs[0].type_expr
                if (
                    isinstance(values_type, TypeTensor)
                    and isinstance(scores_type, TypeTensor)
                    and isinstance(output_type, TypeTensor)
                    and len(values_type.dims) == 4
                    and len(scores_type.dims) == 3
                    and len(output_type.dims) == 3
                    and tuple(scores_type.dims) == tuple(values_type.dims[:3])
                    and tuple(output_type.dims) == (values_type.dims[0], values_type.dims[1], values_type.dims[3])
                ):
                    return values_operand, scores_operand
    scale_name = scale_node.outputs[0].name
    mul_name = mul_node.outputs[0].name
    if value_ref_counts.get(scale_name, 0) != 1:
        return None
    if value_ref_counts.get(mul_name, 0) != 1:
        return None
    scale_prov = local_provenance.get(scale_name)
    if scale_prov is None or scale_prov.kind != "op" or scale_prov.op != "_unsqueeze" or len(scale_prov.args) < 2:
        return None
    if not _is_literal_provenance_value(scale_prov.args[1], -1):
        return None
    scores_operand = _node_operand_for_provenance(
        scale_node,
        scale_prov.args[0],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    if scores_operand is None:
        return None
    mul_prov = local_provenance.get(mul_name)
    if mul_prov is None or mul_prov.kind != "op" or mul_prov.op not in {"_mul", "core.binary.*"} or len(mul_prov.args) != 2:
        return None
    if mul_prov.args[0] == scale_prov:
        values_provenance = mul_prov.args[1]
    elif mul_prov.args[1] == scale_prov:
        values_provenance = mul_prov.args[0]
    else:
        return None
    values_operand = _provenance_to_graph_operand(
        values_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    if values_operand is None:
        return None
    sum_prov = local_provenance.get(sum_node.outputs[0].name)
    if sum_prov is None or sum_prov.kind != "op" or sum_prov.op != "_sum" or len(sum_prov.args) < 3:
        return None
    if sum_prov.args[0] != mul_prov:
        return None
    if not _is_literal_provenance_value(sum_prov.args[1], 2):
        return None
    if not _is_literal_provenance_value(sum_prov.args[2], False):
        return None
    values_type = graph_operand_type(values_operand)
    scores_type = graph_operand_type(scores_operand)
    output_type = sum_node.outputs[0].type_expr
    if not (
        isinstance(values_type, TypeTensor)
        and isinstance(scores_type, TypeTensor)
        and isinstance(output_type, TypeTensor)
    ):
        return None
    if len(values_type.dims) != 4 or len(scores_type.dims) != 3 or len(output_type.dims) != 3:
        return None
    if tuple(scores_type.dims) != tuple(values_type.dims[:3]):
        return None
    if tuple(output_type.dims) != (values_type.dims[0], values_type.dims[1], values_type.dims[3]):
        return None
    return values_operand, scores_operand


def _is_topk_last_slice_bounds(
    start_provenance: GraphProvenance,
    end_provenance: GraphProvenance,
    top_k_provenance: GraphProvenance,
) -> bool:
    if end_provenance != top_k_provenance:
        return False
    expected_start = GraphProvenance(
        "op",
        op="core.binary.-",
        args=(top_k_provenance, GraphProvenance("literal", value=1)),
    )
    if start_provenance == expected_start:
        return True
    if (
        start_provenance.kind == "literal"
        and end_provenance.kind == "literal"
        and isinstance(start_provenance.value, int)
        and isinstance(end_provenance.value, int)
    ):
        return start_provenance.value == end_provenance.value - 1
    return False


def _torch_topk_normalize_inputs_from_provenance(
    *,
    cumsum_prov: GraphProvenance,
    slice_prov: GraphProvenance,
    div_prov: GraphProvenance,
    cast_prov: GraphProvenance,
    cast_node: GraphNode,
    local_provenance: Mapping[str, GraphProvenance],
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
) -> tuple[GraphOperand, ...] | None:
    if cumsum_prov.kind != "op" or cumsum_prov.op != "_cumsum" or len(cumsum_prov.args) < 2:
        return None
    if not _is_literal_provenance_value(cumsum_prov.args[1], -1):
        return None
    topk_weights_prov = cumsum_prov.args[0]
    if topk_weights_prov.kind != "op" or topk_weights_prov.op not in {"_topk", "_topk[0]"} or len(topk_weights_prov.args) < 5:
        return None
    if not (
        _is_literal_provenance_value(topk_weights_prov.args[2], -1)
        and _is_literal_provenance_value(topk_weights_prov.args[3], True)
        and _is_literal_provenance_value(topk_weights_prov.args[4], False)
    ):
        return None
    if slice_prov.kind != "op" or slice_prov.op != "_slice" or len(slice_prov.args) < 4:
        return None
    if slice_prov.args[0] != cumsum_prov:
        return None
    if not _is_literal_provenance_value(slice_prov.args[1], -1):
        return None
    if not _is_topk_last_slice_bounds(slice_prov.args[2], slice_prov.args[3], topk_weights_prov.args[1]):
        return None
    if div_prov.kind != "op" or div_prov.op not in {"_div", "core.binary./"} or len(div_prov.args) != 2:
        return None
    if div_prov.args[0] != topk_weights_prov or div_prov.args[1] != slice_prov:
        return None
    if cast_prov.kind != "op" or cast_prov.op != "_cast_like" or len(cast_prov.args) < 2:
        return None
    if cast_prov.args[0] != div_prov:
        return None
    weights_operand = _provenance_to_graph_operand(
        topk_weights_prov,
        provenance_to_operand=provenance_to_operand,
    )
    ref_operand = _node_operand_for_provenance(
        cast_node,
        cast_prov.args[1],
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    if weights_operand is None or ref_operand is None:
        return None
    weights_type = graph_operand_type(weights_operand)
    output_type = cast_node.outputs[0].type_expr
    if not isinstance(weights_type, TypeTensor) or not isinstance(output_type, TypeTensor):
        return None
    if tuple(weights_type.dims) != tuple(output_type.dims):
        return None
    return weights_operand, ref_operand


def _provenance_contains(
    root: GraphProvenance,
    target: GraphProvenance,
    memo: dict[tuple[str, int, object], object] | None = None,
) -> bool:
    key = ("contains", id(root), id(target))
    if memo is not None and key in memo:
        cached = memo[key]
        return bool(cached) if isinstance(cached, bool) else False
    if root == target:
        if memo is not None:
            memo[key] = True
        return True
    if memo is not None:
        memo[key] = "in_progress"
    result = any(_provenance_contains(arg, target, memo) for arg in root.args)
    if memo is not None:
        memo[key] = result
    return result


def _find_op_provenance(
    root: GraphProvenance,
    op_name: str,
    memo: dict[tuple[str, int, object], object] | None = None,
) -> GraphProvenance | None:
    key = ("find_op", id(root), op_name)
    if memo is not None and key in memo:
        found = memo[key]
        return found if isinstance(found, GraphProvenance) else None
    if root.kind == "op" and root.op == op_name:
        if memo is not None:
            memo[key] = root
        return root
    if memo is not None:
        memo[key] = "in_progress"
    for arg in root.args:
        found = _find_op_provenance(arg, op_name, memo)
        if found is not None:
            if memo is not None:
                memo[key] = found
            return found
    if memo is not None:
        memo[key] = None
    return None


def _match_packed_swiglu_linear_provenance(
    provenance: GraphProvenance,
) -> tuple[GraphProvenance, GraphProvenance, GraphProvenance] | None:
    if provenance.kind != "op" or provenance.op != "_linear" or len(provenance.args) < 8:
        return None
    hidden = provenance.args[1]
    if hidden.kind != "op" or hidden.op not in {"_mul", "core.binary.*"} or len(hidden.args) != 2:
        return None
    left, right = hidden.args
    if left.kind == "op" and left.op == "_activations_silu" and len(left.args) == 1:
        gate, up = left.args[0], right
    elif right.kind == "op" and right.op == "_activations_silu" and len(right.args) == 1:
        gate, up = right.args[0], left
    else:
        return None
    chunk_match = _match_chunk_output_pair(gate, up)
    if chunk_match is None:
        return None
    gate_up, _parts = chunk_match
    if gate_up.kind != "op" or gate_up.op != "_linear" or len(gate_up.args) < 8:
        return None
    if not (
        _is_literal_provenance_value(gate_up.args[3], False)
        and _is_literal_provenance_value(provenance.args[3], False)
        and (
            _is_literal_provenance_value(gate_up.args[4], False)
            or _is_literal_provenance_value(gate_up.args[4], True)
        )
        and gate_up.args[4] == provenance.args[4]
        and _is_literal_provenance_value(gate_up.args[5], None)
        and _is_literal_provenance_value(provenance.args[5], None)
    ):
        return None
    return gate_up, gate_up.args[1], provenance


def _find_packed_swiglu_linear_provenance(
    provenance: GraphProvenance,
    memo: dict[tuple[str, int, object], object] | None = None,
) -> tuple[GraphProvenance, GraphProvenance, GraphProvenance] | None:
    key = ("find_packed_swiglu", id(provenance), None)
    if memo is not None and key in memo:
        found = memo[key]
        return found if isinstance(found, tuple) else None
    matched = _match_packed_swiglu_linear_provenance(provenance)
    if matched is not None:
        if memo is not None:
            memo[key] = matched
        return matched
    if memo is not None:
        memo[key] = "in_progress"
    for arg in provenance.args:
        matched = _find_packed_swiglu_linear_provenance(arg, memo)
        if matched is not None:
            if memo is not None:
                memo[key] = matched
            return matched
    if memo is not None:
        memo[key] = None
    return None


def _match_clamped_packed_swiglu_linear_provenance(
    provenance: GraphProvenance,
) -> tuple[GraphProvenance, GraphProvenance, GraphProvenance, GraphProvenance] | None:
    if provenance.kind != "op" or provenance.op != "_linear" or len(provenance.args) < 8:
        return None
    hidden = provenance.args[1]
    if hidden.kind != "op" or hidden.op not in {"_mul", "core.binary.*"} or len(hidden.args) != 2:
        return None
    for left, right in (hidden.args, hidden.args[::-1]):
        if left.kind != "op" or left.op != "_activations_silu" or len(left.args) != 1:
            continue
        gate = left.args[0]
        up = right
        if gate.kind != "op" or gate.op != "_clamp" or len(gate.args) < 3:
            continue
        if up.kind != "op" or up.op != "_clamp" or len(up.args) < 3:
            continue
        gate_raw, gate_min, gate_max = gate.args[:3]
        up_raw, up_min, up_max = up.args[:3]
        if not _is_literal_provenance_value(gate_min, None):
            continue
        if gate_max != up_max or not _match_limit_negation(up_min, gate_max):
            continue
        chunk_match = _match_chunk_output_pair(gate_raw, up_raw)
        if chunk_match is None:
            continue
        gate_up, _parts = chunk_match
        if gate_up.kind != "op" or gate_up.op != "_linear" or len(gate_up.args) < 8:
            continue
        if not (
            _is_literal_provenance_value(gate_up.args[3], False)
            and _is_literal_provenance_value(provenance.args[3], False)
            and (
                _is_literal_provenance_value(gate_up.args[4], False)
                or _is_literal_provenance_value(gate_up.args[4], True)
            )
            and gate_up.args[4] == provenance.args[4]
            and _is_literal_provenance_value(gate_up.args[5], None)
            and _is_literal_provenance_value(provenance.args[5], None)
        ):
            continue
        return gate_up, gate_up.args[1], provenance, gate_max
    return None


def _find_clamped_packed_swiglu_linear_provenance(
    provenance: GraphProvenance,
    memo: dict[tuple[str, int, object], object] | None = None,
) -> tuple[GraphProvenance, GraphProvenance, GraphProvenance, GraphProvenance] | None:
    key = ("find_clamped_packed_swiglu", id(provenance), None)
    if memo is not None and key in memo:
        found = memo[key]
        return found if isinstance(found, tuple) else None
    matched = _match_clamped_packed_swiglu_linear_provenance(provenance)
    if matched is not None:
        if memo is not None:
            memo[key] = matched
        return matched
    if memo is not None:
        memo[key] = "in_progress"
    for arg in provenance.args:
        matched = _find_clamped_packed_swiglu_linear_provenance(arg, memo)
        if matched is not None:
            if memo is not None:
                memo[key] = matched
            return matched
    if memo is not None:
        memo[key] = None
    return None


def _match_packed_gegelu_linear_provenance(
    provenance: GraphProvenance,
) -> tuple[GraphProvenance, GraphProvenance, GraphProvenance, GraphProvenance] | None:
    if provenance.kind != "op" or provenance.op != "_linear" or len(provenance.args) < 8:
        return None
    hidden = provenance.args[1]
    if hidden.kind != "op" or hidden.op != "_activations_gegelu" or len(hidden.args) < 2:
        return None
    gate_up = hidden.args[0]
    if gate_up.kind != "op" or gate_up.op != "_linear" or len(gate_up.args) < 8:
        return None
    if not (
        (
            _is_literal_provenance_value(gate_up.args[3], False)
            or _is_literal_provenance_value(gate_up.args[3], True)
        )
        and gate_up.args[3] == provenance.args[3]
        and (
            _is_literal_provenance_value(gate_up.args[4], False)
            or _is_literal_provenance_value(gate_up.args[4], True)
        )
        and gate_up.args[4] == provenance.args[4]
        and gate_up.args[5] == provenance.args[5]
    ):
        return None
    return gate_up, gate_up.args[1], provenance, hidden.args[1]


def _find_packed_gegelu_linear_provenance(
    provenance: GraphProvenance,
    memo: dict[tuple[str, int, object], object] | None = None,
) -> tuple[GraphProvenance, GraphProvenance, GraphProvenance, GraphProvenance] | None:
    key = ("find_packed_gegelu", id(provenance), None)
    if memo is not None and key in memo:
        found = memo[key]
        return found if isinstance(found, tuple) else None
    matched = _match_packed_gegelu_linear_provenance(provenance)
    if matched is not None:
        if memo is not None:
            memo[key] = matched
        return matched
    if memo is not None:
        memo[key] = "in_progress"
    for arg in provenance.args:
        matched = _find_packed_gegelu_linear_provenance(arg, memo)
        if matched is not None:
            if memo is not None:
                memo[key] = matched
            return matched
    if memo is not None:
        memo[key] = None
    return None


def _match_relu2_linear_provenance(
    provenance: GraphProvenance,
) -> tuple[GraphProvenance, GraphProvenance, GraphProvenance] | None:
    if provenance.kind != "op" or provenance.op != "_linear" or len(provenance.args) < 8:
        return None
    relu2 = provenance.args[1]
    if relu2.kind != "op" or relu2.op != "_activations_relu2" or len(relu2.args) != 1:
        return None
    up = relu2.args[0]
    if up.kind != "op" or up.op != "_linear" or len(up.args) < 8:
        return None
    if not (
        _is_literal_provenance_value(up.args[3], False)
        and _is_literal_provenance_value(provenance.args[3], False)
        and (
            _is_literal_provenance_value(up.args[4], False)
            or _is_literal_provenance_value(up.args[4], True)
        )
        and up.args[4] == provenance.args[4]
        and _is_literal_provenance_value(up.args[5], None)
        and _is_literal_provenance_value(provenance.args[5], None)
    ):
        return None
    return up, up.args[1], provenance


def _find_relu2_linear_provenance(
    provenance: GraphProvenance,
    memo: dict[tuple[str, int, object], object] | None = None,
) -> tuple[GraphProvenance, GraphProvenance, GraphProvenance] | None:
    key = ("find_relu2", id(provenance), None)
    if memo is not None and key in memo:
        found = memo[key]
        return found if isinstance(found, tuple) else None
    matched = _match_relu2_linear_provenance(provenance)
    if matched is not None:
        if memo is not None:
            memo[key] = matched
        return matched
    if memo is not None:
        memo[key] = "in_progress"
    for arg in provenance.args:
        matched = _find_relu2_linear_provenance(arg, memo)
        if matched is not None:
            if memo is not None:
                memo[key] = matched
            return matched
    if memo is not None:
        memo[key] = None
    return None


def _find_weighted_update_unsqueeze_provenance(
    root: GraphProvenance,
    update_provenance: GraphProvenance,
    memo: dict[tuple[str, int, object], object] | None = None,
) -> GraphProvenance | None:
    key = ("find_weighted_unsqueeze", id(root), id(update_provenance))
    if memo is not None and key in memo:
        found = memo[key]
        return found if isinstance(found, GraphProvenance) else None
    if root.kind == "op" and root.op in {"_mul", "core.binary.*"} and len(root.args) == 2:
        left, right = root.args
        if _provenance_contains(left, update_provenance, memo) and right.kind == "op" and right.op == "_unsqueeze":
            if memo is not None:
                memo[key] = right
            return right
        if _provenance_contains(right, update_provenance, memo) and left.kind == "op" and left.op == "_unsqueeze":
            if memo is not None:
                memo[key] = left
            return left
    if memo is not None:
        memo[key] = "in_progress"
    for arg in root.args:
        found = _find_weighted_update_unsqueeze_provenance(arg, update_provenance, memo)
        if found is not None:
            if memo is not None:
                memo[key] = found
            return found
    if memo is not None:
        memo[key] = None
    return None


def _selected_hidden_source_provenance(
    selected_provenance: GraphProvenance,
    token_index_provenance: GraphProvenance,
    memo: dict[tuple[str, int, object], object] | None = None,
) -> GraphProvenance | None:
    if selected_provenance.kind != "op" or selected_provenance.op != "_gather" or len(selected_provenance.args) < 3:
        return None
    if not _is_literal_provenance_value(selected_provenance.args[2], 0):
        return None
    if not _provenance_contains(selected_provenance.args[1], token_index_provenance, memo):
        return None
    reshaped = selected_provenance.args[0]
    if reshaped.kind != "op" or reshaped.op != "_reshape" or not reshaped.args:
        return None
    return reshaped.args[0]


def _selected_scores_source_provenance(
    scores_provenance: GraphProvenance,
    token_index_provenance: GraphProvenance,
    topk_pos_provenance: GraphProvenance,
    memo: dict[tuple[str, int, object], object] | None = None,
) -> GraphProvenance | None:
    if scores_provenance.kind == "op" and scores_provenance.op == "_reshape" and scores_provenance.args:
        return _selected_scores_source_provenance(
            scores_provenance.args[0],
            token_index_provenance,
            topk_pos_provenance,
            memo,
        )
    if scores_provenance.kind != "op" or scores_provenance.op != "_gather" or len(scores_provenance.args) < 3:
        return None
    if not _provenance_contains(scores_provenance.args[1], topk_pos_provenance, memo):
        return None
    first_gather = scores_provenance.args[0]
    if first_gather.kind != "op" or first_gather.op != "_gather" or len(first_gather.args) < 3:
        return None
    if not _provenance_contains(first_gather.args[1], token_index_provenance, memo):
        return None
    reshaped = first_gather.args[0]
    if reshaped.kind != "op" or reshaped.op != "_reshape" or not reshaped.args:
        return None
    return reshaped.args[0]


def _path_without_repeat_iter_segment(
    provenance: GraphProvenance,
    *,
    repeat_var: str,
) -> GraphPath | None:
    if provenance.kind != "path" or not isinstance(provenance.value, str):
        return None
    absolute, parts = _path_parts_from_token(provenance.value)
    iter_segment = "{" + repeat_var + "}"
    new_parts = tuple(part for part in parts if part != iter_segment)
    if new_parts == parts or not new_parts:
        return None
    return GraphPath(absolute, new_parts)


def _repeat_actual_operands_by_formal(
    repeat_node: GraphNode,
    callee_module: GraphModule,
) -> dict[str, GraphOperand]:
    if len(repeat_node.inputs) < 3:
        return {}
    out: dict[str, GraphOperand] = {}
    for index, formal in enumerate(callee_module.inputs):
        role_operand = repeat_node.attrs.get(f"arg_{index}")
        if not isinstance(role_operand, GraphLiteral) or not isinstance(role_operand.value, str):
            continue
        role = role_operand.value
        if role.startswith("input:"):
            try:
                input_index = int(role.split(":", 1)[1])
            except ValueError:
                continue
            if 0 <= input_index < len(repeat_node.inputs):
                out[formal.name] = repeat_node.inputs[input_index]
        elif role.startswith("carry:"):
            try:
                carry_index = int(role.split(":", 1)[1])
            except ValueError:
                continue
            input_index = 3 + carry_index
            if 0 <= input_index < len(repeat_node.inputs):
                out[formal.name] = repeat_node.inputs[input_index]
        elif role == "iter":
            out[formal.name] = GraphValueRef(
                formal.name,
                formal.type_expr,
                formal.dims,
            )
    return out


def _actual_operand_for_callee_provenance(
    target: GraphProvenance,
    *,
    callee_module: GraphModule,
    callee_local_provenance: Mapping[str, GraphProvenance],
    actual_by_formal: Mapping[str, GraphOperand],
) -> GraphOperand | None:
    matches: list[GraphOperand] = []
    for formal in callee_module.inputs:
        if callee_local_provenance.get(formal.name) == target and formal.name in actual_by_formal:
            matches.append(actual_by_formal[formal.name])
    return matches[0] if len(matches) == 1 else None


def _call_actual_operands_by_formal(
    call_node: GraphNode,
    callee_module: GraphModule,
) -> dict[str, GraphOperand]:
    if len(call_node.inputs) != len(callee_module.inputs):
        return {}
    return {
        formal.name: actual
        for formal, actual in zip(callee_module.inputs, call_node.inputs, strict=True)
    }


def _torch_direct_selected_expert_packed_swiglu_call_candidate(
    call_node: GraphNode,
    *,
    callee_module: GraphModule,
    callee_local_provenance: Mapping[str, GraphProvenance],
) -> tuple[GraphOperand, ...] | None:
    if len(call_node.outputs) != 1 or len(callee_module.outputs) != 1:
        return None
    output = callee_module.outputs[0]
    if not isinstance(output, GraphValueRef):
        return None
    output_prov = callee_local_provenance.get(output.name)
    if output_prov is None or output_prov.kind != "op" or output_prov.op != "_sum" or len(output_prov.args) < 3:
        return None
    if not _is_literal_provenance_value(output_prov.args[1], 2) or not _is_literal_provenance_value(output_prov.args[2], False):
        return None
    weighted_prov = output_prov.args[0]
    if weighted_prov.kind != "op" or weighted_prov.op not in {"_mul", "core.binary.*"} or len(weighted_prov.args) != 2:
        return None
    left, right = weighted_prov.args
    if left.kind == "op" and left.op == "_unsqueeze" and len(left.args) >= 2 and _is_literal_provenance_value(left.args[1], -1):
        scale_prov = left
        values_prov = right
    elif right.kind == "op" and right.op == "_unsqueeze" and len(right.args) >= 2 and _is_literal_provenance_value(right.args[1], -1):
        scale_prov = right
        values_prov = left
    else:
        return None
    if values_prov.kind != "op" or values_prov.op != "_expert_linear" or len(values_prov.args) < 8:
        return None
    if not _is_literal_provenance_value(values_prov.args[4], False):
        return None
    ff_prov = values_prov.args[1]
    topk_indices_prov = values_prov.args[2]
    if ff_prov.kind != "op" or ff_prov.op not in {"_mul", "core.binary.*"} or len(ff_prov.args) != 2:
        return None
    ff_left, ff_right = ff_prov.args
    if ff_left.kind == "op" and ff_left.op == "_activations_silu" and len(ff_left.args) == 1:
        silu_prov = ff_left
        up_prov = ff_right
    elif ff_right.kind == "op" and ff_right.op == "_activations_silu" and len(ff_right.args) == 1:
        silu_prov = ff_right
        up_prov = ff_left
    else:
        return None
    chunk_match = _match_chunk_output_pair(silu_prov.args[0], up_prov)
    if chunk_match is None:
        return None
    gate_up_prov, _parts = chunk_match
    if gate_up_prov.kind != "op" or gate_up_prov.op != "_expert_linear" or len(gate_up_prov.args) < 8:
        return None
    if not _is_literal_provenance_value(gate_up_prov.args[4], False):
        return None
    if gate_up_prov.args[2] != topk_indices_prov or values_prov.args[2] != topk_indices_prov:
        return None
    if gate_up_prov.args[5] != values_prov.args[5]:
        return None
    hidden_prov = _match_expand_unsqueeze_hidden_provenance(gate_up_prov.args[1])
    if hidden_prov is None:
        return None
    actual_by_formal = _call_actual_operands_by_formal(call_node, callee_module)
    hidden = _actual_operand_for_callee_provenance(
        hidden_prov,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    topk_scores = _actual_operand_for_callee_provenance(
        scale_prov.args[0],
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    topk_indices = _actual_operand_for_callee_provenance(
        topk_indices_prov,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    gate_up_weight = _actual_operand_for_callee_provenance(
        gate_up_prov.args[6],
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    down_weight = _actual_operand_for_callee_provenance(
        values_prov.args[6],
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    transpose = _actual_operand_for_callee_provenance(
        gate_up_prov.args[5],
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    if transpose is None:
        transpose = _provenance_to_graph_operand(
            gate_up_prov.args[5],
            provenance_to_operand={
                GraphProvenance("literal", value=True): GraphLiteral(True, TypeBool()),
                GraphProvenance("literal", value=False): GraphLiteral(False, TypeBool()),
            },
        )
    if not (
        hidden is not None
        and topk_scores is not None
        and topk_indices is not None
        and isinstance(gate_up_weight, GraphPath)
        and isinstance(down_weight, GraphPath)
        and isinstance(transpose, GraphLiteral)
        and isinstance(transpose.value, bool)
    ):
        return None
    hidden_type = graph_operand_type(hidden)
    output_type = call_node.outputs[0].type_expr
    if not isinstance(hidden_type, TypeTensor) or not isinstance(output_type, TypeTensor):
        return None
    if not _tensor_prefix_dims_match(output_type, hidden_type, 2):
        return None
    return hidden, topk_scores, topk_indices, gate_up_weight, down_weight, transpose


def _torch_direct_selected_expert_swiglu_call_candidate(
    call_node: GraphNode,
    *,
    callee_module: GraphModule,
    callee_local_provenance: Mapping[str, GraphProvenance],
) -> tuple[GraphOperand, ...] | None:
    if len(call_node.outputs) != 1 or len(callee_module.outputs) != 1:
        return None
    output = callee_module.outputs[0]
    if not isinstance(output, GraphValueRef):
        return None
    output_prov = callee_local_provenance.get(output.name)
    if output_prov is None or output_prov.kind != "op" or output_prov.op != "_sum" or len(output_prov.args) < 3:
        return None
    if not _is_literal_provenance_value(output_prov.args[1], 2) or not _is_literal_provenance_value(output_prov.args[2], False):
        return None
    weighted_prov = output_prov.args[0]
    if weighted_prov.kind != "op" or weighted_prov.op not in {"_mul", "core.binary.*"} or len(weighted_prov.args) != 2:
        return None
    left, right = weighted_prov.args
    if left.kind == "op" and left.op == "_unsqueeze" and len(left.args) >= 2 and _is_literal_provenance_value(left.args[1], -1):
        scale_prov = left
        values_prov = right
    elif right.kind == "op" and right.op == "_unsqueeze" and len(right.args) >= 2 and _is_literal_provenance_value(right.args[1], -1):
        scale_prov = right
        values_prov = left
    else:
        return None
    if values_prov.kind != "op" or values_prov.op != "_expert_linear" or len(values_prov.args) < 8:
        return None
    if not _is_literal_provenance_value(values_prov.args[4], False):
        return None
    ff_prov = values_prov.args[1]
    topk_indices_prov = values_prov.args[2]
    if ff_prov.kind != "op" or ff_prov.op not in {"_mul", "core.binary.*"} or len(ff_prov.args) != 2:
        return None
    ff_left, ff_right = ff_prov.args
    if ff_left.kind == "op" and ff_left.op == "_activations_silu" and len(ff_left.args) == 1:
        silu_prov = ff_left
        up_prov = ff_right
    elif ff_right.kind == "op" and ff_right.op == "_activations_silu" and len(ff_right.args) == 1:
        silu_prov = ff_right
        up_prov = ff_left
    else:
        return None
    gate_prov = silu_prov.args[0]
    if (
        gate_prov.kind != "op"
        or gate_prov.op != "_expert_linear"
        or len(gate_prov.args) < 8
        or up_prov.kind != "op"
        or up_prov.op != "_expert_linear"
        or len(up_prov.args) < 8
    ):
        return None
    if not (
        _is_literal_provenance_value(gate_prov.args[4], False)
        and _is_literal_provenance_value(up_prov.args[4], False)
        and gate_prov.args[1] == up_prov.args[1]
        and gate_prov.args[2] == topk_indices_prov
        and up_prov.args[2] == topk_indices_prov
        and values_prov.args[2] == topk_indices_prov
        and gate_prov.args[5] == up_prov.args[5]
        and gate_prov.args[5] == values_prov.args[5]
    ):
        return None
    hidden_prov = _match_expand_unsqueeze_hidden_provenance(gate_prov.args[1])
    if hidden_prov is None:
        return None
    actual_by_formal = _call_actual_operands_by_formal(call_node, callee_module)
    hidden = _actual_operand_for_callee_provenance(
        hidden_prov,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    topk_scores = _actual_operand_for_callee_provenance(
        scale_prov.args[0],
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    topk_indices = _actual_operand_for_callee_provenance(
        topk_indices_prov,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    gate_weight = _actual_operand_for_callee_provenance(
        gate_prov.args[6],
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    up_weight = _actual_operand_for_callee_provenance(
        up_prov.args[6],
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    down_weight = _actual_operand_for_callee_provenance(
        values_prov.args[6],
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    transpose = _actual_operand_for_callee_provenance(
        gate_prov.args[5],
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    if transpose is None:
        transpose = _provenance_to_graph_operand(
            gate_prov.args[5],
            provenance_to_operand={
                GraphProvenance("literal", value=True): GraphLiteral(True, TypeBool()),
                GraphProvenance("literal", value=False): GraphLiteral(False, TypeBool()),
            },
        )
    if not (
        hidden is not None
        and topk_scores is not None
        and topk_indices is not None
        and isinstance(gate_weight, GraphPath)
        and isinstance(up_weight, GraphPath)
        and isinstance(down_weight, GraphPath)
        and isinstance(transpose, GraphLiteral)
        and isinstance(transpose.value, bool)
    ):
        return None
    hidden_type = graph_operand_type(hidden)
    output_type = call_node.outputs[0].type_expr
    if not isinstance(hidden_type, TypeTensor) or not isinstance(output_type, TypeTensor):
        return None
    if not _tensor_prefix_dims_match(output_type, hidden_type, 2):
        return None
    return hidden, topk_scores, topk_indices, gate_weight, up_weight, down_weight, transpose


def _torch_direct_selected_expert_packed_gegelu_call_candidate(
    call_node: GraphNode,
    *,
    callee_module: GraphModule,
    callee_local_provenance: Mapping[str, GraphProvenance],
) -> tuple[GraphOperand, ...] | None:
    if len(call_node.outputs) != 1 or len(callee_module.outputs) != 1:
        return None
    output = callee_module.outputs[0]
    if not isinstance(output, GraphValueRef):
        return None
    output_prov = callee_local_provenance.get(output.name)
    if output_prov is None or output_prov.kind != "op" or output_prov.op != "_sum" or len(output_prov.args) < 3:
        return None
    if not _is_literal_provenance_value(output_prov.args[1], 2) or not _is_literal_provenance_value(output_prov.args[2], False):
        return None
    weighted_prov = output_prov.args[0]
    if weighted_prov.kind != "op" or weighted_prov.op not in {"_mul", "core.binary.*"} or len(weighted_prov.args) != 2:
        return None
    left, right = weighted_prov.args
    if left.kind == "op" and left.op == "_unsqueeze" and len(left.args) >= 2 and _is_literal_provenance_value(left.args[1], -1):
        scale_prov = left
        values_prov = right
    elif right.kind == "op" and right.op == "_unsqueeze" and len(right.args) >= 2 and _is_literal_provenance_value(right.args[1], -1):
        scale_prov = right
        values_prov = left
    else:
        return None
    if values_prov.kind != "op" or values_prov.op != "_expert_linear" or len(values_prov.args) < 8:
        return None
    activation_match = _match_packed_gegelu_activation_with_alpha(values_prov.args[1])
    if activation_match is None:
        return None
    gate_up_prov, limit_prov, alpha_prov = activation_match
    if gate_up_prov.kind != "op" or gate_up_prov.op != "_expert_linear" or len(gate_up_prov.args) < 8:
        return None
    topk_indices_prov = values_prov.args[2]
    if not (
        gate_up_prov.args[2] == topk_indices_prov
        and values_prov.args[2] == topk_indices_prov
        and gate_up_prov.args[4] == values_prov.args[4]
        and gate_up_prov.args[5] == values_prov.args[5]
    ):
        return None
    hidden_prov = _match_expand_unsqueeze_hidden_provenance(gate_up_prov.args[1])
    if hidden_prov is None:
        return None
    actual_by_formal = _call_actual_operands_by_formal(call_node, callee_module)
    hidden = _actual_operand_for_callee_provenance(
        hidden_prov,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    topk_scores = _actual_operand_for_callee_provenance(
        scale_prov.args[0],
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    topk_indices = _actual_operand_for_callee_provenance(
        topk_indices_prov,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    gate_up_weight = _actual_operand_for_callee_provenance(
        gate_up_prov.args[6],
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    gate_up_bias = _actual_operand_for_callee_provenance(
        gate_up_prov.args[7],
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    down_weight = _actual_operand_for_callee_provenance(
        values_prov.args[6],
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    down_bias = _actual_operand_for_callee_provenance(
        values_prov.args[7],
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    limit = _actual_operand_for_callee_provenance(
        limit_prov,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    if limit is None:
        limit = _provenance_to_graph_operand(limit_prov, provenance_to_operand={})
    alpha = _actual_operand_for_callee_provenance(
        alpha_prov,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    if alpha is None:
        alpha = _provenance_to_graph_operand(alpha_prov, provenance_to_operand={})
    bias = _actual_operand_for_callee_provenance(
        gate_up_prov.args[4],
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    if bias is None:
        bias = _provenance_to_graph_operand(
            gate_up_prov.args[4],
            provenance_to_operand={
                GraphProvenance("literal", value=True): GraphLiteral(True, TypeBool()),
                GraphProvenance("literal", value=False): GraphLiteral(False, TypeBool()),
            },
        )
    transpose = _actual_operand_for_callee_provenance(
        gate_up_prov.args[5],
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    if transpose is None:
        transpose = _provenance_to_graph_operand(
            gate_up_prov.args[5],
            provenance_to_operand={
                GraphProvenance("literal", value=True): GraphLiteral(True, TypeBool()),
                GraphProvenance("literal", value=False): GraphLiteral(False, TypeBool()),
            },
        )
    bias_bool = _graph_bool_literal_operand(bias) if bias is not None else None
    transpose_bool = _graph_bool_literal_operand(transpose) if transpose is not None else None
    if not (
        hidden is not None
        and topk_scores is not None
        and topk_indices is not None
        and isinstance(gate_up_weight, GraphPath)
        and isinstance(gate_up_bias, GraphPath)
        and isinstance(down_weight, GraphPath)
        and isinstance(down_bias, GraphPath)
        and limit is not None
        and alpha is not None
        and bias_bool is not None
        and transpose_bool is not None
    ):
        return None
    hidden_type = graph_operand_type(hidden)
    output_type = call_node.outputs[0].type_expr
    if not isinstance(hidden_type, TypeTensor) or not isinstance(output_type, TypeTensor):
        return None
    if not _tensor_prefix_dims_match(output_type, hidden_type, 2):
        return None
    return (
        hidden,
        topk_scores,
        topk_indices,
        gate_up_weight,
        gate_up_bias,
        down_weight,
        down_bias,
        limit,
        alpha,
        bias_bool,
        transpose_bool,
    )


def _torch_selected_expert_packed_swiglu_candidate(
    repeat_node: GraphNode,
    *,
    callee_module: GraphModule,
    callee_local_provenance: Mapping[str, GraphProvenance],
    provenance_search_memo: dict[tuple[str, int, object], object],
) -> tuple[GraphOperand, ...] | None:
    if repeat_node.op.name != "core.repeat" or len(repeat_node.outputs) != 1:
        return None
    if len(repeat_node.inputs) < 4:
        return None
    if not _is_literal_value(repeat_node.inputs[2], 1):
        return None
    var_operand = repeat_node.attrs.get("var")
    if not isinstance(var_operand, GraphLiteral) or not isinstance(var_operand.value, str):
        return None
    repeat_var = var_operand.value
    if len(callee_module.outputs) != 1:
        return None
    output = callee_module.outputs[0]
    if not isinstance(output, GraphValueRef):
        return None
    output_provenance = callee_local_provenance.get(output.name)
    if output_provenance is None:
        return None
    packed_match = _find_packed_swiglu_linear_provenance(output_provenance, provenance_search_memo)
    if packed_match is None:
        return None
    gate_up_provenance, selected_input_provenance, down_provenance = packed_match
    token_index_provenance = _find_op_provenance(output_provenance, "_where_indices[1]", provenance_search_memo)
    topk_pos_provenance = _find_op_provenance(output_provenance, "_where_indices[0]", provenance_search_memo)
    if token_index_provenance is None or topk_pos_provenance is None:
        return None
    hidden_provenance = _selected_hidden_source_provenance(selected_input_provenance, token_index_provenance, provenance_search_memo)
    if hidden_provenance is None:
        return None
    unsqueeze_provenance = _find_weighted_update_unsqueeze_provenance(output_provenance, down_provenance, provenance_search_memo)
    if unsqueeze_provenance is None or not unsqueeze_provenance.args:
        return None
    topk_scores_provenance = _selected_scores_source_provenance(
        unsqueeze_provenance.args[0],
        token_index_provenance,
        topk_pos_provenance,
        provenance_search_memo,
    )
    if topk_scores_provenance is None:
        return None
    topk_indices_provenance = None
    if token_index_provenance.args:
        transpose = token_index_provenance.args[0]
        if transpose.kind == "op" and transpose.op == "_transpose" and transpose.args:
            eq = transpose.args[0]
            if eq.kind == "op" and eq.op == "_eq" and eq.args:
                reshaped = eq.args[0]
                if reshaped.kind == "op" and reshaped.op == "_reshape" and reshaped.args:
                    topk_indices_provenance = reshaped.args[0]
    if topk_indices_provenance is None:
        return None
    iter_provenance = GraphProvenance("op", op="core.repeat.iter")
    if not (
        _provenance_contains(token_index_provenance, iter_provenance, provenance_search_memo)
        and _provenance_contains(selected_input_provenance, token_index_provenance, provenance_search_memo)
        and _provenance_contains(unsqueeze_provenance, topk_pos_provenance, provenance_search_memo)
    ):
        return None
    actual_by_formal = _repeat_actual_operands_by_formal(repeat_node, callee_module)
    hidden = _actual_operand_for_callee_provenance(
        hidden_provenance,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    topk_scores = _actual_operand_for_callee_provenance(
        topk_scores_provenance,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    topk_indices = _actual_operand_for_callee_provenance(
        topk_indices_provenance,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    if hidden is None or topk_scores is None or topk_indices is None:
        return None
    gate_up_weight = _path_without_repeat_iter_segment(gate_up_provenance.args[6], repeat_var=repeat_var)
    down_weight = _path_without_repeat_iter_segment(down_provenance.args[6], repeat_var=repeat_var)
    if gate_up_weight is None or down_weight is None:
        return None
    transpose = _provenance_to_graph_operand(
        gate_up_provenance.args[4],
        provenance_to_operand={
            GraphProvenance("literal", value=True): GraphLiteral(True, TypeBool()),
            GraphProvenance("literal", value=False): GraphLiteral(False, TypeBool()),
        },
    )
    if not isinstance(transpose, GraphLiteral) or not isinstance(transpose.value, bool):
        return None
    return (
        hidden,
        topk_scores,
        topk_indices,
        gate_up_weight,
        down_weight,
        transpose,
    )


def _torch_selected_expert_packed_gegelu_candidate(
    repeat_node: GraphNode,
    *,
    callee_module: GraphModule,
    callee_local_provenance: Mapping[str, GraphProvenance],
    provenance_search_memo: dict[tuple[str, int, object], object],
) -> tuple[GraphOperand, ...] | None:
    if repeat_node.op.name != "core.repeat" or len(repeat_node.outputs) != 1:
        return None
    if len(repeat_node.inputs) < 4:
        return None
    if not _is_literal_value(repeat_node.inputs[2], 1):
        return None
    var_operand = repeat_node.attrs.get("var")
    if not isinstance(var_operand, GraphLiteral) or not isinstance(var_operand.value, str):
        return None
    repeat_var = var_operand.value
    if len(callee_module.outputs) != 1:
        return None
    output = callee_module.outputs[0]
    if not isinstance(output, GraphValueRef):
        return None
    output_provenance = callee_local_provenance.get(output.name)
    if output_provenance is None:
        return None
    gegelu_match = _find_packed_gegelu_linear_provenance(output_provenance, provenance_search_memo)
    if gegelu_match is None:
        return None
    gate_up_provenance, selected_input_provenance, down_provenance, limit_provenance = gegelu_match
    token_index_provenance = _find_op_provenance(output_provenance, "_where_indices[1]", provenance_search_memo)
    topk_pos_provenance = _find_op_provenance(output_provenance, "_where_indices[0]", provenance_search_memo)
    if token_index_provenance is None or topk_pos_provenance is None:
        return None
    hidden_provenance = _selected_hidden_source_provenance(selected_input_provenance, token_index_provenance, provenance_search_memo)
    if hidden_provenance is None:
        return None
    unsqueeze_provenance = _find_weighted_update_unsqueeze_provenance(output_provenance, down_provenance, provenance_search_memo)
    if unsqueeze_provenance is None or not unsqueeze_provenance.args:
        return None
    topk_scores_provenance = _selected_scores_source_provenance(
        unsqueeze_provenance.args[0],
        token_index_provenance,
        topk_pos_provenance,
        provenance_search_memo,
    )
    if topk_scores_provenance is None:
        return None
    topk_indices_provenance = None
    if token_index_provenance.args:
        transpose = token_index_provenance.args[0]
        if transpose.kind == "op" and transpose.op == "_transpose" and transpose.args:
            eq = transpose.args[0]
            if eq.kind == "op" and eq.op == "_eq" and eq.args:
                reshaped = eq.args[0]
                if reshaped.kind == "op" and reshaped.op == "_reshape" and reshaped.args:
                    topk_indices_provenance = reshaped.args[0]
    if topk_indices_provenance is None:
        return None
    iter_provenance = GraphProvenance("op", op="core.repeat.iter")
    if not (
        _provenance_contains(token_index_provenance, iter_provenance, provenance_search_memo)
        and _provenance_contains(selected_input_provenance, token_index_provenance, provenance_search_memo)
        and _provenance_contains(unsqueeze_provenance, topk_pos_provenance, provenance_search_memo)
    ):
        return None
    actual_by_formal = _repeat_actual_operands_by_formal(repeat_node, callee_module)
    hidden = _actual_operand_for_callee_provenance(
        hidden_provenance,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    topk_scores = _actual_operand_for_callee_provenance(
        topk_scores_provenance,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    topk_indices = _actual_operand_for_callee_provenance(
        topk_indices_provenance,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    provenance_to_operand = _module_provenance_to_operand_map(
        callee_module,
        local_provenance=callee_local_provenance,
    )
    limit = _provenance_to_graph_operand(
        limit_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    if hidden is None or topk_scores is None or topk_indices is None or limit is None:
        return None
    gate_up_weight = _path_without_repeat_iter_segment(gate_up_provenance.args[6], repeat_var=repeat_var)
    down_weight = _path_without_repeat_iter_segment(down_provenance.args[6], repeat_var=repeat_var)
    if gate_up_weight is None or down_weight is None:
        return None
    bias = _provenance_to_graph_operand(
        gate_up_provenance.args[3],
        provenance_to_operand={
            GraphProvenance("literal", value=True): GraphLiteral(True, TypeBool()),
            GraphProvenance("literal", value=False): GraphLiteral(False, TypeBool()),
        },
    )
    transpose = _provenance_to_graph_operand(
        gate_up_provenance.args[4],
        provenance_to_operand={
            GraphProvenance("literal", value=True): GraphLiteral(True, TypeBool()),
            GraphProvenance("literal", value=False): GraphLiteral(False, TypeBool()),
        },
    )
    if not isinstance(bias, GraphLiteral) or not isinstance(bias.value, bool):
        return None
    if not isinstance(transpose, GraphLiteral) or not isinstance(transpose.value, bool):
        return None
    gate_up_bias: GraphOperand = GraphLiteral(None, TypeNull())
    down_bias: GraphOperand = GraphLiteral(None, TypeNull())
    if bias.value:
        gate_up_bias_path = _path_without_repeat_iter_segment(gate_up_provenance.args[7], repeat_var=repeat_var)
        down_bias_path = _path_without_repeat_iter_segment(down_provenance.args[7], repeat_var=repeat_var)
        if gate_up_bias_path is None or down_bias_path is None:
            return None
        gate_up_bias = gate_up_bias_path
        down_bias = down_bias_path
    return (
        hidden,
        topk_scores,
        topk_indices,
        gate_up_weight,
        gate_up_bias,
        down_weight,
        down_bias,
        limit,
        bias,
        transpose,
    )


def _torch_selected_expert_clamped_packed_swiglu_candidate(
    repeat_node: GraphNode,
    *,
    callee_module: GraphModule,
    callee_local_provenance: Mapping[str, GraphProvenance],
    provenance_search_memo: dict[tuple[str, int, object], object],
) -> tuple[GraphOperand, ...] | None:
    if repeat_node.op.name != "core.repeat" or len(repeat_node.outputs) != 1:
        return None
    if len(repeat_node.inputs) < 4:
        return None
    if not _is_literal_value(repeat_node.inputs[2], 1):
        return None
    var_operand = repeat_node.attrs.get("var")
    if not isinstance(var_operand, GraphLiteral) or not isinstance(var_operand.value, str):
        return None
    repeat_var = var_operand.value
    if len(callee_module.outputs) != 1:
        return None
    output = callee_module.outputs[0]
    if not isinstance(output, GraphValueRef):
        return None
    output_provenance = callee_local_provenance.get(output.name)
    if output_provenance is None:
        return None
    swiglu_match = _find_clamped_packed_swiglu_linear_provenance(output_provenance, provenance_search_memo)
    if swiglu_match is None:
        return None
    gate_up_provenance, selected_input_provenance, down_provenance, limit_provenance = swiglu_match
    token_index_provenance = _find_op_provenance(output_provenance, "_where_indices[1]", provenance_search_memo)
    topk_pos_provenance = _find_op_provenance(output_provenance, "_where_indices[0]", provenance_search_memo)
    if token_index_provenance is None or topk_pos_provenance is None:
        return None
    hidden_provenance = _selected_hidden_source_provenance(selected_input_provenance, token_index_provenance, provenance_search_memo)
    if hidden_provenance is None:
        return None
    unsqueeze_provenance = _find_weighted_update_unsqueeze_provenance(output_provenance, down_provenance, provenance_search_memo)
    if unsqueeze_provenance is None or not unsqueeze_provenance.args:
        return None
    topk_scores_provenance = _selected_scores_source_provenance(
        unsqueeze_provenance.args[0],
        token_index_provenance,
        topk_pos_provenance,
        provenance_search_memo,
    )
    if topk_scores_provenance is None:
        return None
    topk_indices_provenance = None
    if token_index_provenance.args:
        transpose = token_index_provenance.args[0]
        if transpose.kind == "op" and transpose.op == "_transpose" and transpose.args:
            eq = transpose.args[0]
            if eq.kind == "op" and eq.op == "_eq" and eq.args:
                reshaped = eq.args[0]
                if reshaped.kind == "op" and reshaped.op == "_reshape" and reshaped.args:
                    topk_indices_provenance = reshaped.args[0]
    if topk_indices_provenance is None:
        return None
    iter_provenance = GraphProvenance("op", op="core.repeat.iter")
    if not (
        _provenance_contains(token_index_provenance, iter_provenance, provenance_search_memo)
        and _provenance_contains(selected_input_provenance, token_index_provenance, provenance_search_memo)
        and _provenance_contains(unsqueeze_provenance, topk_pos_provenance, provenance_search_memo)
    ):
        return None
    actual_by_formal = _repeat_actual_operands_by_formal(repeat_node, callee_module)
    hidden = _actual_operand_for_callee_provenance(
        hidden_provenance,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    topk_scores = _actual_operand_for_callee_provenance(
        topk_scores_provenance,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    topk_indices = _actual_operand_for_callee_provenance(
        topk_indices_provenance,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    provenance_to_operand = _module_provenance_to_operand_map(
        callee_module,
        local_provenance=callee_local_provenance,
    )
    limit = _provenance_to_graph_operand(
        limit_provenance,
        provenance_to_operand=provenance_to_operand,
    )
    if hidden is None or topk_scores is None or topk_indices is None or limit is None:
        return None
    gate_up_weight = _path_without_repeat_iter_segment(gate_up_provenance.args[6], repeat_var=repeat_var)
    down_weight = _path_without_repeat_iter_segment(down_provenance.args[6], repeat_var=repeat_var)
    if gate_up_weight is None or down_weight is None:
        return None
    transpose = _provenance_to_graph_operand(
        gate_up_provenance.args[4],
        provenance_to_operand={
            GraphProvenance("literal", value=True): GraphLiteral(True, TypeBool()),
            GraphProvenance("literal", value=False): GraphLiteral(False, TypeBool()),
        },
    )
    if not isinstance(transpose, GraphLiteral) or not isinstance(transpose.value, bool):
        return None
    return (
        hidden,
        topk_scores,
        topk_indices,
        gate_up_weight,
        down_weight,
        limit,
        transpose,
    )


def _torch_selected_expert_relu2_candidate(
    repeat_node: GraphNode,
    *,
    callee_module: GraphModule,
    callee_local_provenance: Mapping[str, GraphProvenance],
    provenance_search_memo: dict[tuple[str, int, object], object],
) -> tuple[GraphOperand, ...] | None:
    if repeat_node.op.name != "core.repeat" or len(repeat_node.outputs) != 1:
        return None
    if len(repeat_node.inputs) < 4:
        return None
    if not _is_literal_value(repeat_node.inputs[2], 1):
        return None
    var_operand = repeat_node.attrs.get("var")
    if not isinstance(var_operand, GraphLiteral) or not isinstance(var_operand.value, str):
        return None
    repeat_var = var_operand.value
    if len(callee_module.outputs) != 1:
        return None
    output = callee_module.outputs[0]
    if not isinstance(output, GraphValueRef):
        return None
    output_provenance = callee_local_provenance.get(output.name)
    if output_provenance is None:
        return None
    relu2_match = _find_relu2_linear_provenance(output_provenance, provenance_search_memo)
    if relu2_match is None:
        return None
    up_provenance, selected_input_provenance, down_provenance = relu2_match
    token_index_provenance = _find_op_provenance(output_provenance, "_where_indices[1]", provenance_search_memo)
    topk_pos_provenance = _find_op_provenance(output_provenance, "_where_indices[0]", provenance_search_memo)
    if token_index_provenance is None or topk_pos_provenance is None:
        return None
    hidden_provenance = _selected_hidden_source_provenance(selected_input_provenance, token_index_provenance, provenance_search_memo)
    if hidden_provenance is None:
        return None
    unsqueeze_provenance = _find_weighted_update_unsqueeze_provenance(output_provenance, down_provenance, provenance_search_memo)
    if unsqueeze_provenance is None or not unsqueeze_provenance.args:
        return None
    topk_scores_provenance = _selected_scores_source_provenance(
        unsqueeze_provenance.args[0],
        token_index_provenance,
        topk_pos_provenance,
        provenance_search_memo,
    )
    if topk_scores_provenance is None:
        return None
    topk_indices_provenance = None
    if token_index_provenance.args:
        transpose = token_index_provenance.args[0]
        if transpose.kind == "op" and transpose.op == "_transpose" and transpose.args:
            eq = transpose.args[0]
            if eq.kind == "op" and eq.op == "_eq" and eq.args:
                reshaped = eq.args[0]
                if reshaped.kind == "op" and reshaped.op == "_reshape" and reshaped.args:
                    topk_indices_provenance = reshaped.args[0]
    if topk_indices_provenance is None:
        return None
    iter_provenance = GraphProvenance("op", op="core.repeat.iter")
    if not (
        _provenance_contains(token_index_provenance, iter_provenance, provenance_search_memo)
        and _provenance_contains(selected_input_provenance, token_index_provenance, provenance_search_memo)
        and _provenance_contains(unsqueeze_provenance, topk_pos_provenance, provenance_search_memo)
    ):
        return None
    actual_by_formal = _repeat_actual_operands_by_formal(repeat_node, callee_module)
    hidden = _actual_operand_for_callee_provenance(
        hidden_provenance,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    topk_scores = _actual_operand_for_callee_provenance(
        topk_scores_provenance,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    topk_indices = _actual_operand_for_callee_provenance(
        topk_indices_provenance,
        callee_module=callee_module,
        callee_local_provenance=callee_local_provenance,
        actual_by_formal=actual_by_formal,
    )
    if hidden is None or topk_scores is None or topk_indices is None:
        return None
    up_weight = _path_without_repeat_iter_segment(up_provenance.args[6], repeat_var=repeat_var)
    down_weight = _path_without_repeat_iter_segment(down_provenance.args[6], repeat_var=repeat_var)
    if up_weight is None or down_weight is None:
        return None
    transpose = _provenance_to_graph_operand(
        up_provenance.args[4],
        provenance_to_operand={
            GraphProvenance("literal", value=True): GraphLiteral(True, TypeBool()),
            GraphProvenance("literal", value=False): GraphLiteral(False, TypeBool()),
        },
    )
    if not isinstance(transpose, GraphLiteral) or not isinstance(transpose.value, bool):
        return None
    return (
        hidden,
        topk_scores,
        topk_indices,
        up_weight,
        down_weight,
        transpose,
    )


def _has_torch_selected_expert_intrinsic_candidates(graph: GraphProgram) -> bool:
    del graph
    # This is only a cost prefilter; the rewrite itself is guarded by
    # primitive-level provenance.  Wrapper-preserving optimized graphs may not
    # expose `_expert_linear` as a direct node op even when provenance proves the
    # selected-expert pattern, so do not reject here based on surface op names.
    return True


def _rewrite_torch_selected_expert_intrinsics(
    graph: GraphProgram,
    *,
    enabled_intrinsics: frozenset[str],
    op_prefix: str = "__torch",
) -> GraphProgram:
    if not _has_torch_selected_expert_intrinsic_candidates(graph):
        return graph
    modules_by_name = {module.name: module for module in graph.modules}
    provenance = infer_graph_provenance(graph)
    module_provenance_to_operand: dict[str, dict[GraphProvenance, GraphOperand]] = {}
    for module in graph.modules:
        local_provenance = provenance.module_local_provenance.get(module.name, {})
        module_provenance_to_operand[module.name] = _module_provenance_to_operand_map(
            module,
            local_provenance=local_provenance,
        )
    provenance_search_memo: dict[tuple[str, int, object], object] = {}
    changed = False
    new_modules: list[GraphModule] = []
    for module in graph.modules:
        local_provenance = provenance.module_local_provenance.get(module.name, {})
        provenance_to_operand = module_provenance_to_operand.get(module.name, {})
        producer_by_output = {output.name: node for node in module.nodes for output in node.outputs}
        new_nodes: list[GraphNode] = []
        for node in module.nodes:
            callee = modules_by_name.get(node.op.name)
            if callee is not None:
                inputs = _torch_direct_selected_expert_packed_gegelu_call_candidate(
                    node,
                    callee_module=callee,
                    callee_local_provenance=provenance.module_local_provenance.get(callee.name, {}),
                )
                if inputs is not None:
                    op_name = f"{op_prefix}_selected_expert_packed_gegelu_ffn"
                    if _backend_intrinsic_enabled(enabled_intrinsics, op_name):
                        changed = True
                        new_nodes.append(
                            replace(
                                node,
                                op=GraphOp(op_name),
                                inputs=inputs,
                                attrs={},
                            )
                        )
                        continue
                inputs = _torch_direct_selected_expert_swiglu_call_candidate(
                    node,
                    callee_module=callee,
                    callee_local_provenance=provenance.module_local_provenance.get(callee.name, {}),
                )
                if inputs is not None:
                    op_name = f"{op_prefix}_selected_expert_swiglu_ffn"
                    if _backend_intrinsic_enabled(enabled_intrinsics, op_name):
                        changed = True
                        new_nodes.append(
                            replace(
                                node,
                                op=GraphOp(op_name),
                                inputs=inputs,
                                attrs={},
                            )
                        )
                        continue
                inputs = _torch_direct_selected_expert_packed_swiglu_call_candidate(
                    node,
                    callee_module=callee,
                    callee_local_provenance=provenance.module_local_provenance.get(callee.name, {}),
                )
                if inputs is not None:
                    op_name = f"{op_prefix}_selected_expert_packed_swiglu_ffn"
                    if _backend_intrinsic_enabled(enabled_intrinsics, op_name):
                        changed = True
                        new_nodes.append(
                            replace(
                                node,
                                op=GraphOp(op_name),
                                inputs=inputs,
                                attrs={},
                            )
                        )
                        continue
            inputs = _torch_direct_selected_expert_swiglu_candidate(
                node,
                local_provenance=local_provenance,
                provenance_to_operand=provenance_to_operand,
            )
            if inputs is not None:
                op_name = f"{op_prefix}_selected_expert_swiglu_ffn"
                if _backend_intrinsic_enabled(enabled_intrinsics, op_name):
                    changed = True
                    new_nodes.append(
                        replace(
                            node,
                            op=GraphOp(op_name),
                            inputs=inputs,
                            attrs={},
                        )
                    )
                    continue
            inputs = _torch_direct_selected_expert_packed_swiglu_candidate(
                node,
                local_provenance=local_provenance,
                provenance_to_operand=provenance_to_operand,
                producer_by_output=producer_by_output,
            )
            if inputs is not None:
                op_name = f"{op_prefix}_selected_expert_packed_swiglu_ffn"
                if _backend_intrinsic_enabled(enabled_intrinsics, op_name):
                    changed = True
                    new_nodes.append(
                        replace(
                            node,
                            op=GraphOp(op_name),
                            inputs=inputs,
                            attrs={},
                        )
                    )
                    continue
            if node.op.name != "core.repeat":
                new_nodes.append(node)
                continue
            callee_operand = node.attrs.get("callee")
            if not isinstance(callee_operand, GraphLiteral) or not isinstance(callee_operand.value, str):
                new_nodes.append(node)
                continue
            callee = modules_by_name.get(callee_operand.value)
            if callee is None:
                new_nodes.append(node)
                continue
            inputs = _torch_selected_expert_packed_swiglu_candidate(
                node,
                callee_module=callee,
                callee_local_provenance=provenance.module_local_provenance.get(callee.name, {}),
                provenance_search_memo=provenance_search_memo,
            )
            op_name = f"{op_prefix}_selected_expert_packed_swiglu_ffn"
            if inputs is None:
                inputs = _torch_selected_expert_packed_gegelu_candidate(
                    node,
                    callee_module=callee,
                    callee_local_provenance=provenance.module_local_provenance.get(callee.name, {}),
                    provenance_search_memo=provenance_search_memo,
                )
                op_name = f"{op_prefix}_selected_expert_packed_gegelu_ffn"
            if inputs is None:
                inputs = _torch_selected_expert_clamped_packed_swiglu_candidate(
                    node,
                    callee_module=callee,
                    callee_local_provenance=provenance.module_local_provenance.get(callee.name, {}),
                    provenance_search_memo=provenance_search_memo,
                )
                op_name = f"{op_prefix}_selected_expert_clamped_packed_swiglu_ffn"
            if inputs is None:
                inputs = _torch_selected_expert_relu2_candidate(
                    node,
                    callee_module=callee,
                    callee_local_provenance=provenance.module_local_provenance.get(callee.name, {}),
                    provenance_search_memo=provenance_search_memo,
                )
                op_name = f"{op_prefix}_selected_expert_relu2_ffn"
            if inputs is None:
                new_nodes.append(node)
                continue
            if not _backend_intrinsic_enabled(enabled_intrinsics, op_name):
                new_nodes.append(node)
                continue
            changed = True
            new_nodes.append(
                replace(
                    node,
                    op=GraphOp(op_name),
                    inputs=inputs,
                    attrs={},
                )
            )
        new_modules.append(replace(module, nodes=tuple(new_nodes)))
    return replace(graph, modules=tuple(new_modules)) if changed else graph


def _rewrite_torch_weighted_topk_sum_intrinsics(graph: GraphProgram) -> GraphProgram:
    has_candidate_shape = False
    for module in graph.modules:
        names = [node.op.name for node in module.nodes]
        for index in range(len(names) - 2):
            if names[index] == "_unsqueeze" and names[index + 1] in {"_mul", "core.binary.*"} and names[index + 2] == "_sum":
                has_candidate_shape = True
                break
        if has_candidate_shape:
            break
    if not has_candidate_shape:
        return graph
    changed = False
    new_modules: list[GraphModule] = []
    for module in graph.modules:
        value_ref_counts = _module_value_ref_counts(module)
        new_nodes: list[GraphNode] = []
        index = 0
        while index < len(module.nodes):
            if index + 2 >= len(module.nodes):
                new_nodes.append(module.nodes[index])
                index += 1
                continue
            inputs = _torch_weighted_topk_sum_candidate(
                module.nodes[index],
                module.nodes[index + 1],
                module.nodes[index + 2],
                local_provenance={},
                provenance_to_operand={},
                value_ref_counts=value_ref_counts,
            )
            if inputs is None:
                new_nodes.append(module.nodes[index])
                index += 1
                continue
            sum_node = module.nodes[index + 2]
            changed = True
            new_nodes.append(
                replace(
                    sum_node,
                    op=GraphOp("__torch_weighted_topk_sum"),
                    inputs=inputs,
                    attrs={},
                )
            )
            index += 3
        new_modules.append(replace(module, nodes=tuple(new_nodes)))
    return replace(graph, modules=tuple(new_modules)) if changed else graph


def _torch_topk_normalize_candidate(
    cumsum_node: GraphNode,
    slice_node: GraphNode,
    div_node: GraphNode,
    cast_node: GraphNode,
    *,
    local_provenance: Mapping[str, GraphProvenance],
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
    value_ref_counts: Mapping[str, int],
) -> tuple[GraphOperand, ...] | None:
    if (
        len(cumsum_node.outputs) != 1
        or len(slice_node.outputs) != 1
        or len(div_node.outputs) != 1
        or len(cast_node.outputs) != 1
    ):
        return None
    cumsum_name = cumsum_node.outputs[0].name
    slice_name = slice_node.outputs[0].name
    div_name = div_node.outputs[0].name
    if value_ref_counts.get(cumsum_name, 0) != 1:
        return None
    if value_ref_counts.get(slice_name, 0) != 1:
        return None
    if value_ref_counts.get(div_name, 0) != 1:
        return None
    cumsum_prov = local_provenance.get(cumsum_name)
    slice_prov = local_provenance.get(slice_name)
    div_prov = local_provenance.get(div_name)
    cast_prov = local_provenance.get(cast_node.outputs[0].name)
    if cumsum_prov is None or slice_prov is None or div_prov is None or cast_prov is None:
        return None
    return _torch_topk_normalize_inputs_from_provenance(
        cumsum_prov=cumsum_prov,
        slice_prov=slice_prov,
        div_prov=div_prov,
        cast_prov=cast_prov,
        cast_node=cast_node,
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )


def _torch_topk_normalize_nested_candidate(
    cumsum_node: GraphNode,
    slice_node: GraphNode,
    cast_node: GraphNode,
    *,
    local_provenance: Mapping[str, GraphProvenance],
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
    value_ref_counts: Mapping[str, int],
) -> tuple[GraphOperand, ...] | None:
    if len(cumsum_node.outputs) != 1 or len(slice_node.outputs) != 1 or len(cast_node.outputs) != 1:
        return None
    cumsum_name = cumsum_node.outputs[0].name
    slice_name = slice_node.outputs[0].name
    if value_ref_counts.get(cumsum_name, 0) != 1:
        return None
    if value_ref_counts.get(slice_name, 0) != 1:
        return None
    cumsum_prov = local_provenance.get(cumsum_name)
    slice_prov = local_provenance.get(slice_name)
    cast_prov = local_provenance.get(cast_node.outputs[0].name)
    if cumsum_prov is None or slice_prov is None or cast_prov is None:
        return None
    if cast_prov.kind != "op" or cast_prov.op != "_cast_like" or len(cast_prov.args) < 1:
        return None
    div_prov = cast_prov.args[0]
    return _torch_topk_normalize_inputs_from_provenance(
        cumsum_prov=cumsum_prov,
        slice_prov=slice_prov,
        div_prov=div_prov,
        cast_prov=cast_prov,
        cast_node=cast_node,
        local_provenance=local_provenance,
        provenance_to_operand=provenance_to_operand,
    )


def _rewrite_torch_topk_normalize_intrinsics(graph: GraphProgram) -> GraphProgram:
    has_candidate_shape = False
    for module in graph.modules:
        names = [node.op.name for node in module.nodes]
        for index in range(len(names)):
            if names[index : index + 4] == ["_cumsum", "_slice", "core.binary./", "_cast_like"]:
                has_candidate_shape = True
                break
            if names[index : index + 4] == ["_cumsum", "_slice", "_div", "_cast_like"]:
                has_candidate_shape = True
                break
            if names[index : index + 3] == ["_cumsum", "_slice", "_cast_like"]:
                has_candidate_shape = True
                break
        if has_candidate_shape:
            break
    if not has_candidate_shape:
        return graph
    provenance = infer_graph_provenance(graph)
    module_provenance_to_operand: dict[str, dict[GraphProvenance, GraphOperand]] = {}
    for module in graph.modules:
        local_provenance = provenance.module_local_provenance.get(module.name, {})
        module_provenance_to_operand[module.name] = _module_provenance_to_operand_map(
            module,
            local_provenance=local_provenance,
        )
    changed = False
    new_modules: list[GraphModule] = []
    for module in graph.modules:
        local_provenance = provenance.module_local_provenance.get(module.name, {})
        provenance_to_operand = module_provenance_to_operand.get(module.name, {})
        value_ref_counts = _module_value_ref_counts(module)
        new_nodes: list[GraphNode] = []
        index = 0
        while index < len(module.nodes):
            if index + 2 >= len(module.nodes):
                new_nodes.append(module.nodes[index])
                index += 1
                continue
            inputs = None
            if index + 3 < len(module.nodes):
                inputs = _torch_topk_normalize_candidate(
                    module.nodes[index],
                    module.nodes[index + 1],
                    module.nodes[index + 2],
                    module.nodes[index + 3],
                    local_provenance=local_provenance,
                    provenance_to_operand=provenance_to_operand,
                    value_ref_counts=value_ref_counts,
                )
            if inputs is None:
                nested_inputs = _torch_topk_normalize_nested_candidate(
                    module.nodes[index],
                    module.nodes[index + 1],
                    module.nodes[index + 2],
                    local_provenance=local_provenance,
                    provenance_to_operand=provenance_to_operand,
                    value_ref_counts=value_ref_counts,
                )
                if nested_inputs is None:
                    new_nodes.append(module.nodes[index])
                    index += 1
                    continue
                cast_node = module.nodes[index + 2]
                changed = True
                new_nodes.append(
                    replace(
                        cast_node,
                        op=GraphOp("__torch_topk_normalize"),
                        inputs=nested_inputs,
                        attrs={},
                    )
                )
                index += 3
                continue
            cast_node = module.nodes[index + 3]
            changed = True
            new_nodes.append(
                replace(
                    cast_node,
                    op=GraphOp("__torch_topk_normalize"),
                    inputs=inputs,
                    attrs={},
                )
            )
            index += 4
        new_modules.append(replace(module, nodes=tuple(new_nodes)))
    return replace(graph, modules=tuple(new_modules)) if changed else graph


def _match_packed_gegelu_channel(
    provenance: GraphProvenance,
    *,
    start: int,
    end: int,
) -> tuple[GraphProvenance, GraphProvenance] | None:
    if provenance.kind != "op" or provenance.op != "_reshape" or len(provenance.args) < 1:
        return None
    sliced = provenance.args[0]
    if sliced.kind != "op" or sliced.op != "_slice" or len(sliced.args) < 4:
        return None
    pair, dim_prov, start_prov, end_prov = sliced.args[:4]
    if not _is_literal_provenance_value(dim_prov, -1):
        return None
    if not _is_literal_provenance_value(start_prov, start):
        return None
    if not _is_literal_provenance_value(end_prov, end):
        return None
    if pair.kind != "op" or pair.op != "_reshape" or len(pair.args) < 1:
        return None
    return pair.args[0], pair


def _match_limit_negation(
    provenance: GraphProvenance,
    limit: GraphProvenance,
) -> bool:
    args = _provenance_binary_args(provenance, "core.binary.-", "_sub")
    if args is None:
        return False
    left, right = args
    return _is_literal_number_provenance_value(left, 0.0) and right == limit


def _match_packed_gegelu_clamped_channels(
    gate: GraphProvenance,
    up: GraphProvenance,
) -> tuple[GraphProvenance, GraphProvenance] | None:
    if gate.kind != "op" or gate.op != "_clamp" or len(gate.args) < 3:
        return None
    if up.kind != "op" or up.op != "_clamp" or len(up.args) < 3:
        return None
    gate_raw, gate_min, gate_max = gate.args[:3]
    up_raw, up_min, up_max = up.args[:3]
    if not _is_literal_provenance_value(gate_min, None):
        return None
    if gate_max != up_max:
        return None
    if not _match_limit_negation(up_min, gate_max):
        return None
    gate_source = _match_packed_gegelu_channel(gate_raw, start=0, end=1)
    up_source = _match_packed_gegelu_channel(up_raw, start=1, end=2)
    if gate_source is None or up_source is None:
        return None
    gate_input, gate_pair = gate_source
    up_input, up_pair = up_source
    if gate_input != up_input or gate_pair != up_pair:
        return None
    return gate_input, gate_max


def _match_packed_gegelu_activation(
    provenance: GraphProvenance,
) -> tuple[GraphProvenance, GraphProvenance] | None:
    matched = _match_packed_gegelu_activation_with_alpha(provenance)
    if matched is None:
        return None
    source, limit, alpha = matched
    if not _is_literal_number_provenance_value(alpha, 1.702):
        return None
    return source, limit


def _match_packed_gegelu_activation_with_alpha(
    provenance: GraphProvenance,
) -> tuple[GraphProvenance, GraphProvenance, GraphProvenance] | None:
    outer_mul_args = _provenance_binary_args(provenance, "core.binary.*", "_mul")
    if outer_mul_args is None:
        return None
    for up_plus_one, gate_term in (outer_mul_args, outer_mul_args[::-1]):
        add_args = _provenance_binary_args(up_plus_one, "core.binary.+", "_add")
        if add_args is None:
            continue
        up_candidate: GraphProvenance | None = None
        if _is_literal_number_provenance_value(add_args[0], 1.0):
            up_candidate = add_args[1]
        elif _is_literal_number_provenance_value(add_args[1], 1.0):
            up_candidate = add_args[0]
        if up_candidate is None:
            continue
        gate_term_args = _provenance_binary_args(gate_term, "core.binary.*", "_mul")
        if gate_term_args is None:
            continue
        for gate_candidate, sigmoid_candidate in (gate_term_args, gate_term_args[::-1]):
            if sigmoid_candidate.kind != "op" or sigmoid_candidate.op != "_activations_sigmoid" or len(sigmoid_candidate.args) != 1:
                continue
            sigmoid_input_args = _provenance_binary_args(sigmoid_candidate.args[0], "core.binary.*", "_mul")
            if sigmoid_input_args is None:
                continue
            alpha: GraphProvenance | None = None
            if sigmoid_input_args[0] == gate_candidate:
                alpha = sigmoid_input_args[1]
            elif sigmoid_input_args[1] == gate_candidate:
                alpha = sigmoid_input_args[0]
            if alpha is None:
                continue
            matched = _match_packed_gegelu_clamped_channels(gate_candidate, up_candidate)
            if matched is not None:
                source, limit = matched
                return source, limit, alpha
    return None


def _packed_gegelu_type_shape_matches(
    source_operand: GraphOperand,
    output_type: TypeExpr | None,
) -> bool:
    source_type = graph_operand_type(source_operand)
    if not isinstance(source_type, TypeTensor) or not isinstance(output_type, TypeTensor):
        return False
    if source_type.base != output_type.base:
        return False
    if len(source_type.dims) != len(output_type.dims) or not source_type.dims:
        return False
    if tuple(source_type.dims[:-1]) != tuple(output_type.dims[:-1]):
        return False
    source_last = source_type.dims[-1]
    output_last = output_type.dims[-1]
    if isinstance(source_last, int) and isinstance(output_last, int):
        return source_last == 2 * output_last
    if isinstance(source_last, DimExprBinary) and source_last.op == "*":
        return (
            (source_last.left == output_last and source_last.right == 2)
            or (source_last.right == output_last and source_last.left == 2)
        )
    return False


def _rewrite_packed_gegelu_intrinsics(graph: GraphProgram) -> GraphProgram:
    provenance = infer_graph_provenance(graph)
    changed = False
    new_modules: list[GraphModule] = []
    for module in graph.modules:
        local_provenance = provenance.module_local_provenance.get(module.name, {})
        provenance_to_operand = _module_provenance_to_operand_map(
            module,
            local_provenance=local_provenance,
        )
        new_nodes: list[GraphNode] = []
        for node in module.nodes:
            if len(node.outputs) != 1:
                new_nodes.append(node)
                continue
            output_provenance = local_provenance.get(node.outputs[0].name)
            if output_provenance is None:
                new_nodes.append(node)
                continue
            matched = _match_packed_gegelu_activation(output_provenance)
            if matched is None:
                new_nodes.append(node)
                continue
            source_provenance, limit_provenance = matched
            source_operand = _provenance_to_graph_operand(
                source_provenance,
                provenance_to_operand=provenance_to_operand,
            )
            limit_operand = _provenance_to_graph_operand(
                limit_provenance,
                provenance_to_operand=provenance_to_operand,
            )
            if source_operand is None or limit_operand is None:
                new_nodes.append(node)
                continue
            if not _packed_gegelu_type_shape_matches(source_operand, node.outputs[0].type_expr):
                new_nodes.append(node)
                continue
            changed = True
            new_nodes.append(
                replace(
                    node,
                    op=GraphOp("_activations_gegelu"),
                    inputs=(source_operand, limit_operand),
                    attrs={},
                )
            )
        new_modules.append(replace(module, nodes=tuple(new_nodes)))
    return replace(graph, modules=tuple(new_modules)) if changed else graph


def _provenance_to_graph_operand(
    provenance: GraphProvenance,
    *,
    provenance_to_operand: Mapping[GraphProvenance, GraphOperand],
) -> GraphOperand | None:
    if provenance in provenance_to_operand:
        return provenance_to_operand[provenance]
    if provenance.kind == "literal":
        return GraphLiteral(provenance.value, TypeAny())
    return None


def _collect_literal_path_provenance_operands(
    operand: GraphOperand,
    out: dict[GraphProvenance, GraphOperand],
) -> None:
    if isinstance(operand, GraphLiteral):
        out.setdefault(GraphProvenance("literal", value=operand.value), operand)
        return
    if isinstance(operand, GraphPath):
        prefix = "@@" if operand.absolute else "@"
        out.setdefault(GraphProvenance("path", value=prefix + ".".join(operand.parts)), operand)
        return
    if not isinstance(operand, GraphExpr):
        return
    for item in (*operand.inputs, *operand.attrs.values()):
        _collect_literal_path_provenance_operands(item, out)


def _single_rope_apply_fact(
    facts,
) -> GraphRopeApplyFactorsFact | None:
    matches = [
        fact.value
        for fact in facts
        if fact.kind == "rope_apply_factors"
        and isinstance(fact.value, GraphRopeApplyFactorsFact)
        and not fact.value.interleaved
    ]
    return matches[0] if len(matches) == 1 else None


def _rope_fact_actuals(
    fact: GraphRopeApplyFactorsFact,
    *,
    callee: GraphModule,
    node: GraphNode,
) -> tuple[GraphOperand, GraphOperand, GraphOperand] | None:
    formal_to_actual = {
        formal.name: actual
        for formal, actual in zip(callee.inputs, node.inputs, strict=False)
    }
    names = tuple(_input_provenance_name(item) for item in (fact.x, fact.sin, fact.cos))
    if any(name is None for name in names):
        return None
    try:
        return tuple(formal_to_actual[name] for name in names if name is not None)  # type: ignore[return-value]
    except KeyError:
        return None


def _input_provenance_name(provenance: GraphProvenance) -> str | None:
    return provenance.name if provenance.kind == "input" else None


def _rewrite_torch_rope_intrinsics(
    graph: GraphProgram,
    *,
    enabled_intrinsics: frozenset[str],
) -> GraphProgram:
    provenance = infer_graph_provenance(graph)
    modules_by_name = {module.name: module for module in graph.modules}
    changed = False
    new_modules: list[GraphModule] = []
    for module in graph.modules:
        new_nodes: list[GraphNode] = []
        for node in module.nodes:
            rewritten = _maybe_rewrite_node_to_torch_rope_apply_factors(
                node,
                module=module,
                modules_by_name=modules_by_name,
                provenance=provenance,
            )
            if rewritten is None:
                new_nodes.append(node)
            elif not _backend_intrinsic_enabled(enabled_intrinsics, rewritten.op.name):
                new_nodes.append(node)
            else:
                changed = True
                new_nodes.append(rewritten)
        new_modules.append(replace(module, nodes=tuple(new_nodes)))
    return replace(graph, modules=tuple(new_modules)) if changed else graph


def _rewrite_assign_slice(graph: GraphProgram) -> GraphProgram:
    provenance = infer_graph_provenance(graph)
    changed = False
    new_modules: list[GraphModule] = []
    for module in graph.modules:
        local_provenance = provenance.module_local_provenance.get(module.name, {})
        provenance_to_operand: dict[GraphProvenance, GraphOperand] = {}
        for value in module.inputs:
            value_provenance = local_provenance.get(value.name)
            if value_provenance is not None:
                provenance_to_operand.setdefault(
                    value_provenance,
                    GraphValueRef(value.name, value.type_expr, value.dims),
                )
        for node in module.nodes:
            for operand in (*node.inputs, *node.attrs.values()):
                _collect_literal_path_provenance_operands(operand, provenance_to_operand)
            for output in node.outputs:
                value_provenance = local_provenance.get(output.name)
                if value_provenance is not None:
                    provenance_to_operand.setdefault(
                        value_provenance,
                        GraphValueRef(output.name, output.type_expr, output.dims),
                    )
        new_nodes: list[GraphNode] = []
        for node in module.nodes:
            rewritten = _maybe_rewrite_node_to_assign_slice(
                node,
                module=module,
                provenance=provenance,
                provenance_to_operand=provenance_to_operand,
            )
            if rewritten is None:
                new_nodes.append(node)
            else:
                changed = True
                new_nodes.append(rewritten)
        new_modules.append(replace(module, nodes=tuple(new_nodes)))
    return replace(graph, modules=tuple(new_modules)) if changed else graph


def _match_rope_apply_factors_operands(
    node: GraphNode,
) -> tuple[GraphOperand, GraphOperand, GraphOperand] | None:
    if node.attrs:
        return None
    left, right = _match_graph_binary(node.op.name, node.inputs, "core.binary.+")
    if left is None or right is None:
        return None
    first = _match_rope_scaled_operand(left)
    second = _match_rope_scaled_operand(right)
    if first is None or second is None:
        return None
    x_a, factor_a, rotated_a = first
    x_b, factor_b, rotated_b = second
    if not rotated_a and rotated_b and x_a == x_b:
        return x_a, factor_b, factor_a
    if not rotated_b and rotated_a and x_a == x_b:
        return x_a, factor_a, factor_b
    return None


def _match_rope_scaled_operand(
    operand: GraphOperand,
) -> tuple[GraphOperand, GraphOperand, bool] | None:
    left, right = _match_graph_expr_binary(operand, "core.binary.*")
    if left is None or right is None:
        return None
    left_rot = _match_rope_rotate_half_noninterleaved_operand(left)
    if left_rot is not None:
        return left_rot, right, True
    right_rot = _match_rope_rotate_half_noninterleaved_operand(right)
    if right_rot is not None:
        return right_rot, left, True
    if _is_expand_operand(right):
        return left, right, False
    if _is_expand_operand(left):
        return right, left, False
    return None


def _match_rope_rotate_half_noninterleaved_operand(
    operand: GraphOperand,
) -> GraphOperand | None:
    if not isinstance(operand, GraphExpr):
        return None
    if operand.op.name != "_concat" or len(operand.inputs) < 3 or operand.attrs:
        return None
    first, second, dim = operand.inputs[:3]
    if not _is_literal_value(dim, -1):
        return None
    negated = _match_negated_slice_operand(first)
    plain = _match_slice_operand(second)
    if negated is None or plain is None:
        return None
    x_hi, hi_dim, hi_start, _hi_end = negated
    x_lo, lo_dim, lo_start, lo_end = plain
    if x_hi != x_lo:
        return None
    if not _is_literal_value(hi_dim, -1) or not _is_literal_value(lo_dim, -1):
        return None
    if not _is_literal_value(lo_start, 0):
        return None
    if hi_start != lo_end:
        return None
    return x_hi


def _match_negated_slice_operand(
    operand: GraphOperand,
) -> tuple[GraphOperand, GraphOperand, GraphOperand, GraphOperand] | None:
    left, right = _match_graph_expr_binary(operand, "core.binary.-")
    if not _is_literal_value(left, 0) or right is None:
        return None
    return _match_slice_operand(right)


def _match_slice_operand(
    operand: GraphOperand,
) -> tuple[GraphOperand, GraphOperand, GraphOperand, GraphOperand] | None:
    if not isinstance(operand, GraphExpr):
        return None
    if operand.op.name != "_slice" or len(operand.inputs) < 4:
        return None
    return operand.inputs[0], operand.inputs[1], operand.inputs[2], operand.inputs[3]


def _is_expand_operand(operand: GraphOperand) -> bool:
    return isinstance(operand, GraphExpr) and operand.op.name == "_expand" and bool(operand.inputs)


def _match_graph_expr_binary(
    operand: GraphOperand,
    op_name: str,
) -> tuple[GraphOperand | None, GraphOperand | None]:
    if not isinstance(operand, GraphExpr):
        return None, None
    return _match_graph_binary(operand.op.name, operand.inputs, op_name)


def _match_graph_binary(
    actual_op: str,
    inputs: tuple[GraphOperand, ...],
    op_name: str,
) -> tuple[GraphOperand | None, GraphOperand | None]:
    if actual_op == op_name and len(inputs) == 2:
        return inputs[0], inputs[1]
    return None, None


def _is_literal_value(operand: GraphOperand | None, value: object) -> bool:
    return isinstance(operand, GraphLiteral) and operand.value == value


def _rewrite_backend_sdpa_intrinsics(graph: GraphProgram, *, op_name: str) -> GraphProgram:
    provenance = infer_graph_provenance(graph)
    modules_by_name = {module.name: module for module in graph.modules}
    changed = False
    new_modules: list[GraphModule] = []
    for module in graph.modules:
        new_nodes: list[GraphNode] = []
        for node in module.nodes:
            rewritten = _maybe_rewrite_node_to_backend_sdpa(
                node,
                module=module,
                modules_by_name=modules_by_name,
                provenance=provenance,
                op_name=op_name,
            )
            if rewritten is None:
                new_nodes.append(node)
            else:
                changed = True
                new_nodes.append(rewritten)
        new_modules.append(replace(module, nodes=tuple(new_nodes)))
    return replace(graph, modules=tuple(new_modules)) if changed else graph


def _canonical_specialization_operand(
    operand: GraphOperand,
    *,
    global_symbol_names: set[str],
) -> GraphOperand:
    if (
        isinstance(operand, GraphExpr)
        and operand.op.name in global_symbol_names
        and not operand.inputs
        and not operand.attrs
    ):
        return GraphValueRef(
            name=operand.op.name,
            type_expr=operand.type_expr,
            dims=operand.dims,
        )
    return operand


def _specialization_actual_matches_formal(actual: GraphOperand, formal: GraphValue) -> bool:
    actual_type = graph_operand_type(actual)
    if formal.optional and isinstance(actual_type, TypeNull):
        return True
    return graph_type_compatible(actual_type, formal.type_expr)


def _domain_fact_specialization_operand(
    fact: GraphDomainFact | None,
    formal: GraphValue,
) -> GraphOperand | None:
    if fact is None or fact.kind == GraphDomainKind.UNKNOWN:
        return None
    if fact.kind == GraphDomainKind.NULL:
        candidate: GraphOperand = GraphLiteral(None, TypeNull())
    elif fact.kind == GraphDomainKind.LITERAL:
        candidate = GraphLiteral(fact.value, _literal_fact_type(fact.value, formal.type_expr))
    elif fact.kind == GraphDomainKind.PATH and isinstance(fact.value, GraphPath):
        candidate = fact.value
    else:
        return None
    if _specialization_actual_matches_formal(candidate, formal):
        return candidate
    return None


def _is_total_pure_op(op_name: str, module_effects: Mapping[str, GraphEffect] | None = None) -> bool:
    if module_effects is not None and op_name in module_effects:
        return module_effects[op_name] == GraphEffect.TOTAL_PURE
    return graph_op_effect(op_name) == GraphEffect.TOTAL_PURE


def _is_total_pure_node(
    node: GraphNode,
    module_effects: Mapping[str, GraphEffect] | None = None,
) -> bool:
    return graph_node_effect(node, module_effects=dict(module_effects or {})) == GraphEffect.TOTAL_PURE


def _is_unrestricted_node(
    node: GraphNode,
    module_usages: Mapping[str, UsageClass] | None = None,
) -> bool:
    return graph_node_usage(node, module_usages=dict(module_usages or {})) == UsageClass.UNRESTRICTED


def _is_unrestricted_operand(
    operand: GraphOperand,
    module_usages: Mapping[str, UsageClass] | None = None,
) -> bool:
    return graph_operand_usage(operand, module_usages=dict(module_usages or {})) == UsageClass.UNRESTRICTED


def _is_non_effectful(effect: GraphEffect | None) -> bool:
    return effect in {GraphEffect.TOTAL_PURE, GraphEffect.PARTIAL_PURE}


def _literal_like(value: object, type_like: GraphOperand | GraphNode | GraphExpr) -> GraphLiteral:
    type_expr = getattr(type_like, "type_expr")
    return GraphLiteral(value=value, type_expr=type_expr)


def _bool_literal(value: bool) -> GraphLiteral:
    return GraphLiteral(value=value, type_expr=TypeBool())


def _literal_fact_type(value: object, formal_type: TypeExpr) -> TypeExpr:
    if isinstance(formal_type, TypeOptional):
        inner = formal_type.inner
        if value is not None and graph_type_compatible(inner, formal_type):
            return inner
    if isinstance(value, bool):
        return TypeBool()
    if type(value) is int and isinstance(formal_type, TypeOptional):
        inner = formal_type.inner
        if isinstance(inner, TypeDim | TypeInt):
            return inner
    if isinstance(value, float):
        return TypeFloat()
    if value is None:
        return TypeNull()
    return formal_type


def _validate_optimizer_graph(graph: GraphProgram, *, phase: str) -> None:
    key = _graph_program_validation_key(graph)
    if key in _VALIDATED_OPTIMIZER_GRAPH_KEYS:
        return
    try:
        validate_graph_program(graph)
        modules_by_name = {module.name: module for module in graph.modules}
        for module in graph.modules:
            _validate_optimizer_module_metadata(module, modules_by_name=modules_by_name)
    except ValueError as exc:
        raise ValueError(f"graph optimizer phase {phase!r} produced invalid graph: {exc}") from exc
    _VALIDATED_OPTIMIZER_GRAPH_KEYS.add(key)


_VALIDATED_OPTIMIZER_GRAPH_KEYS: set[object] = set()
_REFRESH_GRAPH_PROGRAM_TYPES_CACHE: dict[object, GraphProgram] = {}


def _graph_type_key(type_expr: TypeExpr | None) -> object:
    return repr(type_expr)


def _hashable_graph_metadata(value: object) -> object:
    if isinstance(value, dict):
        return tuple(
            sorted(
                (key, _hashable_graph_metadata(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, list | tuple):
        return tuple(_hashable_graph_metadata(item) for item in value)
    if isinstance(value, set | frozenset):
        return tuple(sorted(_hashable_graph_metadata(item) for item in value))
    return value


def _graph_dim_key(dim: DimToken) -> object:
    return repr(dim)


def _graph_value_validation_key(value: GraphValue | GraphValueRef) -> object:
    return (
        value.name,
        _graph_type_key(value.type_expr),
        tuple(_graph_dim_key(dim) for dim in value.dims or ()),
        value.optional if isinstance(value, GraphValue) else None,
    )


def _graph_operand_validation_key(operand: GraphOperand) -> object:
    if isinstance(operand, GraphValueRef):
        return ("ref", _graph_value_validation_key(operand))
    if isinstance(operand, GraphLiteral):
        return ("lit", operand.value, _graph_type_key(operand.type_expr))
    if isinstance(operand, GraphPath):
        return ("path", operand.absolute, operand.parts)
    return (
        "expr",
        operand.op.name,
        tuple(_graph_operand_validation_key(item) for item in operand.inputs),
        tuple(
            sorted(
                (key, _graph_operand_validation_key(value))
                for key, value in operand.attrs.items()
            )
        ),
        _graph_type_key(operand.type_expr),
        tuple(_graph_dim_key(dim) for dim in operand.dims or ()),
    )


def _graph_node_validation_key(node: GraphNode) -> object:
    return (
        node.id,
        node.op.name,
        tuple(_graph_operand_validation_key(item) for item in node.inputs),
        tuple(
            sorted(
                (key, _graph_operand_validation_key(value))
                for key, value in node.attrs.items()
            )
        ),
        tuple(_graph_value_validation_key(output) for output in node.outputs),
        node.source_module,
        _graph_type_key(node.type_expr),
        tuple(_graph_dim_key(dim) for dim in node.dims or ()),
    )


def _graph_module_validation_key(module: GraphModule) -> object:
    return (
        module.name,
        tuple(_graph_value_validation_key(value) for value in module.inputs),
        tuple(_graph_operand_validation_key(output) for output in module.outputs),
        module.output_names,
        tuple(_graph_node_validation_key(node) for node in module.nodes),
        _graph_type_key(module.return_type_expr),
        repr(module.constraints),
        module.is_global_binding,
    )


def _graph_program_validation_key(graph: GraphProgram) -> object:
    return (
        graph.main_module,
        tuple(
            sorted(
                (key, _hashable_graph_metadata(value))
                for key, value in graph.pragmas.items()
            )
        ),
        tuple(_graph_module_validation_key(module) for module in graph.modules),
    )


def _validate_optimizer_module_metadata(
    module: GraphModule,
    *,
    modules_by_name: Mapping[str, GraphModule],
) -> None:
    dim_values: dict[str, DimToken] = {}
    local_value_types: dict[str, TypeExpr] = {value.name: value.type_expr for value in module.inputs}
    for node in module.nodes:
        local_value_types.update({value.name: value.type_expr for value in node.outputs})
    for value in module.inputs:
        _require_value_dims_match_type(value, context=f"module {module.name!r} input")
        _validate_type_dim_value_refs(
            value.type_expr,
            local_value_types=local_value_types,
            context=f"module {module.name!r} input {value.name!r}",
        )
    for node in module.nodes:
        if (
            isinstance(node.type_expr, TypeTensor)
            and node.dims is not None
            and not _optimizer_dims_metadata_compatible(node.type_expr.dims, node.dims)
        ):
            raise ValueError(
                f"node {node.id!r} has stale dims metadata: "
                f"type has {node.type_expr.dims!r}, dims has {node.dims!r}"
            )
        _validate_type_dim_value_refs(
            node.type_expr,
            local_value_types=local_value_types,
            context=f"node {node.id!r} type",
        )
        for output in node.outputs:
            _require_value_dims_match_type(output, context=f"node {node.id!r} output")
            _validate_type_dim_value_refs(
                output.type_expr,
                local_value_types=local_value_types,
                context=f"node {node.id!r} output {output.name!r}",
            )
        for operand in (*node.inputs, *node.attrs.values()):
            _validate_optimizer_operand_metadata(
                operand,
                context=f"node {node.id!r} operand",
                local_value_types=local_value_types,
            )
            _validate_optimizer_nested_call_results(
                operand,
                modules_by_name=modules_by_name,
                dim_values=dim_values,
                context=f"node {node.id!r} operand",
            )
        _validate_optimizer_call_result(
            node,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
            context=f"node {node.id!r}",
        )
        if len(node.outputs) == 1 and isinstance(node.outputs[0].type_expr, TypeDim | TypeInt):
            dim = _operand_dim_token(
                GraphExpr(
                    op=node.op,
                    inputs=node.inputs,
                    attrs=node.attrs,
                    type_expr=node.type_expr,
                    dims=node.dims,
                ),
                dim_values,
            )
            if dim is not None:
                dim_values[node.outputs[0].name] = dim
    for output in module.outputs:
        _validate_optimizer_operand_metadata(
            output,
            context=f"module {module.name!r} return",
            local_value_types=local_value_types,
        )
        _validate_optimizer_nested_call_results(
            output,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
            context=f"module {module.name!r} return",
        )
    _validate_optimizer_constraints(module, modules_by_name=modules_by_name)


def _is_dim_value_type(type_expr: TypeExpr) -> bool:
    if isinstance(type_expr, TypeOptional):
        return _is_dim_value_type(type_expr.inner)
    return isinstance(type_expr, TypeDim | TypeInt)


def _validate_type_dim_value_refs(
    type_expr: TypeExpr | None,
    *,
    local_value_types: Mapping[str, TypeExpr],
    context: str,
) -> None:
    for name in _type_dim_refs(type_expr):
        if not isinstance(name, str):
            continue
        value_type = local_value_types.get(name)
        if value_type is None or _is_dim_value_type(value_type):
            continue
        # Some hand-written Axon currently passes shape values as term
        # parameters without precise Dim/Int annotations. Do not reject those
        # legacy imprecise formals here. The dangerous case for graph
        # optimization is generated/local temporaries captured as type-level
        # dimensions, because subsequent refresh passes can then substitute an
        # unrelated tensor/float value name into shape metadata.
        if not _is_generated_value_name(name):
            continue
        raise ValueError(
            f"{context} uses local non-dim value {name!r} as a type dimension "
            f"(local type {value_type!r})"
        )


def _require_value_dims_match_type(value: GraphValue, *, context: str) -> None:
    if (
        isinstance(value.type_expr, TypeTensor)
        and value.dims is not None
        and not _optimizer_dims_metadata_compatible(value.type_expr.dims, value.dims)
    ):
        raise ValueError(
            f"{context} {value.name!r} has stale dims metadata: "
            f"type has {value.type_expr.dims!r}, dims has {value.dims!r}"
        )


def _validate_optimizer_operand_metadata(
    operand: GraphOperand,
    *,
    context: str,
    local_value_types: Mapping[str, TypeExpr],
) -> None:
    if isinstance(operand, GraphValueRef):
        if (
            isinstance(operand.type_expr, TypeTensor)
            and operand.dims is not None
            and not _optimizer_dims_metadata_compatible(operand.type_expr.dims, operand.dims)
        ):
            raise ValueError(
                f"{context} ref {operand.name!r} has stale dims metadata: "
                f"type has {operand.type_expr.dims!r}, dims has {operand.dims!r}"
            )
        _validate_type_dim_value_refs(
            operand.type_expr,
            local_value_types=local_value_types,
            context=f"{context} ref {operand.name!r}",
        )
        return
    if isinstance(operand, GraphExpr):
        if (
            isinstance(operand.type_expr, TypeTensor)
            and operand.dims is not None
            and not _optimizer_dims_metadata_compatible(operand.type_expr.dims, operand.dims)
        ):
            raise ValueError(
                f"{context} expr {operand.op.name!r} has stale dims metadata: "
                f"type has {operand.type_expr.dims!r}, dims has {operand.dims!r}"
            )
        _validate_type_dim_value_refs(
            operand.type_expr,
            local_value_types=local_value_types,
            context=f"{context} expr {operand.op.name!r}",
        )
        for item in operand.inputs:
            _validate_optimizer_operand_metadata(
                item,
                context=f"{context} input",
                local_value_types=local_value_types,
            )
        for key, item in operand.attrs.items():
            _validate_optimizer_operand_metadata(
                item,
                context=f"{context} attr {key!r}",
                local_value_types=local_value_types,
            )


def _validate_optimizer_nested_call_results(
    operand: GraphOperand,
    *,
    modules_by_name: Mapping[str, GraphModule],
    dim_values: Mapping[str, DimToken],
    context: str,
) -> None:
    if not isinstance(operand, GraphExpr):
        return
    _validate_optimizer_call_result(
        operand,
        modules_by_name=modules_by_name,
        dim_values=dim_values,
        context=context,
    )
    for item in operand.inputs:
        _validate_optimizer_nested_call_results(
            item,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
            context=f"{context} input",
        )
    for key, item in operand.attrs.items():
        _validate_optimizer_nested_call_results(
            item,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
            context=f"{context} attr {key!r}",
        )


def _validate_optimizer_call_result(
    call: GraphNode | GraphExpr,
    *,
    modules_by_name: Mapping[str, GraphModule],
    dim_values: Mapping[str, DimToken],
    context: str,
) -> None:
    callee = modules_by_name.get(call.op.name)
    if callee is None:
        return
    actuals = _call_actuals(call, callee)
    if len(actuals) != len(callee.inputs):
        raise ValueError(
            f"{context}: call to {call.op.name!r} has arity {len(actuals)}, "
            f"expected {len(callee.inputs)}"
        )
    expected_types = _instantiate_call_output_types(
        callee,
        actuals,
        len(call.outputs) if isinstance(call, GraphNode) else 1,
        dim_values=dim_values,
    )
    actual_types = (
        tuple(output.type_expr for output in call.outputs)
        if isinstance(call, GraphNode)
        else (call.type_expr,)
    )
    if len(actual_types) != len(expected_types):
        raise ValueError(
            f"{context}: call to {call.op.name!r} result arity {len(actual_types)}, "
            f"expected {len(expected_types)}"
        )
    for index, (actual_type, expected_type) in enumerate(zip(actual_types, expected_types, strict=True)):
        if not graph_type_compatible(actual_type, expected_type):
            raise ValueError(
                f"{context}: call to {call.op.name!r} result {index} has stale type "
                f"{actual_type!r}, expected {expected_type!r}"
            )


def _validate_optimizer_constraints(
    module: GraphModule,
    *,
    modules_by_name: Mapping[str, GraphModule],
) -> None:
    value_names = {value.name for value in module.inputs}
    for node in module.nodes:
        value_names.update(value.name for value in node.outputs)
    dim_symbols = _module_dim_refs(module)
    globals_or_modules = set(modules_by_name)
    allowed = value_names | dim_symbols | globals_or_modules
    for constraint in module.constraints:
        refs = _constraint_ref_names(constraint)
        if _constraint_has_callsite_guard(constraint):
            continue
        unknown = sorted(ref for ref in refs if ref not in allowed)
        if unknown:
            raise ValueError(
                f"module {module.name!r} constraint uses undefined refs: "
                + ", ".join(unknown)
            )


def _sanitize_graph_constraints(graph: GraphProgram) -> GraphProgram:
    modules_by_name = {module.name: module for module in graph.modules}
    return replace(
        graph,
        modules=tuple(
            _sanitize_module_constraints(module, modules_by_name=modules_by_name)
            for module in graph.modules
        ),
    )


def _fresh_type_dim_name(base: str, used: set[str]) -> str:
    candidate = f"{base}__dim"
    if candidate not in used:
        used.add(candidate)
        return candidate
    index = 1
    while True:
        candidate = f"{base}__dim{index}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        index += 1


def _rename_type_dims_in_value(value: GraphValue, renames: Mapping[str, str]) -> GraphValue:
    if not renames:
        return value
    return replace(
        value,
        type_expr=_rename_module_type_expr(value.type_expr, renames),
        dims=(
            None
            if value.dims is None
            else tuple(_rename_module_dim_token(dim, renames) for dim in value.dims)
        ),
    )


def _rename_type_dims_in_operand(operand: GraphOperand, renames: Mapping[str, str]) -> GraphOperand:
    if not renames:
        return operand
    if isinstance(operand, GraphValueRef):
        return replace(
            operand,
            type_expr=_rename_module_type_expr(operand.type_expr, renames),
            dims=(
                None
                if operand.dims is None
                else tuple(_rename_module_dim_token(dim, renames) for dim in operand.dims)
            ),
        )
    if isinstance(operand, GraphLiteral):
        return replace(operand, type_expr=_rename_module_type_expr(operand.type_expr, renames))
    if isinstance(operand, GraphExpr):
        return replace(
            operand,
            inputs=tuple(_rename_type_dims_in_operand(item, renames) for item in operand.inputs),
            attrs={key: _rename_type_dims_in_operand(value, renames) for key, value in operand.attrs.items()},
            type_expr=_rename_module_type_expr(operand.type_expr, renames),
            dims=(
                None
                if operand.dims is None
                else tuple(_rename_module_dim_token(dim, renames) for dim in operand.dims)
            ),
        )
    return operand


def _alpha_rename_shadowed_type_dims_in_module(module: GraphModule) -> GraphModule:
    """Keep term-value binders and type-level dimension symbols lexically distinct.

    Graph IR stores term values and type dimensions as names. If a non-dim term
    value named `K` exists in a module, a type annotation `Tensor[B,K]` must not
    silently resolve to that term value. Rename the type-level symbol instead,
    preserving ordinary term references and path templates.
    """

    local_value_types: dict[str, TypeExpr] = {value.name: value.type_expr for value in module.inputs}
    for node in module.nodes:
        local_value_types.update({value.name: value.type_expr for value in node.outputs})
    shadowing_values = {
        name
        for name, type_expr in local_value_types.items()
        if not _is_dim_value_type(type_expr)
    }
    dim_refs = _module_dim_refs(module)
    conflicts = sorted(name for name in (dim_refs & shadowing_values) if _is_plain_dim_symbol(name))
    if not conflicts:
        return module
    used = set(local_value_types) | dim_refs
    renames: dict[str, str] = {}
    for name in conflicts:
        prefix = f"{name}__dim"
        existing = sorted(
            candidate
            for candidate in dim_refs
            if isinstance(candidate, str)
            and (candidate == prefix or (candidate.startswith(prefix) and candidate[len(prefix):].isdigit()))
            and candidate not in local_value_types
        )
        if existing:
            renames[name] = existing[0]
            continue
        renames[name] = _fresh_type_dim_name(name, used)
    return replace(
        module,
        inputs=tuple(_rename_type_dims_in_value(value, renames) for value in module.inputs),
        outputs=tuple(_rename_type_dims_in_operand(output, renames) for output in module.outputs),
        nodes=tuple(
            replace(
                node,
                inputs=tuple(_rename_type_dims_in_operand(item, renames) for item in node.inputs),
                attrs={key: _rename_type_dims_in_operand(value, renames) for key, value in node.attrs.items()},
                outputs=tuple(_rename_type_dims_in_value(output, renames) for output in node.outputs),
                type_expr=_rename_module_type_expr(node.type_expr, renames),
                dims=(
                    None
                    if node.dims is None
                    else tuple(_rename_module_dim_token(dim, renames) for dim in node.dims)
                ),
            )
            for node in module.nodes
        ),
        return_type_expr=(
            None
            if module.return_type_expr is None
            else _rename_module_type_expr(module.return_type_expr, renames)
        ),
    )


def _alpha_rename_shadowed_type_dims(graph: GraphProgram) -> GraphProgram:
    modules = tuple(_alpha_rename_shadowed_type_dims_in_module(module) for module in graph.modules)
    return graph if modules == graph.modules else replace(graph, modules=modules)


def _sanitize_module_constraints(
    module: GraphModule,
    *,
    modules_by_name: Mapping[str, GraphModule],
) -> GraphModule:
    if not module.constraints:
        return module
    value_names = {value.name for value in module.inputs}
    for node in module.nodes:
        value_names.update(value.name for value in node.outputs)
    allowed = value_names | _module_constraint_dim_symbols(module)
    kept: list[Constraint] = []
    for constraint in module.constraints:
        if _constraint_is_trivially_true(constraint):
            continue
        if _constraint_has_callsite_guard(constraint):
            kept.append(constraint)
            continue
        if _constraint_ref_names(constraint) - allowed:
            continue
        kept.append(constraint)
    if len(kept) == len(module.constraints):
        return module
    return replace(module, constraints=tuple(kept))


def _module_constraint_dim_symbols(module: GraphModule) -> set[str]:
    symbols: set[str] = set()
    for value in module.inputs:
        symbols.update(_type_dim_refs(value.type_expr))
        if value.dims is not None:
            for dim in value.dims:
                symbols.update(dim_token_names(dim))
        if isinstance(value.type_expr, TypeDim):
            symbols.add(value.name)
    symbols.update(_type_dim_refs(module.return_type_expr))
    for output in module.outputs:
        symbols.update(_type_dim_refs(graph_operand_type(output)))
        if isinstance(output, GraphExpr) and output.dims is not None:
            for dim in output.dims:
                symbols.update(dim_token_names(dim))
        if isinstance(output, GraphValueRef):
            symbols.update(dim_token_names(output.name))
    for node in module.nodes:
        symbols.update(_type_dim_refs(node.type_expr))
        if node.dims is not None:
            for dim in node.dims:
                symbols.update(dim_token_names(dim))
        for output in node.outputs:
            symbols.update(_type_dim_refs(output.type_expr))
            if output.dims is not None:
                for dim in output.dims:
                    symbols.update(dim_token_names(dim))
            if isinstance(output.type_expr, TypeDim):
                symbols.add(output.name)
    return symbols


def _optimizer_dims_metadata_compatible(
    type_dims: tuple[DimToken, ...],
    metadata_dims: tuple[DimToken, ...],
) -> bool:
    if type_dims == metadata_dims:
        return True
    if any(isinstance(dim, str) and dim.startswith("..") for dim in (*type_dims, *metadata_dims)):
        return True
    if len(type_dims) != len(metadata_dims):
        return False
    for type_dim, metadata_dim in zip(type_dims, metadata_dims, strict=True):
        if type_dim == metadata_dim:
            continue
        if type(type_dim) is int or type(metadata_dim) is int:
            return False
    return True


def _graph_operand_key(operand: GraphOperand) -> object:
    if isinstance(operand, GraphValueRef):
        return ("ref", operand.name)
    if isinstance(operand, GraphLiteral):
        return ("lit", operand.value, _graph_cse_type_key(operand.type_expr))
    if isinstance(operand, GraphPath):
        return ("path", operand.absolute, operand.parts)
    return (
        "expr",
        operand.op.name,
        tuple(_graph_operand_key(item) for item in operand.inputs),
        tuple(sorted((key, _graph_operand_key(value)) for key, value in operand.attrs.items())),
    )


def _graph_cse_type_key(type_expr: TypeExpr | None) -> object:
    if isinstance(type_expr, TypeDim | TypeInt):
        return "DimOrInt"
    return type_expr


def _graph_node_cse_key(node: GraphNode) -> object:
    return (
        node.op.name,
        tuple(_graph_operand_key(item) for item in node.inputs),
        tuple(sorted((key, _graph_operand_key(value)) for key, value in node.attrs.items())),
        len(node.outputs),
    )


def _result_types(type_expr: TypeExpr, output_count: int) -> tuple[TypeExpr, ...]:
    if output_count == 1:
        return (type_expr,)
    if isinstance(type_expr, TypeTuple) and len(type_expr.items) == output_count:
        return type_expr.items
    if isinstance(type_expr, TypeList):
        return tuple(type_expr.item for _ in range(output_count))
    return tuple(TypeAny() for _ in range(output_count))


def _destructured_list_output_types(
    primitive_type: TypeList,
    output_count: int,
) -> tuple[TypeExpr, ...] | None:
    if output_count <= 1 or isinstance(primitive_type.item, TypeAny):
        return None
    return tuple(primitive_type.item for _ in range(output_count))


def _type_contains_inference_var(type_expr: TypeExpr) -> bool:
    if isinstance(type_expr, TypeAny | TypeVar):
        return True
    if isinstance(type_expr, TypeOptional):
        return _type_contains_inference_var(type_expr.inner)
    if isinstance(type_expr, TypeList):
        return _type_contains_inference_var(type_expr.item)
    if isinstance(type_expr, TypeTuple):
        return any(_type_contains_inference_var(item) for item in type_expr.items)
    return False


def _type_contains_unbound_dim(type_expr: TypeExpr, bound_dim_names: set[str]) -> bool:
    if isinstance(type_expr, TypeTensor):
        return any(
            isinstance(name, str) and (name.startswith("..") or name not in bound_dim_names)
            for dim in type_expr.dims
            for name in dim_token_names(dim)
        )
    if isinstance(type_expr, TypeOptional):
        return _type_contains_unbound_dim(type_expr.inner, bound_dim_names)
    if isinstance(type_expr, TypeList):
        return _type_contains_unbound_dim(type_expr.item, bound_dim_names)
    if isinstance(type_expr, TypeTuple):
        return any(_type_contains_unbound_dim(item, bound_dim_names) for item in type_expr.items)
    return False


def _type_references_any_dim_name(type_expr: TypeExpr | None, names: set[str]) -> bool:
    if type_expr is None or not names:
        return False
    if isinstance(type_expr, TypeTensor):
        return any(
            isinstance(name, str) and name in names
            for dim in type_expr.dims
            for name in dim_token_names(dim)
        )
    if isinstance(type_expr, TypeOptional):
        return _type_references_any_dim_name(type_expr.inner, names)
    if isinstance(type_expr, TypeList):
        return _type_references_any_dim_name(type_expr.item, names)
    if isinstance(type_expr, TypeTuple):
        return any(_type_references_any_dim_name(item, names) for item in type_expr.items)
    return False


def _type_has_non_dim_local_dim_ref(
    type_expr: TypeExpr,
    *,
    env: Mapping[str, GraphValue],
    globals_env: Mapping[str, GraphValue],
) -> bool:
    for name in _type_dim_refs(type_expr):
        value = env.get(name) or globals_env.get(name)
        if value is None:
            # Generated names in type positions should either be live local
            # dim/int values or already substituted away.  If they are no
            # longer bound, they are stale metadata from an earlier rewrite and
            # must not be allowed to override freshly computed types.
            if _is_generated_value_name(name):
                return True
            continue
        value_type = _value_ref_type(value)
        if isinstance(value_type, TypeOptional):
            value_type = value_type.inner
        if not isinstance(value_type, TypeDim | TypeInt):
            return True
    return False


def _refined_compatible_type(current: TypeExpr, desired: TypeExpr) -> TypeExpr:
    if not (graph_type_compatible(current, desired) or graph_type_compatible(desired, current)):
        return current
    if (
        isinstance(current, TypeTensor)
        and isinstance(desired, TypeTensor)
        and len(current.dims) == len(desired.dims)
        and any(isinstance(old, DimExprBinary) and isinstance(new, str) for old, new in zip(current.dims, desired.dims, strict=True))
    ):
        return desired
    if _type_contains_inference_var(current) and not _type_contains_inference_var(desired):
        return desired
    return _more_specific_compatible_type(current, desired)


def _refine_operand_type_metadata(operand: GraphOperand, desired: TypeExpr) -> GraphOperand:
    if not isinstance(operand, GraphValueRef | GraphExpr):
        return operand
    refined = _refined_compatible_type(operand.type_expr, desired)
    if refined == operand.type_expr:
        return operand
    dims = refined.dims if isinstance(refined, TypeTensor) else operand.dims
    return replace(operand, type_expr=refined, dims=dims)


def _refine_select_inputs_from_result(
    inputs: tuple[GraphOperand, ...],
    result_type: TypeExpr,
) -> tuple[GraphOperand, ...]:
    if len(inputs) != 3:
        return inputs
    _, true_operand, false_operand = inputs
    if isinstance(result_type, TypeOptional):
        inner = result_type.inner
        if isinstance(graph_operand_type(false_operand), TypeNull):
            true_operand = _refine_operand_type_metadata(true_operand, inner)
        if isinstance(graph_operand_type(true_operand), TypeNull):
            false_operand = _refine_operand_type_metadata(false_operand, inner)
    return (inputs[0], true_operand, false_operand)


def _strip_optional_operand_type(operand: GraphOperand) -> GraphOperand:
    if not isinstance(operand, GraphValueRef | GraphExpr):
        return operand
    if not isinstance(operand.type_expr, TypeOptional):
        return operand
    inner = operand.type_expr.inner
    return replace(
        operand,
        type_expr=inner,
        dims=inner.dims if isinstance(inner, TypeTensor) else operand.dims,
    )


def _strip_optional_refs_in_operand(
    operand: GraphOperand,
    non_null_names: set[str],
) -> GraphOperand:
    if isinstance(operand, GraphValueRef):
        if operand.name in non_null_names:
            return _strip_optional_operand_type(operand)
        return operand
    if isinstance(operand, GraphExpr):
        return _strip_optional_operand_type(
            replace(
                operand,
                inputs=tuple(_strip_optional_refs_in_operand(item, non_null_names) for item in operand.inputs),
                attrs={
                    key: _strip_optional_refs_in_operand(value, non_null_names)
                    for key, value in operand.attrs.items()
                },
            )
        )
    return operand


def _domain_refined_operand_type(operand: GraphOperand, fact: GraphDomainFact | None) -> GraphOperand:
    if not isinstance(operand, GraphValueRef | GraphExpr) or fact is None:
        return operand
    operand_type = operand.type_expr
    if fact.kind == GraphDomainKind.NULL:
        return replace(operand, type_expr=TypeNull(), dims=operand.dims)
    if fact.kind != GraphDomainKind.NOT_NULL or not isinstance(operand_type, TypeOptional):
        return operand
    inner = operand_type.inner
    return replace(
        operand,
        type_expr=inner,
        dims=inner.dims if isinstance(inner, TypeTensor) else operand.dims,
    )


def _refine_operand_types_from_domain_facts(
    operand: GraphOperand,
    local_facts: Mapping[str, GraphDomainFact],
) -> GraphOperand:
    if isinstance(operand, GraphValueRef):
        return _domain_refined_operand_type(operand, local_facts.get(operand.name))
    if isinstance(operand, GraphExpr):
        rewritten = replace(
            operand,
            inputs=tuple(_refine_operand_types_from_domain_facts(item, local_facts) for item in operand.inputs),
            attrs={
                key: _refine_operand_types_from_domain_facts(value, local_facts)
                for key, value in operand.attrs.items()
            },
        )
        return _domain_refined_operand_type(rewritten, None)
    return operand


def _condition_non_null_names(
    condition: GraphOperand,
    *,
    branch_value: bool,
    conditions: Mapping[str, GraphExpr],
) -> set[str]:
    if isinstance(condition, GraphValueRef):
        condition = conditions.get(condition.name, condition)
    if (
        not isinstance(condition, GraphExpr)
        or not condition.op.name.startswith("core.binary.")
        or len(condition.inputs) != 2
        or condition.attrs
    ):
        return set()
    op = condition.op.name.removeprefix("core.binary.")
    if op not in {"==", "!="}:
        return set()
    equality_branch = branch_value if op == "==" else not branch_value
    left, right = condition.inputs
    names: set[str] = set()
    if isinstance(left, GraphValueRef) and isinstance(right, GraphLiteral) and right.value is None and not equality_branch:
        names.add(left.name)
    if isinstance(right, GraphValueRef) and isinstance(left, GraphLiteral) and left.value is None and not equality_branch:
        names.add(right.name)
    return names


def _refine_select_inputs_from_condition(
    inputs: tuple[GraphOperand, ...],
    *,
    conditions: Mapping[str, GraphExpr],
) -> tuple[GraphOperand, ...]:
    if len(inputs) != 3:
        return inputs
    condition, true_operand, false_operand = inputs
    true_non_null = _condition_non_null_names(condition, branch_value=True, conditions=conditions)
    false_non_null = _condition_non_null_names(condition, branch_value=False, conditions=conditions)
    if true_non_null:
        true_operand = _strip_optional_refs_in_operand(true_operand, true_non_null)
    if false_non_null:
        false_operand = _strip_optional_refs_in_operand(false_operand, false_non_null)
    return (condition, true_operand, false_operand)


def _module_output_types(module: GraphModule) -> tuple[TypeExpr, ...]:
    if module.return_type_expr is not None:
        return _result_types(module.return_type_expr, len(module.outputs))
    return tuple(graph_operand_type(output) for output in module.outputs)


def _module_output_types_for_arity(module: GraphModule, output_count: int) -> tuple[TypeExpr, ...]:
    if module.return_type_expr is not None:
        return _result_types(module.return_type_expr, output_count)
    if output_count == len(module.outputs):
        return tuple(graph_operand_type(output) for output in module.outputs)
    if len(module.outputs) == 1:
        return _result_types(graph_operand_type(module.outputs[0]), output_count)
    return tuple(TypeAny() for _ in range(output_count))


def _return_type_expr_from_outputs(outputs: tuple[GraphOperand, ...]) -> TypeExpr:
    output_types = tuple(graph_operand_type(output) for output in outputs)
    return output_types[0] if len(output_types) == 1 else TypeTuple(output_types)


class _DimUnificationError(ValueError):
    pass


def _is_plain_dim_symbol(dim: DimToken) -> bool:
    return isinstance(dim, str) and dim.isidentifier() and not dim.startswith("..")


def _choose_dim_representative(left: DimToken, right: DimToken) -> DimToken:
    if type(left) is int or type(right) is int:
        return left if type(left) is int else right
    if isinstance(left, str) and _is_plain_dim_symbol(left) and not isinstance(right, str):
        return left
    if isinstance(right, str) and _is_plain_dim_symbol(right) and not isinstance(left, str):
        return right
    left_score = _dim_specificity_score(left)
    right_score = _dim_specificity_score(right)
    if left_score != right_score:
        return left if left_score > right_score else right
    if isinstance(left, str) and isinstance(right, str):
        return left if (len(left), left) <= (len(right), right) else right
    return left


def _dim_substitution_would_cycle(
    name: str,
    replacement: DimToken,
    dim_map: Mapping[str, DimToken],
) -> bool:
    stack = [replacement]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        for symbol in dim_token_names(current):
            if not isinstance(symbol, str):
                continue
            if symbol == name:
                return True
            if symbol in seen:
                continue
            seen.add(symbol)
            mapped = dim_map.get(symbol)
            if mapped is not None:
                stack.append(mapped)
    return False


def _set_dim_binding_if_acyclic(
    dim_map: dict[str, DimToken],
    name: str,
    replacement: DimToken,
) -> None:
    if replacement == name:
        dim_map.pop(name, None)
        return
    if _dim_substitution_would_cycle(name, replacement, dim_map):
        return
    dim_map[name] = replacement


def _unify_dim_binding(
    formal_name: str,
    actual_dim: DimToken,
    dim_map: dict[str, DimToken],
) -> None:
    actual = substitute_dim_token(actual_dim, dim_map)
    existing = dim_map.get(formal_name)
    if existing is None:
        if actual != formal_name:
            _set_dim_binding_if_acyclic(dim_map, formal_name, actual)
        return
    existing = substitute_dim_token(existing, dim_map)
    if existing == actual:
        _set_dim_binding_if_acyclic(dim_map, formal_name, existing)
        return
    if type(existing) is int and type(actual) is int:
        return
    if _is_plain_dim_symbol(existing) or _is_plain_dim_symbol(actual):
        representative = _choose_dim_representative(existing, actual)
        if representative == formal_name:
            dim_map.pop(formal_name, None)
        else:
            _set_dim_binding_if_acyclic(dim_map, formal_name, representative)
        if _is_plain_dim_symbol(existing) and existing != representative:
            _set_dim_binding_if_acyclic(dim_map, existing, representative)
        if _is_plain_dim_symbol(actual) and actual != representative and actual != formal_name:
            _set_dim_binding_if_acyclic(dim_map, actual, representative)
        return
    return


def _bind_dim_sequence_map(
    formal_dims: tuple[DimToken, ...],
    actual_dims: tuple[DimToken, ...],
    dim_map: dict[str, DimToken],
    row_map: dict[str, tuple[DimToken, ...]] | None = None,
    *,
    bind_singleton_plain_dims: bool = True,
) -> None:
    variadic_indexes = [
        index
        for index, dim in enumerate(formal_dims)
        if isinstance(dim, str) and dim.startswith("..")
    ]
    if len(variadic_indexes) > 1:
        return
    pairs: list[tuple[DimToken, DimToken]] = []
    if not variadic_indexes:
        pairs = list(zip(formal_dims, actual_dims, strict=False))
    else:
        variadic_index = variadic_indexes[0]
        prefix = formal_dims[:variadic_index]
        suffix = formal_dims[variadic_index + 1 :]
        if len(actual_dims) < len(prefix) + len(suffix):
            return
        variadic_dim = formal_dims[variadic_index]
        if (
            row_map is not None
            and isinstance(variadic_dim, str)
            and not any(isinstance(dim, str) and dim.startswith("..") for dim in actual_dims)
        ):
            row_end = len(actual_dims) - len(suffix) if suffix else len(actual_dims)
            row_map.setdefault(variadic_dim, actual_dims[len(prefix) : row_end])
        pairs.extend(zip(prefix, actual_dims[: len(prefix)], strict=False))
        if suffix:
            pairs.extend(zip(suffix, actual_dims[-len(suffix) :], strict=False))
    for formal_dim, actual_dim in pairs:
        if isinstance(formal_dim, str) and not formal_dim.startswith(".."):
            if any(isinstance(name, str) and name.startswith("..") for name in dim_token_names(actual_dim)):
                continue
            if (
                not bind_singleton_plain_dims
                and actual_dim == 1
                and _is_plain_dim_symbol(formal_dim)
            ):
                continue
            _unify_dim_binding(formal_dim, actual_dim, dim_map)


def _bind_type_dim_map(
    formal: TypeExpr,
    actual: TypeExpr,
    dim_map: dict[str, DimToken],
    row_map: dict[str, tuple[DimToken, ...]] | None = None,
    *,
    bind_singleton_plain_dims: bool = True,
) -> None:
    if isinstance(formal, TypeTensor) and isinstance(actual, TypeTensor):
        _bind_dim_sequence_map(
            formal.dims,
            actual.dims,
            dim_map,
            row_map=row_map,
            bind_singleton_plain_dims=bind_singleton_plain_dims,
        )
        return
    if isinstance(formal, TypeNamed) and isinstance(actual, TypeNamed):
        _bind_dim_sequence_map(
            formal.args,
            actual.args,
            dim_map,
            bind_singleton_plain_dims=bind_singleton_plain_dims,
        )
        return
    if isinstance(formal, TypeOptional) and isinstance(actual, TypeOptional):
        _bind_type_dim_map(
            formal.inner,
            actual.inner,
            dim_map,
            row_map=row_map,
            bind_singleton_plain_dims=bind_singleton_plain_dims,
        )
        return
    if isinstance(formal, TypeOptional):
        _bind_type_dim_map(
            formal.inner,
            actual,
            dim_map,
            row_map=row_map,
            bind_singleton_plain_dims=bind_singleton_plain_dims,
        )
        return
    if isinstance(actual, TypeOptional):
        _bind_type_dim_map(
            formal,
            actual.inner,
            dim_map,
            row_map=row_map,
            bind_singleton_plain_dims=bind_singleton_plain_dims,
        )
        return
    if isinstance(formal, TypeList) and isinstance(actual, TypeList):
        _bind_type_dim_map(
            formal.item,
            actual.item,
            dim_map,
            row_map=row_map,
            bind_singleton_plain_dims=bind_singleton_plain_dims,
        )
        return
    if isinstance(formal, TypeTuple) and isinstance(actual, TypeTuple):
        for formal_item, actual_item in zip(formal.items, actual.items, strict=False):
            _bind_type_dim_map(
                formal_item,
                actual_item,
                dim_map,
                row_map=row_map,
                bind_singleton_plain_dims=bind_singleton_plain_dims,
            )


def _operand_dim_token(
    operand: GraphOperand,
    dim_values: Mapping[str, DimToken] | None = None,
) -> DimToken | None:
    if isinstance(operand, GraphLiteral) and type(operand.value) is int:
        return operand.value
    if isinstance(operand, GraphValueRef):
        operand_type = operand.type_expr
        if isinstance(operand_type, TypeOptional):
            operand_type = operand_type.inner
        if not isinstance(operand_type, TypeDim | TypeInt):
            return None
        if dim_values is not None and operand.name in dim_values:
            return dim_values[operand.name]
        return operand.name
    if (
        isinstance(operand, GraphExpr)
        and not operand.inputs
        and not operand.attrs
        and isinstance(operand.type_expr, TypeDim | TypeInt)
    ):
        if dim_values is not None and operand.op.name in dim_values:
            return dim_values[operand.op.name]
        return operand.op.name
    if (
        isinstance(operand, GraphExpr)
        and operand.op.name.startswith("core.binary.")
        and len(operand.inputs) == 2
        and isinstance(operand.type_expr, TypeDim | TypeInt)
    ):
        op = operand.op.name.removeprefix("core.binary.")
        if op not in {"+", "-", "*", "/"}:
            return None
        left = _operand_dim_token(operand.inputs[0], dim_values)
        right = _operand_dim_token(operand.inputs[1], dim_values)
        if left is None or right is None:
            return None
        return substitute_dim_token(DimExprBinary(op=op, left=left, right=right), {})
    return None


def _dim_token_operand(dim: DimToken, type_expr: TypeExpr | None = None) -> GraphOperand:
    scalar_type = type_expr if isinstance(type_expr, TypeDim | TypeInt) else TypeDim()
    if type(dim) is int:
        return GraphLiteral(value=dim, type_expr=scalar_type)
    if isinstance(dim, str):
        return GraphValueRef(name=dim, type_expr=scalar_type)
    if isinstance(dim, DimExprBinary):
        return GraphExpr(
            op=GraphOp(f"core.binary.{dim.op}"),
            inputs=(
                _dim_token_operand(dim.left, TypeDim()),
                _dim_token_operand(dim.right, TypeDim()),
            ),
            attrs={},
            type_expr=scalar_type,
        )
    return GraphValueRef(name=str(dim), type_expr=scalar_type)


def _fold_dim_binary_operand(
    op_name: str,
    left: GraphOperand,
    right: GraphOperand,
    *,
    type_expr: TypeExpr,
    dim_values: Mapping[str, DimToken],
) -> GraphOperand | None:
    op = op_name.removeprefix("core.binary.")
    if op not in {"+", "-", "*", "/"} or not isinstance(type_expr, TypeDim | TypeInt):
        return None
    left_dim = _operand_dim_token(left, dim_values)
    right_dim = _operand_dim_token(right, dim_values)
    if left_dim is None or right_dim is None:
        return None
    original = DimExprBinary(op=op, left=left_dim, right=right_dim)
    simplified = substitute_dim_token(original, {})
    if simplified == original:
        return None
    return _dim_token_operand(simplified, type_expr)


def _bind_value_dim_map(
    formal: GraphValue,
    actual: GraphOperand,
    dim_map: dict[str, DimToken],
    *,
    dim_values: Mapping[str, DimToken] | None = None,
) -> None:
    formal_type = formal.type_expr
    if isinstance(formal_type, TypeOptional):
        formal_type = formal_type.inner
    if not isinstance(formal_type, TypeDim):
        return
    actual_dim = _operand_dim_token(actual, dim_values)
    if actual_dim is not None:
        if formal.name not in dim_map:
            _set_dim_binding_if_acyclic(dim_map, formal.name, actual_dim)


def _call_actuals(
    node: GraphNode | GraphExpr,
    callee: GraphModule,
) -> tuple[GraphOperand, ...]:
    actuals: list[GraphOperand | None] = [None] * len(callee.inputs)
    for index, operand in enumerate(node.inputs):
        if index < len(actuals):
            actuals[index] = operand
    formal_by_name = {formal.name: index for index, formal in enumerate(callee.inputs)}
    for key, operand in node.attrs.items():
        index = formal_by_name.get(key)
        if index is not None:
            actuals[index] = operand
    return tuple(actual for actual in actuals if actual is not None)


def _call_dim_subst(
    callee: GraphModule,
    actuals: tuple[GraphOperand, ...],
    *,
    dim_values: Mapping[str, DimToken] | None = None,
) -> dict[str, DimToken]:
    dim_map: dict[str, DimToken] = {}
    for formal, actual in zip(callee.inputs, actuals, strict=False):
        _bind_type_dim_map(
            formal.type_expr,
            graph_operand_type(actual),
            dim_map,
            bind_singleton_plain_dims=False,
        )
        _bind_value_dim_map(formal, actual, dim_map, dim_values=dim_values)
    return dim_map


def _call_node_dim_subst(
    callee: GraphModule,
    node: GraphNode,
    *,
    dim_values: Mapping[str, DimToken] | None = None,
) -> dict[str, DimToken]:
    dim_map = _call_dim_subst(callee, node.inputs, dim_values=dim_values)
    formal_return = _return_type_expr_from_outputs(callee.outputs)
    actual_return = _return_type_expr_from_outputs(
        tuple(
            GraphValueRef(
                name=output.name,
                type_expr=output.type_expr,
                dims=output.dims,
            )
            for output in node.outputs
        )
    )
    _bind_type_dim_map(
        formal_return,
        actual_return,
        dim_map,
        bind_singleton_plain_dims=False,
    )
    return dim_map


def _call_type_substitutions(
    callee: GraphModule,
    actuals: tuple[GraphOperand, ...],
    *,
    dim_values: Mapping[str, DimToken] | None = None,
) -> tuple[dict[str, DimToken], dict[str, tuple[DimToken, ...]]]:
    dim_map: dict[str, DimToken] = {}
    row_map: dict[str, tuple[DimToken, ...]] = {}
    for formal, actual in zip(callee.inputs, actuals, strict=False):
        _bind_type_dim_map(
            formal.type_expr,
            graph_operand_type(actual),
            dim_map,
            row_map=row_map,
        )
        _bind_value_dim_map(formal, actual, dim_map, dim_values=dim_values)
    return dim_map, row_map


def _substitute_type_expr_graph(
    type_expr: TypeExpr,
    *,
    dim_map: Mapping[str, DimToken],
    row_map: Mapping[str, tuple[DimToken, ...]],
) -> TypeExpr:
    if isinstance(type_expr, TypeTensor):
        dims: list[DimToken] = []
        for dim in type_expr.dims:
            if isinstance(dim, str) and dim in row_map:
                dims.extend(row_map[dim])
                continue
            dims.append(substitute_dim_token(dim, dim_map))
        return TypeTensor(base=type_expr.base, dims=tuple(dims))
    if isinstance(type_expr, TypeOptional):
        return TypeOptional(
            _substitute_type_expr_graph(type_expr.inner, dim_map=dim_map, row_map=row_map)
        )
    if isinstance(type_expr, TypeList):
        return TypeList(
            _substitute_type_expr_graph(type_expr.item, dim_map=dim_map, row_map=row_map)
        )
    if isinstance(type_expr, TypeTuple):
        return TypeTuple(
            tuple(
                _substitute_type_expr_graph(item, dim_map=dim_map, row_map=row_map)
                for item in type_expr.items
            )
        )
    return substitute_type_expr(type_expr, dim_map)


def _instantiate_call_output_types(
    callee: GraphModule,
    actuals: tuple[GraphOperand, ...],
    output_count: int,
    *,
    dim_values: Mapping[str, DimToken] | None = None,
) -> tuple[TypeExpr, ...]:
    dim_map, row_map = _call_type_substitutions(
        callee,
        actuals,
        dim_values=dim_values,
    )
    return tuple(
        _substitute_type_expr_graph(type_expr, dim_map=dim_map, row_map=row_map)
        for type_expr in _module_output_types_for_arity(callee, output_count)
    )


def _dim_specificity_score(dim: DimToken) -> int:
    if type(dim) is int:
        return 4
    if isinstance(dim, str):
        return 0 if dim.startswith("..") else 1
    if isinstance(dim, DimExprBinary):
        if any(isinstance(name, str) for name in dim_token_names(dim)):
            return 1
        return _dim_specificity_score(dim.left) + _dim_specificity_score(dim.right)
    return 1


def _type_specificity_score(type_expr: TypeExpr) -> int:
    if isinstance(type_expr, TypeAny):
        return 0
    if isinstance(type_expr, TypeTensor):
        score = 2
        for dim in type_expr.dims:
            score += _dim_specificity_score(dim)
        return score
    if isinstance(type_expr, TypeOptional):
        return 1 + _type_specificity_score(type_expr.inner)
    if isinstance(type_expr, TypeList):
        return 1 + _type_specificity_score(type_expr.item)
    if isinstance(type_expr, TypeTuple):
        return 1 + sum(_type_specificity_score(item) for item in type_expr.items)
    if isinstance(type_expr, TypeNamed):
        return 2 + 2 * len(type_expr.args)
    return 2


def _type_dims(type_expr: TypeExpr) -> tuple[DimToken, ...] | None:
    return type_expr.dims if isinstance(type_expr, TypeTensor) else None


def _graph_operand_dim_token_for_type_rule(
    operand: GraphOperand,
    dim_values: Mapping[str, DimToken] | None,
) -> DimToken | None:
    if isinstance(operand, GraphLiteral):
        if type(operand.value) is int:
            return operand.value
        if isinstance(operand.value, str) and isinstance(operand.type_expr, TypeDim | TypeInt):
            return operand.value
        return None
    if isinstance(operand, GraphValueRef):
        if dim_values is not None and operand.name in dim_values:
            return dim_values[operand.name]
        operand_type = operand.type_expr
        if isinstance(operand_type, TypeOptional):
            operand_type = operand_type.inner
        if isinstance(operand_type, TypeDim | TypeInt):
            return operand.name
        return None
    if isinstance(operand, GraphExpr):
        if operand.op.name in {"core.alias", "core.ascribe"} and len(operand.inputs) == 1:
            return _graph_operand_dim_token_for_type_rule(operand.inputs[0], dim_values)
        if operand.op.name.startswith("core.binary.") and len(operand.inputs) == 2:
            op = operand.op.name.removeprefix("core.binary.")
            if op in {"+", "-", "*", "/"}:
                left = _graph_operand_dim_token_for_type_rule(operand.inputs[0], dim_values)
                right = _graph_operand_dim_token_for_type_rule(operand.inputs[1], dim_values)
                if left is not None and right is not None:
                    return substitute_dim_token(DimExprBinary(op=op, left=left, right=right), {})
        if not operand.inputs and not operand.attrs and isinstance(operand.type_expr, TypeDim | TypeInt):
            return operand.op.name
    return None


def _infer_primitive_graph_type(
    op_name: str,
    inputs: tuple[GraphOperand, ...],
    attrs: Mapping[str, GraphOperand],
    *,
    dim_values: Mapping[str, DimToken] | None,
) -> TypeExpr | None:
    type_rule = get_op_type_rule(op_name[1:] if op_name.startswith("_") else op_name)
    if type_rule is None:
        return None
    inferred = type_rule(
        arg_types=tuple(graph_operand_type(item) for item in inputs),
        kwarg_types={key: graph_operand_type(value) for key, value in attrs.items()},
        args=inputs,
        kwargs=dict(attrs),
        helpers=_PrimitiveTypeHelpers(
            type_dims=_type_dims,
            expr_to_dim_token=lambda value: _graph_operand_dim_token_for_type_rule(value, dim_values)
            if isinstance(value, GraphValueRef | GraphLiteral | GraphPath | GraphExpr)
            else None,
            type_tensor=lambda *, dims: TypeTensor(base="Tensor", dims=tuple(dims)),
            resolve_name_expr=lambda name: GraphValueRef(
                name=name,
                type_expr=TypeDim(),
            )
            if dim_values is not None and name in dim_values
            else None,
            broadcast_tensor_dims=lambda left, right: _broadcast_graph_dims(left, right),
            dim_equivalent=lambda left, right: substitute_dim_token(left, {}) == substitute_dim_token(right, {}),
            unify_dim=_unify_graph_dim_for_type_rule,
        ),
    )
    return inferred if isinstance(inferred, TypeExpr) else None


def _unify_graph_dim_for_type_rule(left: DimToken, right: DimToken) -> DimToken:
    left = substitute_dim_token(left, {})
    right = substitute_dim_token(right, {})
    if left == right:
        return left
    if isinstance(left, int) and isinstance(right, int):
        raise ValueError(f"graph primitive dim mismatch {left!r} vs {right!r}")
    if isinstance(left, str) and not left.startswith(".."):
        return right if not isinstance(right, str) else _choose_dim_representative(left, right)
    if isinstance(right, str) and not right.startswith(".."):
        return left if not isinstance(left, str) else _choose_dim_representative(left, right)
    if isinstance(left, DimExprBinary) and isinstance(right, DimExprBinary) and left.op == right.op:
        return substitute_dim_token(
            DimExprBinary(
                op=left.op,
                left=_unify_graph_dim_for_type_rule(left.left, right.left),
                right=_unify_graph_dim_for_type_rule(left.right, right.right),
            ),
            {},
        )
    raise ValueError(f"graph primitive dim mismatch {left!r} vs {right!r}")


def _broadcast_graph_dim(left: DimToken, right: DimToken) -> DimToken | None:
    left = substitute_dim_token(left, {})
    right = substitute_dim_token(right, {})
    if left == right:
        return left
    if left == 1:
        return right
    if right == 1:
        return left
    if isinstance(left, str) and left.startswith("..") and isinstance(right, str) and right.startswith(".."):
        return None
    if isinstance(left, str) and left.startswith(".."):
        return right
    if isinstance(right, str) and right.startswith(".."):
        return left
    if isinstance(left, str) and not isinstance(right, int):
        return left
    if isinstance(right, str) and not isinstance(left, int):
        return right
    if isinstance(left, int) and isinstance(right, str):
        return left
    if isinstance(right, int) and isinstance(left, str):
        return right
    if isinstance(left, str):
        return left
    if isinstance(right, str):
        return right
    return None


def _broadcast_graph_dims(
    left: tuple[DimToken, ...] | None,
    right: tuple[DimToken, ...] | None,
) -> tuple[DimToken, ...] | None:
    if left is None:
        return right
    if right is None:
        return left
    max_rank = max(len(left), len(right))
    left_full = (1,) * (max_rank - len(left)) + left
    right_full = (1,) * (max_rank - len(right)) + right
    dims: list[DimToken] = []
    for left_dim, right_dim in zip(left_full, right_full, strict=True):
        merged = _broadcast_graph_dim(left_dim, right_dim)
        if merged is None:
            return None
        dims.append(merged)
    return tuple(dims)


def _core_binary_result_type(
    op: str,
    left: TypeExpr,
    right: TypeExpr,
) -> TypeExpr | None:
    left_dims = _type_dims(left)
    right_dims = _type_dims(right)
    if op in {"==", "!=", "<", "<=", ">", ">="}:
        if isinstance(left, TypeNull) or isinstance(right, TypeNull):
            return TypeBool()
        dims = _broadcast_graph_dims(left_dims, right_dims)
        return TypeTensor(base="Tensor", dims=dims) if dims is not None else TypeBool()
    if op not in {"+", "-", "*", "/"}:
        return None
    if left_dims is not None or right_dims is not None:
        dims = _broadcast_graph_dims(left_dims, right_dims)
        if dims is not None:
            return TypeTensor(base="Tensor", dims=dims)
    if isinstance(left, TypeFloat) or isinstance(right, TypeFloat):
        return TypeFloat()
    if isinstance(left, TypeDim) or isinstance(right, TypeDim):
        return TypeDim()
    if isinstance(left, TypeInt) and isinstance(right, TypeInt):
        return TypeInt()
    return None


def _dim_token_uses_any_name(dim: DimToken, names: set[str]) -> bool:
    if isinstance(dim, str):
        return dim in names
    if isinstance(dim, DimExprBinary):
        return any(name in names for name in dim_token_names(dim) if isinstance(name, str))
    return False


def _more_specific_compatible_type(
    existing: TypeExpr,
    refreshed: TypeExpr,
    *,
    preferred_dim_names: set[str] | None = None,
    prefer_refreshed_dim_names: set[str] | None = None,
) -> TypeExpr:
    prefer_refreshed_dim_names = prefer_refreshed_dim_names or set()
    if (
        prefer_refreshed_dim_names
        and graph_type_compatible(existing, refreshed)
        and isinstance(existing, TypeTensor)
        and isinstance(refreshed, TypeTensor)
        and len(existing.dims) == len(refreshed.dims)
    ):
        for existing_dim, refreshed_dim in zip(existing.dims, refreshed.dims, strict=True):
            refreshed_preferred = _dim_token_uses_any_name(refreshed_dim, prefer_refreshed_dim_names)
            existing_preferred = _dim_token_uses_any_name(existing_dim, prefer_refreshed_dim_names)
            if refreshed_preferred and not existing_preferred:
                return refreshed
    if (
        isinstance(existing, TypeTensor)
        and isinstance(refreshed, TypeTensor)
        and len(existing.dims) == len(refreshed.dims)
        and graph_type_compatible(existing, refreshed)
    ):
        if any(old == 1 and new != 1 for old, new in zip(existing.dims, refreshed.dims, strict=True)):
            return refreshed
        if any(
            isinstance(old, str)
            and old.startswith("..")
            and not (isinstance(new, str) and new.startswith(".."))
            for old, new in zip(existing.dims, refreshed.dims, strict=True)
        ):
            return refreshed
    existing_score = _type_specificity_score(existing)
    refreshed_score = _type_specificity_score(refreshed)
    if graph_type_compatible(existing, refreshed) and existing_score > refreshed_score:
        return existing
    preferred_dim_names = preferred_dim_names or set()
    if (
        preferred_dim_names
        and graph_type_compatible(existing, refreshed)
        and existing_score == refreshed_score
        and isinstance(existing, TypeTensor)
        and isinstance(refreshed, TypeTensor)
        and len(existing.dims) == len(refreshed.dims)
    ):
        for existing_dim, refreshed_dim in zip(existing.dims, refreshed.dims, strict=True):
            existing_preferred = _dim_token_uses_any_name(existing_dim, preferred_dim_names)
            refreshed_preferred = _dim_token_uses_any_name(refreshed_dim, preferred_dim_names)
            if existing_preferred and not refreshed_preferred:
                return existing
            if refreshed_preferred and not existing_preferred:
                return refreshed
    return refreshed


def _select_result_type(existing: TypeExpr, true_type: TypeExpr, false_type: TypeExpr) -> TypeExpr:
    del existing
    if isinstance(true_type, TypeNull):
        return TypeOptional(false_type) if not isinstance(false_type, TypeOptional) else false_type
    if isinstance(false_type, TypeNull):
        return TypeOptional(true_type) if not isinstance(true_type, TypeOptional) else true_type
    if isinstance(true_type, TypeOptional) and isinstance(false_type, TypeOptional):
        return TypeOptional(_select_result_type(true_type.inner, true_type.inner, false_type.inner))
    if isinstance(true_type, TypeOptional):
        return TypeOptional(_select_result_type(true_type.inner, true_type.inner, false_type))
    if isinstance(false_type, TypeOptional):
        return TypeOptional(_select_result_type(false_type.inner, true_type, false_type.inner))
    if (
        isinstance(true_type, TypeTensor)
        and isinstance(false_type, TypeTensor)
        and true_type.base == false_type.base
    ):
        if len(true_type.dims) != len(false_type.dims):
            return true_type
        return TypeTensor(
            true_type.base,
            tuple(
                _select_result_dim(left_dim, right_dim)
                for left_dim, right_dim in zip(true_type.dims, false_type.dims, strict=True)
            ),
        )
    if isinstance(true_type, TypeTuple) and isinstance(false_type, TypeTuple):
        if len(true_type.items) != len(false_type.items):
            return true_type
        return TypeTuple(
            tuple(
                _select_result_type(left_item, left_item, right_item)
                for left_item, right_item in zip(true_type.items, false_type.items, strict=True)
            )
        )
    if isinstance(true_type, TypeList) and isinstance(false_type, TypeList):
        return TypeList(_select_result_type(true_type.item, true_type.item, false_type.item))
    if graph_type_compatible(true_type, false_type):
        return _more_specific_compatible_type(true_type, false_type)
    if graph_type_compatible(false_type, true_type):
        return _more_specific_compatible_type(false_type, true_type)
    return true_type


def _stable_join_dim_name(left: DimToken, right: DimToken) -> str:
    payload = repr((substitute_dim_token(left, {}), substitute_dim_token(right, {}))).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:10]
    return f"__join_{digest}"


def _select_result_dim(left: DimToken, right: DimToken) -> DimToken:
    left = substitute_dim_token(left, {})
    right = substitute_dim_token(right, {})
    if left == right:
        return left
    if left == 1:
        return right
    if right == 1:
        return left
    return _stable_join_dim_name(left, right)


def _primitive_semantics(op_name: str) -> dict[str, object]:
    return get_op_semantics(op_name[1:] if op_name.startswith("_") else op_name)


def _primitive_value_dependent_output_types(node: GraphNode) -> tuple[TypeExpr, ...] | None:
    groups = _primitive_semantics(node.op.name).get("value_dependent_output_dim_groups")
    if not isinstance(groups, tuple) or not groups:
        return None
    output_types = [output.type_expr for output in node.outputs]
    changed = False
    for group in groups:
        if not isinstance(group, tuple):
            continue
        positions: list[tuple[int, int]] = []
        for item in group:
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and type(item[0]) is int
                and type(item[1]) is int
            ):
                positions.append(item)
        if not positions:
            continue
        output_value_names = {output.name for output in node.outputs}
        dynamic_dim: DimToken | None = None
        needs_rewrite = False
        for output_index, dim_index in positions:
            if output_index < 0 or output_index >= len(node.outputs):
                continue
            type_expr = node.outputs[output_index].type_expr
            if not isinstance(type_expr, TypeTensor):
                continue
            if dim_index < 0 or dim_index >= len(type_expr.dims):
                continue
            dim = type_expr.dims[dim_index]
            if isinstance(dim, str) and dim.startswith(".."):
                continue
            if isinstance(dim, str):
                if dim not in output_value_names:
                    dynamic_dim = dim
                    break
                needs_rewrite = True
                continue
            needs_rewrite = True
        if dynamic_dim is None and not needs_rewrite:
            continue
        if dynamic_dim is None:
            first_output_index = positions[0][0]
            if first_output_index < 0 or first_output_index >= len(node.outputs):
                continue
            dynamic_dim = f"{node.outputs[first_output_index].name}_dim"
        for output_index, dim_index in positions:
            if output_index < 0 or output_index >= len(output_types):
                continue
            type_expr = output_types[output_index]
            if not isinstance(type_expr, TypeTensor):
                continue
            if dim_index < 0 or dim_index >= len(type_expr.dims):
                continue
            dims = list(type_expr.dims)
            if dims[dim_index] != dynamic_dim:
                dims[dim_index] = dynamic_dim
                output_types[output_index] = replace(type_expr, dims=tuple(dims))
                changed = True
    return tuple(output_types) if changed else None


def _primitive_dim_output_value(
    node: GraphNode,
    *,
    output_index: int,
) -> DimToken | None:
    metadata = _primitive_semantics(node.op.name).get("dim_output_from_tensor_axis")
    if not isinstance(metadata, dict):
        return None
    if metadata.get("output") != output_index:
        return None
    tensor_arg = metadata.get("tensor_arg")
    axis_arg = metadata.get("axis_arg")
    if type(tensor_arg) is not int or type(axis_arg) is not int:
        return None
    if tensor_arg < 0 or axis_arg < 0 or max(tensor_arg, axis_arg) >= len(node.inputs):
        return None
    dims = _type_dims(graph_operand_type(node.inputs[tensor_arg]))
    axis = _literal_int(node.inputs[axis_arg])
    if dims is None or axis is None:
        return None
    resolved = axis if axis >= 0 else len(dims) + axis
    if resolved < 0 or resolved >= len(dims):
        return None
    return dims[resolved]


def _module_input_dim_symbols(module: GraphModule) -> set[str]:
    symbols: set[str] = set()
    for value in module.inputs:
        symbols.update(_type_dim_refs(value.type_expr))
        if isinstance(value.type_expr, TypeDim | TypeInt):
            symbols.add(value.name)
        if isinstance(value.type_expr, TypeOptional) and isinstance(value.type_expr.inner, TypeDim | TypeInt):
            symbols.add(value.name)
    return symbols


def _dim_has_unbound_names(dim: DimToken, bound_names: set[str]) -> bool:
    return any(
        isinstance(name, str) and not name.startswith("..") and name not in bound_names
        for name in dim_token_names(dim)
    )


def _preserve_unbound_output_dims(
    instantiated: TypeExpr,
    candidate: TypeExpr,
    *,
    bound_dim_names: set[str],
) -> TypeExpr:
    if (
        isinstance(instantiated, TypeTensor)
        and isinstance(candidate, TypeTensor)
        and len(instantiated.dims) == len(candidate.dims)
    ):
        dims = tuple(
            instantiated_dim
            if _dim_has_unbound_names(instantiated_dim, bound_dim_names)
            else candidate_dim
            for instantiated_dim, candidate_dim in zip(instantiated.dims, candidate.dims, strict=True)
        )
        return replace(candidate, dims=dims)
    if isinstance(instantiated, TypeOptional) and isinstance(candidate, TypeOptional):
        return TypeOptional(
            _preserve_unbound_output_dims(
                instantiated.inner,
                candidate.inner,
                bound_dim_names=bound_dim_names,
            )
        )
    if isinstance(instantiated, TypeList) and isinstance(candidate, TypeList):
        return TypeList(
            _preserve_unbound_output_dims(
                instantiated.item,
                candidate.item,
                bound_dim_names=bound_dim_names,
            )
        )
    if (
        isinstance(instantiated, TypeTuple)
        and isinstance(candidate, TypeTuple)
        and len(instantiated.items) == len(candidate.items)
    ):
        return TypeTuple(
            tuple(
                _preserve_unbound_output_dims(
                    instantiated_item,
                    candidate_item,
                    bound_dim_names=bound_dim_names,
                )
                for instantiated_item, candidate_item in zip(
                    instantiated.items,
                    candidate.items,
                    strict=True,
                )
            )
        )
    return candidate


def _module_call_result_type(
    existing: TypeExpr,
    instantiated: TypeExpr,
    *,
    bound_dim_names: set[str],
    preferred_dim_names: set[str] | None = None,
) -> TypeExpr:
    if isinstance(instantiated, TypeAny):
        return existing
    if not _type_contains_unbound_dim(instantiated, bound_dim_names):
        return instantiated
    candidate = _more_specific_compatible_type(
        existing,
        instantiated,
        preferred_dim_names=preferred_dim_names,
    )
    return _preserve_unbound_output_dims(
        instantiated,
        candidate,
        bound_dim_names=bound_dim_names,
    )


def _dim_value_symbol_names(dim_values: Mapping[str, DimToken] | None) -> set[str]:
    names = set(dim_values or {})
    for dim in (dim_values or {}).values():
        names.update(name for name in dim_token_names(dim) if isinstance(name, str))
    return names


def _refresh_graph_operand_types(
    operand: GraphOperand,
    *,
    env: Mapping[str, GraphValue],
    globals_env: Mapping[str, GraphValue],
    modules_by_name: Mapping[str, GraphModule],
    dim_values: Mapping[str, DimToken] | None = None,
    local_domain_facts: Mapping[str, GraphDomainFact] | None = None,
    local_conditions: Mapping[str, GraphExpr] | None = None,
) -> GraphOperand:
    if isinstance(operand, GraphValueRef):
        value = env.get(operand.name) or globals_env.get(operand.name)
        if value is None:
            refreshed = operand
        else:
            refreshed = replace(operand, type_expr=_value_ref_type(value), dims=value.dims)
        if local_domain_facts:
            refreshed = _refine_operand_types_from_domain_facts(refreshed, local_domain_facts)
        return refreshed
    if not isinstance(operand, GraphExpr):
        return operand
    if operand.op.name == "core.select" and len(operand.inputs) == 3 and not operand.attrs:
        facts = local_domain_facts or {}
        conditions = local_conditions or {}
        cond = _refresh_graph_operand_types(
            operand.inputs[0],
            env=env,
            globals_env=globals_env,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
            local_domain_facts=facts,
            local_conditions=conditions,
        )
        true_facts = refine_graph_domain_facts_for_branch(cond, True, facts, conditions)
        false_facts = refine_graph_domain_facts_for_branch(cond, False, facts, conditions)
        true_operand = _refresh_graph_operand_types(
            operand.inputs[1],
            env=env,
            globals_env=globals_env,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
            local_domain_facts=true_facts,
            local_conditions=conditions,
        )
        false_operand = _refresh_graph_operand_types(
            operand.inputs[2],
            env=env,
            globals_env=globals_env,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
            local_domain_facts=false_facts,
            local_conditions=conditions,
        )
        inputs = _refine_select_inputs_from_condition((cond, true_operand, false_operand), conditions=conditions)
        result_type = _select_result_type(
            operand.type_expr,
            graph_operand_type(inputs[1]),
            graph_operand_type(inputs[2]),
        )
        inputs = _refine_select_inputs_from_result(inputs, result_type)
        return replace(
            operand,
            inputs=inputs,
            attrs={},
            type_expr=result_type,
            dims=result_type.dims if isinstance(result_type, TypeTensor) else operand.dims,
        )
    inputs = tuple(
        _refresh_graph_operand_types(
            item,
            env=env,
            globals_env=globals_env,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
            local_domain_facts=local_domain_facts,
            local_conditions=local_conditions,
        )
        for item in operand.inputs
    )
    attrs = {
        key: _refresh_graph_operand_types(
            value,
            env=env,
            globals_env=globals_env,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
            local_domain_facts=local_domain_facts,
            local_conditions=local_conditions,
        )
        for key, value in operand.attrs.items()
    }
    refreshed_dim_names = _dim_value_symbol_names(dim_values)
    preferred_dim_names = set(globals_env) | refreshed_dim_names
    callee = modules_by_name.get(operand.op.name)
    if callee is None:
        primitive_type = _infer_primitive_graph_type(
            operand.op.name,
            inputs,
            attrs,
            dim_values=dim_values,
        )
        if primitive_type is not None:
            result_type = operand.type_expr if isinstance(primitive_type, TypeAny) else primitive_type
            return replace(
                operand,
                inputs=inputs,
                attrs=attrs,
                type_expr=result_type,
                dims=result_type.dims if isinstance(result_type, TypeTensor) else operand.dims,
            )
        if operand.op.name.startswith("core.binary.") and len(inputs) == 2:
            op = operand.op.name.removeprefix("core.binary.")
            result_type = _core_binary_result_type(
                op,
                graph_operand_type(inputs[0]),
                graph_operand_type(inputs[1]),
            )
            if result_type is not None:
                return replace(
                    operand,
                    inputs=inputs,
                    attrs=attrs,
                    type_expr=result_type,
                    dims=result_type.dims if isinstance(result_type, TypeTensor) else operand.dims,
                )
        if operand.op.name == "core.select" and len(inputs) == 3:
            result_type = _select_result_type(
                operand.type_expr,
                graph_operand_type(inputs[1]),
                graph_operand_type(inputs[2]),
            )
            inputs = _refine_select_inputs_from_result(inputs, result_type)
            return replace(
                operand,
                inputs=inputs,
                attrs=attrs,
                type_expr=result_type,
                dims=result_type.dims if isinstance(result_type, TypeTensor) else operand.dims,
            )
        return replace(operand, inputs=inputs, attrs=attrs)
    call = replace(operand, inputs=inputs, attrs=attrs)
    result_types = _instantiate_call_output_types(
        callee,
        _call_actuals(call, callee),
        1,
        dim_values=dim_values,
    )
    if len(result_types) != 1:
        return replace(call, type_expr=TypeTuple(result_types))
    existing_type = call.type_expr
    if _type_has_non_dim_local_dim_ref(
        existing_type,
        env=env,
        globals_env=globals_env,
    ):
        existing_type = result_types[0]
    result_type = _module_call_result_type(
        existing_type,
        result_types[0],
        bound_dim_names=_module_input_dim_symbols(callee),
        preferred_dim_names=preferred_dim_names,
    )
    dims = result_type.dims if isinstance(result_type, TypeTensor) else None
    return replace(call, type_expr=result_type, dims=dims)


def _refresh_graph_module_types(
    module: GraphModule,
    *,
    globals_env: Mapping[str, GraphValue],
    modules_by_name: Mapping[str, GraphModule],
    global_dim_values: Mapping[str, DimToken] | None = None,
) -> GraphModule:
    env = {value.name: value for value in module.inputs}
    shadowed_dim_names = _module_signature_dim_refs(module)
    shadowed_global_dim_names = shadowed_dim_names & set(global_dim_values or {})
    dim_values: dict[str, DimToken] = {
        name: value
        for name, value in (global_dim_values or {}).items()
        if name not in shadowed_dim_names
    }
    desired_output_types: dict[str, TypeExpr] = {}
    local_output_names = {
        output.name
        for node in module.nodes
        for output in node.outputs
    }
    for output in module.outputs:
        if isinstance(output, GraphValueRef) and (
            not isinstance(output.type_expr, TypeTensor)
            or
            _type_contains_inference_var(output.type_expr)
            or (
                isinstance(output.type_expr, TypeTensor)
                and any(isinstance(dim, DimExprBinary) for dim in output.type_expr.dims)
            )
        ) and not (_type_dim_refs(output.type_expr) & local_output_names):
            desired_output_types[output.name] = output.type_expr
    for node in module.nodes:
        if node.op.name != "core.repeat":
            continue
        for carry_index, output in enumerate(node.outputs):
            input_index = 3 + carry_index
            if input_index >= len(node.inputs):
                continue
            carry_input = node.inputs[input_index]
            if not isinstance(carry_input, GraphValueRef):
                continue
            current = desired_output_types.get(carry_input.name)
            desired_output_types[carry_input.name] = (
                output.type_expr
                if current is None
                else _refined_compatible_type(current, output.type_expr)
            )
    for _ in range(8):
        changed_desired_types = False
        for node in module.nodes:
            if (
                not node.op.name.startswith("core.binary.")
                or node.op.name.removeprefix("core.binary.") not in {"+", "-", "*", "/"}
                or len(node.inputs) != 2
                or len(node.outputs) != 1
                or node.outputs[0].name not in desired_output_types
            ):
                continue
            desired = desired_output_types[node.outputs[0].name]
            if not isinstance(desired, TypeTensor):
                continue
            for item in node.inputs:
                if not isinstance(item, GraphValueRef):
                    continue
                item_type = graph_operand_type(item)
                if not isinstance(item_type, TypeTensor):
                    continue
                if not (
                    _type_contains_inference_var(item_type)
                    or _type_contains_unbound_dim(item_type, shadowed_dim_names | set(globals_env))
                ):
                    continue
                current = desired_output_types.get(item.name)
                refined = desired if current is None else _refined_compatible_type(current, desired)
                if current != refined:
                    desired_output_types[item.name] = refined
                    changed_desired_types = True
        if not changed_desired_types:
            break
    nodes: list[GraphNode] = []
    conditions: dict[str, GraphExpr] = {}
    for node in module.nodes:
        refreshed_dim_names = _dim_value_symbol_names(dim_values)
        preferred_dim_names = set(globals_env) | refreshed_dim_names
        inputs = tuple(
            _refresh_graph_operand_types(
                item,
                env=env,
                globals_env=globals_env,
                modules_by_name=modules_by_name,
                dim_values=dim_values,
                local_domain_facts={},
                local_conditions=conditions,
            )
            for item in node.inputs
        )
        attrs = {
            key: _refresh_graph_operand_types(
                value,
                env=env,
                globals_env=globals_env,
                modules_by_name=modules_by_name,
                dim_values=dim_values,
                local_domain_facts={},
                local_conditions=conditions,
            )
            for key, value in node.attrs.items()
        }
        type_expr = node.type_expr
        output_types = _result_types(type_expr, len(node.outputs))
        callee = modules_by_name.get(node.op.name)
        if callee is not None:
            call = replace(node, inputs=inputs, attrs=attrs)
            output_types = _instantiate_call_output_types(
                callee,
                _call_actuals(call, callee),
                len(node.outputs),
                dim_values=dim_values,
            )
            output_types = tuple(
                _module_call_result_type(
                    (
                        output_type
                        if _type_has_non_dim_local_dim_ref(
                            node.outputs[index].type_expr,
                            env=env,
                            globals_env=globals_env,
                        )
                        else node.outputs[index].type_expr
                    ),
                    output_type,
                    bound_dim_names=_module_input_dim_symbols(callee),
                    preferred_dim_names=preferred_dim_names,
                )
                if index < len(node.outputs)
                else output_type
                for index, output_type in enumerate(output_types)
            )
            type_expr = output_types[0] if len(output_types) == 1 else TypeTuple(output_types)
        elif node.op.name == "core.tuple":
            output_types = tuple(graph_operand_type(item) for item in inputs)
            type_expr = output_types[0] if len(output_types) == 1 else TypeTuple(output_types)
        elif node.op.name == "core.alias" and len(inputs) == 1:
            input_type = graph_operand_type(inputs[0])
            output_types = _result_types(input_type, len(node.outputs))
            type_expr = output_types[0] if len(output_types) == 1 else TypeTuple(output_types)
        elif node.op.name == "core.ascribe" and len(inputs) == 1:
            output_types = _result_types(type_expr, len(node.outputs))
        elif node.op.name.startswith("core.binary.") and len(inputs) == 2:
            op = node.op.name.removeprefix("core.binary.")
            binary_type = _core_binary_result_type(
                op,
                graph_operand_type(inputs[0]),
                graph_operand_type(inputs[1]),
            )
            if binary_type is not None:
                # The result of a core binary node is determined by its
                # refreshed operands. Existing node metadata can be stale after
                # specialization/inlining, so do not let equally-specific old
                # symbolic names (for example K vs S) override the operand
                # result.
                type_expr = binary_type
                if len(node.outputs) == 1 and node.outputs[0].name in desired_output_types:
                    type_expr = _refined_compatible_type(
                        type_expr,
                        desired_output_types[node.outputs[0].name],
                    )
                if op in {"+", "-", "*", "/"} and isinstance(type_expr, TypeTensor):
                    inputs = tuple(
                        _refine_operand_type_metadata(item, type_expr)
                        if isinstance(graph_operand_type(item), TypeTensor)
                        and (
                            _type_contains_inference_var(graph_operand_type(item))
                            or _type_contains_unbound_dim(
                                graph_operand_type(item),
                                shadowed_dim_names | set(globals_env),
                            )
                        )
                        else item
                        for item in inputs
                    )
                output_types = (type_expr,)
        elif node.op.name == "core.select" and len(inputs) == 3:
            inputs = _refine_select_inputs_from_condition(inputs, conditions=conditions)
            type_expr = _select_result_type(
                type_expr,
                graph_operand_type(inputs[1]),
                graph_operand_type(inputs[2]),
            )
            if len(node.outputs) == 1 and node.outputs[0].name in desired_output_types:
                type_expr = _refined_compatible_type(
                    type_expr,
                    desired_output_types[node.outputs[0].name],
                )
            inputs = _refine_select_inputs_from_result(inputs, type_expr)
            output_types = _result_types(type_expr, len(node.outputs))
        else:
            primitive_type = _infer_primitive_graph_type(
                node.op.name,
                inputs,
                attrs,
                dim_values=dim_values,
            )
            if primitive_type is not None:
                destructured_types = (
                    _destructured_list_output_types(primitive_type, len(node.outputs))
                    if isinstance(primitive_type, TypeList)
                    else None
                )
                if destructured_types is not None:
                    output_types = destructured_types
                    type_expr = TypeTuple(output_types)
                else:
                    type_expr = type_expr if isinstance(primitive_type, TypeAny) else primitive_type
                    output_types = _result_types(type_expr, len(node.outputs))
        value_dependent_output_types = _primitive_value_dependent_output_types(
            replace(
                node,
                inputs=inputs,
                attrs=attrs,
                outputs=tuple(
                    replace(
                        output,
                        type_expr=output_types[index] if index < len(output_types) else output.type_expr,
                        dims=(
                            output_types[index].dims
                            if index < len(output_types) and isinstance(output_types[index], TypeTensor)
                            else output.dims
                        ),
                    )
                    for index, output in enumerate(node.outputs)
                ),
            )
        )
        if value_dependent_output_types is not None:
            output_types = value_dependent_output_types
            type_expr = output_types[0] if len(output_types) == 1 else TypeTuple(output_types)
        outputs = tuple(
            replace(
                output,
                type_expr=(
                    _refined_compatible_type(output_types[index], desired_output_types[output.name])
                    if index < len(output_types) and output.name in desired_output_types
                    else output_types[index]
                    if index < len(output_types)
                    else output.type_expr
                ),
                dims=(
                    (
                        _refined_compatible_type(
                            output_types[index],
                            desired_output_types[output.name],
                        ).dims
                        if output.name in desired_output_types
                        and isinstance(
                            _refined_compatible_type(
                                output_types[index],
                                desired_output_types[output.name],
                            ),
                            TypeTensor,
                        )
                        else output_types[index].dims
                    )
                    if index < len(output_types) and isinstance(output_types[index], TypeTensor)
                    else output.dims
                ),
            )
            for index, output in enumerate(node.outputs)
        )
        type_expr = outputs[0].type_expr if len(outputs) == 1 else TypeTuple(tuple(output.type_expr for output in outputs))
        rewritten = replace(
            node,
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            type_expr=type_expr,
            dims=type_expr.dims if isinstance(type_expr, TypeTensor) else node.dims,
        )
        nodes.append(rewritten)
        env.update({output.name: output for output in outputs})
        if (
            len(outputs) == 1
            and rewritten.op.name.startswith("core.binary.")
            and len(rewritten.inputs) == 2
            and not rewritten.attrs
        ):
            conditions[outputs[0].name] = GraphExpr(
                op=rewritten.op,
                inputs=rewritten.inputs,
                attrs=rewritten.attrs,
                type_expr=rewritten.type_expr,
                dims=rewritten.dims,
            )
        if len(outputs) == 1:
            output = outputs[0]
            if node.op.name in globals_env and isinstance(globals_env[node.op.name].type_expr, TypeDim | TypeInt):
                dim_values[output.name] = node.op.name
            elif node.op.name in {"core.alias", "core.ascribe"} and len(inputs) == 1:
                dim = _operand_dim_token(inputs[0], dim_values)
                if dim is not None:
                    dim_values[output.name] = dim
                elif (
                    isinstance(inputs[0], GraphExpr)
                    and not inputs[0].inputs
                    and not inputs[0].attrs
                    and inputs[0].op.name in globals_env
                    and isinstance(globals_env[inputs[0].op.name].type_expr, TypeDim | TypeInt)
                ):
                    dim_values[output.name] = inputs[0].op.name
            elif node.op.name.startswith("core.binary.") and isinstance(
                output.type_expr,
                TypeDim | TypeInt,
            ):
                dim = _operand_dim_token(
                    GraphExpr(
                        op=rewritten.op,
                        inputs=rewritten.inputs,
                        attrs=rewritten.attrs,
                        type_expr=rewritten.type_expr,
                        dims=rewritten.dims,
                    ),
                    dim_values,
                )
                if dim is not None:
                    dim_values[output.name] = dim
            elif isinstance(output.type_expr, TypeDim | TypeInt):
                dim = _primitive_dim_output_value(rewritten, output_index=0)
                if dim is not None:
                    dim_values[output.name] = dim
    use_types: dict[str, TypeExpr] = {}
    for node in nodes:
        for operand in (*node.inputs, *node.attrs.values()):
            _collect_ref_type_uses(operand, use_types)
    for output in module.outputs:
        _collect_ref_type_uses(output, use_types)
    refined_nodes: list[GraphNode] = []
    for node in nodes:
        refined_outputs: list[GraphValue] = []
        changed_outputs = False
        for output in node.outputs:
            used_type = use_types.get(output.name)
            if used_type is None:
                refined_outputs.append(output)
                continue
            if _type_has_non_dim_local_dim_ref(
                used_type,
                env=env,
                globals_env=globals_env,
            ):
                refined_outputs.append(output)
                continue
            refined_type = _refined_compatible_type(output.type_expr, used_type)
            if refined_type != output.type_expr:
                changed_outputs = True
                refined_outputs.append(
                    replace(
                        output,
                        type_expr=refined_type,
                        dims=refined_type.dims if isinstance(refined_type, TypeTensor) else output.dims,
                    )
                )
            else:
                refined_outputs.append(output)
        if not changed_outputs:
            refined_nodes.append(node)
            continue
        node_type = (
            refined_outputs[0].type_expr
            if len(refined_outputs) == 1
            else TypeTuple(tuple(output.type_expr for output in refined_outputs))
        )
        refined_nodes.append(
            replace(
                node,
                outputs=tuple(refined_outputs),
                type_expr=node_type,
                dims=node_type.dims if isinstance(node_type, TypeTensor) else node.dims,
            )
        )
    nodes = refined_nodes
    outputs = tuple(
        _refresh_graph_operand_types(
            output,
            env=env,
            globals_env=globals_env,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
            local_domain_facts={},
            local_conditions=conditions,
        )
        for output in module.outputs
    )
    return replace(
        module,
        nodes=tuple(nodes),
        outputs=outputs,
        return_type_expr=_return_type_expr_from_outputs(outputs),
    )


def _repeat_attr_value(node: GraphNode, key: str) -> object | None:
    value = node.attrs.get(key)
    return value.value if isinstance(value, GraphLiteral) else None


def _repeat_call_actuals_for_refresh(
    node: GraphNode,
    callee: GraphModule,
) -> tuple[GraphOperand, ...] | None:
    if node.op.name != "core.repeat":
        return None
    arg_count = _repeat_attr_value(node, "arg_count")
    carry_count = _repeat_attr_value(node, "carry_count")
    if type(arg_count) is not int or type(carry_count) is not int:
        return None
    if arg_count != len(callee.inputs) or carry_count != len(node.outputs):
        return None
    actuals: list[GraphOperand] = []
    for index in range(arg_count):
        role = _repeat_attr_value(node, f"arg_{index}")
        if not isinstance(role, str):
            return None
        if role == "iter":
            actuals.append(GraphLiteral(0, TypeInt()))
            continue
        if role.startswith("carry:"):
            carry_index = int(role.removeprefix("carry:"))
            if carry_index < 0 or carry_index >= carry_count:
                return None
            carry_name = _repeat_attr_value(node, f"carry_{carry_index}")
            actuals.append(
                GraphValueRef(
                    name=carry_name if isinstance(carry_name, str) else node.outputs[carry_index].name,
                    type_expr=node.outputs[carry_index].type_expr,
                    dims=node.outputs[carry_index].dims,
                )
            )
            continue
        if role.startswith("input:"):
            input_index = int(role.removeprefix("input:"))
            if input_index < 0 or input_index >= len(node.inputs):
                return None
            actuals.append(node.inputs[input_index])
            continue
        return None
    return tuple(actuals)


def _collect_ref_type_uses(operand: GraphOperand, out: dict[str, TypeExpr]) -> None:
    if isinstance(operand, GraphValueRef):
        current = out.get(operand.name)
        out[operand.name] = (
            operand.type_expr if current is None else _refined_compatible_type(current, operand.type_expr)
        )
        return
    if not isinstance(operand, GraphExpr):
        return
    for item in operand.inputs:
        _collect_ref_type_uses(item, out)
    for item in operand.attrs.values():
        _collect_ref_type_uses(item, out)


def _refine_repeat_callee_signatures(graph: GraphProgram) -> GraphProgram:
    modules_by_name = {module.name: module for module in graph.modules}
    repeat_calls: dict[str, list[tuple[GraphNode, tuple[GraphOperand, ...]]]] = {}
    non_repeat_counts: Counter[str] = Counter()
    module_names = set(modules_by_name)
    for module in graph.modules:
        for node in module.nodes:
            if node.op.name == "core.repeat":
                callee_name = _repeat_attr_value(node, "callee")
                if isinstance(callee_name, str) and callee_name in modules_by_name:
                    actuals = _repeat_call_actuals_for_refresh(node, modules_by_name[callee_name])
                    if actuals is not None:
                        repeat_calls.setdefault(callee_name, []).append((node, actuals))
                continue
            if node.op.name in module_names:
                non_repeat_counts[node.op.name] += 1
            for operand in (*node.inputs, *node.attrs.values()):
                for callee_name in _operand_called_module_names(operand, module_names):
                    non_repeat_counts[callee_name] += 1
        for output in module.outputs:
            for callee_name in _operand_called_module_names(output, module_names):
                non_repeat_counts[callee_name] += 1

    rewritten_modules: list[GraphModule] = []
    changed = False
    for module in graph.modules:
        calls = repeat_calls.get(module.name, [])
        if len(calls) != 1 or non_repeat_counts[module.name]:
            rewritten_modules.append(module)
            continue
        repeat_node, actuals = calls[0]
        inputs: list[GraphValue] = []
        for formal, actual in zip(module.inputs, actuals, strict=True):
            actual_type = graph_operand_type(actual)
            refined_type = _refined_compatible_type(formal.type_expr, actual_type)
            if refined_type != formal.type_expr:
                changed = True
                inputs.append(
                    replace(
                        formal,
                        type_expr=refined_type,
                        dims=refined_type.dims if isinstance(refined_type, TypeTensor) else formal.dims,
                    )
                )
            else:
                inputs.append(formal)
        outputs: list[GraphOperand] = []
        for index, output in enumerate(module.outputs):
            if index >= len(repeat_node.outputs):
                outputs.append(output)
                continue
            desired_type = repeat_node.outputs[index].type_expr
            refined_output = (
                replace(
                    output,
                    type_expr=desired_type,
                    dims=desired_type.dims if isinstance(desired_type, TypeTensor) else output.dims,
                )
                if isinstance(output, GraphValueRef | GraphExpr)
                and (
                    graph_type_compatible(graph_operand_type(output), desired_type)
                    or graph_type_compatible(desired_type, graph_operand_type(output))
                )
                else output
            )
            if refined_output != output:
                changed = True
            outputs.append(refined_output)
        return_type_expr = (
            outputs[0].type_expr
            if len(outputs) == 1 and isinstance(outputs[0], GraphValueRef | GraphExpr)
            else TypeTuple(tuple(graph_operand_type(output) for output in outputs))
        )
        rewritten_modules.append(
            replace(
                module,
                inputs=tuple(inputs),
                outputs=tuple(outputs),
                return_type_expr=return_type_expr,
            )
        )
    return replace(graph, modules=tuple(rewritten_modules)) if changed else graph


def _refresh_graph_program_types(graph: GraphProgram) -> GraphProgram:
    cache_key = _graph_program_validation_key(graph)
    cached = _REFRESH_GRAPH_PROGRAM_TYPES_CACHE.get(cache_key)
    if cached is not None:
        return cached
    current = graph
    for _ in range(16):
        modules_by_name = {module.name: module for module in current.modules}
        global_dim_values = _atomic_int_constant_dims(current)
        globals_env = {
            module.name: GraphValue(
                name=module.name,
                type_expr=_module_output_types(module)[0],
                dims=None,
            )
            for module in current.modules
            if not module.inputs and len(module.outputs) == 1
        }
        refreshed = replace(
            current,
            modules=tuple(
                _refresh_graph_module_types(
                    module,
                    globals_env=globals_env,
                    modules_by_name=modules_by_name,
                    global_dim_values=global_dim_values,
                )
                for module in current.modules
            ),
        )
        refreshed = _refine_repeat_callee_signatures(refreshed)
        if refreshed == current:
            if len(_REFRESH_GRAPH_PROGRAM_TYPES_CACHE) > 128:
                _REFRESH_GRAPH_PROGRAM_TYPES_CACHE.clear()
            _REFRESH_GRAPH_PROGRAM_TYPES_CACHE[cache_key] = current
            return current
        current = refreshed
    raise RuntimeError("graph type refresh did not converge after 16 iterations")


def _refresh_single_graph_module_in_program(graph: GraphProgram, module: GraphModule) -> GraphModule:
    modules_by_name = {item.name: item for item in graph.modules}
    modules_by_name[module.name] = module
    global_dim_values = _atomic_int_constant_dims(graph)
    globals_env = {
        item.name: GraphValue(
            name=item.name,
            type_expr=_module_output_types(module if item.name == module.name else item)[0],
            dims=None,
        )
        for item in graph.modules
        if not item.inputs and len(item.outputs) == 1
    }
    return _refresh_graph_module_types(
        module,
        globals_env=globals_env,
        modules_by_name=modules_by_name,
        global_dim_values=global_dim_values,
    )


def _atomic_literal_constants(graph: GraphProgram) -> dict[str, GraphLiteral]:
    modules_by_name = {module.name: module for module in graph.modules}
    evaluating: set[str] = set()
    memo: dict[str, GraphLiteral] = {}

    def eval_operand(operand: GraphOperand, env: Mapping[str, GraphLiteral]) -> GraphLiteral | None:
        if isinstance(operand, GraphLiteral):
            return operand
        if isinstance(operand, GraphValueRef):
            return env.get(operand.name)
        if not isinstance(operand, GraphExpr):
            return None
        values = tuple(eval_operand(item, env) for item in operand.inputs)
        if any(value is None for value in values):
            return None
        if operand.op.name == "core.alias" and len(values) == 1:
            return values[0]
        if operand.op.name == "core.ascribe" and len(values) == 1:
            return replace(values[0], type_expr=operand.type_expr)
        if operand.op.name == "core.select" and len(values) == 3 and isinstance(values[0].value, bool):
            selected = values[1] if values[0].value else values[2]
            replacement = _select_fold_replacement(selected, operand.type_expr)
            return replacement if isinstance(replacement, GraphLiteral) else None
        if operand.op.name.startswith("core.binary.") and len(values) == 2:
            left, right = values
            return _fold_graph_binary(operand.op.name, left, right, operand)
        if operand.op.name in modules_by_name and not operand.inputs and not operand.attrs:
            return eval_module(operand.op.name)
        return None

    def eval_module(name: str) -> GraphLiteral | None:
        if name in memo:
            return memo[name]
        if name in evaluating:
            return None
        module = modules_by_name.get(name)
        if module is None or module.inputs or len(module.outputs) != 1:
            return None
        evaluating.add(name)
        env: dict[str, GraphLiteral] = {}
        for node in module.nodes:
            if len(node.outputs) != 1 or node.attrs:
                evaluating.remove(name)
                return None
            if node.op.name in modules_by_name and not node.inputs:
                value = eval_module(node.op.name)
            else:
                value = eval_operand(
                    GraphExpr(
                        op=node.op,
                        inputs=node.inputs,
                        attrs=node.attrs,
                        type_expr=node.type_expr,
                        dims=node.dims,
                    ),
                    env,
                )
            if not isinstance(value, GraphLiteral):
                evaluating.remove(name)
                return None
            env[node.outputs[0].name] = value
        value = eval_operand(module.outputs[0], env)
        evaluating.remove(name)
        if isinstance(value, GraphLiteral):
            memo[name] = value
            return value
        return None

    for module in graph.modules:
        value = eval_module(module.name)
        if value is not None:
            memo[module.name] = value
    return memo


def _atomic_int_constant_dims(graph: GraphProgram) -> dict[str, DimToken]:
    return {
        name: literal.value
        for name, literal in _atomic_literal_constants(graph).items()
        if type(literal.value) is int
    }


def _module_dim_refs(module: GraphModule) -> set[str]:
    refs: set[str] = set()
    _module_metadata_refs(module, set(), refs)

    def collect_type(tp: TypeExpr | None) -> None:
        if tp is None:
            return
        if isinstance(tp, TypeTensor):
            for dim in tp.dims:
                refs.update(dim_token_names(dim))
            return
        if isinstance(tp, TypeNamed):
            for dim in tp.args:
                refs.update(dim_token_names(dim))
            return
        if isinstance(tp, TypeOptional):
            collect_type(tp.inner)
            return
        if isinstance(tp, TypeList):
            collect_type(tp.item)
            return
        if isinstance(tp, TypeTuple):
            for item in tp.items:
                collect_type(item)

    for value in module.inputs:
        collect_type(value.type_expr)
        if value.dims is not None:
            for dim in value.dims:
                refs.update(dim_token_names(dim))
    for node in module.nodes:
        collect_type(node.type_expr)
        if node.dims is not None:
            for dim in node.dims:
                refs.update(dim_token_names(dim))
        for value in node.outputs:
            collect_type(value.type_expr)
            if value.dims is not None:
                for dim in value.dims:
                    refs.update(dim_token_names(dim))
        for operand in (*node.inputs, *node.attrs.values()):
            collect_type(graph_operand_type(operand))
    for output in module.outputs:
        collect_type(graph_operand_type(output))
    collect_type(module.return_type_expr)
    return refs


def _module_dim_value_names(module: GraphModule) -> set[str]:
    names = {
        value.name
        for value in module.inputs
        if isinstance(value.type_expr, TypeDim | TypeInt)
    }
    for node in module.nodes:
        names.update(
            value.name
            for value in node.outputs
            if isinstance(value.type_expr, TypeDim | TypeInt)
        )
    return names


def _type_dim_refs(type_expr: TypeExpr | None) -> set[str]:
    if type_expr is None:
        return set()
    if isinstance(type_expr, TypeTensor):
        refs: set[str] = set()
        for dim in type_expr.dims:
            refs.update(dim_token_names(dim))
        return refs
    if isinstance(type_expr, TypeNamed):
        refs: set[str] = set()
        for dim in type_expr.args:
            refs.update(dim_token_names(dim))
        return refs
    if isinstance(type_expr, TypeOptional):
        return _type_dim_refs(type_expr.inner)
    if isinstance(type_expr, TypeList):
        return _type_dim_refs(type_expr.item)
    if isinstance(type_expr, TypeTuple):
        refs: set[str] = set()
        for item in type_expr.items:
            refs.update(_type_dim_refs(item))
        return refs
    return set()


def _dims_dim_refs(dims: tuple[DimToken, ...] | None) -> set[str]:
    refs: set[str] = set()
    for dim in dims or ():
        refs.update(dim_token_names(dim))
    return refs


@dataclass
class _ModuleFreeSymbols:
    value_refs: set[str]
    path_refs: set[str]
    type_dim_refs: set[str]
    term_dim_refs: set[str]
    constraint_refs: set[str]


def _collect_operand_free_symbols(operand: GraphOperand, symbols: _ModuleFreeSymbols) -> None:
    symbols.type_dim_refs.update(_type_dim_refs(graph_operand_type(operand)))
    if isinstance(operand, GraphValueRef):
        symbols.value_refs.add(operand.name)
        if isinstance(operand.type_expr, TypeDim):
            symbols.term_dim_refs.add(operand.name)
        symbols.type_dim_refs.update(_dims_dim_refs(operand.dims))
        return
    if isinstance(operand, GraphPath):
        symbols.path_refs.update(graph_path_template_names(operand))
        return
    if isinstance(operand, GraphExpr):
        symbols.type_dim_refs.update(_dims_dim_refs(operand.dims))
        for item in operand.inputs:
            _collect_operand_free_symbols(item, symbols)
        for item in operand.attrs.values():
            _collect_operand_free_symbols(item, symbols)


def _collect_module_free_symbols(module: GraphModule) -> _ModuleFreeSymbols:
    symbols = _ModuleFreeSymbols(
        value_refs=set(),
        path_refs=set(),
        type_dim_refs=set(),
        term_dim_refs=set(),
        constraint_refs=set(),
    )
    for value in module.inputs:
        symbols.type_dim_refs.update(_type_dim_refs(value.type_expr))
        symbols.type_dim_refs.update(_dims_dim_refs(value.dims))
    for node in module.nodes:
        symbols.type_dim_refs.update(_type_dim_refs(node.type_expr))
        symbols.type_dim_refs.update(_dims_dim_refs(node.dims))
        for operand in (*node.inputs, *node.attrs.values()):
            _collect_operand_free_symbols(operand, symbols)
        for value in node.outputs:
            symbols.type_dim_refs.update(_type_dim_refs(value.type_expr))
            symbols.type_dim_refs.update(_dims_dim_refs(value.dims))
    for output in module.outputs:
        _collect_operand_free_symbols(output, symbols)
    symbols.type_dim_refs.update(_type_dim_refs(module.return_type_expr))
    for constraint in module.constraints:
        symbols.constraint_refs.update(_constraint_ref_names(constraint))
    return symbols


def _module_signature_dim_refs(module: GraphModule) -> set[str]:
    refs: set[str] = set()
    for value in module.inputs:
        refs.update(_type_dim_refs(value.type_expr))
        refs.update(_dims_dim_refs(value.dims))
    if module.return_type_expr is not None:
        refs.update(_type_dim_refs(module.return_type_expr))
    else:
        for output in module.outputs:
            refs.update(_type_dim_refs(graph_operand_type(output)))
            dims = getattr(output, "dims", None)
            refs.update(_dims_dim_refs(dims))
    return refs


def _module_return_dim_refs(module: GraphModule) -> set[str]:
    if module.return_type_expr is not None:
        return _type_dim_refs(module.return_type_expr)
    refs: set[str] = set()
    for output in module.outputs:
        refs.update(_type_dim_refs(graph_operand_type(output)))
        refs.update(_dims_dim_refs(getattr(output, "dims", None)))
    return refs


def _specialized_module_render_closure_safe(
    module: GraphModule,
    *,
    global_symbol_names: set[str],
) -> bool:
    input_names = {value.name for value in module.inputs}
    local_names = {
        value.name
        for node in module.nodes
        for value in node.outputs
    }
    signature_dim_refs = _module_signature_dim_refs(module)
    return_dim_refs = _module_return_dim_refs(module)
    symbols = _collect_module_free_symbols(module)

    value_bound = input_names | local_names | global_symbol_names
    dim_bound = signature_dim_refs | global_symbol_names
    if (symbols.value_refs | symbols.path_refs) - value_bound - dim_bound:
        return False
    if symbols.constraint_refs - value_bound - dim_bound:
        return False

    term_dim_bound = input_names | signature_dim_refs | return_dim_refs | global_symbol_names
    return not (symbols.term_dim_refs - term_dim_bound)


def _replace_atomic_literal_globals(
    operand: GraphOperand,
    constants: Mapping[str, GraphLiteral],
) -> GraphOperand:
    if isinstance(operand, GraphValueRef):
        return constants.get(operand.name, operand)
    if not isinstance(operand, GraphExpr):
        return operand
    if operand.op.name in constants and not operand.inputs and not operand.attrs:
        return constants[operand.op.name]
    return replace(
        operand,
        inputs=tuple(_replace_atomic_literal_globals(item, constants) for item in operand.inputs),
        attrs={
            key: _replace_atomic_literal_globals(value, constants)
            for key, value in operand.attrs.items()
        },
    )


def _substitute_atomic_constant_dims_local(graph: GraphProgram) -> GraphProgram:
    literal_constants = _atomic_literal_constants(graph)
    if not literal_constants:
        return graph
    dim_subst = {
        name: literal.value
        for name, literal in literal_constants.items()
        if type(literal.value) is int
    }
    modules: list[GraphModule] = []
    for module in graph.modules:
        module_constants = {
            name: literal
            for name, literal in literal_constants.items()
            if name != module.name
        }
        candidate_module = substitute_graph_module_dims(module, dim_subst)
        candidate_module = replace(
            candidate_module,
            nodes=tuple(
                replace(
                    node,
                    inputs=tuple(
                        _replace_atomic_literal_globals(item, module_constants)
                        for item in node.inputs
                    ),
                    attrs={
                        key: _replace_atomic_literal_globals(value, module_constants)
                        for key, value in node.attrs.items()
                    },
                )
                for node in candidate_module.nodes
            ),
            outputs=tuple(
                _replace_atomic_literal_globals(output, module_constants)
                for output in candidate_module.outputs
            ),
        )
        modules.append(candidate_module)
    pre_refresh_modules = tuple(modules)
    candidate = replace(graph, modules=pre_refresh_modules)
    if candidate == graph:
        return graph
    candidate = _refresh_graph_program_types(candidate)
    candidate = replace(
        candidate,
        modules=_preserve_unchanged_module_types(
            graph.modules,
            pre_refresh_modules,
            candidate.modules,
            main_module=graph.main_module,
        ),
    )
    candidate = _sanitize_graph_constraints(candidate)
    try:
        _validate_optimizer_graph(candidate, phase="constant_dim_substitution")
    except ValueError:
        accepted = list(graph.modules)
        changed = False
        for index, (original_module, candidate_module) in enumerate(
            zip(graph.modules, modules, strict=True)
        ):
            if candidate_module == original_module:
                continue
            candidate_modules = list(accepted)
            candidate_modules[index] = candidate_module
            pre_refresh_candidate_modules = tuple(candidate_modules)
            module_candidate = replace(graph, modules=tuple(candidate_modules))
            module_candidate = _refresh_graph_program_types(module_candidate)
            module_candidate = replace(
                module_candidate,
                modules=_preserve_unchanged_module_types(
                    graph.modules,
                    pre_refresh_candidate_modules,
                    module_candidate.modules,
                    main_module=graph.main_module,
                ),
            )
            module_candidate = _sanitize_graph_constraints(module_candidate)
            try:
                _validate_optimizer_graph(
                    module_candidate,
                    phase="constant_dim_substitution.module",
                )
            except ValueError:
                continue
            accepted = list(module_candidate.modules)
            changed = True
        if not changed:
            return graph
        return replace(graph, modules=tuple(accepted))
    return candidate


def _preserve_unchanged_module_types(
    original_modules: tuple[GraphModule, ...],
    pre_refresh_modules: tuple[GraphModule, ...],
    refreshed_modules: tuple[GraphModule, ...],
    *,
    main_module: str,
) -> tuple[GraphModule, ...]:
    return tuple(
        replace(
            refreshed_module,
            inputs=original_module.inputs,
            outputs=original_module.outputs,
            return_type_expr=original_module.return_type_expr,
        )
        if (
            original_module.name != main_module
            and
            pre_refresh_module.inputs == original_module.inputs
            and pre_refresh_module.outputs == original_module.outputs
            and pre_refresh_module.nodes == original_module.nodes
            and pre_refresh_module.return_type_expr == original_module.return_type_expr
            and refreshed_module != original_module
        )
        else refreshed_module
        for original_module, pre_refresh_module, refreshed_module in zip(
            original_modules,
            pre_refresh_modules,
            refreshed_modules,
            strict=True,
        )
    )


def _simplify_symbolic_graph_dims(graph: GraphProgram) -> GraphProgram:
    candidate = replace(
        graph,
        modules=tuple(substitute_graph_module_dims(module, {}) for module in graph.modules),
    )
    candidate = _sanitize_graph_constraints(candidate)
    _validate_optimizer_graph(candidate, phase="symbolic_dim_simplification")
    return candidate


def _fold_graph_binary(op_name: str, left: GraphLiteral, right: GraphLiteral, template: GraphExpr | GraphNode) -> GraphLiteral | None:
    op = op_name.removeprefix("core.binary.")
    lval = left.value
    rval = right.value
    if isinstance(lval, bool) and isinstance(rval, bool):
        if op == "and":
            return _bool_literal(lval and rval)
        if op == "or":
            return _bool_literal(lval or rval)
        if op == "==":
            return _bool_literal(lval == rval)
        if op == "!=":
            return _bool_literal(lval != rval)
    if isinstance(lval, str) and isinstance(rval, str):
        if op == "==":
            return _bool_literal(lval == rval)
        if op == "!=":
            return _bool_literal(lval != rval)
    if type(lval) is int and type(rval) is int:
        if op == "+":
            return _literal_like(lval + rval, template)
        if op == "-":
            return _literal_like(lval - rval, template)
        if op == "*":
            return _literal_like(lval * rval, template)
        if op == "/" and rval != 0 and lval % rval == 0:
            return _literal_like(lval // rval, template)
        if op == "==":
            return _bool_literal(lval == rval)
        if op == "!=":
            return _bool_literal(lval != rval)
        if op == "<":
            return _bool_literal(lval < rval)
        if op == "<=":
            return _bool_literal(lval <= rval)
        if op == ">":
            return _bool_literal(lval > rval)
        if op == ">=":
            return _bool_literal(lval >= rval)
    if isinstance(lval, float) and isinstance(rval, float):
        if op == "+":
            return _literal_like(lval + rval, template)
        if op == "-":
            return _literal_like(lval - rval, template)
        if op == "*":
            return _literal_like(lval * rval, template)
        if op == "/" and rval != 0.0:
            return _literal_like(lval / rval, template)
        if op == "==":
            return _bool_literal(lval == rval)
        if op == "!=":
            return _bool_literal(lval != rval)
        if op == "<":
            return _bool_literal(lval < rval)
        if op == "<=":
            return _bool_literal(lval <= rval)
        if op == ">":
            return _bool_literal(lval > rval)
        if op == ">=":
            return _bool_literal(lval >= rval)
    if lval is None and rval is None:
        if op == "==":
            return _bool_literal(True)
        if op == "!=":
            return _bool_literal(False)
    if (lval is None) != (rval is None):
        if op == "==":
            return _bool_literal(False)
        if op == "!=":
            return _bool_literal(True)
    return None


def _is_null_literal(operand: GraphOperand) -> bool:
    return isinstance(operand, GraphLiteral) and operand.value is None


def _operand_is_statically_non_null_for_fold(operand: GraphOperand) -> bool:
    if isinstance(operand, GraphPath):
        return True
    if isinstance(operand, GraphLiteral):
        return operand.value is not None
    return False


def _fold_typed_null_comparison(op_name: str, left: GraphOperand, right: GraphOperand) -> GraphLiteral | None:
    op = op_name.removeprefix("core.binary.")
    if op not in {"==", "!="}:
        return None
    if _is_null_literal(left) and _operand_is_statically_non_null_for_fold(right):
        return _bool_literal(op == "!=")
    if _is_null_literal(right) and _operand_is_statically_non_null_for_fold(left):
        return _bool_literal(op == "!=")
    return None


def _operand_domain_fact(
    operand: GraphOperand,
    local_domain_facts: Mapping[str, GraphDomainFact] | None,
) -> GraphDomainFact | None:
    if isinstance(operand, GraphLiteral):
        if operand.value is None:
            return GraphDomainFact(GraphDomainKind.NULL)
        if isinstance(operand.value, bool | int | float | str):
            return GraphDomainFact(GraphDomainKind.LITERAL, operand.value)
        return None
    if isinstance(operand, GraphPath):
        return GraphDomainFact(GraphDomainKind.PATH, operand)
    if isinstance(operand, GraphValueRef) and local_domain_facts is not None:
        return local_domain_facts.get(operand.name)
    return None


def _domain_facts_equal(left: GraphDomainFact, right: GraphDomainFact) -> bool | None:
    if left.kind == GraphDomainKind.UNKNOWN or right.kind == GraphDomainKind.UNKNOWN:
        return None
    if left.kind != right.kind:
        if {left.kind, right.kind} == {GraphDomainKind.NULL, GraphDomainKind.NOT_NULL}:
            return False
        if left.kind == GraphDomainKind.NULL and right.kind in {
            GraphDomainKind.LITERAL,
            GraphDomainKind.PATH,
            GraphDomainKind.GLOBAL_VALUE,
        }:
            return False
        if right.kind == GraphDomainKind.NULL and left.kind in {
            GraphDomainKind.LITERAL,
            GraphDomainKind.PATH,
            GraphDomainKind.GLOBAL_VALUE,
        }:
            return False
        return None
    return left.value == right.value


def _fold_domain_binary_comparison(
    op_name: str,
    left: GraphOperand,
    right: GraphOperand,
    *,
    local_domain_facts: Mapping[str, GraphDomainFact] | None,
) -> GraphLiteral | None:
    op = op_name.removeprefix("core.binary.")
    if op not in {"==", "!="}:
        return None
    left_fact = _operand_domain_fact(left, local_domain_facts)
    right_fact = _operand_domain_fact(right, local_domain_facts)
    if left_fact is None or right_fact is None:
        return None
    equality = _domain_facts_equal(left_fact, right_fact)
    if equality is None:
        return None
    return _bool_literal(equality if op == "==" else not equality)


def _domain_bool_value(
    operand: GraphOperand,
    local_domain_facts: Mapping[str, GraphDomainFact] | None,
) -> bool | None:
    if isinstance(operand, GraphLiteral) and type(operand.value) is bool:
        return operand.value
    fact = _operand_domain_fact(operand, local_domain_facts)
    if (
        fact is not None
        and fact.kind == GraphDomainKind.LITERAL
        and isinstance(fact.value, bool)
    ):
        return fact.value
    return None


def _select_fold_replacement(selected: GraphOperand, result_type: TypeExpr) -> GraphOperand | None:
    selected_type = graph_operand_type(selected)
    if selected_type == result_type:
        return selected
    if graph_type_compatible(selected_type, result_type):
        return selected
    if graph_type_compatible(result_type, selected_type):
        if isinstance(selected, GraphValueRef):
            return replace(selected, type_expr=result_type, dims=_type_dims(result_type))
        if isinstance(selected, GraphExpr):
            return replace(selected, type_expr=result_type, dims=_type_dims(result_type))
    return None


def _fold_numeric_primitive(expr: GraphExpr | GraphNode) -> GraphLiteral | None:
    if expr.attrs:
        return None
    if not all(isinstance(item, GraphLiteral) for item in expr.inputs):
        return None
    values = tuple(item.value for item in expr.inputs if isinstance(item, GraphLiteral))
    op_name = expr.op.name[1:] if expr.op.name.startswith("_") else expr.op.name
    if op_name == "sqrt" and len(values) == 1 and type(values[0]) in {int, float} and values[0] >= 0:
        return GraphLiteral(value=math.sqrt(values[0]), type_expr=expr.type_expr)
    return None


def _dim_token_to_operand(dim: DimToken) -> GraphOperand:
    if type(dim) is int:
        return GraphLiteral(value=dim, type_expr=TypeDim())
    if isinstance(dim, str):
        return GraphValueRef(name=dim, type_expr=TypeDim())
    if isinstance(dim, DimExprBinary):
        return GraphExpr(
            op=GraphOp(f"core.binary.{dim.op}"),
            inputs=(_dim_token_to_operand(dim.left), _dim_token_to_operand(dim.right)),
            attrs={},
            type_expr=TypeDim(),
        )
    return GraphValueRef(name=str(dim), type_expr=TypeDim())


def _typed_shape_operands(operand: GraphOperand) -> tuple[GraphOperand, ...] | None:
    type_expr = graph_operand_type(operand)
    if not isinstance(type_expr, TypeTensor):
        return None
    if any(isinstance(dim, str) and dim.startswith("..") for dim in type_expr.dims):
        return None
    return tuple(_dim_token_to_operand(dim) for dim in type_expr.dims)


def _literal_int(operand: GraphOperand) -> int | None:
    if isinstance(operand, GraphLiteral) and type(operand.value) is int:
        return operand.value
    return None


def _indexed_operand(items: tuple[GraphOperand, ...], index: int) -> GraphOperand | None:
    resolved = index if index >= 0 else len(items) + index
    if resolved < 0 or resolved >= len(items):
        return None
    return items[resolved]


def _shape_items_operand(
    operand: GraphOperand,
    *,
    modules_by_name: Mapping[str, GraphModule] | None,
) -> tuple[GraphOperand, ...] | None:
    if isinstance(operand, GraphExpr) and operand.op.name == "core.list":
        return operand.inputs
    if isinstance(operand, GraphExpr) and operand.op.name == "_shape" and len(operand.inputs) == 1:
        return _typed_shape_operands(operand.inputs[0])
    return None


def _shape_index_forwarder(module: GraphModule) -> tuple[int, int] | None:
    if module.is_global_binding:
        return None
    if len(module.nodes) != 2 or len(module.outputs) != 1:
        return None
    shape_node, index_node = module.nodes
    if (
        shape_node.op.name != "_shape"
        or index_node.op.name != "_list_index"
        or shape_node.attrs
        or index_node.attrs
        or len(shape_node.inputs) != 1
        or len(shape_node.outputs) != 1
        or len(index_node.inputs) != 2
        or len(index_node.outputs) != 1
    ):
        return None
    shape_input = shape_node.inputs[0]
    if not isinstance(shape_input, GraphValueRef):
        return None
    indexed_shape, index_input = index_node.inputs
    if not isinstance(indexed_shape, GraphValueRef):
        return None
    if indexed_shape.name != shape_node.outputs[0].name:
        return None
    if not isinstance(index_input, GraphValueRef):
        return None
    returned = module.outputs[0]
    if not isinstance(returned, GraphValueRef) or returned.name != index_node.outputs[0].name:
        return None
    formal_indexes = {formal.name: index for index, formal in enumerate(module.inputs)}
    tensor_index = formal_indexes.get(shape_input.name)
    dim_index = formal_indexes.get(index_input.name)
    if tensor_index is None or dim_index is None:
        return None
    return tensor_index, dim_index


def _tensor_size_forwarder(module: GraphModule) -> tuple[int, int] | None:
    if module.is_global_binding:
        return None
    if len(module.nodes) != 1 or len(module.outputs) != 1:
        return None
    node = module.nodes[0]
    if node.op.name != "_tensor_size" or node.attrs or len(node.inputs) != 2 or len(node.outputs) != 1:
        return None
    tensor_input, dim_input = node.inputs
    if not isinstance(tensor_input, GraphValueRef) or not isinstance(dim_input, GraphValueRef):
        return None
    returned = module.outputs[0]
    if not isinstance(returned, GraphValueRef) or returned.name != node.outputs[0].name:
        return None
    formal_indexes = {formal.name: index for index, formal in enumerate(module.inputs)}
    tensor_index = formal_indexes.get(tensor_input.name)
    dim_index = formal_indexes.get(dim_input.name)
    if tensor_index is None or dim_index is None:
        return None
    return tensor_index, dim_index


def _shape_query_replacement(
    expr: GraphExpr,
    *,
    modules_by_name: Mapping[str, GraphModule] | None,
    stable_shape_values: set[str] | None = None,
    blocked_dim_ref_names: set[str] | None = None,
) -> GraphOperand | None:
    stable_shape_values = stable_shape_values or set()
    blocked_dim_ref_names = blocked_dim_ref_names or set()

    def allowed(candidate: GraphOperand | None) -> GraphOperand | None:
        if (
            isinstance(candidate, GraphValueRef)
            and candidate.name in blocked_dim_ref_names
            and isinstance(candidate.type_expr, TypeDim | TypeInt)
        ):
            return None
        return candidate

    if expr.op.name == "_shape" and len(expr.inputs) == 1:
        if not (
            isinstance(expr.inputs[0], GraphValueRef)
            and expr.inputs[0].name in stable_shape_values
        ):
            return None
        items = _typed_shape_operands(expr.inputs[0])
        if items is None:
            return None
        if any(
            isinstance(item, GraphValueRef)
            and item.name in blocked_dim_ref_names
            and isinstance(item.type_expr, TypeDim | TypeInt)
            for item in items
        ):
            return None
        return GraphExpr(
            op=GraphOp("core.list"),
            inputs=items,
            attrs={},
            type_expr=TypeList(TypeDim()),
        )
    if expr.op.name == "_list_index" and len(expr.inputs) == 2:
        items = _shape_items_operand(expr.inputs[0], modules_by_name=modules_by_name)
        index = _literal_int(expr.inputs[1])
        if items is None or index is None:
            return None
        return allowed(_indexed_operand(items, index))
    if expr.op.name == "_tensor_size" and len(expr.inputs) == 2:
        if not (
            isinstance(expr.inputs[0], GraphValueRef)
            and expr.inputs[0].name in stable_shape_values
        ):
            return None
        items = _typed_shape_operands(expr.inputs[0])
        index = _literal_int(expr.inputs[1])
        if items is None or index is None:
            return None
        return allowed(_indexed_operand(items, index))
    if modules_by_name is None:
        return None
    callee = modules_by_name.get(expr.op.name)
    if callee is None:
        return None
    forwarder = _tensor_size_forwarder(callee)
    if forwarder is not None:
        tensor_index, dim_index = forwarder
        actuals = _call_actuals(expr, callee)
        if len(actuals) <= max(tensor_index, dim_index):
            return None
        if not (
            isinstance(actuals[tensor_index], GraphValueRef)
            and actuals[tensor_index].name in stable_shape_values
        ):
            return None
        items = _typed_shape_operands(actuals[tensor_index])
        index = _literal_int(actuals[dim_index])
        if items is None or index is None:
            return None
        return allowed(_indexed_operand(items, index))
    forwarder = _shape_index_forwarder(callee)
    if forwarder is None:
        return None
    tensor_index, dim_index = forwarder
    actuals = _call_actuals(expr, callee)
    if len(actuals) <= max(tensor_index, dim_index):
        return None
    if not (
        isinstance(actuals[tensor_index], GraphValueRef)
        and actuals[tensor_index].name in stable_shape_values
    ):
        return None
    items = _typed_shape_operands(actuals[tensor_index])
    index = _literal_int(actuals[dim_index])
    if items is None or index is None:
        return None
    return allowed(_indexed_operand(items, index))


def _operand_refs(operand: GraphOperand, out: set[str]) -> None:
    if isinstance(operand, GraphValueRef):
        out.add(operand.name)
        return
    if isinstance(operand, GraphPath):
        out.update(graph_path_template_names(operand))
        return
    if isinstance(operand, GraphExpr):
        for item in operand.inputs:
            _operand_refs(item, out)
        for item in operand.attrs.values():
            _operand_refs(item, out)


def _operand_module_calls(operand: GraphOperand, module_names: set[str], out: set[str]) -> None:
    if isinstance(operand, GraphValueRef):
        if operand.name in module_names:
            out.add(operand.name)
        return
    if isinstance(operand, GraphPath):
        out.update(name for name in graph_path_template_names(operand) if name in module_names)
        return
    if not isinstance(operand, GraphExpr):
        return
    if operand.op.name in module_names:
        out.add(operand.op.name)
    for item in operand.inputs:
        _operand_module_calls(item, module_names, out)
    for item in operand.attrs.values():
        _operand_module_calls(item, module_names, out)


def _operand_called_module_names(operand: GraphOperand, module_names: set[str]) -> set[str]:
    calls: set[str] = set()
    _operand_module_calls(operand, module_names, calls)
    return calls


def _find_operand_call(operand: GraphOperand, callee_name: str) -> GraphExpr | None:
    if not isinstance(operand, GraphExpr):
        return None
    if operand.op.name == callee_name:
        return operand
    for item in operand.inputs:
        found = _find_operand_call(item, callee_name)
        if found is not None:
            return found
    for item in operand.attrs.values():
        found = _find_operand_call(item, callee_name)
        if found is not None:
            return found
    return None


def _rename_module_dim_token(dim: DimToken, renames: Mapping[str, str]) -> DimToken:
    if isinstance(dim, str):
        return renames.get(dim, dim)
    if isinstance(dim, DimExprBinary):
        return DimExprBinary(
            op=dim.op,
            left=_rename_module_dim_token(dim.left, renames),
            right=_rename_module_dim_token(dim.right, renames),
        )
    return dim


def _rename_module_type_expr(type_expr: TypeExpr, renames: Mapping[str, str]) -> TypeExpr:
    if isinstance(type_expr, TypeTensor):
        return TypeTensor(
            base=type_expr.base,
            dims=tuple(_rename_module_dim_token(dim, renames) for dim in type_expr.dims),
        )
    if isinstance(type_expr, TypeNamed):
        return TypeNamed(
            name=type_expr.name,
            args=tuple(_rename_module_dim_token(dim, renames) for dim in type_expr.args),
        )
    if isinstance(type_expr, TypeOptional):
        return TypeOptional(_rename_module_type_expr(type_expr.inner, renames))
    if isinstance(type_expr, TypeList):
        return TypeList(_rename_module_type_expr(type_expr.item, renames))
    if isinstance(type_expr, TypeTuple):
        return TypeTuple(tuple(_rename_module_type_expr(item, renames) for item in type_expr.items))
    return type_expr


def _rename_module_value(value: GraphValue, renames: Mapping[str, str]) -> GraphValue:
    return replace(
        value,
        type_expr=_rename_module_type_expr(value.type_expr, renames),
        dims=(
            None
            if value.dims is None
            else tuple(_rename_module_dim_token(dim, renames) for dim in value.dims)
        ),
    )


def _rename_module_operand(operand: GraphOperand, renames: Mapping[str, str]) -> GraphOperand:
    if isinstance(operand, GraphValueRef):
        return replace(
            operand,
            name=renames.get(operand.name, operand.name),
            type_expr=_rename_module_type_expr(operand.type_expr, renames),
            dims=(
                None
                if operand.dims is None
                else tuple(_rename_module_dim_token(dim, renames) for dim in operand.dims)
            ),
        )
    if isinstance(operand, GraphLiteral):
        return replace(operand, type_expr=_rename_module_type_expr(operand.type_expr, renames))
    if isinstance(operand, GraphExpr):
        return replace(
            operand,
            op=GraphOp(renames.get(operand.op.name, operand.op.name)),
            inputs=tuple(_rename_module_operand(item, renames) for item in operand.inputs),
            attrs={key: _rename_module_operand(value, renames) for key, value in operand.attrs.items()},
            type_expr=_rename_module_type_expr(operand.type_expr, renames),
            dims=(
                None
                if operand.dims is None
                else tuple(_rename_module_dim_token(dim, renames) for dim in operand.dims)
            ),
        )
    return operand


def _rename_module_constraint_operand(
    operand: ConstraintOperand,
    renames: Mapping[str, str],
) -> ConstraintOperand:
    if isinstance(operand, tuple):
        return tuple(_rename_module_constraint_operand(item, renames) for item in operand)
    if isinstance(operand, str):
        return renames.get(operand, operand)
    if isinstance(operand, DimExprBinary):
        return _rename_module_dim_token(operand, renames)
    return operand


def _rename_module_constraint(constraint: Constraint, renames: Mapping[str, str]) -> Constraint:
    return Constraint(
        relation=constraint.relation,
        left=_rename_module_constraint_operand(constraint.left, renames),
        right=(
            None
            if constraint.right is None
            else _rename_module_constraint_operand(constraint.right, renames)
        ),
        guards=tuple(_rename_module_constraint(guard, renames) for guard in constraint.guards),
    )


def _rename_modules(graph: GraphProgram, renames: Mapping[str, str]) -> GraphProgram:
    if not renames:
        return graph
    modules: list[GraphModule] = []
    for module in graph.modules:
        modules.append(
            replace(
                module,
                name=renames.get(module.name, module.name),
                inputs=tuple(_rename_module_value(value, renames) for value in module.inputs),
                outputs=tuple(_rename_module_operand(output, renames) for output in module.outputs),
                nodes=tuple(
                    replace(
                        node,
                        op=GraphOp(renames.get(node.op.name, node.op.name)),
                        inputs=tuple(_rename_module_operand(item, renames) for item in node.inputs),
                        attrs={key: _rename_module_operand(value, renames) for key, value in node.attrs.items()},
                        outputs=tuple(_rename_module_value(output, renames) for output in node.outputs),
                        type_expr=_rename_module_type_expr(node.type_expr, renames),
                        dims=(
                            None
                            if node.dims is None
                            else tuple(_rename_module_dim_token(dim, renames) for dim in node.dims)
                        ),
                    )
                    for node in module.nodes
                ),
                return_type_expr=(
                    None
                    if module.return_type_expr is None
                    else _rename_module_type_expr(module.return_type_expr, renames)
                ),
                constraints=tuple(_rename_module_constraint(item, renames) for item in module.constraints),
            )
        )
    return replace(
        graph,
        modules=tuple(modules),
        main_module=renames.get(graph.main_module, graph.main_module),
    )


def _specialized_module_base(name: str) -> str | None:
    marker = "__spec_"
    if marker not in name:
        return None
    base, suffix = name.rsplit(marker, 1)
    if not base or not suffix.isdigit():
        return None
    return base


def _canonicalize_generated_module_names(graph: GraphProgram) -> GraphProgram:
    names = {module.name for module in graph.modules}
    generated_by_base: dict[str, list[str]] = {}
    for module in graph.modules:
        if module.name == graph.main_module or module.is_global_binding:
            continue
        base = _specialized_module_base(module.name)
        if base is None:
            continue
        generated_by_base.setdefault(base, []).append(module.name)
    renames: dict[str, str] = {}
    reserved = set(names)
    for base, generated_names in sorted(generated_by_base.items()):
        for index, generated_name in enumerate(sorted(generated_names), start=1):
            if base not in names and base not in renames.values() and generated_name == generated_names[0]:
                target = base
            else:
                suffix = 1
                while True:
                    target = f"{base}__s{suffix}"
                    if target not in reserved and target not in renames.values():
                        break
                    suffix += 1
            if target == generated_name:
                continue
            reserved.add(target)
            renames[generated_name] = target
    if not renames:
        return graph
    renamed = _rename_modules(graph, renames)
    _validate_optimizer_graph(renamed, phase="canonicalize_module_names")
    return renamed


def _is_generated_value_name(name: str) -> bool:
    return (
        name.startswith("__")
        or "__inl_" in name
        or "___" in name
        or (name.startswith("_v") and "_arg" in name)
        or (name.startswith("_v") and name[2:].isdigit())
    )


def _fresh_canonical_value_name(used: set[str], next_index: int) -> tuple[str, int]:
    index = next_index
    while True:
        candidate = f"_v{index}"
        if candidate not in used:
            used.add(candidate)
            return candidate, index + 1
        index += 1


def _rename_value(
    value: GraphValue,
    renames: Mapping[str, str],
    *,
    type_renames: Mapping[str, str] | None = None,
) -> GraphValue:
    type_renames = renames if type_renames is None else type_renames
    return replace(
        value,
        name=renames.get(value.name, value.name),
        type_expr=_rename_module_type_expr(value.type_expr, type_renames),
        dims=(
            None
            if value.dims is None
            else tuple(_rename_module_dim_token(dim, type_renames) for dim in value.dims)
        ),
    )


def _rename_value_operand(
    operand: GraphOperand,
    renames: Mapping[str, str],
    *,
    type_renames: Mapping[str, str] | None = None,
) -> GraphOperand:
    type_renames = renames if type_renames is None else type_renames
    if isinstance(operand, GraphValueRef):
        return replace(
            operand,
            name=renames.get(operand.name, operand.name),
            type_expr=_rename_module_type_expr(operand.type_expr, type_renames),
            dims=(
                None
                if operand.dims is None
                else tuple(_rename_module_dim_token(dim, type_renames) for dim in operand.dims)
            ),
        )
    if isinstance(operand, GraphLiteral):
        return replace(operand, type_expr=_rename_module_type_expr(operand.type_expr, type_renames))
    if isinstance(operand, GraphPath):
        return rename_operand(operand, renames)
    if isinstance(operand, GraphExpr):
        return replace(
            operand,
            inputs=tuple(
                _rename_value_operand(item, renames, type_renames=type_renames)
                for item in operand.inputs
            ),
            attrs={
                key: _rename_value_operand(value, renames, type_renames=type_renames)
                for key, value in operand.attrs.items()
            },
            type_expr=_rename_module_type_expr(operand.type_expr, type_renames),
            dims=(
                None
                if operand.dims is None
                else tuple(_rename_module_dim_token(dim, type_renames) for dim in operand.dims)
            ),
        )
    return operand


def _rename_value_constraint_operand(
    operand: ConstraintOperand,
    renames: Mapping[str, str],
) -> ConstraintOperand:
    if isinstance(operand, tuple):
        return tuple(_rename_value_constraint_operand(item, renames) for item in operand)
    if isinstance(operand, str):
        return renames.get(operand, operand)
    if isinstance(operand, DimExprBinary):
        return _rename_module_dim_token(operand, renames)
    return operand


def _rename_value_constraint(constraint: Constraint, renames: Mapping[str, str]) -> Constraint:
    return Constraint(
        relation=constraint.relation,
        left=_rename_value_constraint_operand(constraint.left, renames),
        right=(
            None
            if constraint.right is None
            else _rename_value_constraint_operand(constraint.right, renames)
        ),
        guards=tuple(_rename_value_constraint(guard, renames) for guard in constraint.guards),
    )


def _is_zero_arg_global_call_node(
    node: GraphNode,
    *,
    global_symbol_names: set[str],
) -> bool:
    return (
        node.op.name in global_symbol_names
        and not node.inputs
        and not node.attrs
        and len(node.outputs) == 1
    )


def _fresh_hidden_global_value_name(used: set[str], next_index: int) -> tuple[str, int]:
    index = next_index
    while True:
        candidate = f"__global_{index}"
        if candidate not in used:
            used.add(candidate)
            return candidate, index + 1
        index += 1


def _collect_dim_value_ref_names(operand: GraphOperand, out: set[str]) -> None:
    if isinstance(operand, GraphValueRef):
        if isinstance(operand.type_expr, TypeDim | TypeInt):
            out.add(operand.name)
        return
    if isinstance(operand, GraphPath):
        out.update(graph_path_template_names(operand))
        return
    if isinstance(operand, GraphExpr):
        for item in operand.inputs:
            _collect_dim_value_ref_names(item, out)
        for item in operand.attrs.values():
            _collect_dim_value_ref_names(item, out)


def _module_unbound_dim_value_ref_names(module: GraphModule) -> set[str]:
    defined = {value.name for value in module.inputs}
    for node in module.nodes:
        defined.update(value.name for value in node.outputs)
    refs: set[str] = set()
    for node in module.nodes:
        for operand in node.inputs:
            _collect_dim_value_ref_names(operand, refs)
        for operand in node.attrs.values():
            _collect_dim_value_ref_names(operand, refs)
    for output in module.outputs:
        _collect_dim_value_ref_names(output, refs)
    return refs - defined


def _is_canonical_hidden_global_value_name(name: str) -> bool:
    return name.startswith("__global_") and name.removeprefix("__global_").isdigit()


def _is_canonical_local_value_name(name: str) -> bool:
    return name.startswith("_v") and name[2:].isdigit()


def _canonicalize_generated_value_names_in_module(
    module: GraphModule,
    *,
    global_symbol_names: set[str],
) -> GraphModule:
    renames: dict[str, str] = {}
    # Dim symbols are first-class in graph IR. A value name must not capture a
    # type-dim symbol or an intentionally unbound dim term reference.
    used = {value.name for value in module.inputs} | _module_unbound_dim_value_ref_names(module)
    for name in module.output_names:
        if not _is_generated_value_name(name):
            used.add(name)
    for node in module.nodes:
        for output in node.outputs:
            if not _is_generated_value_name(output.name):
                used.add(output.name)
    next_hidden_global_index = 1
    for node in module.nodes:
        if not _is_zero_arg_global_call_node(node, global_symbol_names=global_symbol_names):
            continue
        output = node.outputs[0]
        if _is_canonical_hidden_global_value_name(output.name) and output.name not in used:
            used.add(output.name)
            continue
        target, next_hidden_global_index = _fresh_hidden_global_value_name(used, next_hidden_global_index)
        if target != output.name:
            renames[output.name] = target
    next_index = 1
    for node in module.nodes:
        if _is_zero_arg_global_call_node(node, global_symbol_names=global_symbol_names):
            continue
        for output in node.outputs:
            if not _is_generated_value_name(output.name):
                continue
            if _is_canonical_local_value_name(output.name) and output.name not in used:
                used.add(output.name)
                continue
            target, next_index = _fresh_canonical_value_name(used, next_index)
            if target != output.name:
                renames[output.name] = target
    if not renames:
        return module
    local_value_types = {value.name: value.type_expr for value in module.inputs}
    for node in module.nodes:
        local_value_types.update({value.name: value.type_expr for value in node.outputs})
    type_renames = {
        old: new
        for old, new in renames.items()
        if _is_dim_value_type(local_value_types.get(old, TypeAny()))
        or old in _module_dim_refs(module)
    }
    return replace(
        module,
        inputs=tuple(_rename_value(value, renames, type_renames=type_renames) for value in module.inputs),
        outputs=tuple(
            _rename_value_operand(output, renames, type_renames=type_renames)
            for output in module.outputs
        ),
        output_names=tuple(renames.get(name, name) for name in module.output_names),
        nodes=tuple(
            replace(
                node,
                inputs=tuple(
                    _rename_value_operand(item, renames, type_renames=type_renames)
                    for item in node.inputs
                ),
                attrs={
                    key: _rename_value_operand(value, renames, type_renames=type_renames)
                    for key, value in node.attrs.items()
                },
                outputs=tuple(
                    _rename_value(output, renames, type_renames=type_renames)
                    for output in node.outputs
                ),
                type_expr=_rename_module_type_expr(node.type_expr, type_renames),
                dims=(
                    None
                    if node.dims is None
                    else tuple(_rename_module_dim_token(dim, type_renames) for dim in node.dims)
                ),
            )
            for node in module.nodes
        ),
        return_type_expr=(
            None
            if module.return_type_expr is None
            else _rename_module_type_expr(module.return_type_expr, type_renames)
        ),
        constraints=tuple(_rename_value_constraint(item, renames) for item in module.constraints),
    )


def _canonicalize_generated_value_names(graph: GraphProgram) -> GraphProgram:
    global_symbol_names = {
        module.name
        for module in graph.modules
        if module.is_global_binding and not module.inputs and len(module.outputs) == 1
    }
    modules = tuple(
        _canonicalize_generated_value_names_in_module(
            module,
            global_symbol_names=global_symbol_names,
        )
        for module in graph.modules
    )
    if modules == graph.modules:
        return graph
    renamed = _alpha_rename_shadowed_type_dims(replace(graph, modules=modules))
    try:
        _validate_optimizer_graph(renamed, phase="canonicalize_value_names")
    except ValueError:
        # Value-name canonicalization is a readability-only transform. If a
        # generated value name is also used as a type-level dimension symbol,
        # renaming can create accidental term/type capture in rare stale-type
        # cases. Keep the already-valid graph instead of changing semantics.
        return graph
    return renamed


def _type_module_refs(type_expr: TypeExpr | None, module_names: set[str], out: set[str]) -> None:
    if type_expr is None:
        return
    if isinstance(type_expr, TypeTensor):
        for dim in type_expr.dims:
            out.update(name for name in dim_token_names(dim) if name in module_names)
        return
    if isinstance(type_expr, TypeNamed):
        for dim in type_expr.args:
            out.update(name for name in dim_token_names(dim) if name in module_names)
        return
    if isinstance(type_expr, TypeOptional):
        _type_module_refs(type_expr.inner, module_names, out)
        return
    if isinstance(type_expr, TypeList):
        _type_module_refs(type_expr.item, module_names, out)
        return
    if isinstance(type_expr, TypeTuple):
        for item in type_expr.items:
            _type_module_refs(item, module_names, out)


def _dim_module_refs(dim: DimToken, module_names: set[str], out: set[str]) -> None:
    out.update(name for name in dim_token_names(dim) if name in module_names)


def _constraint_atom_module_refs(
    atom: ConstraintAtom,
    module_names: set[str],
    out: set[str],
) -> None:
    if isinstance(atom, str):
        if atom in module_names:
            out.add(atom)
        return
    if isinstance(atom, DimExprBinary):
        _dim_module_refs(atom, module_names, out)


def _constraint_operand_module_refs(
    operand: ConstraintOperand,
    module_names: set[str],
    out: set[str],
) -> None:
    if isinstance(operand, tuple):
        for item in operand:
            _constraint_atom_module_refs(item, module_names, out)
        return
    _constraint_atom_module_refs(operand, module_names, out)


def _constraint_module_refs(
    constraint: Constraint,
    module_names: set[str],
    out: set[str],
) -> None:
    _constraint_operand_module_refs(constraint.left, module_names, out)
    if constraint.right is not None:
        _constraint_operand_module_refs(constraint.right, module_names, out)
    for guard in constraint.guards:
        _constraint_module_refs(guard, module_names, out)


def _module_metadata_refs(module: GraphModule, module_names: set[str], out: set[str]) -> None:
    for value in module.inputs:
        _type_module_refs(value.type_expr, module_names, out)
        if value.dims is not None:
            for dim in value.dims:
                _dim_module_refs(dim, module_names, out)
    for node in module.nodes:
        _type_module_refs(node.type_expr, module_names, out)
        if node.dims is not None:
            for dim in node.dims:
                _dim_module_refs(dim, module_names, out)
        for value in node.outputs:
            _type_module_refs(value.type_expr, module_names, out)
            if value.dims is not None:
                for dim in value.dims:
                    _dim_module_refs(dim, module_names, out)
        for operand in (*node.inputs, *node.attrs.values()):
            _type_module_refs(graph_operand_type(operand), module_names, out)
    for output in module.outputs:
        _type_module_refs(graph_operand_type(output), module_names, out)
    _type_module_refs(module.return_type_expr, module_names, out)
    for constraint in module.constraints:
        _constraint_module_refs(constraint, module_names, out)


def _count_operand_module_calls(
    operand: GraphOperand,
    module_names: set[str],
    counts: Counter[str],
) -> None:
    if not isinstance(operand, GraphExpr):
        return
    if operand.op.name in module_names:
        counts[operand.op.name] += 1
    for item in operand.inputs:
        _count_operand_module_calls(item, module_names, counts)
    for item in operand.attrs.values():
        _count_operand_module_calls(item, module_names, counts)


def _replace_operand_refs(
    operand: GraphOperand,
    subst: Mapping[str, GraphOperand],
    *,
    fold: bool = True,
    module_effects: Mapping[str, GraphEffect] | None = None,
    modules_by_name: Mapping[str, GraphModule] | None = None,
    local_domain_facts: Mapping[str, GraphDomainFact] | None = None,
) -> GraphOperand:
    rewritten = replace_operand_refs(
        operand,
        subst,
        fold_operand=(
            None
            if not fold
            else lambda item: _fold_operand(
                item,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                local_domain_facts=local_domain_facts,
            )
        ),
    )
    dim_subst = _dim_alias_subst(subst)
    return substitute_graph_operand_dims(rewritten, dim_subst) if dim_subst else rewritten


def _dim_alias_subst(subst: Mapping[str, GraphOperand]) -> dict[str, DimToken]:
    dim_subst: dict[str, DimToken] = {}
    for name, replacement in subst.items():
        replacement_dim = _operand_dim_token(replacement)
        if replacement_dim is not None:
            dim_subst[name] = replacement_dim
        elif isinstance(replacement, GraphValueRef):
            # Alias cleanup can replace a local value before all type metadata
            # that refers to it has been revisited.  If that old value name
            # appears in a type-level dimension position, it must follow the
            # same alias as the term-level reference.
            dim_subst[name] = replacement.name
    return dim_subst


def _fold_operand(
    operand: GraphOperand,
    *,
    module_effects: Mapping[str, GraphEffect] | None = None,
    modules_by_name: Mapping[str, GraphModule] | None = None,
    local_domain_facts: Mapping[str, GraphDomainFact] | None = None,
) -> GraphOperand:
    if not isinstance(operand, GraphExpr):
        return operand
    expr = replace(
        operand,
        inputs=tuple(
            _fold_operand(
                item,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                local_domain_facts=local_domain_facts,
            )
            for item in operand.inputs
        ),
        attrs={
            key: _fold_operand(
                value,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                local_domain_facts=local_domain_facts,
            )
            for key, value in operand.attrs.items()
        },
    )
    shape_replacement = _shape_query_replacement(
        expr,
        modules_by_name=modules_by_name,
        stable_shape_values=set(),
    )
    if shape_replacement is not None:
        return shape_replacement
    if expr.op.name == "core.ascribe" and len(expr.inputs) == 1:
        return expr.inputs[0]
    if expr.op.name == "core.select" and len(expr.inputs) == 3:
        cond_value = _domain_bool_value(expr.inputs[0], local_domain_facts)
        if cond_value is not None:
            selected = expr.inputs[1] if cond_value else expr.inputs[2]
            replacement = _select_fold_replacement(selected, expr.type_expr)
            if replacement is not None:
                return replacement
    if expr.op.name.startswith("core.binary.") and len(expr.inputs) == 2:
        left, right = expr.inputs
        domain_fold = _fold_domain_binary_comparison(
            expr.op.name,
            left,
            right,
            local_domain_facts=local_domain_facts,
        )
        if domain_fold is not None:
            return domain_fold
        typed_null_fold = _fold_typed_null_comparison(expr.op.name, left, right)
        if typed_null_fold is not None:
            return typed_null_fold
        if isinstance(left, GraphLiteral) and isinstance(right, GraphLiteral):
            folded = _fold_graph_binary(expr.op.name, left, right, expr)
            if folded is not None:
                return folded
    if _is_total_pure_op(expr.op.name, module_effects) and (folded := _fold_numeric_primitive(expr)) is not None:
        return folded
    return expr


def _rewrite_node_operands(
    node: GraphNode,
    subst: Mapping[str, GraphOperand],
    *,
    fold: bool = True,
    module_effects: Mapping[str, GraphEffect] | None = None,
    modules_by_name: Mapping[str, GraphModule] | None = None,
    local_domain_facts: Mapping[str, GraphDomainFact] | None = None,
) -> GraphNode:
    rewritten = replace(
        node,
        inputs=tuple(
            _replace_operand_refs(
                item,
                subst,
                fold=fold,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                local_domain_facts=local_domain_facts,
            )
            for item in node.inputs
        ),
        attrs={
            key: _replace_operand_refs(
                value,
                subst,
                fold=fold,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                local_domain_facts=local_domain_facts,
            )
            for key, value in node.attrs.items()
        },
    )
    dim_subst = _dim_alias_subst(subst)
    return substitute_graph_node_dims(rewritten, dim_subst) if dim_subst else rewritten


def _node_replacement(
    node: GraphNode,
    *,
    config: GraphOptimizeConfig,
    module_effects: Mapping[str, GraphEffect],
    modules_by_name: Mapping[str, GraphModule],
    dim_values: Mapping[str, DimToken] | None = None,
    local_domain_facts: Mapping[str, GraphDomainFact] | None = None,
    stable_shape_values: set[str] | None = None,
    blocked_dim_ref_names: set[str] | None = None,
) -> GraphOperand | None:
    if len(node.outputs) != 1:
        return None
    if config.constant_folding:
        shape_replacement = _shape_query_replacement(
            GraphExpr(
                op=node.op,
                inputs=node.inputs,
                attrs=node.attrs,
                type_expr=node.type_expr,
                dims=node.dims,
            ),
            modules_by_name=modules_by_name,
            stable_shape_values=stable_shape_values,
            blocked_dim_ref_names=blocked_dim_ref_names,
        )
        if shape_replacement is not None:
            return shape_replacement
    if config.atomic_alias_cleanup and node.op.name == "core.alias" and len(node.inputs) == 1:
        return node.inputs[0]
    if config.atomic_alias_cleanup and node.op.name == "core.ascribe" and len(node.inputs) == 1:
        return node.inputs[0]
    if config.atomic_alias_cleanup and node.op.name in {"core.list", "core.tuple"}:
        return GraphExpr(
            op=node.op,
            inputs=node.inputs,
            attrs=node.attrs,
            type_expr=node.outputs[0].type_expr,
            dims=node.outputs[0].dims,
        )
    if (
        config.constant_folding
        and node.op.name == "core.select"
        and len(node.inputs) == 3
    ):
        cond_value = _domain_bool_value(node.inputs[0], local_domain_facts)
        if cond_value is not None:
            selected = node.inputs[1] if cond_value else node.inputs[2]
            selected_replacement = _select_fold_replacement(selected, node.type_expr)
            if selected_replacement is None:
                return None
            if _is_atomic_operand(selected_replacement) or graph_operand_effect(
                selected_replacement,
                module_effects=dict(module_effects),
            ) == GraphEffect.TOTAL_PURE:
                return selected_replacement
    if config.constant_folding and node.op.name.startswith("core.binary.") and len(node.inputs) == 2:
        left, right = node.inputs
        domain_fold = _fold_domain_binary_comparison(
            node.op.name,
            left,
            right,
            local_domain_facts=local_domain_facts,
        )
        if domain_fold is not None:
            return domain_fold
        typed_null_fold = _fold_typed_null_comparison(node.op.name, left, right)
        if typed_null_fold is not None:
            return typed_null_fold
        if isinstance(left, GraphLiteral) and isinstance(right, GraphLiteral):
            return _fold_graph_binary(node.op.name, left, right, node)
        dim_fold = _fold_dim_binary_operand(
            node.op.name,
            left,
            right,
            type_expr=node.type_expr,
            dim_values=dim_values or {},
        )
        if dim_fold is not None:
            return dim_fold
    if (
        config.constant_folding
        and _is_total_pure_op(node.op.name, module_effects)
        and (folded := _fold_numeric_primitive(node)) is not None
    ):
        return folded
    if node.op.name == "core.tuple" and len(node.inputs) == len(node.outputs):
        return None
    return None


def _multi_output_tuple_alias_subst(node: GraphNode) -> dict[str, GraphOperand] | None:
    if node.op.name != "core.tuple":
        return None
    if node.attrs or len(node.inputs) != len(node.outputs) or len(node.outputs) <= 1:
        return None
    if not all(_is_repackaging_operand(item) for item in node.inputs):
        return None
    return {
        output.name: input_operand
        for output, input_operand in zip(node.outputs, node.inputs, strict=True)
    }


def _is_repackaging_operand(operand: GraphOperand) -> bool:
    if _is_atomic_operand(operand):
        return True
    return (
        isinstance(operand, GraphExpr)
        and operand.op.name == "core.tuple"
        and not operand.attrs
        and all(_is_repackaging_operand(item) for item in operand.inputs)
    )


def _literal_select_selected_node(
    node: GraphNode,
    module_effects: Mapping[str, GraphEffect],
    *,
    local_domain_facts: Mapping[str, GraphDomainFact] | None = None,
) -> GraphNode | None:
    if (
        node.op.name != "core.select"
        or len(node.inputs) != 3
        or node.attrs
    ):
        return None
    cond_value = _domain_bool_value(node.inputs[0], local_domain_facts)
    if cond_value is None:
        return None
    selected = node.inputs[1] if cond_value else node.inputs[2]
    selected_replacement = _select_fold_replacement(selected, node.type_expr)
    if selected_replacement is None:
        selected_replacement = selected
    if not isinstance(selected_replacement, GraphExpr):
        return replace(
            node,
            op=GraphOp("core.alias"),
            inputs=(selected_replacement,),
            attrs={},
        )
    if graph_operand_effect(selected_replacement, module_effects=dict(module_effects)) == GraphEffect.TOTAL_PURE:
        return replace(
            node,
            op=GraphOp("core.alias"),
            inputs=(selected_replacement,),
            attrs={},
        )
    return replace(
        node,
        op=selected_replacement.op,
        inputs=selected_replacement.inputs,
        attrs=selected_replacement.attrs,
    )


def _call_node_output_dim_value_subst(
    node: GraphNode,
    *,
    modules_by_name: Mapping[str, GraphModule],
    local_names: set[str],
) -> dict[str, GraphOperand]:
    candidates: dict[str, GraphOperand] = {}
    conflicts: set[str] = set()

    def add_candidate(dim_name: str, operand: GraphOperand) -> None:
        if dim_name in local_names:
            return
        existing = candidates.get(dim_name)
        if existing is None:
            candidates[dim_name] = operand
        elif existing != operand:
            conflicts.add(dim_name)

    def collect_from_dim(
        formal_dim: DimToken,
        actual_dim: DimToken,
        formal_dim_values: Mapping[str, GraphOperand],
    ) -> None:
        if isinstance(formal_dim, str) and isinstance(actual_dim, str):
            replacement = formal_dim_values.get(formal_dim)
            if replacement is not None:
                add_candidate(actual_dim, replacement)
            return
        if isinstance(formal_dim, DimExprBinary) and isinstance(actual_dim, DimExprBinary):
            if formal_dim.op != actual_dim.op:
                return
            collect_from_dim(formal_dim.left, actual_dim.left, formal_dim_values)
            collect_from_dim(formal_dim.right, actual_dim.right, formal_dim_values)

    def collect_from_type(
        formal_type: TypeExpr,
        actual_type: TypeExpr,
        formal_dim_values: Mapping[str, GraphOperand],
    ) -> None:
        if isinstance(formal_type, TypeTensor) and isinstance(actual_type, TypeTensor):
            if formal_type.base != actual_type.base or len(formal_type.dims) != len(actual_type.dims):
                return
            for formal_dim, actual_dim in zip(formal_type.dims, actual_type.dims, strict=True):
                collect_from_dim(formal_dim, actual_dim, formal_dim_values)
            return
        if isinstance(formal_type, TypeNamed) and isinstance(actual_type, TypeNamed):
            if formal_type.name != actual_type.name or len(formal_type.args) != len(actual_type.args):
                return
            for formal_dim, actual_dim in zip(formal_type.args, actual_type.args, strict=True):
                collect_from_dim(formal_dim, actual_dim, formal_dim_values)
            return
        if isinstance(formal_type, TypeOptional) and isinstance(actual_type, TypeOptional):
            collect_from_type(formal_type.inner, actual_type.inner, formal_dim_values)
            return
        if isinstance(formal_type, TypeList) and isinstance(actual_type, TypeList):
            collect_from_type(formal_type.item, actual_type.item, formal_dim_values)
            return
        if isinstance(formal_type, TypeTuple) and isinstance(actual_type, TypeTuple):
            if len(formal_type.items) != len(actual_type.items):
                return
            for formal_item, actual_item in zip(formal_type.items, actual_type.items, strict=True):
                collect_from_type(formal_item, actual_item, formal_dim_values)

    callee = modules_by_name.get(node.op.name)
    if callee is None or len(node.inputs) != len(callee.inputs):
        return {}
    formal_dim_values: dict[str, GraphOperand] = {}
    for formal, actual in zip(callee.inputs, node.inputs, strict=True):
        if not isinstance(formal.type_expr, TypeDim | TypeInt):
            continue
        if isinstance(actual, GraphValueRef) and isinstance(actual.type_expr, TypeDim | TypeInt):
            formal_dim_values[formal.name] = actual
        elif isinstance(actual, GraphLiteral) and type(actual.value) is int:
            formal_dim_values[formal.name] = actual
    if not formal_dim_values:
        return {}
    formal_output_types = _module_output_types_for_arity(callee, len(node.outputs))
    for formal_type, output in zip(formal_output_types, node.outputs, strict=True):
        collect_from_type(formal_type, output.type_expr, formal_dim_values)
    for name in conflicts:
        candidates.pop(name, None)
    return candidates


def _optimize_module_local(
    module: GraphModule,
    *,
    config: GraphOptimizeConfig,
    module_effects: Mapping[str, GraphEffect],
    modules_by_name: Mapping[str, GraphModule] | None = None,
    local_domain_facts: Mapping[str, GraphDomainFact] | None = None,
    global_dim_values: Mapping[str, DimToken] | None = None,
    global_literals: Mapping[str, GraphLiteral] | None = None,
) -> GraphModule:
    modules_by_name = modules_by_name or {}
    global_literals = global_literals or {}
    before_outputs = module.outputs
    shadowed_dim_names = _module_signature_dim_refs(module)
    global_symbol_names = {
        name
        for name, global_module in modules_by_name.items()
        if not global_module.inputs and len(global_module.outputs) == 1
    }
    shadowed_global_dim_names = shadowed_dim_names & global_symbol_names
    subst: dict[str, GraphOperand] = {}
    local_names = _module_value_names(module)
    stable_shape_values = {value.name for value in module.inputs}
    dim_values: dict[str, DimToken] = {
        name: value
        for name, value in (global_dim_values or {}).items()
        if name not in shadowed_dim_names
    }
    nodes: list[GraphNode] = []
    for node in module.nodes:
        rewritten = _rewrite_node_operands(
            node,
            subst,
            fold=config.constant_folding,
            module_effects=module_effects,
            modules_by_name=modules_by_name,
            local_domain_facts=local_domain_facts,
        )
        if global_literals:
            rewritten = replace(
                rewritten,
                inputs=tuple(
                    _replace_atomic_literal_globals(item, global_literals)
                    for item in rewritten.inputs
                ),
                attrs={
                    key: _replace_atomic_literal_globals(value, global_literals)
                    for key, value in rewritten.attrs.items()
                },
            )
        selected_node = (
            _literal_select_selected_node(
                rewritten,
                module_effects,
                local_domain_facts=local_domain_facts,
            )
            if config.constant_folding
            else None
        )
        if selected_node is not None:
            rewritten = selected_node
        if (
            len(rewritten.outputs) == 1
            and not rewritten.inputs
            and not rewritten.attrs
            and rewritten.op.name in global_literals
            and rewritten.op.name != module.name
        ):
            output_name = rewritten.outputs[0].name
            replacement = global_literals[rewritten.op.name]
            if _constraints_reference_any(module.constraints, {output_name}) and _specialize_constraints(
                module.constraints,
                {output_name: replacement},
            ) is None:
                nodes.append(rewritten)
                continue
            subst[output_name] = replacement
            if type(replacement.value) is int and isinstance(
                rewritten.outputs[0].type_expr,
                TypeDim | TypeInt,
            ):
                dim_values[output_name] = replacement.value
            continue
        if (
            len(rewritten.outputs) == 1
            and not rewritten.inputs
            and not rewritten.attrs
            and rewritten.op.name != module.name
        ):
            global_module = modules_by_name.get(rewritten.op.name)
            if (
                global_module is not None
                and global_module.is_global_binding
                and not global_module.inputs
                and len(global_module.outputs) == 1
            ):
                output_name = rewritten.outputs[0].name
                replacement = GraphValueRef(
                    name=rewritten.op.name,
                    type_expr=rewritten.outputs[0].type_expr,
                    dims=rewritten.outputs[0].dims,
                )
                if _constraints_reference_any(module.constraints, {output_name}) and _specialize_constraints(
                    module.constraints,
                    {output_name: replacement},
                ) is None:
                    nodes.append(rewritten)
                    continue
                subst[output_name] = replacement
                replacement_dim = _operand_dim_token(replacement, dim_values)
                if replacement_dim is not None:
                    dim_values[output_name] = replacement_dim
                continue
        tuple_subst = (
            _multi_output_tuple_alias_subst(rewritten)
            if config.atomic_alias_cleanup
            else None
        )
        if tuple_subst is not None:
            if _constraints_reference_any(module.constraints, set(tuple_subst)) and _specialize_constraints(
                module.constraints,
                tuple_subst,
            ) is None:
                nodes.append(rewritten)
                continue
            subst.update(tuple_subst)
            continue
        replacement = _node_replacement(
            rewritten,
            config=config,
            module_effects=module_effects,
            modules_by_name=modules_by_name,
            dim_values=dim_values,
            local_domain_facts=local_domain_facts,
            stable_shape_values=stable_shape_values,
            blocked_dim_ref_names=shadowed_global_dim_names,
        )
        if replacement is not None and len(rewritten.outputs) == 1:
            output_name = rewritten.outputs[0].name
            if _constraints_reference_any(module.constraints, {output_name}) and _specialize_constraints(
                module.constraints,
                {output_name: replacement},
            ) is None:
                nodes.append(rewritten)
                continue
            subst[output_name] = replacement
            replacement_dim = _operand_dim_token(replacement, dim_values)
            if replacement_dim is not None:
                dim_values[output_name] = replacement_dim
            continue
        nodes.append(rewritten)
        if len(rewritten.outputs) == 1 and isinstance(
            rewritten.outputs[0].type_expr, TypeDim | TypeInt
        ):
            dim_expr = _operand_dim_token(
                GraphExpr(
                    op=rewritten.op,
                    inputs=rewritten.inputs,
                    attrs=rewritten.attrs,
                    type_expr=rewritten.type_expr,
                    dims=rewritten.dims,
                ),
                dim_values,
            )
            if dim_expr is not None:
                dim_values[rewritten.outputs[0].name] = dim_expr
        subst.update(
            _call_node_output_dim_value_subst(
                rewritten,
                modules_by_name=modules_by_name,
                local_names=local_names,
            )
        )
    outputs = tuple(
        _replace_operand_refs(
            item,
            subst,
            fold=config.constant_folding,
            module_effects=module_effects,
            modules_by_name=modules_by_name,
            local_domain_facts=local_domain_facts,
        )
        for item in module.outputs
    )
    constraints = _specialize_constraints(module.constraints, subst)
    module = replace(
        module,
        nodes=tuple(nodes),
        outputs=outputs,
        constraints=module.constraints if constraints is None else constraints,
    )
    module = _returned_name_preserving_module(module, before_outputs)
    if not config.dead_temp_elimination:
        return module
    return _dead_temp_eliminate_module(module, module_effects=module_effects)


def _optimize_modules_local_with_fresh_domain_facts(
    graph: GraphProgram,
    *,
    config: GraphOptimizeConfig,
    phase: str,
) -> GraphProgram:
    current = graph
    for iteration in range(config.max_iterations):
        before = current
        module_effects = infer_graph_module_effects(current.modules)
        modules_by_name = {module.name: module for module in current.modules}
        domain_analysis = infer_main_module_domain_facts(current)
        global_literals = _atomic_literal_constants(current)
        global_dim_values = _atomic_int_constant_dims(current)
        pre_refresh_modules = tuple(
            _optimize_module_local(
                module,
                config=config,
                module_effects=module_effects,
                modules_by_name=modules_by_name,
                local_domain_facts=domain_analysis.module_local_facts.get(module.name),
                global_dim_values=global_dim_values,
                global_literals=global_literals,
            )
            for module in current.modules
        )
        candidate = replace(
            current,
            modules=pre_refresh_modules,
        )
        candidate = _refresh_graph_program_types(candidate)
        candidate = replace(
            candidate,
            modules=_preserve_unchanged_module_types(
                current.modules,
                pre_refresh_modules,
                candidate.modules,
                main_module=current.main_module,
            ),
        )
        candidate = _sanitize_graph_constraints(candidate)
        try:
            _validate_optimizer_graph(candidate, phase=f"{phase}.{iteration}")
        except ValueError:
            accepted = current
            accepted_modules_by_name = {module.name: module for module in accepted.modules}
            for before_module, after_module in zip(current.modules, pre_refresh_modules, strict=True):
                if before_module == after_module:
                    continue
                trial_modules = tuple(
                    after_module if module.name == before_module.name else accepted_modules_by_name[module.name]
                    for module in accepted.modules
                )
                trial = replace(accepted, modules=trial_modules)
                trial = _sanitize_graph_constraints(trial)
                try:
                    _validate_optimizer_graph(trial, phase=f"{phase}.{iteration}.{before_module.name}")
                except ValueError:
                    continue
                accepted = trial
                accepted_modules_by_name = {module.name: module for module in accepted.modules}
            if accepted == current:
                return current
            current = accepted
            continue
        current = candidate
        if current == before:
            return current
    raise RuntimeError(
        f"graph local cleanup did not converge after {config.max_iterations} iterations"
    )


def _dead_temp_eliminate_module(
    module: GraphModule,
    *,
    module_effects: Mapping[str, GraphEffect] | None = None,
) -> GraphModule:
    live: set[str] = set()
    live.update(_module_dim_refs(module))
    for output in module.outputs:
        _operand_refs(output, live)
    kept_rev: list[GraphNode] = []
    for node in reversed(module.nodes):
        output_names = {value.name for value in node.outputs}
        if output_names and not (output_names & live) and _is_total_pure_node(
            node,
            module_effects,
        ):
            continue
        live.difference_update(output_names)
        for operand in node.inputs:
            _operand_refs(operand, live)
        for operand in node.attrs.values():
            _operand_refs(operand, live)
        kept_rev.append(node)
    kept_nodes = tuple(reversed(kept_rev))
    value_names = {value.name for value in module.inputs}
    for node in kept_nodes:
        value_names.update(value.name for value in node.outputs)
    kept_constraints = tuple(
        constraint
        for constraint in module.constraints
        if _constraint_has_callsite_guard(constraint)
        or _constraint_ref_names(constraint) <= value_names | _module_dim_refs(module)
    )
    return replace(module, nodes=kept_nodes, constraints=kept_constraints)


def _collect_value_ref_counts(
    operand: GraphOperand,
    *,
    counts: Counter[str],
    path_template_refs: set[str],
) -> None:
    if isinstance(operand, GraphValueRef):
        counts[operand.name] += 1
        return
    if isinstance(operand, GraphPath):
        path_template_refs.update(graph_path_template_names(operand))
        return
    if isinstance(operand, GraphExpr):
        for item in operand.inputs:
            _collect_value_ref_counts(
                item,
                counts=counts,
                path_template_refs=path_template_refs,
            )
        for item in operand.attrs.values():
            _collect_value_ref_counts(
                item,
                counts=counts,
                path_template_refs=path_template_refs,
            )


def _inlineable_single_use_expr_node(node: GraphNode) -> bool:
    if len(node.outputs) != 1:
        return False
    if node.op.name in {"core.alias", "core.ascribe"}:
        return len(node.inputs) == 1 and not node.attrs
    if node.op.name == "core.list":
        return bool(node.inputs) and not node.attrs
    if node.op.name.startswith("core.binary."):
        return len(node.inputs) == 2 and not node.attrs
    if node.op.name == "core.select":
        return len(node.inputs) == 3 and not node.attrs
    return False


def _safe_inline_single_use_expr_node(
    node: GraphNode,
    *,
    module_effects: Mapping[str, GraphEffect],
    module_usages: Mapping[str, UsageClass],
) -> bool:
    if graph_node_effect(
        node,
        module_effects=dict(module_effects),
    ) == GraphEffect.TOTAL_PURE:
        return _is_unrestricted_node(node, module_usages)
    if node.op.name in {"core.alias", "core.ascribe"}:
        return (
            len(node.inputs) == 1
            and not node.attrs
            and _is_atomic_operand(node.inputs[0])
            and _is_unrestricted_operand(node.inputs[0], module_usages)
        )
    if node.op.name in {"core.list", "core.tuple"}:
        return (
            not node.attrs
            and all(_is_atomic_operand(item) for item in node.inputs)
            and all(_is_unrestricted_operand(item, module_usages) for item in node.inputs)
        )
    return False


def _safe_duplicate_atomic_container_node(
    node: GraphNode,
    *,
    module_usages: Mapping[str, UsageClass],
) -> bool:
    return (
        len(node.outputs) == 1
        and node.op.name in {"core.list", "core.tuple"}
        and not node.attrs
        and all(_is_atomic_operand(item) for item in node.inputs)
        and all(_is_unrestricted_operand(item, module_usages) for item in node.inputs)
    )


def _inline_single_use_total_pure_exprs_module(
    module: GraphModule,
    *,
    module_effects: Mapping[str, GraphEffect],
    module_usages: Mapping[str, UsageClass],
    modules_by_name: Mapping[str, GraphModule],
) -> GraphModule:
    counts: Counter[str] = Counter()
    path_template_refs: set[str] = set()
    output_refs: set[str] = set()
    for node in module.nodes:
        for operand in (*node.inputs, *node.attrs.values()):
            _collect_value_ref_counts(
                operand,
                counts=counts,
                path_template_refs=path_template_refs,
            )
        if node.op.name == "core.repeat":
            for operand in node.inputs:
                _operand_refs(operand, output_refs)
    for output in module.outputs:
        _collect_value_ref_counts(
            output,
            counts=counts,
            path_template_refs=path_template_refs,
        )
        _operand_refs(output, output_refs)

    hard_blocked_refs = path_template_refs | output_refs
    type_dim_refs = _collect_module_free_symbols(module).type_dim_refs
    blocked_refs = hard_blocked_refs | type_dim_refs
    subst: dict[str, GraphOperand] = {}
    nodes: list[GraphNode] = []
    changed = False
    for node in module.nodes:
        rewritten = _rewrite_node_operands(
            node,
            subst,
            fold=True,
            module_effects=module_effects,
            modules_by_name=modules_by_name,
        )
        if not _inlineable_single_use_expr_node(rewritten):
            nodes.append(rewritten)
            continue
        output = rewritten.outputs[0]
        can_duplicate_container = _safe_duplicate_atomic_container_node(
            rewritten,
            module_usages=module_usages,
        )
        if (
            output.name in hard_blocked_refs
            or (
                output.name in blocked_refs
                and not can_duplicate_container
            )
            or (counts[output.name] != 1 and not can_duplicate_container)
        ):
            nodes.append(rewritten)
            continue
        replacement = GraphExpr(
            op=rewritten.op,
            inputs=rewritten.inputs,
            attrs=rewritten.attrs,
            type_expr=output.type_expr,
            dims=output.dims,
        )
        if _constraints_reference_any(module.constraints, {output.name}) and _specialize_constraints(
            module.constraints,
            {**subst, output.name: replacement},
        ) is None:
            nodes.append(rewritten)
            continue
        if not _safe_inline_single_use_expr_node(
            rewritten,
            module_effects=module_effects,
            module_usages=module_usages,
        ):
            nodes.append(rewritten)
            continue
        subst[output.name] = replacement
        changed = True

    if not changed:
        return module
    outputs = tuple(
        _replace_operand_refs(
            output,
            subst,
            fold=True,
            module_effects=module_effects,
            modules_by_name=modules_by_name,
        )
        for output in module.outputs
    )
    constraints = _specialize_constraints(module.constraints, subst)
    if constraints is None:
        return module
    return replace(module, nodes=tuple(nodes), outputs=outputs, constraints=constraints)


def _collect_nested_total_expr_counts(
    operand: GraphOperand,
    *,
    module_effects: Mapping[str, GraphEffect],
    module_usages: Mapping[str, UsageClass],
    counts: Counter[object],
) -> None:
    if not isinstance(operand, GraphExpr):
        return
    for item in operand.inputs:
        _collect_nested_total_expr_counts(
            item,
            module_effects=module_effects,
            module_usages=module_usages,
            counts=counts,
        )
    for item in operand.attrs.values():
        _collect_nested_total_expr_counts(
            item,
            module_effects=module_effects,
            module_usages=module_usages,
            counts=counts,
        )
    if (
        graph_operand_effect(operand, module_effects=dict(module_effects)) == GraphEffect.TOTAL_PURE
        and _is_unrestricted_operand(operand, module_usages)
    ):
        counts[_graph_operand_key(operand)] += 1


def _fresh_graph_value_name(used_names: set[str], preferred: str) -> str:
    if preferred not in used_names:
        used_names.add(preferred)
        return preferred
    index = 1
    while True:
        candidate = f"{preferred}_{index}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def _hoist_repeated_nested_total_exprs_module(
    module: GraphModule,
    *,
    module_effects: Mapping[str, GraphEffect],
    module_usages: Mapping[str, UsageClass],
) -> GraphModule:
    counts: Counter[object] = Counter()
    for node in module.nodes:
        for operand in (*node.inputs, *node.attrs.values()):
            _collect_nested_total_expr_counts(
                operand,
                module_effects=module_effects,
                module_usages=module_usages,
                counts=counts,
            )
    for output in module.outputs:
        _collect_nested_total_expr_counts(
            output,
            module_effects=module_effects,
            module_usages=module_usages,
            counts=counts,
        )
    repeated = {key for key, count in counts.items() if count > 1}
    if not repeated:
        return module

    used_names = _module_value_names(module)
    emitted: dict[object, GraphValueRef] = {}
    temp_index = 0
    changed = False

    def rewrite_operand(
        operand: GraphOperand,
        *,
        inserted_nodes: list[GraphNode],
        source_id: str,
    ) -> GraphOperand:
        nonlocal changed, temp_index
        if not isinstance(operand, GraphExpr):
            return operand
        original_key = _graph_operand_key(operand)
        rewritten = replace(
            operand,
            inputs=tuple(
                rewrite_operand(item, inserted_nodes=inserted_nodes, source_id=source_id)
                for item in operand.inputs
            ),
            attrs={
                key: rewrite_operand(value, inserted_nodes=inserted_nodes, source_id=source_id)
                for key, value in operand.attrs.items()
            },
        )
        if original_key not in repeated:
            return rewritten
        existing = emitted.get(original_key)
        if existing is not None:
            changed = True
            return existing
        temp_index += 1
        name = _fresh_graph_value_name(used_names, f"__cse{temp_index}")
        value = GraphValue(name=name, type_expr=rewritten.type_expr, dims=rewritten.dims)
        ref = GraphValueRef(name=name, type_expr=value.type_expr, dims=value.dims)
        emitted[original_key] = ref
        inserted_nodes.append(
            GraphNode(
                id=f"{module.name}:nested-cse:{source_id}:{temp_index}",
                op=rewritten.op,
                inputs=rewritten.inputs,
                attrs=rewritten.attrs,
                outputs=(value,),
                source_module=module.name,
                type_expr=rewritten.type_expr,
                dims=rewritten.dims,
            )
        )
        changed = True
        return ref

    nodes: list[GraphNode] = []
    for node in module.nodes:
        inserted: list[GraphNode] = []
        rewritten = replace(
            node,
            inputs=tuple(
                rewrite_operand(item, inserted_nodes=inserted, source_id=node.id)
                for item in node.inputs
            ),
            attrs={
                key: rewrite_operand(value, inserted_nodes=inserted, source_id=node.id)
                for key, value in node.attrs.items()
            },
        )
        nodes.extend(inserted)
        nodes.append(rewritten)
    output_inserted: list[GraphNode] = []
    outputs = tuple(
        rewrite_operand(output, inserted_nodes=output_inserted, source_id="return")
        for output in module.outputs
    )
    nodes.extend(output_inserted)
    if not changed:
        return module
    return replace(module, nodes=tuple(nodes), outputs=outputs)


def _hoist_eager_nested_exprs_module(
    module: GraphModule,
    *,
    module_effects: Mapping[str, GraphEffect],
    module_usages: Mapping[str, UsageClass],
) -> GraphModule:
    used_names = _module_value_names(module)
    next_index = 1
    changed = False

    def fresh_name() -> str:
        nonlocal next_index
        name, next_index = _fresh_canonical_value_name(used_names, next_index)
        return name

    def hoist_operand(
        operand: GraphOperand,
        *,
        inserted_nodes: list[GraphNode],
        source_id: str,
        eager: bool,
    ) -> GraphOperand:
        nonlocal changed
        if not isinstance(operand, GraphExpr):
            return operand
        if operand.op.name == "core.select" and len(operand.inputs) == 3 and not operand.attrs:
            rewritten = replace(
                operand,
                inputs=(
                    hoist_operand(
                        operand.inputs[0],
                        inserted_nodes=inserted_nodes,
                        source_id=source_id,
                        eager=True,
                    ),
                    hoist_operand(
                        operand.inputs[1],
                        inserted_nodes=inserted_nodes,
                        source_id=source_id,
                        eager=False,
                    ),
                    hoist_operand(
                        operand.inputs[2],
                        inserted_nodes=inserted_nodes,
                        source_id=source_id,
                        eager=False,
                    ),
                ),
            )
        else:
            rewritten = replace(
                operand,
                inputs=tuple(
                    hoist_operand(
                        item,
                        inserted_nodes=inserted_nodes,
                        source_id=source_id,
                        eager=True,
                    )
                    for item in operand.inputs
                ),
                attrs={
                    key: hoist_operand(
                        value,
                        inserted_nodes=inserted_nodes,
                        source_id=source_id,
                        eager=True,
                    )
                    for key, value in operand.attrs.items()
                },
            )
        if (
            not eager
            or rewritten.op.name in {"core.list", "core.tuple"}
            or graph_operand_effect(rewritten, module_effects=dict(module_effects)) != GraphEffect.TOTAL_PURE
            or not _is_unrestricted_operand(rewritten, module_usages)
        ):
            return rewritten
        name = fresh_name()
        value = GraphValue(name=name, type_expr=rewritten.type_expr, dims=rewritten.dims)
        inserted_nodes.append(
            GraphNode(
                id=f"{module.name}:hoist:{source_id}:{name}",
                op=rewritten.op,
                inputs=rewritten.inputs,
                attrs=rewritten.attrs,
                outputs=(value,),
                source_module=module.name,
                type_expr=rewritten.type_expr,
                dims=rewritten.dims,
            )
        )
        changed = True
        return GraphValueRef(name=name, type_expr=value.type_expr, dims=value.dims)

    nodes: list[GraphNode] = []
    for node in module.nodes:
        inserted: list[GraphNode] = []
        if node.op.name == "core.select" and len(node.inputs) == 3 and not node.attrs:
            inputs = (
                hoist_operand(node.inputs[0], inserted_nodes=inserted, source_id=node.id, eager=True),
                hoist_operand(node.inputs[1], inserted_nodes=inserted, source_id=node.id, eager=False),
                hoist_operand(node.inputs[2], inserted_nodes=inserted, source_id=node.id, eager=False),
            )
            attrs = node.attrs
        else:
            inputs = tuple(
                hoist_operand(item, inserted_nodes=inserted, source_id=node.id, eager=True)
                for item in node.inputs
            )
            attrs = {
                key: hoist_operand(value, inserted_nodes=inserted, source_id=node.id, eager=True)
                for key, value in node.attrs.items()
            }
        nodes.extend(inserted)
        nodes.append(replace(node, inputs=inputs, attrs=attrs))
    if not changed:
        return module
    return replace(module, nodes=tuple(nodes))


def _hoist_eager_nested_exprs(
    graph: GraphProgram,
    *,
    module_effects: Mapping[str, GraphEffect],
    module_usages: Mapping[str, UsageClass],
) -> GraphProgram:
    modules = tuple(
        _hoist_eager_nested_exprs_module(
            module,
            module_effects=module_effects,
            module_usages=module_usages,
        )
        for module in graph.modules
    )
    if modules == graph.modules:
        return graph
    return replace(graph, modules=modules)


def _common_subexpression_eliminate_module(
    module: GraphModule,
    *,
    module_effects: Mapping[str, GraphEffect],
    module_usages: Mapping[str, UsageClass],
    fold: bool,
) -> GraphModule:
    subst: dict[str, GraphOperand] = {}
    seen: dict[object, tuple[GraphValue, ...]] = {}
    nodes: list[GraphNode] = []
    changed = False
    for node in module.nodes:
        rewritten = _rewrite_node_operands(
            node,
            subst,
            fold=fold,
            module_effects=module_effects,
        )
        if (
            len(rewritten.outputs) == 1
            and _is_total_pure_node(rewritten, module_effects)
            and _is_unrestricted_node(rewritten, module_usages)
        ):
            key = _graph_node_cse_key(rewritten)
            previous = seen.get(key)
            if previous is not None:
                output_name = rewritten.outputs[0].name
                if previous[0].name in subst:
                    nodes.append(rewritten)
                    continue
                replacement = GraphValueRef(
                    name=previous[0].name,
                    type_expr=rewritten.outputs[0].type_expr,
                    dims=rewritten.outputs[0].dims,
                )
                replacement_refs: set[str] = set()
                _operand_refs(replacement, replacement_refs)
                if output_name in replacement_refs:
                    nodes.append(rewritten)
                    continue
                if _constraints_reference_any(module.constraints, {output_name}) and _specialize_constraints(
                    module.constraints,
                    {output_name: replacement},
                ) is None:
                    nodes.append(rewritten)
                    continue
                subst[output_name] = replacement
                changed = True
                continue
            seen[key] = rewritten.outputs
        nodes.append(rewritten)
    outputs = tuple(
        _replace_operand_refs(
            output,
            subst,
            fold=fold,
            module_effects=module_effects,
        )
        for output in module.outputs
    )
    constraints = _specialize_constraints(module.constraints, subst)
    rewritten_module = (
        module
        if not changed
        else replace(
            module,
            nodes=tuple(nodes),
            outputs=outputs,
            constraints=module.constraints if constraints is None else constraints,
        )
    )
    return _hoist_repeated_nested_total_exprs_module(
        rewritten_module,
        module_effects=module_effects,
        module_usages=module_usages,
    )


def _returned_name_preserving_module(module: GraphModule, before_outputs: tuple[GraphOperand, ...]) -> GraphModule:
    renames: dict[str, str] = {}
    input_names = {value.name for value in module.inputs}
    defined_names = set(input_names)
    node_output_names: set[str] = set()
    for node in module.nodes:
        for value in node.outputs:
            defined_names.add(value.name)
            node_output_names.add(value.name)
    for before, after in zip(before_outputs, module.outputs, strict=False):
        if not isinstance(before, GraphValueRef) or not isinstance(after, GraphValueRef):
            continue
        if before.name == after.name:
            continue
        if after.name not in node_output_names:
            continue
        if before.name in defined_names:
            continue
        renames[after.name] = before.name
        defined_names.add(before.name)
    if not renames:
        return module
    constraint_subst = {
        old: GraphValueRef(name=new, type_expr=TypeAny())
        for old, new in renames.items()
        if old != new
    }
    constraints = _specialize_constraints(module.constraints, constraint_subst)
    return replace(
        module,
        nodes=tuple(
            replace(
                node,
                inputs=tuple(rename_operand(item, renames) for item in node.inputs),
                attrs={key: rename_operand(value, renames) for key, value in node.attrs.items()},
                outputs=tuple(
                    replace(output, name=renames.get(output.name, output.name))
                    for output in node.outputs
                ),
            )
            for node in module.nodes
        ),
        outputs=tuple(rename_operand(output, renames) for output in module.outputs),
        constraints=module.constraints if constraints is None else constraints,
    )


def _graph_call_graph(graph: GraphProgram) -> dict[str, set[str]]:
    module_names = {module.name for module in graph.modules}
    calls: dict[str, set[str]] = {module.name: set() for module in graph.modules}
    for module in graph.modules:
        for node in module.nodes:
            if node.op.name in module_names:
                calls[module.name].add(node.op.name)
            repeat_callee = _core_repeat_callee(node, module_names)
            if repeat_callee is not None:
                calls[module.name].add(repeat_callee)
            for operand in node.inputs:
                _operand_module_calls(operand, module_names, calls[module.name])
            for operand in node.attrs.values():
                _operand_module_calls(operand, module_names, calls[module.name])
        for operand in module.outputs:
            _operand_module_calls(operand, module_names, calls[module.name])
        _module_metadata_refs(module, module_names, calls[module.name])
    return calls


def _core_repeat_callee(node: GraphNode, module_names: set[str]) -> str | None:
    if node.op.name != "core.repeat":
        return None
    callee = node.attrs.get("callee")
    if isinstance(callee, GraphLiteral) and isinstance(callee.value, str) and callee.value in module_names:
        return callee.value
    return None


def _strongly_connected_components(edges: Mapping[str, set[str]]) -> list[set[str]]:
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[set[str]] = []

    def visit(name: str) -> None:
        nonlocal index
        indices[name] = index
        lowlinks[name] = index
        index += 1
        stack.append(name)
        on_stack.add(name)
        for target in sorted(edges.get(name, ())):
            if target not in edges:
                continue
            if target not in indices:
                visit(target)
                lowlinks[name] = min(lowlinks[name], lowlinks[target])
            elif target in on_stack:
                lowlinks[name] = min(lowlinks[name], indices[target])
        if lowlinks[name] != indices[name]:
            return
        component: set[str] = set()
        while True:
            item = stack.pop()
            on_stack.remove(item)
            component.add(item)
            if item == name:
                break
        components.append(component)

    for name in sorted(edges):
        if name not in indices:
            visit(name)
    return components


def _recursive_modules(graph: GraphProgram) -> set[str]:
    edges = _graph_call_graph(graph)
    recursive: set[str] = set()
    for component in _strongly_connected_components(edges):
        if len(component) > 1:
            recursive.update(component)
            continue
        name = next(iter(component))
        if name in edges.get(name, ()):
            recursive.add(name)
    return recursive


def prune_graph_to_main(graph: GraphProgram) -> GraphProgram:
    graph = _sanitize_graph_constraints(graph)
    _validate_optimizer_graph(graph, phase="prune.input")
    calls = _graph_call_graph(graph)
    seen: set[str] = set()
    stack = [graph.main_module]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(sorted(calls.get(name, ())))
    pruned = replace(graph, modules=tuple(module for module in graph.modules if module.name in seen))
    pruned = _sanitize_graph_constraints(pruned)
    _validate_optimizer_graph(pruned, phase="prune")
    return pruned


def _call_counts(graph: GraphProgram) -> Counter[str]:
    module_names = {module.name for module in graph.modules}
    counts: Counter[str] = Counter()
    for module in graph.modules:
        for node in module.nodes:
            if node.op.name in module_names:
                counts[node.op.name] += 1
            for operand in node.inputs:
                _count_operand_module_calls(operand, module_names, counts)
            for operand in node.attrs.values():
                _count_operand_module_calls(operand, module_names, counts)
        for operand in module.outputs:
            _count_operand_module_calls(operand, module_names, counts)
    return counts


def _top_level_calls_by_callee(graph: GraphProgram) -> dict[str, list[tuple[str, GraphNode]]]:
    module_names = {module.name for module in graph.modules}
    calls: dict[str, list[tuple[str, GraphNode]]] = {name: [] for name in module_names}
    for module in graph.modules:
        for node in module.nodes:
            if node.op.name in module_names:
                calls[node.op.name].append((module.name, node))
    return calls


def _all_calls_by_callee(graph: GraphProgram) -> dict[str, list[tuple[str, GraphNode | GraphExpr]]]:
    module_names = {module.name for module in graph.modules}
    calls: dict[str, list[tuple[str, GraphNode | GraphExpr]]] = {name: [] for name in module_names}

    def collect_operand(caller_name: str, operand: GraphOperand) -> None:
        if not isinstance(operand, GraphExpr):
            return
        if operand.op.name in module_names:
            calls[operand.op.name].append((caller_name, operand))
        for item in operand.inputs:
            collect_operand(caller_name, item)
        for item in operand.attrs.values():
            collect_operand(caller_name, item)

    for module in graph.modules:
        for node in module.nodes:
            if node.op.name in module_names:
                calls[node.op.name].append((module.name, node))
            for operand in node.inputs:
                collect_operand(module.name, operand)
            for operand in node.attrs.values():
                collect_operand(module.name, operand)
        for output in module.outputs:
            collect_operand(module.name, output)
    return calls


def _dead_formal_referenced_names(module: GraphModule) -> set[str]:
    symbols = _collect_module_free_symbols(module)
    refs = set(symbols.value_refs)
    refs.update(symbols.path_refs)
    refs.update(symbols.type_dim_refs)
    refs.update(symbols.term_dim_refs)
    refs.update(symbols.constraint_refs)
    return refs


def _formal_bound_dim_names(formal: GraphValue) -> set[str]:
    refs = _type_dim_refs(formal.type_expr)
    refs.update(_dims_dim_refs(formal.dims))
    return refs


def _safe_to_drop_dead_actual(
    operand: GraphOperand,
    *,
    module_effects: dict[str, GraphEffect],
    modules_by_name: dict[str, GraphModule],
) -> bool:
    if isinstance(operand, GraphValueRef | GraphLiteral | GraphPath):
        return True
    return (
        graph_operand_effect(
            operand,
            module_effects=module_effects,
            modules_by_name=modules_by_name,
        )
        == GraphEffect.TOTAL_PURE
    )


def _rewrite_dead_formal_call_operand(
    operand: GraphOperand,
    *,
    drop_indices_by_module: Mapping[str, set[int]],
) -> GraphOperand:
    if not isinstance(operand, GraphExpr):
        return operand
    inputs = tuple(
        _rewrite_dead_formal_call_operand(
            item,
            drop_indices_by_module=drop_indices_by_module,
        )
        for item in operand.inputs
    )
    attrs = {
        name: _rewrite_dead_formal_call_operand(
            item,
            drop_indices_by_module=drop_indices_by_module,
        )
        for name, item in operand.attrs.items()
    }
    drop_indices = drop_indices_by_module.get(operand.op.name)
    if drop_indices:
        inputs = tuple(
            item
            for index, item in enumerate(inputs)
            if index not in drop_indices
        )
    if inputs == operand.inputs and attrs == operand.attrs:
        return operand
    return replace(operand, inputs=inputs, attrs=attrs)


def _rewrite_dead_formal_call_node(
    node: GraphNode,
    *,
    drop_indices_by_module: Mapping[str, set[int]],
) -> GraphNode:
    inputs = tuple(
        _rewrite_dead_formal_call_operand(
            item,
            drop_indices_by_module=drop_indices_by_module,
        )
        for item in node.inputs
    )
    attrs = {
        name: _rewrite_dead_formal_call_operand(
            item,
            drop_indices_by_module=drop_indices_by_module,
        )
        for name, item in node.attrs.items()
    }
    drop_indices = drop_indices_by_module.get(node.op.name)
    if drop_indices:
        inputs = tuple(
            item
            for index, item in enumerate(inputs)
            if index not in drop_indices
        )
    if inputs == node.inputs and attrs == node.attrs:
        return node
    return replace(node, inputs=inputs, attrs=attrs)


def _prune_dead_formals(graph: GraphProgram) -> GraphProgram:
    module_by_name = {module.name: module for module in graph.modules}
    module_effects = infer_graph_module_effects(graph.modules)
    recursive_modules = _recursive_modules(graph)
    calls_by_callee = _all_calls_by_callee(graph)
    drop_indices_by_module: dict[str, set[int]] = {}

    for module in graph.modules:
        if module.name == graph.main_module or module.is_global_binding:
            continue
        if module.name in recursive_modules:
            continue
        formal_names = {formal.name for formal in module.inputs}
        if _type_references_any_dim_name(module.return_type_expr, formal_names):
            continue
        callsites = calls_by_callee.get(module.name, [])
        if not callsites:
            continue
        if any(len(call.inputs) != len(module.inputs) for _, call in callsites):
            continue
        referenced = _dead_formal_referenced_names(module)
        drop_indices: set[int] = set()
        for index, formal in enumerate(module.inputs):
            if formal.name in referenced:
                continue
            if _formal_bound_dim_names(formal) & referenced:
                continue
            if not all(
                _safe_to_drop_dead_actual(
                    call.inputs[index],
                    module_effects=module_effects,
                    modules_by_name=module_by_name,
                )
                for _, call in callsites
            ):
                continue
            drop_indices.add(index)
        if drop_indices:
            drop_indices_by_module[module.name] = drop_indices

    if not drop_indices_by_module:
        return graph

    rewritten_modules: list[GraphModule] = []
    for module in graph.modules:
        drop_indices = drop_indices_by_module.get(module.name, set())
        inputs = tuple(
            value
            for index, value in enumerate(module.inputs)
            if index not in drop_indices
        )
        nodes = tuple(
            _rewrite_dead_formal_call_node(
                node,
                drop_indices_by_module=drop_indices_by_module,
            )
            for node in module.nodes
        )
        outputs = tuple(
            _rewrite_dead_formal_call_operand(
                output,
                drop_indices_by_module=drop_indices_by_module,
            )
            for output in module.outputs
        )
        rewritten_modules.append(replace(module, inputs=inputs, nodes=nodes, outputs=outputs))

    pruned = replace(graph, modules=tuple(rewritten_modules))
    pruned = _refresh_graph_program_types(pruned)
    pruned = _sanitize_graph_constraints(pruned)
    original_return_types = {module.name: module.return_type_expr for module in graph.modules}
    for module in pruned.modules:
        if module.return_type_expr != original_return_types.get(module.name):
            return graph
    try:
        _validate_optimizer_graph(pruned, phase="prune_dead_formals")
    except ValueError:
        return graph
    return pruned


def _top_level_call_counts(graph: GraphProgram) -> Counter[str]:
    module_names = {module.name for module in graph.modules}
    counts: Counter[str] = Counter()
    for module in graph.modules:
        for node in module.nodes:
            if node.op.name in module_names:
                counts[node.op.name] += 1
        for output in module.outputs:
            if isinstance(output, GraphExpr) and output.op.name in module_names:
                counts[output.op.name] += 1
    return counts


def _has_safe_specialization_actual(
    node: GraphNode,
    module: GraphModule,
    *,
    global_symbol_names: set[str],
    input_domain_facts: Mapping[str, GraphDomainFact] | None = None,
) -> bool:
    if len(node.inputs) != len(module.inputs):
        return False
    return any(
        (
            _domain_fact_specialization_operand(
                None if input_domain_facts is None else input_domain_facts.get(formal.name),
                formal,
            )
            is not None
            or _is_safe_callsite_specialization_operand(
                item,
                global_symbol_names=global_symbol_names,
            )
        )
        and _specialization_actual_matches_formal(item, formal)
        for formal, item in zip(module.inputs, node.inputs, strict=True)
    )


def _callsite_specialization_subst(
    module: GraphModule,
    inputs: tuple[GraphOperand, ...],
    *,
    global_symbol_names: set[str],
    caller_name: str | None = None,
    caller_modules: Mapping[str, GraphModule] | None = None,
    input_domain_facts: Mapping[str, GraphDomainFact] | None = None,
) -> dict[str, GraphOperand]:
    if len(inputs) != len(module.inputs):
        return {}
    subst: dict[str, GraphOperand] = {}
    for formal, actual in zip(module.inputs, inputs, strict=True):
        if not _specialization_actual_matches_formal(actual, formal):
            continue
        if _is_safe_callsite_specialization_operand(
            actual,
            global_symbol_names=global_symbol_names,
        ):
            subst[formal.name] = _canonical_specialization_operand(
                actual,
                global_symbol_names=global_symbol_names,
            )
            continue
        if caller_name is None or caller_modules is None:
            continue
        traced = _candidate_actual_from_operand(
            actual,
            caller_name=caller_name,
            caller_modules=caller_modules,
            candidates={},
            global_symbol_names=global_symbol_names,
        )
        if traced is not None and _specialization_actual_matches_formal(traced, formal):
            subst[formal.name] = _canonical_specialization_operand(
                traced,
                global_symbol_names=global_symbol_names,
            )
            continue
        domain_actual = _domain_fact_specialization_operand(
            None if input_domain_facts is None else input_domain_facts.get(formal.name),
            formal,
        )
        if domain_actual is not None:
            subst[formal.name] = _canonical_specialization_operand(
                domain_actual,
                global_symbol_names=global_symbol_names,
            )
    return subst


def _can_specialize_module(module: GraphModule, *, recursive_modules: set[str], main_module: str) -> bool:
    if module.name == main_module:
        return False
    if _specialized_module_base(module.name) is not None:
        return False
    if module.name in recursive_modules:
        return False
    return True


def _constraint_operand_ref_names(operand: ConstraintOperand) -> set[str]:
    if isinstance(operand, tuple):
        names: set[str] = set()
        for item in operand:
            names.update(_constraint_atom_ref_names(item))
        return names
    return _constraint_atom_ref_names(operand)


def _constraint_atom_ref_names(atom: ConstraintAtom) -> set[str]:
    if isinstance(atom, str):
        return {atom}
    if isinstance(atom, DimExprBinary):
        return set(dim_token_names(atom))
    return set()


def _constraint_ref_names(constraint: Constraint) -> set[str]:
    if constraint.relation == "callsite":
        return set()
    names = _constraint_operand_ref_names(constraint.left)
    if constraint.right is not None:
        names.update(_constraint_operand_ref_names(constraint.right))
    for guard in constraint.guards:
        names.update(_constraint_ref_names(guard))
    return names


def _constraint_has_callsite_guard(constraint: Constraint) -> bool:
    return any(
        guard.relation == "callsite" or _constraint_has_callsite_guard(guard)
        for guard in constraint.guards
    )


def _constraints_reference_any(
    constraints: tuple[Constraint, ...],
    names: set[str],
) -> bool:
    return any(_constraint_ref_names(constraint) & names for constraint in constraints)


def _constraint_is_trivially_true(constraint: Constraint) -> bool:
    if constraint.guards and not all(_constraint_is_trivially_true(guard) for guard in constraint.guards):
        return False
    left = constraint.left
    right = constraint.right
    if constraint.relation == "=":
        return right is not None and left == right
    if constraint.relation == "!=":
        return right is not None and _constraint_literals_comparable(left, right) and left != right
    if constraint.relation == "is_true":
        return left is True and right is None
    if constraint.relation == "is_false":
        return left is False and right is None
    if constraint.relation == "is_null":
        return left is None and right is None
    if constraint.relation == "not_null":
        return left is not None and not isinstance(left, str | tuple | DimExprBinary) and right is None
    if (
        constraint.relation in {"<", "<=", ">", ">="}
        and type(left) is int
        and type(right) is int
    ):
        return _evaluate_int_relation(left, constraint.relation, right)
    return False


def _constraint_is_trivially_false(constraint: Constraint) -> bool:
    if constraint.guards and not all(_constraint_is_trivially_true(guard) for guard in constraint.guards):
        return False
    left = constraint.left
    right = constraint.right
    if constraint.relation == "=":
        return right is not None and _constraint_literals_comparable(left, right) and left != right
    if constraint.relation == "!=":
        return right is not None and left == right
    if constraint.relation == "is_true":
        return left is False and right is None
    if constraint.relation == "is_false":
        return left is True and right is None
    if constraint.relation == "is_null":
        return left is not None and not isinstance(left, str | tuple | DimExprBinary) and right is None
    if constraint.relation == "not_null":
        return left is None and right is None
    if (
        constraint.relation in {"<", "<=", ">", ">="}
        and type(left) is int
        and type(right) is int
    ):
        return not _evaluate_int_relation(left, constraint.relation, right)
    return False


def _constraint_literals_comparable(left: ConstraintOperand, right: ConstraintOperand) -> bool:
    literal_types = (int, bool, type(None))
    return isinstance(left, literal_types) and isinstance(right, literal_types)


def _evaluate_int_relation(left: int, relation: str, right: int) -> bool:
    if relation == "<":
        return left < right
    if relation == "<=":
        return left <= right
    if relation == ">":
        return left > right
    if relation == ">=":
        return left >= right
    raise ValueError(f"unsupported int constraint relation {relation!r}")


def _specialize_constraints(
    constraints: tuple[Constraint, ...],
    subst: Mapping[str, GraphOperand],
) -> tuple[Constraint, ...] | None:
    rewritten: list[Constraint] = []
    for constraint in constraints:
        try:
            candidate = replace_constraint_refs(constraint, subst)
        except UnsupportedConstraintSubstitution:
            if _constraint_has_callsite_guard(constraint):
                continue
            return None
        if _constraint_is_trivially_false(candidate):
            return None
        if _constraint_is_trivially_true(candidate):
            continue
        rewritten.append(candidate)
    return tuple(rewritten)


def _specialized_module(
    module: GraphModule,
    *,
    name: str,
    call_node: GraphNode,
    subst_override: Mapping[str, GraphOperand] | None = None,
    global_symbol_names: set[str] | None = None,
) -> GraphModule | None:
    if len(call_node.inputs) != len(module.inputs):
        return None
    subst: dict[str, GraphOperand] = dict(subst_override or {})
    dim_subst: dict[str, DimToken] = {}
    kept_inputs: list[GraphValue] = []
    for formal, actual in zip(module.inputs, call_node.inputs, strict=True):
        if formal.name in subst:
            actual = subst[formal.name]
            subst[formal.name] = actual
            if (
                isinstance(formal.type_expr, TypeDim | TypeInt)
                and isinstance(actual, GraphLiteral)
                and type(actual.value) is int
            ):
                dim_subst[formal.name] = actual.value
            continue
        if subst_override is None and _is_safe_callsite_specialization_operand(
            actual,
            global_symbol_names=global_symbol_names or set(),
        ):
            subst[formal.name] = _canonical_specialization_operand(
                actual,
                global_symbol_names=global_symbol_names or set(),
            )
            if (
                isinstance(formal.type_expr, TypeDim | TypeInt)
                and isinstance(actual, GraphLiteral)
                and type(actual.value) is int
            ):
                dim_subst[formal.name] = actual.value
            continue
        kept_inputs.append(formal)
    if not subst:
        return None
    kept_constraints = _specialize_constraints(module.constraints, subst)
    if kept_constraints is None:
        return None
    nodes = tuple(_rewrite_node_operands(node, subst) for node in module.nodes)
    outputs = tuple(_replace_operand_refs(output, subst) for output in module.outputs)
    specialized = replace(
        module,
        name=name,
        inputs=tuple(kept_inputs),
        nodes=nodes,
        outputs=outputs,
        constraints=tuple(kept_constraints),
    )
    if dim_subst:
        dim_specialized = substitute_graph_module_dims(specialized, dim_subst)
        specialized = dim_specialized
    cleanup_config = GraphOptimizeConfig(
        prune_to_main=False,
        common_subexpression_elimination=False,
        specialize_definitions="off",
        inline_safe=False,
    )
    for _ in range(64):
        before = specialized
        specialized = _optimize_module_local(
            specialized,
            config=cleanup_config,
            module_effects={},
        )
        if specialized == before:
            break
    else:
        raise RuntimeError(f"specialized module {name!r} local cleanup did not converge")
    bound_names = (
        {value.name for value in specialized.inputs}
        | {value.name for node in specialized.nodes for value in node.outputs}
        | _module_signature_dim_refs(specialized)
        | (global_symbol_names or set())
    )
    specialized = replace(
        specialized,
        constraints=tuple(
            constraint
            for constraint in specialized.constraints
            if not (
                _constraint_has_callsite_guard(constraint)
                and not (_constraint_ref_names(constraint) <= bound_names)
            )
        ),
    )
    if not _specialized_module_render_closure_safe(
        specialized,
        global_symbol_names=global_symbol_names or set(),
    ):
        return None
    return specialized


def _specialized_candidate_valid(
    graph: GraphProgram,
    modules: tuple[GraphModule, ...],
    *,
    phase: str,
) -> bool:
    candidate = replace(graph, modules=modules)
    candidate = _sanitize_graph_constraints(candidate)
    try:
        _validate_optimizer_graph(candidate, phase=phase)
    except ValueError:
        return False
    return True


def _rewrite_call_to_specialized(node: GraphNode, original: GraphModule, specialized_name: str) -> GraphNode:
    return _rewrite_call_to_specialized_with_subst(node, original, specialized_name, None)


def _rewrite_call_to_specialized_with_subst(
    node: GraphNode,
    original: GraphModule,
    specialized_name: str,
    subst_names: set[str] | None,
) -> GraphNode:
    inputs = tuple(
        actual
        for formal, actual in zip(original.inputs, node.inputs, strict=True)
        if not (
            formal.name in subst_names
            if subst_names is not None
            else _is_safe_specialization_operand(actual)
        )
    )
    return replace(node, op=GraphOp(specialized_name), inputs=inputs)


def _rewrite_call_expr_to_specialized_with_subst(
    expr: GraphExpr,
    original: GraphModule,
    specialized_name: str,
    subst_names: set[str],
) -> GraphExpr:
    inputs = tuple(
        actual
        for formal, actual in zip(original.inputs, expr.inputs, strict=True)
        if formal.name not in subst_names
    )
    return replace(expr, op=GraphOp(specialized_name), inputs=inputs)


def _rewrite_recursive_specialized_operand(
    operand: GraphOperand,
    *,
    originals_by_name: Mapping[str, GraphModule],
    clone_names: Mapping[str, str],
    subst_names_by_module: Mapping[str, set[str]],
) -> GraphOperand:
    if not isinstance(operand, GraphExpr):
        return operand
    inputs = tuple(
        _rewrite_recursive_specialized_operand(
            item,
            originals_by_name=originals_by_name,
            clone_names=clone_names,
            subst_names_by_module=subst_names_by_module,
        )
        for item in operand.inputs
    )
    attrs = {
        key: _rewrite_recursive_specialized_operand(
            value,
            originals_by_name=originals_by_name,
            clone_names=clone_names,
            subst_names_by_module=subst_names_by_module,
        )
        for key, value in operand.attrs.items()
    }
    rewritten = replace(operand, inputs=inputs, attrs=attrs)
    clone_name = clone_names.get(rewritten.op.name)
    if clone_name is None:
        return rewritten
    original = originals_by_name[rewritten.op.name]
    subst_names = subst_names_by_module[rewritten.op.name]
    return _rewrite_call_expr_to_specialized_with_subst(
        rewritten,
        original,
        clone_name,
        subst_names,
    )


def _rewrite_specialized_nested_operand(
    operand: GraphOperand,
    *,
    replacements: Mapping[str, tuple[str, GraphModule, set[str]]],
) -> GraphOperand:
    if not isinstance(operand, GraphExpr):
        return operand
    rewritten = replace(
        operand,
        inputs=tuple(
            _rewrite_specialized_nested_operand(item, replacements=replacements)
            for item in operand.inputs
        ),
        attrs={
            key: _rewrite_specialized_nested_operand(value, replacements=replacements)
            for key, value in operand.attrs.items()
        },
    )
    replacement = replacements.get(rewritten.op.name)
    if replacement is None:
        return rewritten
    clone_name, original, subst_names = replacement
    return _rewrite_call_expr_to_specialized_with_subst(
        rewritten,
        original,
        clone_name,
        subst_names,
    )


def _rewrite_recursive_specialized_node(
    node: GraphNode,
    *,
    originals_by_name: Mapping[str, GraphModule],
    clone_names: Mapping[str, str],
    subst_names_by_module: Mapping[str, set[str]],
) -> GraphNode:
    inputs = tuple(
        _rewrite_recursive_specialized_operand(
            item,
            originals_by_name=originals_by_name,
            clone_names=clone_names,
            subst_names_by_module=subst_names_by_module,
        )
        for item in node.inputs
    )
    attrs = {
        key: _rewrite_recursive_specialized_operand(
            value,
            originals_by_name=originals_by_name,
            clone_names=clone_names,
            subst_names_by_module=subst_names_by_module,
        )
        for key, value in node.attrs.items()
    }
    rewritten = replace(node, inputs=inputs, attrs=attrs)
    clone_name = clone_names.get(rewritten.op.name)
    if clone_name is None:
        return rewritten
    original = originals_by_name[rewritten.op.name]
    subst_names = subst_names_by_module[rewritten.op.name]
    return _rewrite_call_to_specialized_with_subst(
        rewritten,
        original,
        clone_name,
        subst_names,
    )


def _shared_constant_specialization_subst(
    module: GraphModule,
    calls: list[tuple[str, GraphNode | GraphExpr]],
    *,
    global_symbol_names: set[str],
    caller_modules: Mapping[str, GraphModule],
    input_domain_facts: Mapping[str, GraphDomainFact] | None = None,
) -> dict[str, GraphOperand]:
    if not calls:
        return {}
    candidates: dict[int, GraphOperand] = {}
    for index, formal in enumerate(module.inputs):
        actuals: list[GraphOperand] = []
        for caller_name, node in calls:
            if len(node.inputs) != len(module.inputs):
                break
            actual = node.inputs[index]
            if not _specialization_actual_matches_formal(actual, formal):
                break
            if _is_safe_shared_specialization_operand(
                actual,
                global_symbol_names=global_symbol_names,
            ):
                actuals.append(
                    _canonical_specialization_operand(
                        actual,
                        global_symbol_names=global_symbol_names,
                    )
                )
                continue
            traced = _candidate_actual_from_operand(
                actual,
                caller_name=caller_name,
                caller_modules=caller_modules,
                candidates={},
                global_symbol_names=global_symbol_names,
            )
            if traced is not None and _specialization_actual_matches_formal(traced, formal):
                actuals.append(
                    _canonical_specialization_operand(
                        traced,
                        global_symbol_names=global_symbol_names,
                    )
                )
                continue
            domain_actual = _domain_fact_specialization_operand(
                None if input_domain_facts is None else input_domain_facts.get(formal.name),
                formal,
            )
            if domain_actual is not None:
                actuals.append(
                    _canonical_specialization_operand(
                        domain_actual,
                        global_symbol_names=global_symbol_names,
                    )
                )
                continue
            break
        else:
            first = actuals[0]
            if all(_graph_operand_key(actual) == _graph_operand_key(first) for actual in actuals[1:]):
                candidates[index] = first
    return {
        module.inputs[index].name: actual
        for index, actual in candidates.items()
    }


def _external_top_level_scc_calls(
    graph: GraphProgram,
    component: set[str],
) -> list[tuple[str, GraphNode]]:
    calls: list[tuple[str, GraphNode]] = []
    for module in graph.modules:
        for node in module.nodes:
            if node.op.name in component and module.name not in component:
                calls.append((module.name, node))
    return calls


def _internal_top_level_scc_calls(
    graph: GraphProgram,
    component: set[str],
) -> list[tuple[str, GraphNode | GraphExpr]]:
    calls: list[tuple[str, GraphNode | GraphExpr]] = []

    def collect_operand(caller_name: str, operand: GraphOperand) -> None:
        if not isinstance(operand, GraphExpr):
            return
        if operand.op.name in component:
            calls.append((caller_name, operand))
        for item in operand.inputs:
            collect_operand(caller_name, item)
        for item in operand.attrs.values():
            collect_operand(caller_name, item)

    for module in graph.modules:
        if module.name not in component:
            continue
        for node in module.nodes:
            if node.op.name in component:
                calls.append((module.name, node))
            for operand in node.inputs:
                collect_operand(module.name, operand)
            for operand in node.attrs.values():
                collect_operand(module.name, operand)
        for output in module.outputs:
            collect_operand(module.name, output)
    return calls


def _candidate_actual_from_operand(
    operand: GraphOperand,
    *,
    caller_name: str,
    caller_modules: Mapping[str, GraphModule],
    candidates: Mapping[tuple[str, int], GraphOperand],
    global_symbol_names: set[str],
) -> GraphOperand | None:
    if _is_safe_shared_specialization_operand(
        operand,
        global_symbol_names=global_symbol_names,
    ):
        return _canonical_specialization_operand(
            operand,
            global_symbol_names=global_symbol_names,
        )
    if not isinstance(operand, GraphValueRef):
        return None
    caller = caller_modules.get(caller_name)
    if caller is None:
        return None
    for index, formal in enumerate(caller.inputs):
        if formal.name == operand.name:
            return candidates.get((caller_name, index))
    producers = {
        output.name: node
        for node in caller.nodes
        if len(node.outputs) == 1
        for output in node.outputs
    }
    producer = producers.get(operand.name)
    if producer is None:
        return None
    if producer.op.name in {"core.alias", "core.ascribe"} and len(producer.inputs) == 1 and not producer.attrs:
        return _candidate_actual_from_operand(
            producer.inputs[0],
            caller_name=caller_name,
            caller_modules=caller_modules,
            candidates=candidates,
            global_symbol_names=global_symbol_names,
        )
    candidate = GraphExpr(
        op=producer.op,
        inputs=producer.inputs,
        attrs=producer.attrs,
        type_expr=producer.type_expr,
        dims=producer.dims,
    )
    if _is_safe_shared_specialization_operand(
        candidate,
        global_symbol_names=global_symbol_names,
    ):
        return _canonical_specialization_operand(
            candidate,
            global_symbol_names=global_symbol_names,
        )
    return None


def _recursive_scc_specialization_substs(
    graph: GraphProgram,
    component: set[str],
    *,
    global_symbol_names: set[str],
) -> dict[str, dict[str, GraphOperand]]:
    modules_by_name = {module.name: module for module in graph.modules}
    external_calls = _external_top_level_scc_calls(graph, component)
    if len(external_calls) != 1:
        return {}
    entry_caller_name, entry_call = external_calls[0]
    entry = modules_by_name[entry_call.op.name]
    if len(entry_call.inputs) != len(entry.inputs):
        return {}

    candidates: dict[tuple[str, int], GraphOperand] = {}
    for index, actual in enumerate(entry_call.inputs):
        formal = entry.inputs[index]
        if not _specialization_actual_matches_formal(actual, formal):
            continue
        candidate_actual = _candidate_actual_from_operand(
            actual,
            caller_name=entry_caller_name,
            caller_modules=modules_by_name,
            candidates=candidates,
            global_symbol_names=global_symbol_names,
        )
        if candidate_actual is not None and _specialization_actual_matches_formal(candidate_actual, formal):
            candidates[(entry.name, index)] = candidate_actual

    internal_calls = _internal_top_level_scc_calls(graph, component)
    changed = True
    while changed:
        changed = False
        for caller_name, node in internal_calls:
            callee = modules_by_name[node.op.name]
            if len(node.inputs) != len(callee.inputs):
                continue
            for index, actual in enumerate(node.inputs):
                formal = callee.inputs[index]
                if not _specialization_actual_matches_formal(actual, formal):
                    continue
                propagated = _candidate_actual_from_operand(
                    actual,
                    caller_name=caller_name,
                    caller_modules=modules_by_name,
                    candidates=candidates,
                    global_symbol_names=global_symbol_names,
                )
                if propagated is None or not _specialization_actual_matches_formal(propagated, formal):
                    continue
                key = (callee.name, index)
                existing = candidates.get(key)
                if existing is None:
                    candidates[key] = propagated
                    changed = True
                elif _graph_operand_key(existing) != _graph_operand_key(propagated):
                    del candidates[key]
                    changed = True

        for caller_name, node in internal_calls:
            callee = modules_by_name[node.op.name]
            if len(node.inputs) != len(callee.inputs):
                continue
            for index, actual in enumerate(node.inputs):
                key = (callee.name, index)
                expected = candidates.get(key)
                if expected is None:
                    continue
                formal = callee.inputs[index]
                if not _specialization_actual_matches_formal(actual, formal):
                    del candidates[key]
                    changed = True
                    continue
                propagated = _candidate_actual_from_operand(
                    actual,
                    caller_name=caller_name,
                    caller_modules=modules_by_name,
                    candidates=candidates,
                    global_symbol_names=global_symbol_names,
                )
                if (
                    propagated is None
                    or not _specialization_actual_matches_formal(propagated, formal)
                    or _graph_operand_key(propagated) != _graph_operand_key(expected)
                ):
                    del candidates[key]
                    changed = True

    substs: dict[str, dict[str, GraphOperand]] = {}
    for module_name in component:
        module = modules_by_name[module_name]
        module_subst = {
            formal.name: candidates[(module_name, index)]
            for index, formal in enumerate(module.inputs)
            if (module_name, index) in candidates
        }
        if module_subst:
            substs[module_name] = module_subst
    return substs


def _specialize_recursive_sccs(graph: GraphProgram, *, config: GraphOptimizeConfig) -> GraphProgram:
    if config.specialize_definitions == "off":
        return graph
    modules_by_name = {module.name: module for module in graph.modules}
    edges = _graph_call_graph(graph)
    global_symbol_names = {
        module.name
        for module in graph.modules
        if _is_global_symbol_module(module)
    }
    used_module_names = {module.name for module in graph.modules}
    clone_index = 0
    cloned_modules: list[GraphModule] = []
    external_replacements: dict[tuple[str, str], tuple[str, GraphModule, set[str]]] = {}

    for component in _strongly_connected_components(edges):
        if graph.main_module in component:
            continue
        if any(_specialized_module_base(name) is not None for name in component):
            continue
        recursive = len(component) > 1 or any(name in edges.get(name, ()) for name in component)
        if not recursive:
            continue
        substs = _recursive_scc_specialization_substs(
            graph,
            component,
            global_symbol_names=global_symbol_names,
        )
        if not substs:
            continue
        clone_names: dict[str, str] = {}
        for module_name in sorted(component):
            if module_name not in substs:
                continue
            while True:
                clone_index += 1
                clone_name = f"{module_name}__spec_{clone_index}"
                if clone_name not in used_module_names:
                    used_module_names.add(clone_name)
                    clone_names[module_name] = clone_name
                    break
        if not clone_names:
            continue
        specialized_by_original: dict[str, GraphModule] = {}
        for module_name, clone_name in clone_names.items():
            module = modules_by_name[module_name]
            fake_inputs = tuple(
                substs[module_name].get(
                    formal.name,
                    GraphValueRef(formal.name, formal.type_expr, formal.dims),
                )
                for formal in module.inputs
            )
            fake_call = GraphNode(
                id=f"{module_name}:recursive_specialization",
                op=GraphOp(module_name),
                inputs=fake_inputs,
                attrs={},
                outputs=(),
                source_module=module_name,
                type_expr=module.return_type_expr or TypeAny(),
            )
            specialized = _specialized_module(
                module,
                name=clone_name,
                call_node=fake_call,
                subst_override=substs[module_name],
                global_symbol_names=global_symbol_names,
            )
            if specialized is None:
                specialized_by_original.clear()
                break
            specialized_by_original[module_name] = specialized
        if not specialized_by_original:
            continue
        subst_names_by_module = {name: set(subst) for name, subst in substs.items()}
        rewritten_specialized: list[GraphModule] = []
        for module_name, specialized in specialized_by_original.items():
            rewritten_nodes = tuple(
                _rewrite_recursive_specialized_node(
                    node,
                    originals_by_name=modules_by_name,
                    clone_names=clone_names,
                    subst_names_by_module=subst_names_by_module,
                )
                for node in specialized.nodes
            )
            rewritten_outputs = tuple(
                _rewrite_recursive_specialized_operand(
                    output,
                    originals_by_name=modules_by_name,
                    clone_names=clone_names,
                    subst_names_by_module=subst_names_by_module,
                )
                for output in specialized.outputs
            )
            rewritten_specialized.append(
                replace(
                    specialized,
                    nodes=rewritten_nodes,
                    outputs=rewritten_outputs,
                )
            )
        cloned_modules.extend(rewritten_specialized)
        for caller_name, node in _external_top_level_scc_calls(graph, component):
            clone_name = clone_names.get(node.op.name)
            if clone_name is None:
                continue
            external_replacements[(caller_name, node.id)] = (
                clone_name,
                modules_by_name[node.op.name],
                subst_names_by_module[node.op.name],
            )

    if not cloned_modules:
        return graph
    rewritten_modules: list[GraphModule] = []
    for module in graph.modules:
        nodes: list[GraphNode] = []
        for node in module.nodes:
            replacement_info = external_replacements.get((module.name, node.id))
            if replacement_info is None:
                nodes.append(node)
                continue
            clone_name, original, subst_names = replacement_info
            nodes.append(
                _rewrite_call_to_specialized_with_subst(
                    node,
                    original,
                    clone_name,
                    subst_names,
                )
            )
        rewritten_modules.append(replace(module, nodes=tuple(nodes)))
    specialized_graph = replace(graph, modules=tuple((*rewritten_modules, *cloned_modules)))
    specialized_graph = _refresh_graph_program_types(specialized_graph)
    module_effects = infer_graph_module_effects(specialized_graph.modules)
    specialized_modules_by_name = {module.name: module for module in specialized_graph.modules}
    global_literals = _atomic_literal_constants(specialized_graph)
    global_dim_values = _atomic_int_constant_dims(specialized_graph)
    specialized_graph = replace(
        specialized_graph,
        modules=tuple(
            _optimize_module_local(
                module,
                config=config,
                module_effects=module_effects,
                modules_by_name=specialized_modules_by_name,
                global_dim_values=global_dim_values,
                global_literals=global_literals,
            )
            for module in specialized_graph.modules
        ),
    )
    specialized_graph = _sanitize_graph_constraints(specialized_graph)
    _validate_optimizer_graph(specialized_graph, phase="recursive_specialize")
    return specialized_graph


def _specialize_definitions(graph: GraphProgram, *, config: GraphOptimizeConfig) -> GraphProgram:
    if config.specialize_definitions not in _SPECIALIZE_MODES:
        raise ValueError(
            "GraphOptimizeConfig.specialize_definitions must be one of: "
            + ", ".join(sorted(_SPECIALIZE_MODES))
        )
    if config.specialize_definitions == "off":
        return graph
    modules_by_name = {module.name: module for module in graph.modules}
    counts = _call_counts(graph)
    top_level_calls = _top_level_calls_by_callee(graph)
    all_calls = _all_calls_by_callee(graph)
    recursive = _recursive_modules(graph)
    global_symbol_names = {
        module.name
        for module in graph.modules
        if _is_global_symbol_module(module)
    }
    domain_analysis = infer_main_module_domain_facts(graph)
    replacements: dict[tuple[str, str], str] = {}
    replacement_subst_names: dict[tuple[str, str], set[str] | None] = {}
    nested_replacements: dict[str, tuple[str, GraphModule, set[str]]] = {}
    new_modules: list[GraphModule] = list(graph.modules)
    used_module_names = {module.name for module in graph.modules}
    clone_index = 0
    for callee in graph.modules:
        if not _can_specialize_module(
            callee,
            recursive_modules=recursive,
            main_module=graph.main_module,
        ):
            continue
        calls = all_calls.get(callee.name, [])
        if len(calls) < 2:
            continue
        subst = _shared_constant_specialization_subst(
            callee,
            calls,
            global_symbol_names=global_symbol_names,
            caller_modules=modules_by_name,
            input_domain_facts=domain_analysis.module_input_facts.get(callee.name),
        )
        if not subst:
            continue
        while True:
            clone_index += 1
            clone_name = f"{callee.name}__spec_{clone_index}"
            if clone_name not in used_module_names:
                used_module_names.add(clone_name)
                break
        representative = calls[0][1]
        specialized = _specialized_module(
            callee,
            name=clone_name,
            call_node=representative,
            subst_override=subst,
            global_symbol_names=global_symbol_names,
        )
        if specialized is None:
            continue
        if not _specialized_candidate_valid(
            graph,
            tuple((*new_modules, specialized)),
            phase="specialize.candidate.shared",
        ):
            continue
        new_modules.append(specialized)
        subst_names = set(subst)
        for caller_name, node in calls:
            if isinstance(node, GraphNode):
                replacements[(caller_name, node.id)] = clone_name
                replacement_subst_names[(caller_name, node.id)] = subst_names
            else:
                nested_replacements[callee.name] = (clone_name, callee, subst_names)
    for caller in graph.modules:
        for node in caller.nodes:
            if (caller.name, node.id) in replacements:
                continue
            callee = modules_by_name.get(node.op.name)
            if callee is None:
                continue
            if not _can_specialize_module(
                callee,
                recursive_modules=recursive,
                main_module=graph.main_module,
            ):
                continue
            if config.specialize_definitions == "single-callsite" and counts[callee.name] != 1:
                continue
            subst = _callsite_specialization_subst(
                callee,
                node.inputs,
                global_symbol_names=global_symbol_names,
                caller_name=caller.name,
                caller_modules=modules_by_name,
                input_domain_facts=domain_analysis.module_input_facts.get(callee.name),
            )
            if not subst:
                continue
            if not _has_safe_specialization_actual(
                node,
                callee,
                global_symbol_names=global_symbol_names,
                input_domain_facts=domain_analysis.module_input_facts.get(callee.name),
            ):
                continue
            while True:
                clone_index += 1
                clone_name = f"{callee.name}__spec_{clone_index}"
                if clone_name not in used_module_names:
                    used_module_names.add(clone_name)
                    break
            specialized = _specialized_module(
                callee,
                name=clone_name,
                call_node=node,
                subst_override=subst,
                global_symbol_names=global_symbol_names,
            )
            if specialized is None:
                continue
            if not _specialized_candidate_valid(
                graph,
                tuple((*new_modules, specialized)),
                phase="specialize.candidate.callsite",
            ):
                continue
            replacements[(caller.name, node.id)] = clone_name
            replacement_subst_names[(caller.name, node.id)] = set(subst)
            new_modules.append(specialized)
    for caller in graph.modules:
        for node in caller.nodes:
            for operand in (*node.inputs, *node.attrs.values()):
                for callee_name in sorted(_operand_called_module_names(operand, set(modules_by_name))):
                    if callee_name in nested_replacements:
                        continue
                    callee = modules_by_name[callee_name]
                    if not _can_specialize_module(
                        callee,
                        recursive_modules=recursive,
                        main_module=graph.main_module,
                    ):
                        continue
                    if counts[callee.name] != 1:
                        continue
                    expr = _find_operand_call(operand, callee.name)
                    if expr is None or expr.attrs:
                        continue
                    subst = _callsite_specialization_subst(
                        callee,
                        expr.inputs,
                        global_symbol_names=global_symbol_names,
                        caller_name=caller.name,
                        caller_modules=modules_by_name,
                        input_domain_facts=domain_analysis.module_input_facts.get(callee.name),
                    )
                    if not subst:
                        continue
                    while True:
                        clone_index += 1
                        clone_name = f"{callee.name}__spec_{clone_index}"
                        if clone_name not in used_module_names:
                            used_module_names.add(clone_name)
                            break
                    fake_call = GraphNode(
                        id=f"{caller.name}:nested_specialize:{callee.name}",
                        op=GraphOp(callee.name),
                        inputs=expr.inputs,
                        attrs={},
                        outputs=(),
                        source_module=caller.name,
                        type_expr=expr.type_expr,
                        dims=expr.dims,
                    )
                    specialized = _specialized_module(
                        callee,
                        name=clone_name,
                        call_node=fake_call,
                        subst_override=subst,
                        global_symbol_names=global_symbol_names,
                    )
                    if specialized is None:
                        continue
                    if not _specialized_candidate_valid(
                        graph,
                        tuple((*new_modules, specialized)),
                        phase="specialize.candidate.nested",
                    ):
                        continue
                    nested_replacements[callee.name] = (clone_name, callee, set(subst))
                    new_modules.append(specialized)
        for output in caller.outputs:
            for callee_name in sorted(_operand_called_module_names(output, set(modules_by_name))):
                if callee_name in nested_replacements:
                    continue
                callee = modules_by_name[callee_name]
                if not _can_specialize_module(
                    callee,
                    recursive_modules=recursive,
                    main_module=graph.main_module,
                ):
                    continue
                if counts[callee.name] != 1:
                    continue
                expr = _find_operand_call(output, callee.name)
                if expr is None or expr.attrs:
                    continue
                subst = _callsite_specialization_subst(
                    callee,
                    expr.inputs,
                    global_symbol_names=global_symbol_names,
                    caller_name=caller.name,
                    caller_modules=modules_by_name,
                )
                if not subst:
                    continue
                while True:
                    clone_index += 1
                    clone_name = f"{callee.name}__spec_{clone_index}"
                    if clone_name not in used_module_names:
                        used_module_names.add(clone_name)
                        break
                fake_call = GraphNode(
                    id=f"{caller.name}:nested_specialize:{callee.name}",
                    op=GraphOp(callee.name),
                    inputs=expr.inputs,
                    attrs={},
                    outputs=(),
                    source_module=caller.name,
                    type_expr=expr.type_expr,
                    dims=expr.dims,
                )
                specialized = _specialized_module(
                    callee,
                    name=clone_name,
                    call_node=fake_call,
                    subst_override=subst,
                    global_symbol_names=global_symbol_names,
                )
                if specialized is None:
                    continue
                if not _specialized_candidate_valid(
                    graph,
                    tuple((*new_modules, specialized)),
                    phase="specialize.candidate.output",
                ):
                    continue
                nested_replacements[callee.name] = (clone_name, callee, set(subst))
                new_modules.append(specialized)
    if not replacements:
        if not nested_replacements:
            return graph
    original_by_name = modules_by_name
    rewritten_modules: list[GraphModule] = []
    for module in new_modules:
        if module.name not in {item.name for item in graph.modules}:
            rewritten_modules.append(module)
            continue
        nodes: list[GraphNode] = []
        for node in module.nodes:
            clone_name = replacements.get((module.name, node.id))
            if clone_name is not None:
                original = original_by_name[node.op.name]
                nodes.append(
                    _rewrite_call_to_specialized_with_subst(
                        node,
                        original,
                        clone_name,
                        replacement_subst_names.get((module.name, node.id)),
                    )
                )
            else:
                nodes.append(
                    replace(
                        node,
                        inputs=tuple(
                            _rewrite_specialized_nested_operand(
                                item,
                                replacements=nested_replacements,
                            )
                            for item in node.inputs
                        ),
                        attrs={
                            key: _rewrite_specialized_nested_operand(
                                value,
                                replacements=nested_replacements,
                            )
                            for key, value in node.attrs.items()
                        },
                    )
                )
        rewritten_modules.append(
            replace(
                module,
                nodes=tuple(nodes),
                outputs=tuple(
                    _rewrite_specialized_nested_operand(
                        output,
                        replacements=nested_replacements,
                    )
                    for output in module.outputs
                ),
            )
        )
    specialized_graph = replace(graph, modules=tuple(rewritten_modules))
    specialized_graph = _optimize_modules_local_with_fresh_domain_facts(
        specialized_graph,
        config=config,
        phase="specialize.local_cleanup",
    )
    try:
        _validate_optimizer_graph(specialized_graph, phase="specialize")
    except ValueError:
        return graph
    return specialized_graph


def _specialize_definitions_to_fixpoint(
    graph: GraphProgram,
    *,
    config: GraphOptimizeConfig,
) -> GraphProgram:
    current = graph
    for iteration in range(config.max_iterations):
        before = current
        current = _specialize_recursive_sccs(current, config=config)
        _validate_optimizer_graph(current, phase=f"recursive_specialize.fixpoint.{iteration}")
        current = _specialize_definitions(current, config=config)
        _validate_optimizer_graph(current, phase=f"specialize.fixpoint.{iteration}")
        current = _optimize_modules_local_with_fresh_domain_facts(
            current,
            config=config,
            phase=f"specialize.cleanup.{iteration}",
        )
        if current == before:
            return current
    raise RuntimeError(
        f"graph specialization did not converge after {config.max_iterations} iterations"
    )


def _module_value_names(module: GraphModule) -> set[str]:
    names = {value.name for value in module.inputs}
    for node in module.nodes:
        names.update(value.name for value in node.outputs)
    for output in module.outputs:
        _operand_refs(output, names)
    return names


def _operand_has_core_select(operand: GraphOperand) -> bool:
    if not isinstance(operand, GraphExpr):
        return False
    if operand.op.name == "core.select":
        return True
    return any(_operand_has_core_select(item) for item in operand.inputs) or any(
        _operand_has_core_select(value) for value in operand.attrs.values()
    )


def _module_has_core_select(module: GraphModule) -> bool:
    for node in module.nodes:
        if node.op.name == "core.select":
            return True
        if any(_operand_has_core_select(item) for item in node.inputs):
            return True
        if any(_operand_has_core_select(value) for value in node.attrs.values()):
            return True
    return any(_operand_has_core_select(output) for output in module.outputs)


def _select_branches_do_not_use_local_outputs(
    operand: GraphOperand,
    *,
    local_output_names: set[str],
) -> bool:
    if not isinstance(operand, GraphExpr):
        return True
    if operand.op.name == "core.select":
        if len(operand.inputs) != 3:
            return False
        for branch in operand.inputs[1:]:
            refs: set[str] = set()
            _operand_refs(branch, refs)
            if refs & local_output_names:
                return False
    return all(
        _select_branches_do_not_use_local_outputs(item, local_output_names=local_output_names)
        for item in operand.inputs
    ) and all(
        _select_branches_do_not_use_local_outputs(value, local_output_names=local_output_names)
        for value in operand.attrs.values()
    )


def _module_select_branches_do_not_use_local_outputs(module: GraphModule) -> bool:
    local_output_names = {
        output.name
        for node in module.nodes
        for output in node.outputs
    }
    for node in module.nodes:
        if node.op.name == "core.select":
            if len(node.inputs) != 3:
                return False
            for branch in node.inputs[1:]:
                refs: set[str] = set()
                _operand_refs(branch, refs)
                if refs & local_output_names:
                    return False
        if not all(
            _select_branches_do_not_use_local_outputs(item, local_output_names=local_output_names)
            for item in node.inputs
        ):
            return False
        if not all(
            _select_branches_do_not_use_local_outputs(value, local_output_names=local_output_names)
            for value in node.attrs.values()
        ):
            return False
    return all(
        _select_branches_do_not_use_local_outputs(output, local_output_names=local_output_names)
        for output in module.outputs
    )


def _can_inline_module(
    module: GraphModule,
    *,
    module_effects: Mapping[str, GraphEffect],
    recursive_modules: set[str],
    main_module: str,
    allow_control_select: bool,
    allow_multi_node_tensor: bool = False,
) -> bool:
    if module.name == main_module:
        return False
    if module.name in recursive_modules:
        return False
    forwarding = _forwarding_node(module)
    if _module_signature_has_variadic_rows(module) and forwarding is None:
        return False
    if forwarding is not None:
        if forwarding.op.name == "core.select" and not allow_control_select:
            return False
        return True
    forwarding_expr = _forwarding_expr(module)
    if forwarding_expr is not None:
        return not _operand_has_core_select(forwarding_expr)
    if len(module.nodes) > 1 and _module_has_tensor_values(module) and not allow_multi_node_tensor:
        return False
    if module.is_global_binding and not _is_atomic_constant_module(module):
        return False
    if _module_has_core_select(module):
        if not allow_control_select:
            return False
    return _is_non_effectful(module_effects.get(module.name))


def _module_signature_has_variadic_rows(module: GraphModule) -> bool:
    return (
        any(_type_has_variadic_rows(value.type_expr) for value in module.inputs)
        or _type_has_variadic_rows(module.return_type_expr)
        or any(_type_has_variadic_rows(graph_operand_type(output)) for output in module.outputs)
    )


def _type_has_variadic_rows(type_expr: TypeExpr | None) -> bool:
    if isinstance(type_expr, TypeTensor):
        return any(isinstance(dim, str) and dim.startswith("..") for dim in type_expr.dims)
    if isinstance(type_expr, TypeOptional):
        return _type_has_variadic_rows(type_expr.inner)
    if isinstance(type_expr, TypeList):
        return _type_has_variadic_rows(type_expr.item)
    if isinstance(type_expr, TypeTuple):
        return any(_type_has_variadic_rows(item) for item in type_expr.items)
    if isinstance(type_expr, TypeNamed):
        return any(isinstance(dim, str) and dim.startswith("..") for dim in type_expr.args)
    return False


def _is_small_inline_candidate(module: GraphModule, counts: Counter[str]) -> bool:
    return (
        counts[module.name] > 0
        and not module.is_global_binding
        and len(module.nodes) <= _SMALL_INLINE_NODE_LIMIT
        and not _is_atomic_constant_module(module)
    )


def _module_uses_runtime_shape_queries(module: GraphModule) -> bool:
    for node in module.nodes:
        if node.op.name in {"_shape", "_tensor_size"}:
            return True
        for operand in (*node.inputs, *node.attrs.values()):
            if _operand_uses_runtime_shape_queries(operand):
                return True
    for output in module.outputs:
        if _operand_uses_runtime_shape_queries(output):
            return True
    return False


def _module_has_tensor_values(module: GraphModule) -> bool:
    def has_tensor_type(type_expr: TypeExpr | None) -> bool:
        if isinstance(type_expr, TypeTensor):
            return True
        if isinstance(type_expr, TypeOptional):
            return has_tensor_type(type_expr.inner)
        if isinstance(type_expr, TypeList):
            return has_tensor_type(type_expr.item)
        if isinstance(type_expr, TypeTuple):
            return any(has_tensor_type(item) for item in type_expr.items)
        return False

    if any(has_tensor_type(value.type_expr) for value in module.inputs):
        return True
    if has_tensor_type(module.return_type_expr):
        return True
    for node in module.nodes:
        if has_tensor_type(node.type_expr):
            return True
        if any(has_tensor_type(value.type_expr) for value in node.outputs):
            return True
        if any(has_tensor_type(graph_operand_type(operand)) for operand in (*node.inputs, *node.attrs.values())):
            return True
    return any(has_tensor_type(graph_operand_type(output)) for output in module.outputs)


def _operand_uses_runtime_shape_queries(operand: GraphOperand) -> bool:
    if not isinstance(operand, GraphExpr):
        return False
    if operand.op.name in {"_shape", "_tensor_size"}:
        return True
    return any(_operand_uses_runtime_shape_queries(item) for item in operand.inputs) or any(
        _operand_uses_runtime_shape_queries(item) for item in operand.attrs.values()
    )


def _is_atomic_constant_module(module: GraphModule) -> bool:
    return (
        not module.inputs
        and not module.nodes
        and len(module.outputs) == 1
        and _is_atomic_operand(module.outputs[0])
    )


def _is_global_symbol_module(module: GraphModule) -> bool:
    return not module.inputs and len(module.outputs) == 1


def _promote_total_pure_zero_arg_modules_to_globals(
    graph: GraphProgram,
    *,
    module_effects: Mapping[str, GraphEffect],
    module_usages: Mapping[str, UsageClass],
) -> GraphProgram:
    modules: list[GraphModule] = []
    changed = False
    for module in graph.modules:
        should_promote = (
            module.name != graph.main_module
            and not module.is_global_binding
            and not module.inputs
            and len(module.outputs) == 1
            and module_effects.get(module.name) == GraphEffect.TOTAL_PURE
            and module_usages.get(module.name) == UsageClass.UNRESTRICTED
        )
        if should_promote:
            modules.append(replace(module, is_global_binding=True))
            changed = True
        else:
            modules.append(module)
    if not changed:
        return graph
    promoted = replace(graph, modules=tuple(modules))
    _validate_optimizer_graph(promoted, phase="promote_zero_arg_globals")
    return promoted


def _inline_call_substitution_is_closed(
    callee: GraphModule,
    actuals: tuple[GraphOperand, ...],
) -> bool:
    formal_subst = {
        formal.name: actual
        for formal, actual in zip(callee.inputs, actuals, strict=True)
    }
    dim_subst = _call_dim_subst(callee, actuals)
    allowed_refs: set[str] = set()
    for actual in actuals:
        _operand_refs(actual, allowed_refs)
        for dim in _type_dims(graph_operand_type(actual)) or ():
            if isinstance(dim, str):
                allowed_refs.add(dim)
            else:
                allowed_refs.update(str(name) for name in dim_token_names(dim))
    for dim in dim_subst.values():
        if isinstance(dim, str):
            allowed_refs.add(dim)
        else:
            allowed_refs.update(str(name) for name in dim_token_names(dim))
    local_refs = {
        output.name
        for node in callee.nodes
        for output in node.outputs
    }

    def rewritten_refs(operand: GraphOperand) -> set[str]:
        rewritten = _replace_operand_refs(
            operand,
            formal_subst,
            fold=True,
        )
        rewritten = substitute_graph_operand_dims(rewritten, dim_subst) if dim_subst else rewritten
        refs: set[str] = set()
        _operand_refs(rewritten, refs)
        return refs

    for node in callee.nodes:
        for operand in (*node.inputs, *node.attrs.values()):
            if not rewritten_refs(operand) <= allowed_refs | local_refs:
                return False
    for output in callee.outputs:
        if not rewritten_refs(output) <= allowed_refs | local_refs:
            return False
    return True


def _can_inline_call_node(node: GraphNode, callee: GraphModule) -> bool:
    if len(node.inputs) != len(callee.inputs):
        return False
    if len(node.outputs) != len(callee.outputs):
        if not (
            len(callee.outputs) == 1
            and isinstance(graph_operand_type(callee.outputs[0]), TypeTuple)
            and len(graph_operand_type(callee.outputs[0]).items) == len(node.outputs)
        ):
            return False
    if node.attrs:
        return False
    for actual, formal in zip(node.inputs, callee.inputs, strict=True):
        if formal.optional and isinstance(graph_operand_type(actual), TypeNull):
            continue
        if not graph_type_compatible(graph_operand_type(actual), formal.type_expr):
            return False
    expected_output_types = _instantiate_call_output_types(
        callee,
        node.inputs,
        len(node.outputs),
    )
    if len(expected_output_types) != len(node.outputs):
        return False
    for call_output, expected in zip(node.outputs, expected_output_types, strict=True):
        if not (
            graph_type_compatible(expected, call_output.type_expr)
            or graph_type_compatible(call_output.type_expr, expected)
        ):
            return False
    if not _inline_call_substitution_is_closed(callee, node.inputs):
        return False
    return True


def _can_inline_forwarded_call_node(node: GraphNode, callee: GraphModule, inner: GraphNode) -> bool:
    if len(node.inputs) != len(callee.inputs):
        return False
    if node.attrs:
        return False
    for actual, formal in zip(node.inputs, callee.inputs, strict=True):
        if formal.optional and isinstance(graph_operand_type(actual), TypeNull):
            continue
        if not graph_type_compatible(graph_operand_type(actual), formal.type_expr):
            return False
    forwarded = _rewrite_forwarded_call_node(node, callee, inner, fold=True)
    # A forwarded one-op module can have a surface signature that is less
    # precise than the primitive/type rule it forwards to.  In that case the
    # call node's existing metadata may already be stale (for example a
    # broadcast wrapper preserving Tensor[B,1,1,K] instead of the primitive's
    # Tensor[B,1,Q,K]).  Trust the rewritten forwarded op and let the graph type
    # refresh propagate its inferred output types.
    forwarded_output_types = _forwarded_node_output_types(forwarded, len(node.outputs))
    if _module_signature_has_variadic_rows(callee):
        return (
            len(node.outputs) > 1
            and
            forwarded_output_types is not None
            and len(forwarded_output_types) == len(node.outputs)
            and not any(_type_has_variadic_rows(output_type) for output_type in forwarded_output_types)
            and _inline_call_substitution_is_closed(callee, node.inputs)
        )
    if forwarded_output_types is not None and len(forwarded_output_types) == len(node.outputs):
        return _inline_call_substitution_is_closed(callee, node.inputs)
    expected_output_types = _instantiate_call_output_types(
        callee,
        node.inputs,
        len(node.outputs),
    )
    if len(expected_output_types) == len(node.outputs) and all(
        graph_type_compatible(expected, output.type_expr)
        or graph_type_compatible(output.type_expr, expected)
        for expected, output in zip(expected_output_types, node.outputs, strict=True)
    ):
        return True
    return (
        _inline_call_substitution_is_closed(callee, node.inputs)
        and (
            graph_type_compatible(forwarded.type_expr, node.type_expr)
            or graph_type_compatible(
                node.type_expr,
                forwarded.type_expr,
            )
        )
    )


def _forwarded_node_outputs_compatible(
    forwarded: GraphNode,
    original_outputs: tuple[GraphValue, ...],
) -> bool:
    output_types = _forwarded_node_output_types(forwarded, len(original_outputs))
    if output_types is None:
        return True
    if len(output_types) != len(original_outputs):
        return False
    for output, inferred in zip(original_outputs, output_types, strict=True):
        if output.type_expr == inferred:
            continue
        if not (
            graph_type_compatible(inferred, output.type_expr)
            or graph_type_compatible(output.type_expr, inferred)
        ):
            return False
    return True


def _forwarded_node_output_types(
    forwarded: GraphNode,
    output_count: int,
) -> tuple[TypeExpr, ...] | None:
    primitive_type = _infer_primitive_graph_type(
        forwarded.op.name,
        forwarded.inputs,
        forwarded.attrs,
        dim_values={},
    )
    if primitive_type is None or isinstance(primitive_type, TypeAny):
        return None
    output_types = (
        _destructured_list_output_types(primitive_type, output_count)
        if isinstance(primitive_type, TypeList)
        else None
    )
    if output_types is None:
        output_types = _result_types(primitive_type, output_count)
    return output_types


def _can_inline_call_expr(expr: GraphExpr, callee: GraphModule) -> bool:
    if len(expr.inputs) != len(callee.inputs):
        return False
    if len(callee.outputs) != 1:
        return False
    if expr.attrs:
        return False
    for actual, formal in zip(expr.inputs, callee.inputs, strict=True):
        if formal.optional and isinstance(graph_operand_type(actual), TypeNull):
            continue
        if not graph_type_compatible(graph_operand_type(actual), formal.type_expr):
            return False
    expected_output_types = _instantiate_call_output_types(callee, expr.inputs, 1)
    if len(expected_output_types) != 1:
        return False
    expected = expected_output_types[0]
    if expr.type_expr != expected:
        return False
    if not _inline_call_substitution_is_closed(callee, expr.inputs):
        return False
    return True


def _can_inline_forwarded_call_expr(expr: GraphExpr, callee: GraphModule, inner: GraphNode) -> bool:
    if len(expr.inputs) != len(callee.inputs):
        return False
    if expr.attrs:
        return False
    for actual, formal in zip(expr.inputs, callee.inputs, strict=True):
        if formal.optional and isinstance(graph_operand_type(actual), TypeNull):
            continue
        if not graph_type_compatible(graph_operand_type(actual), formal.type_expr):
            return False
    if not _inline_call_substitution_is_closed(callee, expr.inputs):
        return False
    if len(callee.outputs) == 1 and graph_type_compatible(graph_operand_type(callee.outputs[0]), expr.type_expr):
        return True
    forwarded = _rewrite_forwarded_call_expr(expr, callee, inner, fold=True)
    return graph_type_compatible(graph_operand_type(forwarded), expr.type_expr) or graph_type_compatible(
        expr.type_expr,
        graph_operand_type(forwarded),
    )


def _can_inline_forwarder_inside_lazy_branch(
    callee: GraphModule,
    *,
    module_effects: Mapping[str, GraphEffect],
    module_usages: Mapping[str, UsageClass],
) -> bool:
    return (
        _is_non_effectful(module_effects.get(callee.name))
        and module_usages.get(callee.name) in {UsageClass.UNRESTRICTED, UsageClass.AFFINE}
    )


def _is_single_callsite_inline_candidate(
    module: GraphModule,
    counts: Counter[str],
    top_level_counts: Counter[str],
) -> bool:
    return (
        counts[module.name] == 1
        and top_level_counts[module.name] <= 1
        and not _is_atomic_constant_module(module)
        and (
            not module.is_global_binding
            or _forwarding_expr(module) is not None
        )
    )


def _forwarding_node(module: GraphModule) -> GraphNode | None:
    if module.is_global_binding:
        return None
    if len(module.nodes) != 1:
        return None
    inner = module.nodes[0]
    if len(module.outputs) != len(inner.outputs):
        return None
    for returned, output in zip(module.outputs, inner.outputs, strict=True):
        if not isinstance(returned, GraphValueRef) or returned.name != output.name:
            return None
    return inner


def _forwarding_expr(module: GraphModule) -> GraphExpr | None:
    if module.is_global_binding:
        return None
    if module.nodes or len(module.outputs) != 1:
        return None
    output = module.outputs[0]
    if isinstance(output, GraphExpr):
        return output
    return None


def _rewrite_inlined_node(
    inner: GraphNode,
    *,
    module_name: str,
    node_id: str,
    renames: Mapping[str, str],
    formal_subst: Mapping[str, GraphOperand],
    dim_subst: Mapping[str, DimToken],
    fold: bool,
) -> GraphNode:
    renamed_outputs = tuple(
        replace(output, name=renames.get(output.name, output.name))
        for output in inner.outputs
    )
    rewritten = replace(
        inner,
        id=node_id,
        inputs=tuple(
            _replace_operand_refs(
                rename_operand(item, renames),
                formal_subst,
                fold=fold,
            )
            for item in inner.inputs
        ),
        attrs={
            key: _replace_operand_refs(
                rename_operand(value, renames),
                formal_subst,
                fold=fold,
            )
            for key, value in inner.attrs.items()
        },
        outputs=renamed_outputs,
        source_module=module_name,
    )
    return substitute_graph_node_dims(rewritten, dim_subst) if dim_subst else rewritten


def _inlined_dim_subst(
    callee: GraphModule,
    *,
    renames: Mapping[str, str],
    dim_subst: Mapping[str, DimToken],
    used_names: set[str],
) -> dict[str, DimToken]:
    """Dimension substitution for an inlined callee.

    Inlined value names are freshened to avoid collisions in the caller.  Some
    local values are also first-class dimension/int values and may be referenced
    from later type annotations.  Those type-level references must be renamed
    with the value; otherwise stale local names can later be rebound to
    unrelated caller temporaries during type refresh.
    """

    combined = dict(dim_subst)
    for node in callee.nodes:
        for output in node.outputs:
            if output.name not in renames:
                continue
            output_type = output.type_expr
            if isinstance(output_type, TypeOptional):
                output_type = output_type.inner
            if isinstance(output_type, TypeDim | TypeInt):
                combined.setdefault(output.name, renames[output.name])
    input_dim_symbols = _module_input_dim_symbols(callee)
    free_symbols = _collect_module_free_symbols(callee)
    value_level_names = free_symbols.value_refs | free_symbols.term_dim_refs | set(callee.output_names)
    for dim_name in sorted(_module_dim_refs(callee)):
        if (
            dim_name in combined
            or dim_name in input_dim_symbols
            or dim_name in value_level_names
            or not _is_plain_dim_symbol(dim_name)
        ):
            continue
        fresh_name = _fresh_graph_value_name(
            used_names,
            f"__dim_{callee.name.replace('.', '_')}__{dim_name}",
        )
        combined[dim_name] = fresh_name
    return combined


def _can_inline_with_dim_subst(
    callee: GraphModule,
    *,
    dim_subst: Mapping[str, DimToken],
    caller_dim_refs: set[str],
) -> bool:
    """Reject inline candidates that would leak callee-local type symbols.

    A callee input signature may use a symbolic dimension that does not bind to
    a caller-visible symbol at a specific call site.  Inlining that body would
    move the callee symbol into the caller module, where graph validation must
    reject it as an undefined type/value reference.  Skip those candidates
    before building and validating large invalid graphs.
    """

    for dim_name in _module_dim_refs(callee):
        if (
            _is_plain_dim_symbol(dim_name)
            and dim_name not in dim_subst
            and dim_name not in caller_dim_refs
        ):
            return False
    return True


def _rewrite_forwarded_call_node(
    node: GraphNode,
    callee: GraphModule,
    inner: GraphNode,
    *,
    fold: bool,
) -> GraphNode:
    formal_subst = {
        formal.name: actual
        for formal, actual in zip(callee.inputs, node.inputs, strict=True)
    }
    dim_subst = _call_node_dim_subst(callee, node)
    rewritten = replace(
        node,
        op=inner.op,
        inputs=tuple(
            _replace_operand_refs(item, formal_subst, fold=fold)
            for item in inner.inputs
        ),
        attrs={
            key: _replace_operand_refs(value, formal_subst, fold=fold)
            for key, value in inner.attrs.items()
        },
    )
    rewritten = substitute_graph_node_dims(rewritten, dim_subst) if dim_subst else rewritten
    output_types = _forwarded_node_output_types(rewritten, len(node.outputs))
    outputs = node.outputs
    type_expr = node.type_expr
    dims = node.dims
    if output_types is not None and len(output_types) == len(node.outputs):
        outputs = tuple(
            replace(
                output,
                type_expr=output_type,
                dims=output_type.dims if isinstance(output_type, TypeTensor) else output.dims,
            )
            for output, output_type in zip(node.outputs, output_types, strict=True)
        )
        type_expr = output_types[0] if len(output_types) == 1 else TypeTuple(output_types)
        dims = output_types[0].dims if len(output_types) == 1 and isinstance(output_types[0], TypeTensor) else dims
    return replace(
        rewritten,
        outputs=outputs,
        type_expr=type_expr,
        dims=dims,
        source_module=node.source_module,
    )


def _rewrite_forwarded_call_expr(
    expr: GraphExpr,
    callee: GraphModule,
    inner: GraphNode,
    *,
    fold: bool,
) -> GraphExpr:
    formal_subst = {
        formal.name: actual
        for formal, actual in zip(callee.inputs, expr.inputs, strict=True)
    }
    dim_subst = _call_dim_subst(callee, expr.inputs)
    rewritten = GraphExpr(
        op=inner.op,
        inputs=tuple(
            _replace_operand_refs(item, formal_subst, fold=fold)
            for item in inner.inputs
        ),
        attrs={
            key: _replace_operand_refs(value, formal_subst, fold=fold)
            for key, value in inner.attrs.items()
        },
        type_expr=expr.type_expr,
        dims=expr.dims,
    )
    rewritten = substitute_graph_operand_dims(rewritten, dim_subst) if dim_subst else rewritten
    return replace(
        rewritten,
        type_expr=expr.type_expr,
        dims=expr.dims,
    )


def _rewrite_forwarded_expr_call_node(
    node: GraphNode,
    callee: GraphModule,
    expr: GraphExpr,
    *,
    fold: bool,
) -> GraphNode:
    formal_subst = {
        formal.name: actual
        for formal, actual in zip(callee.inputs, node.inputs, strict=True)
    }
    dim_subst = _call_node_dim_subst(callee, node)
    rewritten_expr = _replace_operand_refs(expr, formal_subst, fold=fold)
    rewritten_expr = substitute_graph_operand_dims(rewritten_expr, dim_subst) if dim_subst else rewritten_expr
    if not isinstance(rewritten_expr, GraphExpr):
        return replace(
            node,
            op=GraphOp("core.alias"),
            inputs=(rewritten_expr,),
            attrs={},
            outputs=node.outputs,
            type_expr=node.type_expr,
            dims=node.dims,
            source_module=node.source_module,
        )
    return replace(
        node,
        op=rewritten_expr.op,
        inputs=rewritten_expr.inputs,
        attrs=rewritten_expr.attrs,
        outputs=node.outputs,
        type_expr=node.type_expr,
        dims=node.dims,
        source_module=node.source_module,
    )


def _rewrite_forwarded_expr_call_expr(
    expr: GraphExpr,
    callee: GraphModule,
    forwarded: GraphExpr,
    *,
    fold: bool,
) -> GraphExpr:
    formal_subst = {
        formal.name: actual
        for formal, actual in zip(callee.inputs, expr.inputs, strict=True)
    }
    dim_subst = _call_dim_subst(callee, expr.inputs)
    rewritten = _replace_operand_refs(forwarded, formal_subst, fold=fold)
    rewritten = substitute_graph_operand_dims(rewritten, dim_subst) if dim_subst else rewritten
    if not isinstance(rewritten, GraphExpr):
        return GraphExpr(
            op=GraphOp("core.alias"),
            inputs=(rewritten,),
            attrs={},
            type_expr=expr.type_expr,
            dims=expr.dims,
        )
    return replace(
        rewritten,
        type_expr=expr.type_expr,
        dims=expr.dims,
    )


def _rewrite_inlined_return(
    returned: GraphOperand,
    *,
    renames: Mapping[str, str],
    formal_subst: Mapping[str, GraphOperand],
    dim_subst: Mapping[str, DimToken],
    fold: bool,
) -> GraphOperand:
    rewritten = _replace_operand_refs(
        rename_operand(returned, renames),
        formal_subst,
        fold=fold,
    )
    return substitute_graph_operand_dims(rewritten, dim_subst) if dim_subst else rewritten


def _inlined_constraints(
    callee: GraphModule,
    *,
    renames: Mapping[str, str],
    formal_subst: Mapping[str, GraphOperand],
) -> tuple[Constraint, ...] | None:
    if not callee.constraints:
        return ()
    subst: dict[str, GraphOperand] = dict(formal_subst)
    subst.update(
        {
            old: GraphValueRef(name=new, type_expr=TypeAny())
            for old, new in renames.items()
            if old != new
        }
    )
    return _specialize_constraints(callee.constraints, subst)


def _inline_safe_modules(graph: GraphProgram, *, config: GraphOptimizeConfig) -> GraphProgram:
    counts = _call_counts(graph)
    top_level_counts = _top_level_call_counts(graph)
    modules_by_name = {module.name: module for module in graph.modules}
    module_effects = infer_graph_module_effects(graph.modules)
    module_usages = infer_graph_module_usages(graph.modules)
    recursive = _recursive_modules(graph)
    single_callsite_candidates = {
        module.name
        for module in graph.modules
        if _is_single_callsite_inline_candidate(module, counts, top_level_counts)
    }
    small_inline_candidates = {
        module.name
        for module in graph.modules
        if _is_small_inline_candidate(module, counts)
    }
    inlineable = {
        module.name: module
        for module in graph.modules
        if (
            _forwarding_node(module) is not None
            or _forwarding_expr(module) is not None
            or module.name in single_callsite_candidates
            or module.name in small_inline_candidates
            or (
                _is_atomic_constant_module(module)
                and config.constant_dim_substitution
            )
        )
        and _can_inline_module(
            module,
            module_effects=module_effects,
            recursive_modules=recursive,
            main_module=graph.main_module,
            allow_control_select=True,
            allow_multi_node_tensor=(
                module.name in single_callsite_candidates
                or module.name in small_inline_candidates
            ),
        )
    }
    if not inlineable:
        return graph
    rewritten_modules: list[GraphModule] = []
    for module in graph.modules:
        nodes: list[GraphNode] = []
        subst: dict[str, GraphOperand] = {}
        constraints: list[Constraint] = list(module.constraints)
        temp_index = 0
        used_names = _module_value_names(module)
        caller_dim_refs = _module_dim_refs(module) | _module_dim_value_names(module)
        local_conditions: dict[str, GraphExpr] = {}

        def _inline_expr_call(expr: GraphExpr, *, prefix: str) -> GraphOperand:
            nonlocal temp_index
            callee = inlineable.get(expr.op.name)
            if callee is None or not _can_inline_call_expr(expr, callee):
                return expr
            formal_subst = {
                formal.name: actual
                for formal, actual in zip(callee.inputs, expr.inputs, strict=True)
            }
            dim_subst = _call_dim_subst(callee, expr.inputs)
            renames: dict[str, str] = {}
            for inner in callee.nodes:
                for output in inner.outputs:
                    while True:
                        temp_index += 1
                        candidate = f"{prefix}__inl_{temp_index}_{output.name}"
                        if candidate not in used_names:
                            used_names.add(candidate)
                            renames[output.name] = candidate
                            break
            inline_dim_subst = _inlined_dim_subst(
                callee,
                renames=renames,
                dim_subst=dim_subst,
                used_names=used_names,
            )
            if not _can_inline_with_dim_subst(callee, dim_subst=inline_dim_subst, caller_dim_refs=caller_dim_refs):
                return expr
            inlined_constraints = _inlined_constraints(
                callee,
                renames=renames,
                formal_subst=formal_subst,
            )
            if inlined_constraints is None:
                return expr
            for inner in callee.nodes:
                nodes.append(
                    _rewrite_inlined_node(
                        inner,
                        module_name=module.name,
                        node_id=f"{module.name}:inl:output:{expr.op.name}:{inner.id}",
                        renames=renames,
                        formal_subst=formal_subst,
                        dim_subst=inline_dim_subst,
                        fold=config.constant_folding,
                    )
                )
            constraints.extend(inlined_constraints)
            return _rewrite_inlined_return(
                callee.outputs[0],
                renames=renames,
                formal_subst=formal_subst,
                dim_subst=inline_dim_subst,
                fold=config.constant_folding,
            )

        def _inline_nested_expr_calls(
            operand: GraphOperand,
            *,
            prefix: str,
            allow_general_inline: bool = True,
            expected_type: TypeExpr | None = None,
            local_domain_facts: Mapping[str, GraphDomainFact] | None = None,
        ) -> GraphOperand:
            if local_domain_facts:
                operand = _refine_operand_types_from_domain_facts(operand, local_domain_facts)
            if not isinstance(operand, GraphExpr):
                return operand
            forwarding_callee = inlineable.get(operand.op.name)
            forwarding = _forwarding_node(forwarding_callee) if forwarding_callee is not None else None
            forwarded_expr = _forwarding_expr(forwarding_callee) if forwarding_callee is not None else None
            allow_forwarding_inline = allow_general_inline or (
                forwarding_callee is not None
                and _can_inline_forwarder_inside_lazy_branch(
                    forwarding_callee,
                    module_effects=module_effects,
                    module_usages=module_usages,
                )
            )
            if (
                forwarding_callee is not None
                and forwarding is not None
                and allow_forwarding_inline
                and _can_inline_forwarded_call_expr(operand, forwarding_callee, forwarding)
            ):
                dim_subst = _call_dim_subst(forwarding_callee, operand.inputs)
                if not _can_inline_with_dim_subst(
                    forwarding_callee,
                    dim_subst=dim_subst,
                    caller_dim_refs=caller_dim_refs,
                ):
                    return operand
                forwarded = _rewrite_forwarded_call_expr(
                    operand,
                    forwarding_callee,
                    forwarding,
                    fold=config.constant_folding,
                )
                if allow_general_inline or graph_type_compatible(
                    graph_operand_type(forwarded),
                    expected_type or operand.type_expr,
                ):
                    return forwarded
            if (
                forwarding_callee is not None
                and forwarded_expr is not None
                and allow_forwarding_inline
                and _can_inline_call_expr(operand, forwarding_callee)
            ):
                dim_subst = _call_dim_subst(forwarding_callee, operand.inputs)
                if not _can_inline_with_dim_subst(
                    forwarding_callee,
                    dim_subst=dim_subst,
                    caller_dim_refs=caller_dim_refs,
                ):
                    return operand
                forwarded = _rewrite_forwarded_expr_call_expr(
                    operand,
                    forwarding_callee,
                    forwarded_expr,
                    fold=config.constant_folding,
                )
                if allow_general_inline or graph_type_compatible(
                    graph_operand_type(forwarded),
                    expected_type or operand.type_expr,
                ):
                    return forwarded
            if operand.op.name == "core.select" and len(operand.inputs) == 3 and not operand.attrs:
                true_domain_facts = refine_graph_domain_facts_for_branch(
                    operand.inputs[0],
                    True,
                    local_domain_facts or {},
                    local_conditions,
                )
                false_domain_facts = refine_graph_domain_facts_for_branch(
                    operand.inputs[0],
                    False,
                    local_domain_facts or {},
                    local_conditions,
                )
                return replace(
                    operand,
                    inputs=(
                        _inline_nested_expr_calls(
                            operand.inputs[0],
                            prefix=f"{prefix}_cond",
                            allow_general_inline=allow_general_inline,
                            local_domain_facts=local_domain_facts,
                        ),
                        _inline_nested_expr_calls(
                            operand.inputs[1],
                            prefix=f"{prefix}_then",
                            allow_general_inline=False,
                            expected_type=operand.type_expr,
                            local_domain_facts=true_domain_facts,
                        ),
                        _inline_nested_expr_calls(
                            operand.inputs[2],
                            prefix=f"{prefix}_else",
                            allow_general_inline=False,
                            expected_type=operand.type_expr,
                            local_domain_facts=false_domain_facts,
                        ),
                    ),
                )
            rewritten = replace(
                operand,
                inputs=tuple(
                    _inline_nested_expr_calls(
                        item,
                        prefix=f"{prefix}_arg{index + 1}",
                        allow_general_inline=allow_general_inline,
                        local_domain_facts=local_domain_facts,
                    )
                    for index, item in enumerate(operand.inputs)
                ),
                attrs={
                    key: _inline_nested_expr_calls(
                        value,
                        prefix=f"{prefix}_{key}",
                        allow_general_inline=allow_general_inline,
                        local_domain_facts=local_domain_facts,
                    )
                    for key, value in operand.attrs.items()
                },
            )
            if not allow_general_inline:
                return rewritten
            return _inline_expr_call(rewritten, prefix=prefix)

        for node in module.nodes:
            node = _rewrite_node_operands(node, subst, fold=config.constant_folding)
            if node.op.name == "core.select" and len(node.inputs) == 3 and not node.attrs:
                node = replace(
                    node,
                    inputs=(
                        _inline_nested_expr_calls(
                            node.inputs[0],
                            prefix=f"{node.outputs[0].name}_cond",
                        ),
                        _inline_nested_expr_calls(
                            node.inputs[1],
                            prefix=f"{node.outputs[0].name}_then",
                            allow_general_inline=False,
                            expected_type=node.type_expr,
                        ),
                        _inline_nested_expr_calls(
                            node.inputs[2],
                            prefix=f"{node.outputs[0].name}_else",
                            allow_general_inline=False,
                            expected_type=node.type_expr,
                        ),
                    ),
                )
            else:
                node = replace(
                    node,
                    inputs=tuple(
                        _inline_nested_expr_calls(
                            item,
                            prefix=f"{node.outputs[0].name}_arg{index + 1}",
                        )
                        for index, item in enumerate(node.inputs)
                    ),
                    attrs={
                        key: _inline_nested_expr_calls(
                            value,
                            prefix=f"{node.outputs[0].name}_{key}",
                        )
                        for key, value in node.attrs.items()
                    },
                )
            if (
                len(node.outputs) == 1
                and node.op.name.startswith("core.binary.")
                and len(node.inputs) == 2
                and not node.attrs
            ):
                local_conditions[node.outputs[0].name] = GraphExpr(
                    op=node.op,
                    inputs=node.inputs,
                    attrs=node.attrs,
                    type_expr=node.type_expr,
                    dims=node.dims,
                )
            callee = inlineable.get(node.op.name)
            if callee is None:
                nodes.append(node)
                continue
            forwarding = _forwarding_node(callee)
            if (
                forwarding is not None
                and _can_inline_forwarded_call_node(node, callee, forwarding)
            ):
                dim_subst = _call_node_dim_subst(callee, node)
                if not _can_inline_with_dim_subst(callee, dim_subst=dim_subst, caller_dim_refs=caller_dim_refs):
                    nodes.append(node)
                    continue
                inlined_constraints = _inlined_constraints(
                    callee,
                    renames={
                        output.name: call_output.name
                        for output, call_output in zip(forwarding.outputs, node.outputs, strict=False)
                    },
                    formal_subst={
                        formal.name: actual
                        for formal, actual in zip(callee.inputs, node.inputs, strict=True)
                    },
                )
                if inlined_constraints is not None:
                    forwarded_node = _rewrite_forwarded_call_node(
                        node,
                        callee,
                        forwarding,
                        fold=config.constant_folding,
                    )
                    nodes.append(forwarded_node)
                    for output in forwarded_node.outputs:
                        subst[output.name] = GraphValueRef(
                            output.name,
                            output.type_expr,
                            output.dims,
                        )
                    constraints.extend(inlined_constraints)
                    continue
            forwarded_expr = _forwarding_expr(callee)
            if (
                forwarded_expr is not None
                and _can_inline_call_node(node, callee)
            ):
                dim_subst = _call_node_dim_subst(callee, node)
                if not _can_inline_with_dim_subst(callee, dim_subst=dim_subst, caller_dim_refs=caller_dim_refs):
                    nodes.append(node)
                    continue
                inlined_constraints = _inlined_constraints(
                    callee,
                    renames={},
                    formal_subst={
                        formal.name: actual
                        for formal, actual in zip(callee.inputs, node.inputs, strict=True)
                    },
                )
                if inlined_constraints is not None:
                    forwarded_node = _rewrite_forwarded_expr_call_node(
                        node,
                        callee,
                        forwarded_expr,
                        fold=config.constant_folding,
                    )
                    nodes.append(forwarded_node)
                    for output in forwarded_node.outputs:
                        subst[output.name] = GraphValueRef(
                            output.name,
                            output.type_expr,
                            output.dims,
                        )
                    constraints.extend(inlined_constraints)
                    continue
            if _module_signature_has_variadic_rows(callee):
                nodes.append(node)
                continue
            if not _can_inline_call_node(node, callee):
                nodes.append(node)
                continue
            formal_subst = {formal.name: actual for formal, actual in zip(callee.inputs, node.inputs, strict=True)}
            dim_subst = _call_node_dim_subst(callee, node)
            renames: dict[str, str] = {}
            for inner in callee.nodes:
                for output in inner.outputs:
                    while True:
                        temp_index += 1
                        candidate = f"{node.outputs[0].name}__inl_{temp_index}_{output.name}"
                        if candidate not in used_names:
                            used_names.add(candidate)
                            renames[output.name] = candidate
                            break
            inline_dim_subst = _inlined_dim_subst(
                callee,
                renames=renames,
                dim_subst=dim_subst,
                used_names=used_names,
            )
            if not _can_inline_with_dim_subst(callee, dim_subst=inline_dim_subst, caller_dim_refs=caller_dim_refs):
                nodes.append(node)
                continue
            inlined_constraints = _inlined_constraints(
                callee,
                renames=renames,
                formal_subst=formal_subst,
            )
            if inlined_constraints is None:
                nodes.append(node)
                continue
            for inner in callee.nodes:
                nodes.append(
                    _rewrite_inlined_node(
                        inner,
                        module_name=module.name,
                        node_id=f"{module.name}:inl:{node.id}:{inner.id}",
                        renames=renames,
                        formal_subst=formal_subst,
                        dim_subst=inline_dim_subst,
                        fold=config.constant_folding,
                    )
                )
            constraints.extend(inlined_constraints)
            if len(callee.outputs) == len(node.outputs):
                for output, returned in zip(node.outputs, callee.outputs, strict=True):
                    subst[output.name] = _rewrite_inlined_return(
                        returned,
                        renames=renames,
                        formal_subst=formal_subst,
                        dim_subst=inline_dim_subst,
                        fold=config.constant_folding,
                    )
                continue
            if len(callee.outputs) == 1:
                returned = _rewrite_inlined_return(
                    callee.outputs[0],
                    renames=renames,
                    formal_subst=formal_subst,
                    dim_subst=inline_dim_subst,
                    fold=config.constant_folding,
                )
                alias_type = TypeTuple(tuple(output.type_expr for output in node.outputs))
                nodes.append(
                    GraphNode(
                        id=f"{module.name}:inl:{node.id}:destructure",
                        op=GraphOp("core.ascribe"),
                        inputs=(returned,),
                        attrs={},
                        outputs=node.outputs,
                        source_module=module.name,
                        type_expr=alias_type,
                    )
                )
                for output in node.outputs:
                    subst[output.name] = GraphValueRef(
                        output.name,
                        output.type_expr,
                        output.dims,
                    )
                continue
            nodes.append(node)
        outputs = tuple(
            _inline_nested_expr_calls(
                _replace_operand_refs(
                    output,
                    subst,
                    fold=config.constant_folding,
                ),
                prefix=f"__out_{index + 1}",
            )
            for index, output in enumerate(module.outputs)
        )
        rewritten_modules.append(
            replace(
                module,
                nodes=tuple(nodes),
                outputs=outputs,
                constraints=tuple(constraints),
            )
        )
    inlined = _alpha_rename_shadowed_type_dims(replace(graph, modules=tuple(rewritten_modules)))
    inlined = _refresh_graph_program_types(inlined)
    inlined = _alpha_rename_shadowed_type_dims(inlined)
    inlined = _sanitize_graph_constraints(inlined)
    try:
        _validate_optimizer_graph(inlined, phase="inline.candidate")
    except ValueError:
        changed_indices = [
            index
            for index, (original_module, rewritten_module) in enumerate(
                zip(graph.modules, rewritten_modules, strict=True)
            )
            if rewritten_module != original_module
        ]
        if len(changed_indices) > 32:
            return graph
        accepted = list(graph.modules)
        changed = False
        for index in changed_indices:
            rewritten_module = rewritten_modules[index]
            candidate_modules = list(accepted)
            candidate_modules[index] = _refresh_single_graph_module_in_program(
                replace(graph, modules=tuple(candidate_modules)),
                rewritten_module,
            )
            candidate = replace(graph, modules=tuple(candidate_modules))
            candidate = _alpha_rename_shadowed_type_dims(candidate)
            candidate = _sanitize_graph_constraints(candidate)
            try:
                _validate_optimizer_graph(candidate, phase="inline.candidate.module")
            except ValueError:
                continue
            accepted = list(candidate.modules)
            changed = True
        if not changed:
            return graph
        inlined = _alpha_rename_shadowed_type_dims(replace(graph, modules=tuple(accepted)))
    inlined = prune_graph_to_main(inlined)
    inlined = _alpha_rename_shadowed_type_dims(inlined)
    inlined = _sanitize_graph_constraints(inlined)
    _validate_optimizer_graph(inlined, phase="inline")
    return inlined


def optimize_graph_program(
    graph: GraphProgram,
    *,
    config: GraphOptimizeConfig | None = None,
) -> GraphProgram:
    _VALIDATED_OPTIMIZER_GRAPH_KEYS.clear()
    _REFRESH_GRAPH_PROGRAM_TYPES_CACHE.clear()
    config = config or GraphOptimizeConfig()
    if config.specialize_definitions not in _SPECIALIZE_MODES:
        raise ValueError(
            "GraphOptimizeConfig.specialize_definitions must be one of: "
            + ", ".join(sorted(_SPECIALIZE_MODES))
        )
    backend_intrinsic_target, enabled_backend_intrinsics = _parse_backend_intrinsics(config.backend_intrinsics)
    graph = _alpha_rename_shadowed_type_dims(graph)
    graph = _sanitize_graph_constraints(graph)
    _validate_optimizer_graph(graph, phase="input")
    current = prune_graph_to_main(graph) if config.prune_to_main else graph
    current = _alpha_rename_shadowed_type_dims(current)
    _validate_optimizer_graph(current, phase="initial_prune" if config.prune_to_main else "initial")
    for _ in range(config.max_iterations):
        current = _alpha_rename_shadowed_type_dims(current)
        before = current
        if config.constant_dim_substitution:
            current = _substitute_atomic_constant_dims_local(current)
            current = _alpha_rename_shadowed_type_dims(current)
            _validate_optimizer_graph(current, phase="constant_dim_substitution")
        if config.constant_folding:
            current = _simplify_symbolic_graph_dims(current)
            current = _alpha_rename_shadowed_type_dims(current)
        module_effects = infer_graph_module_effects(current.modules)
        module_usages = infer_graph_module_usages(current.modules)
        current = _promote_total_pure_zero_arg_modules_to_globals(
            current,
            module_effects=module_effects,
            module_usages=module_usages,
        )
        current = _alpha_rename_shadowed_type_dims(current)
        _validate_optimizer_graph(current, phase="promote_zero_arg_globals")
        current = _optimize_modules_local_with_fresh_domain_facts(
            current,
            config=config,
            phase="local_cleanup",
        )
        current = _alpha_rename_shadowed_type_dims(current)
        current = _prune_dead_formals(current)
        current = _alpha_rename_shadowed_type_dims(current)
        if config.constant_folding:
            module_effects = infer_graph_module_effects(current.modules)
            module_usages = infer_graph_module_usages(current.modules)
            modules_by_name = {module.name: module for module in current.modules}
            pre_refresh_modules = tuple(
                _inline_single_use_total_pure_exprs_module(
                    module,
                    module_effects=module_effects,
                    module_usages=module_usages,
                    modules_by_name=modules_by_name,
                )
                for module in current.modules
            )
            candidate = replace(current, modules=pre_refresh_modules)
            candidate = _refresh_graph_program_types(candidate)
            candidate = _alpha_rename_shadowed_type_dims(candidate)
            candidate = replace(
                candidate,
                modules=_preserve_unchanged_module_types(
                    current.modules,
                    pre_refresh_modules,
                    candidate.modules,
                    main_module=current.main_module,
                ),
            )
            candidate = _alpha_rename_shadowed_type_dims(candidate)
            candidate = _sanitize_graph_constraints(candidate)
            try:
                _validate_optimizer_graph(candidate, phase="inline_single_use_exprs")
            except ValueError:
                pass
            else:
                current = candidate
        if config.common_subexpression_elimination:
            module_effects = infer_graph_module_effects(current.modules)
            module_usages = infer_graph_module_usages(current.modules)
            candidate = replace(
                current,
                modules=tuple(
                    _common_subexpression_eliminate_module(
                        module,
                        module_effects=module_effects,
                        module_usages=module_usages,
                        fold=config.constant_folding,
                    )
                    for module in current.modules
                ),
            )
            candidate = _refresh_graph_program_types(candidate)
            candidate = _alpha_rename_shadowed_type_dims(candidate)
            candidate = _sanitize_graph_constraints(candidate)
            try:
                _validate_optimizer_graph(candidate, phase="common_subexpression_elimination")
            except ValueError:
                pass
            else:
                current = candidate
        current = _specialize_definitions_to_fixpoint(current, config=config)
        current = _alpha_rename_shadowed_type_dims(current)
        _validate_optimizer_graph(current, phase="specialize")
        if config.inline_safe:
            current = _inline_safe_modules(current, config=config)
            current = _alpha_rename_shadowed_type_dims(current)
            _validate_optimizer_graph(current, phase="inline")
            if config.common_subexpression_elimination:
                module_effects = infer_graph_module_effects(current.modules)
                module_usages = infer_graph_module_usages(current.modules)
                candidate = replace(
                    current,
                    modules=tuple(
                        _common_subexpression_eliminate_module(
                            module,
                            module_effects=module_effects,
                            module_usages=module_usages,
                            fold=config.constant_folding,
                        )
                        for module in current.modules
                    ),
                )
                candidate = _refresh_graph_program_types(candidate)
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                try:
                    _validate_optimizer_graph(candidate, phase="post_inline_cse")
                except ValueError:
                    pass
                else:
                    current = candidate
        candidate = _rewrite_assign_slice(current)
        if candidate != current:
            candidate = _refresh_graph_program_types(candidate)
            candidate = _alpha_rename_shadowed_type_dims(candidate)
            candidate = _sanitize_graph_constraints(candidate)
            try:
                _validate_optimizer_graph(candidate, phase="assign_slice")
            except ValueError:
                pass
            else:
                current = candidate
        candidate = _rewrite_linear_projection_packs(current)
        if candidate != current:
            candidate = _refresh_graph_program_types(candidate)
            candidate = _alpha_rename_shadowed_type_dims(candidate)
            candidate = _sanitize_graph_constraints(candidate)
            _validate_optimizer_graph(candidate, phase="linear_projection_pack")
            current = candidate
        candidate = _rewrite_dense_gate_up_linear_pairs(current)
        if candidate != current:
            candidate = _refresh_graph_program_types(candidate)
            candidate = _alpha_rename_shadowed_type_dims(candidate)
            candidate = _sanitize_graph_constraints(candidate)
            _validate_optimizer_graph(candidate, phase="dense_gate_up_linear_pair")
            current = candidate
        if backend_intrinsic_target == "codegen2-torch":
            candidate = (
                _rewrite_torch_rope_intrinsics(current, enabled_intrinsics=enabled_backend_intrinsics)
                if (
                    _backend_intrinsic_enabled(enabled_backend_intrinsics, "__torch_rope_apply_factors")
                    or _backend_intrinsic_enabled(enabled_backend_intrinsics, "__torch_rope_pair_apply_factors")
                )
                else current
            )
            if candidate != current:
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                _validate_optimizer_graph(candidate, phase="torch_rope_intrinsics")
                current = candidate
            candidate = (
                _rewrite_backend_sdpa_intrinsics(current, op_name="__torch_sdpa")
                if _backend_intrinsic_enabled(enabled_backend_intrinsics, "__torch_sdpa")
                else current
            )
            if candidate != current:
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                _validate_optimizer_graph(candidate, phase="torch_sdpa_intrinsics")
                current = candidate
            candidate = (
                _rewrite_torch_swiglu_ffn_intrinsics(current)
                if _backend_intrinsic_enabled(enabled_backend_intrinsics, "__torch_swiglu_ffn")
                else current
            )
            if candidate != current:
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                _validate_optimizer_graph(candidate, phase="torch_swiglu_ffn_intrinsics")
                current = candidate
            candidate = (
                _rewrite_torch_expert_swiglu_ffn_intrinsics(current)
                if _backend_intrinsic_enabled(enabled_backend_intrinsics, "__torch_expert_swiglu_ffn")
                else current
            )
            if candidate != current:
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                _validate_optimizer_graph(candidate, phase="torch_expert_swiglu_ffn_intrinsics")
                current = candidate
            selected_expert_intrinsics = {
                "__torch_selected_expert_clamped_packed_swiglu_ffn",
                "__torch_selected_expert_packed_gegelu_ffn",
                "__torch_selected_expert_packed_swiglu_ffn",
                "__torch_selected_expert_relu2_ffn",
                "__torch_selected_expert_swiglu_ffn",
            }
            candidate = (
                _rewrite_torch_selected_expert_intrinsics(
                    current,
                    enabled_intrinsics=enabled_backend_intrinsics,
                )
                if selected_expert_intrinsics & enabled_backend_intrinsics
                else current
            )
            if candidate != current:
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                _validate_optimizer_graph(candidate, phase="torch_selected_expert_intrinsics")
                current = candidate
            candidate = (
                _rewrite_torch_expert_packed_swiglu_ffn_intrinsics(current)
                if _backend_intrinsic_enabled(enabled_backend_intrinsics, "__torch_expert_packed_swiglu_ffn")
                else current
            )
            if candidate != current:
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                _validate_optimizer_graph(candidate, phase="torch_expert_packed_swiglu_ffn_intrinsics")
                current = candidate
            candidate = (
                _rewrite_torch_weighted_topk_sum_intrinsics(current)
                if _backend_intrinsic_enabled(enabled_backend_intrinsics, "__torch_weighted_topk_sum")
                else current
            )
            if candidate != current:
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                _validate_optimizer_graph(candidate, phase="torch_weighted_topk_sum_intrinsics")
                current = candidate
            candidate = (
                _rewrite_torch_topk_normalize_intrinsics(current)
                if _backend_intrinsic_enabled(enabled_backend_intrinsics, "__torch_topk_normalize")
                else current
            )
            if candidate != current:
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                _validate_optimizer_graph(candidate, phase="torch_topk_normalize_intrinsics")
                current = candidate
        if backend_intrinsic_target == "codegen2-tinygrad":
            candidate = (
                _rewrite_backend_sdpa_intrinsics(current, op_name="__tinygrad_sdpa")
                if _backend_intrinsic_enabled(enabled_backend_intrinsics, "__tinygrad_sdpa")
                else current
            )
            if candidate != current:
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                _validate_optimizer_graph(candidate, phase="tinygrad_sdpa_intrinsics")
                current = candidate
        if backend_intrinsic_target == "codegen2-triton":
            candidate = (
                _rewrite_triton_rmsnorm_scaled_intrinsics(current)
                if _backend_intrinsic_enabled(enabled_backend_intrinsics, "__triton_rmsnorm_scaled")
                else current
            )
            if candidate != current:
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                _validate_optimizer_graph(candidate, phase="triton_rmsnorm_scaled_intrinsics")
                current = candidate
            candidate = (
                _rewrite_triton_rmsnorm_noscale_intrinsics(current)
                if _backend_intrinsic_enabled(enabled_backend_intrinsics, "__triton_rmsnorm_noscale")
                else current
            )
            if candidate != current:
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                _validate_optimizer_graph(candidate, phase="triton_rmsnorm_noscale_intrinsics")
                current = candidate
            candidate = (
                _rewrite_triton_rmsnorm_unit_offset_scaled_intrinsics(current)
                if _backend_intrinsic_enabled(enabled_backend_intrinsics, "__triton_rmsnorm_unit_offset_scaled")
                else current
            )
            if candidate != current:
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                _validate_optimizer_graph(candidate, phase="triton_rmsnorm_unit_offset_scaled_intrinsics")
                current = candidate
            candidate = (
                _rewrite_triton_geglu_tanh_activation_intrinsics(current)
                if _backend_intrinsic_enabled(enabled_backend_intrinsics, "__triton_geglu_tanh_activation")
                else current
            )
            if candidate != current:
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                _validate_optimizer_graph(candidate, phase="triton_geglu_tanh_activation_intrinsics")
                current = candidate
            triton_selected_expert_intrinsics = {
                "__triton_selected_expert_packed_swiglu_ffn",
            }
            candidate = (
                _rewrite_torch_selected_expert_intrinsics(
                    current,
                    enabled_intrinsics=enabled_backend_intrinsics,
                    op_prefix="__triton",
                )
                if triton_selected_expert_intrinsics & enabled_backend_intrinsics
                else current
            )
            if candidate != current:
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                _validate_optimizer_graph(candidate, phase="triton_selected_expert_intrinsics")
                current = candidate
            candidate = (
                _rewrite_triton_swiglu_activation_intrinsics(current)
                if _backend_intrinsic_enabled(enabled_backend_intrinsics, "__triton_swiglu_activation")
                else current
            )
            if candidate != current:
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                _validate_optimizer_graph(candidate, phase="triton_swiglu_activation_intrinsics")
                current = candidate
        if backend_intrinsic_target == "codegen2-vllm":
            candidate = (
                _rewrite_backend_sdpa_intrinsics(current, op_name="__vllm_paged_attention")
                if _backend_intrinsic_enabled(enabled_backend_intrinsics, "__vllm_paged_attention")
                else current
            )
            if candidate != current:
                candidate = _alpha_rename_shadowed_type_dims(candidate)
                candidate = _sanitize_graph_constraints(candidate)
                _validate_optimizer_graph(candidate, phase="vllm_paged_attention_intrinsics")
                current = candidate
        if config.constant_folding:
            module_effects = infer_graph_module_effects(current.modules)
            module_usages = infer_graph_module_usages(current.modules)
            hoist_candidate = _hoist_eager_nested_exprs(
                current,
                module_effects=module_effects,
                module_usages=module_usages,
            )
            hoist_candidate = _refresh_graph_program_types(hoist_candidate)
            hoist_candidate = _alpha_rename_shadowed_type_dims(hoist_candidate)
            hoist_candidate = _sanitize_graph_constraints(hoist_candidate)
            try:
                _validate_optimizer_graph(hoist_candidate, phase="hoist_eager_nested_exprs")
            except ValueError:
                pass
            else:
                current = hoist_candidate
            module_effects = infer_graph_module_effects(current.modules)
            module_usages = infer_graph_module_usages(current.modules)
            modules_by_name = {module.name: module for module in current.modules}
            pre_refresh_modules = tuple(
                _inline_single_use_total_pure_exprs_module(
                    module,
                    module_effects=module_effects,
                    module_usages=module_usages,
                    modules_by_name=modules_by_name,
                )
                for module in current.modules
            )
            candidate = replace(current, modules=pre_refresh_modules)
            candidate = _refresh_graph_program_types(candidate)
            candidate = _alpha_rename_shadowed_type_dims(candidate)
            candidate = replace(
                candidate,
                modules=_preserve_unchanged_module_types(
                    current.modules,
                    pre_refresh_modules,
                    candidate.modules,
                    main_module=current.main_module,
                ),
            )
            candidate = _alpha_rename_shadowed_type_dims(candidate)
            candidate = _sanitize_graph_constraints(candidate)
            try:
                _validate_optimizer_graph(candidate, phase="post_hoist_inline_single_use_exprs")
            except ValueError:
                pass
            else:
                current = candidate
        if config.prune_to_main:
            current = prune_graph_to_main(current)
            current = _alpha_rename_shadowed_type_dims(current)
            _validate_optimizer_graph(current, phase="prune")
        current = _canonicalize_generated_module_names(current)
        current = _alpha_rename_shadowed_type_dims(current)
        _validate_optimizer_graph(current, phase="canonicalize_module_names")
        current = _canonicalize_generated_value_names(current)
        current = _alpha_rename_shadowed_type_dims(current)
        _validate_optimizer_graph(current, phase="canonicalize_value_names")
        _validate_optimizer_graph(current, phase="iteration")
        if current == before:
            return current
    raise RuntimeError(
        f"graph optimizer did not converge after {config.max_iterations} iterations"
    )


__all__ = ["GraphOptimizeConfig", "optimize_graph_program", "prune_graph_to_main"]
