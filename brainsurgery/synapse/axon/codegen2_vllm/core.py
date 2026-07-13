from __future__ import annotations

import re
from typing import Any, Iterable

from ..codegen2_torch.core import (
    _DirectTorchEmitter,
    _dim_ident,
    _py_ident,
    graph_main_output_names,
)
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
    validate_graph_program,
)
from ..ast import TypeDim, TypeNamed, TypeOptional, TypePath, TypeTensor
from .classify import (
    VLLMLayerClassification,
    VLLMLayerType,
    _is_linear_call,
    classify_graph_for_vllm,
)


def _graph_path_key(path: GraphPath) -> str:
    return ".".join(path.parts)


_SELECTED_EXPERT_OPS = frozenset(
    {
        "_expert_linear",
        "__vllm_selected_expert_clamped_packed_swiglu_ffn",
        "__vllm_selected_expert_packed_gegelu_ffn",
        "__vllm_selected_expert_packed_swiglu_ffn",
        "__vllm_selected_expert_relu2_ffn",
        "__vllm_selected_expert_swiglu_ffn",
        "__torch_selected_expert_clamped_packed_swiglu_ffn",
        "__torch_selected_expert_packed_gegelu_ffn",
        "__torch_selected_expert_packed_swiglu_ffn",
        "__torch_selected_expert_relu2_ffn",
        "__torch_selected_expert_swiglu_ffn",
        "__jax_selected_expert_clamped_packed_swiglu_ffn",
        "__jax_selected_expert_packed_gegelu_ffn",
        "__jax_selected_expert_packed_swiglu_ffn",
        "__jax_selected_expert_relu2_ffn",
        "__jax_selected_expert_swiglu_ffn",
        "__triton_selected_expert_packed_swiglu_ffn",
    }
)


def _graph_uses_selected_expert_moe(program: GraphProgram) -> bool:
    for module in program.modules:
        for node in module.nodes:
            if node.op.name in _SELECTED_EXPERT_OPS:
                return True
    return False


_BINOP_SYMBOLS = {
    "core.binary.*": "*",
    "core.binary./": "/",
    "core.binary.+": "+",
    "core.binary.-": "-",
    "core.binary.%": "%",
    "core.binary.==": "==",
    "core.binary.!=": "!=",
    "core.binary.>=": ">=",
    "core.binary.and": "and",
    "core.binary.or": "or",
}

_CONFIG_ALIASES: dict[str, tuple[str, ...]] = {
    "hidden_size": ("n_embd", "d_model"),
    "num_hidden_layers": ("n_layer", "num_layers"),
    "num_attention_heads": ("n_head", "num_heads"),
    "num_key_value_heads": ("num_kv_heads", "n_kv_heads"),
    "max_position_embeddings": ("n_positions", "n_ctx"),
    "num_local_experts": ("num_experts", "n_routed_experts"),
}


def _simplify_binop(op: str, left: str, right: str) -> str:
    """Simplify a binary operation expression, removing trivial operands."""
    sym = _BINOP_SYMBOLS.get(op, op)
    if op == "core.binary.and":
        if right == "True":
            return f"({left})"
        if left == "True":
            return f"({right})"
        if right == "False" or left == "False":
            return "False"
    if op == "core.binary.or":
        if right == "False":
            return f"({left})"
        if left == "False":
            return f"({right})"
        if right == "True" or left == "True":
            return "True"
    return f"({left} {sym} {right})"


def _literal_value(operand: GraphOperand, default: Any = None) -> Any:
    if isinstance(operand, GraphLiteral):
        return operand.value
    return default


def _bool_arg(node: GraphNode, index: int, default: bool = False) -> bool:
    if len(node.inputs) <= index:
        return default
    val = _literal_value(node.inputs[index], default)
    if isinstance(val, bool):
        return val
    return default


def _linear_bool_args(node: GraphNode) -> tuple[bool, ...]:
    return tuple(
        operand.value
        for operand in node.inputs[3:]
        if isinstance(operand, GraphLiteral) and isinstance(operand.value, bool)
    )


def _linear_bias_arg(node: GraphNode, default: bool = False) -> bool:
    bools = _linear_bool_args(node)
    if bools:
        return bools[0]
    if len(node.inputs) < 5:
        return default
    bias_leaf = node.inputs[-1]
    if isinstance(bias_leaf, GraphLiteral) and bias_leaf.value is None:
        return False
    if isinstance(bias_leaf, GraphPath):
        return True
    return default


def _linear_transpose_arg(node: GraphNode, default: bool = False) -> bool:
    bools = _linear_bool_args(node)
    if len(bools) >= 2:
        return bools[1]
    return default


def _str_arg(node: GraphNode, index: int, default: str | None = None) -> str | None:
    if len(node.inputs) <= index:
        return default
    val = _literal_value(node.inputs[index], default)
    if isinstance(val, str):
        return val
    return default


def _int_arg(node: GraphNode, index: int, default: int | None = None) -> int | None:
    if len(node.inputs) <= index:
        return default
    val = _literal_value(node.inputs[index], default)
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    return default


def _linear_base_key(node: GraphNode) -> str:
    if len(node.inputs) < 1:
        return ""
    # Search all inputs for a GraphPath — the first one that doesn't end
    # with 'weight'/'bias' is the module path (e.g. q_proj, not q_proj.weight).
    # If only weight/bias paths exist, return the first GraphPath found.
    fallback = ""
    for inp in node.inputs:
        if isinstance(inp, GraphPath) and inp.parts:
            key = _graph_path_key(inp)
            if not key.endswith(".weight") and not key.endswith(".bias"):
                return key
            if not fallback:
                fallback = key
    return fallback


def _linear_base_keys_from_expr_operand(operand: GraphOperand) -> tuple[str, ...]:
    """Return primitive _linear base paths contained in an expression operand."""
    keys: list[str] = []
    if isinstance(operand, GraphExpr):
        if operand.op.name == "_linear" and operand.inputs:
            base = operand.inputs[0]
            if isinstance(base, GraphPath) and base.parts:
                keys.append(_graph_path_key(base))
        for item in operand.inputs:
            keys.extend(_linear_base_keys_from_expr_operand(item))
        for item in operand.attrs.values():
            if isinstance(item, (GraphExpr, GraphValueRef, GraphValue, GraphLiteral, GraphPath)):
                keys.extend(_linear_base_keys_from_expr_operand(item))
    return tuple(dict.fromkeys(key for key in keys if key))


def _safe_ident(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe


def _value_name(operand: GraphOperand) -> str | None:
    if isinstance(operand, (GraphValueRef, GraphValue)):
        return operand.name
    return None


class _DirectVLLMEmitter(_DirectTorchEmitter):
    """vLLM backend emitter.

    Generates a vLLM-compatible model class that uses vLLM's parallel layers
    (QKVParallelLinear, RowParallelLinear, VocabParallelEmbedding, etc.)
    and vllm.attention.Attention for PagedAttention.

    vLLM-specific graph intrinsics (__vllm_paged_attention, etc.) are
    consumed here; all other ops fall through to the torch emitter.
    """

    def __init__(self, *args: Any, model_config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._vllm_classification: VLLMLayerClassification = classify_graph_for_vllm(self.program)
        self._ffn_down_node_ids: set[str] = {
            g.down_node_id
            for g in self._vllm_classification.ffn_groups
            if g.down_node_id is not None
        }
        self._use_clean_forward: bool = False
        self._vllm_emitted_layer_node_ids: set[str] = set()
        self._model_config_data: dict[str, Any] = model_config or {}
        raw_prefixes = self._model_config_data.get("__axon_checkpoint_prefixes", ())
        self._checkpoint_prefixes: set[str] = {
            str(item) for item in raw_prefixes if isinstance(item, str) and item
        }

    def _primitive_expr(self, primitive: str, node: Any, *, local: set[str], symbols_dict: str) -> str:
        if primitive == "params_has_root":
            args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
            if not args:
                raise ValueError("params_has_root missing root argument")
            return (
                f"any(k == {args[0]} or k.startswith(str({args[0]}) + '.') "
                "for k in self._loaded_state_keys)"
            )
        return super()._primitive_expr(primitive, node, local=local, symbols_dict=symbols_dict)

    def _static_path_key(self, operand: Any) -> str | None:
        if not isinstance(operand, GraphPath) or not operand.absolute:
            return None
        parts = tuple(part for part in operand.parts if part)
        if any("{" in part or "}" in part for part in parts):
            template_names = {
                part[1:-1]
                for part in parts
                if part.startswith("{") and part.endswith("}")
            }
            if any(name != "__scope" for name in template_names):
                return None
            formatted = self._format_template_parts(parts)
            if any("{" in part or "}" in part for part in formatted):
                return None
            return ".".join(part for part in formatted if part)
        return super()._static_path_key(operand)

    def emit(self) -> str:
        self._validate_vllm_lowering_completeness()
        lines: list[str] = [f"class {self.class_name}(nn.Module):"]
        self._emit_common(lines)
        if not self._use_clean_forward:
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
        self._add(lines, 4, f"axon_vllm_legacy_forward = {not self._use_clean_forward!r}")
        lines.append("")
        self._emit_generate(lines)
        return "\n".join(lines)

    def _validate_vllm_lowering_completeness(self) -> None:
        cls = self._vllm_classification
        if cls.mamba_mixer_module_names:
            return
        if cls.attention_node_ids and not cls.qkv_groups:
            raise NotImplementedError(
                "codegen2-vllm found attention but could not prove a complete "
                "Q/K/V group for vLLM paged-attention generation; legacy "
                "explicit matmul attention is unsafe with vLLM KV cache"
            )
        repeated_mod_name = self._clean_forward_repeated_module_name(cls)
        if repeated_mod_name is None:
            if cls.qkv_groups and cls.repeated_module_names:
                raise NotImplementedError(
                    "codegen2-vllm could not prove a single complete repeated "
                    "transformer layer for vLLM paged-attention generation; "
                    "legacy explicit matmul attention is unsafe with vLLM KV cache"
                )
            return
        repeated_mod = self.modules_by_name.get(repeated_mod_name)
        if repeated_mod is None:
            return
        unsupported_reason = self._clean_forward_unsupported_reason(cls)
        if unsupported_reason is not None:
            raise NotImplementedError(unsupported_reason)
        for group in cls.qkv_groups:
            qkv_ids = (group.q_node_id, group.k_node_id, group.v_node_id)
            if len(set(qkv_ids)) != len(qkv_ids):
                raise NotImplementedError(
                    "codegen2-vllm found an ambiguous Q/K/V group with shared "
                    "projection nodes; refusing to lower it to vLLM packed "
                    "attention"
                )
        layer_norms = self._analyze_layer_norms(repeated_mod, cls)
        if not layer_norms:
            return
        if cls.qkv_groups and all(group.attention_node_id for group in cls.qkv_groups):
            return
        raise NotImplementedError(
            "codegen2-vllm could not prove a complete vLLM attention lowering "
            f"for repeated module {repeated_mod_name!r}; refusing to emit a "
            "partial transformer forward"
        )

    def _clean_forward_repeated_module_name(
        self,
        classification: VLLMLayerClassification | None = None,
    ) -> str | None:
        """Return the repeated module that clean vLLM forward can emit.

        Clean forward is a hand-scheduled transformer layer. It is only valid
        when the selected repeated module contains the complete layer schedule
        needed by that hand scheduler. Helper modules that contain only
        attention or only FFN may still be repeated and may still get vLLM
        layer objects, but using one as the whole clean-forward layer silently
        drops the sibling helper.
        """
        cls = classification or self._vllm_classification
        if not cls.repeated_module_names:
            return None
        candidates = sorted(
            cls.repeated_module_names,
            key=lambda n: len(self.modules_by_name[n].nodes) if n in self.modules_by_name else 0,
            reverse=True,
        )
        matches: list[str] = []
        for name in candidates:
            mod = self.modules_by_name.get(name)
            if mod is None:
                continue
            if not self._analyze_layer_norms(mod, cls):
                continue
            node_ids = self._reachable_module_node_ids(name, depth=0, visited=set())
            if not cls.qkv_groups:
                continue
            if cls.qkv_groups:
                if not any(group.attention_node_id in node_ids for group in cls.qkv_groups):
                    continue
                if cls.ffn_groups and not any(
                    (group.down_node_id is not None and group.down_node_id in node_ids)
                    or (
                        group.gate_up_intrinsic_node_id is not None
                        and group.gate_up_intrinsic_node_id in node_ids
                    )
                    for group in cls.ffn_groups
                ):
                    continue
            matches.append(name)
        return matches[0] if len(matches) == 1 else None

    def _reachable_module_node_ids(
        self,
        name: str,
        *,
        depth: int,
        visited: set[str],
    ) -> set[str]:
        if depth > 8 or name in visited:
            return set()
        visited.add(name)
        mod = self.modules_by_name.get(name)
        if mod is None:
            return set()
        node_ids = {node.id for node in mod.nodes}
        for node in mod.nodes:
            if node.op.name in self.modules_by_name:
                node_ids.update(
                    self._reachable_module_node_ids(
                        node.op.name,
                        depth=depth + 1,
                        visited=set(visited),
                    )
                )
            for operand in (*node.inputs, *node.attrs.values()):
                node_ids.update(
                    self._reachable_operand_module_node_ids(
                        operand,
                        depth=depth + 1,
                        visited=set(visited),
                    )
                )
        return node_ids

    def _reachable_operand_module_node_ids(
        self,
        operand: GraphOperand,
        *,
        depth: int,
        visited: set[str],
    ) -> set[str]:
        if depth > 8:
            return set()
        if not isinstance(operand, GraphExpr):
            return set()
        node_ids: set[str] = set()
        if operand.op.name in self.modules_by_name:
            node_ids.update(
                self._reachable_module_node_ids(
                    operand.op.name,
                    depth=depth + 1,
                    visited=set(visited),
                )
            )
        for item in operand.inputs:
            node_ids.update(
                self._reachable_operand_module_node_ids(
                    item,
                    depth=depth + 1,
                    visited=set(visited),
                )
            )
        for item in operand.attrs.values():
            if isinstance(item, (GraphExpr, GraphValueRef, GraphValue, GraphLiteral, GraphPath)):
                node_ids.update(
                    self._reachable_operand_module_node_ids(
                        item,
                        depth=depth + 1,
                        visited=set(visited),
                    )
                )
        return node_ids

    def _clean_forward_unsupported_reason(
        self,
        classification: VLLMLayerClassification | None = None,
    ) -> str | None:
        cls = classification or self._vllm_classification
        if cls.mamba_mixer_module_names:
            return None
        if cls.attention_node_ids and not cls.qkv_groups:
            return (
                "codegen2-vllm found attention but could not prove a complete "
                "Q/K/V group for vLLM paged-attention generation; legacy "
                "explicit matmul attention is unsafe with vLLM KV cache"
            )
        if self._clean_forward_repeated_module_name(cls) is None:
            if cls.qkv_groups and cls.repeated_module_names:
                return (
                    "codegen2-vllm could not prove a single complete repeated "
                    "transformer layer for vLLM paged-attention generation; "
                    "legacy explicit matmul attention is unsafe with vLLM KV cache"
                )
            return None
        main_module = self.modules_by_name.get(self.program.main_module)
        if main_module is None:
            return None
        for node in main_module.nodes:
            if node.op.name != "core.repeat":
                continue
            carry_count = _literal_value(node.attrs.get("carry_count"), 0)
            if not isinstance(carry_count, int) or carry_count <= 0:
                continue
            for carry_index, carry in enumerate(node.inputs[3 : 3 + carry_count]):
                type_expr = getattr(carry, "type_expr", None)
                while isinstance(type_expr, TypeOptional):
                    type_expr = type_expr.inner
                if not isinstance(type_expr, TypeTensor):
                    continue
                dims = tuple(getattr(carry, "dims", None) or type_expr.dims)
                if len(dims) != 3:
                    return (
                        "codegen2-vllm clean transformer forward currently supports "
                        "rank-3 tensor layer carries [B,S,D] only; "
                        f"repeat carry {carry_index} has rank {len(dims)}"
                    )
        for group in cls.qkv_groups:
            qkv_ids = (group.q_node_id, group.k_node_id, group.v_node_id)
            if len(set(qkv_ids)) != len(qkv_ids):
                return (
                    "codegen2-vllm found an ambiguous Q/K/V group with shared "
                    "projection nodes; refusing to lower it to vLLM packed "
                    "attention"
                )
        return None

    def _resolve_const_value(self, name: str, visited: set[str] | None = None) -> str | None:
        """Resolve a top-level constant name to a config expression."""
        def literal_output_value(operand: GraphOperand) -> Any:
            if isinstance(operand, GraphLiteral):
                return operand.value
            if (
                isinstance(operand, GraphExpr)
                and operand.op.name in {"core.ascribe", "core.alias"}
                and len(operand.inputs) == 1
            ):
                return literal_output_value(operand.inputs[0])
            return None

        if visited is None:
            visited = set()
        if name in visited:
            return None
        visited.add(name)
        for module in self.program.modules:
            if module.name != name:
                continue
            if len(module.outputs) == 1:
                value = literal_output_value(module.outputs[0])
                if isinstance(value, str):
                    return repr(value)
                if value is None:
                    if isinstance(module.outputs[0], GraphLiteral):
                        return "None"
                if isinstance(value, (bool, int, float)):
                    return repr(value)
            resolved: str | None = None
            for node in module.nodes:
                if node.op.name in ("_config_int", "_config_dim"):
                    path_inp = node.inputs[0] if node.inputs else None
                    default = _literal_value(node.inputs[1], None) if len(node.inputs) >= 2 else None
                    if isinstance(path_inp, GraphPath) and path_inp.parts:
                        field = self._config_field_from_path(path_inp)
                        if field == "head_dim":
                            resolved = self._head_dim_expr()
                            continue
                        if isinstance(default, int):
                            resolved = self._config_expr(field, default=default)
                            continue
                        resolved = self._config_expr(field)
                        continue
                elif node.op.name == "_config_float":
                    path_inp = node.inputs[0] if node.inputs else None
                    default = _literal_value(node.inputs[1], None) if len(node.inputs) >= 2 else None
                    if isinstance(path_inp, GraphPath) and path_inp.parts:
                        field = self._config_field_from_path(path_inp)
                        if isinstance(default, (int, float)):
                            resolved = self._config_expr(field, default=float(default))
                            continue
                        resolved = self._config_expr(field, default=0.0)
                        continue
                elif node.op.name == "_config_str":
                    path_inp = node.inputs[0] if node.inputs else None
                    default = _literal_value(node.inputs[1], None) if len(node.inputs) >= 2 else None
                    if isinstance(path_inp, GraphPath) and path_inp.parts:
                        field = self._config_field_from_path(path_inp)
                        if isinstance(default, str):
                            resolved = self._config_expr(field, default=default)
                            continue
                        resolved = self._config_expr(field)
                        continue
                elif node.op.name == "_config_bool":
                    path_inp = node.inputs[0] if node.inputs else None
                    default = _literal_value(node.inputs[1], None) if len(node.inputs) >= 2 else None
                    if isinstance(path_inp, GraphPath) and path_inp.parts:
                        field = self._config_field_from_path(path_inp)
                        if isinstance(default, bool):
                            resolved = self._config_expr(field, default=default)
                            continue
                        resolved = self._config_expr(field, default=False)
                        continue
                elif node.op.name == "_params_has_root" and node.inputs:
                    root = _literal_value(node.inputs[0], None)
                    if isinstance(root, str):
                        resolved = self._params_has_root_expr(root)
                        continue
                traced = self._trace_node_dim_expr(node, module, "i", set(visited))
                if traced is not None:
                    resolved = traced
            return resolved
        return None

    def _global_symbol_expr(self, name: str, symbols_dict: str = "self._symbols") -> str:
        resolved = self._resolve_const_value(name)
        if resolved is not None:
            return resolved
        return f"{symbols_dict}[{name!r}]"

    def _config_field_from_path(self, path: GraphPath) -> str:
        parts: list[str] = []
        for part in path.parts:
            if part.startswith("{") and part.endswith("}"):
                resolved = self._resolve_const_value(part[1:-1])
                if resolved is not None:
                    parts.append(resolved.strip("'\""))
                    continue
            parts.append(part)
        if parts and parts[0] == "text_config":
            return parts[-1]
        return parts[-1] if parts else ""

    def _trace_dim_expr(
        self,
        operand: GraphOperand,
        repeated_mod: Any,
        loop_var: str = "i",
        visited: set[str] | None = None,
    ) -> str | None:
        """Trace a dimension operand to a Python expression string."""
        if visited is None:
            visited = set()
        if isinstance(operand, GraphLiteral):
            if operand.value is None:
                return "None"
            if isinstance(operand.value, str):
                return repr(operand.value)
            return str(operand.value)
        if isinstance(operand, GraphValueRef):
            name = operand.name
            if name is None:
                return None
            if name == loop_var:
                return loop_var
            if name in visited:
                return None
            visited.add(name)
            for node in repeated_mod.nodes:
                for out in node.outputs:
                    if hasattr(out, "name") and out.name == name:
                        return self._trace_node_dim_expr(node, repeated_mod, loop_var, visited)
            for inp in repeated_mod.inputs:
                if hasattr(inp, "name") and inp.name == name:
                    result = self._trace_param_to_call_site(name, repeated_mod, loop_var, visited)
                    if result is not None:
                        return result
                    return None
            const_val = self._resolve_const_value(name, set(visited) - {name})
            if const_val is not None:
                return const_val
            return None
        if isinstance(operand, GraphExpr):
            return self._trace_expr_dim_expr(operand, repeated_mod, loop_var, visited)
        return None

    def _find_call_exprs(
        self, module_name: str, obj: Any, found: list | None = None,
    ) -> list:
        """Recursively find all GraphExpr with op.name == module_name."""
        if found is None:
            found = []
        if isinstance(obj, GraphExpr):
            if obj.op.name == module_name:
                found.append(obj)
            for inp in obj.inputs:
                self._find_call_exprs(module_name, inp, found)
        elif isinstance(obj, GraphNode):
            for inp in obj.inputs:
                self._find_call_exprs(module_name, inp, found)
        return found

    def _trace_param_to_call_site(
        self,
        param_name: str,
        called_mod: Any,
        loop_var: str,
        visited: set[str],
    ) -> str | None:
        """Trace a module parameter to its call-site argument."""
        param_idx = None
        for i, inp in enumerate(called_mod.inputs):
            if hasattr(inp, "name") and inp.name == param_name:
                param_idx = i
                break
        if param_idx is None:
            return None
        for caller_mod in self.program.modules:
            for node in caller_mod.nodes:
                call_exprs = self._find_call_exprs(called_mod.name, node)
                for call_expr in call_exprs:
                    if param_idx < len(call_expr.inputs):
                        arg = call_expr.inputs[param_idx]
                        result = self._trace_dim_expr(arg, caller_mod, loop_var, visited)
                        if result is not None:
                            return result
        return None

    def _trace_node_dim_expr(
        self,
        node: GraphNode,
        repeated_mod: Any,
        loop_var: str,
        visited: set[str],
    ) -> str | None:
        op = node.op.name
        bin_ops = {
            "core.binary.*": "*",
            "core.binary./": "/",
            "core.binary.+": "+",
            "core.binary.-": "-",
            "core.binary.%": "%",
            "core.binary.==": "==",
            "core.binary.!=": "!=",
            "core.binary.>=": ">=",
            "core.binary.and": "and",
        }
        if op in bin_ops and len(node.inputs) >= 2:
            left = self._trace_dim_expr(node.inputs[0], repeated_mod, loop_var, set(visited))
            right = self._trace_dim_expr(node.inputs[1], repeated_mod, loop_var, set(visited))
            if left and right:
                return _simplify_binop(op, left, right)
        elif op in ("_tensor_size", "tensor_size") and len(node.inputs) >= 2:
            return self._trace_tensor_dim_at_index(
                node.inputs[0],
                node.inputs[1],
                repeated_mod,
                loop_var,
                set(visited),
            )
        elif op == "_list_index" and len(node.inputs) >= 2:
            return self._trace_shape_list_index(
                node.inputs[0],
                node.inputs[1],
                repeated_mod,
                loop_var,
                set(visited),
            )
        elif op in ("_config_int", "_config_dim", "_config_value") and node.inputs:
            expr = self._config_lookup_expr(node.inputs, default_none=None)
            if expr is not None:
                return expr
        elif op == "_config_float" and node.inputs:
            expr = self._config_lookup_expr(node.inputs, default_none=0.0)
            if expr is not None:
                return expr
        elif op == "_config_bool" and node.inputs:
            expr = self._config_lookup_expr(node.inputs, default_none=False)
            if expr is not None:
                return expr
        elif op == "_config_has" and node.inputs:
            expr = self._config_has_expr(node.inputs[0])
            if expr is not None:
                return expr
        elif op == "_params_has_root" and node.inputs:
            root = _literal_value(node.inputs[0], None)
            if isinstance(root, str):
                return self._params_has_root_expr(root)
        elif op == "core.select" and len(node.inputs) >= 3:
            cond = self._trace_dim_expr(node.inputs[0], repeated_mod, loop_var, set(visited))
            true_val = self._trace_dim_expr(node.inputs[1], repeated_mod, loop_var, set(visited))
            false_val = self._trace_dim_expr(node.inputs[2], repeated_mod, loop_var, set(visited))
            if cond and true_val and false_val:
                return f"({true_val} if {cond} else {false_val})"
        elif op == "_sqrt" and node.inputs:
            inner = self._trace_dim_expr(node.inputs[0], repeated_mod, loop_var, set(visited))
            if inner:
                return f"(({inner}) ** 0.5)"
        elif op in self.modules_by_name:
            return self._trace_module_call_dim_expr(
                op, node.inputs, repeated_mod, loop_var, visited
            )
        return None

    def _trace_expr_dim_expr(
        self,
        expr: GraphExpr,
        repeated_mod: Any,
        loop_var: str,
        visited: set[str],
    ) -> str | None:
        op = expr.op.name
        bin_ops = {
            "core.binary.*": "*",
            "core.binary./": "/",
            "core.binary.+": "+",
            "core.binary.-": "-",
            "core.binary.%": "%",
            "core.binary.==": "==",
            "core.binary.!=": "!=",
            "core.binary.>=": ">=",
            "core.binary.and": "and",
        }
        if op in bin_ops and len(expr.inputs) >= 2:
            left = self._trace_dim_expr(expr.inputs[0], repeated_mod, loop_var, set(visited))
            right = self._trace_dim_expr(expr.inputs[1], repeated_mod, loop_var, set(visited))
            if left and right:
                return _simplify_binop(op, left, right)
        elif op in ("_tensor_size", "tensor_size") and len(expr.inputs) >= 2:
            return self._trace_tensor_dim_at_index(
                expr.inputs[0],
                expr.inputs[1],
                repeated_mod,
                loop_var,
                set(visited),
            )
        elif op == "_list_index" and len(expr.inputs) >= 2:
            return self._trace_shape_list_index(
                expr.inputs[0],
                expr.inputs[1],
                repeated_mod,
                loop_var,
                set(visited),
            )
        elif op in ("_config_int", "_config_dim", "_config_value") and expr.inputs:
            return self._config_lookup_expr(expr.inputs, default_none=None)
        elif op == "_config_float" and expr.inputs:
            return self._config_lookup_expr(expr.inputs, default_none=0.0)
        elif op == "_config_bool" and expr.inputs:
            return self._config_lookup_expr(expr.inputs, default_none=False)
        elif op == "_config_has" and expr.inputs:
            return self._config_has_expr(expr.inputs[0])
        elif op == "_params_has_root" and expr.inputs:
            root = _literal_value(expr.inputs[0], None)
            if isinstance(root, str):
                return self._params_has_root_expr(root)
            return None
        elif op == "core.select" and len(expr.inputs) >= 3:
            cond = self._trace_dim_expr(expr.inputs[0], repeated_mod, loop_var, set(visited))
            true_val = self._trace_dim_expr(expr.inputs[1], repeated_mod, loop_var, set(visited))
            false_val = self._trace_dim_expr(expr.inputs[2], repeated_mod, loop_var, set(visited))
            if cond and true_val and false_val:
                return f"({true_val} if {cond} else {false_val})"
            return None
        elif op == "_sqrt" and expr.inputs:
            inner = self._trace_dim_expr(expr.inputs[0], repeated_mod, loop_var, set(visited))
            if inner:
                return f"(({inner}) ** 0.5)"
        elif op in self.modules_by_name:
            return self._trace_module_call_dim_expr(
                op, expr.inputs, repeated_mod, loop_var, visited
            )
        return None

    def _trace_module_call_dim_expr(
        self,
        module_name: str,
        args: tuple[GraphOperand, ...],
        caller_mod: Any,
        loop_var: str,
        visited: set[str],
    ) -> str | None:
        callee = self.modules_by_name.get(module_name)
        if callee is None or not callee.outputs:
            return None
        call_key = f"call:{module_name}:{tuple(_value_name(arg) or repr(arg) for arg in args)}"
        if call_key in visited:
            return None
        bindings: dict[str, GraphOperand] = {}
        for idx, inp in enumerate(callee.inputs):
            if idx >= len(args):
                break
            name = getattr(inp, "name", None)
            if name:
                bindings[name] = args[idx]
        callee_visited = {item for item in visited if item.startswith("call:")}
        callee_visited.add(call_key)
        return self._trace_bound_dim_expr(
            callee.outputs[0],
            callee,
            bindings,
            caller_mod,
            loop_var,
            callee_visited,
        )

    def _bind_operand(
        self,
        operand: GraphOperand,
        bindings: dict[str, GraphOperand],
    ) -> GraphOperand:
        if isinstance(operand, GraphValueRef) and operand.name in bindings:
            return bindings[operand.name]
        if isinstance(operand, GraphExpr):
            return GraphExpr(
                op=operand.op,
                inputs=tuple(self._bind_operand(item, bindings) for item in operand.inputs),
                attrs=operand.attrs,
                type_expr=operand.type_expr,
                dims=operand.dims,
            )
        return operand

    def _trace_bound_dim_expr(
        self,
        operand: GraphOperand,
        callee_mod: Any,
        bindings: dict[str, GraphOperand],
        caller_mod: Any,
        loop_var: str,
        visited: set[str],
    ) -> str | None:
        bound = self._bind_operand(operand, bindings)
        if bound is not operand:
            return self._trace_dim_expr(bound, caller_mod, loop_var, set(visited))
        if isinstance(operand, GraphLiteral):
            if operand.value is None:
                return "None"
            if isinstance(operand.value, str):
                return repr(operand.value)
            return str(operand.value)
        if isinstance(operand, GraphValueRef):
            name = operand.name
            if name in visited:
                return None
            for node in callee_mod.nodes:
                if any(getattr(out, "name", None) == name for out in node.outputs):
                    return self._trace_bound_node_dim_expr(
                        node,
                        callee_mod,
                        bindings,
                        caller_mod,
                        loop_var,
                        set(visited) | {name},
                    )
            const_val = self._resolve_const_value(name, set(visited))
            if const_val is not None:
                return const_val
        if isinstance(operand, GraphExpr):
            return self._trace_bound_expr_dim_expr(
                operand,
                callee_mod,
                bindings,
                caller_mod,
                loop_var,
                visited,
            )
        return None

    def _trace_bound_node_dim_expr(
        self,
        node: GraphNode,
        callee_mod: Any,
        bindings: dict[str, GraphOperand],
        caller_mod: Any,
        loop_var: str,
        visited: set[str],
    ) -> str | None:
        op = node.op.name
        if op in _BINOP_SYMBOLS and len(node.inputs) >= 2:
            left = self._trace_bound_dim_expr(
                node.inputs[0], callee_mod, bindings, caller_mod, loop_var, set(visited)
            )
            right = self._trace_bound_dim_expr(
                node.inputs[1], callee_mod, bindings, caller_mod, loop_var, set(visited)
            )
            if left and right:
                return _simplify_binop(op, left, right)
        bound_inputs = tuple(self._bind_operand(item, bindings) for item in node.inputs)
        if op in ("_tensor_size", "tensor_size") and len(bound_inputs) >= 2:
            return self._trace_tensor_dim_at_index(
                bound_inputs[0],
                bound_inputs[1],
                caller_mod,
                loop_var,
                set(visited),
            )
        if op == "_list_index" and len(bound_inputs) >= 2:
            return self._trace_shape_list_index(
                bound_inputs[0],
                bound_inputs[1],
                caller_mod,
                loop_var,
                set(visited),
            )
        if op in ("_config_int", "_config_dim", "_config_value") and bound_inputs:
            return self._config_lookup_expr(bound_inputs, default_none=None)
        if op == "_config_float" and bound_inputs:
            return self._config_lookup_expr(bound_inputs, default_none=0.0)
        if op == "_config_bool" and bound_inputs:
            return self._config_lookup_expr(bound_inputs, default_none=False)
        if op == "_config_has" and bound_inputs:
            return self._config_has_expr(bound_inputs[0])
        if op == "_params_has_root" and bound_inputs:
            root = _literal_value(bound_inputs[0], None)
            if isinstance(root, str):
                return self._params_has_root_expr(root)
        if op == "core.select" and len(node.inputs) >= 3:
            cond = self._trace_bound_dim_expr(
                node.inputs[0], callee_mod, bindings, caller_mod, loop_var, set(visited)
            )
            true_val = self._trace_bound_dim_expr(
                node.inputs[1], callee_mod, bindings, caller_mod, loop_var, set(visited)
            )
            false_val = self._trace_bound_dim_expr(
                node.inputs[2], callee_mod, bindings, caller_mod, loop_var, set(visited)
            )
            if cond and true_val and false_val:
                return f"({true_val} if {cond} else {false_val})"
        if op == "_sqrt" and node.inputs:
            inner = self._trace_bound_dim_expr(
                node.inputs[0], callee_mod, bindings, caller_mod, loop_var, set(visited)
            )
            if inner:
                return f"(({inner}) ** 0.5)"
        if op in self.modules_by_name:
            return self._trace_module_call_dim_expr(
                op, bound_inputs, caller_mod, loop_var, visited
            )
        return None

    def _trace_bound_expr_dim_expr(
        self,
        expr: GraphExpr,
        callee_mod: Any,
        bindings: dict[str, GraphOperand],
        caller_mod: Any,
        loop_var: str,
        visited: set[str],
    ) -> str | None:
        op = expr.op.name
        if op in _BINOP_SYMBOLS and len(expr.inputs) >= 2:
            left = self._trace_bound_dim_expr(
                expr.inputs[0], callee_mod, bindings, caller_mod, loop_var, set(visited)
            )
            right = self._trace_bound_dim_expr(
                expr.inputs[1], callee_mod, bindings, caller_mod, loop_var, set(visited)
            )
            if left and right:
                return _simplify_binop(op, left, right)
        bound_inputs = tuple(self._bind_operand(item, bindings) for item in expr.inputs)
        if op in ("_tensor_size", "tensor_size") and len(bound_inputs) >= 2:
            return self._trace_tensor_dim_at_index(
                bound_inputs[0],
                bound_inputs[1],
                caller_mod,
                loop_var,
                set(visited),
            )
        if op == "_list_index" and len(bound_inputs) >= 2:
            return self._trace_shape_list_index(
                bound_inputs[0],
                bound_inputs[1],
                caller_mod,
                loop_var,
                set(visited),
            )
        if op in ("_config_int", "_config_dim", "_config_value") and bound_inputs:
            return self._config_lookup_expr(bound_inputs, default_none=None)
        if op == "_config_float" and bound_inputs:
            return self._config_lookup_expr(bound_inputs, default_none=0.0)
        if op == "_config_bool" and bound_inputs:
            return self._config_lookup_expr(bound_inputs, default_none=False)
        if op == "_config_has" and bound_inputs:
            return self._config_has_expr(bound_inputs[0])
        if op == "_params_has_root" and bound_inputs:
            root = _literal_value(bound_inputs[0], None)
            if isinstance(root, str):
                return self._params_has_root_expr(root)
        if op == "core.select" and len(expr.inputs) >= 3:
            cond = self._trace_bound_dim_expr(
                expr.inputs[0], callee_mod, bindings, caller_mod, loop_var, set(visited)
            )
            true_val = self._trace_bound_dim_expr(
                expr.inputs[1], callee_mod, bindings, caller_mod, loop_var, set(visited)
            )
            false_val = self._trace_bound_dim_expr(
                expr.inputs[2], callee_mod, bindings, caller_mod, loop_var, set(visited)
            )
            if cond and true_val and false_val:
                return f"({true_val} if {cond} else {false_val})"
        if op == "_sqrt" and expr.inputs:
            inner = self._trace_bound_dim_expr(
                expr.inputs[0], callee_mod, bindings, caller_mod, loop_var, set(visited)
            )
            if inner:
                return f"(({inner}) ** 0.5)"
        if op in self.modules_by_name:
            return self._trace_module_call_dim_expr(
                op, bound_inputs, caller_mod, loop_var, visited
            )
        return None

    def _params_has_root_expr(self, root: str) -> str:
        return (
            f"any(k == {root!r} or k.startswith({root!r} + '.') "
            "for k in self._loaded_state_keys)"
        )

    def _tensor_type_dims(self, operand: GraphOperand) -> tuple[Any, ...] | None:
        dims = getattr(operand, "dims", None)
        if dims is not None:
            return tuple(dims)
        type_expr = getattr(operand, "type_expr", None)
        while isinstance(type_expr, TypeOptional):
            type_expr = type_expr.inner
        if isinstance(type_expr, TypeTensor):
            return tuple(type_expr.dims)
        return None

    @staticmethod
    def _is_mask_input(type_expr: Any) -> bool:
        while isinstance(type_expr, TypeOptional):
            type_expr = type_expr.inner
        inner = type_expr
        if isinstance(inner, TypeNamed) and inner.name == "Mask":
            return True
        if isinstance(inner, TypeTensor):
            return len(inner.dims) == 2
        return False

    def _static_index_value(
        self,
        operand: GraphOperand,
        repeated_mod: Any,
        loop_var: str,
        visited: set[str],
    ) -> int | None:
        value = _literal_value(operand, None)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        traced = self._trace_dim_expr(operand, repeated_mod, loop_var, set(visited))
        if traced is None:
            return None
        try:
            return int(traced)
        except ValueError:
            return None

    def _trace_tensor_dim_at_index(
        self,
        tensor_operand: GraphOperand,
        index_operand: GraphOperand,
        repeated_mod: Any,
        loop_var: str,
        visited: set[str],
    ) -> str | None:
        dims = self._tensor_type_dims(tensor_operand)
        if not dims:
            return None
        index = self._static_index_value(index_operand, repeated_mod, loop_var, set(visited))
        if index is None:
            return None
        if index < 0:
            index += len(dims)
        if index < 0 or index >= len(dims):
            return None
        return self._dim_expr_to_python(dims[index], repeated_mod, loop_var)

    def _trace_shape_list_index(
        self,
        shape_operand: GraphOperand,
        index_operand: GraphOperand,
        repeated_mod: Any,
        loop_var: str,
        visited: set[str],
    ) -> str | None:
        if isinstance(shape_operand, GraphExpr) and shape_operand.op.name == "_shape" and shape_operand.inputs:
            return self._trace_tensor_dim_at_index(
                shape_operand.inputs[0],
                index_operand,
                repeated_mod,
                loop_var,
                set(visited),
            )
        if isinstance(shape_operand, GraphValueRef):
            name = shape_operand.name
            if name in visited:
                return None
            for node in repeated_mod.nodes:
                if (
                    node.op.name == "_shape"
                    and node.inputs
                    and any(getattr(out, "name", None) == name for out in node.outputs)
                ):
                    return self._trace_tensor_dim_at_index(
                        node.inputs[0],
                        index_operand,
                        repeated_mod,
                        loop_var,
                        set(visited) | {name},
                    )
        return None

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
            if optional:
                return (
                    f"(self._vllm_state_tensor({key!r}, self.state_dict_tensors[{key!r}]) "
                    f"if {key!r} in self.state_dict_tensors else None)"
                )
            return f"self._vllm_state_tensor({key!r}, self.state_dict_tensors[{key!r}])"
        return super()._param_expr_for_path(
            base,
            leaf,
            optional=optional,
            local=local,
            symbols_dict=symbols_dict,
        )

    def _config_lookup_expr(
        self,
        inputs: tuple[GraphOperand, ...],
        *,
        default_none: object | None,
    ) -> str | None:
        path_inp = inputs[0] if inputs else None
        if not isinstance(path_inp, GraphPath) or not path_inp.parts:
            return None
        field = self._config_field_from_path(path_inp)
        if field == "head_dim":
            return self._head_dim_expr()
        default = _literal_value(inputs[1], None) if len(inputs) >= 2 else None
        if isinstance(default, int):
            return self._config_expr(field, default=default)
        if isinstance(default, float):
            return self._config_expr(field, default=default)
        if default is not None:
            return self._config_expr(field, default=default)
        if len(inputs) >= 2:
            names = (field,) + _CONFIG_ALIASES.get(field, ())
            expr = "None"
            for name in reversed(names):
                expr = (
                    f"(getattr(self.config, {name!r}, None) "
                    f"if getattr(self.config, {name!r}, None) is not None "
                    f"else self._model_config.get({name!r}, {expr}))"
                )
            return expr
        if default_none is not None:
            return self._config_expr(field, default=default_none)
        return self._config_expr(field)

    def _config_has_expr(self, operand: GraphOperand) -> str | None:
        if not isinstance(operand, GraphPath) or not operand.parts:
            return None
        key = ".".join(operand.parts)
        leaf = operand.parts[-1]
        return (
            f"(getattr(self.config, {leaf!r}, None) is not None "
            f"or self._model_config.get({key!r}) is not None)"
        )

    def _detect_head_dim_expr(self, repeated_mod: Any, loop_var: str = "i") -> str | None:
        """Detect per-layer head_dim expression from core.select in repeated module.

        When multiple selects with literal int branches exist, prefer the one
        whose values are closest to a typical head_dim range (filtering out
        very small values like head counts, and very large values like
        projection sizes) by picking the candidate with the smallest max value.
        """
        candidates: list[tuple[int, int, str, int, int]] = []
        for node in repeated_mod.nodes:
            if node.op.name == "core.select" and len(node.inputs) >= 3:
                true_val = _literal_value(node.inputs[1], None)
                false_val = _literal_value(node.inputs[2], None)
                if isinstance(true_val, int) and isinstance(false_val, int) and true_val != false_val:
                    cond = self._trace_dim_expr(node.inputs[0], repeated_mod, loop_var)
                    if cond:
                        lo = min(true_val, false_val)
                        hi = max(true_val, false_val)
                        candidates.append((lo, hi, cond, true_val, false_val))
        if not candidates:
            return None
        # Filter out very small values (head counts) and prefer smallest max
        # (head_dim is smaller than derived projection sizes).
        filtered = [c for c in candidates if c[0] >= 32]
        pool = filtered if filtered else candidates
        pool.sort(key=lambda c: c[1])
        lo, hi, cond, true_val, false_val = pool[0]
        return f"({true_val} if {cond} else {false_val})"

    def _detect_kv_heads_expr(self, repeated_mod: Any, loop_var: str = "i") -> str | None:
        """Detect per-layer KV head count expression from core.select in repeated module.

        Looks for selects with small literal int values (< 32) that represent
        head counts rather than head dimensions.  Among multiple candidates,
        prefers the one with the smallest max value.
        """
        candidates: list[tuple[int, int, str, int, int]] = []
        for node in repeated_mod.nodes:
            if node.op.name == "core.select" and len(node.inputs) >= 3:
                true_val = _literal_value(node.inputs[1], None)
                false_val = _literal_value(node.inputs[2], None)
                if isinstance(true_val, int) and isinstance(false_val, int) and true_val != false_val:
                    lo = min(true_val, false_val)
                    hi = max(true_val, false_val)
                    if hi < 32:
                        cond = self._trace_dim_expr(node.inputs[0], repeated_mod, loop_var)
                        if cond:
                            candidates.append((lo, hi, cond, true_val, false_val))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[1])
        lo, hi, cond, true_val, false_val = candidates[0]
        return f"({true_val} if {cond} else {false_val})"

    def _attention_head_dim_expr(
        self,
        node: GraphNode,
        repeated_mod: Any | None,
        loop_var: str = "i",
    ) -> str | None:
        if not node.inputs:
            return None
        dims = self._tensor_type_dims(node.inputs[0])
        if not dims:
            return None
        mod = repeated_mod or self.modules_by_name.get(self._node_module_name(node))
        if mod is None:
            return None
        rendered = self._dim_expr_to_python(dims[-1], mod, loop_var)
        if rendered is None:
            return None
        # A missing optional head_dim config field must not become head_size=0
        # for vLLM Attention. If graph typing traces to that invalid fallback,
        # use the standard hidden_size / num_attention_heads derivation.
        if "head_dim" in rendered and "_model_config.get('head_dim', 0)" in rendered:
            return self._head_dim_expr()
        return rendered

    def _node_module_name(self, node: Any) -> str:
        source_module = getattr(node, "source_module", None)
        if isinstance(source_module, str) and source_module in self.modules_by_name:
            return source_module
        nid = getattr(node, "id", "")
        return nid.rsplit(":", 1)[0] if ":" in nid else ""

    def _is_repeated_node(self, node: Any) -> bool:
        nid = getattr(node, "id", "")
        for mod_name in self._vllm_classification.repeated_module_names:
            if nid == mod_name or nid.startswith(mod_name + ":"):
                return True
        return False

    def _is_vllm_repeated_layer_node(self, node: GraphNode) -> bool:
        if self._is_repeated_node(node) or "{i}" in self._node_prefix(node):
            return True
        if self._vllm_classification.node_types.get(node.id) == VLLMLayerType.ATTENTION:
            group = next(
                (
                    g
                    for g in self._vllm_classification.qkv_groups
                    if g.attention_node_id == node.id
                ),
                None,
            )
            q_node = self._find_node_by_id(group.q_node_id) if group is not None else None
            if q_node is not None:
                return self._is_repeated_node(q_node) or "{i}" in self._node_prefix(q_node)
        return False

    def _get_repeated_module(self) -> Any:
        cls = self._vllm_classification
        if not cls.repeated_module_names:
            return None
        name = max(
            cls.repeated_module_names,
            key=lambda n: len(self.modules_by_name[n].nodes) if n in self.modules_by_name else 0,
        )
        return self.modules_by_name.get(name)

    def _get_node_modules(self, node: Any) -> list:
        """Return the node's own module and the primary repeated module, in priority order."""
        mods = []
        node_mod_name = self._node_module_name(node)
        if node_mod_name and node_mod_name in self.modules_by_name:
            mods.append(self.modules_by_name[node_mod_name])
        primary = self._get_repeated_module()
        if primary is not None and primary not in mods:
            mods.append(primary)
        return mods

    def _node_loop_index(self, node: Any) -> str:
        mod = self._node_module_name(node)
        return self._vllm_classification.loop_index_param.get(mod, "i")

    def _vllm_repeated_layer_index_expr(
        self,
        node: Any,
        local: set[str] | dict[str, str],
        *,
        anchor_exprs: Iterable[str] = (),
    ) -> str | None:
        local_names = set(local.values()) if isinstance(local, dict) else set(local)
        anchor_texts = tuple(str(expr) for expr in anchor_exprs)
        index_name = _safe_ident(self._node_loop_index(node))
        if index_name in local_names:
            return index_name

        for output in getattr(node, "outputs", ()):
            output_name = getattr(output, "name", "")
            match = re.match(r"^(.+)_\d+__", output_name)
            if match:
                candidate = f"{match.group(1)}_{index_name}"
                if candidate in local_names:
                    return candidate

        node_id = str(getattr(node, "id", ""))
        mod = self._node_module_name(node)
        scope_parts = self._vllm_classification.module_scope_parts.get(mod)

        loop_names = [index_name]
        if scope_parts:
            for part in scope_parts:
                for match in re.finditer(r"\{([^}]+)\}", part):
                    safe = _safe_ident(match.group(1))
                    if safe not in loop_names:
                        loop_names.append(safe)

        module = self.modules_by_name.get(mod)
        if scope_parts and module is not None and module.inputs:
            loop_token = "{" + self._node_loop_index(node) + "}"
            try:
                segment_index = next(
                    i for i, part in enumerate(scope_parts) if loop_token in part
                )
            except StopIteration:
                segment_index = None
            scope_param = _value_name(module.inputs[0])
            if segment_index is not None and scope_param:
                scope_local = (
                    local.get(scope_param) or local.get(_safe_ident(scope_param))
                    if isinstance(local, dict)
                    else _safe_ident(scope_param)
                )
                if scope_local in local_names:
                    return f"self._loop_index_from_scope({scope_local}, {segment_index})"

        if anchor_texts:
            scoped_locals: list[tuple[int, int, str]] = []
            for name in local_names:
                if not name.endswith("__scope"):
                    continue
                prefix = name[: -len("__scope")]
                if not prefix:
                    continue
                score = sum(
                    1
                    for anchor in anchor_texts
                    if anchor.startswith(prefix) or prefix in anchor
                )
                if score:
                    scoped_locals.append((score, len(prefix), name))
            if scoped_locals:
                scoped_locals.sort(reverse=True)
                return f"self._loop_index_from_scope({scoped_locals[0][2]}, -1)"

        # Repeated modules are often emitted through the generic repeat inliner.
        # The inliner substitutes a callee parameter `i` as
        # `_loop_inline_<repeat-node-id>_i`; use that generated binding when it is
        # the only live canonical candidate in the current emitted scope.
        for loop_name in loop_names:
            candidates = sorted(
                name
                for name in local_names
                if "loop_inline" in name and name.endswith(f"_{loop_name}")
            )
            canonical_candidates = [
                name
                for name in candidates
                if re.match(rf"^.*_\d+_{re.escape(loop_name)}$", name)
            ]
            if len(canonical_candidates) == 1:
                return canonical_candidates[0]
            if len(canonical_candidates) > 1:
                scored = []
                for candidate in canonical_candidates:
                    prefix = candidate[: -(len(loop_name) + 1)]
                    anchor_score = sum(
                        1
                        for anchor in anchor_texts
                        if anchor.startswith(prefix + "_") or prefix in anchor
                    )
                    support = sum(
                        1 for name in local_names if name.startswith(prefix + "_")
                    )
                    scored.append((anchor_score, support, candidate))
                scored.sort(reverse=True)
                if len(scored) == 1 or scored[0][:2] > scored[1][:2]:
                    return scored[0][2]
            if len(candidates) == 1:
                return candidates[0]

        default_canonical_i = [
            name
            for name in local_names
            if "loop_inline" in name and re.match(r"^.*_\d+_i$", name)
        ]
        if len(default_canonical_i) == 1:
            return default_canonical_i[0]
        if len(default_canonical_i) > 1:
            scored = []
            for candidate in default_canonical_i:
                prefix = candidate[:-2]
                support = sum(1 for name in local_names if name.startswith(prefix + "_"))
                scored.append((support, candidate))
            scored.sort(reverse=True)
            if len(scored) == 1 or scored[0][0] > scored[1][0]:
                return scored[0][1]

        if ":inline:" in node_id:
            repeat_node_id = node_id.split(":inline:", 1)[0]
            repeat_prefix = _safe_ident(
                "__loop_inline_" + repeat_node_id.replace(":", "_")
            )
            candidate = f"{repeat_prefix}_{index_name}"
            if candidate in local_names:
                return candidate

        return None

    def _vllm_attr_access(
        self,
        node: Any,
        *,
        local: set[str] | dict[str, str] | None = None,
        index_node: Any | None = None,
        anchor_exprs: Iterable[str] = (),
    ) -> str:
        attr = self._vllm_layer_attr_name(node)
        if self._is_vllm_repeated_layer_node(node):
            index_node = node if index_node is None else index_node
            idx = (
                _safe_ident(self._node_loop_index(node))
                if local is None
                else self._vllm_repeated_layer_index_expr(
                    index_node,
                    local,
                    anchor_exprs=anchor_exprs,
                )
            )
            if not idx:
                local_hint = ""
                if local is not None:
                    local_names = set(local.values()) if isinstance(local, dict) else set(local)
                    loopish = sorted(name for name in local_names if "loop" in name or name.endswith("_i"))
                    local_hint = f"; loop candidates={loopish[:12]!r}"
                raise ValueError(
                    f"cannot resolve loop index for repeated vLLM layer {node.id!r}"
                    f"{local_hint}"
                )
            return f"self.{attr}[{idx}]"
        return f"self.{attr}"

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
        node_id = getattr(node, "id", None)
        if node_id is not None:
            layer_type = self._vllm_classification.node_types.get(
                node_id, VLLMLayerType.DEFAULT
            )
            if (
                layer_type != VLLMLayerType.DEFAULT
                and self._use_clean_forward
            ):
                if self._emit_vllm_layer_call(
                    lines, node, layer_type,
                    targets=targets, indent=indent, local=local,
                    symbols_dict=symbols_dict,
                ):
                    return True
        return super()._emit_direct_module_call_node(
            lines, node,
            targets=targets, module_name=module_name,
            indent=indent, local=local, symbols_dict=symbols_dict,
        )

    def _emit_vllm_layer_call(
        self,
        lines: list[str],
        node: Any,
        layer_type: VLLMLayerType,
        *,
        targets: tuple[str, ...],
        indent: int,
        local: set[str],
        symbols_dict: str,
    ) -> bool:
        add = self._add
        local_dict = {k: k for k in local} if isinstance(local, set) else local
        args = self._collect_args(node, local_dict)

        if layer_type == VLLMLayerType.VOCAB_PARALLEL_EMBEDDING:
            attr = self._vllm_attr_access(node, local=local)
            if len(args) < 2:
                return False
            expr = f"{attr}({args[1]})"
        elif layer_type == VLLMLayerType.QKV_PARALLEL_LINEAR:
            attr = self._vllm_attr_access(node, local=local)
            if len(args) < 2:
                return False
            expr = f"{attr}({args[1]})[0]"
        elif layer_type in (VLLMLayerType.COLUMN_PARALLEL_LINEAR, VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR):
            attr = self._vllm_attr_access(node, local=local)
            if len(args) < 2:
                return False
            expr = f"{attr}({args[1]})[0]"
        elif layer_type == VLLMLayerType.ROW_PARALLEL_LINEAR:
            attr = self._vllm_attr_access(node, local=local)
            if len(args) < 2:
                return False
            expr = f"{attr}({args[1]})[0]"
        elif layer_type == VLLMLayerType.PARALLEL_LM_HEAD:
            hidden_expr = self._lm_head_hidden_expr(
                node,
                local=local_dict,
                symbols_dict=symbols_dict,
            )
            if hidden_expr is not None:
                expr = hidden_expr
            elif len(args) < 2:
                return False
            else:
                expr = args[1]
        elif layer_type == VLLMLayerType.ATTENTION:
            attr = self._vllm_attr_access(node, local=local, anchor_exprs=args)
            expr = self._vllm_attention_call_expr(
                attr,
                args,
                node=node,
                local=local_dict,
                output_rank=self._node_output_rank(node),
            )
        elif layer_type == VLLMLayerType.RMSNORM:
            attr = self._vllm_attr_access(node, local=local)
            if len(args) < 1:
                return False
            data_idx = 1 if (len(node.inputs) >= 2 and isinstance(node.inputs[0], GraphPath)) else 0
            if data_idx >= len(args):
                return False
            expr = f"{attr}({args[data_idx]})"
        elif layer_type == VLLMLayerType.LAYERNORM:
            attr = self._vllm_attr_access(node, local=local)
            if len(args) < 1:
                return False
            data_idx = 1 if (len(node.inputs) >= 2 and isinstance(node.inputs[0], GraphPath)) else 0
            if data_idx >= len(args):
                return False
            expr = f"{attr}({args[data_idx]})"
        else:
            return False

        joined = ", ".join(targets)
        add(lines, indent, f"{joined} = {expr}")
        return True

    def _emit_node(
        self,
        lines: list[str],
        node: Any,
        *,
        module_name: str,
        indent: int,
        local: set[str],
        symbols_dict: str,
    ) -> None:
        if (
            self._use_clean_forward
            and
            module_name == self.program.main_module
            and len(getattr(node, "outputs", ())) == 1
            and graph_main_output_names(self.program, self.modules_by_name[module_name])
            and node.outputs[0].name
            == graph_main_output_names(self.program, self.modules_by_name[module_name])[0]
        ):
            hidden_expr = self._lm_head_hidden_expr(
                node,
                local={name: name for name in local},
                symbols_dict=symbols_dict,
            )
            if hidden_expr is not None:
                self._add(lines, indent, f"{node.outputs[0].name} = {hidden_expr}")
                return
        super()._emit_node(
            lines,
            node,
            module_name=module_name,
            indent=indent,
            local=local,
            symbols_dict=symbols_dict,
        )

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
        eps = (
            self._operand_expr(node.inputs[2], local=local, symbols_dict=symbols_dict)
            if len(node.inputs) > 2
            else "1e-5"
        )
        bias_flag = (
            self._operand_expr(node.inputs[5], local=local, symbols_dict=symbols_dict)
            if len(node.inputs) > 5
            else "True"
        )
        bias_literal = self._literal_bool_arg(node.inputs[5]) if len(node.inputs) > 5 else True
        weight_leaf = node.inputs[4] if len(node.inputs) > 4 else "weight"
        bias_leaf = node.inputs[6] if len(node.inputs) > 6 else "bias"
        weight_key = self._static_param_key(node.inputs[0], weight_leaf)
        bias_key = self._static_param_key(node.inputs[0], bias_leaf)
        if weight_key is not None:
            weight = (
                f"self._vllm_state_tensor({weight_key!r}, "
                f"self.state_dict_tensors[{weight_key!r}])"
            )
        else:
            weight = self._param_expr_for_path(
                node.inputs[0],
                weight_leaf,
                local=local,
                symbols_dict=symbols_dict,
            )
        if bias_key is not None:
            bias_value = (
                f"(self._vllm_state_tensor({bias_key!r}, self.state_dict_tensors[{bias_key!r}]) "
                f"if {bias_key!r} in self.state_dict_tensors else None)"
            )
        else:
            bias_value = self._param_expr_for_path(
                node.inputs[0],
                bias_leaf,
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
            flag_expr=bias_flag,
            flag_literal=bias_literal,
            indent=indent,
        )
        x_expr = f"{args[1]}.to({weight_name}.device)"
        bias_expr = f"({bias_name}.to({weight_name}.device) if {bias_name} is not None else None)"
        op_expr = (
            f"F.layer_norm({x_expr}, "
            f"({args[1]}.shape[-1],), weight={weight_name}, "
            f"bias={bias_expr}, eps=float({eps}))"
        )
        if self.profile:
            self._add(lines, indent, f"{target} = self._profile_call({f'node:{target}:_layernorm'!r}, lambda: {op_expr})")
        else:
            self._add(lines, indent, f"{target} = {op_expr}")
        return True

    def _lm_head_hidden_expr(
        self,
        node: GraphNode,
        *,
        local: dict[str, str],
        symbols_dict: str,
    ) -> str | None:
        def is_linear_expr(expr: GraphExpr) -> bool:
            if expr.op.name == "_linear":
                return True
            module = self.modules_by_name.get(expr.op.name)
            return module is not None and any(
                inner.op.name == "_linear" for inner in module.nodes
            )

        def hidden_operand(operand: GraphOperand) -> GraphOperand | None:
            if not isinstance(operand, GraphExpr):
                return None
            if is_linear_expr(operand) and len(operand.inputs) >= 2:
                return operand.inputs[1]
            if operand.op.name == "core.select" and len(operand.inputs) >= 3:
                then_hidden = hidden_operand(operand.inputs[1])
                else_hidden = hidden_operand(operand.inputs[2])
                if then_hidden is None or else_hidden is None:
                    return None
                then_expr = self._operand_expr(
                    then_hidden,
                    local=local,
                    symbols_dict=symbols_dict,
                )
                else_expr = self._operand_expr(
                    else_hidden,
                    local=local,
                    symbols_dict=symbols_dict,
                )
                return then_hidden if then_expr == else_expr else None
            return None

        if node.op.name in ("core.alias", "core.ascribe") and node.inputs:
            hidden = hidden_operand(node.inputs[0])
            if hidden is not None:
                return self._operand_expr(hidden, local=local, symbols_dict=symbols_dict)
        if node.op.name == "core.select" and len(node.inputs) >= 3:
            then_hidden = hidden_operand(node.inputs[1])
            else_hidden = hidden_operand(node.inputs[2])
            if then_hidden is not None and else_hidden is not None:
                then_expr = self._operand_expr(
                    then_hidden,
                    local=local,
                    symbols_dict=symbols_dict,
                )
                else_expr = self._operand_expr(
                    else_hidden,
                    local=local,
                    symbols_dict=symbols_dict,
                )
                if then_expr == else_expr:
                    return then_expr
        if _is_linear_call(node, self.modules_by_name) and len(node.inputs) >= 2:
            return self._operand_expr(node.inputs[1], local=local, symbols_dict=symbols_dict)
        return None

    def _detect_rope(self, repeated_mod: Any) -> str | None:
        def _search_expr(expr, depth=0):
            if depth > 5:
                return False
            if "rope" in expr.op.name.lower():
                return True
            inner_mod = self.modules_by_name.get(expr.op.name)
            if inner_mod is not None:
                for inner_node in inner_mod.nodes:
                    if "rope" in inner_node.op.name.lower():
                        return True
            for inp in expr.inputs:
                if isinstance(inp, GraphExpr) and _search_expr(inp, depth + 1):
                    return True
            return False

        for node in repeated_mod.nodes:
            if "rope" in node.op.name.lower():
                return node.id
            for inp in node.inputs:
                if isinstance(inp, GraphExpr) and _search_expr(inp):
                    return node.id
        return None

    def _detect_rope_variants(self) -> tuple[str, str, str, str, str, str, str, str, str] | None:
        """Detect per-layer RoPE variants (local vs full) from top-level module.

        Returns
        (local_hd, local_theta, local_scale, local_partial,
         full_hd, full_theta, full_scale, full_partial, rope_period_expr)
        or None.
        """
        rope_calls: list[tuple[str, str, str, str]] = []
        for module in self.program.modules:
            for node in module.nodes:
                op_name = node.op.name.lower()
                if (
                    "rope_base_factors" in op_name
                    and len(node.inputs) >= 2
                ):
                    if len(node.inputs) >= 3:
                        hd = self._trace_dim_expr(node.inputs[1], module, "i", set())
                        theta = self._trace_dim_expr(node.inputs[2], module, "i", set())
                    else:
                        # Specialized/materialized forms may already have the
                        # rotary dimension baked into the called definition.
                        hd = self._head_dim_expr()
                        theta = self._trace_dim_expr(node.inputs[1], module, "i", set())
                    if hd is not None and theta is not None:
                        rope_calls.append((
                            hd,
                            theta,
                            "1.0",
                            "1.0",
                        ))
                    continue
                if "rope_proportional" not in op_name and "rope" not in op_name:
                    continue
                if len(node.inputs) >= 5:
                    hd = self._trace_dim_expr(node.inputs[1], module, "i", set())
                    theta = self._trace_dim_expr(node.inputs[2], module, "i", set())
                    scale = self._trace_dim_expr(node.inputs[3], module, "i", set())
                    partial = self._trace_dim_expr(node.inputs[4], module, "i", set())
                    if hd is not None and theta is not None:
                        rope_calls.append((
                            hd,
                            theta,
                            scale if scale is not None else "1.0",
                            partial if partial is not None else "1.0",
                        ))
        if len(rope_calls) < 2:
            return None
        local_hd, local_theta, local_scale, local_partial = rope_calls[0]
        full_hd, full_theta, full_scale, full_partial = rope_calls[-1]
        if (
            local_hd,
            local_theta,
            local_scale,
            local_partial,
        ) == (
            full_hd,
            full_theta,
            full_scale,
            full_partial,
        ):
            return None
        rope_period = self._resolve_const_value("ROPE_PERIOD")
        if rope_period is None:
            rope_period = "6"
        return (
            str(local_hd),
            str(local_theta),
            str(local_scale),
            str(local_partial),
            str(full_hd),
            str(full_theta),
            str(full_scale),
            str(full_partial),
            rope_period,
        )

    def _detect_fused_residual(self, repeated_mod: Any) -> bool:
        """Detect if the model uses fused (Gemma3) or non-fused (Gemma4) residual pattern."""
        norm_output_names: set[str] = set()
        for node in repeated_mod.nodes:
            if self._vllm_classification.node_types.get(node.id) == VLLMLayerType.RMSNORM:
                for out in node.outputs:
                    if hasattr(out, "name"):
                        norm_output_names.add(out.name)
        for node in repeated_mod.nodes:
            if node.op.name == "core.binary.+":
                out_name = None
                for out in node.outputs:
                    if hasattr(out, "name"):
                        out_name = out.name
                        break
                if out_name is None:
                    continue
                for other in repeated_mod.nodes:
                    if self._vllm_classification.node_types.get(other.id) == VLLMLayerType.RMSNORM:
                        for inp in other.inputs:
                            if hasattr(inp, "name") and inp.name == out_name:
                                return True
        return False

    def _detect_activation_node(self, repeated_mod: Any) -> GraphNode | None:
        def _expr_is_activation(expr: GraphExpr) -> bool:
            if expr.op.name.startswith("_activations_"):
                return True
            callee = self.modules_by_name.get(expr.op.name)
            if callee is None:
                return False
            return any(inner.op.name.startswith("_activations_") for inner in callee.nodes)

        for node in repeated_mod.nodes:
            op_name = node.op.name
            if op_name.startswith("Activations.") or op_name.startswith("_activations_"):
                return node
            if op_name == "core.select" and any(
                isinstance(inp, GraphExpr) and _expr_is_activation(inp)
                for inp in node.inputs[1:3]
            ):
                return node
            callee = self.modules_by_name.get(op_name)
            if callee is not None and any(
                inner.op.name.startswith("_activations_")
                or (
                    inner.op.name == "core.select"
                    and any(
                        isinstance(inp, GraphExpr) and _expr_is_activation(inp)
                        for inp in inner.inputs[1:3]
                    )
                )
                for inner in callee.nodes
            ):
                return node
        return None

    def _detect_activation(self, repeated_mod: Any) -> str | None:
        node = self._detect_activation_node(repeated_mod)
        return node.op.name if node is not None else None

    def _activation_expr_to_code(
        self,
        operand: GraphOperand,
        *,
        x: str,
        repeated_mod: Any,
        loop_var: str = "i",
        depth: int = 0,
    ) -> str | None:
        if depth > 4 or not isinstance(operand, GraphExpr):
            return None
        op_name = operand.op.name
        if op_name.startswith("_activations_"):
            return self._activation_to_code(op_name).format(x=x)
        callee = self.modules_by_name.get(op_name)
        if callee is not None:
            for inner in callee.nodes:
                if inner.op.name.startswith("_activations_"):
                    return self._activation_to_code(inner.op.name).format(x=x)
        if op_name == "core.select" and len(operand.inputs) >= 3:
            cond = self._trace_dim_expr(operand.inputs[0], repeated_mod, loop_var, set())
            true_code = self._activation_expr_to_code(
                operand.inputs[1],
                x=x,
                repeated_mod=repeated_mod,
                loop_var=loop_var,
                depth=depth + 1,
            )
            false_code = self._activation_expr_to_code(
                operand.inputs[2],
                x=x,
                repeated_mod=repeated_mod,
                loop_var=loop_var,
                depth=depth + 1,
            )
            if cond is not None and true_code is not None and false_code is not None:
                return f"({true_code} if {cond} else {false_code})"
        return None

    def _substitute_graph_operand(
        self,
        operand: GraphOperand,
        mapping: dict[str | None, GraphOperand],
    ) -> GraphOperand:
        if isinstance(operand, GraphValueRef) and operand.name in mapping:
            return mapping[operand.name]
        if isinstance(operand, GraphExpr):
            return GraphExpr(
                op=operand.op,
                inputs=tuple(
                    self._substitute_graph_operand(item, mapping)
                    for item in operand.inputs
                ),
                attrs={
                    key: self._substitute_graph_operand(value, mapping)
                    for key, value in operand.attrs.items()
                },
                type_expr=operand.type_expr,
                dims=operand.dims,
            )
        return operand

    def _activation_node_to_code(
        self,
        node: GraphNode | None,
        *,
        x: str,
        repeated_mod: Any,
        loop_var: str = "i",
    ) -> str | None:
        if node is None:
            return None
        if node.op.name.startswith("_activations_"):
            return self._activation_to_code(node.op.name).format(x=x)
        if node.op.name == "core.select" and len(node.inputs) >= 3:
            expr = GraphExpr(op=node.op, inputs=node.inputs, attrs=node.attrs)
            return self._activation_expr_to_code(
                expr,
                x=x,
                repeated_mod=repeated_mod,
                loop_var=loop_var,
            )
        callee = self.modules_by_name.get(node.op.name)
        if callee is not None:
            for inner in callee.nodes:
                if inner.op.name.startswith("_activations_"):
                    return self._activation_to_code(inner.op.name).format(x=x)
                if inner.op.name == "core.select" and len(inner.inputs) >= 3:
                    # Instantiate the callee-local select condition with the
                    # call-site actuals before rendering a Python ternary.
                    formal_to_actual = {
                        getattr(formal, "name", None): actual
                        for formal, actual in zip(callee.inputs, node.inputs, strict=False)
                    }
                    expr_inputs = tuple(
                        self._substitute_graph_operand(inp, formal_to_actual)
                        for inp in inner.inputs
                    )
                    expr = GraphExpr(
                        op=inner.op,
                        inputs=expr_inputs,
                        attrs=inner.attrs,
                        type_expr=inner.outputs[0].type_expr if inner.outputs else inner.type_expr,
                        dims=inner.outputs[0].dims if inner.outputs else inner.dims,
                    )
                    rendered = self._activation_expr_to_code(
                        expr,
                        x=x,
                        repeated_mod=repeated_mod,
                        loop_var=loop_var,
                    )
                    if rendered is not None:
                        return rendered
        return None

    def _activation_arg_expr(
        self,
        repeated_mod: Any,
        *,
        input_index: int,
        loop_var: str = "i",
        default: str = "None",
    ) -> str:
        node = self._detect_activation_node(repeated_mod)
        if node is None or len(node.inputs) <= input_index:
            return default
        expr = self._trace_dim_expr(node.inputs[input_index], repeated_mod, loop_var, set())
        return expr if expr is not None else default

    @staticmethod
    def _activation_to_code(act_name: str) -> str:
        if act_name.endswith("gegelu"):
            return "self._gegelu({x})"
        if act_name.endswith(("gelu_pytorch_tanh", "gelu_tanh", "gelu_new")):
            return "F.gelu({x}, approximate='tanh')"
        if act_name.endswith("gelu"):
            return "F.gelu({x})"
        if act_name.endswith("silu") or act_name.endswith("swish"):
            return "F.silu({x})"
        if act_name.endswith("relu"):
            return "F.relu({x})"
        return "F.gelu({x})"

    def _analyze_layer_norms(
        self,
        repeated_mod: Any,
        classification: VLLMLayerClassification,
    ) -> list[tuple[str, str, bool, bool]]:
        """Return ordered list of (node_id, attr_expr, uses_residual, fused) for non-QK norms."""
        norms: list[tuple[int, str]] = []
        for node in repeated_mod.nodes:
            node_type = classification.node_types.get(node.id)
            if node_type in (VLLMLayerType.RMSNORM, VLLMLayerType.LAYERNORM):
                if node.id in classification.qk_norm_node_ids:
                    continue
                if node.id in classification.v_norm_node_ids:
                    continue
                if node.id == classification.pli_norm_node_id:
                    continue
                idx = int(node.id.rsplit(":", 1)[-1]) if ":" in node.id else 0
                norms.append((idx, node.id))

        norms = self._sort_repeated_norms_by_dataflow(repeated_mod, norms)
        if not norms:
            return []

        value_to_node: dict[str, str] = {}
        for node in repeated_mod.nodes:
            for out in node.outputs:
                if hasattr(out, "name"):
                    value_to_node[out.name] = node.id

        add_nodes: list[str] = []
        for node in repeated_mod.nodes:
            if node.op.name == "core.binary.+":
                add_nodes.append(node.id)

        add_to_consumers: dict[str, list[str]] = {}
        for node in repeated_mod.nodes:
            for inp in node.inputs:
                if hasattr(inp, "name") and inp.name in value_to_node:
                    producer_id = value_to_node[inp.name]
                    if producer_id in add_nodes:
                        add_to_consumers.setdefault(producer_id, []).append(node.id)

        norm_ids = [nid for _, nid in norms]
        uses_residual_set: set[str] = set()
        fused_set: set[str] = set()
        uses_residual_set.add(norm_ids[0])

        has_explicit_adds = False
        block_output_names: set[str] = set()
        for o in repeated_mod.outputs:
            if hasattr(o, "name"):
                block_output_names.add(o.name)

        for add_id in add_nodes:
            add_node = self._find_node_by_id(add_id)
            add_output_name = None
            if add_node.outputs:
                add_output_name = getattr(add_node.outputs[0], "name", None)
            consumers = add_to_consumers.get(add_id, [])
            is_block_output = add_output_name in block_output_names
            if not consumers and not is_block_output:
                continue
            has_explicit_adds = True
            # Case 1: a norm consumes the add's output → the add was already done
            # before the norm.  The norm just normalizes the sum; it should NOT
            # add residual after itself.
            # Case 2: a norm produces the add's input → the add happens AFTER the
            # norm, so the norm should emit an explicit residual add after.
            for inp in add_node.inputs:
                if hasattr(inp, "name") and inp.name in value_to_node:
                    producer = value_to_node[inp.name]
                    if producer in norm_ids:
                        uses_residual_set.add(producer)

        # If there are no explicit adds in the block (Gemma2 pattern), the
        # residual adds are implicit and must be folded into alternating norms.
        # Even-indexed norms (input_norm, pre_ffn_norm) are fused; odd-indexed
        # norms (post_attn_norm, post_ffn_norm) are non-fused.
        if not has_explicit_adds:
            for i, nid in enumerate(norm_ids):
                if i > 0 and i % 2 == 0:
                    fused_set.add(nid)
                    uses_residual_set.add(nid)

        result: list[tuple[str, str, bool, bool]] = []
        for _, nid in norms:
            node = self._find_node_by_id(nid)
            attr = self._vllm_attr_access(node)
            result.append((nid, attr, nid in uses_residual_set, nid in fused_set))
        return result

    def _sort_repeated_norms_by_dataflow(
        self,
        repeated_mod: GraphModule,
        norms: list[tuple[int, str]],
    ) -> list[tuple[int, str]]:
        """Order repeated-layer norms by graph dependency, not generated ids.

        Inlined helper modules can produce node ids whose numeric suffixes do
        not match execution order.  The clean vLLM scheduler is only valid when
        the norm sequence follows dataflow, so prefer dependency edges and use
        the source/id order only as a deterministic tie breaker.
        """
        if len(norms) <= 1:
            return norms

        by_id = {nid: (idx, nid) for idx, nid in norms}
        order_index = {nid: pos for pos, (_, nid) in enumerate(norms)}
        edges: dict[str, set[str]] = {nid: set() for _, nid in norms}
        indegree: dict[str, int] = {nid: 0 for _, nid in norms}

        for _, before_id in norms:
            for _, after_id in norms:
                if before_id == after_id:
                    continue
                after_node = self._find_node_by_id(after_id)
                if after_node is None:
                    continue
                if any(
                    self._operand_depends_on_any_node(
                        repeated_mod,
                        inp,
                        {before_id},
                    )
                    for inp in after_node.inputs
                ):
                    if after_id not in edges[before_id]:
                        edges[before_id].add(after_id)
                        indegree[after_id] += 1

        ready = sorted(
            (nid for nid, deg in indegree.items() if deg == 0),
            key=lambda nid: (by_id[nid][0], order_index[nid], nid),
        )
        sorted_ids: list[str] = []
        while ready:
            nid = ready.pop(0)
            sorted_ids.append(nid)
            for succ in sorted(edges[nid], key=lambda item: (by_id[item][0], order_index[item], item)):
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)
                    ready.sort(key=lambda item: (by_id[item][0], order_index[item], item))

        if len(sorted_ids) != len(norms):
            return sorted(norms, key=lambda item: (item[0], order_index[item[1]], item[1]))
        return [by_id[nid] for nid in sorted_ids]

    def _ffn_norm_index_from_dataflow(
        self,
        repeated_mod: GraphModule,
        layer_norms: list[tuple[str, str, bool, bool]],
        ffn_group: Any,
    ) -> int | None:
        if ffn_group is None:
            return None
        ffn_input_node_id = (
            ffn_group.gate_node_id
            or ffn_group.up_node_id
            or ffn_group.gate_up_intrinsic_node_id
        )
        if ffn_input_node_id is None:
            return None
        ffn_input_node = self._find_node_by_id(ffn_input_node_id)
        if ffn_input_node is None or len(ffn_input_node.inputs) < 2:
            return None
        ffn_input = ffn_input_node.inputs[1]
        result: int | None = None
        for norm_i, (norm_id, _, _, _) in enumerate(layer_norms):
            if self._operand_depends_on_any_node(repeated_mod, ffn_input, {norm_id}):
                result = norm_i
        return result

    def _infer_lm_head_path(
        self,
        classification: VLLMLayerClassification,
    ) -> str | None:
        if classification.lm_head_node_id:
            lm_head_node = self._find_node_by_id(classification.lm_head_node_id)
            if lm_head_node is not None and not self._is_repeated_node(lm_head_node):
                return _linear_base_key(lm_head_node)
        if self._use_clean_forward:
            main_mod = self.modules_by_name.get(self.program.main_module)
            if main_mod is not None:
                for node in reversed(main_mod.nodes):
                    node_layer_type = self._vllm_classification.node_types.get(
                        node.id, VLLMLayerType.DEFAULT
                    )
                    if (
                        node_layer_type
                        in (
                            VLLMLayerType.COLUMN_PARALLEL_LINEAR,
                            VLLMLayerType.ROW_PARALLEL_LINEAR,
                            VLLMLayerType.PARALLEL_LM_HEAD,
                        )
                        and len(node.inputs) >= 3
                        and _literal_value(node.inputs[2], None) is None
                    ):
                        return self._linear_prefix_from_node(node)
                inferred_from_return = self._returned_projection_path(
                    self.program.main_module,
                    output_index=0,
                )
                if inferred_from_return:
                    return inferred_from_return
        inferred = self._unique_emitted_vocab_projection_path()
        if inferred:
            return inferred
        return None

    def _linear_weight_key_from_inputs(
        self,
        inputs: tuple[GraphOperand, ...],
        bindings: dict[str, GraphOperand] | None = None,
    ) -> str | None:
        if not inputs:
            return None
        base = self._bound_preloop_operand(inputs[0], bindings)
        leaf: GraphOperand | str = "weight"
        if len(inputs) >= 4:
            leaf = self._bound_preloop_operand(inputs[3], bindings)
        return self._static_param_key(base, leaf)

    def _linear_prefix_from_inputs(
        self,
        inputs: tuple[GraphOperand, ...],
        bindings: dict[str, GraphOperand] | None = None,
    ) -> str | None:
        key = self._linear_weight_key_from_inputs(inputs, bindings)
        if key:
            return key.rsplit(".", 1)[0] if key.endswith(".weight") else key
        if inputs:
            base = self._bound_preloop_operand(inputs[0], bindings)
            return self._static_path_key(base)
        return None

    def _linear_prefix_from_node(
        self,
        node: GraphNode,
        bindings: dict[str, GraphOperand] | None = None,
    ) -> str | None:
        prefix = self._linear_prefix_from_inputs(node.inputs, bindings)
        return prefix or _linear_base_key(node)

    def _returned_projection_path(
        self,
        module_name: str,
        *,
        actuals: tuple[GraphOperand, ...] = (),
        output_index: int = 0,
        bindings: dict[str, GraphOperand] | None = None,
        depth: int = 0,
    ) -> str | None:
        module = self.modules_by_name.get(module_name)
        if module is None or depth > 12 or output_index < 0 or output_index >= len(module.outputs):
            return None
        bound_actuals = tuple(self._bound_preloop_operand(item, bindings) for item in actuals)
        callee_bindings = {
            name: bound_actuals[index]
            for index, formal in enumerate(module.inputs)
            if index < len(bound_actuals) and (name := _value_name(formal)) is not None
        }
        return self._returned_projection_path_from_operand(
            module,
            module.outputs[output_index],
            output_index=0,
            bindings=callee_bindings,
            depth=depth,
            visited=set(),
        )

    def _returned_projection_path_from_operand(
        self,
        module: GraphModule,
        operand: GraphOperand,
        *,
        output_index: int,
        bindings: dict[str, GraphOperand] | None,
        depth: int,
        visited: set[tuple[str, str, int]],
    ) -> str | None:
        operand = self._bound_preloop_operand(operand, bindings)
        if isinstance(operand, GraphExpr):
            return self._returned_projection_path_from_expr(
                module,
                operand,
                output_index=output_index,
                bindings=bindings,
                depth=depth,
                visited=visited,
            )
        name = _value_name(operand)
        if name is None:
            return None
        key = (module.name, name, output_index)
        if key in visited:
            return None
        visited.add(key)
        for node in reversed(module.nodes):
            for idx, output in enumerate(node.outputs):
                if _value_name(output) == name:
                    return self._returned_projection_path_from_node(
                        module,
                        node,
                        output_index=idx,
                        bindings=bindings,
                        depth=depth,
                        visited=visited,
                    )
        return None

    def _returned_projection_path_from_expr(
        self,
        module: GraphModule,
        expr: GraphExpr,
        *,
        output_index: int,
        bindings: dict[str, GraphOperand] | None,
        depth: int,
        visited: set[tuple[str, str, int]],
    ) -> str | None:
        if expr.op.name in {"NN.linear", "_linear"}:
            return self._linear_prefix_from_inputs(expr.inputs, bindings)
        if expr.op.name in self.modules_by_name:
            return self._returned_projection_path(
                expr.op.name,
                actuals=expr.inputs,
                output_index=output_index,
                bindings=bindings,
                depth=depth + 1,
            )
        if expr.op.name == "core.select" and len(expr.inputs) >= 3:
            paths = [
                path
                for branch in expr.inputs[1:]
                if (
                    path := self._returned_projection_path_from_operand(
                        module,
                        branch,
                        output_index=output_index,
                        bindings=bindings,
                        depth=depth + 1,
                        visited=visited,
                    )
                )
            ]
            unique = list(dict.fromkeys(paths))
            if len(unique) == 1:
                return unique[0]
            if self._checkpoint_prefixes:
                weighted = {
                    path
                    for path in unique
                    if path in self._checkpoint_prefixes
                    or any(item.startswith(path + ".") for item in self._checkpoint_prefixes)
                }
                if len(weighted) == 1:
                    return next(iter(weighted))
            return None
        return None

    def _returned_projection_path_from_node(
        self,
        module: GraphModule,
        node: GraphNode,
        *,
        output_index: int,
        bindings: dict[str, GraphOperand] | None,
        depth: int,
        visited: set[tuple[str, str, int]],
    ) -> str | None:
        if node.op.name in {"NN.linear", "_linear"} and output_index == 0:
            return self._linear_prefix_from_node(node, bindings)
        if node.op.name in self.modules_by_name:
            return self._returned_projection_path(
                node.op.name,
                actuals=node.inputs,
                output_index=output_index,
                bindings=bindings,
                depth=depth + 1,
            )
        if node.op.name == "core.alias":
            if len(node.inputs) == len(node.outputs):
                return self._returned_projection_path_from_operand(
                    module,
                    node.inputs[output_index],
                    output_index=0,
                    bindings=bindings,
                    depth=depth + 1,
                    visited=visited,
                )
            if len(node.inputs) == 1:
                return self._returned_projection_path_from_operand(
                    module,
                    node.inputs[0],
                    output_index=output_index,
                    bindings=bindings,
                    depth=depth + 1,
                    visited=visited,
                )
        if node.op.name == "core.select":
            paths = [
                path
                for branch in node.inputs[1:]
                if (
                    path := self._returned_projection_path_from_operand(
                        module,
                        branch,
                        output_index=output_index,
                        bindings=bindings,
                        depth=depth + 1,
                        visited=visited,
                    )
                )
            ]
            unique = list(dict.fromkeys(paths))
            if len(unique) == 1:
                return unique[0]
            if self._checkpoint_prefixes:
                weighted = {
                    path
                    for path in unique
                    if path in self._checkpoint_prefixes
                    or any(item.startswith(path + ".") for item in self._checkpoint_prefixes)
                }
                if len(weighted) == 1:
                    return next(iter(weighted))
        return None

    def _node_output_dim_int(self, node: GraphNode) -> int | None:
        dim = _int_arg(node, 2)
        if dim is not None:
            return dim
        if not node.outputs:
            return None
        out = node.outputs[0]
        out_dims = getattr(out, "dims", None) or getattr(getattr(out, "type_expr", None), "dims", None)
        if not out_dims:
            return None
        last = out_dims[-1]
        if isinstance(last, int):
            return last
        if isinstance(last, str):
            resolved = self._resolve_const_value(last)
            if resolved is not None:
                try:
                    return int(resolved)
                except ValueError:
                    return None
        return None

    def _unique_emitted_vocab_projection_path(self) -> str | None:
        vocab = self._model_config_data.get("vocab_size")
        if not isinstance(vocab, int) or isinstance(vocab, bool) or vocab <= 0:
            return None
        matches: list[str] = []
        emitted = getattr(self, "_vllm_emitted_layer_node_ids", set())
        if not emitted:
            return None
        for module in self.program.modules:
            for node in module.nodes:
                if node.id not in emitted or self._is_repeated_node(node):
                    continue
                node_layer_type = self._vllm_classification.node_types.get(
                    node.id, VLLMLayerType.DEFAULT
                )
                if node_layer_type not in (
                    VLLMLayerType.COLUMN_PARALLEL_LINEAR,
                    VLLMLayerType.ROW_PARALLEL_LINEAR,
                    VLLMLayerType.PARALLEL_LM_HEAD,
                ):
                    continue
                if self._node_output_dim_int(node) != vocab:
                    continue
                path = _linear_base_key(node) or self._node_prefix(node)
                if path:
                    matches.append(path)
        unique = list(dict.fromkeys(matches))
        return unique[0] if len(unique) == 1 else None

    def _emitted_linear_attr_for_path(self, path: str | None) -> str | None:
        if not path:
            return None
        matches: list[str] = []
        for module in self.program.modules:
            for node in module.nodes:
                if node.id not in getattr(self, "_vllm_emitted_layer_node_ids", set()):
                    continue
                if self._is_repeated_node(node):
                    continue
                node_layer_type = self._vllm_classification.node_types.get(
                    node.id, VLLMLayerType.DEFAULT
                )
                if node_layer_type not in (
                    VLLMLayerType.COLUMN_PARALLEL_LINEAR,
                    VLLMLayerType.ROW_PARALLEL_LINEAR,
                    VLLMLayerType.PARALLEL_LM_HEAD,
                ):
                    continue
                if _linear_base_key(node) == path or self._node_prefix(node) == path:
                    matches.append(self._vllm_attr_access(node))
        unique = list(dict.fromkeys(matches))
        return unique[0] if len(unique) == 1 else None

    def _emitted_embedding_attr_for_prefix(self, prefix: str | None) -> str | None:
        if not prefix:
            return None
        matches: list[str] = []
        for module in self.program.modules:
            for node in module.nodes:
                if node.id not in getattr(self, "_vllm_emitted_layer_node_ids", set()):
                    continue
                node_layer_type = self._vllm_classification.node_types.get(
                    node.id, VLLMLayerType.DEFAULT
                )
                if node_layer_type != VLLMLayerType.VOCAB_PARALLEL_EMBEDDING:
                    continue
                if self._vllm_embedding_layer_prefix(node) == prefix:
                    matches.append(self._vllm_attr_access(node))
        unique = list(dict.fromkeys(matches))
        return unique[0] if len(unique) == 1 else None

    def _token_embedding_attr_and_prefix(self, node: GraphNode) -> tuple[str, str] | None:
        prefix = self._vllm_embedding_layer_prefix(node)
        if not prefix:
            return None
        if node.id in getattr(self, "_vllm_emitted_layer_node_ids", set()):
            return self._vllm_attr_access(node), prefix
        attr = self._emitted_embedding_attr_for_prefix(prefix)
        if attr is None:
            return None
        return attr, prefix

    def _operand_contains_value(self, operand: GraphOperand, value_name: str) -> bool:
        if isinstance(operand, GraphValueRef):
            return operand.name == value_name
        if isinstance(operand, GraphExpr):
            return any(self._operand_contains_value(inp, value_name) for inp in operand.inputs)
        return False

    def _selects_block_input_vs_current_hidden(
        self,
        operand: GraphOperand,
        *,
        repeated_mod: Any,
    ) -> str | None:
        """Return condition code for select(block input, current hidden), if proven."""
        if not isinstance(operand, GraphExpr) or operand.op.name != "core.select":
            return None
        if len(operand.inputs) < 3 or not repeated_mod.inputs:
            return None
        block_input_name = _value_name(repeated_mod.inputs[1]) if len(repeated_mod.inputs) > 1 else None
        if block_input_name is None:
            return None
        true_name = _value_name(operand.inputs[1])
        false_name = _value_name(operand.inputs[2])
        if true_name == block_input_name and false_name != block_input_name:
            return self._trace_dim_expr(operand.inputs[0], repeated_mod, "i", set())
        if false_name == block_input_name and true_name != block_input_name:
            cond = self._trace_dim_expr(operand.inputs[0], repeated_mod, "i", set())
            return f"not ({cond})" if cond is not None else None
        return None

    def _module_output_select_contains_value(
        self,
        repeated_mod: Any,
        value_name: str,
    ) -> bool:
        """True when the first module output is a select and all branches use value_name."""
        if not repeated_mod.outputs:
            return False
        output_name = _value_name(repeated_mod.outputs[0])
        if output_name is None:
            return False
        producer = None
        for node in repeated_mod.nodes:
            if any(_value_name(out) == output_name for out in node.outputs):
                producer = node
                break
        if producer is None or producer.op.name != "core.select" or len(producer.inputs) < 3:
            return False
        return all(
            self._operand_contains_value(branch, value_name)
            for branch in producer.inputs[1:3]
        )

    def _token_embedding_scale_expr(self, embedding_node: GraphNode) -> str | None:
        if embedding_node.op.name in {"NN.embedding", "_embedding"} and len(embedding_node.inputs) >= 4:
            scale_operand = embedding_node.inputs[3]
            if not (
                isinstance(scale_operand, GraphLiteral)
                and scale_operand.value is None
            ):
                module = self._module_for_node(embedding_node) or self.modules_by_name.get(
                    self.program.main_module
                )
                if module is not None:
                    traced = self._trace_dim_expr(scale_operand, module, "i", set())
                    if traced is not None and traced != "None":
                        return traced
        output_name = _value_name(embedding_node.outputs[0]) if embedding_node.outputs else None
        if output_name is None:
            return None
        main_module = self.modules_by_name.get(self.program.main_module)
        if main_module is None:
            return None
        for node in main_module.nodes:
            if node.op.name != "core.binary.*" or len(node.inputs) < 2:
                continue
            left_name = _value_name(node.inputs[0])
            right_name = _value_name(node.inputs[1])
            if left_name == output_name:
                return self._trace_dim_expr(node.inputs[1], main_module, "i", set())
            if right_name == output_name:
                return self._trace_dim_expr(node.inputs[0], main_module, "i", set())
        return None

    def _main_value_node_map(self, module: Any) -> dict[str, GraphNode]:
        mapping: dict[str, GraphNode] = {}
        for node in module.nodes:
            for out in node.outputs:
                name = _value_name(out)
                if name is not None:
                    mapping[name] = node
        return mapping

    def _module_for_node(self, node: GraphNode) -> Any | None:
        module_name = self._node_module_name(node)
        if module_name:
            module = self.modules_by_name.get(module_name)
            if module is not None:
                return module
        for module in self.program.modules:
            if any(candidate.id == node.id for candidate in module.nodes):
                return module
        return None

    def _main_callsite_operand_for_module_input(
        self,
        node: GraphNode,
        operand: GraphOperand,
    ) -> GraphOperand | None:
        module = self._module_for_node(node)
        main_module = self.modules_by_name.get(self.program.main_module)
        if module is None or main_module is None or module.name == main_module.name:
            return None
        input_name = _value_name(operand)
        if input_name is None:
            return None
        input_index = None
        for index, module_input in enumerate(module.inputs):
            if _value_name(module_input) == input_name:
                input_index = index
                break
        if input_index is None:
            return None
        candidates: list[GraphOperand] = []
        for call_node in main_module.nodes:
            if call_node.op.name != module.name or len(call_node.inputs) <= input_index:
                continue
            candidates.append(call_node.inputs[input_index])
        if not candidates:
            return None
        first = candidates[0]
        first_expr = self._position_indices_expr(first, main_module)
        for candidate in candidates[1:]:
            if self._position_indices_expr(candidate, main_module) != first_expr:
                return None
        return first

    def _position_indices_expr_for_node(self, node: GraphNode) -> str | None:
        if len(node.inputs) < 2:
            return None
        module = self._module_for_node(node)
        if module is None:
            return None
        operand = self._main_callsite_operand_for_module_input(node, node.inputs[1])
        if operand is not None:
            main_module = self.modules_by_name.get(self.program.main_module)
            if main_module is not None:
                return self._position_indices_expr(operand, main_module)
        return self._position_indices_expr(node.inputs[1], module)

    def _token_embedding_addend_expr(self, embedding_node: GraphNode) -> str | None:
        output_name = _value_name(embedding_node.outputs[0]) if embedding_node.outputs else None
        if output_name is None:
            return None
        main_module = self.modules_by_name.get(self.program.main_module)
        if main_module is None:
            return None
        token_path_names = {output_name}
        for node in main_module.nodes:
            if node.op.name != "core.binary.*" or len(node.inputs) < 2:
                continue
            out_name = _value_name(node.outputs[0]) if node.outputs else None
            if out_name is None:
                continue
            if _value_name(node.inputs[0]) == output_name or _value_name(node.inputs[1]) == output_name:
                token_path_names.add(out_name)
        for node in main_module.nodes:
            if node.op.name != "core.binary.+" or len(node.inputs) < 2:
                continue
            left_name = _value_name(node.inputs[0])
            right_name = _value_name(node.inputs[1])
            if left_name in token_path_names:
                return self._preloop_operand_expr(node.inputs[1], main_module)
            if right_name in token_path_names:
                return self._preloop_operand_expr(node.inputs[0], main_module)
        return None

    def _bound_preloop_operand(
        self,
        operand: GraphOperand,
        bindings: dict[str, GraphOperand] | None,
    ) -> GraphOperand:
        if bindings is None:
            return operand
        name = _value_name(operand)
        if name is not None:
            return bindings.get(name, operand)
        return operand

    def _embedding_weight_keys_from_module_call(
        self,
        module_name: str,
        actuals: tuple[GraphOperand, ...],
        bindings: dict[str, GraphOperand] | None,
        depth: int,
    ) -> tuple[str, ...]:
        callee = self.modules_by_name.get(module_name)
        if callee is None or depth > 8:
            return ()
        bound_actuals = tuple(self._bound_preloop_operand(item, bindings) for item in actuals)
        callee_bindings = {
            name: bound_actuals[index]
            for index, formal in enumerate(callee.inputs)
            if index < len(bound_actuals) and (name := _value_name(formal)) is not None
        }
        keys: list[str] = []
        for node in callee.nodes:
            keys.extend(
                self._embedding_weight_keys_from_node(
                    node,
                    callee_bindings,
                    depth=depth + 1,
                )
            )
        return tuple(dict.fromkeys(key for key in keys if key))

    def _embedding_weight_keys_from_operand(
        self,
        operand: GraphOperand,
        bindings: dict[str, GraphOperand] | None = None,
        *,
        depth: int = 0,
    ) -> tuple[str, ...]:
        operand = self._bound_preloop_operand(operand, bindings)
        if not isinstance(operand, GraphExpr):
            return ()
        if operand.op.name in {"NN.embedding", "_embedding"} and operand.inputs:
            base = self._bound_preloop_operand(operand.inputs[0], bindings)
            key = self._static_param_key(base, "weight")
            return (key,) if key is not None else ()
        if operand.op.name in self.modules_by_name:
            return self._embedding_weight_keys_from_module_call(
                operand.op.name,
                operand.inputs,
                bindings,
                depth,
            )
        keys: list[str] = []
        for item in operand.inputs:
            keys.extend(
                self._embedding_weight_keys_from_operand(
                    item,
                    bindings,
                    depth=depth + 1,
                )
            )
        for item in operand.attrs.values():
            if isinstance(item, (GraphExpr, GraphValueRef, GraphValue, GraphLiteral, GraphPath)):
                keys.extend(
                    self._embedding_weight_keys_from_operand(
                        item,
                        bindings,
                        depth=depth + 1,
                    )
                )
        return tuple(dict.fromkeys(key for key in keys if key))

    def _embedding_weight_keys_from_node(
        self,
        node: GraphNode,
        bindings: dict[str, GraphOperand] | None = None,
        *,
        depth: int = 0,
    ) -> tuple[str, ...]:
        if node.op.name in {"NN.embedding", "_embedding"} and node.inputs:
            base = self._bound_preloop_operand(node.inputs[0], bindings)
            key = self._static_param_key(base, "weight")
            return (key,) if key is not None else ()
        if node.op.name in self.modules_by_name:
            return self._embedding_weight_keys_from_module_call(
                node.op.name,
                node.inputs,
                bindings,
                depth,
            )
        keys: list[str] = []
        for item in node.inputs:
            keys.extend(
                self._embedding_weight_keys_from_operand(
                    item,
                    bindings,
                    depth=depth + 1,
                )
            )
        for item in node.attrs.values():
            if isinstance(item, (GraphExpr, GraphValueRef, GraphValue, GraphLiteral, GraphPath)):
                keys.extend(
                    self._embedding_weight_keys_from_operand(
                        item,
                        bindings,
                        depth=depth + 1,
                    )
                )
        return tuple(dict.fromkeys(key for key in keys if key))

    def _vllm_embedding_layer_prefix(self, node: GraphNode) -> str | None:
        keys = self._embedding_weight_keys_from_node(node)
        if len(keys) != 1:
            return None
        key = keys[0]
        return key.rsplit(".", 1)[0] if key.endswith(".weight") else key

    def _embedding_weight_suffix_from_node(self, node: GraphNode) -> str | None:
        """Return a graph-derived state-key suffix for templated embedding paths.

        Some generic Axon definitions take a root path argument and call
        `_embedding @@'{root}.foo' ...`.  At the definition site there is no
        single absolute key, but the non-template suffix is still semantic path
        metadata.  The generated vLLM model can resolve it at load time if the
        loaded state dict has exactly one matching key.
        """

        def suffix_from_path(path: GraphPath) -> str | None:
            parts = tuple(part for part in path.parts if part)
            if not parts:
                return None
            last_template = -1
            for index, part in enumerate(parts):
                if "{" in part or "}" in part:
                    last_template = index
            suffix_parts = parts[last_template + 1 :]
            if not suffix_parts:
                return None
            return ".".join((*suffix_parts, "weight"))

        def suffix_from_operand(operand: GraphOperand, depth: int = 0) -> str | None:
            if depth > 8:
                return None
            if isinstance(operand, GraphExpr):
                if operand.op.name in {"NN.embedding", "_embedding"} and operand.inputs:
                    base = operand.inputs[0]
                    if isinstance(base, GraphPath):
                        return suffix_from_path(base)
                if operand.op.name in self.modules_by_name:
                    callee = self.modules_by_name.get(operand.op.name)
                    if callee is not None:
                        bindings = {
                            name: operand.inputs[index]
                            for index, formal in enumerate(callee.inputs)
                            if index < len(operand.inputs) and (name := _value_name(formal)) is not None
                        }
                        for callee_node in callee.nodes:
                            suffix = suffix_from_node(callee_node, bindings, depth + 1)
                            if suffix is not None:
                                return suffix
                for item in operand.inputs:
                    suffix = suffix_from_operand(item, depth + 1)
                    if suffix is not None:
                        return suffix
                for item in operand.attrs.values():
                    if isinstance(item, (GraphExpr, GraphValueRef, GraphValue, GraphLiteral, GraphPath)):
                        suffix = suffix_from_operand(item, depth + 1)
                        if suffix is not None:
                            return suffix
            if isinstance(operand, GraphPath):
                return suffix_from_path(operand)
            return None

        def suffix_from_node(
            candidate: GraphNode,
            bindings: dict[str, GraphOperand] | None,
            depth: int,
        ) -> str | None:
            if candidate.op.name in {"NN.embedding", "_embedding"} and candidate.inputs:
                base = self._bound_preloop_operand(candidate.inputs[0], bindings)
                if isinstance(base, GraphPath):
                    return suffix_from_path(base)
            if candidate.op.name in self.modules_by_name:
                callee = self.modules_by_name.get(candidate.op.name)
                if callee is None or depth > 8:
                    return None
                bound_actuals = tuple(self._bound_preloop_operand(item, bindings) for item in candidate.inputs)
                callee_bindings = {
                    name: bound_actuals[index]
                    for index, formal in enumerate(callee.inputs)
                    if index < len(bound_actuals) and (name := _value_name(formal)) is not None
                }
                for callee_node in callee.nodes:
                    suffix = suffix_from_node(callee_node, callee_bindings, depth + 1)
                    if suffix is not None:
                        return suffix
                return None
            for item in candidate.inputs:
                suffix = suffix_from_operand(self._bound_preloop_operand(item, bindings), depth + 1)
                if suffix is not None:
                    return suffix
            for item in candidate.attrs.values():
                if isinstance(item, (GraphExpr, GraphValueRef, GraphValue, GraphLiteral, GraphPath)):
                    suffix = suffix_from_operand(self._bound_preloop_operand(item, bindings), depth + 1)
                    if suffix is not None:
                        return suffix
            return None

        return suffix_from_node(node, None, 0)

    def _preloop_operand_expr(
        self,
        operand: GraphOperand,
        main_module: Any,
        bindings: dict[str, GraphOperand] | None = None,
    ) -> str | None:
        if bindings is not None:
            name = _value_name(operand)
            if name is not None and name in bindings:
                return self._preloop_operand_expr(bindings[name], main_module, None)
        if isinstance(operand, GraphExpr):
            if operand.op.name == "core.select" and len(operand.inputs) >= 3:
                cond = self._trace_dim_expr(operand.inputs[0], main_module, "i", set())
                true_expr = self._preloop_operand_expr(operand.inputs[1], main_module, bindings)
                false_expr = self._preloop_operand_expr(operand.inputs[2], main_module, bindings)
                if cond and true_expr and false_expr:
                    return f"({true_expr} if {cond} else {false_expr})"
                return None
            embed_expr = self._preloop_embedding_expr(operand, main_module, bindings)
            if embed_expr is not None:
                return embed_expr
            sinusoidal_expr = self._preloop_sinusoidal_expr(operand, main_module)
            if sinusoidal_expr is not None:
                return sinusoidal_expr
            if operand.op.name in self.modules_by_name:
                return self._preloop_module_call_expr(
                    operand.op.name,
                    operand.inputs,
                    main_module,
                )
            return None
        name = _value_name(operand)
        if name is None:
            return None
        node = self._main_value_node_map(main_module).get(name)
        if node is None:
            return None
        if node.op.name == "core.select" and len(node.inputs) >= 3:
            out_idx = next(
                (idx for idx, out in enumerate(node.outputs) if _value_name(out) == name),
                None,
            )
            cond = self._trace_dim_expr(node.inputs[0], main_module, "i", set())
            branches: list[str | None] = []
            for branch in node.inputs[1:3]:
                if (
                    out_idx is not None
                    and isinstance(branch, GraphExpr)
                    and branch.op.name == "core.tuple"
                    and out_idx < len(branch.inputs)
                ):
                    branches.append(self._preloop_operand_expr(branch.inputs[out_idx], main_module, bindings))
                else:
                    branches.append(self._preloop_operand_expr(branch, main_module, bindings))
            if cond and branches[0] and branches[1]:
                return f"({branches[0]} if {cond} else {branches[1]})"
        if node.op.name in self.modules_by_name:
            return self._preloop_module_call_expr(node.op.name, node.inputs, main_module)
        return None

    def _preloop_module_call_expr(
        self,
        module_name: str,
        actuals: tuple[GraphOperand, ...],
        caller_module: Any,
    ) -> str | None:
        callee = self.modules_by_name.get(module_name)
        if callee is None or len(callee.outputs) != 1:
            return None
        output_name = _value_name(callee.outputs[0])
        if output_name is None:
            return None
        bindings = {
            name: actuals[index]
            for index, formal in enumerate(callee.inputs)
            if index < len(actuals) and (name := _value_name(formal)) is not None
        }
        return self._preloop_operand_expr(
            GraphValueRef(output_name, callee.outputs[0].type_expr, callee.outputs[0].dims),
            callee,
            bindings,
        )

    def _preloop_embedding_expr(
        self,
        expr: GraphExpr,
        main_module: Any,
        bindings: dict[str, GraphOperand] | None = None,
    ) -> str | None:
        if expr.op.name not in {"NN.embedding", "_embedding"} or len(expr.inputs) < 2:
            return None
        base = expr.inputs[0]
        if bindings is not None:
            base_name = _value_name(base)
            if base_name is not None and base_name in bindings:
                base = bindings[base_name]
        weight_key = self._static_param_key(base, "weight")
        if weight_key is None:
            return None
        indices_operand = expr.inputs[1]
        if bindings is not None:
            indices_name = _value_name(indices_operand)
            if indices_name is not None and indices_name in bindings:
                indices_operand = bindings[indices_name]
        indices = self._position_indices_expr(indices_operand, main_module)
        if indices is None:
            return None
        key = repr(weight_key)
        weight = f"self._vllm_state_tensor({key}, self.state_dict_tensors[{key}])"
        return f"(lambda _w: F.embedding(({indices}).to(_w.device), _w))({weight})"

    def _position_embedding_expr(self, node: GraphNode) -> str | None:
        if len(node.inputs) < 2:
            return None
        weight_key = self._static_param_key(node.inputs[0], "weight")
        if weight_key is None:
            return None
        indices = self._position_indices_expr_for_node(node)
        if indices is None:
            return None
        key = repr(weight_key)
        weight = f"self._vllm_state_tensor({key}, self.state_dict_tensors[{key}])"
        return f"(lambda _w: F.embedding(({indices}).to(_w.device), _w))({weight})"

    def _is_sinusoidal_position_module(self, name: str) -> bool:
        ops = self._reachable_module_ops(name, depth=0, visited=set())
        return {"_sin", "_cos", "_concat", "_arange"}.issubset(ops)

    def _reachable_module_ops(self, name: str, *, depth: int, visited: set[str]) -> set[str]:
        if depth > 6 or name in visited:
            return set()
        visited.add(name)
        mod = self.modules_by_name.get(name)
        if mod is None:
            return set()
        ops: set[str] = set()
        for node in mod.nodes:
            ops.add(node.op.name)
            if node.op.name in self.modules_by_name:
                ops.update(self._reachable_module_ops(node.op.name, depth=depth + 1, visited=set(visited)))
        return ops

    def _preloop_sinusoidal_expr(self, expr: GraphExpr, main_module: Any) -> str | None:
        if expr.op.name not in self.modules_by_name:
            return None
        if not self._is_sinusoidal_position_module(expr.op.name):
            return None
        if len(expr.inputs) < 2:
            return None
        indices = self._position_indices_expr(expr.inputs[1], main_module)
        if indices is None:
            return None
        theta = _literal_value(expr.inputs[2], 10000.0) if len(expr.inputs) >= 3 else 10000.0
        offset = _literal_value(expr.inputs[3], 2) if len(expr.inputs) >= 4 else 2
        padding_idx = _literal_value(expr.inputs[4], None) if len(expr.inputs) >= 5 else None
        mode = _literal_value(expr.inputs[5], None) if len(expr.inputs) >= 6 else None
        return (
            "self._vllm_sinusoidal_positions("
            f"hidden_states, {indices}, theta={float(theta)!r}, offset={int(offset)!r}, "
            f"padding_idx={padding_idx!r}, mode={mode!r})"
        )

    def _position_indices_expr(self, operand: GraphOperand, main_module: Any) -> str | None:
        if isinstance(operand, GraphExpr):
            if operand.op.name == "core.binary.+" and len(operand.inputs) >= 2:
                left_lit = _literal_value(operand.inputs[0], None)
                right_lit = _literal_value(operand.inputs[1], None)
                if isinstance(left_lit, int):
                    return f"(positions + {left_lit})"
                if isinstance(right_lit, int):
                    return f"(positions + {right_lit})"
            return "positions"
        name = _value_name(operand)
        if name is None:
            return None
        node = self._main_value_node_map(main_module).get(name)
        if node is None:
            return "positions"
        if node.op.name == "core.binary.+" and len(node.inputs) >= 2:
            left_lit = _literal_value(node.inputs[0], None)
            right_lit = _literal_value(node.inputs[1], None)
            if isinstance(left_lit, int):
                return f"(positions + {left_lit})"
            if isinstance(right_lit, int):
                return f"(positions + {right_lit})"
        return "positions"

    def _find_o_proj_id(
        self,
        repeated_mod: Any,
        classification: VLLMLayerClassification,
    ) -> str | None:
        ffn_down_ids = {g.down_node_id for g in classification.ffn_groups}
        attention_ids = set(classification.attention_node_ids)
        for node in repeated_mod.nodes:
            if not _is_linear_call(node, self.modules_by_name):
                continue
            if node.id in ffn_down_ids:
                continue
            if len(node.inputs) >= 2 and self._operand_depends_on_any_node(
                repeated_mod,
                node.inputs[1],
                attention_ids,
            ):
                return node.id
        for node in repeated_mod.nodes:
            if classification.node_types.get(node.id) == VLLMLayerType.ROW_PARALLEL_LINEAR:
                if node.id not in ffn_down_ids:
                    return node.id
        return None

    def _operand_depends_on_any_node(
        self,
        module: Any,
        operand: GraphOperand,
        target_node_ids: set[str],
        *,
        depth: int = 0,
        visited: set[str] | None = None,
    ) -> bool:
        if depth > 24:
            return False
        if visited is None:
            visited = set()
        if isinstance(operand, GraphExpr):
            return any(
                self._operand_depends_on_any_node(
                    module,
                    item,
                    target_node_ids,
                    depth=depth + 1,
                    visited=visited,
                )
                for item in operand.inputs
            )
        name = _value_name(operand)
        if name is None:
            return False
        key = f"{getattr(module, 'name', '<module>')}:{name}"
        if key in visited:
            return False
        visited.add(key)
        producer = self._main_value_node_map(module).get(name)
        if producer is None:
            return False
        if producer.id in target_node_ids:
            return True
        return any(
            self._operand_depends_on_any_node(
                module,
                item,
                target_node_ids,
                depth=depth + 1,
                visited=visited,
            )
            for item in producer.inputs
        )

    def _uses_parallel_ffn_input(
        self,
        repeated_mod: Any,
        classification: VLLMLayerClassification,
        layer_norms: list[tuple[str, str, bool, bool]],
        ffn_group: Any,
        o_proj_id: str | None,
        *,
        has_standalone_attn_add: bool,
        has_standalone_ffn_add: bool,
    ) -> bool:
        """Detect blocks where attention and FFN both consume the same norm.

        Some decoder blocks compute ``x + attention(norm(x)) + ffn(norm(x))``.
        The clean vLLM forward can use vLLM linears/attention for this topology,
        but it must not feed the post-attention residual into the FFN.  The
        predicate is structural: it only checks data dependencies in the graph.
        """
        if not has_standalone_attn_add or not has_standalone_ffn_add:
            return False
        if len(layer_norms) != 1 or ffn_group is None:
            return False
        ffn_input_node_id = ffn_group.gate_node_id or ffn_group.up_node_id
        if ffn_input_node_id is None:
            return False
        ffn_input_node = self._find_node_by_id(ffn_input_node_id)
        if ffn_input_node is None or len(ffn_input_node.inputs) < 2:
            return False
        first_norm_id = layer_norms[0][0]
        ffn_input = ffn_input_node.inputs[1]
        if not self._operand_depends_on_any_node(repeated_mod, ffn_input, {first_norm_id}):
            return False
        forbidden = set(classification.attention_node_ids)
        if o_proj_id is not None:
            forbidden.add(o_proj_id)
        if forbidden and self._operand_depends_on_any_node(repeated_mod, ffn_input, forbidden):
            return False
        return True

    def _emit_common(self, lines: list[str]) -> None:
        add = self._add
        classification = self._vllm_classification
        indent = 4

        add(lines, indent, '"""Generated vLLM model from Axon Graph IR."""')
        add(lines, indent, "")
        if self._needs_mamba_cache_placeholders(classification) or classification.mamba_mixer_module_names:
            add(lines, indent, "is_attention_free = True")
            add(lines, indent, "")
        has_merged_ffn = any(
            g.gate_node_id and g.up_node_id for g in classification.ffn_groups
        )
        if classification.qkv_groups:
            add(lines, indent, "packed_modules_mapping = {")
            add(lines, indent + 1, '"qkv_proj": ["q_proj", "k_proj", "v_proj"],')
            if has_merged_ffn:
                add(lines, indent + 1, '"gate_up_proj": ["gate_proj", "up_proj"],')
            add(lines, indent, "}")
            add(lines, indent, "")
        add(lines, indent, "def __init__(self, *, vllm_config, prefix: str = ''):")
        add(lines, indent * 2, "super().__init__()")
        add(lines, indent * 2, "config = vllm_config.model_config.hf_config")
        add(lines, indent * 2, "if hasattr(config, 'text_config'):")
        add(lines, indent * 3, "config = config.text_config")
        add(lines, indent * 2, "self.vllm_config = vllm_config")
        add(lines, indent * 2, "self._prefix = prefix")
        add(lines, indent * 2, "quant_config = getattr(vllm_config, 'quant_config', None)")
        add(lines, indent * 2, "params_dtype = vllm_config.model_config.dtype")
        add(lines, indent * 2, "cache_config = getattr(vllm_config, 'cache_config', None)")
        add(lines, indent * 2, "_raw_model_config = dict(_MODEL_CONFIG or {})")
        add(lines, indent * 2, "self._model_config = dict(_raw_model_config)")
        add(lines, indent * 2, "if isinstance(_raw_model_config.get('text_config'), dict):")
        add(lines, indent * 3, "self._model_config.update(_raw_model_config['text_config'])")
        add(lines, indent * 2, "self.config = config")
        add(lines, indent * 2, "from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size")
        add(lines, indent * 2, "_tp_size = get_tensor_model_parallel_world_size()")
        add(lines, indent * 2, "_tp_rank = get_tensor_model_parallel_rank()")
        add(lines, indent * 2, "self._tp_size = _tp_size")
        add(lines, indent * 2, f"_num_heads = {self._config_expr('num_attention_heads', default=1)} // _tp_size")
        add(lines, indent * 2, f"_num_kv_heads = max(1, {self._config_expr('num_key_value_heads', alt='num_attention_heads', default=1)} // _tp_size)")
        add(lines, indent * 2, "def _axon_alibi_slopes(_total_num_heads, _scale=1.0):")
        add(lines, indent * 3, "import math as _math")
        add(lines, indent * 3, "_closest_power_of_2 = 2 ** _math.floor(_math.log2(_total_num_heads))")
        add(lines, indent * 3, "_base = torch.tensor(2 ** (-(2 ** -(_math.log2(_closest_power_of_2) - 3))), dtype=torch.float32)")
        add(lines, indent * 3, "_powers = torch.arange(1, 1 + _closest_power_of_2, dtype=torch.int32)")
        add(lines, indent * 3, "_slopes = torch.pow(_base, _powers)")
        add(lines, indent * 3, "if _closest_power_of_2 != _total_num_heads:")
        add(lines, indent * 4, "_extra_base = torch.tensor(2 ** (-(2 ** -(_math.log2(2 * _closest_power_of_2) - 3))), dtype=torch.float32)")
        add(lines, indent * 4, "_num_remaining_heads = min(_closest_power_of_2, _total_num_heads - _closest_power_of_2)")
        add(lines, indent * 4, "_extra_powers = torch.arange(start=1, end=1 + 2 * _num_remaining_heads, step=2, dtype=torch.int32)")
        add(lines, indent * 4, "_slopes = torch.cat([_slopes, torch.pow(_extra_base, _extra_powers)], dim=0)")
        add(lines, indent * 3, "return _slopes * float(_scale)")
        add(lines, indent * 2, "")
        add(lines, indent * 2, "# --- vLLM layer instantiation ---")
        add(lines, indent * 2, "from vllm.model_executor.layers.linear import (")
        add(lines, indent * 3, "ColumnParallelLinear,")
        add(lines, indent * 3, "MergedColumnParallelLinear,")
        add(lines, indent * 3, "QKVParallelLinear,")
        add(lines, indent * 3, "RowParallelLinear,")
        add(lines, indent * 3, "ReplicatedLinear,")
        add(lines, indent * 2, ")")
        add(lines, indent * 2, "from vllm.model_executor.layers.vocab_parallel_embedding import (")
        add(lines, indent * 3, "VocabParallelEmbedding,")
        add(lines, indent * 3, "ParallelLMHead,")
        add(lines, indent * 2, ")")
        add(lines, indent * 2, "from vllm.model_executor.layers.attention import Attention")
        add(lines, indent * 2, "from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm")
        add(lines, indent * 2, "")

        add(lines, indent * 2, "self.state_dict_tensors = {}")
        add(lines, indent * 2, "self._loaded_state_keys = set()")
        self._emit_vllm_layer_inits(lines, classification, indent * 2)

        # --- RoPE (rotary embedding) ---
        repeated_mod = self._get_repeated_module()
        has_rope = False
        if repeated_mod is not None:
            has_rope = self._detect_rope(repeated_mod) is not None
        if has_rope:
            num_layers_expr = self._config_expr("num_hidden_layers")
            head_dim_expr = self._head_dim_expr()
            add(lines, indent * 2, "from vllm.model_executor.layers.rotary_embedding import get_rope")
            rope_variants = self._detect_rope_variants()
            if rope_variants:
                (
                    local_hd,
                    local_theta,
                    local_scale,
                    local_partial,
                    full_hd,
                    full_theta,
                    full_scale,
                    full_partial,
                    cond_expr,
                ) = rope_variants
                add(lines, indent * 2, f"_rope_period = {cond_expr}")
                full_params = (
                    f"{{'rope_type': 'default', 'rope_theta': {full_theta}, 'factor': {full_scale}, 'partial_rotary_factor': {full_partial}}}"
                    if str(full_scale) in {"1", "1.0"}
                    else f"{{'rope_type': 'proportional', 'rope_theta': {full_theta}, 'factor': {full_scale}, 'partial_rotary_factor': {full_partial}}}"
                )
                add(lines, indent * 2, "self.rotary_emb = nn.ModuleList([")
                add(lines, indent * 3, "get_rope(")
                add(lines, indent * 4, f"({full_hd} if ((i + 1) % _rope_period == 0) else {local_hd}),")
                add(lines, indent * 4, f"max_position={self._config_expr('max_position_embeddings', default=4096)},")
                add(lines, indent * 4, "is_neox_style=True,")
                add(lines, indent * 4, "rope_parameters=(")
                add(lines, indent * 5, "dict(config.rope_parameters[config.layer_types[i]])")
                add(lines, indent * 5, "if hasattr(config, 'rope_parameters') and hasattr(config, 'layer_types')")
                add(lines, indent * 5, "and isinstance(config.rope_parameters, dict) and i < len(config.layer_types)")
                add(lines, indent * 5, "and config.layer_types[i] in config.rope_parameters")
                add(lines, indent * 5, "else (")
                add(lines, indent * 6, full_params)
                add(lines, indent * 6, "if ((i + 1) % _rope_period == 0)")
                add(lines, indent * 6, f"else {{'rope_type': 'default', 'rope_theta': {local_theta}, 'factor': {local_scale}, 'partial_rotary_factor': {local_partial}}}")
                add(lines, indent * 5, ")")
                add(lines, indent * 4, "),")
                add(lines, indent * 3, f") for i in range({num_layers_expr})")
                add(lines, indent * 2, "])")
            else:
                add(lines, indent * 2, f"_rope_theta = {self._config_expr('rope_theta', default=10000.0)}")
                add(lines, indent * 2, f"_rope_params = {{'rope_type': 'default', 'rope_theta': _rope_theta, 'partial_rotary_factor': {self._config_expr('partial_rotary_factor', default=1.0)}}}")
                add(lines, indent * 2, "self.rotary_emb = nn.ModuleList([")
                add(lines, indent * 3, f"get_rope({head_dim_expr},")
                add(lines, indent * 3, f"max_position={self._config_expr('max_position_embeddings', default=4096)},")
                add(lines, indent * 3, "is_neox_style=True,")
                add(lines, indent * 3, "rope_parameters=_rope_params,")
                add(lines, indent * 3, f") for i in range({num_layers_expr})")
                add(lines, indent * 2, "])")
            add(lines, indent * 2, "")

        # --- Per-layer scalar ---
        if classification.per_layer_scalar_node_id:
            num_layers_expr = self._config_expr("num_hidden_layers")
            add(lines, indent * 2, f"self.layer_scalars = nn.ParameterList([")
            add(lines, indent * 3, "nn.Parameter(torch.ones(1))")
            add(lines, indent * 3, f"for _ in range({num_layers_expr})")
            add(lines, indent * 2, "])")
            add(lines, indent * 2, "")

        # --- lm_head and logits_processor ---
        lm_head_attr = None
        if classification.lm_head_node_id:
            lm_head_node = self._find_node_by_id(classification.lm_head_node_id)
            if lm_head_node is not None and not self._is_repeated_node(lm_head_node):
                lm_head_path = _linear_base_key(lm_head_node)
                if not lm_head_path and lm_head_node.op.name == "core.select":
                    token_embedding_paths_for_lm = set()
                    for emb_id in classification.token_embedding_node_ids:
                        emb_node = self._find_node_by_id(emb_id)
                        if emb_node is not None and not self._is_repeated_node(emb_node):
                            token_embedding_paths_for_lm.add(_linear_base_key(emb_node))
                    branch_paths: list[str] = []
                    for branch in lm_head_node.inputs[1:]:
                        branch_paths.extend(_linear_base_keys_from_expr_operand(branch))
                    for path in branch_paths:
                        if path and path not in token_embedding_paths_for_lm:
                            lm_head_path = path
                            break
                if lm_head_node.id in self._vllm_emitted_layer_node_ids:
                    lm_head_attr = self._vllm_attr_access(lm_head_node)
            else:
                lm_head_path = None
        else:
            lm_head_node = None
            lm_head_path = None
        if lm_head_path is None and self._use_clean_forward:
            main_mod = self.modules_by_name.get(self.program.main_module)
            if main_mod is not None:
                for node in reversed(main_mod.nodes):
                    node_layer_type = self._vllm_classification.node_types.get(
                        node.id, VLLMLayerType.DEFAULT
                    )
                    if (
                        node_layer_type
                        in (
                            VLLMLayerType.COLUMN_PARALLEL_LINEAR,
                            VLLMLayerType.ROW_PARALLEL_LINEAR,
                            VLLMLayerType.PARALLEL_LM_HEAD,
                        )
                        and len(node.inputs) >= 3
                        and _literal_value(node.inputs[2], None) is None
                    ):
                        lm_head_path = _linear_base_key(node)
                        break
        inferred_lm_head_path = self._infer_lm_head_path(classification)
        if inferred_lm_head_path is not None:
            lm_head_path = inferred_lm_head_path
        if lm_head_attr:
            add(lines, indent * 2, f"self.lm_head = {lm_head_attr}")
        else:
            add(lines, indent * 2, "self.lm_head = ParallelLMHead(")
            add(lines, indent * 3, f"{self._ctor_int_expr(self._config_expr('vocab_size'))},")
            add(lines, indent * 3, f"{self._ctor_int_expr(self._config_expr('hidden_size'))},")
            add(lines, indent * 3, "bias=False,")
            add(lines, indent * 3, "params_dtype=params_dtype,")
            add(lines, indent * 3, f"org_num_embeddings={self._ctor_int_expr(self._config_expr('vocab_size'))},")
            add(lines, indent * 3, "quant_config=quant_config,")
            add(lines, indent * 3, "prefix='lm_head',")
            add(lines, indent * 2, ")")
        # Handle tied word embeddings. Prefer graph-proven path equality over
        # HF config flags; some configs omit tie_word_embeddings even when the
        # Axon graph uses the same parameter path for embedding and LM head.
        config_tie_emb_attr = None
        graph_tie_emb_attr = None
        load_time_tie_emb_attr = None
        for emb_id in classification.token_embedding_node_ids:
            if emb_id == classification.pli_embed_node_id:
                continue
            emb_node = self._find_node_by_id(emb_id)
            if emb_node is not None and not self._is_repeated_node(emb_node):
                emb_info = self._token_embedding_attr_and_prefix(emb_node)
                if emb_info is None:
                    continue
                emb_attr, emb_path = emb_info
                config_tie_emb_attr = config_tie_emb_attr or emb_attr
                if lm_head_path is not None and emb_path == lm_head_path:
                    graph_tie_emb_attr = emb_attr
                    break
                if (
                    lm_head_node is not None
                    and lm_head_node.op.name == "core.select"
                    and any(
                        path == emb_path
                        for branch in lm_head_node.inputs[1:]
                        for path in _linear_base_keys_from_expr_operand(branch)
                    )
                ):
                    load_time_tie_emb_attr = load_time_tie_emb_attr or emb_attr
        if graph_tie_emb_attr is not None:
            add(lines, indent * 2, f"self.lm_head = {graph_tie_emb_attr}")
        elif config_tie_emb_attr is not None:
            add(lines, indent * 2, "if getattr(config, 'tie_word_embeddings', False):")
            add(lines, indent * 3, f"self.lm_head = {config_tie_emb_attr}")
        present_lm_head_attr = self._emitted_linear_attr_for_path(lm_head_path)
        if present_lm_head_attr is not None and present_lm_head_attr != lm_head_attr:
            add(lines, indent * 2, f"self._axon_lm_head_present_attr = {present_lm_head_attr}")
        if load_time_tie_emb_attr is not None and lm_head_path:
            add(lines, indent * 2, f"self._axon_lm_head_state_root = {lm_head_path!r}")
            add(lines, indent * 2, f"self._axon_lm_head_fallback = {load_time_tie_emb_attr}")
        add(lines, indent * 2, "from vllm.model_executor.layers.logits_processor import LogitsProcessor")
        add(lines, indent * 2, "_soft_cap = getattr(config, 'final_logit_softcapping', None)")
        add(lines, indent * 2, "self.logits_processor = LogitsProcessor(")
        add(lines, indent * 3, f"{self._config_expr('vocab_size')},")
        add(lines, indent * 3, "soft_cap=_soft_cap,")
        add(lines, indent * 2, ")")
        add(lines, indent * 2, "")

        # Register embedding normalizer buffer (matches native vLLM: float32
        # buffer that gets dtype-converted to bf16 when model is loaded).
        emb_scale = self._model_config_data.get("embedding_scale")
        if emb_scale:
            add(lines, indent * 2, f"self.register_buffer('normalizer', torch.tensor({float(emb_scale)}), persistent=False)")

        # PLI scale factor buffers (matches native vLLM Gemma4Model).
        if classification.pli_gate_node_id is not None:
            hs_expr = self._config_expr("hidden_size")
            pli_dim_expr = self._resolve_const_value("PLI") or self._config_expr("per_layer_input_dim", default=256)
            add(lines, indent * 2, f"self.register_buffer('_embed_scale_per_layer', torch.tensor({pli_dim_expr} ** 0.5), persistent=False)")
            add(lines, indent * 2, f"self.register_buffer('_per_layer_projection_scale', torch.tensor({hs_expr} ** -0.5), persistent=False)")
            add(lines, indent * 2, "self.register_buffer('_per_layer_input_scale', torch.tensor(2 ** -0.5), persistent=False)")

        # Fused activation for merged gate_up_proj (GeluAndMul)
        has_merged_ffn = any(
            g.gate_node_id and g.up_node_id for g in classification.ffn_groups
        )
        if has_merged_ffn:
            add(lines, indent * 2, "from vllm.model_executor.layers.activation import GeluAndMul")
            add(lines, indent * 2, "self._ffn_act = GeluAndMul(approximate='tanh')")

        # Mamba/SSM state cache placeholder layers
        if self._needs_mamba_cache_placeholders(classification):
            mixer_prefix = self._derive_mamba_mixer_prefix(classification)
            if mixer_prefix:
                i_expr = self._config_expr("intermediate_size")
                n_expr = self._config_expr("state_size")
                k_expr = self._config_expr("conv_kernel")
                num_layers_expr = self._config_expr("num_hidden_layers")
                add(lines, indent * 2, f"self._mamba_placeholders = nn.ModuleList([")
                add(lines, indent * 3, f"_MambaPlaceholderLayer(")
                add(lines, indent * 4, f"prefix={mixer_prefix},")
                add(lines, indent * 4, f"intermediate_size={i_expr},")
                add(lines, indent * 4, f"state_size={n_expr},")
                add(lines, indent * 4, f"conv_kernel={k_expr},")
                add(lines, indent * 3, f")")
                add(lines, indent * 3, f"for i in range({num_layers_expr})")
                add(lines, indent * 2, "])")

        add(lines, indent * 2, "self._build_state_dict_tensors()")
        if not self._use_clean_forward:
            add(lines, indent * 2, "self._eval_symbols()")
        add(lines, indent, "")
        add(lines, indent, "def embed_input_ids(self, input_ids, positions=None):")
        emitted_token_embedding = False
        emitted_position_addend = False
        add(lines, indent * 2, "hidden_states = None")
        for node_id in sorted(classification.token_embedding_node_ids):
            if node_id == classification.pli_embed_node_id:
                continue
            node = self._find_node_by_id(node_id)
            if node is not None:
                native_prefix = self._vllm_embedding_layer_prefix(node)
                if native_prefix is None:
                    suffix = self._embedding_weight_suffix_from_node(node)
                    if suffix is not None:
                        add(lines, indent * 2, "if hidden_states is None:")
                        add(lines, indent * 3, f"_suffix = {suffix!r}")
                        add(
                            lines,
                            indent * 3,
                            "_matches = [(k, v) for k, v in self.state_dict_tensors.items() "
                            "if k == _suffix or str(k).endswith('.' + _suffix)]",
                        )
                        add(lines, indent * 3, "if len(_matches) != 1:")
                        add(
                            lines,
                            indent * 4,
                            "raise KeyError(f'expected exactly one token embedding weight matching suffix {_suffix!r}, found {[k for k, _ in _matches]}')",
                        )
                        add(lines, indent * 3, "_key, _w = _matches[0]")
                        add(lines, indent * 3, "_w = self._vllm_state_tensor(_key, _w)")
                        add(lines, indent * 3, "_ids = input_ids")
                        add(lines, indent * 3, "if _ids is None:")
                        add(
                            lines,
                            indent * 4,
                            "_n = int(positions.shape[0]) if positions is not None else 1",
                        )
                        add(
                            lines,
                            indent * 4,
                            "_ids = torch.zeros((_n,), device=_w.device, dtype=torch.long)",
                        )
                        add(lines, indent * 3, "hidden_states = F.embedding(_ids.to(_w.device), _w)")
                        scale_expr = self._token_embedding_scale_expr(node)
                        if scale_expr is not None:
                            add(lines, indent * 3, f"hidden_states = hidden_states * ({scale_expr})")
                        elif hasattr(self, '_model_config_data') and self._model_config_data.get("embedding_scale"):
                            add(lines, indent * 3, "hidden_states = hidden_states * self.normalizer")
                        emitted_token_embedding = True
                        continue
                    module = self._module_for_node(node) or self.modules_by_name.get(self.program.main_module)
                    expr = None
                    if module is not None and node.outputs:
                        output_name = _value_name(node.outputs[0])
                        if output_name is not None:
                            expr = self._preloop_operand_expr(
                                GraphValueRef(output_name, node.outputs[0].type_expr, node.outputs[0].dims),
                                module,
                            )
                    if expr is None:
                        continue
                    add(lines, indent * 2, "if hidden_states is None:")
                    add(lines, indent * 3, f"hidden_states = {expr}")
                    emitted_token_embedding = True
                    continue
                attr = self._vllm_attr_access(node)
                weight_key = f"{native_prefix}.weight"
                if weight_key is not None:
                    add(lines, indent * 2, f"if hidden_states is None and {weight_key!r} in self._loaded_state_keys:")
                    branch_indent = indent * 3
                else:
                    add(lines, indent * 2, "if hidden_states is None:")
                    branch_indent = indent * 3
                scale_expr = self._token_embedding_scale_expr(node)
                if scale_expr is not None:
                    add(lines, branch_indent, f"hidden_states = {attr}(input_ids) * ({scale_expr})")
                elif hasattr(self, '_model_config_data') and self._model_config_data.get("embedding_scale"):
                    add(lines, branch_indent, f"hidden_states = {attr}(input_ids) * self.normalizer")
                else:
                    add(lines, branch_indent, f"hidden_states = {attr}(input_ids)")
                addend_expr = self._token_embedding_addend_expr(node)
                if addend_expr is not None:
                    add(lines, branch_indent, "if positions is not None:")
                    add(lines, branch_indent + 1, f"hidden_states = hidden_states + ({addend_expr})")
                    emitted_position_addend = True
                emitted_token_embedding = True
        if not emitted_token_embedding:
            add(lines, indent * 2, "pass")
        else:
            add(lines, indent * 2, "if hidden_states is None:")
            add(lines, indent * 3, "raise KeyError('no compatible token embedding weight was loaded')")
        if not emitted_position_addend:
            for node_id in sorted(classification.position_embedding_node_ids):
                if node_id == classification.pli_embed_node_id:
                    continue
                node = self._find_node_by_id(node_id)
                if node is None:
                    continue
                expr = self._position_embedding_expr(node)
                if expr is None:
                    continue
                weight_key = self._static_param_key(node.inputs[0], "weight") if node.inputs else None
                if weight_key is not None:
                    add(lines, indent * 2, f"if positions is not None and {weight_key!r} in self._loaded_state_keys:")
                else:
                    add(lines, indent * 2, "if positions is not None:")
                add(lines, indent * 3, f"hidden_states = hidden_states + ({expr})")
        add(lines, indent * 2, "return hidden_states")
        add(lines, indent, "")

        add(lines, indent, "def _loop_index_from_scope(self, scope, segment_index):")
        add(lines, indent * 2, "parts = str(scope).strip().lstrip('@').split('.')")
        add(lines, indent * 2, "if segment_index < 0:")
        add(lines, indent * 3, "segment_index += len(parts)")
        add(lines, indent * 2, "if segment_index < 0 or segment_index >= len(parts):")
        add(lines, indent * 3, "raise ValueError(f'cannot extract loop index from scope={scope!r} segment={segment_index}')")
        add(lines, indent * 2, "return int(parts[segment_index])")
        add(lines, indent, "")

        add(lines, indent, "def load_weights(self, weights):")
        add(lines, indent * 2, "import re as _re")
        add(lines, indent * 2, "stacked_params_mapping = [")
        add(lines, indent * 3, '(".qkv_proj", ".q_proj", "q"),')
        add(lines, indent * 3, '(".qkv_proj", ".k_proj", "k"),')
        add(lines, indent * 3, '(".qkv_proj", ".v_proj", "v"),')
        has_merged_ffn = any(
            g.gate_node_id and g.up_node_id for g in classification.ffn_groups
        )
        if has_merged_ffn:
            add(lines, indent * 3, '(".gate_up_proj", ".gate_proj", 0),')
            add(lines, indent * 3, '(".gate_up_proj", ".up_proj", 1),')
        add(lines, indent * 2, "]")
        add(lines, indent * 2, "params_dict = dict(self.named_parameters(remove_duplicate=False))")
        add(lines, indent * 2, "params_dict.update(dict(self.named_buffers(remove_duplicate=False)))")
        add(lines, indent * 2, "loaded_params = set()")
        add(lines, indent * 2, "_debug_seen_weights = 0")
        add(lines, indent * 2, "_transpose_model_param_names = set()")
        add(lines, indent * 2, "_interleaved_qkv_model_param_names = set()")
        add(lines, indent * 2, "_grouped_qkv_model_param_names = set()")
        for module in self.program.modules:
            for node in module.nodes:
                if not _is_linear_call(node, self.modules_by_name):
                    continue
                if node.id not in self._vllm_emitted_layer_node_ids:
                    continue
                if not self._linear_transpose_arg(node, False):
                    continue
                if node.id not in classification.node_types:
                    continue
                attr_name = self._vllm_layer_attr_name(node)
                if self._is_repeated_node(node) or "{i}" in self._node_prefix(node):
                    add(lines, indent * 2, f"for _i, _mod in enumerate(self.{attr_name}):")
                    add(lines, indent * 3, "if hasattr(_mod, 'weight'):")
                    add(lines, indent * 4, f"_transpose_model_param_names.add(f'{attr_name}.{{_i}}.weight')")
                else:
                    add(lines, indent * 2, f"if hasattr(self.{attr_name}, 'weight'):")
                    add(lines, indent * 3, f"_transpose_model_param_names.add('{attr_name}.weight')")
        for group in classification.qkv_groups:
            if group.layout != "interleaved":
                continue
            if group.q_node_id != group.k_node_id or group.q_node_id != group.v_node_id:
                continue
            node = self._find_node_by_id(group.q_node_id)
            if node is None:
                continue
            if node.id not in self._vllm_emitted_layer_node_ids:
                continue
            attr_name = self._vllm_layer_attr_name(node)
            if self._is_repeated_node(node) or "{i}" in self._node_prefix(node):
                add(lines, indent * 2, f"for _i, _mod in enumerate(self.{attr_name}):")
                add(lines, indent * 3, "if hasattr(_mod, 'weight'):")
                add(lines, indent * 4, f"_interleaved_qkv_model_param_names.add(f'{attr_name}.{{_i}}.weight')")
                add(lines, indent * 3, "if hasattr(_mod, 'bias') and _mod.bias is not None:")
                add(lines, indent * 4, f"_interleaved_qkv_model_param_names.add(f'{attr_name}.{{_i}}.bias')")
            else:
                add(lines, indent * 2, f"if hasattr(self.{attr_name}, 'weight'):")
                add(lines, indent * 3, f"_interleaved_qkv_model_param_names.add('{attr_name}.weight')")
                add(lines, indent * 2, f"if hasattr(self.{attr_name}, 'bias') and self.{attr_name}.bias is not None:")
                add(lines, indent * 3, f"_interleaved_qkv_model_param_names.add('{attr_name}.bias')")
        for group in classification.qkv_groups:
            if group.layout != "grouped":
                continue
            if group.q_node_id != group.k_node_id or group.q_node_id != group.v_node_id:
                continue
            node = self._find_node_by_id(group.q_node_id)
            if node is None:
                continue
            if node.id not in self._vllm_emitted_layer_node_ids:
                continue
            attr_name = self._vllm_layer_attr_name(node)
            if self._is_repeated_node(node) or "{i}" in self._node_prefix(node):
                add(lines, indent * 2, f"for _i, _mod in enumerate(self.{attr_name}):")
                add(lines, indent * 3, "if hasattr(_mod, 'weight'):")
                add(lines, indent * 4, f"_grouped_qkv_model_param_names.add(f'{attr_name}.{{_i}}.weight')")
                add(lines, indent * 3, "if hasattr(_mod, 'bias') and _mod.bias is not None:")
                add(lines, indent * 4, f"_grouped_qkv_model_param_names.add(f'{attr_name}.{{_i}}.bias')")
            else:
                add(lines, indent * 2, f"if hasattr(self.{attr_name}, 'weight'):")
                add(lines, indent * 3, f"_grouped_qkv_model_param_names.add('{attr_name}.weight')")
                add(lines, indent * 2, f"if hasattr(self.{attr_name}, 'bias') and self.{attr_name}.bias is not None:")
                add(lines, indent * 3, f"_grouped_qkv_model_param_names.add('{attr_name}.bias')")
        add(lines, indent * 2, "def _maybe_transpose_for_param(param, tensor, *, force=False):")
        add(lines, indent * 3, "if force and torch.is_tensor(tensor) and tensor.ndim == 2:")
        add(lines, indent * 4, "return tensor.t().contiguous()")
        add(lines, indent * 3, "return tensor")
        add(lines, indent * 2, "def _maybe_reorder_interleaved_qkv(target_name, param, tensor):")
        add(lines, indent * 3, "if target_name not in _interleaved_qkv_model_param_names:")
        add(lines, indent * 4, "return tensor")
        add(lines, indent * 3, "if not torch.is_tensor(tensor):")
        add(lines, indent * 4, "return tensor")
        add(lines, indent * 3, "output_dim = getattr(param, 'output_dim', None)")
        add(lines, indent * 3, "if output_dim is None:")
        add(lines, indent * 4, "output_dim = 0")
        add(lines, indent * 3, "num_heads = getattr(self.config, 'num_attention_heads', None)")
        add(lines, indent * 3, "if num_heads is None:")
        add(lines, indent * 4, "num_heads = getattr(self.config, 'n_head', None)")
        add(lines, indent * 3, "if num_heads is None:")
        add(lines, indent * 4, "raise ValueError(f'interleaved QKV load for {target_name} requires num_attention_heads/n_head')")
        add(lines, indent * 3, "num_heads = int(num_heads)")
        add(lines, indent * 3, "shape = tuple(tensor.shape)")
        add(lines, indent * 3, "if output_dim < 0:")
        add(lines, indent * 4, "output_dim += len(shape)")
        add(lines, indent * 3, "if output_dim < 0 or output_dim >= len(shape):")
        add(lines, indent * 4, "raise ValueError(f'interleaved QKV load for {target_name} has invalid output_dim={output_dim} shape={shape}')")
        add(lines, indent * 3, "if shape[output_dim] % (3 * num_heads) != 0:")
        add(lines, indent * 4, "raise ValueError(f'interleaved QKV load for {target_name} cannot split shape={shape} num_heads={num_heads}')")
        add(lines, indent * 3, "view_shape = shape[:output_dim] + (num_heads, 3, -1) + shape[output_dim + 1:]")
        add(lines, indent * 3, "return tensor.view(view_shape).transpose(output_dim, output_dim + 1).reshape(shape).contiguous()")
        add(lines, indent * 2, "def _maybe_reorder_grouped_qkv(target_name, param, tensor):")
        add(lines, indent * 3, "if target_name not in _grouped_qkv_model_param_names:")
        add(lines, indent * 4, "return tensor")
        add(lines, indent * 3, "if not torch.is_tensor(tensor):")
        add(lines, indent * 4, "return tensor")
        add(lines, indent * 3, "output_dim = getattr(param, 'output_dim', None)")
        add(lines, indent * 3, "if output_dim is None:")
        add(lines, indent * 4, "output_dim = 0")
        add(lines, indent * 3, "num_heads = getattr(self.config, 'num_attention_heads', None)")
        add(lines, indent * 3, "if num_heads is None:")
        add(lines, indent * 4, "num_heads = getattr(self.config, 'n_head', None)")
        add(lines, indent * 3, "num_kv_heads = getattr(self.config, 'num_key_value_heads', None)")
        add(lines, indent * 3, "if num_kv_heads is None:")
        add(lines, indent * 4, "num_kv_heads = getattr(self.config, 'num_attention_heads', None)")
        add(lines, indent * 3, "if num_kv_heads is None:")
        add(lines, indent * 4, "num_kv_heads = getattr(self.config, 'n_head', None)")
        add(lines, indent * 3, "if num_heads is None or num_kv_heads is None:")
        add(lines, indent * 4, "raise ValueError(f'grouped QKV load for {target_name} requires attention head config')")
        add(lines, indent * 3, "num_heads = int(num_heads)")
        add(lines, indent * 3, "num_kv_heads = int(num_kv_heads)")
        add(lines, indent * 3, "if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:")
        add(lines, indent * 4, "raise ValueError(f'grouped QKV load for {target_name} has invalid heads num_heads={num_heads} num_kv_heads={num_kv_heads}')")
        add(lines, indent * 3, "shape = tuple(tensor.shape)")
        add(lines, indent * 3, "if output_dim < 0:")
        add(lines, indent * 4, "output_dim += len(shape)")
        add(lines, indent * 3, "if output_dim < 0 or output_dim >= len(shape):")
        add(lines, indent * 4, "raise ValueError(f'grouped QKV load for {target_name} has invalid output_dim={output_dim} shape={shape}')")
        add(lines, indent * 3, "out = int(shape[output_dim])")
        add(lines, indent * 3, "denom = num_heads + 2 * num_kv_heads")
        add(lines, indent * 3, "if denom <= 0 or out % denom != 0:")
        add(lines, indent * 4, "raise ValueError(f'grouped QKV load for {target_name} cannot infer head_dim from shape={shape} heads={num_heads} kv_heads={num_kv_heads}')")
        add(lines, indent * 3, "head_dim = out // denom")
        add(lines, indent * 3, "q_groups = num_heads // num_kv_heads")
        add(lines, indent * 3, "moved = tensor.movedim(output_dim, 0).contiguous()")
        add(lines, indent * 3, "rest = tuple(moved.shape[1:])")
        add(lines, indent * 3, "view = moved.view(num_kv_heads, q_groups + 2, head_dim, *rest)")
        add(lines, indent * 3, "q = view[:, :q_groups, :, ...].reshape(num_heads * head_dim, *rest)")
        add(lines, indent * 3, "k = view[:, q_groups:q_groups + 1, :, ...].reshape(num_kv_heads * head_dim, *rest)")
        add(lines, indent * 3, "v = view[:, q_groups + 1:q_groups + 2, :, ...].reshape(num_kv_heads * head_dim, *rest)")
        add(lines, indent * 3, "packed = torch.cat([q, k, v], dim=0)")
        add(lines, indent * 3, "return packed.movedim(0, output_dim).contiguous()")
        add(lines, indent * 2, "def _compatible_direct_load(param, tensor):")
        add(lines, indent * 3, "pdata = getattr(param, 'data', None)")
        add(lines, indent * 3, "if getattr(param, 'weight_loader', None) is not None:")
        add(lines, indent * 4, "return True")
        add(lines, indent * 3, "if not (torch.is_tensor(pdata) and torch.is_tensor(tensor)):")
        add(lines, indent * 4, "return True")
        add(lines, indent * 3, "if tuple(pdata.shape) == tuple(tensor.shape):")
        add(lines, indent * 4, "return True")
        add(lines, indent * 3, "if pdata.ndim == 2 and tensor.ndim == 2 and tuple(pdata.shape) == tuple(tensor.t().shape):")
        add(lines, indent * 4, "return True")
        add(lines, indent * 3, "return tensor.numel() >= pdata.numel()")
        add(lines, indent * 2, "def _try_load_param(param, tensor, shard_id=None):")
        add(lines, indent * 3, "pdata = getattr(param, 'data', None)")
        add(lines, indent * 3, "if (")
        add(lines, indent * 4, "torch.is_tensor(pdata)")
        add(lines, indent * 4, "and torch.is_tensor(tensor)")
        add(lines, indent * 4, "and pdata.is_floating_point()")
        add(lines, indent * 4, "and tensor.is_floating_point()")
        add(lines, indent * 4, "and pdata.dtype != tensor.dtype")
        add(lines, indent * 3, "):")
        add(lines, indent * 4, "tensor = tensor.to(dtype=pdata.dtype)")
        add(lines, indent * 3, "weight_loader = getattr(param, 'weight_loader', None)")
        add(lines, indent * 3, "try:")
        add(lines, indent * 4, "if weight_loader is not None:")
        add(lines, indent * 5, "if shard_id is None:")
        add(lines, indent * 6, "weight_loader(param, tensor)")
        add(lines, indent * 5, "else:")
        add(lines, indent * 6, "weight_loader(param, tensor, shard_id)")
        add(lines, indent * 4, "else:")
        add(lines, indent * 5, "param.data.copy_(tensor)")
        add(lines, indent * 4, "return True")
        add(lines, indent * 3, "except RuntimeError as _exc:")
        add(lines, indent * 4, "_msg = str(_exc)")
        add(lines, indent * 4, "if (")
        add(lines, indent * 5, "'exceeds dimension size' in _msg")
        add(lines, indent * 5, "or 'size mismatch' in _msg")
        add(lines, indent * 5, "or 'must match' in _msg")
        add(lines, indent * 5, "or 'The size of tensor' in _msg")
        add(lines, indent * 4, "):")
        add(lines, indent * 5, "return False")
        add(lines, indent * 4, "raise")
        add(lines, indent * 2, "def _load_named_param(target_name, tensor, shard_id=None):")
        add(lines, indent * 3, "if target_name not in params_dict:")
        add(lines, indent * 4, "return False")
        add(lines, indent * 3, "param = params_dict[target_name]")
        add(lines, indent * 3, "tensor = _maybe_reorder_interleaved_qkv(target_name, param, tensor)")
        add(lines, indent * 3, "tensor = _maybe_reorder_grouped_qkv(target_name, param, tensor)")
        add(lines, indent * 3, "tensor = _maybe_transpose_for_param(param, tensor, force=target_name in _transpose_model_param_names)")
        add(lines, indent * 3, "if not _compatible_direct_load(param, tensor):")
        add(lines, indent * 4, "return False")
        add(lines, indent * 3, "if not _try_load_param(param, tensor, shard_id):")
        add(lines, indent * 4, "return False")
        add(lines, indent * 3, "loaded_params.add(target_name)")
        add(lines, indent * 3, "return True")
        add(lines, indent * 2, "")
        add(lines, indent * 2, "# Build checkpoint name -> model param name mapping from module prefixes")
        add(lines, indent * 2, "_ckpt_to_model = {}")
        add(lines, indent * 2, "for _mod_name, _module in self.named_modules():")
        add(lines, indent * 3, "_prefix = getattr(_module, 'prefix', None)")
        add(lines, indent * 3, "if _prefix:")
        add(lines, indent * 4, "_resolved = _prefix")
        add(lines, indent * 4, "if '{i}' in _prefix:")
        add(lines, indent * 5, "_idx = None")
        add(lines, indent * 5, "for _p in reversed(_mod_name.split('.')):")
        add(lines, indent * 6, "if _p.isdigit():")
        add(lines, indent * 7, "_idx = int(_p)")
        add(lines, indent * 7, "break")
        add(lines, indent * 5, "if _idx is None:")
        add(lines, indent * 6, "continue")
        add(lines, indent * 5, "_resolved = _prefix.replace('{i}', str(_idx))")
        add(lines, indent * 4, "for _pname, _ in _module.named_parameters(recurse=False):")
        add(lines, indent * 5, "_ck = f'{_resolved}.{_pname}'")
        add(lines, indent * 5, "_ckpt_to_model.setdefault(_ck, []).append(f'{_mod_name}.{_pname}')")
        add(lines, indent * 4, "for _bname, _ in _module.named_buffers(recurse=False):")
        add(lines, indent * 5, "_ck = f'{_resolved}.{_bname}'")
        add(lines, indent * 5, "_ckpt_to_model.setdefault(_ck, []).append(f'{_mod_name}.{_bname}')")
        add(lines, indent * 2, "")
        for module in self.program.modules:
            for node in module.nodes:
                layer_type = classification.node_types.get(node.id)
                if layer_type not in (VLLMLayerType.RMSNORM, VLLMLayerType.LAYERNORM):
                    continue
                emitted_ids = getattr(self, "_vllm_emitted_layer_node_ids", set())
                if emitted_ids and node.id not in emitted_ids:
                    continue
                attr_name = self._vllm_layer_attr_name(node)
                prefix_expr = self._layer_prefix(node)
                loop_name = self._node_loop_index(node)
                prefix_expr_for_i = prefix_expr.replace("{" + loop_name + "}", "{_i}")
                fields = ("weight", "bias") if layer_type == VLLMLayerType.LAYERNORM else ("weight",)
                if self._is_repeated_node(node) or "{i}" in self._node_prefix(node):
                    add(lines, indent * 2, f"for _i, _mod in enumerate(self.{attr_name}):")
                    add(lines, indent * 3, f"_norm_prefix = {prefix_expr_for_i}")
                    for field in fields:
                        add(lines, indent * 3, f"_ckpt_to_model.setdefault(f'{{_norm_prefix}}.{field}', []).append(f'{attr_name}.{{_i}}.{field}')")
                else:
                    add(lines, indent * 2, f"_norm_prefix = {prefix_expr}")
                    for field in fields:
                        add(lines, indent * 2, f"_ckpt_to_model.setdefault(f'{{_norm_prefix}}.{field}', []).append('{attr_name}.{field}')")
        add(lines, indent * 2, "")
        lm_head_path = self._infer_lm_head_path(classification)
        token_embedding_paths: set[str] = set()
        for emb_id in classification.token_embedding_node_ids:
            emb_node = self._find_node_by_id(emb_id)
            if emb_node is not None and not self._is_repeated_node(emb_node):
                token_embedding_paths.add(_linear_base_key(emb_node))
        if lm_head_path and lm_head_path not in token_embedding_paths:
            lm_head_targets = ["lm_head.weight"]
            if classification.lm_head_node_id:
                lm_head_node = self._find_node_by_id(classification.lm_head_node_id)
                if (
                    lm_head_node is not None
                    and lm_head_node.id in getattr(self, "_vllm_emitted_layer_node_ids", set())
                ):
                    lm_head_targets.append(f"{self._vllm_attr_access(lm_head_node)}.weight")
            for target in dict.fromkeys(lm_head_targets):
                add(
                    lines,
                    indent * 2,
                    f"_ckpt_to_model.setdefault({(lm_head_path + '.weight')!r}, []).append({target!r})",
                )
            add(lines, indent * 2, "")
        if classification.per_layer_scalar_node_id:
            # Derive the layer prefix from scope_parts of the module containing
            # the per_layer_scalar node (e.g. "model.language_model.layers.{i}").
            scalar_node = self._find_node_by_id(classification.per_layer_scalar_node_id)
            layer_prefix_expr = self._derive_layer_prefix_expr(scalar_node, classification)
            add(lines, indent * 2, "# Map layer_scalar checkpoint names to ParameterList entries")
            add(lines, indent * 2, "if hasattr(self, 'layer_scalars'):")
            add(lines, indent * 3, "for _i in range(len(self.layer_scalars)):")
            add(lines, indent * 4, f"_layer_prefix = {layer_prefix_expr}")
            add(lines, indent * 4, f'_ckpt_to_model.setdefault(f"{{_layer_prefix}}.layer_scalar", []).append(f"layer_scalars.{{_i}}")')
            add(lines, indent * 2, "")
        if classification.has_k_eq_v:
            add(lines, indent * 2, "_config = self.config")
            add(lines, indent * 2, "_use_k_eq_v = getattr(_config, 'attention_k_eq_v', False)")
            add(lines, indent * 2, "_k_eq_v_layers = set()")
            add(lines, indent * 2, "if _use_k_eq_v:")
            add(lines, indent * 3, "for _idx, _lt in enumerate(getattr(_config, 'layer_types', [])):")
            add(lines, indent * 4, "if _lt == 'full_attention':")
            add(lines, indent * 5, "_k_eq_v_layers.add(_idx)")
            add(lines, indent * 2, "")
        add(lines, indent * 2, "for name, loaded_weight in weights:")
        add(lines, indent * 3, "_debug_seen_weights += 1")
        self._emit_weight_loading_body(lines, classification, indent * 3)
        add(lines, indent * 2, "_fallback_lm_head = getattr(self, '_axon_lm_head_fallback', None)")
        add(lines, indent * 2, "_present_lm_head = getattr(self, '_axon_lm_head_present_attr', None)")
        add(lines, indent * 2, "_lm_head_root = getattr(self, '_axon_lm_head_state_root', None)")
        add(lines, indent * 2, "if _fallback_lm_head is not None and _lm_head_root is not None:")
        add(lines, indent * 3, "_has_lm_head = any(k == _lm_head_root or str(k).startswith(str(_lm_head_root) + '.') for k in self._loaded_state_keys)")
        add(lines, indent * 3, "if _has_lm_head and _present_lm_head is not None:")
        add(lines, indent * 4, "self.lm_head = _present_lm_head")
        add(lines, indent * 3, "elif not _has_lm_head:")
        add(lines, indent * 4, "self.lm_head = _fallback_lm_head")
        add(lines, indent * 2, "elif _present_lm_head is not None:")
        add(lines, indent * 3, "self.lm_head = _present_lm_head")
        add(lines, indent * 2, "self._debug_load_summary(_debug_seen_weights, loaded_params, params_dict)")
        add(lines, indent * 2, "if hasattr(self, '_eval_symbols'):")
        add(lines, indent * 3, "self._eval_symbols()")
        add(lines, indent, "")

        add(lines, indent, "def compute_logits(self, hidden_states):")
        if self._use_clean_forward:
            add(lines, indent * 2, "logits = self.logits_processor(self.lm_head, hidden_states)")
            add(lines, indent * 2, "self._debug_tensor_stats('compute_logits.hidden_states', hidden_states)")
            add(lines, indent * 2, "self._debug_tensor_stats('compute_logits.logits', logits)")
            add(lines, indent * 2, "return logits")
        else:
            add(lines, indent * 2, "self._debug_tensor_stats('compute_logits.precomputed_logits', hidden_states)")
            add(lines, indent * 2, "return hidden_states")
        add(lines, indent, "")
        add(lines, indent, "def make_empty_intermediate_tensors(")
        add(lines, indent * 2, "self, batch_size, dtype, device,")
        add(lines, indent * 2, "):")
        add(lines, indent * 2, "return {}")
        add(lines, indent, "")
        add(lines, indent, "def _build_state_dict_tensors(self):")
        add(lines, indent * 2, "_num_layers = getattr(self.config, 'num_hidden_layers', 0)")
        add(lines, indent * 2, "for _mod_name, _module in self.named_modules():")
        add(lines, indent * 3, "_prefix = getattr(_module, 'prefix', None)")
        add(lines, indent * 3, "if not _prefix:")
        add(lines, indent * 4, "continue")
        add(lines, indent * 3, "_resolved = _prefix")
        add(lines, indent * 3, "if '{i}' in _prefix:")
        add(lines, indent * 4, "_idx = None")
        add(lines, indent * 4, "for _p in reversed(_mod_name.split('.')):")
        add(lines, indent * 5, "if _p.isdigit():")
        add(lines, indent * 6, "_idx = int(_p)")
        add(lines, indent * 6, "break")
        add(lines, indent * 4, "if _idx is not None:")
        add(lines, indent * 5, "_resolved = _prefix.replace('{i}', str(_idx))")
        add(lines, indent * 4, "else:")
        add(lines, indent * 5, "continue")
        add(lines, indent * 3, "for _pname, _param in _module.named_parameters(recurse=False):")
        add(lines, indent * 4, "self.state_dict_tensors[f'{_resolved}.{_pname}'] = _param")
        add(lines, indent * 3, "for _bname, _buf in _module.named_buffers(recurse=False):")
        add(lines, indent * 4, "self.state_dict_tensors[f'{_resolved}.{_bname}'] = _buf")
        add(lines, indent, "")
        add(lines, indent, "def _debug_enabled(self):")
        add(lines, indent * 2, "import os")
        add(lines, indent * 2, "return os.environ.get('AXON_VLLM_DEBUG_STATS') not in (None, '', '0', 'false', 'False')")
        add(lines, indent, "")
        add(lines, indent, "def _debug_tensor_stats(self, label, value):")
        add(lines, indent * 2, "if not self._debug_enabled() or not torch.is_tensor(value):")
        add(lines, indent * 3, "return")
        add(lines, indent * 2, "with torch.no_grad():")
        add(lines, indent * 3, "sample = value.detach()")
        add(lines, indent * 3, "if sample.numel() == 0:")
        add(lines, indent * 4, "print(f'[axon-vllm-debug] {label}: shape={tuple(sample.shape)} dtype={sample.dtype} device={sample.device} empty=True', flush=True)")
        add(lines, indent * 4, "return")
        add(lines, indent * 3, "stats = sample.float() if sample.is_floating_point() else sample.to(torch.float32)")
        add(lines, indent * 3, "print(f'[axon-vllm-debug] {label}: shape={tuple(sample.shape)} dtype={sample.dtype} device={sample.device} mean={stats.mean().item():.6g} std={stats.std(unbiased=False).item():.6g} min={stats.min().item():.6g} max={stats.max().item():.6g}', flush=True)")
        add(lines, indent, "")
        add(lines, indent, "def _debug_load_summary(self, seen_weights, loaded_params, params_dict):")
        add(lines, indent * 2, "if not self._debug_enabled():")
        add(lines, indent * 3, "return")
        add(lines, indent * 2, "missing = sorted(set(params_dict) - set(loaded_params))")
        add(lines, indent * 2, "loaded = sorted(loaded_params)")
        add(lines, indent * 2, "print(f'[axon-vllm-debug] load_weights: seen={seen_weights} loaded={len(loaded)} params={len(params_dict)} state_tensors={len(self.state_dict_tensors)} missing={len(missing)}', flush=True)")
        add(lines, indent * 2, "print(f'[axon-vllm-debug] load_weights.loaded_first={loaded[:20]}', flush=True)")
        add(lines, indent * 2, "print(f'[axon-vllm-debug] load_weights.missing_first={missing[:20]}', flush=True)")
        add(lines, indent * 2, "interesting = [name for name in loaded if any(frag in name for frag in ('embed', 'lm_head', 'ln_f', 'block', 'qkv', 'proj', 'norm', 'weight'))]")
        add(lines, indent * 2, "for _name in interesting[:20]:")
        add(lines, indent * 3, "self._debug_tensor_stats(f'param.{_name}', params_dict[_name])")
        add(lines, indent, "")
        add(lines, indent, "@staticmethod")
        add(lines, indent, "def _gegelu(x, limit=None, alpha=1.702):")
        add(lines, indent * 2, "if x.shape[-1] % 2 != 0:")
        add(lines, indent * 3, "raise ValueError('gegelu requires even last dimension')")
        add(lines, indent * 2, "x_gelu = x[..., ::2]")
        add(lines, indent * 2, "x_linear = x[..., 1::2]")
        add(lines, indent * 2, "if limit is not None:")
        add(lines, indent * 3, "limit = float(limit)")
        add(lines, indent * 3, "x_gelu = torch.where(torch.isinf(x_gelu), x_gelu, x_gelu.clamp(max=limit))")
        add(lines, indent * 3, "x_linear = torch.where(torch.isinf(x_linear), x_linear, x_linear.clamp(min=-limit, max=limit))")
        add(lines, indent * 2, "return x_gelu * torch.sigmoid(float(alpha) * x_gelu) * (x_linear + 1.0)")
        add(lines, indent, "")
        add(lines, indent, "@staticmethod")
        add(lines, indent, "def _vllm_sinusoidal_positions(ref, position_ids, theta=10000.0, offset=2, padding_idx=None, mode=None):")
        add(lines, indent * 2, "if position_ids is None:")
        add(lines, indent * 3, "raise ValueError('sinusoidal positions require vLLM positions')")
        add(lines, indent * 2, "d = int(ref.shape[-1])")
        add(lines, indent * 2, "half = d // 2")
        add(lines, indent * 2, "pos = position_ids.to(device=ref.device, dtype=torch.float32) + float(offset)")
        add(lines, indent * 2, "pos = pos.reshape(-1, 1)")
        add(lines, indent * 2, "freq_idx = torch.arange(0, half, device=ref.device, dtype=torch.float32)")
        add(lines, indent * 2, "if str(mode) == 'rope':")
        add(lines, indent * 3, "den = max(1, half)")
        add(lines, indent * 2, "else:")
        add(lines, indent * 3, "den = max(1, half - 1)")
        add(lines, indent * 2, "inv_freq = torch.exp(-(torch.log(torch.tensor(float(theta), device=ref.device, dtype=torch.float32)) * (freq_idx / den)))")
        add(lines, indent * 2, "angles = pos * inv_freq.reshape(1, half)")
        add(lines, indent * 2, "emb = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)")
        add(lines, indent * 2, "if d % 2 != 0:")
        add(lines, indent * 3, "emb = torch.cat((emb, torch.zeros((emb.shape[0], 1), device=emb.device, dtype=emb.dtype)), dim=-1)")
        add(lines, indent * 2, "if padding_idx is not None:")
        add(lines, indent * 3, "mask = (position_ids.to(device=ref.device) == int(padding_idx)).reshape(-1, 1)")
        add(lines, indent * 3, "emb = torch.where(mask, torch.zeros_like(emb), emb)")
        add(lines, indent * 2, "return emb.to(dtype=ref.dtype)")
        add(lines, indent, "")
        add(lines, indent, "def _vllm_active_device(self):")
        add(lines, indent * 2, "device = getattr(self, '_active_device', None)")
        add(lines, indent * 2, "if device is not None:")
        add(lines, indent * 3, "return torch.device(device)")
        add(lines, indent * 2, "for param in self.parameters():")
        add(lines, indent * 3, "return param.device")
        add(lines, indent * 2, "return torch.device('cpu')")
        add(lines, indent, "")
        add(lines, indent, "def _vllm_active_dtype(self):")
        add(lines, indent * 2, "dtype = getattr(self, '_active_dtype', None)")
        add(lines, indent * 2, "if dtype is not None:")
        add(lines, indent * 3, "return dtype")
        add(lines, indent * 2, "for param in self.parameters():")
        add(lines, indent * 3, "if param.is_floating_point():")
        add(lines, indent * 4, "return param.dtype")
        add(lines, indent * 2, "return None")
        add(lines, indent, "")
        add(lines, indent, "@staticmethod")
        add(lines, indent, "def _path_template_part(value):")
        add(lines, indent * 2, "if isinstance(value, str) and value.startswith('@@'):")
        add(lines, indent * 3, "return value[2:].strip('.')")
        add(lines, indent * 2, "if isinstance(value, str) and value.startswith('@'):")
        add(lines, indent * 3, "return value[1:].strip('.')")
        add(lines, indent * 2, "return value")
        add(lines, indent, "")
        add(lines, indent, "@classmethod")
        add(lines, indent, "def _move_to(cls, value, device):")
        add(lines, indent * 2, "if torch.is_tensor(value):")
        add(lines, indent * 3, "return value.to(device=device)")
        add(lines, indent * 2, "if isinstance(value, tuple):")
        add(lines, indent * 3, "return tuple(cls._move_to(item, device) for item in value)")
        add(lines, indent * 2, "if isinstance(value, list):")
        add(lines, indent * 3, "return [cls._move_to(item, device) for item in value]")
        add(lines, indent * 2, "return value")
        add(lines, indent, "")
        add(lines, indent, "def _scatter(self, x, index, src, dim):")
        add(lines, indent * 2, "dim = int(dim)")
        add(lines, indent * 2, "index = self._move_to(index, x.device)")
        add(lines, indent * 2, "if torch.is_tensor(src):")
        add(lines, indent * 3, "src = self._move_to(src, x.device)")
        add(lines, indent * 3, "return torch.scatter(x, dim=dim, index=index, src=src)")
        add(lines, indent * 2, "return torch.scatter(x, dim=dim, index=index, value=src)")
        add(lines, indent, "")
        add(lines, indent, "def _vllm_state_tensor(self, key, value):")
        add(lines, indent * 2, "if not torch.is_tensor(value):")
        add(lines, indent * 3, "return value")
        add(lines, indent * 2, "device = self._vllm_active_device()")
        add(lines, indent * 2, "dtype = self._vllm_active_dtype() if value.is_floating_point() else None")
        add(lines, indent * 2, "if value.device == device and (dtype is None or value.dtype == dtype):")
        add(lines, indent * 3, "return value")
        add(lines, indent * 2, "moved = value.to(device=device, dtype=dtype) if dtype is not None else value.to(device=device)")
        add(lines, indent * 2, "self.state_dict_tensors[key] = moved")
        add(lines, indent * 2, "return moved")
        if self._use_clean_forward and _graph_uses_selected_expert_moe(self.program):
            add(lines, indent, "")
            self._emit_clean_selected_expert_helpers(lines, indent)
        if not self._use_clean_forward:
            add(lines, indent, "")
            add(lines, indent, "def _config(self, path, default=None):")
            add(lines, indent * 2, "sentinel = object()")
            add(lines, indent * 2, "value = _common_config_value(self.config, path, sentinel)")
            add(lines, indent * 2, "if value is not sentinel:")
            add(lines, indent * 3, "return value")
            add(lines, indent * 2, "return _common_config_value(self._model_config, path, default)")
            add(lines, indent, "")
            add(lines, indent, "def _has_config(self, path):")
            add(lines, indent * 2, "return _common_has_config_value(self.config, path) or _common_has_config_value(self._model_config, path)")
            add(lines, indent, "")
            self._emit_runtime_helpers(lines)
            add(lines, indent, "")
            add(lines, indent, "def _linear_param(self, path, expert, *, optional=False, field='linear.weight'):")
            add(lines, indent * 2, "resolved = path[2:] if isinstance(path, str) and path.startswith('@@') else path")
            add(lines, indent * 2, "value = self.state_dict_tensors.get(resolved)")
            add(lines, indent * 2, "if torch.is_tensor(value):")
            add(lines, indent * 3, "return self._vllm_state_tensor(resolved, value), expert")
            add(lines, indent * 2, "self._materialize_expert_bank_for_path(self.state_dict_tensors, resolved)")
            add(lines, indent * 2, "value = self.state_dict_tensors.get(resolved)")
            add(lines, indent * 2, "if torch.is_tensor(value):")
            add(lines, indent * 3, "return self._vllm_state_tensor(resolved, value), expert")
            add(lines, indent * 2, "bank = self._expert_bank_lookup(self.state_dict_tensors, resolved)")
            add(lines, indent * 2, "if bank is not None:")
            add(lines, indent * 3, "bank_key, path_expert = bank")
            add(lines, indent * 3, "bank_value = self.state_dict_tensors.get(bank_key)")
            add(lines, indent * 3, "if torch.is_tensor(bank_value):")
            add(lines, indent * 4, "bank_value = self._vllm_state_tensor(bank_key, bank_value)")
            add(lines, indent * 4, "if expert is None or int(expert) == path_expert:")
            add(lines, indent * 5, "return bank_value[path_expert], None")
            add(lines, indent * 4, "return bank_value, expert")
            add(lines, indent * 2, "if optional:")
            add(lines, indent * 3, "return None, expert")
            add(lines, indent * 2, "return self._vllm_state_tensor(resolved, _common_required_state_value(self.state_dict_tensors, resolved)), expert")
        add(lines, indent, "")

    def _emit_clean_selected_expert_helpers(self, lines: list[str], indent: int) -> None:
        add = self._add
        add(lines, indent, "@staticmethod")
        add(lines, indent, "def _collapsed_numeric_segments(key):")
        add(lines, indent * 2, "parts = str(key).split('.')")
        add(lines, indent * 2, "return [('.'.join(parts[:index] + parts[index + 1:]), int(part), index) for index, part in enumerate(parts) if part.isdigit()]")
        add(lines, indent, "")
        add(lines, indent, "@classmethod")
        add(lines, indent, "def _keys_for_collapsed_bank(cls, state, bank_key):")
        add(lines, indent * 2, "items = {}")
        add(lines, indent * 2, "numeric_index = None")
        add(lines, indent * 2, "for key in state:")
        add(lines, indent * 3, "for collapsed_key, expert, index in cls._collapsed_numeric_segments(str(key)):")
        add(lines, indent * 4, "if collapsed_key != bank_key:")
        add(lines, indent * 5, "continue")
        add(lines, indent * 4, "if numeric_index is None:")
        add(lines, indent * 5, "numeric_index = index")
        add(lines, indent * 4, "elif numeric_index != index:")
        add(lines, indent * 5, "continue")
        add(lines, indent * 4, "items[expert] = str(key)")
        add(lines, indent * 4, "break")
        add(lines, indent * 2, "if not items:")
        add(lines, indent * 3, "return []")
        add(lines, indent * 2, "ordered = [items[i] for i in range(len(items)) if i in items]")
        add(lines, indent * 2, "return ordered if len(ordered) == len(items) else []")
        add(lines, indent, "")
        add(lines, indent, "@staticmethod")
        add(lines, indent, "def _fused_gate_up_source_bank_keys(bank_key):")
        add(lines, indent * 2, "parts = str(bank_key).split('.')")
        add(lines, indent * 2, "for index, part in enumerate(parts):")
        add(lines, indent * 3, "if 'gate_up' not in part:")
        add(lines, indent * 4, "continue")
        add(lines, indent * 3, "gate_parts = list(parts)")
        add(lines, indent * 3, "up_parts = list(parts)")
        add(lines, indent * 3, "gate_parts[index] = part.replace('gate_up', 'gate', 1)")
        add(lines, indent * 3, "up_parts[index] = part.replace('gate_up', 'up', 1)")
        add(lines, indent * 3, "return '.'.join(gate_parts), '.'.join(up_parts)")
        add(lines, indent * 2, "return None")
        add(lines, indent, "")
        add(lines, indent, "@classmethod")
        add(lines, indent, "def _materialize_expert_bank_for_path(cls, state, bank_key):")
        add(lines, indent * 2, "existing = state.get(bank_key)")
        add(lines, indent * 2, "if torch.is_tensor(existing):")
        add(lines, indent * 3, "return existing")
        add(lines, indent * 2, "ordered_keys = cls._keys_for_collapsed_bank(state, bank_key)")
        add(lines, indent * 2, "if ordered_keys:")
        add(lines, indent * 3, "first = state[ordered_keys[0]]")
        add(lines, indent * 3, "first_shape = tuple(first.shape)")
        add(lines, indent * 3, "if all(torch.is_tensor(state[key]) and tuple(state[key].shape) == first_shape for key in ordered_keys):")
        add(lines, indent * 4, "return _materialize_joined_parameter(state, bank_key, ordered_keys, dim=0, mode='stack', remove_inputs=True)")
        add(lines, indent * 2, "fused_sources = cls._fused_gate_up_source_bank_keys(bank_key)")
        add(lines, indent * 2, "if fused_sources is None:")
        add(lines, indent * 3, "return None")
        add(lines, indent * 2, "gate_key, up_key = fused_sources")
        add(lines, indent * 2, "gate = cls._materialize_expert_bank_for_path(state, gate_key)")
        add(lines, indent * 2, "up = cls._materialize_expert_bank_for_path(state, up_key)")
        add(lines, indent * 2, "if not torch.is_tensor(gate) or not torch.is_tensor(up):")
        add(lines, indent * 3, "return None")
        add(lines, indent * 2, "if gate.shape[:-2] != up.shape[:-2] or gate.shape[-1:] != up.shape[-1:]:")
        add(lines, indent * 3, "return None")
        add(lines, indent * 2, "concat_dim = -2 if gate.ndim >= 2 else -1")
        add(lines, indent * 2, "return _materialize_joined_parameter(state, bank_key, (gate_key, up_key), dim=concat_dim, mode='cat', remove_inputs=True)")
        add(lines, indent, "")
        add(lines, indent, "def _required_param(self, path, field='parameter'):")
        add(lines, indent * 2, "resolved = str(path).lstrip('@')")
        add(lines, indent * 2, "self._materialize_expert_bank_for_path(self.state_dict_tensors, resolved)")
        add(lines, indent * 2, "value = self.state_dict_tensors.get(resolved)")
        add(lines, indent * 2, "if torch.is_tensor(value):")
        add(lines, indent * 3, "return self._vllm_state_tensor(resolved, value)")
        add(lines, indent * 2, "raise KeyError(f'missing {field} {resolved!r}')")
        add(lines, indent, "")
        add(lines, indent, "def _expert_linear_weight(self, x, expert_idx, weight_path, bias_value=None, transpose=False):")
        add(lines, indent * 2, "weight = self._required_param(str(weight_path), field='expert weight')")
        add(lines, indent * 2, "bias_value = bias_value.to(device=weight.device) if torch.is_tensor(bias_value) else bias_value")
        add(lines, indent * 2, "return _grouped_expert_linear_torch(x, weight, expert_idx, bias_value, transpose=transpose)")
        add(lines, indent, "")
        add(lines, indent, "def _expert_packed_swiglu_ffn(self, x, expert_idx, gate_up_weight_path, down_weight_path, transpose=False):")
        add(lines, indent * 2, "gate_up = self._expert_linear_weight(x, expert_idx, gate_up_weight_path, transpose=transpose)")
        add(lines, indent * 2, "gate, up = torch.chunk(gate_up, 2, dim=-1)")
        add(lines, indent * 2, "hidden = F.silu(gate) * up")
        add(lines, indent * 2, "return self._expert_linear_weight(hidden, expert_idx, down_weight_path, transpose=transpose)")
        add(lines, indent, "")
        add(lines, indent, "def _selected_expert_packed_swiglu_ffn(self, x, topk_scores, topk_indices, gate_up_weight_path, down_weight_path, transpose=False):")
        add(lines, indent * 2, "topk_indices = topk_indices.long()")
        add(lines, indent * 2, "expanded = torch.unsqueeze(x, -2).expand((*topk_indices.shape, x.shape[-1]))")
        add(lines, indent * 2, "values = self._expert_packed_swiglu_ffn(expanded, topk_indices, gate_up_weight_path, down_weight_path, transpose=transpose)")
        add(lines, indent * 2, "weights = torch.unsqueeze(topk_scores.to(device=values.device, dtype=values.dtype), -1)")
        add(lines, indent * 2, "return torch.sum(values * weights, dim=-2, keepdim=False)")
        add(lines, indent, "")
        add(lines, indent, "def _selected_expert_swiglu_ffn(self, x, topk_scores, topk_indices, gate_weight_path, up_weight_path, down_weight_path, transpose=False):")
        add(lines, indent * 2, "topk_indices = topk_indices.long()")
        add(lines, indent * 2, "expanded = torch.unsqueeze(x, -2).expand((*topk_indices.shape, x.shape[-1]))")
        add(lines, indent * 2, "gate = self._expert_linear_weight(expanded, topk_indices, gate_weight_path, transpose=transpose)")
        add(lines, indent * 2, "up = self._expert_linear_weight(expanded, topk_indices, up_weight_path, transpose=transpose)")
        add(lines, indent * 2, "hidden = F.silu(gate) * up")
        add(lines, indent * 2, "values = self._expert_linear_weight(hidden, topk_indices, down_weight_path, transpose=transpose)")
        add(lines, indent * 2, "weights = torch.unsqueeze(topk_scores.to(device=values.device, dtype=values.dtype), -1)")
        add(lines, indent * 2, "return torch.sum(values * weights, dim=-2, keepdim=False)")
        add(lines, indent, "")
        add(lines, indent, "def _selected_expert_packed_gegelu_ffn(self, x, topk_scores, topk_indices, gate_up_weight_path, gate_up_bias_path, down_weight_path, down_bias_path, limit, alpha=1.702, bias=False, transpose=False):")
        add(lines, indent * 2, "topk_indices = topk_indices.long()")
        add(lines, indent * 2, "expanded = torch.unsqueeze(x, -2).expand((*topk_indices.shape, x.shape[-1]))")
        add(lines, indent * 2, "gate_up_bias = self._required_param(str(gate_up_bias_path), field='gate_up bias') if bias else None")
        add(lines, indent * 2, "gate_up = self._expert_linear_weight(expanded, topk_indices, gate_up_weight_path, bias_value=gate_up_bias, transpose=transpose)")
        add(lines, indent * 2, "hidden = self._gegelu(gate_up, limit, alpha)")
        add(lines, indent * 2, "down_bias = self._required_param(str(down_bias_path), field='down bias') if bias else None")
        add(lines, indent * 2, "values = self._expert_linear_weight(hidden, topk_indices, down_weight_path, bias_value=down_bias, transpose=transpose)")
        add(lines, indent * 2, "weights = torch.unsqueeze(topk_scores.to(device=values.device, dtype=values.dtype), -1)")
        add(lines, indent * 2, "return torch.sum(values * weights, dim=-2, keepdim=False)")
        add(lines, indent, "")
        add(lines, indent, "def _selected_expert_clamped_packed_swiglu_ffn(self, x, topk_scores, topk_indices, gate_up_weight_path, down_weight_path, limit, transpose=False):")
        add(lines, indent * 2, "topk_indices = topk_indices.long()")
        add(lines, indent * 2, "expanded = torch.unsqueeze(x, -2).expand((*topk_indices.shape, x.shape[-1]))")
        add(lines, indent * 2, "gate_up = self._expert_linear_weight(expanded, topk_indices, gate_up_weight_path, transpose=transpose)")
        add(lines, indent * 2, "gate, up = torch.chunk(gate_up, 2, dim=-1)")
        add(lines, indent * 2, "limit = float(limit)")
        add(lines, indent * 2, "gate = torch.where(torch.isinf(gate), gate, gate.clamp(max=limit))")
        add(lines, indent * 2, "up = torch.where(torch.isinf(up), up, up.clamp(min=-limit, max=limit))")
        add(lines, indent * 2, "hidden = F.silu(gate) * up")
        add(lines, indent * 2, "values = self._expert_linear_weight(hidden, topk_indices, down_weight_path, transpose=transpose)")
        add(lines, indent * 2, "weights = torch.unsqueeze(topk_scores.to(device=values.device, dtype=values.dtype), -1)")
        add(lines, indent * 2, "return torch.sum(values * weights, dim=-2, keepdim=False)")
        add(lines, indent, "")
        add(lines, indent, "def _selected_expert_relu2_ffn(self, x, topk_scores, topk_indices, up_weight_path, down_weight_path, transpose=False):")
        add(lines, indent * 2, "topk_indices = topk_indices.long()")
        add(lines, indent * 2, "expanded = torch.unsqueeze(x, -2).expand((*topk_indices.shape, x.shape[-1]))")
        add(lines, indent * 2, "up = self._expert_linear_weight(expanded, topk_indices, up_weight_path, transpose=transpose)")
        add(lines, indent * 2, "hidden = F.relu(up) * F.relu(up)")
        add(lines, indent * 2, "values = self._expert_linear_weight(hidden, topk_indices, down_weight_path, transpose=transpose)")
        add(lines, indent * 2, "weights = torch.unsqueeze(topk_scores.to(device=values.device, dtype=values.dtype), -1)")
        add(lines, indent * 2, "return torch.sum(values * weights, dim=-2, keepdim=False)")
        add(lines, indent, "")

    def _emit_forward(self, lines: list[str]) -> None:
        add = self._add
        cls = self._vllm_classification

        repeated_mod_name = self._clean_forward_repeated_module_name(cls)
        repeated_mod = self.modules_by_name.get(repeated_mod_name) if repeated_mod_name else None

        if cls.mamba_mixer_module_names:
            self._emit_forward_ssm(lines, cls)
            return

        if repeated_mod is None:
            self._emit_forward_legacy(lines)
            return

        # --- Analyze per-layer structure ---
        layer_norms = self._analyze_layer_norms(repeated_mod, cls)
        if not layer_norms:
            self._emit_forward_legacy(lines)
            return

        qkv_group = cls.qkv_groups[0] if cls.qkv_groups else None
        ffn_group = cls.ffn_groups[0] if cls.ffn_groups else None
        o_proj_id = self._find_o_proj_id(repeated_mod, cls)
        rope_id = self._detect_rope(repeated_mod)
        act_name = self._detect_activation(repeated_mod)

        # Find non-repeated norms in source order. Prefix norms before the
        # repeated block (for example BLOOM's embedding layernorm) must run
        # before the layer loop; the last one is the final norm. Some model
        # graphs wrap the repeated block in a model helper, so this cannot be
        # restricted to the top-level main module.
        main_norm_nodes: list[GraphNode] = []
        for module in self.program.modules:
            if module.name == repeated_mod_name:
                continue
            for node in module.nodes:
                layer_type = cls.node_types.get(node.id)
                if (
                    layer_type in (VLLMLayerType.RMSNORM, VLLMLayerType.LAYERNORM)
                    and not self._is_vllm_repeated_layer_node(node)
                ):
                    main_norm_nodes.append(node)
        pre_loop_norm_attrs = [
            self._vllm_attr_access(node)
            for node in main_norm_nodes[:-1]
        ]
        final_norm_attr = None
        if main_norm_nodes:
            final_norm_attr = self._vllm_attr_access(main_norm_nodes[-1])

        # --- Config expressions ---
        num_layers_expr = self._config_expr("num_hidden_layers")
        num_heads_expr = self._config_expr("num_attention_heads")
        num_kv_heads_expr = self._config_expr("num_key_value_heads", alt="num_attention_heads")
        head_dim_expr = self._head_dim_expr()

        # --- Generate forward ---
        add(lines, 4, "def forward(")
        add(lines, 8, "self,")
        add(lines, 8, "input_ids: torch.Tensor | None = None,")
        add(lines, 8, "positions: torch.Tensor | None = None,")
        add(lines, 8, "intermediate_tensors=None,")
        add(lines, 8, "inputs_embeds: torch.Tensor | None = None,")
        add(lines, 8, "**kwargs,")
        add(lines, 8, "):")
        add(lines, 8, "if input_ids is not None and input_ids.dim() > 1:")
        add(lines, 12, "input_ids = input_ids.reshape(-1)")
        add(lines, 8, "if positions is not None and positions.dim() > 1:")
        add(lines, 12, "positions = positions.reshape(-1)")
        add(lines, 8, "if inputs_embeds is not None:")
        add(lines, 12, "hidden_states = inputs_embeds")
        add(lines, 12, "if hidden_states.dim() > 2:")
        add(lines, 16, "hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])")
        add(lines, 8, "else:")
        add(lines, 12, "hidden_states = self.embed_input_ids(input_ids, positions)")
        add(lines, 8, "if positions is None:")
        add(lines, 12, "positions = torch.arange(hidden_states.shape[0], device=hidden_states.device, dtype=torch.long)")
        for attr in pre_loop_norm_attrs:
            add(lines, 8, f"hidden_states = {attr}(hidden_states)")
        add(lines, 8, "self._debug_tensor_stats('forward.after_embed', hidden_states)")
        add(lines, 8, "")
        add(lines, 8, "if hidden_states is not None:")
        add(lines, 12, "self._active_device = hidden_states.device")
        add(lines, 12, "self._active_dtype = hidden_states.dtype")
        add(lines, 8, "elif input_ids is not None:")
        add(lines, 12, "self._active_device = input_ids.device")
        add(lines, 8, "else:")
        add(lines, 12, "self._active_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        add(lines, 8, "if hidden_states is None:")
        add(lines, 12, "self._active_dtype = next((p.dtype for p in self.parameters() if p.is_floating_point()), torch.float32)")
        add(lines, 8, "")
        add(lines, 8, "config = self.config")
        add(lines, 8, "_tp_size = self._tp_size")
        add(lines, 8, f"_num_heads = {num_heads_expr} // _tp_size")
        add(lines, 8, f"_num_kv_heads = max(1, {num_kv_heads_expr} // _tp_size)")
        add(lines, 8, f"_head_dim = {head_dim_expr}")
        add(lines, 8, f"_num_layers = {num_layers_expr}")
        self._emit_clean_forward_shape_bindings(lines, 8, repeated_mod)

        # Detect per-layer head dim variation
        hd_expr = None
        if repeated_mod is not None:
            hd_expr = self._detect_head_dim_expr(repeated_mod, "i")

        # Detect per-layer KV head count variation
        kv_heads_expr = None
        if repeated_mod is not None:
            kv_heads_expr = self._detect_kv_heads_expr(repeated_mod, "i")

        # Only emit pre-loop q/kv sizes when there's no per-layer variation
        if not hd_expr and not kv_heads_expr:
            add(lines, 8, f"_q_size = _num_heads * _head_dim")
            add(lines, 8, f"_kv_size = _num_kv_heads * _head_dim")

        has_per_layer_scalar = cls.per_layer_scalar_node_id is not None
        has_pli = cls.pli_gate_node_id is not None
        any_fused = any(f for _, _, _, f in layer_norms) and not has_pli

        # Precompute loop-invariant getattr(config, ...) calls found in
        # per-layer head_dim / kv_heads expressions so they are not
        # re-evaluated every iteration.
        _cfg_cache: dict[str, str] = {}

        def _precompute_getattr(expr: str) -> str:
            import re as _re
            pattern = r"getattr\(config,\s*'([^']+)',\s*([^)]+)\)"
            def _replace(m: _re.Match) -> str:
                field = m.group(1)
                default = m.group(2).strip()
                key = f"getattr(config, '{field}', {default})"
                if key not in _cfg_cache:
                    _cfg_cache[key] = f"_cfg_{field}"
                return _cfg_cache[key]
            return _re.sub(pattern, _replace, expr)

        if hd_expr:
            hd_expr = _precompute_getattr(hd_expr)
        if kv_heads_expr:
            kv_heads_expr = _precompute_getattr(kv_heads_expr)
        for orig, var in _cfg_cache.items():
            add(lines, 8, f"{var} = {orig}")

        # PLI computation (before the loop)
        if has_pli:
            pli_embed_node = self._find_node_by_id(cls.pli_embed_node_id) if cls.pli_embed_node_id else None
            pli_model_proj_node = self._find_node_by_id(cls.pli_model_proj_node_id) if cls.pli_model_proj_node_id else None
            pli_proj_norm_node = self._find_node_by_id(cls.pli_proj_norm_node_id) if cls.pli_proj_norm_node_id else None
            if pli_embed_node and pli_model_proj_node and pli_proj_norm_node:
                pli_embed_attr = self._vllm_attr_access(pli_embed_node)
                pli_model_proj_attr = self._vllm_attr_access(pli_model_proj_node)
                pli_proj_norm_attr = self._vllm_attr_access(pli_proj_norm_node)
                pli_dim = self._resolve_const_value("PLI") or self._config_expr("per_layer_input_dim", default=256)
                add(lines, 8, f"_pli_dim = {pli_dim}")
                add(lines, 8, f"_per_layer_inputs = {pli_embed_attr}(input_ids)")
                add(lines, 8, f"_per_layer_inputs = _per_layer_inputs * self._embed_scale_per_layer")
                add(lines, 8, f"_per_layer_inputs = _per_layer_inputs.reshape(-1, _num_layers, _pli_dim)")
                add(lines, 8, f"_pli_proj = {pli_model_proj_attr}(hidden_states)[0]")
                add(lines, 8, f"_pli_proj = _pli_proj * self._per_layer_projection_scale")
                add(lines, 8, f"_pli_proj = _pli_proj.reshape(-1, _num_layers, _pli_dim)")
                add(lines, 8, f"_pli_proj = {pli_proj_norm_attr}(_pli_proj)")
                add(lines, 8, f"_per_layer_inputs = (_pli_proj + _per_layer_inputs) * self._per_layer_input_scale")
                add(lines, 8, "")

        add(lines, 8, f"residual = None")
        add(lines, 8, f"_first_kv_shared = {self._config_expr('num_hidden_layers')} - {self._config_expr('num_kv_shared_layers')}")
        add(lines, 8, f"for i in range(_num_layers):")

        indent = 12

        # KV sharing: determine if this layer shares KV with an earlier layer
        add(lines, indent, "_is_kv_shared = (i >= _first_kv_shared > 0)")

        if hd_expr or kv_heads_expr:
            if kv_heads_expr:
                add(lines, indent, f"_num_kv_heads = max(1, {kv_heads_expr} // _tp_size)")
            if hd_expr:
                add(lines, indent, f"_hd = {hd_expr}")
                add(lines, indent, f"_q_size = _num_heads * _hd")
                add(lines, indent, f"_kv_size = _num_kv_heads * _hd")
            elif kv_heads_expr:
                add(lines, indent, f"_kv_size = _num_kv_heads * _head_dim")

        # Determine which norm index is the pre-FFN norm
        ffn_norm_idx = len(layer_norms) - 1
        if ffn_group and ffn_group.gate_node_id:
            dataflow_ffn_norm_idx = self._ffn_norm_index_from_dataflow(
                repeated_mod,
                layer_norms,
                ffn_group,
            )
            if dataflow_ffn_norm_idx is not None:
                ffn_norm_idx = dataflow_ffn_norm_idx
            else:
                gate_idx = int(ffn_group.gate_node_id.rsplit(":", 1)[-1]) if ":" in ffn_group.gate_node_id else 0
                for i, (nid, _, _, _) in enumerate(layer_norms):
                    nidx = int(nid.rsplit(":", 1)[-1]) if ":" in nid else 0
                    if nidx < gate_idx:
                        ffn_norm_idx = i

        # Detect standalone residual adds: adds whose inputs don't come from
        # norms (e.g. o_proj output + residual in standard Llama/SmolLM2).
        # These are NOT handled by the fused norm optimization and must be
        # emitted explicitly after the blocks.
        has_standalone_attn_add = False
        has_standalone_ffn_add = False
        if repeated_mod is not None:
            # Build value→producer map for this module
            _v2n: dict[str, str] = {}
            for node in repeated_mod.nodes:
                for out in node.outputs:
                    if hasattr(out, "name"):
                        _v2n[out.name] = node.id
            norm_ids_set = {nid for nid, _, _, _ in layer_norms}
            for node in repeated_mod.nodes:
                if node.op.name != "core.binary.+":
                    continue
                from_norm = False
                for inp in node.inputs:
                    if hasattr(inp, "name") and inp.name in _v2n:
                        producer = _v2n.get(inp.name)
                        if producer in norm_ids_set:
                            from_norm = True
                            break
                if not from_norm:
                    add_idx = int(node.id.rsplit(":", 1)[-1]) if ":" in node.id else 0
                    # Only classify as standalone add if it's after o_proj
                    # (attention residual) or after down_proj (FFN residual)
                    if o_proj_id:
                        o_idx = int(o_proj_id.rsplit(":", 1)[-1]) if ":" in o_proj_id else 0
                        if add_idx > o_idx and not has_standalone_attn_add:
                            has_standalone_attn_add = True
                    ffn_output_id = None
                    if ffn_group:
                        ffn_output_id = (
                            ffn_group.down_node_id
                            or ffn_group.gate_up_intrinsic_node_id
                        )
                    if ffn_output_id:
                        ffn_output_idx = int(ffn_output_id.rsplit(":", 1)[-1]) if ":" in ffn_output_id else 0
                        if add_idx > ffn_output_idx and not has_standalone_ffn_add:
                            has_standalone_ffn_add = True
        if (
            not has_standalone_ffn_add
            and ffn_group
            and (ffn_group.down_node_id or ffn_group.gate_up_intrinsic_node_id)
        ):
            ffn_output_node = self._find_node_by_id(
                ffn_group.down_node_id or ffn_group.gate_up_intrinsic_node_id
            )
            ffn_output_name = (
                _value_name(ffn_output_node.outputs[0])
                if ffn_output_node is not None and ffn_output_node.outputs
                else None
            )
            if ffn_output_name is not None and self._module_output_select_contains_value(
                repeated_mod,
                ffn_output_name,
            ):
                has_standalone_ffn_add = True

        uses_parallel_ffn_input = self._uses_parallel_ffn_input(
            repeated_mod,
            cls,
            layer_norms,
            ffn_group,
            o_proj_id,
            has_standalone_attn_add=has_standalone_attn_add,
            has_standalone_ffn_add=has_standalone_ffn_add,
        )

        # Do NOT fuse add+norm: native vLLM always uses separate rms_norm + add.
        # fused_add_rms_norm returns a tuple which changes inductor graph splitting
        # and results in ~17% slower CUDA graph execution.

        # Determine if first norm should fuse residual for subsequent layers.
        # True when residual is carried across layers (Gemma2 pattern),
        # False when param_scale consumes the residual within the block (Gemma4 pattern).
        # Also False when PLI is present: PLI sits between the last norm and the
        # final norm, so the FFN residual add must be explicit (before PLI),
        # not fused into the final norm.
        # Never fuse first norm residual: native vLLM always uses separate
        # rms_norm + add, never passes residual to RMSNorm.
        first_norm_fused = False

        # When first_norm_fused, the last norm's add is fused into the final
        # norm (or the next layer's first norm).  Don't emit the explicit add.
        if first_norm_fused and layer_norms:
            _last = len(layer_norms) - 1
            if not layer_norms[_last][3]:  # not already fused
                _ln = list(layer_norms[_last])
                _ln[2] = False  # uses_res = False
                layer_norms[_last] = tuple(_ln)

        # Unified per-norm pattern: each norm uses its own fused flag.
        for norm_i, (nid, attr, uses_res, fused) in enumerate(layer_norms):
            if norm_i == 0:
                add(lines, indent, f"block_input = hidden_states")
                if first_norm_fused:
                    add(lines, indent, f"if residual is None:")
                    add(lines, indent + 4, f"residual = hidden_states")
                    add(lines, indent + 4, f"hidden_states = {attr}(hidden_states)")
                    add(lines, indent, f"else:")
                    add(lines, indent + 4, f"hidden_states, residual = {attr}(hidden_states, residual)")
                else:
                    add(lines, indent, f"residual = hidden_states")
                    add(lines, indent, f"hidden_states = {attr}(hidden_states)")
            elif fused and uses_res:
                add(lines, indent, f"hidden_states, residual = {attr}(hidden_states, residual)")
                add(lines, indent, f"hidden_states = hidden_states + residual")
                add(lines, indent, f"residual = hidden_states")
            elif fused:
                add(lines, indent, f"hidden_states, residual = {attr}(hidden_states, residual)")
            elif uses_res:
                add(lines, indent, f"hidden_states = {attr}(hidden_states)")
                add(lines, indent, f"hidden_states = hidden_states + residual")
                add(lines, indent, f"residual = hidden_states")
            else:
                norm_node = self._find_node_by_id(nid)
                selected_block_input_cond = None
                if norm_node is not None and len(norm_node.inputs) >= 2:
                    selected_block_input_cond = self._selects_block_input_vs_current_hidden(
                        norm_node.inputs[1],
                        repeated_mod=repeated_mod,
                    )
                if selected_block_input_cond is not None:
                    add(
                        lines,
                        indent,
                        f"hidden_states = {attr}((block_input if {selected_block_input_cond} else hidden_states))",
                    )
                else:
                    add(lines, indent, f"hidden_states = {attr}(hidden_states)")

            if norm_i == 0 and qkv_group:
                if uses_parallel_ffn_input:
                    add(lines, indent, f"_parallel_ffn_input = hidden_states")
                self._emit_attn_block(lines, indent, cls, qkv_group, rope_id, o_proj_id)
                if has_standalone_attn_add:
                    add(lines, indent, f"hidden_states = hidden_states + residual")
                    add(lines, indent, f"residual = hidden_states")
            if norm_i == ffn_norm_idx and ffn_group:
                if uses_parallel_ffn_input:
                    add(lines, indent, f"hidden_states = _parallel_ffn_input")
                current_hidden_value = None
                norm_node = self._find_node_by_id(nid)
                if norm_node is not None and norm_node.outputs:
                    current_hidden_value = getattr(norm_node.outputs[0], "name", None)
                self._emit_ffn_block(
                    lines,
                    indent,
                    cls,
                    ffn_group,
                    act_name,
                    current_hidden_value=current_hidden_value,
                )
                if has_standalone_ffn_add:
                    add(lines, indent, f"hidden_states = hidden_states + residual")

        if has_pli:
            self._emit_pli_block(lines, indent, cls)
        if has_per_layer_scalar:
            if cls.per_layer_scalar_has_residual_add:
                add(lines, indent, f"hidden_states = hidden_states + residual")
                add(lines, indent, f"hidden_states = hidden_states * self.layer_scalars[i]")
                add(lines, indent, f"residual = hidden_states")
            else:
                add(lines, indent, f"hidden_states = hidden_states * self.layer_scalars[i]")

        # Final norm
        if final_norm_attr:
            if first_norm_fused:
                add(lines, 8, f"hidden_states, _ = {final_norm_attr}(hidden_states, residual)")
            else:
                add(lines, 8, f"hidden_states = {final_norm_attr}(hidden_states)")
            add(lines, 8, "self._debug_tensor_stats('forward.after_final_norm', hidden_states)")

        add(lines, 8, "return hidden_states")

    def _emit_pli_block(
        self,
        lines: list[str],
        indent: int,
        cls: VLLMLayerClassification,
    ) -> None:
        """Emit per-layer input application code."""
        add = self._add
        pli_gate_node = self._find_node_by_id(cls.pli_gate_node_id) if cls.pli_gate_node_id else None
        pli_proj_node = self._find_node_by_id(cls.pli_proj_node_id) if cls.pli_proj_node_id else None
        pli_norm_node = self._find_node_by_id(cls.pli_norm_node_id) if cls.pli_norm_node_id else None
        if not (pli_gate_node and pli_proj_node and pli_norm_node):
            return
        gate_attr = self._vllm_attr_access(pli_gate_node)
        proj_attr = self._vllm_attr_access(pli_proj_node)
        norm_attr = self._vllm_attr_access(pli_norm_node)
        add(lines, indent, f"_p_i = _per_layer_inputs[:, i, :]")
        add(lines, indent, f"_g = {gate_attr}(hidden_states)[0]")
        add(lines, indent, f"_g = F.gelu(_g, approximate='tanh')")
        add(lines, indent, f"_p = _g * _p_i")
        add(lines, indent, f"_p = {proj_attr}(_p)[0]")
        add(lines, indent, f"_p = {norm_attr}(_p)")
        add(lines, indent, f"hidden_states = hidden_states + _p")

    def _emit_clean_forward_shape_bindings(
        self,
        lines: list[str],
        indent: int,
        repeated_mod: GraphModule | None,
    ) -> None:
        """Bind repeated-module shape symbols for vLLM's flattened token API."""

        if repeated_mod is None:
            return
        hidden_input = next(
            (
                inp
                for inp in repeated_mod.inputs
                if isinstance(getattr(inp, "type_expr", None), TypeTensor)
            ),
            None,
        )
        if hidden_input is None:
            return
        dims = tuple(getattr(hidden_input.type_expr, "dims", ()) or ())
        if len(dims) < 2:
            return

        def bind_dim(dim: Any, expr: str) -> None:
            if not isinstance(dim, str) or dim in self.global_symbol_names:
                return
            self._add(lines, indent, f"{_dim_ident(dim)} = {expr}")

        # vLLM forwards flattened tokens: hidden_states is [total_tokens, D].
        # Any leading batch/group axes in the Axon block input are collapsed,
        # the penultimate axis is the flattened token count, and the final axis
        # remains the hidden width.
        for dim in dims[:-2]:
            bind_dim(dim, "1")
        bind_dim(dims[-2], "hidden_states.shape[0]")
        bind_dim(dims[-1], "hidden_states.shape[-1]")

    def _emit_attn_block(
        self,
        lines: list[str],
        indent: int,
        cls: VLLMLayerClassification,
        qkv_group: Any,
        rope_id: str | None,
        o_proj_id: str | None,
    ) -> None:
        add = self._add
        q_node = self._find_node_by_id(qkv_group.q_node_id)
        k_node = self._find_node_by_id(qkv_group.k_node_id)
        v_node = self._find_node_by_id(qkv_group.v_node_id)
        attn_node = self._find_node_by_id(qkv_group.attention_node_id) if qkv_group.attention_node_id else None

        q_attr = self._vllm_attr_access(q_node)
        has_var_hd = self._detect_head_dim_expr(self._get_repeated_module() or repeated_mod, "i") is not None if self._get_repeated_module() else False
        head_dim_ref = "_hd" if has_var_hd else "_head_dim"

        # Always use packed QKV: k/v graph nodes are skipped in init
        # (they're part of the single QKVParallelLinear)
        add(lines, indent, f"qkv, _ = {q_attr}(hidden_states)")
        add(lines, indent, f"q, k, v = qkv.split([_q_size, _kv_size, _kv_size], dim=-1)")

        q_norm_ids = sorted(cls.q_norm_node_ids)
        k_norm_ids = sorted(cls.k_norm_node_ids)
        unknown_qk_norm_ids = sorted(
            cls.qk_norm_node_ids - cls.q_norm_node_ids - cls.k_norm_node_ids
        )

        for nid in [*q_norm_ids, *unknown_qk_norm_ids]:
            node = self._find_node_by_id(nid)
            attr = self._vllm_attr_access(node)
            add(lines, indent, f"q = q.unflatten(-1, (_num_heads, {head_dim_ref}))")
            add(lines, indent, f"q = {attr}(q)")
            add(lines, indent, f"q = q.flatten(-2, -1)")

        # Match native vLLM ordering: K norm → RoPE → V norm (all inside one
        # if not _is_kv_shared block), with shared layers only applying RoPE to Q
        k_norm_attrs = [
            self._vllm_attr_access(self._find_node_by_id(nid))
            for nid in k_norm_ids
        ]
        v_norm_attr = None
        if cls.v_norm_node_ids:
            nid = sorted(cls.v_norm_node_ids)[0]
            node = self._find_node_by_id(nid)
            v_norm_attr = self._vllm_attr_access(node)

        if rope_id:
            add(lines, indent, f"if not _is_kv_shared:")
            for k_norm_attr in k_norm_attrs:
                add(lines, indent + 1, f"k = k.unflatten(-1, (_num_kv_heads, {head_dim_ref}))")
                add(lines, indent + 1, f"k = {k_norm_attr}(k)")
                add(lines, indent + 1, f"k = k.flatten(-2, -1)")
            add(lines, indent + 1, f"q, k = self.rotary_emb[i](positions, q, k)")
            if v_norm_attr:
                add(lines, indent + 1, f"v = v.unflatten(-1, (_num_kv_heads, {head_dim_ref}))")
                add(lines, indent + 1, f"v = {v_norm_attr}(v)")
                add(lines, indent + 1, f"v = v.flatten(-2, -1)")
            add(lines, indent, f"else:")
            add(lines, indent + 1, f"q = self.rotary_emb[i](positions, q, k)[0]")
        else:
            if k_norm_attrs:
                add(lines, indent, f"if not _is_kv_shared:")
            for k_norm_attr in k_norm_attrs:
                add(lines, indent + 1, f"k = k.unflatten(-1, (_num_kv_heads, {head_dim_ref}))")
                add(lines, indent + 1, f"k = {k_norm_attr}(k)")
                add(lines, indent + 1, f"k = k.flatten(-2, -1)")
            if v_norm_attr:
                if not k_norm_attrs:
                    add(lines, indent, f"if not _is_kv_shared:")
                add(lines, indent + 1, f"v = v.unflatten(-1, (_num_kv_heads, {head_dim_ref}))")
                add(lines, indent + 1, f"v = {v_norm_attr}(v)")
                add(lines, indent + 1, f"v = v.flatten(-2, -1)")

        # Attention
        if attn_node:
            attn_attr = self._vllm_attr_access(attn_node)
            add(lines, indent, "q = q.contiguous()")
            add(lines, indent, "k = k.contiguous()")
            add(lines, indent, "v = v.contiguous()")
            add(lines, indent, f"attn_out = {attn_attr}(q, k, v)")
        else:
            add(lines, indent, f"attn_out = v")

        # O projection
        if o_proj_id:
            o_node = self._find_node_by_id(o_proj_id)
            o_attr = self._vllm_attr_access(o_node)
            add(lines, indent, f"hidden_states = {o_attr}(attn_out)[0]")

    def _emit_ffn_block(
        self,
        lines: list[str],
        indent: int,
        cls: VLLMLayerClassification,
        ffn_group: Any,
        act_name: str | None,
        *,
        current_hidden_value: str | None = None,
    ) -> None:
        add = self._add

        if ffn_group.gate_up_intrinsic_node_id:
            node = self._find_node_by_id(ffn_group.gate_up_intrinsic_node_id)
            if node is None:
                return
            local = self._emit_selected_expert_dependencies(
                lines,
                indent,
                node,
                current_hidden_value=current_hidden_value,
            )
            expr = self._selected_expert_intrinsic_expr(node, local=local)
            add(lines, indent, f"hidden_states = {expr}")
            return

        if ffn_group.gate_node_id and ffn_group.up_node_id:
            gate_node = self._find_node_by_id(ffn_group.gate_node_id)
            gate_attr = self._vllm_attr_access(gate_node)
            # Use merged gate_up_proj + GeluAndMul for numerical parity with native vLLM
            add(lines, indent, f"gate_up, _ = {gate_attr}(hidden_states)")
            if act_name and act_name.endswith("gegelu"):
                add(lines, indent, f"hidden_states = self._gegelu(gate_up, {self._activation_arg_expr(self._get_repeated_module() or cls, input_index=1)})")
            elif act_name and act_name.endswith(("gelu_pytorch_tanh", "gelu_tanh")):
                add(lines, indent, f"hidden_states = self._ffn_act(gate_up)")
            elif act_name:
                act_code = self._activation_to_code(act_name)
                add(lines, indent, f"_gate, _up = gate_up.chunk(2, dim=-1)")
                add(lines, indent, f"hidden_states = {act_code.format(x='_gate')} * _up")
            else:
                add(lines, indent, f"_gate, _up = gate_up.chunk(2, dim=-1)")
                add(lines, indent, f"hidden_states = F.silu(_gate) * _up")
        elif ffn_group.up_node_id:
            up_node = self._find_node_by_id(ffn_group.up_node_id)
            up_attr = self._vllm_attr_access(up_node)
            add(lines, indent, f"hidden_states = {up_attr}(hidden_states)[0]")
            act_node = self._detect_activation_node(self._get_repeated_module() or cls)
            act_expr = self._activation_node_to_code(
                act_node,
                x="hidden_states",
                repeated_mod=self._get_repeated_module() or cls,
            )
            if act_expr is not None:
                add(lines, indent, f"hidden_states = {act_expr}")
            elif act_name:
                if act_name.endswith("gegelu"):
                    add(lines, indent, f"hidden_states = self._gegelu(hidden_states, {self._activation_arg_expr(self._get_repeated_module() or cls, input_index=1)})")
                else:
                    act_code = self._activation_to_code(act_name)
                    add(lines, indent, f"hidden_states = {act_code.format(x='hidden_states')}")
            elif ffn_group.down_node_id:
                down_node = self._find_node_by_id(ffn_group.down_node_id)
                repeated_mod = self._get_repeated_module() or cls
                if down_node is not None and len(down_node.inputs) >= 2:
                    act_expr = self._activation_expr_to_code(
                        down_node.inputs[1],
                        x="hidden_states",
                        repeated_mod=repeated_mod,
                    )
                    if act_expr is not None:
                        add(lines, indent, f"hidden_states = {act_expr}")

        if ffn_group.down_node_id:
            down_node = self._find_node_by_id(ffn_group.down_node_id)
            down_attr = self._vllm_attr_access(down_node)
            add(lines, indent, f"hidden_states = {down_attr}(hidden_states)[0]")

    def _emit_selected_expert_dependencies(
        self,
        lines: list[str],
        indent: int,
        node: GraphNode,
        *,
        current_hidden_value: str | None,
    ) -> dict[str, str]:
        """Emit the primitive provenance feeding a clean selected-expert call."""

        local = self._clean_forward_local_map(node)
        if current_hidden_value:
            local[current_hidden_value] = "hidden_states"
        if node.inputs and isinstance(node.inputs[0], (GraphValueRef, GraphValue)):
            local[node.inputs[0].name] = "hidden_states"
        module_name = self._node_module_name(node)
        module = self.modules_by_name.get(module_name) if module_name else None
        if module is None:
            return local
        producer: dict[str, GraphNode] = {}
        for candidate in module.nodes:
            for output in candidate.outputs:
                if getattr(output, "name", None):
                    producer[output.name] = candidate

        ordered: list[GraphNode] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit_operand(operand: GraphOperand) -> None:
            if isinstance(operand, (GraphValueRef, GraphValue)):
                name = operand.name
                if name in local:
                    return
                dep = producer.get(name)
                if dep is not None and dep is not node:
                    visit_node(dep)
                return
            if isinstance(operand, GraphExpr):
                for child in operand.inputs:
                    visit_operand(child)
                for child in operand.attrs.values():
                    visit_operand(child)

        def visit_node(dep: GraphNode) -> None:
            if dep.id in visited or dep.id in visiting:
                return
            if all(getattr(out, "name", None) in local for out in dep.outputs):
                visited.add(dep.id)
                return
            visiting.add(dep.id)
            for child in dep.inputs:
                visit_operand(child)
            for child in dep.attrs.values():
                visit_operand(child)
            visiting.remove(dep.id)
            visited.add(dep.id)
            ordered.append(dep)

        # Only the data, top-k weights, and top-k indices feed the clean
        # runtime operation. Path/literal operands are rendered directly.
        for operand in node.inputs[:3]:
            visit_operand(operand)

        for dep in ordered:
            targets = tuple(out.name for out in dep.outputs if getattr(out, "name", None))
            if not targets:
                continue
            if all(target in local for target in targets):
                continue
            if self._emit_clean_dependency_node(
                lines,
                indent,
                dep,
                targets=targets,
                local=local,
            ):
                continue
            expr = self._node_expr(dep, local=local, symbols_dict="self._symbols")
            self._add(lines, indent, f"{', '.join(targets)} = {expr}")
            for target in targets:
                local[target] = target
        return local

    def _emit_clean_dependency_node(
        self,
        lines: list[str],
        indent: int,
        node: GraphNode,
        *,
        targets: tuple[str, ...],
        local: dict[str, str],
    ) -> bool:
        layer_type = self._vllm_classification.node_types.get(
            node.id,
            VLLMLayerType.DEFAULT,
        )
        if layer_type != VLLMLayerType.DEFAULT:
            if self._emit_vllm_layer_call(
                lines,
                node,
                layer_type,
                targets=targets,
                indent=indent,
                local=local,  # type: ignore[arg-type]
                symbols_dict="self._symbols",
            ):
                for target in targets:
                    local[target] = target
                return True
        if node.op.name in self.modules_by_name:
            if self._emit_clean_inline_module_call(
                lines,
                indent,
                node,
                targets=targets,
                local=local,
            ):
                return True
        return False

    def _emit_clean_inline_module_call(
        self,
        lines: list[str],
        indent: int,
        node: GraphNode,
        *,
        targets: tuple[str, ...],
        local: dict[str, str],
    ) -> bool:
        callee = self.modules_by_name.get(node.op.name)
        if callee is None or len(callee.inputs) != len(node.inputs):
            return False
        call_local = dict(local)
        for formal, actual in zip(callee.inputs, node.inputs, strict=True):
            call_local[formal.name] = self._operand_expr(
                actual,
                local=local,
                symbols_dict="self._symbols",
            )

        safe_prefix = _py_ident(f"__clean_inline_{node.id.replace(':', '_')}")
        for index, inner in enumerate(callee.nodes, start=1):
            inner_targets = tuple(
                f"{safe_prefix}_{index}_{_py_ident(out.name)}"
                for out in inner.outputs
                if getattr(out, "name", None)
            )
            if not inner_targets:
                continue
            if self._emit_clean_dependency_node(
                lines,
                indent,
                inner,
                targets=inner_targets,
                local=call_local,
            ):
                for original, target in zip(inner.outputs, inner_targets, strict=True):
                    call_local[original.name] = target
                continue
            expr = self._node_expr(inner, local=call_local, symbols_dict="self._symbols")
            self._add(lines, indent, f"{', '.join(inner_targets)} = {expr}")
            for original, target in zip(inner.outputs, inner_targets, strict=True):
                call_local[original.name] = target

        output_exprs = [
            self._operand_expr(output, local=call_local, symbols_dict="self._symbols")
            for output in callee.outputs
        ]
        if len(targets) == 1:
            rhs = output_exprs[0] if len(output_exprs) == 1 else f"({', '.join(output_exprs)})"
        elif len(output_exprs) == len(targets):
            rhs = ", ".join(output_exprs)
        elif len(output_exprs) == 1:
            rhs = output_exprs[0]
        else:
            return False
        self._add(lines, indent, f"{', '.join(targets)} = {rhs}")
        for target in targets:
            local[target] = target
        return True

    def _selected_expert_intrinsic_expr(self, node: GraphNode, *, local: dict[str, str] | None = None) -> str:
        if local is None:
            local = self._clean_forward_local_map(node)
        args = self._collect_args(node, local)
        op = node.op.name
        if op == "__vllm_selected_expert_packed_swiglu_ffn":
            if len(args) < 6:
                raise ValueError("__vllm_selected_expert_packed_swiglu_ffn expects input, top-k scores/indices, gate-up/down weight paths, and transpose")
            return (
                "self._selected_expert_packed_swiglu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, "
                f"transpose={_bool_arg(node, 5)})"
            )
        if op == "__vllm_selected_expert_swiglu_ffn":
            if len(args) < 7:
                raise ValueError("__vllm_selected_expert_swiglu_ffn expects input, top-k scores/indices, gate/up/down weight paths, and transpose")
            return (
                "self._selected_expert_swiglu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, {args[5]}, "
                f"transpose={_bool_arg(node, 6)})"
            )
        if op == "__vllm_selected_expert_packed_gegelu_ffn":
            if len(args) < 10:
                raise ValueError("__vllm_selected_expert_packed_gegelu_ffn expects input, top-k scores/indices, gate-up/down weight/bias paths, limit, optional alpha, bias, and transpose")
            alpha_arg = args[8] if len(args) >= 11 else "1.702"
            bias_idx = 9 if len(args) >= 11 else 8
            transpose_idx = 10 if len(args) >= 11 else 9
            return (
                "self._selected_expert_packed_gegelu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, "
                f"{args[5]}, {args[6]}, {args[7]}, alpha={alpha_arg}, "
                f"bias={_bool_arg(node, bias_idx)}, transpose={_bool_arg(node, transpose_idx)})"
            )
        if op == "__vllm_selected_expert_clamped_packed_swiglu_ffn":
            if len(args) < 7:
                raise ValueError("__vllm_selected_expert_clamped_packed_swiglu_ffn expects input, top-k scores/indices, gate-up/down weight paths, limit, and transpose")
            return (
                "self._selected_expert_clamped_packed_swiglu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, {args[5]}, "
                f"transpose={_bool_arg(node, 6)})"
            )
        if op == "__vllm_selected_expert_relu2_ffn":
            if len(args) < 6:
                raise ValueError("__vllm_selected_expert_relu2_ffn expects input, top-k scores/indices, up/down weight paths, and transpose")
            return (
                "self._selected_expert_relu2_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, "
                f"transpose={_bool_arg(node, 5)})"
            )
        raise ValueError(f"unsupported selected-expert intrinsic {op!r}")

    def _clean_forward_local_map(self, node: GraphNode) -> dict[str, str]:
        local: dict[str, str] = {"i": "i"}
        mod_name = self._node_module_name(node)
        module = self.modules_by_name.get(mod_name)
        scope_parts = self._vllm_classification.module_scope_parts.get(mod_name)
        if module is None or scope_parts is None:
            return local
        loop_var = self._node_loop_index(node)
        fparts = self._format_template_parts(
            scope_parts,
            loop_var=loop_var,
            map_unknown_templates_to_loop=True,
        )
        scope_expr = f'f"{".".join(fparts)}"'
        for inp in module.inputs:
            if isinstance(inp.type_expr, TypePath):
                local[inp.name] = scope_expr
        for inp in module.inputs:
            if isinstance(inp.type_expr, TypeTensor) and inp.name not in local:
                local[inp.name] = "hidden_states"
                break
        return local

    def _emit_forward_ssm(self, lines: list[str], cls: VLLMLayerClassification) -> None:
        add = self._add
        num_layers_expr = self._config_expr("num_hidden_layers")

        # Find the pre-mixer norm: per-layer RMSNorm (repeated, not in mamba_mixer)
        norm_attr = None
        for nid in sorted(cls.rmsnorm_node_ids):
            node = self._find_node_by_id(nid)
            if node is None:
                continue
            mod_name = self._node_module_name(node)
            if mod_name in cls.mamba_mixer_module_names:
                continue
            is_repeated = self._is_repeated_node(node) or '{i}' in self._node_prefix(node)
            if is_repeated:
                norm_attr = self._vllm_attr_access(node)
                break

        # Find final norm: non-repeated RMSNorm/LayerNorm (not in any repeated module)
        final_norm_attr = None
        for nid, layer_type in sorted(cls.node_types.items()):
            if layer_type not in (VLLMLayerType.RMSNORM, VLLMLayerType.LAYERNORM):
                continue
            node = self._find_node_by_id(nid)
            if node is None:
                continue
            mod_name = self._node_module_name(node)
            if mod_name in cls.repeated_module_names:
                continue
            if mod_name in cls.mamba_mixer_module_names:
                continue
            is_repeated = self._is_repeated_node(node) or '{i}' in self._node_prefix(node)
            if not is_repeated:
                final_norm_attr = self._vllm_attr_access(node)
                break

        add(lines, 4, "def forward(")
        add(lines, 8, "self,")
        add(lines, 8, "input_ids: torch.Tensor | None = None,")
        add(lines, 8, "positions: torch.Tensor | None = None,")
        add(lines, 8, "intermediate_tensors=None,")
        add(lines, 8, "inputs_embeds: torch.Tensor | None = None,")
        add(lines, 8, "**kwargs,")
        add(lines, 8, "):")
        add(lines, 8, "if inputs_embeds is not None:")
        add(lines, 12, "hidden_states = inputs_embeds")
        add(lines, 8, "else:")
        add(lines, 12, "hidden_states = self.embed_input_ids(input_ids, positions)")
        add(lines, 8, "")
        add(lines, 8, "if hidden_states is not None:")
        add(lines, 12, "self._active_device = hidden_states.device")
        add(lines, 12, "self._active_dtype = hidden_states.dtype")
        add(lines, 8, "elif input_ids is not None:")
        add(lines, 12, "self._active_device = input_ids.device")
        add(lines, 8, "else:")
        add(lines, 12, "self._active_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        add(lines, 8, "if hidden_states is None:")
        add(lines, 12, "self._active_dtype = next((p.dtype for p in self.parameters() if p.is_floating_point()), torch.float32)")
        add(lines, 8, "")
        add(lines, 8, "config = self.config")
        add(lines, 8, f"_num_layers = {num_layers_expr}")
        add(lines, 8, "residual = None")
        add(lines, 8, "for i in range(_num_layers):")
        if norm_attr is not None:
            add(lines, 12, "if residual is not None:")
            add(lines, 16, f"hidden_states, residual = {norm_attr}[i](hidden_states, residual)")
            add(lines, 12, "else:")
            add(lines, 16, "residual = hidden_states")
            add(lines, 16, f"hidden_states = {norm_attr}[i](hidden_states)")
        add(lines, 12, "output = torch.empty_like(hidden_states)")
        add(lines, 12, "self._vllm_mamba_mixer[i](hidden_states, output)")
        add(lines, 12, "hidden_states = output")
        add(lines, 8, "")
        if final_norm_attr is not None:
            add(lines, 8, "if residual is not None:")
            add(lines, 12, "hidden_states = hidden_states + residual")
            add(lines, 8, f"hidden_states = {final_norm_attr}(hidden_states)")
            add(lines, 8, "")
        add(lines, 8, "return hidden_states")
        add(lines, 4, "")

    def _emit_forward_legacy(self, lines: list[str]) -> None:
        add = self._add
        main = self.modules_by_name[self.program.main_module]
        mask_inputs = [
            inp
            for inp in main.inputs[1:]
            if self._is_mask_input(getattr(inp, "type_expr", None))
        ]
        add(lines, 4, "def forward(")
        add(lines, 8, "self,")
        add(lines, 8, "input_ids: torch.Tensor | None = None,")
        add(lines, 8, "positions: torch.Tensor | None = None,")
        add(lines, 8, "intermediate_tensors=None,")
        add(lines, 8, "inputs_embeds: torch.Tensor | None = None,")
        add(lines, 8, "attn_metadata=None,")
        add(lines, 8, "):")
        add(lines, 8, "if inputs_embeds is not None:")
        add(lines, 12, "hidden_states = inputs_embeds")
        add(lines, 8, "else:")
        add(lines, 12, "hidden_states = self.embed_input_ids(input_ids, positions)")
        add(lines, 8, "")
        add(lines, 8, "if hidden_states is not None:")
        add(lines, 12, "self._active_device = hidden_states.device")
        add(lines, 12, "self._active_dtype = hidden_states.dtype")
        add(lines, 8, "elif input_ids is not None:")
        add(lines, 12, "self._active_device = input_ids.device")
        add(lines, 8, "else:")
        add(lines, 12, "self._active_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        add(lines, 8, "if hidden_states is None:")
        add(lines, 12, "self._active_dtype = next((p.dtype for p in self.parameters() if p.is_floating_point()), torch.float32)")
        add(lines, 8, "")
        add(lines, 8, "self._attn_metadata = attn_metadata")
        add(lines, 8, "self._positions = positions")
        add(lines, 8, "def _mask_for_positions(_ids, _pos):")
        add(lines, 12, "if _ids is None:")
        add(lines, 16, "return None")
        add(lines, 12, "if _pos is None:")
        add(lines, 16, "return torch.ones_like(_ids, dtype=torch.long)")
        add(lines, 12, "_flat_pos = _pos.reshape(-1).to(device=_ids.device)")
        add(lines, 12, "if _flat_pos.numel() == 0:")
        add(lines, 16, "return torch.ones_like(_ids, dtype=torch.long)")
        add(lines, 12, "_length = int(_flat_pos.max().item()) + 1")
        add(lines, 12, "_length = max(_length, int(_ids.shape[-1]))")
        add(lines, 12, "return torch.ones((_ids.shape[0], _length), dtype=torch.long, device=_ids.device)")
        add(lines, 8, "")
        add(lines, 8, "if input_ids is not None and input_ids.dim() == 1 and positions is not None and positions.dim() == 1:")
        add(lines, 12, "_pos = positions.detach()")
        add(lines, 12, "_reset = torch.nonzero(_pos[1:] < _pos[:-1], as_tuple=False).flatten() + 1")
        add(lines, 12, "if _reset.numel() > 0:")
        add(lines, 16, "_starts = [0, *[int(x) for x in _reset.cpu().tolist()]]")
        add(lines, 16, "_ends = [*_starts[1:], int(input_ids.shape[0])]")
        add(lines, 16, "_parts = []")
        add(lines, 16, "for _start, _end in zip(_starts, _ends):")
        add(lines, 20, "_seg = input_ids[_start:_end].unsqueeze(0)")
        if mask_inputs:
            add(lines, 20, "_seg_pos = positions[_start:_end]")
            add(lines, 20, "_seg_mask = _mask_for_positions(_seg, _seg_pos)")
            add(
                lines,
                20,
                "_result = "
                + f"self.{self.method_names[main.name]}("
                + ", ".join(
                    ["input_ids=_seg"]
                    + [f"{inp.name}=_seg_mask" for inp in mask_inputs]
                )
                + ")",
            )
        else:
            add(lines, 20, "_result = " + f"self.{self.method_names[main.name]}(input_ids=_seg)")
        add(lines, 20, "if isinstance(_result, (tuple, list)):")
        add(lines, 24, "_result = _result[0]")
        add(lines, 20, "if _result.dim() == 3:")
        add(lines, 24, "_result = _result.reshape(-1, _result.shape[-1])")
        add(lines, 20, "_parts.append(_result)")
        add(lines, 16, "return torch.cat(_parts, dim=0)")
        add(lines, 8, "if input_ids is not None and input_ids.dim() == 1:")
        add(lines, 12, "input_ids = input_ids.unsqueeze(0)")
        if mask_inputs:
            add(lines, 8, "_mask = _mask_for_positions(input_ids, positions)")
            add(
                lines,
                8,
                f"result = self.{self.method_names[main.name]}("
                + ", ".join(
                    ["input_ids=input_ids"]
                    + [f"{inp.name}=_mask" for inp in mask_inputs]
                )
                + ")",
            )
        else:
            add(lines, 8, f"result = self.{self.method_names[main.name]}(input_ids=input_ids)")
        add(lines, 8, "if isinstance(result, (tuple, list)):")
        add(lines, 12, "result = result[0]")
        add(lines, 8, "if result.dim() == 3:")
        add(lines, 12, "result = result.reshape(-1, result.shape[-1])")
        add(lines, 8, "return result")

    def _emit_vllm_layer_inits(
        self,
        lines: list[str],
        classification: VLLMLayerClassification,
        indent: int,
    ) -> None:
        add = self._add
        seen = set()
        self._vllm_emitted_layer_node_ids = set()
        self._vllm_attention_prefix_counts: dict[str, int] = {}
        self._vllm_attention_prefix_suffixes: dict[str, str] = {}
        num_layers_expr = self._config_expr("num_hidden_layers")
        skip_node_ids: set[str] = set()
        use_clean_forward = False
        _repeated_mod_name = self._clean_forward_repeated_module_name(classification)
        if _repeated_mod_name is not None:
            _repeated_mod = self.modules_by_name.get(_repeated_mod_name)
            if _repeated_mod is not None:
                _layer_norms = self._analyze_layer_norms(_repeated_mod, classification)
                if _layer_norms and self._clean_forward_unsupported_reason(classification) is None:
                    use_clean_forward = True
        if classification.mamba_mixer_module_names:
            use_clean_forward = True
            for mamba_mod_name in classification.mamba_mixer_module_names:
                mamba_mod = self.modules_by_name.get(mamba_mod_name)
                if mamba_mod is not None:
                    for node in mamba_mod.nodes:
                        skip_node_ids.add(node.id)
        self._use_clean_forward = use_clean_forward
        if use_clean_forward:
            for g in classification.qkv_groups:
                if g.q_node_id != g.k_node_id:
                    skip_node_ids.add(g.k_node_id)
                    skip_node_ids.add(g.v_node_id)
                    _q_mod = g.q_node_id.split(":", 1)[0] if ":" in g.q_node_id else ""
                    for mod_name in [g.k_node_id, g.v_node_id]:
                        top = mod_name.split(":", 1)[0] if ":" in mod_name else ""
                        if top and top in self.modules_by_name and top != _q_mod:
                            kv_mod = self.modules_by_name[top]
                            for kn in kv_mod.nodes:
                                if _is_linear_call(kn, self.modules_by_name):
                                    if kn.id not in (g.q_node_id, g.k_node_id, g.v_node_id):
                                        skip_node_ids.add(kn.id)
                                if classification.node_types.get(kn.id) == VLLMLayerType.RMSNORM:
                                    if kn.id not in classification.qk_norm_node_ids and kn.id not in classification.v_norm_node_ids:
                                        skip_node_ids.add(kn.id)
        # Track modules that need prefix set after creation (don't accept prefix in __init__)
        prefix_setters: list[tuple[str, str, bool]] = []
        # Build map of FFN gate→up pairs for merged gate_up_proj emission.
        # When a gate node has a matching up node, emit a single
        # MergedColumnParallelLinear and skip the up node.
        ffn_up_to_skip: set[str] = set()
        ffn_gate_to_up: dict[str, str] = {}
        ffn_gate_to_down: dict[str, str] = {}
        for g in classification.ffn_groups:
            if g.gate_node_id and g.up_node_id:
                if g.gate_node_id != g.up_node_id:
                    ffn_gate_to_up[g.gate_node_id] = g.up_node_id
                    ffn_up_to_skip.add(g.up_node_id)
                if g.down_node_id:
                    ffn_gate_to_down[g.gate_node_id] = g.down_node_id
        self._ffn_gate_to_up = ffn_gate_to_up
        self._ffn_gate_to_down = ffn_gate_to_down
        # Deduplicate v_norms: keep only one (they are identical RMSNorm has_weight=False).
        # Multiple v_norms arise from alternative KV modules (e.g. gemma4_kv_shared
        # vs gemma4_kv_separate) where only one path is taken at runtime.
        v_norm_ids = sorted(classification.v_norm_node_ids)
        if len(v_norm_ids) > 1:
            for nid in v_norm_ids[1:]:
                skip_node_ids.add(nid)
        if use_clean_forward:
            for module in self.program.modules:
                for node in module.nodes:
                    node_layer_type = classification.node_types.get(
                        node.id, VLLMLayerType.DEFAULT
                    )
                    if (
                        not self._is_repeated_node(node)
                        and node_layer_type
                        in (
                            VLLMLayerType.COLUMN_PARALLEL_LINEAR,
                            VLLMLayerType.ROW_PARALLEL_LINEAR,
                            VLLMLayerType.PARALLEL_LM_HEAD,
                        )
                        and len(node.inputs) >= 3
                        and isinstance(node.inputs[2], GraphLiteral)
                        and node.inputs[2].value is None
                    ):
                        skip_node_ids.add(node.id)
        # Precompute KV sharing targets: for layers in the last
        # num_kv_shared_layers, map to the earlier layer of the same
        # attention type whose KV cache they reuse.
        add(lines, indent, "_kv_sharing_targets = {}")
        add(lines, indent, f"_nkv = {self._config_expr('num_kv_shared_layers')}")
        add(lines, indent, "if _nkv > 0:")
        add(lines, indent + 1, f"_nl = {self._config_expr('num_hidden_layers')}")
        add(lines, indent + 1, "_first_shared = _nl - _nkv")
        add(lines, indent + 1, "_ltypes = getattr(config, 'layer_types', [])")
        add(lines, indent + 1, "for _i in range(_first_shared, _nl):")
        add(lines, indent + 2, "if _i < len(_ltypes):")
        add(lines, indent + 3, "_t = _ltypes[_i]")
        add(lines, indent + 3, "_candidates = [j for j in range(_first_shared) if j < len(_ltypes) and _ltypes[j] == _t]")
        add(lines, indent + 3, "if _candidates:")
        add(lines, indent + 4, "_kv_sharing_targets[_i] = _candidates[-1]")

        for module in self.program.modules:
            for node in module.nodes:
                if node.id in seen:
                    continue
                if node.id in skip_node_ids:
                    continue
                if node.id in ffn_up_to_skip:
                    continue
                if node.id in classification.position_embedding_node_ids:
                    continue
                layer_type = classification.layer_type(node)
                if layer_type == VLLMLayerType.DEFAULT:
                    continue
                if (
                    layer_type == VLLMLayerType.VOCAB_PARALLEL_EMBEDDING
                    and self._vllm_embedding_layer_prefix(node) is None
                ):
                    continue
                if (
                    not use_clean_forward
                    and layer_type
                    in (
                        VLLMLayerType.QKV_PARALLEL_LINEAR,
                        VLLMLayerType.COLUMN_PARALLEL_LINEAR,
                        VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR,
                        VLLMLayerType.ROW_PARALLEL_LINEAR,
                        VLLMLayerType.PARALLEL_LM_HEAD,
                        VLLMLayerType.RMSNORM,
                        VLLMLayerType.LAYERNORM,
                    )
                ):
                    continue
                seen.add(node.id)
                attr_name = self._vllm_layer_attr_name(node)
                is_repeated = self._is_vllm_repeated_layer_node(node)
                inner: list[str] = []
                self._emit_single_layer_init(inner, node, layer_type, classification, indent + 1)
                if is_repeated:
                    add(lines, indent, f"self.{attr_name} = nn.ModuleList([")
                    lines.extend(inner)
                    add(lines, indent + 1, f"for i in range({num_layers_expr})")
                    add(lines, indent, "])")
                else:
                    add(lines, indent, f"self.{attr_name} = " + inner[0].strip())
                    lines.extend(inner[1:])
                self._vllm_emitted_layer_node_ids.add(node.id)
                # Modules that don't store prefix in __init__ need it set after creation.
                # PARALLEL_LM_HEAD is excluded: when tied to embeddings it shares the
                # same prefix, which would overwrite the embedding entry in the
                # checkpoint-to-model mapping.
                if layer_type in (
                    VLLMLayerType.RMSNORM,
                    VLLMLayerType.LAYERNORM,
                    VLLMLayerType.VOCAB_PARALLEL_EMBEDDING,
                ):
                    prefix_expr = self._layer_prefix(node)
                    prefix_setters.append((attr_name, prefix_expr, is_repeated))
        # Emit prefix-setting code for modules that need it
        if prefix_setters:
            add(lines, indent, "# Set prefix on modules that don't accept it in __init__")
            for attr_name, prefix_expr, is_repeated in prefix_setters:
                if is_repeated:
                    fixed_expr = prefix_expr.replace("{i}", "{_i}")
                    add(lines, indent, f"for _i, _mod in enumerate(self.{attr_name}):")
                    add(lines, indent + 1, f"_mod.prefix = {fixed_expr}")
                else:
                    add(lines, indent, f"self.{attr_name}.prefix = {prefix_expr}")

        if classification.mamba_mixer_module_names:
            add(lines, indent, "from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer")
            hidden_expr = self._config_expr("hidden_size")
            state_expr = self._config_expr("state_size")
            conv_k_expr = self._config_expr("conv_kernel")
            inter_expr = self._config_expr("intermediate_size")
            tsr_expr = self._config_expr("time_step_rank")
            num_layers_expr = self._config_expr("num_hidden_layers")
            eps_expr = (
                "(getattr(config, 'layer_norm_epsilon', None) "
                "if getattr(config, 'layer_norm_epsilon', None) is not None "
                "else (getattr(config, 'rms_norm_eps', None) "
                "if getattr(config, 'rms_norm_eps', None) is not None "
                "else self._model_config.get('layer_norm_epsilon', self._model_config.get('rms_norm_eps', 1e-5))))"
            )
            mixer_prefix = self._derive_mamba_mixer_prefix_from_scope(classification)
            add(lines, indent, f"self._vllm_mamba_mixer = nn.ModuleList([")
            add(lines, indent + 1, f"MambaMixer(")
            add(lines, indent + 2, f"hidden_size={hidden_expr},")
            add(lines, indent + 2, f"ssm_state_size={state_expr},")
            add(lines, indent + 2, f"conv_kernel_size={conv_k_expr},")
            add(lines, indent + 2, f"intermediate_size={inter_expr},")
            add(lines, indent + 2, f"time_step_rank={tsr_expr},")
            add(lines, indent + 2, f"use_conv_bias={self._config_expr('use_conv_bias', default=True)},")
            add(lines, indent + 2, f"use_bias={self._config_expr('use_bias', default=False)},")
            add(lines, indent + 2, f"use_rms_norm=False,")
            add(lines, indent + 2, f"rms_norm_eps={eps_expr},")
            add(lines, indent + 2, f"activation={self._config_expr('hidden_act', default='silu')},")
            add(lines, indent + 2, f"model_config=vllm_config.model_config,")
            add(lines, indent + 2, f"cache_config=cache_config,")
            add(lines, indent + 2, f"prefix={mixer_prefix},")
            add(lines, indent + 1, f")")
            add(lines, indent + 1, f"for i in range({num_layers_expr})")
            add(lines, indent, "])")

        # Emit non-vLLM parameters (Params.param nodes classified as DEFAULT)
        # that are inside repeated modules. These need to be created as
        # nn.ParameterList and registered in state_dict_tensors directly.
        if not use_clean_forward:
            legacy_params: list[tuple[str, str, list[str], bool]] = []
            seen_param_paths: set[str] = set()
            for module in self.program.modules:
                for node in module.nodes:
                    if node.op.name != 'Params.param':
                        continue
                    if classification.layer_type(node) != VLLMLayerType.DEFAULT:
                        continue
                    path_inp = node.inputs[0] if node.inputs else None
                    if not isinstance(path_inp, GraphPath) or not path_inp.parts:
                        continue
                    path = '.'.join(path_inp.parts)
                    if path in seen_param_paths:
                        continue
                    seen_param_paths.add(path)
                    is_repeated = self._is_vllm_repeated_layer_node(node)
                    attr_name = self._vllm_layer_attr_name(node)
                    prefix_expr = self._layer_prefix(node)
                    # Resolve shape from output type dims
                    shape_parts: list[str] = []
                    for out in node.outputs:
                        if hasattr(out, 'dims') and out.dims:
                            for d in out.dims:
                                if isinstance(d, int):
                                    shape_parts.append(str(d))
                                else:
                                    val = self._resolve_const_value(str(d))
                                    if val is not None:
                                        shape_parts.append(val)
                                    else:
                                        shape_parts.append(self._config_expr(str(d), default=1))
                            break
                    if not shape_parts:
                        continue
                    legacy_params.append((attr_name, prefix_expr, shape_parts, is_repeated))

            for attr_name, prefix_expr, shape_parts, is_repeated in legacy_params:
                shape_str = ', '.join(shape_parts)
                if is_repeated:
                    add(lines, indent, f"self.{attr_name} = nn.ParameterList([")
                    add(lines, indent + 1, f"nn.Parameter(torch.zeros(({shape_str},), dtype=params_dtype))")
                    add(lines, indent + 1, f"for i in range({num_layers_expr})")
                    add(lines, indent, "])")
                else:
                    add(lines, indent, f"self.{attr_name} = nn.Parameter(torch.zeros(({shape_str},), dtype=params_dtype))")

            # Register non-vLLM parameters in state_dict_tensors
            if legacy_params:
                add(lines, indent, "# Register non-vLLM parameters in state_dict_tensors")
                for attr_name, prefix_expr, shape_parts, is_repeated in legacy_params:
                    if is_repeated:
                        fixed_expr = prefix_expr.replace("{i}", "{_i}")
                        add(lines, indent, f"for _i, _param in enumerate(self.{attr_name}):")
                        add(lines, indent + 1, f"self.state_dict_tensors[{fixed_expr}] = _param")
                    else:
                        add(lines, indent, f"self.state_dict_tensors[{prefix_expr}] = self.{attr_name}")

    def _emit_single_layer_init(
        self,
        lines: list[str],
        node: GraphNode,
        layer_type: VLLMLayerType,
        classification: VLLMLayerClassification,
        indent: int,
    ) -> None:
        add = self._add
        prefix = self._layer_prefix(node)
        if layer_type == VLLMLayerType.VOCAB_PARALLEL_EMBEDDING:
            native_prefix = self._vllm_embedding_layer_prefix(node)
            if native_prefix is not None:
                prefix = repr(native_prefix)
            if node.id in classification.position_embedding_node_ids:
                vocab = (
                    "getattr(config, 'max_position_embeddings', "
                    "getattr(config, 'n_positions', "
                    "self._model_config.get('max_position_embeddings', "
                    "self._model_config.get('n_positions', 0))))"
                )
            else:
                vocab = self._config_expr("vocab_size")
            if node.id == self._vllm_classification.pli_embed_node_id:
                num_layers = self._config_expr("num_hidden_layers")
                pli_dim = self._resolve_const_value("PLI") or self._config_expr("per_layer_input_dim", default=256)
                dim = f"({num_layers} * {pli_dim})"
            else:
                dim = self._config_expr("hidden_size")
            add(lines, indent, "VocabParallelEmbedding(")
            add(lines, indent + 1, f"{self._ctor_int_expr(vocab)}, {self._ctor_int_expr(dim)},")
            add(lines, indent + 1, f"prefix={prefix},")
            add(lines, indent, ")")
        elif layer_type == VLLMLayerType.PARALLEL_LM_HEAD:
            vocab = self._config_expr("vocab_size")
            dim = self._config_expr("hidden_size")
            add(lines, indent, "ParallelLMHead(")
            add(lines, indent + 1, f"{self._ctor_int_expr(vocab)}, {self._ctor_int_expr(dim)},")
            add(lines, indent + 1, f"prefix={prefix},")
            add(lines, indent, ")")
        elif layer_type in (
            VLLMLayerType.QKV_PARALLEL_LINEAR,
            VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR,
            VLLMLayerType.COLUMN_PARALLEL_LINEAR,
            VLLMLayerType.ROW_PARALLEL_LINEAR,
        ):
            pli_linear_ids = {
                classification.pli_gate_node_id,
                classification.pli_proj_node_id,
            }
            pli_linear_ids.discard(None)
            in_dim = self._node_input_dim_expr(node)
            out_dim = self._node_output_dim_expr(node)
            bias = self._linear_bias_arg(node)
            if node.id == classification.pli_model_proj_node_id:
                add(lines, indent, "ColumnParallelLinear(")
                add(lines, indent + 1, f"{self._ctor_int_expr(in_dim)}, {self._ctor_int_expr(out_dim)},")
                add(lines, indent + 1, f"bias={bias}, gather_output=True,")
                add(lines, indent + 1, f"prefix={prefix},")
                add(lines, indent + 1, "quant_config=quant_config,")
                add(lines, indent + 1, "params_dtype=params_dtype,")
                add(lines, indent, ")")
            elif node.id in pli_linear_ids:
                add(lines, indent, "ReplicatedLinear(")
                add(lines, indent + 1, f"{self._ctor_int_expr(in_dim)}, {self._ctor_int_expr(out_dim)},")
                add(lines, indent + 1, f"bias={bias}, skip_bias_add=False,")
                add(lines, indent + 1, f"prefix={prefix},")
                add(lines, indent + 1, "params_dtype=params_dtype,")
                add(lines, indent, ")")
            elif layer_type == VLLMLayerType.QKV_PARALLEL_LINEAR:
                self._emit_qkv_init(lines, node, classification, indent)
            elif layer_type == VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR:
                up_node_id = self._ffn_gate_to_up.get(node.id)
                if up_node_id:
                    up_node = self._find_node_by_id(up_node_id)
                    down_node = self._find_node_by_id(
                        getattr(self, "_ffn_gate_to_down", {}).get(node.id, "")
                    )
                    if down_node is not None:
                        out_dim = self._node_input_dim_expr(down_node)
                        up_out_dim = out_dim
                    else:
                        up_out_dim = self._node_output_dim_expr(up_node) if up_node else out_dim
                    merged_prefix = prefix.replace(".gate_proj", ".gate_up_proj")
                    add(lines, indent, "MergedColumnParallelLinear(")
                    add(
                        lines,
                        indent + 1,
                        f"{self._ctor_int_expr(in_dim)}, [{self._ctor_int_expr(out_dim)}, {self._ctor_int_expr(up_out_dim)}],",
                    )
                    add(lines, indent + 1, f"bias={bias},")
                    add(lines, indent + 1, f"prefix={merged_prefix},")
                    add(lines, indent + 1, "quant_config=quant_config,")
                    add(lines, indent + 1, "params_dtype=params_dtype,")
                    add(lines, indent, ")")
                else:
                    add(lines, indent, "ColumnParallelLinear(")
                    add(lines, indent + 1, f"{self._ctor_int_expr(in_dim)}, {self._ctor_int_expr(out_dim)},")
                    add(lines, indent + 1, f"bias={bias}, skip_bias_add=False,")
                    add(lines, indent + 1, f"prefix={prefix},")
                    add(lines, indent + 1, "quant_config=quant_config,")
                    add(lines, indent + 1, "params_dtype=params_dtype,")
                    add(lines, indent, ")")
            elif layer_type == VLLMLayerType.COLUMN_PARALLEL_LINEAR:
                add(lines, indent, "ColumnParallelLinear(")
                add(lines, indent + 1, f"{self._ctor_int_expr(in_dim)}, {self._ctor_int_expr(out_dim)},")
                add(lines, indent + 1, f"bias={bias}, skip_bias_add=False,")
                add(lines, indent + 1, f"prefix={prefix},")
                add(lines, indent + 1, "quant_config=quant_config,")
                add(lines, indent + 1, "params_dtype=params_dtype,")
                add(lines, indent, ")")
            else:
                add(lines, indent, "RowParallelLinear(")
                add(lines, indent + 1, f"{self._ctor_int_expr(in_dim)}, {self._ctor_int_expr(out_dim)},")
                add(lines, indent + 1, f"bias={bias}, skip_bias_add=False,")
                add(lines, indent + 1, f"prefix={prefix},")
                add(lines, indent + 1, "quant_config=quant_config,")
                add(lines, indent + 1, "params_dtype=params_dtype,")
                add(lines, indent, ")")
        elif layer_type == VLLMLayerType.ATTENTION:
            head_size = self._head_dim_expr()
            kv_heads_arg = "_num_kv_heads"
            if self._is_repeated_node(node):
                repeated_mod = self._get_repeated_module()
                if repeated_mod is not None:
                    loop_var = self._node_loop_index(node)
                    hd_expr = self._attention_head_dim_expr(node, repeated_mod, loop_var)
                    if hd_expr:
                        head_size = hd_expr
                    kvh_expr = self._detect_kv_heads_expr(repeated_mod, loop_var)
                    if kvh_expr:
                        kv_heads_arg = f"max(1, {kvh_expr} // _tp_size)"
            explicit_scale_expr = self._attention_scale_expr_from_graph(node)
            attn_scale = self._model_config_data.get("attention_scale")
            if explicit_scale_expr is not None:
                scale_expr = explicit_scale_expr
            elif attn_scale is not None:
                scale_expr = f"{float(attn_scale)}"
            else:
                scale_cfg = self._query_pre_attn_scalar_expr()
                scale_expr = f"1.0 / (float({scale_cfg}) ** 0.5)"
            add(lines, indent, "Attention(")
            add(lines, indent + 1, "_num_heads,")
            add(lines, indent + 1, f"{self._ctor_int_expr(head_size)},")
            add(lines, indent + 1, f"scale={scale_expr},")
            add(lines, indent + 1, f"num_kv_heads={self._ctor_int_expr(kv_heads_arg)},")
            alibi_attr = node.attrs.get("alibi_slopes")
            use_alibi = isinstance(alibi_attr, GraphLiteral) and bool(alibi_attr.value)
            if use_alibi:
                alibi_scale_attr = node.attrs.get("alibi_scale")
                alibi_scale = (
                    alibi_scale_attr.value
                    if isinstance(alibi_scale_attr, GraphLiteral)
                    and isinstance(alibi_scale_attr.value, (int, float))
                    else 1.0
                )
                total_heads_expr = self._config_expr("num_attention_heads")
                add(lines, indent + 1, "alibi_slopes=(")
                add(lines, indent + 2, f"_axon_alibi_slopes(int({total_heads_expr}), {float(alibi_scale)})")
                add(lines, indent + 2, "[_tp_rank * _num_heads:(_tp_rank + 1) * _num_heads].tolist()),")
            add(lines, indent + 1, "cache_config=cache_config,")
            add(lines, indent + 1, "quant_config=quant_config,")
            add(lines, indent + 1, "logits_soft_cap=getattr(config, 'attn_logit_softcapping', None),")
            is_rep = self._is_vllm_repeated_layer_node(node)
            if is_rep:
                add(lines, indent + 1, "per_layer_sliding_window=(")
                add(lines, indent + 2, "getattr(config, 'sliding_window', None)")
                add(lines, indent + 2, "if ((hasattr(config, 'layer_types') and i < len(config.layer_types) and config.layer_types[i] != 'full_attention')")
                add(lines, indent + 2, "    or (not hasattr(config, 'layer_types') and getattr(config, 'sliding_window_pattern', 0) and (i + 1) % getattr(config, 'sliding_window_pattern', 0) != 0))")
                add(lines, indent + 2, "else None),")
            else:
                add(lines, indent + 1, "per_layer_sliding_window=getattr(config, 'sliding_window', None),")
            # Attention has no checkpoint weights; vLLM uses its prefix as a
            # KV-cache layer name. Repeated modules must be globally unique but
            # contain exactly one integer for vLLM cache binding.
            attn_prefix = self._attention_layer_prefix(node, fallback_prefix=prefix)
            attn_prefix_suffix = self._attention_prefix_suffix_for_node(node, attn_prefix)
            attn_prefix = self._with_attention_prefix_suffix(
                attn_prefix,
                attn_prefix_suffix,
            )
            if is_rep:
                target_prefix = self._attention_layer_prefix(
                    node,
                    fallback_prefix=prefix,
                    loop_var="_kv_sharing_targets[i]",
                    suffix=attn_prefix_suffix,
                )
                add(lines, indent + 1, "kv_sharing_target_layer_name=(")
                add(lines, indent + 2, f"{target_prefix}")
                add(lines, indent + 2, "if i in _kv_sharing_targets else None),")
            else:
                add(lines, indent + 1, "kv_sharing_target_layer_name=None,")
            add(lines, indent + 1, f"prefix={attn_prefix},")
            add(lines, indent, ")")
        elif layer_type == VLLMLayerType.RMSNORM:
            if node.id in self._vllm_classification.qk_norm_node_ids:
                dim = self._head_dim_expr()
                if self._is_repeated_node(node):
                    repeated_mod = self._get_repeated_module()
                    if repeated_mod is not None:
                        loop_var = self._node_loop_index(node)
                        hd_expr = self._detect_head_dim_expr(repeated_mod, loop_var)
                        if hd_expr:
                            dim = hd_expr
            elif node.id in self._vllm_classification.v_norm_node_ids:
                dim = self._head_dim_expr()
                if self._is_repeated_node(node):
                    repeated_mod = self._get_repeated_module()
                    if repeated_mod is not None:
                        loop_var = self._node_loop_index(node)
                        hd_expr = self._detect_head_dim_expr(repeated_mod, loop_var)
                        if hd_expr:
                            dim = hd_expr
            elif node.id == self._vllm_classification.pli_proj_norm_node_id:
                dim = self._resolve_const_value("PLI") or self._config_expr("per_layer_input_dim", default=256)
            else:
                dim = self._config_expr("hidden_size")
            eps = self._node_rmsnorm_eps(node)
            has_weight = "rmsnorm_noscale" not in node.op.name
            norm_cls = "GemmaRMSNorm" if self._rmsnorm_uses_unit_offset(node) else "RMSNorm"
            if has_weight:
                add(lines, indent, f"{norm_cls}(")
                add(lines, indent + 1, f"{self._ctor_int_expr(dim)}, eps={eps},")
                add(lines, indent, ")")
            else:
                add(lines, indent, "RMSNorm(")
                add(lines, indent + 1, f"{self._ctor_int_expr(dim)}, eps={eps}, has_weight=False,")
                add(lines, indent, ")")
        elif layer_type == VLLMLayerType.LAYERNORM:
            dim = self._config_expr("hidden_size")
            eps = self._node_layernorm_eps(node)
            add(lines, indent, "nn.LayerNorm(")
            add(lines, indent + 1, f"{self._ctor_int_expr(dim)}, eps={eps},")
            add(lines, indent, ")")

    def _attention_scale_expr_from_graph(self, node: GraphNode) -> str | None:
        """Derive vLLM Attention scale from primitive structure when explicit.

        The detector intentionally does not rely on ordinary Axon definition
        names. It recognizes explicit scale carried by the vLLM backend
        intrinsic and the direct primitive pattern
        `_matmul(q, k^T) -> core.binary.*(..., scalar)` inside a callee body.
        If the structure is absent, default, or ambiguous, the existing vLLM
        default scaling path remains in force.
        """

        if node.op.name == "__vllm_paged_attention" and len(node.inputs) >= 5:
            scale = node.inputs[4]
            if isinstance(scale, GraphLiteral):
                if scale.value is None:
                    return None
                if isinstance(scale.value, (int, float)):
                    return repr(float(scale.value))
            return None

        callee = self.modules_by_name.get(node.op.name)
        if callee is None:
            return None
        matmul_outputs: set[str] = set()
        for inner in callee.nodes:
            if inner.op.name == "_matmul":
                matmul_outputs.update(out.name for out in inner.outputs)
        if not matmul_outputs:
            return None
        scale_exprs: list[str] = []
        for inner in callee.nodes:
            if inner.op.name != "core.binary.*" or len(inner.inputs) != 2:
                continue
            left, right = inner.inputs
            left_name = getattr(left, "name", None)
            right_name = getattr(right, "name", None)
            scalar = None
            if left_name in matmul_outputs:
                scalar = right
            elif right_name in matmul_outputs:
                scalar = left
            if not isinstance(scalar, GraphLiteral) or not isinstance(scalar.value, (int, float)):
                continue
            scale_exprs.append(repr(float(scalar.value)))
        unique = sorted(set(scale_exprs))
        return unique[0] if len(unique) == 1 else None

    def _emit_qkv_init(
        self,
        lines: list[str],
        node: GraphNode,
        classification: VLLMLayerClassification,
        indent: int,
    ) -> None:
        add = self._add

        is_q_node = any(g.q_node_id == node.id for g in classification.qkv_groups)
        if not is_q_node or not self._use_clean_forward:
            in_dim = self._node_input_dim_expr(node)
            out_dim = self._node_output_dim_expr(node)
            bias = self._linear_bias_arg(node)
            prefix = self._layer_prefix(node)
            add(lines, indent, "ColumnParallelLinear(")
            add(lines, indent + 1, f"{self._ctor_int_expr(in_dim)}, {self._ctor_int_expr(out_dim)},")
            add(lines, indent + 1, f"bias={bias}, skip_bias_add=False,")
            add(lines, indent + 1, f"prefix={prefix},")
            add(lines, indent + 1, "quant_config=quant_config,")
            add(lines, indent + 1, "params_dtype=params_dtype,")
            add(lines, indent, ")")
            return

        hidden_size = self._node_input_dim_expr(node)
        head_size = self._head_dim_expr()
        if self._is_repeated_node(node):
            repeated_mod = self._get_repeated_module()
            if repeated_mod is not None:
                loop_var = self._node_loop_index(node)
                hd_expr = self._detect_head_dim_expr(repeated_mod, loop_var)
                if hd_expr:
                    head_size = hd_expr
        total_num_heads = self._config_expr("num_attention_heads")
        total_num_kv_heads = self._config_expr("num_key_value_heads", alt="num_attention_heads")
        if self._is_repeated_node(node):
            repeated_mod = self._get_repeated_module()
            if repeated_mod is not None:
                loop_var = self._node_loop_index(node)
                kvh_expr = self._detect_kv_heads_expr(repeated_mod, loop_var)
                if kvh_expr:
                    total_num_kv_heads = kvh_expr
        bias = self._linear_bias_arg(node)

        qkv_group = None
        for g in classification.qkv_groups:
            if g.q_node_id == node.id:
                qkv_group = g
                break

        if qkv_group is not None and qkv_group.q_node_id != qkv_group.k_node_id:
            qkv_prefix = self._qkv_layer_prefix(node)
        else:
            qkv_prefix = self._layer_prefix(node)

        add(lines, indent, "QKVParallelLinear(")
        add(lines, indent + 1, f"{self._ctor_int_expr(hidden_size)}, {self._ctor_int_expr(head_size)},")
        add(lines, indent + 1, f"{self._ctor_int_expr(total_num_heads)}, {self._ctor_int_expr(total_num_kv_heads)},")
        add(lines, indent + 1, f"bias={bias}, skip_bias_add=False,")
        add(lines, indent + 1, f"prefix={qkv_prefix},")
        add(lines, indent + 1, "quant_config=quant_config,")
        add(lines, indent + 1, "params_dtype=params_dtype,")
        add(lines, indent, ")")

    def _qkv_layer_prefix(self, q_node: GraphNode) -> str:
        """Compute prefix for fused QKV layer from Q node's path."""
        if not self._is_repeated_node(q_node):
            base = _linear_base_key(q_node)
            if base and "." in base:
                parent = base.rsplit(".", 1)[0] + ".qkv_proj"
                if "{" in parent:
                    parts = self._format_template_parts(
                        parent.split("."),
                        loop_var=self._node_loop_index(q_node),
                        map_unknown_templates_to_loop=True,
                    )
                    return f'f"{".".join(parts)}"'
                return repr(parent)
            return repr("qkv_proj")
        mod_name = self._node_module_name(q_node)
        scope_parts = self._vllm_classification.module_scope_parts.get(mod_name)
        if scope_parts is None:
            base = _linear_base_key(q_node)
            if base and "{" in base:
                parent = base.rsplit(".", 1)[0] + ".qkv_proj"
                parts = self._format_template_parts(
                    parent.split("."),
                    loop_var=self._node_loop_index(q_node),
                    map_unknown_templates_to_loop=True,
                )
                return f'f"{".".join(parts)}"'
            return f'f"{{prefix}}.layers.{{i}}.qkv_proj"'
        base = _linear_base_key(q_node)
        if base:
            sub_parts = base.split(".")
            if sub_parts and sub_parts[0] == "{__scope}":
                full_parts = list(scope_parts) + sub_parts[1:]
            elif (
                sub_parts
                and sub_parts[0].startswith("{")
                and sub_parts[0].endswith("}")
                and sub_parts[0][1:-1] in self._module_path_formal_names(mod_name)
            ):
                full_parts = list(scope_parts) + sub_parts[1:]
            else:
                full_parts = sub_parts
        else:
            full_parts = list(scope_parts) + ["self_attn"]
        if full_parts:
            full_parts[-1] = "qkv_proj"
        else:
            full_parts = ["qkv_proj"]
        fparts = self._format_template_parts(
            full_parts,
            loop_var=self._node_loop_index(q_node),
            map_unknown_templates_to_loop=True,
        )
        return f'f"{".".join(fparts)}"'

    def _attention_layer_prefix(
        self,
        attn_node: GraphNode,
        *,
        fallback_prefix: str,
        loop_var: str | None = None,
        suffix: str | None = None,
    ) -> str:
        """Compute vLLM Attention prefix from graph path metadata.

        Attention nodes do not carry parameter paths, but the corresponding QKV
        projection does. Use that lexical path and replace only the projection
        leaf with vLLM's cache-layer leaf (`attn`). This avoids hardcoding
        family-specific names such as `self_attn`.
        """
        group = next(
            (
                g
                for g in self._vllm_classification.qkv_groups
                if g.attention_node_id == attn_node.id
            ),
            None,
        )
        q_node = self._find_node_by_id(group.q_node_id) if group is not None else None
        if q_node is None:
            return self._with_attention_prefix_suffix(fallback_prefix, suffix)
        if loop_var is None:
            loop_var = self._node_loop_index(q_node)
        base = _linear_base_key(q_node)
        if not self._is_repeated_node(q_node):
            if base and "." in base:
                parent = base.rsplit(".", 1)[0] + ".attn"
                if "{" in parent:
                    parts = self._format_template_parts(
                        parent.split("."),
                        loop_var=loop_var,
                        map_unknown_templates_to_loop=True,
                    )
                    return self._with_attention_prefix_suffix(
                        f'f"{".".join(parts)}"', suffix
                    )
                return self._with_attention_prefix_suffix(repr(parent), suffix)
            return self._with_attention_prefix_suffix(fallback_prefix, suffix)
        mod_name = self._node_module_name(q_node)
        scope_parts = self._vllm_classification.module_scope_parts.get(mod_name)
        if scope_parts is None:
            if base and "{" in base:
                parent = base.rsplit(".", 1)[0] + ".attn"
                parts = self._format_template_parts(
                    parent.split("."),
                    loop_var=loop_var,
                    map_unknown_templates_to_loop=True,
                )
                return self._with_attention_prefix_suffix(
                    f'f"{".".join(parts)}"', suffix
                )
            return self._with_attention_prefix_suffix(fallback_prefix, suffix)
        if base:
            sub_parts = base.split(".")
            if sub_parts and sub_parts[0] == "{__scope}":
                full_parts = list(scope_parts) + sub_parts[1:]
            elif (
                sub_parts
                and sub_parts[0].startswith("{")
                and sub_parts[0].endswith("}")
                and sub_parts[0][1:-1] in self._module_path_formal_names(mod_name)
            ):
                full_parts = list(scope_parts) + sub_parts[1:]
            else:
                full_parts = sub_parts
        else:
            full_parts = list(scope_parts) + ["attn"]
        if full_parts:
            full_parts[-1] = "attn"
        else:
            full_parts = ["attn"]
        fparts = self._format_template_parts(
            full_parts,
            loop_var=loop_var,
            map_unknown_templates_to_loop=True,
        )
        return self._with_attention_prefix_suffix(f'f"{".".join(fparts)}"', suffix)

    @staticmethod
    def _with_attention_prefix_suffix(prefix_expr: str, suffix: str | None) -> str:
        if not suffix:
            return prefix_expr
        return f"({prefix_expr} + '.{suffix}')"

    def _attention_prefix_suffix_for_node(
        self,
        node: GraphNode,
        prefix_expr: str,
    ) -> str | None:
        cached = self._vllm_attention_prefix_suffixes.get(node.id)
        if cached is not None:
            return cached or None
        count = self._vllm_attention_prefix_counts.get(prefix_expr, 0)
        self._vllm_attention_prefix_counts[prefix_expr] = count + 1
        suffix = "" if count == 0 else f"__axon_attn_{count + 1}"
        self._vllm_attention_prefix_suffixes[node.id] = suffix
        return suffix or None

    def _emit_weight_loading_body(
        self,
        lines: list[str],
        classification: VLLMLayerClassification,
        indent: int,
    ) -> None:
        add = self._add
        if classification.has_k_eq_v:
            add(lines, indent, "_names = [name]")
            add(lines, indent, "if _use_k_eq_v and '.self_attn.k_proj.' in name:")
            add(lines, indent + 1, "_m = _re.search(r'layers\\.(\\d+)\\.', name)")
            add(lines, indent + 1, "if _m and int(_m.group(1)) in _k_eq_v_layers:")
            add(lines, indent + 2, "_names.append(name.replace('k_proj', 'v_proj'))")
            add(lines, indent, "for name in _names:")
            self._emit_weight_loading_inner(lines, indent + 1)
        else:
            self._emit_weight_loading_inner(lines, indent)

    def _emit_weight_loading_inner(
        self,
        lines: list[str],
        indent: int,
    ) -> None:
        add = self._add
        add(lines, indent, "_orig_name = name")
        add(lines, indent, "self._loaded_state_keys.add(_orig_name)")
        add(lines, indent, "for _pname, _wname, _sid in stacked_params_mapping:")
        add(lines, indent + 1, "if _wname not in name:")
        add(lines, indent + 2, "continue")
        add(lines, indent + 1, "_mapped_name = name.replace(_wname, _pname)")
        add(lines, indent + 1, "_loaded_any = False")
        add(lines, indent + 1, "for _target_name in _ckpt_to_model.get(_mapped_name, [_mapped_name]):")
        add(lines, indent + 2, "if _load_named_param(_target_name, loaded_weight, _sid):")
        add(lines, indent + 3, "_loaded_any = True")
        add(lines, indent + 1, "if not _loaded_any:")
        add(lines, indent + 2, "self.state_dict_tensors[_orig_name] = loaded_weight")
        add(lines, indent + 2, "break")
        add(lines, indent + 1, "self.state_dict_tensors[_orig_name] = loaded_weight")
        add(lines, indent + 1, "break")
        add(lines, indent, "else:")
        add(lines, indent + 1, "_loaded_any = False")
        add(lines, indent + 1, "for _target_name in _ckpt_to_model.get(name, [name]):")
        add(lines, indent + 2, "if _load_named_param(_target_name, loaded_weight):")
        add(lines, indent + 3, "_loaded_any = True")
        add(lines, indent + 1, "if not _loaded_any:")
        add(lines, indent + 2, "self.state_dict_tensors[_orig_name] = loaded_weight")
        add(lines, indent + 2, "continue")
        add(lines, indent + 1, "self.state_dict_tensors[_orig_name] = loaded_weight")

    def _vllm_layer_attr_name(self, node: GraphNode) -> str:
        return f"_vllm_{_safe_ident(node.id)}"

    def _resolve_prefix_from_called_module(self, node: GraphNode) -> str:
        """Search called module's internal nodes for a GraphPath prefix.

        The optimizer may move GraphPath inputs from a call node into the
        called module's internal nodes (e.g. NN.embedding, NN.rmsnorm_noscale).
        This searches the called module's nodes for a non-weight/bias GraphPath.
        """
        mod_name = node.op.name
        mod = self.modules_by_name.get(mod_name)
        if mod is None:
            return ""
        fallback = ""
        for inner_node in mod.nodes:
            for inp in inner_node.inputs:
                if isinstance(inp, GraphPath) and inp.parts:
                    key = _graph_path_key(inp)
                    if not key.endswith(".weight") and not key.endswith(".bias"):
                        return key
                    if not fallback:
                        fallback = key
        if fallback.endswith(".weight") or fallback.endswith(".bias"):
            return fallback.rsplit(".", 1)[0]
        return fallback

    def _node_prefix(self, node: GraphNode) -> str:
        base = _linear_base_key(node)
        if not base:
            base = self._resolve_prefix_from_called_module(node)
        if not base:
            return node.id
        if base.endswith(".weight") or base.endswith(".bias"):
            return base.rsplit(".", 1)[0]
        return base

    def _module_path_formal_names(self, module_name: str) -> set[str]:
        module = self.modules_by_name.get(module_name)
        if module is None:
            return set()
        return {
            value.name
            for value in module.inputs
            if isinstance(value.type_expr, TypePath)
        }

    def _layer_prefix(self, node: GraphNode) -> str:
        if not self._is_repeated_node(node):
            base = self._node_prefix(node)
            if "{" in base:
                parts = self._format_template_parts(
                    base.split("."),
                    loop_var=self._node_loop_index(node),
                    map_unknown_templates_to_loop=True,
                )
                return f'f"{".".join(parts)}"'
            return repr(base)
        mod_name = self._node_module_name(node)
        scope_parts = self._vllm_classification.module_scope_parts.get(mod_name)
        if scope_parts is None:
            base = self._node_prefix(node)
            if "{" in base:
                parts = self._format_template_parts(
                    base.split("."),
                    loop_var=self._node_loop_index(node),
                    map_unknown_templates_to_loop=True,
                )
                return f'f"{".".join(parts)}"'
            return f'f"{{prefix}}.layers.{{i}}.{_safe_ident(base)}"'
        base = _linear_base_key(node)
        if not base:
            base = self._resolve_prefix_from_called_module(node)
        if base and (base.endswith(".weight") or base.endswith(".bias")):
            base = base.rsplit(".", 1)[0]
        if not base:
            for inp in node.inputs:
                if isinstance(inp, GraphPath) and inp.parts:
                    base = ".".join(inp.parts)
                    break
        if base and (base.endswith(".weight") or base.endswith(".bias")):
            base = base.rsplit(".", 1)[0]
        if base:
            sub_parts = base.split(".")
            if sub_parts and sub_parts[0] == "{__scope}":
                full_parts = list(scope_parts) + sub_parts[1:]
            elif (
                sub_parts
                and sub_parts[0].startswith("{")
                and sub_parts[0].endswith("}")
                and sub_parts[0][1:-1] in self._module_path_formal_names(mod_name)
            ):
                full_parts = list(scope_parts) + sub_parts[1:]
            else:
                full_parts = sub_parts
        else:
            full_parts = list(scope_parts)
        fparts = self._format_template_parts(
            full_parts,
            loop_var=self._node_loop_index(node),
            map_unknown_templates_to_loop=True,
        )
        return f'f"{".".join(fparts)}"'

    def _derive_layer_prefix_expr(
        self, node: GraphNode, classification: VLLMLayerClassification
    ) -> str:
        """Build a runtime f-string expression for the layer-level checkpoint prefix.

        Uses scope_parts from the node's module to produce e.g.
        ``f"model.language_model.layers.{_i}"`` — the prefix used for
        per-layer checkpoint names like ``model.language_model.layers.0.layer_scalar``.
        Falls back to ``f"{self._prefix}.layers.{_i}"`` when scope_parts are unavailable.
        """
        mod_name = self._node_module_name(node)
        scope_parts = classification.module_scope_parts.get(mod_name)
        if scope_parts is None:
            # Optimization may strip scope_parts from the scalar node's module.
            # Search any module's scope_parts that contain '{i}' and use the
            # prefix up to and including '{i}'.
            for parts in classification.module_scope_parts.values():
                if "{i}" in parts:
                    idx = parts.index("{i}")
                    scope_parts = parts[: idx + 1]
                    break
        if scope_parts is None:
            return 'f"{self._prefix}.layers.{_i}"'
        fparts = self._format_template_parts(
            scope_parts,
            loop_var=self._node_loop_index(node),
            loop_target="_i",
            map_unknown_templates_to_loop=True,
        )
        return f'f"{".".join(fparts)}"'

    def _config_expr(
        self,
        config_name: str,
        alt: str | None = None,
        *,
        default: int | float | str = 0,
    ) -> str:
        def _chain(names: tuple[str, ...], fallback: str) -> str:
            expr = fallback
            for name in reversed(names):
                expr = (
                    f"(getattr(self.config, {name!r}, None) "
                    f"if getattr(self.config, {name!r}, None) is not None "
                    f"else self._model_config.get({name!r}, {expr}))"
                )
            return expr

        names = (config_name,) + _CONFIG_ALIASES.get(config_name, ())
        if alt is not None:
            inner = self._config_expr(alt)
            return _chain(names, inner)
        default_repr = repr(default) if isinstance(default, str) else str(default)
        return _chain(names, default_repr)

    def _ctor_int_expr(self, expr: str) -> str:
        expr = str(expr)
        if re.fullmatch(r"[0-9]+", expr):
            return expr
        return f"int({expr})"

    def _format_template_parts(
        self,
        parts: Iterable[str],
        *,
        loop_var: str | None = None,
        loop_target: str = "i",
        map_unknown_templates_to_loop: bool = False,
    ) -> list[str]:
        formatted: list[str] = []
        parts_tuple = tuple(parts)
        for idx, part in enumerate(parts_tuple):
            if part == "{__scope}":
                replacement = self._resolve_path_template_part(parts_tuple, idx)
                if replacement is not None:
                    formatted.append(replacement)
                elif map_unknown_templates_to_loop:
                    formatted.append("self._prefix")
                else:
                    formatted.append("{prefix}")
                continue
            if part.startswith("{") and part.endswith("}"):
                name = part[1:-1]
                if loop_var is not None and name == loop_var:
                    formatted.append(f"{{{loop_target}}}")
                elif name == "i":
                    formatted.append(f"{{{loop_target}}}")
                else:
                    replacement = self._resolve_path_template_part(parts_tuple, idx)
                    if replacement is not None:
                        formatted.append(replacement)
                    elif map_unknown_templates_to_loop:
                        raise ValueError(
                            "Cannot resolve non-loop path template "
                            f"{part!r} in vLLM prefix {'.'.join(parts_tuple)!r}"
                        )
                    else:
                        formatted.append(part)
                continue
            formatted.append(part)
        return formatted

    def _resolve_path_template_part(
        self,
        parts: tuple[str, ...],
        template_index: int,
    ) -> str | None:
        if not self._checkpoint_prefixes:
            return None
        before = [
            part for part in parts[:template_index]
            if not (part.startswith("{") and part.endswith("}"))
        ]
        after: list[str] = []
        for part in parts[template_index + 1:]:
            if part.startswith("{") and part.endswith("}"):
                break
            after.append(part)
        candidates: list[tuple[int, str]] = []
        for prefix in self._checkpoint_prefixes:
            prefix_parts = prefix.split(".")
            combined = [*before, *prefix_parts, *after]
            if not combined:
                continue
            combined_prefix = ".".join(combined)
            min_matched_parts = len(before) + len(prefix_parts)
            if after:
                min_matched_parts += 1
            ancestor_matches = any(
                ".".join(combined[:end]) in self._checkpoint_prefixes
                for end in range(min_matched_parts, len(combined) + 1)
            )
            if (
                combined_prefix in self._checkpoint_prefixes
                or ancestor_matches
                or any(
                    item.startswith(combined_prefix + ".")
                    for item in self._checkpoint_prefixes
                )
            ):
                candidates.append((len(prefix_parts), prefix))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][1]

    def _head_dim_expr(self) -> str:
        hidden_size = f"int({self._config_expr('hidden_size')})"
        num_heads = f"max(1, int({self._config_expr('num_attention_heads', default=1)}))"
        derived = f"({hidden_size} // {num_heads})"
        return (
            "(getattr(config, 'head_dim', None) "
            "if getattr(config, 'head_dim', None) is not None "
            "else (self._model_config.get('head_dim') "
            "if self._model_config.get('head_dim') is not None "
            f"else {derived}))"
        )

    def _query_pre_attn_scalar_expr(self) -> str:
        return (
            "getattr(config, 'query_pre_attn_scalar', "
            f"self._model_config.get('query_pre_attn_scalar', {self._head_dim_expr()}))"
        )

    def _node_input_dim_expr(self, node: GraphNode) -> str:
        if node.id in self._ffn_down_node_ids:
            typed = self._node_input_dim_from_type(node)
            if typed is not None:
                return typed
            # Repeated modules can contain local dim aliases that only resolve
            # in the repeated body. Prefer that semantic type evidence before
            # falling back to config conventions.
            if self._is_repeated_node(node):
                for mod in self._get_node_modules(node):
                    if mod is not None and len(node.inputs) >= 2:
                        loop_var = self._node_loop_index(node)
                        inp = node.inputs[1]
                        inp_dims = getattr(inp, "dims", None) or getattr(getattr(inp, "type_expr", None), "dims", None)
                        if inp_dims and len(inp_dims) > 0:
                            last = inp_dims[-1]
                            py_expr = self._dim_expr_to_python(last, mod, loop_var)
                            if py_expr:
                                return py_expr
            return self._config_expr("intermediate_size")
        layer_type = self._vllm_classification.node_types.get(node.id, VLLMLayerType.DEFAULT)
        if layer_type == VLLMLayerType.ROW_PARALLEL_LINEAR:
            if self._is_repeated_node(node):
                for mod in self._get_node_modules(node):
                    if mod is not None:
                        loop_var = self._node_loop_index(node)
                        # Try extracting dim from input tensor type
                        if len(node.inputs) >= 2:
                            inp = node.inputs[1]
                            inp_dims = getattr(inp, "dims", None) or getattr(getattr(inp, "type_expr", None), "dims", None)
                            if inp_dims and len(inp_dims) > 0:
                                last = inp_dims[-1]
                                py_expr = self._dim_expr_to_python(last, mod, loop_var)
                                if py_expr:
                                    return py_expr
            num_heads = self._config_expr("num_attention_heads")
            head_dim = self._head_dim_expr()
            return f"({num_heads} * {head_dim})"
        typed = self._node_input_dim_from_type(node)
        if typed is not None:
            return typed
        return self._config_expr("hidden_size")

    def _node_input_dim_from_type(self, node: GraphNode) -> str | None:
        """Extract input dim from the node's input tensor type (last dim)."""
        if len(node.inputs) >= 2:
            inp = node.inputs[1]
            inp_dims = getattr(inp, "dims", None) or getattr(getattr(inp, "type_expr", None), "dims", None)
            if inp_dims and len(inp_dims) > 0:
                last = inp_dims[-1]
                typed = self._type_dim_expr_to_python(node, last)
                if typed is not None:
                    return typed
                if isinstance(last, int):
                    return str(last)
                if isinstance(last, str):
                    const = self._resolve_const_value(last)
                    if const:
                        return const
                if hasattr(last, "op") and hasattr(last, "left") and hasattr(last, "right"):
                    left = str(last.left) if isinstance(last.left, int) else (
                        self._resolve_const_value(last.left) if isinstance(last.left, str) else None)
                    right = str(last.right) if isinstance(last.right, int) else (
                        self._resolve_const_value(last.right) if isinstance(last.right, str) else None)
                    if left and right:
                        return f"({left} {last.op} {right})"
        return None

    def _type_dim_expr_to_python(self, node: GraphNode, dim: Any) -> str | None:
        """Render a Graph IR type dimension in the node's lexical context."""
        loop_var = self._node_loop_index(node)
        for mod in self._get_node_modules(node):
            if mod is None:
                continue
            rendered = self._dim_expr_to_python(dim, mod, loop_var)
            if rendered:
                return rendered
        return None

    def _dim_expr_to_python(
        self, dim: Any, repeated_mod: Any, loop_var: str = "i",
    ) -> str | None:
        """Convert a dim expression (int, str, DimExprBinary) to a Python expression."""
        if isinstance(dim, int):
            return str(dim)
        if isinstance(dim, str):
            const = self._resolve_const_value(dim)
            if const:
                return const
            # Try tracing through the module (including cross-module)
            traced = self._trace_dim_expr(GraphValueRef(name=dim, type_expr=None), repeated_mod, loop_var)
            if traced:
                return traced
            hd_expr = self._detect_head_dim_expr(repeated_mod, loop_var)
            if hd_expr and dim == "hd":
                return hd_expr
            return None
        if hasattr(dim, "op") and hasattr(dim, "left") and hasattr(dim, "right"):
            left = self._dim_expr_to_python(dim.left, repeated_mod, loop_var)
            right = self._dim_expr_to_python(dim.right, repeated_mod, loop_var)
            if left and right:
                return f"({left} {dim.op} {right})"
        return None

    def _node_output_dim_expr(self, node: GraphNode) -> str:
        dim = _int_arg(node, 2)
        if dim is not None:
            return str(dim)
        moe_router_dim = self._moe_router_output_dim_from_path(node)
        if moe_router_dim is not None:
            return moe_router_dim
        # Try tracing with the node's own module first, then the primary
        # repeated module. Linear output dim arguments are semantic graph
        # operands; they can be local expressions even outside repeated layers.
        for mod in self._get_node_modules(node):
            if mod is not None and len(node.inputs) >= 3:
                loop_var = self._node_loop_index(node)
                traced = self._trace_dim_expr(node.inputs[2], mod, loop_var)
                if traced and traced != "None":
                    return traced
        # Fallback: extract output dim from the node's output type
        if node.outputs:
            out = node.outputs[0]
            out_dims = getattr(out, "dims", None) or getattr(getattr(out, "type_expr", None), "dims", None)
            if out_dims and len(out_dims) > 0:
                last = out_dims[-1]
                typed = self._type_dim_expr_to_python(node, last)
                if typed is not None:
                    return typed
                if isinstance(last, int):
                    return str(last)
                if isinstance(last, str):
                    const = self._resolve_const_value(last)
                    if const:
                        return const
                    if self._is_repeated_node(node):
                        for mod in self._get_node_modules(node):
                            if mod is not None:
                                loop_var = self._node_loop_index(node)
                                traced = self._trace_dim_expr(
                                    GraphValueRef(name=last, type_expr=None),
                                    mod,
                                    loop_var,
                                    set(),
                                )
                                if traced:
                                    return traced
                if hasattr(last, "op") and hasattr(last, "left") and hasattr(last, "right"):
                    if self._is_repeated_node(node):
                        for mod in self._get_node_modules(node):
                            if mod is not None:
                                loop_var = self._node_loop_index(node)
                                traced = self._dim_expr_to_python(last, mod, loop_var)
                                if traced:
                                    return traced
        return self._config_expr("hidden_size")

    def _moe_router_output_dim_from_path(self, node: GraphNode) -> str | None:
        """Infer router logits width from parameter path metadata.

        Router projections in MoE blocks often omit the explicit linear output
        dim and rely on the gate weight shape. vLLM needs the module size at
        construction time before weights are loaded, so use generic MoE path
        evidence plus expert-count config fields. This is intentionally based
        on parameter path structure, not on Axon definition names.
        """
        prefix = self._node_prefix(node)
        parts = [part for part in prefix.replace("{", ".").replace("}", ".").split(".") if part]
        if not parts:
            return None
        leaf = parts[-1]
        lowered = [part.lower() for part in parts]
        has_moe_context = any(
            "moe" in part or "expert" in part or "router" in part
            for part in lowered[:-1]
        )
        has_router_component = "router" in lowered
        is_router_leaf = leaf in {"gate", "router"}
        if not has_router_component and (not has_moe_context or not is_router_leaf):
            return None
        return self._config_expr("num_local_experts")

    def _node_rmsnorm_eps(self, node: GraphNode) -> str:
        eps = _literal_value(node.inputs[1], None) if len(node.inputs) >= 2 else None
        if isinstance(eps, (int, float)):
            return repr(float(eps))
        eps = self._primitive_arg_literal_from_called_module(
            node.op.name, "_rmsnorm", input_index=1
        )
        if isinstance(eps, (int, float)):
            return repr(float(eps))
        return "1e-6"

    def _rmsnorm_uses_unit_offset(self, node: GraphNode) -> bool:
        module = self.modules_by_name.get(node.op.name)
        if module is None or len(module.outputs) != 1:
            return False
        output = module.outputs[0]
        if not isinstance(output, GraphExpr) or output.op.name != "core.binary.+":
            return False
        left = _value_name(output.inputs[0]) if len(output.inputs) >= 1 else None
        right = _value_name(output.inputs[1]) if len(output.inputs) >= 2 else None
        if left is None or right is None:
            return False
        producers: dict[str, GraphNode] = {}
        for inner in module.nodes:
            for out in inner.outputs:
                name = _value_name(out)
                if name is not None:
                    producers[name] = inner
        left_node = producers.get(left)
        right_node = producers.get(right)
        for scaled_name, base_name, scaled_node, base_node in (
            (left, right, left_node, right_node),
            (right, left, right_node, left_node),
        ):
            if scaled_node is None or base_node is None:
                continue
            if scaled_node.op.name != "Params.param_scale":
                continue
            if base_node.op.name not in {"NN.rmsnorm_noscale", "_rmsnorm"}:
                continue
            if not scaled_node.inputs or _value_name(scaled_node.inputs[0]) != base_name:
                continue
            return True
        return False

    def _linear_bias_arg(self, node: GraphNode, default: bool = False) -> bool:
        bools = _linear_bool_args(node)
        if bools:
            return bools[0]
        primitive_bias = self._primitive_arg_literal_from_called_module(
            node.op.name, "_linear", input_index=3
        )
        if isinstance(primitive_bias, bool):
            return primitive_bias
        if len(node.inputs) < 5:
            return default
        bias_leaf = node.inputs[-1]
        if isinstance(bias_leaf, GraphLiteral) and bias_leaf.value is None:
            return False
        if isinstance(bias_leaf, GraphPath):
            return True
        return default

    def _linear_transpose_arg(self, node: GraphNode, default: bool = False) -> bool:
        bools = _linear_bool_args(node)
        if len(bools) >= 2:
            return bools[1]
        primitive_transpose = self._primitive_arg_literal_from_called_module(
            node.op.name, "_linear", input_index=4
        )
        if isinstance(primitive_transpose, bool):
            return primitive_transpose
        return default

    def _node_layernorm_eps(self, node: GraphNode) -> str:
        eps = _literal_value(node.inputs[2], None) if len(node.inputs) >= 3 else None
        if isinstance(eps, (int, float)):
            return repr(float(eps))
        eps = self._primitive_arg_literal_from_called_module(
            node.op.name, "_layernorm", input_index=2
        )
        if isinstance(eps, (int, float)):
            return repr(float(eps))
        return "1e-5"

    def _primitive_arg_literal_from_called_module(
        self,
        module_name: str,
        primitive_name: str,
        *,
        input_index: int,
        _seen: frozenset[str] = frozenset(),
    ) -> object | None:
        if module_name in _seen:
            return None
        mod = self.modules_by_name.get(module_name)
        if mod is None:
            return None
        seen = frozenset((*_seen, module_name))
        for inner in mod.nodes:
            if inner.op.name == primitive_name and len(inner.inputs) > input_index:
                value = _literal_value(inner.inputs[input_index], None)
                if value is not None:
                    return value
            if inner.op.name in self.modules_by_name:
                value = self._primitive_arg_literal_from_called_module(
                    inner.op.name,
                    primitive_name,
                    input_index=input_index,
                    _seen=seen,
                )
                if value is not None:
                    return value
        return None

    def _find_node_by_id(self, node_id: str) -> GraphNode | None:
        for module in self.program.modules:
            for node in module.nodes:
                if node.id == node_id:
                    return node
        return None

    def _unique_classified_node_for_op(
        self,
        op_name: str,
        layer_type: VLLMLayerType,
    ) -> GraphNode | None:
        matches: list[GraphNode] = []
        for module in self.program.modules:
            for node in module.nodes:
                if (
                    node.op.name == op_name
                    and self._vllm_classification.node_types.get(node.id) == layer_type
                ):
                    matches.append(node)
        return matches[0] if len(matches) == 1 else None

    def _primitive_expr(
        self,
        primitive: str,
        node: GraphNode,
        *,
        local: dict[str, str],
        symbols_dict: str,
    ) -> str | None:
        classification = self._vllm_classification
        node_id = getattr(node, "id", None)
        layer_type = (
            classification.node_types.get(node_id, VLLMLayerType.DEFAULT)
            if node_id is not None
            else VLLMLayerType.DEFAULT
        )

        if primitive == "params_has_root":
            args = self._collect_args(node, local)
            if not args:
                raise ValueError("params_has_root missing root argument")
            return (
                f"any(k == {args[0]} or k.startswith(str({args[0]}) + '.') "
                "for k in self._loaded_state_keys)"
            )

        if primitive == "embedding":
            args = self._collect_args(node, local)
            static_weight_key = self._static_param_key(node.inputs[0], "weight")
            if static_weight_key is not None and len(args) >= 2:
                key = repr(static_weight_key)
                weight = f"self._vllm_state_tensor({key}, self.state_dict_tensors[{key}])"
                return f"(lambda _w: F.embedding({args[1]}.to(_w.device), _w))({weight})"

        if primitive == "reshape":
            shape_expr = self._clean_flattened_reshape_shape_expr(node, local)
            if shape_expr is not None:
                args = self._collect_args(node, local)
                if not args:
                    raise ValueError("reshape expects input tensor")
                return f"torch.reshape({args[0]}, {shape_expr})"

        if primitive == "arange":
            args = self._collect_args(node, local)
            if len(args) < 3:
                raise ValueError("arange expects reference, start, and end")
            end = f"{args[0]}.shape[-1]" if args[2] == "None" else args[2]
            return (
                f"torch.arange({args[1]}, {end}, "
                f"device=({args[0]}.device if torch.is_tensor({args[0]}) else self._vllm_active_device()), "
                "dtype=torch.long)"
            )

        if primitive == "_vllm_selected_expert_packed_swiglu_ffn":
            args = self._collect_args(node, local)
            if len(args) < 6:
                raise ValueError("__vllm_selected_expert_packed_swiglu_ffn expects input, top-k scores/indices, gate-up/down weight paths, and transpose")
            return (
                "self._selected_expert_packed_swiglu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, "
                f"transpose={_bool_arg(node, 5)})"
            )
        if primitive == "_vllm_selected_expert_swiglu_ffn":
            args = self._collect_args(node, local)
            if len(args) < 7:
                raise ValueError("__vllm_selected_expert_swiglu_ffn expects input, top-k scores/indices, gate/up/down weight paths, and transpose")
            return (
                "self._selected_expert_swiglu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, {args[5]}, "
                f"transpose={_bool_arg(node, 6)})"
            )
        if primitive == "_vllm_selected_expert_packed_gegelu_ffn":
            args = self._collect_args(node, local)
            if len(args) < 10:
                raise ValueError("__vllm_selected_expert_packed_gegelu_ffn expects input, top-k scores/indices, gate-up/down weight/bias paths, limit, optional alpha, bias, and transpose")
            alpha_arg = args[8] if len(args) >= 11 else "1.702"
            bias_idx = 9 if len(args) >= 11 else 8
            transpose_idx = 10 if len(args) >= 11 else 9
            return (
                "self._selected_expert_packed_gegelu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, "
                f"{args[5]}, {args[6]}, {args[7]}, alpha={alpha_arg}, "
                f"bias={_bool_arg(node, bias_idx)}, transpose={_bool_arg(node, transpose_idx)})"
            )
        if primitive == "_vllm_selected_expert_clamped_packed_swiglu_ffn":
            args = self._collect_args(node, local)
            if len(args) < 7:
                raise ValueError("__vllm_selected_expert_clamped_packed_swiglu_ffn expects input, top-k scores/indices, gate-up/down weight paths, limit, and transpose")
            return (
                "self._selected_expert_clamped_packed_swiglu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, {args[5]}, "
                f"transpose={_bool_arg(node, 6)})"
            )
        if primitive == "_vllm_selected_expert_relu2_ffn":
            args = self._collect_args(node, local)
            if len(args) < 6:
                raise ValueError("__vllm_selected_expert_relu2_ffn expects input, top-k scores/indices, up/down weight paths, and transpose")
            return (
                "self._selected_expert_relu2_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, "
                f"transpose={_bool_arg(node, 5)})"
            )

        if not self._use_clean_forward:
            if primitive == "_vllm_paged_attention":
                attn_node = node
                if layer_type != VLLMLayerType.ATTENTION:
                    attn_node = self._unique_classified_node_for_op(
                        node.op.name,
                        VLLMLayerType.ATTENTION,
                    ) or node
                args = self._collect_args(node, local)
                attr = self._vllm_attr_access(
                    attn_node,
                    local=local,
                    index_node=attn_node,
                    anchor_exprs=args,
                )
                return self._vllm_attention_call_expr(
                    attr,
                    args,
                    node=node,
                    local=local,
                    output_rank=self._node_output_rank(node),
                )
            return super()._primitive_expr(
                primitive, node, local=local, symbols_dict=symbols_dict
            )

        if layer_type == VLLMLayerType.VOCAB_PARALLEL_EMBEDDING:
            attr = self._vllm_attr_access(node, local=local)
            args = self._collect_args(node, local)
            return f"{attr}({args[1]})"

        if layer_type == VLLMLayerType.QKV_PARALLEL_LINEAR:
            attr = self._vllm_attr_access(node, local=local)
            args = self._collect_args(node, local)
            return f"{attr}({args[1]})[0]"

        if layer_type in (
            VLLMLayerType.COLUMN_PARALLEL_LINEAR,
            VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR,
        ):
            attr = self._vllm_attr_access(node, local=local)
            args = self._collect_args(node, local)
            return f"{attr}({args[1]})[0]"

        if layer_type == VLLMLayerType.ROW_PARALLEL_LINEAR:
            attr = self._vllm_attr_access(node, local=local)
            args = self._collect_args(node, local)
            return f"{attr}({args[1]})[0]"

        if layer_type == VLLMLayerType.PARALLEL_LM_HEAD:
            args = self._collect_args(node, local)
            return args[1]

        if layer_type == VLLMLayerType.ATTENTION:
            args = self._collect_args(node, local)
            attr = self._vllm_attr_access(node, local=local, anchor_exprs=args)
            return self._vllm_attention_call_expr(
                attr,
                args,
                node=node,
                local=local,
                output_rank=self._node_output_rank(node),
            )

        if layer_type == VLLMLayerType.RMSNORM:
            attr = self._vllm_attr_access(node, local=local)
            args = self._collect_args(node, local)
            return f"{attr}({args[0]})"

        if layer_type == VLLMLayerType.LAYERNORM:
            attr = self._vllm_attr_access(node, local=local)
            args = self._collect_args(node, local)
            return f"{attr}({args[0]})"

        if primitive == "_vllm_paged_attention":
            attn_node = node
            if layer_type != VLLMLayerType.ATTENTION:
                attn_node = self._unique_classified_node_for_op(
                    node.op.name,
                    VLLMLayerType.ATTENTION,
                ) or node
            args = self._collect_args(node, local)
            attr = self._vllm_attr_access(
                attn_node,
                local=local,
                index_node=attn_node,
                anchor_exprs=args,
            )
            return self._vllm_attention_call_expr(
                attr,
                args,
                node=node,
                local=local,
                output_rank=self._node_output_rank(node),
            )

        return super()._primitive_expr(primitive, node, local=local, symbols_dict=symbols_dict)

    def _clean_flattened_reshape_shape_expr(
        self,
        node: GraphNode,
        local: dict[str, str],
    ) -> str | None:
        if not self._use_clean_forward or len(node.inputs) < 2:
            return None
        shape = node.inputs[1]
        if not isinstance(shape, GraphExpr) or shape.op.name != "core.list":
            return None
        module_name = self._node_module_name(node)
        module = self.modules_by_name.get(module_name) if module_name else None
        hidden_input = next(
            (
                inp
                for inp in (module.inputs if module is not None else ())
                if isinstance(getattr(inp, "type_expr", None), TypeTensor)
            ),
            None,
        )
        if hidden_input is None:
            return None
        dims = tuple(getattr(hidden_input.type_expr, "dims", ()) or ())
        token_dims = dims[:-1]
        if not token_dims or len(shape.inputs) < len(token_dims):
            return None

        def symbolic_name(operand: GraphOperand) -> str | None:
            while (
                isinstance(operand, GraphExpr)
                and operand.op.name in {"core.alias", "core.ascribe"}
                and operand.inputs
            ):
                operand = operand.inputs[0]
            if isinstance(operand, (GraphValueRef, GraphValue)):
                return operand.name
            return None

        for actual, expected in zip(shape.inputs, token_dims, strict=False):
            if not isinstance(expected, str) or symbolic_name(actual) != expected:
                return None
        tail = [
            self._operand_expr(item, local=local, symbols_dict="self._symbols")
            for item in shape.inputs[len(token_dims):]
        ]
        return "(-1,)" if not tail else f"(-1, {', '.join(tail)})"

    @staticmethod
    def _node_output_rank(node: GraphNode) -> int | None:
        if not node.outputs:
            return None
        dims = getattr(getattr(node.outputs[0], "type_expr", None), "dims", None)
        return len(dims) if dims is not None else None

    def _vllm_attention_call_expr(
        self,
        attr: str,
        args: list[str],
        *,
        node: GraphNode | None = None,
        module_name: str | None = None,
        local: set[str] | dict[str, str] | None = None,
        output_rank: int | None = None,
    ) -> str:
        if len(args) >= 3:
            if node is not None:
                args = self._vllm_attention_compact_kv_args(
                    node,
                    args,
                    module_name=module_name,
                    local=local,
                )
            q = f"{args[0]}.contiguous()"
            k = f"{args[1]}.contiguous()"
            v = f"{args[2]}.contiguous()"
            call = f"{attr}({q}, {k}, {v}, self._attn_metadata)"
            if output_rank == 4:
                return (
                    f"(lambda _q, _k, _v: "
                    f"(lambda _y: _y.reshape(_q.shape[0], _q.shape[2], _q.shape[1], -1)"
                    f".permute(0, 2, 1, 3))("
                    f"{attr}("
                    f"_q.permute(0, 2, 1, 3).reshape(-1, _q.shape[1] * _q.shape[3]).contiguous(), "
                    f"_k.permute(0, 2, 1, 3).reshape(-1, _k.shape[1] * _k.shape[3]).contiguous(), "
                    f"_v.permute(0, 2, 1, 3).reshape(-1, _v.shape[1] * _v.shape[3]).contiguous(), "
                    f"self._attn_metadata)))({args[0]}, {args[1]}, {args[2]})"
                )
            return call
        return f"{attr}(self._attn_metadata)"

    def _vllm_attention_compact_kv_args(
        self,
        node: GraphNode,
        args: list[str],
        *,
        module_name: str | None = None,
        local: set[str] | dict[str, str] | None = None,
    ) -> list[str]:
        """Feed vLLM Attention compact KV tensors, not Axon-expanded GQA KV.

        Axon's backend-neutral attention commonly expands K/V heads with
        `_repeat(..., dim=1)` before a plain matmul attention. vLLM Attention is
        constructed with `num_kv_heads` and performs grouped-query handling
        itself, so passing the expanded K/V changes semantics for GQA models.
        """

        if len(args) < 3:
            return args
        module = self._module_for_node(node, module_name=module_name)
        local_map: set[str] | dict[str, str]
        if local is None:
            local_map = set()
        else:
            local_map = local
        out = list(args)
        for index in (1, 2):
            if index >= len(node.inputs):
                continue
            compact = self._compact_kv_repeat_source_expr(
                module,
                node.inputs[index],
                local=local_map,
            )
            if compact is not None:
                out[index] = compact
        return out

    def _module_for_node(
        self,
        node: GraphNode,
        *,
        module_name: str | None = None,
    ) -> GraphModule | None:
        if module_name is not None:
            module = self.modules_by_name.get(module_name)
            if module is not None:
                return module
        node_mod_name = self._node_module_name(node)
        if node_mod_name:
            module = self.modules_by_name.get(node_mod_name)
            if module is not None:
                return module
        for module in self.program.modules:
            if any(candidate is node or candidate.id == node.id for candidate in module.nodes):
                return module
        return None

    def _compact_kv_repeat_source_expr(
        self,
        module: GraphModule | None,
        operand: GraphOperand,
        *,
        local: set[str] | dict[str, str],
    ) -> str | None:
        producer = self._producer_for_operand(module, operand)
        if producer is None or producer.op.name != "_repeat" or len(producer.inputs) < 3:
            return None
        dim = _literal_value(producer.inputs[2], None)
        if dim != 1:
            return None
        # The repeat count may itself be symbolic; for vLLM Attention any
        # head-dim repeat is the backend-neutral GQA expansion to undo.
        return self._render_operand_for_vllm_attention(producer.inputs[0], local=local)

    def _producer_for_operand(
        self,
        module: GraphModule | None,
        operand: GraphOperand,
    ) -> GraphNode | None:
        name = getattr(operand, "name", None)
        if name is None:
            return None
        emitted = self._current_emitted_defs().get(name)
        if emitted is not None:
            return emitted
        if module is None:
            return None
        for node in module.nodes:
            for out in node.outputs:
                if getattr(out, "name", None) == name:
                    return node
        return None

    def _render_operand_for_vllm_attention(
        self,
        operand: GraphOperand,
        *,
        local: set[str] | dict[str, str],
    ) -> str:
        if isinstance(operand, (GraphValueRef, GraphValue)):
            name = operand.name
            if name in self.global_symbol_names:
                return self._global_symbol_expr(name)
            if isinstance(local, dict):
                return local.get(
                    name,
                    _dim_ident(name) if isinstance(operand.type_expr, TypeDim) else name,
                )
            if isinstance(operand.type_expr, TypeDim):
                return _dim_ident(name)
            return name if name in local else name
        if isinstance(operand, GraphLiteral):
            return repr(operand.value)
        if isinstance(operand, GraphPath):
            return self._vllm_path_key_expr(operand, local=local)
        if isinstance(operand, GraphExpr):
            op = operand.op.name
            if op in ("core.alias", "core.ascribe") and operand.inputs:
                return self._render_operand_for_vllm_attention(operand.inputs[0], local=local)
            if op.startswith("core.binary.") and len(operand.inputs) >= 2:
                sym = _BINOP_SYMBOLS.get(op, op.removeprefix("core.binary."))
                left = self._render_operand_for_vllm_attention(operand.inputs[0], local=local)
                right = self._render_operand_for_vllm_attention(operand.inputs[1], local=local)
                if op == "core.binary.and":
                    return f"({left} and {right})"
                if op == "core.binary.or":
                    return f"({left} or {right})"
                return f"({left} {sym} {right})"
            if op == "core.select" and len(operand.inputs) >= 3:
                cond = self._render_operand_for_vllm_attention(operand.inputs[0], local=local)
                yes = self._render_operand_for_vllm_attention(operand.inputs[1], local=local)
                no = self._render_operand_for_vllm_attention(operand.inputs[2], local=local)
                return f"({yes} if {cond} else {no})"
            if op in self.method_names:
                args = [
                    self._render_operand_for_vllm_attention(item, local=local)
                    for item in operand.inputs
                ]
                args.extend(
                    f"{key}={self._render_operand_for_vllm_attention(value, local=local)}"
                    for key, value in operand.attrs.items()
                )
                call = f"self.{self.method_names[op]}({', '.join(args)})"
                module = self.modules_by_name[op]
                return f"{call}[0]" if len(module.outputs) == 1 else call
        return repr(operand)

    def _operand_expr(
        self,
        operand: GraphOperand,
        *,
        local: set[str] | dict[str, str],
        symbols_dict: str,
    ) -> str:
        if not isinstance(local, dict):
            return super()._operand_expr(operand, local=local, symbols_dict=symbols_dict)
        if isinstance(operand, (GraphValueRef, GraphValue)):
            name = operand.name
            if name in local:
                return local[name]
            if name in self.global_symbol_names:
                return self._global_symbol_expr(name, symbols_dict)
            if isinstance(operand.type_expr, TypeDim):
                return _dim_ident(name)
            return f"{symbols_dict}[{name!r}]"
        if isinstance(operand, GraphLiteral):
            return repr(operand.value)
        if isinstance(operand, GraphPath):
            prefix = "@@" if operand.absolute else "@"
            return f"({prefix!r} + {self._vllm_path_key_expr(operand, local=local)})"
        if isinstance(operand, GraphExpr):
            if (
                operand.op.name in self.global_symbol_names
                and not operand.inputs
                and not operand.attrs
            ):
                return self._global_symbol_expr(operand.op.name, symbols_dict)
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
                    "id": getattr(operand, "id", "<expr>"),
                    "op": operand.op,
                    "inputs": operand.inputs,
                    "attrs": operand.attrs,
                    "outputs": (),
                    "type_expr": operand.type_expr,
                },
            )()
            return self._node_expr(pseudo, local=local, symbols_dict=symbols_dict)
        if isinstance(operand, tuple):
            return "[" + ", ".join(
                self._operand_expr(item, local=local, symbols_dict=symbols_dict)
                for item in operand
            ) + "]"
        raise TypeError(f"unsupported graph operand: {operand!r}")

    def _vllm_path_key_expr(
        self,
        path: GraphPath,
        *,
        local: set[str] | dict[str, str],
    ) -> str:
        template = ".".join(part for part in path.parts if part)
        if "{" not in template and "}" not in template:
            return repr(template)
        pieces: list[str] = []
        cursor = 0
        for match in re.finditer(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", template):
            pieces.append(template[cursor:match.start()].replace("{", "{{").replace("}", "}}"))
            name = match.group(1)
            if isinstance(local, dict):
                expr = local.get(name, self._global_symbol_expr(name))
            else:
                expr = _py_ident(name) if name in local else self._global_symbol_expr(name)
            pieces.append("{self._path_template_part(" + expr + ")}")
            cursor = match.end()
        pieces.append(template[cursor:].replace("{", "{{").replace("}", "}}"))
        return "f" + repr("".join(pieces))

    def _collect_args(self, node: GraphNode, local: set[str] | dict[str, str]) -> list[str]:
        def render_operand(inp: GraphOperand) -> str:
            if isinstance(inp, (GraphValueRef, GraphValue)):
                name = inp.name
                if name in self.global_symbol_names:
                    return self._global_symbol_expr(name)
                if isinstance(local, dict):
                    return local.get(
                        name,
                        _dim_ident(name) if isinstance(inp.type_expr, TypeDim) else name,
                    )
                if isinstance(inp.type_expr, TypeDim):
                    return _dim_ident(name)
                return name if name in local else name
            if isinstance(inp, GraphLiteral):
                return repr(inp.value)
            if isinstance(inp, GraphPath):
                return self._vllm_path_key_expr(inp, local=local)
            if isinstance(inp, GraphExpr):
                op = inp.op.name
                if op in ("core.alias", "core.ascribe") and inp.inputs:
                    return render_operand(inp.inputs[0])
                if op.startswith("core.binary.") and len(inp.inputs) >= 2:
                    sym = _BINOP_SYMBOLS.get(op, op.removeprefix("core.binary."))
                    left = render_operand(inp.inputs[0])
                    right = render_operand(inp.inputs[1])
                    if op == "core.binary.and":
                        return f"({left} and {right})"
                    if op == "core.binary.or":
                        return f"({left} or {right})"
                    return f"({left} {sym} {right})"
                if op == "core.select" and len(inp.inputs) >= 3:
                    cond = render_operand(inp.inputs[0])
                    yes = render_operand(inp.inputs[1])
                    no = render_operand(inp.inputs[2])
                    return f"({yes} if {cond} else {no})"
                if op in self.method_names:
                    args = [render_operand(item) for item in inp.inputs]
                    args.extend(
                        f"{key}={render_operand(value)}"
                        for key, value in inp.attrs.items()
                    )
                    call = f"self.{self.method_names[op]}({', '.join(args)})"
                    module = self.modules_by_name[op]
                    return f"{call}[0]" if len(module.outputs) == 1 else call
            return repr(inp)

        args: list[str] = []
        for inp in node.inputs:
            args.append(render_operand(inp))
        return args

    def _emit_generate(self, lines: list[str]) -> None:
        add = self._add
        add(lines, 4, "def generate(self, *args, **kwargs):")
        add(lines, 8, 'raise NotImplementedError("vLLM handles generation externally")')

    def _needs_mamba_cache_placeholders(self, classification: VLLMLayerClassification) -> bool:
        if classification.mamba_mixer_module_names:
            return False
        has_attention = any(
            lt == VLLMLayerType.ATTENTION for lt in classification.node_types.values()
        )
        if has_attention:
            return False
        for module in self.program.modules:
            for node in module.nodes:
                if 'SSM' in node.op.name or 'ssm' in node.op.name:
                    return True
        return False

    def _derive_mamba_mixer_prefix(self, classification: VLLMLayerClassification) -> str:
        for module in self.program.modules:
            for node in module.nodes:
                lt = classification.node_types.get(node.id)
                if lt in (VLLMLayerType.COLUMN_PARALLEL_LINEAR, VLLMLayerType.ROW_PARALLEL_LINEAR):
                    prefix = self._layer_prefix(node)
                    if '{i}' in prefix:
                        prefix_str = prefix.strip('f"')
                        parts = prefix_str.rsplit('.', 1)
                        if len(parts) == 2:
                            parent = parts[0]
                            if '{i}' in parent:
                                return f'f"{parent}"'
        return ""

    def _derive_mamba_mixer_prefix_from_scope(self, classification: VLLMLayerClassification) -> str:
        for mod_name in classification.mamba_mixer_module_names:
            scope_parts = classification.module_scope_parts.get(mod_name)
            if scope_parts:
                parts = []
                for p in scope_parts:
                    if p == "{i}":
                        parts.append("{i}")
                    else:
                        parts.append(p)
                return f'f"{".".join(parts)}"'
        return 'f"backbone.layers.{i}.mixer"'


def emit_model_code_from_graph_ir(
    program: GraphProgram,
    *,
    class_name: str = "GeneratedVLLMModel",
    model_config: dict[str, Any] | None = None,
    profile: bool = False,
    align_devices: bool = False,
) -> str:
    """Emit a vLLM-compatible model class from Axon Graph IR."""

    validate_graph_program(program)
    uses_selected_expert_moe = _graph_uses_selected_expert_moe(program)
    emitter = _DirectVLLMEmitter(
        program=program,
        class_name=class_name,
        profile=profile,
        align_devices=align_devices,
        model_config=model_config,
    )
    body = emitter.emit()
    header = [
        "",
    ]
    if profile:
        header.append("import time")
    header.extend(
        [
            "import math",
            "import torch",
            "from torch import nn",
            "from torch.nn import functional as F",
            "from brainsurgery.synapse.axon.codegen2_common import (",
            "    config_value as _common_config_value,",
            "    has_config_value as _common_has_config_value,",
            "    required_state_value as _common_required_state_value,",
            ")",
            "",
            f"_MODEL_CONFIG = {model_config!r}",
            "",
        ]
    )
    if uses_selected_expert_moe:
        header.extend(
            [
                "from brainsurgery.synapse.axon.codegen2_torch.core import (",
                "    _grouped_expert_linear_torch,",
                "    _materialize_joined_parameter,",
                ")",
                "",
            ]
        )
    if not emitter._use_clean_forward:
        header.extend(
            [
                "def _common_compose_path(base, leaf):",
                "    base_key = '' if base is None else str(base).strip().lstrip('@')",
                "    leaf_text = '' if leaf is None else str(leaf).strip()",
                "    if leaf_text.startswith('@@'):",
                "        return leaf_text.lstrip('@')",
                "    leaf_key = leaf_text.lstrip('@')",
                "    if not base_key:",
                "        return leaf_key",
                "    if not leaf_key:",
                "        return base_key",
                "    return f'{base_key}.{leaf_key}'",
                "",
                "def _common_render_path(prefix, parts):",
                "    clean = []",
                "    for part in parts:",
                "        if part is None:",
                "            continue",
                "        text = str(part).strip()",
                "        if not text or text == 'None':",
                "            continue",
                "        clean.append(text.strip('@'))",
                "    return str(prefix) + '.'.join(clean)",
                "",
                "def _common_required_state_value(state, path):",
                "    key = str(path).lstrip('@')",
                "    try:",
                "        return state[key]",
                "    except KeyError as exc:",
                "        raise KeyError(f'missing parameter {key!r}') from exc",
                "",
                "def _common_require_value(value):",
                "    if value is None:",
                "        raise ValueError('require expected non-null value')",
                "    return value",
                "",
                "def _materialize_joined_parameter(state, output_key, input_keys, *, dim, mode, remove_inputs=True):",
                "    existing = state.get(output_key)",
                "    if torch.is_tensor(existing):",
                "        return existing",
                "    tensors = [state.get(key) for key in input_keys]",
                "    if not tensors or not all(torch.is_tensor(item) for item in tensors):",
                "        return None",
                "    if mode == 'cat':",
                "        joined = torch.cat(tensors, dim=int(dim))",
                "    elif mode == 'stack':",
                "        joined = torch.stack(tensors, dim=int(dim))",
                "    else:",
                "        raise ValueError(f'unknown parameter join mode {mode!r}')",
                "    state[output_key] = joined",
                "    if remove_inputs:",
                "        for key in input_keys:",
                "            state.pop(key, None)",
                "    return joined",
                "",
            ]
        )
    if emitter._needs_mamba_cache_placeholders(emitter._vllm_classification):
        header.extend(
            [
                "",
                "from vllm.model_executor.layers.mamba.abstract import MambaBase",
                "from vllm.v1.attention.selector import get_mamba_attn_backend",
                "",
                "class _MambaPlaceholderLayer(nn.Module, MambaBase):",
                "    def __init__(self, prefix, intermediate_size, state_size, conv_kernel, dtype=torch.float32):",
                "        super().__init__()",
                "        self.prefix = prefix",
                "        self._intermediate_size = intermediate_size",
                "        self._state_size = state_size",
                "        self._conv_kernel = conv_kernel",
                "        self._dtype = dtype",
                "        self.kv_cache = ()",
                "        from vllm.config import get_current_vllm_config",
                "        _cc = get_current_vllm_config().compilation_config",
                "        _cc.static_forward_context[prefix] = self",
                "    def get_state_shape(self):",
                "        return ((self._intermediate_size, self._conv_kernel), (self._intermediate_size, self._state_size))",
                "    @property",
                "    def mamba_type(self):",
                "        return 'mamba1'",
                "    def get_state_dtype(self):",
                "        return (self._dtype, self._dtype)",
                "    def get_attn_backend(self):",
                "        return get_mamba_attn_backend('mamba1')",
                "",
            ]
        )
    header.extend(
        [
            "from vllm.compilation.decorators import support_torch_compile",
            "",
            "@support_torch_compile",
            body,
        ]
    )
    return "\n".join(header)


def unsupported_reason_for_vllm_graph(program: GraphProgram) -> str | None:
    """Return the backend validation reason for graphs vLLM cannot lower safely."""

    validate_graph_program(program)
    emitter = _DirectVLLMEmitter(
        program=program,
        class_name="GeneratedVLLMUnsupportedProbe",
        profile=False,
        align_devices=False,
        model_config=None,
    )
    try:
        emitter._validate_vllm_lowering_completeness()
    except NotImplementedError as exc:
        return str(exc)
    if any(
        group.gate_up_intrinsic_node_id is not None
        for group in emitter._vllm_classification.ffn_groups
    ):
        try:
            code = emitter.emit()
        except ValueError as exc:
            message = str(exc)
            if "cannot resolve loop index for repeated vLLM layer" in message:
                return message
        if "self._param(" in code or "self._optional_param(" in code:
            return (
                "codegen2-vllm emitted raw parameter helper access in a clean "
                "vLLM transformer forward; this lowering is not supported"
            )
    return None


__all__ = [
    "emit_model_code_from_graph_ir",
    "unsupported_reason_for_vllm_graph",
]
