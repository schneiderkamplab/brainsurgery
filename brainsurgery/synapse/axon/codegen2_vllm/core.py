from __future__ import annotations

import re
from typing import Any

from ..codegen2_torch.core import _DirectTorchEmitter
from ..graph_ir.core import (
    GraphExpr,
    GraphLiteral,
    GraphNode,
    GraphOperand,
    GraphPath,
    GraphProgram,
    GraphValue,
    GraphValueRef,
    validate_graph_program,
)
from .classify import (
    VLLMLayerClassification,
    VLLMLayerType,
    classify_graph_for_vllm,
)


def _graph_path_key(path: GraphPath) -> str:
    return ".".join(path.parts)


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
    base = node.inputs[0]
    if isinstance(base, GraphPath):
        return _graph_path_key(base)
    return ""


def _safe_ident(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe


class _DirectVLLMEmitter(_DirectTorchEmitter):
    """vLLM backend emitter.

    Generates a vLLM-compatible model class that uses vLLM's parallel layers
    (QKVParallelLinear, RowParallelLinear, VocabParallelEmbedding, etc.)
    and vllm.attention.Attention for PagedAttention.

    vLLM-specific graph intrinsics (__vllm_paged_attention, etc.) are
    consumed here; all other ops fall through to the torch emitter.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._vllm_classification: VLLMLayerClassification = classify_graph_for_vllm(self.program)
        self._ffn_down_node_ids: set[str] = {
            g.down_node_id for g in self._vllm_classification.ffn_groups
        }
        self._use_clean_forward: bool = False

    def _resolve_const_value(self, name: str) -> str | None:
        """Resolve a top-level constant name to a config expression."""
        for module in self.program.modules:
            if module.name != name:
                continue
            for node in module.nodes:
                if node.op.name == "_config_int":
                    path_inp = node.inputs[0] if node.inputs else None
                    default = _literal_value(node.inputs[1], None) if len(node.inputs) >= 2 else None
                    if isinstance(path_inp, GraphPath) and path_inp.parts:
                        field = path_inp.parts[-1]
                        if isinstance(default, int):
                            return f"getattr(config, '{field}', {default})"
                        return f"getattr(config, '{field}', 0)"
                elif node.op.name == "_config_float":
                    path_inp = node.inputs[0] if node.inputs else None
                    default = _literal_value(node.inputs[1], None) if len(node.inputs) >= 2 else None
                    if isinstance(path_inp, GraphPath) and path_inp.parts:
                        field = path_inp.parts[-1]
                        if isinstance(default, (int, float)):
                            return f"getattr(config, '{field}', {default})"
                        return f"getattr(config, '{field}', 0.0)"
        return None

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
            const_val = self._resolve_const_value(name)
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
            "core.binary.+": "+",
            "core.binary.%": "%",
            "core.binary.==": "==",
            "core.binary.>=": ">=",
            "core.binary.and": "and",
        }
        if op in bin_ops and len(node.inputs) >= 2:
            left = self._trace_dim_expr(node.inputs[0], repeated_mod, loop_var, visited)
            right = self._trace_dim_expr(node.inputs[1], repeated_mod, loop_var, visited)
            if left and right:
                return f"({left} {bin_ops[op]} {right})"
        elif op == "core.select" and len(node.inputs) >= 3:
            cond = self._trace_dim_expr(node.inputs[0], repeated_mod, loop_var, visited)
            true_val = self._trace_dim_expr(node.inputs[1], repeated_mod, loop_var, visited)
            false_val = self._trace_dim_expr(node.inputs[2], repeated_mod, loop_var, visited)
            if cond and true_val and false_val:
                return f"({true_val} if {cond} else {false_val})"
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
            "core.binary.+": "+",
            "core.binary.%": "%",
            "core.binary.==": "==",
            "core.binary.>=": ">=",
            "core.binary.and": "and",
        }
        if op in bin_ops and len(expr.inputs) >= 2:
            left = self._trace_dim_expr(expr.inputs[0], repeated_mod, loop_var, visited)
            right = self._trace_dim_expr(expr.inputs[1], repeated_mod, loop_var, visited)
            if left and right:
                return f"({left} {bin_ops[op]} {right})"
        elif op == "core.select" and len(expr.inputs) >= 3:
            cond = self._trace_dim_expr(expr.inputs[0], repeated_mod, loop_var, visited)
            true_val = self._trace_dim_expr(expr.inputs[1], repeated_mod, loop_var, visited)
            false_val = self._trace_dim_expr(expr.inputs[2], repeated_mod, loop_var, visited)
            if cond and true_val and false_val:
                return f"({true_val} if {cond} else {false_val})"
            return None

    def _detect_head_dim_expr(self, repeated_mod: Any, loop_var: str = "i") -> str | None:
        """Detect per-layer head_dim expression from core.select in repeated module."""
        for node in repeated_mod.nodes:
            if node.op.name == "core.select" and len(node.inputs) >= 3:
                true_val = _literal_value(node.inputs[1], None)
                false_val = _literal_value(node.inputs[2], None)
                if isinstance(true_val, int) and isinstance(false_val, int) and true_val != false_val:
                    cond = self._trace_dim_expr(node.inputs[0], repeated_mod, loop_var)
                    if cond:
                        return f"({true_val} if {cond} else {false_val})"
        return None

    @staticmethod
    def _node_module_name(node: Any) -> str:
        nid = getattr(node, "id", "")
        return nid.rsplit(":", 1)[0] if ":" in nid else ""

    def _is_repeated_node(self, node: Any) -> bool:
        nid = getattr(node, "id", "")
        for mod_name in self._vllm_classification.repeated_module_names:
            if nid == mod_name or nid.startswith(mod_name + ":"):
                return True
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

    def _vllm_attr_access(self, node: Any) -> str:
        attr = self._vllm_layer_attr_name(node)
        if self._is_repeated_node(node):
            idx = self._node_loop_index(node)
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
            if layer_type != VLLMLayerType.DEFAULT:
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
        attr = self._vllm_attr_access(node)
        local_dict = {k: k for k in local} if isinstance(local, set) else local
        args = self._collect_args(node, local_dict)

        if layer_type == VLLMLayerType.VOCAB_PARALLEL_EMBEDDING:
            if len(args) < 2:
                return False
            expr = f"{attr}({args[1]})"
        elif layer_type == VLLMLayerType.QKV_PARALLEL_LINEAR:
            if len(args) < 2:
                return False
            expr = f"{attr}({args[1]})[0]"
        elif layer_type in (VLLMLayerType.COLUMN_PARALLEL_LINEAR, VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR):
            if len(args) < 2:
                return False
            expr = f"{attr}({args[1]})[0]"
        elif layer_type == VLLMLayerType.ROW_PARALLEL_LINEAR:
            if len(args) < 2:
                return False
            expr = f"{attr}({args[1]})[0]"
        elif layer_type == VLLMLayerType.PARALLEL_LM_HEAD:
            if len(args) < 2:
                return False
            expr = f"{attr}({args[1]})"
        elif layer_type == VLLMLayerType.ATTENTION:
            if len(args) >= 3:
                expr = f"{attr}({args[0]}, {args[1]}, {args[2]}, attn_metadata=self._attn_metadata)"
            else:
                expr = f"{attr}(attn_metadata=self._attn_metadata)"
        elif layer_type == VLLMLayerType.RMSNORM:
            if len(args) < 1:
                return False
            expr = f"{attr}({args[0]})"
        elif layer_type == VLLMLayerType.LAYERNORM:
            if len(args) < 1:
                return False
            expr = f"{attr}({args[0]})"
        else:
            return False

        joined = ", ".join(targets)
        add(lines, indent, f"{joined} = {expr}")
        return True

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

    def _detect_rope_variants(self) -> tuple[str, str, str, str, str] | None:
        """Detect per-layer RoPE variants (local vs full) from top-level module.

        Returns (local_hd, local_theta, full_hd, full_theta, rope_period_expr) or None.
        """
        main_module = self.modules_by_name.get(self.program.main_module)
        if main_module is None:
            return None
        rope_calls = []
        for node in main_module.nodes:
            if "rope_proportional" in node.op.name.lower() or "rope" in node.op.name.lower():
                if len(node.inputs) >= 3:
                    hd = _literal_value(node.inputs[1], None)
                    theta = _literal_value(node.inputs[2], None)
                    if isinstance(hd, int) and isinstance(theta, (int, float)):
                        rope_calls.append((int(hd), float(theta)))
        if len(rope_calls) < 2:
            return None
        rope_calls.sort()
        local_hd, local_theta = rope_calls[0]
        full_hd, full_theta = rope_calls[-1]
        if local_hd == full_hd:
            return None
        rope_period = self._resolve_const_value("ROPE_PERIOD")
        if rope_period is None:
            rope_period = "6"
        return (str(local_hd), repr(local_theta), str(full_hd), repr(full_theta), rope_period)

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

    def _detect_activation(self, repeated_mod: Any) -> str | None:
        for node in repeated_mod.nodes:
            op_name = node.op.name
            if op_name.startswith("Activations.") or op_name.startswith("_activations_"):
                return op_name
        return None

    @staticmethod
    def _activation_to_code(act_name: str) -> str:
        if act_name.endswith("gelu_pytorch_tanh"):
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
            if ":inl:" in node.id:
                continue
            if classification.node_types.get(node.id) == VLLMLayerType.RMSNORM:
                if node.id in classification.qk_norm_node_ids:
                    continue
                idx = int(node.id.rsplit(":", 1)[-1]) if ":" in node.id else 0
                norms.append((idx, node.id))

        norms.sort(key=lambda x: x[0])
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

        for add_id in add_nodes:
            consumers = add_to_consumers.get(add_id, [])
            if not consumers:
                continue
            # Case 1: a norm consumes the add's output → that norm is fused
            for c in consumers:
                if c in norm_ids:
                    uses_residual_set.add(c)
                    fused_set.add(c)
            # Case 2: a norm produces the add's input → that norm uses residual (non-fused)
            for inp in self._find_node_by_id(add_id).inputs:
                if hasattr(inp, "name") and inp.name in value_to_node:
                    producer = value_to_node[inp.name]
                    if producer in norm_ids:
                        uses_residual_set.add(producer)

        result: list[tuple[str, str, bool, bool]] = []
        for _, nid in norms:
            node = self._find_node_by_id(nid)
            attr = self._vllm_attr_access(node)
            result.append((nid, attr, nid in uses_residual_set, nid in fused_set))
        return result

    def _find_o_proj_id(
        self,
        repeated_mod: Any,
        classification: VLLMLayerClassification,
    ) -> str | None:
        ffn_down_ids = {g.down_node_id for g in classification.ffn_groups}
        for node in repeated_mod.nodes:
            if classification.node_types.get(node.id) == VLLMLayerType.ROW_PARALLEL_LINEAR:
                if node.id not in ffn_down_ids:
                    return node.id
        return None

    def _emit_common(self, lines: list[str]) -> None:
        add = self._add
        classification = self._vllm_classification
        indent = 4

        add(lines, indent, '"""Generated vLLM model from Axon Graph IR."""')
        add(lines, indent, "")
        if classification.qkv_groups:
            add(lines, indent, "packed_modules_mapping = {")
            add(lines, indent + 1, '"qkv_proj": ["q_proj", "k_proj", "v_proj"],')
            add(lines, indent, "}")
            add(lines, indent, "")
        add(lines, indent, "def __init__(self, vllm_config, prefix: str = ''):")
        add(lines, indent * 2, "super().__init__()")
        add(lines, indent * 2, "config = vllm_config.model_config.hf_config")
        add(lines, indent * 2, "self.vllm_config = vllm_config")
        add(lines, indent * 2, "self._prefix = prefix")
        add(lines, indent * 2, "quant_config = getattr(vllm_config, 'quant_config', None)")
        add(lines, indent * 2, "params_dtype = vllm_config.model_config.dtype")
        add(lines, indent * 2, "cache_config = getattr(vllm_config, 'cache_config', None)")
        add(lines, indent * 2, "from vllm.distributed import get_tensor_model_parallel_world_size")
        add(lines, indent * 2, "_tp_size = get_tensor_model_parallel_world_size()")
        add(lines, indent * 2, "_num_heads = config.num_attention_heads // _tp_size")
        add(lines, indent * 2, "_num_kv_heads = max(1, getattr(config, 'num_key_value_heads', config.num_attention_heads) // _tp_size)")
        add(lines, indent * 2, "self._model_config = dict(_MODEL_CONFIG)")
        add(lines, indent * 2, "self.config = config")
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
        add(lines, indent * 2, "from vllm.model_executor.layers.layernorm import RMSNorm, LayerNorm")
        add(lines, indent * 2, "")

        self._emit_vllm_layer_inits(lines, classification, indent * 2)

        # --- RoPE (rotary embedding) ---
        repeated_mod = self._get_repeated_module()
        has_rope = False
        if repeated_mod is not None:
            has_rope = self._detect_rope(repeated_mod) is not None
        if has_rope:
            num_layers_expr = self._config_expr("num_hidden_layers")
            head_dim_expr = self._config_expr("head_dim")
            add(lines, indent * 2, "from vllm.model_executor.layers.rotary_embedding import get_rope")
            rope_variants = self._detect_rope_variants()
            if rope_variants:
                local_hd, local_theta, full_hd, full_theta, cond_expr = rope_variants
                add(lines, indent * 2, f"_rope_period = {cond_expr}")
                add(lines, indent * 2, "self.rotary_emb = nn.ModuleList([")
                add(lines, indent * 3, "get_rope(")
                add(lines, indent * 4, f"({full_hd} if ((i + 1) % _rope_period == 0) else {local_hd}),")
                add(lines, indent * 4, "max_position=getattr(config, 'max_position_embeddings', 4096),")
                add(lines, indent * 4, "is_neox_style=True,")
                add(lines, indent * 4, f"rope_parameters={{'rope_type': 'default', 'rope_theta': ({full_theta} if ((i + 1) % _rope_period == 0) else {local_theta})}},")
                add(lines, indent * 3, f") for i in range({num_layers_expr})")
                add(lines, indent * 2, "])")
            else:
                add(lines, indent * 2, "_rope_theta = getattr(config, 'rope_theta',")
                add(lines, indent * 3, "self._model_config.get('rope_theta', 10000.0))")
                add(lines, indent * 2, "_rope_params = {'rope_type': 'default', 'rope_theta': _rope_theta}")
                add(lines, indent * 2, "self.rotary_emb = nn.ModuleList([")
                add(lines, indent * 3, f"get_rope({head_dim_expr},")
                add(lines, indent * 3, "max_position=getattr(config, 'max_position_embeddings', 4096),")
                add(lines, indent * 3, "is_neox_style=True,")
                add(lines, indent * 3, "rope_parameters=_rope_params,")
                add(lines, indent * 3, f") for i in range({num_layers_expr})")
                add(lines, indent * 2, "])")
            add(lines, indent * 2, "")

        # --- Per-layer scalar ---
        if classification.per_layer_scalar_node_id:
            num_layers_expr = self._config_expr("num_hidden_layers")
            add(lines, indent * 2, "import torch")
            add(lines, indent * 2, f"self.layer_scalars = nn.ParameterList([")
            add(lines, indent * 3, "nn.Parameter(torch.ones(1))")
            add(lines, indent * 3, f"for _ in range({num_layers_expr})")
            add(lines, indent * 2, "])")
            add(lines, indent * 2, "")

        # --- lm_head and logits_processor ---
        if classification.lm_head_node_id:
            lm_head_node = self._find_node_by_id(classification.lm_head_node_id)
            lm_head_attr = self._vllm_attr_access(lm_head_node) if lm_head_node else None
            if lm_head_attr:
                add(lines, indent * 2, f"self.lm_head = {lm_head_attr}")
                # Handle tied word embeddings
                add(lines, indent * 2, "if getattr(config, 'tie_word_embeddings', False):")
                for emb_id in classification.embedding_node_ids:
                    if emb_id == classification.pli_embed_node_id:
                        continue
                    emb_node = self._find_node_by_id(emb_id)
                    if emb_node is not None:
                        emb_attr = self._vllm_attr_access(emb_node)
                        add(lines, indent * 3, f"self.lm_head = self.lm_head.tie_weights({emb_attr})")
                        break
        add(lines, indent * 2, "from vllm.model_executor.layers.logits_processor import LogitsProcessor")
        add(lines, indent * 2, "_soft_cap = getattr(config, 'final_logit_softcapping', None)")
        add(lines, indent * 2, "self.logits_processor = LogitsProcessor(")
        add(lines, indent * 3, f"{self._config_expr('vocab_size')},")
        add(lines, indent * 3, "soft_cap=_soft_cap,")
        add(lines, indent * 2, ")")
        add(lines, indent * 2, "")

        add(lines, indent, "")
        add(lines, indent, "def embed_input_ids(self, input_ids):")
        for node_id in sorted(classification.embedding_node_ids):
            if node_id == classification.pli_embed_node_id:
                continue
            node = self._find_node_by_id(node_id)
            if node is not None:
                attr = self._vllm_attr_access(node)
                add(lines, indent * 2, f"return {attr}(input_ids)")
                break
        add(lines, indent * 2, "return None")
        add(lines, indent, "")

        add(lines, indent, "def load_weights(self, weights):")
        add(lines, indent * 2, "stacked_params_mapping = [")
        add(lines, indent * 3, '(".qkv_proj", ".q_proj", "q"),')
        add(lines, indent * 3, '(".qkv_proj", ".k_proj", "k"),')
        add(lines, indent * 3, '(".qkv_proj", ".v_proj", "v"),')
        add(lines, indent * 2, "]")
        add(lines, indent * 2, "params_dict = dict(self.named_parameters())")
        add(lines, indent * 2, "params_dict.update(dict(self.named_buffers()))")
        add(lines, indent * 2, "loaded_params = set()")
        add(lines, indent * 2, "")
        add(lines, indent * 2, "for name, loaded_weight in weights:")
        self._emit_weight_loading_body(lines, classification, indent * 3)
        add(lines, indent, "")

        add(lines, indent, "def compute_logits(self, hidden_states):")
        add(lines, indent * 2, "return self.logits_processor(self.lm_head, hidden_states)")
        add(lines, indent, "")
        add(lines, indent, "def make_empty_intermediate_tensors(")
        add(lines, indent * 2, "self, batch_size, dtype, device,")
        add(lines, indent * 2, "):")
        add(lines, indent * 2, "return {}")
        add(lines, indent, "")

    def _emit_forward(self, lines: list[str]) -> None:
        add = self._add
        cls = self._vllm_classification

        repeated_mod_name = max(
            cls.repeated_module_names,
            key=lambda n: len(self.modules_by_name[n].nodes) if n in self.modules_by_name else 0,
        ) if cls.repeated_module_names else None
        repeated_mod = self.modules_by_name.get(repeated_mod_name) if repeated_mod_name else None

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

        # Find final norm (non-repeated RMSNorm)
        final_norm_attr = None
        for nid in sorted(cls.rmsnorm_node_ids):
            mod_name = self._node_module_name(self._find_node_by_id(nid))
            if mod_name not in cls.repeated_module_names:
                node = self._find_node_by_id(nid)
                final_norm_attr = self._vllm_attr_access(node)
                break

        # --- Config expressions ---
        num_layers_expr = self._config_expr("num_hidden_layers")
        num_heads_expr = self._config_expr("num_attention_heads")
        num_kv_heads_expr = self._config_expr("num_key_value_heads", alt="num_attention_heads")
        head_dim_expr = self._config_expr("head_dim")

        # --- Generate forward ---
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
        add(lines, 12, "hidden_states = self.embed_input_ids(input_ids)")
        add(lines, 8, "")
        add(lines, 8, "self._attn_metadata = attn_metadata")
        add(lines, 8, "self._positions = positions")
        add(lines, 8, "config = self.config")
        add(lines, 8, "")
        add(lines, 8, "from vllm.distributed import get_tensor_model_parallel_world_size")
        add(lines, 8, "_tp_size = get_tensor_model_parallel_world_size()")
        add(lines, 8, f"_num_heads = {num_heads_expr} // _tp_size")
        add(lines, 8, f"_num_kv_heads = max(1, {num_kv_heads_expr} // _tp_size)")
        add(lines, 8, f"_head_dim = {head_dim_expr}")
        add(lines, 8, f"_q_size = _num_heads * _head_dim")
        add(lines, 8, f"_kv_size = _num_kv_heads * _head_dim")
        add(lines, 8, "")

        # Detect per-layer head dim variation
        hd_expr = None
        if repeated_mod is not None:
            hd_expr = self._detect_head_dim_expr(repeated_mod, "i")

        has_per_layer_scalar = cls.per_layer_scalar_node_id is not None
        has_pli = cls.pli_gate_node_id is not None
        any_fused = any(f for _, _, _, f in layer_norms) and not has_pli

        # PLI computation (before the loop)
        if has_pli:
            pli_embed_node = self._find_node_by_id(cls.pli_embed_node_id) if cls.pli_embed_node_id else None
            pli_model_proj_node = self._find_node_by_id(cls.pli_model_proj_node_id) if cls.pli_model_proj_node_id else None
            pli_proj_norm_node = self._find_node_by_id(cls.pli_proj_norm_node_id) if cls.pli_proj_norm_node_id else None
            if pli_embed_node and pli_model_proj_node and pli_proj_norm_node:
                pli_embed_attr = self._vllm_attr_access(pli_embed_node)
                pli_model_proj_attr = self._vllm_attr_access(pli_model_proj_node)
                pli_proj_norm_attr = self._vllm_attr_access(pli_proj_norm_node)
                pli_dim = self._resolve_const_value("PLI") or "getattr(config, 'per_layer_input_dim', 256)"
                add(lines, 8, f"_pli_dim = {pli_dim}")
                add(lines, 8, f"_per_layer_inputs = {pli_embed_attr}(input_ids)")
                add(lines, 8, f"_per_layer_inputs = _per_layer_inputs.unflatten(-1, ({num_layers_expr}, _pli_dim))")
                add(lines, 8, f"_pli_proj = {pli_model_proj_attr}(hidden_states)[0]")
                add(lines, 8, f"_pli_proj = _pli_proj.unflatten(-1, ({num_layers_expr}, _pli_dim))")
                add(lines, 8, f"_pli_proj = {pli_proj_norm_attr}(_pli_proj)")
                add(lines, 8, f"_per_layer_inputs = (_pli_proj + _per_layer_inputs)")
                add(lines, 8, "")

        add(lines, 8, f"residual = None")
        add(lines, 8, f"for i in range({num_layers_expr}):")

        indent = 12

        if hd_expr:
            add(lines, indent, f"_hd = {hd_expr}")
            add(lines, indent, f"_q_size = _num_heads * _hd")
            add(lines, indent, f"_kv_size = _num_kv_heads * _hd")

        # Determine which norm index is the pre-FFN norm
        ffn_norm_idx = len(layer_norms) - 1
        if ffn_group and ffn_group.gate_node_id:
            gate_idx = int(ffn_group.gate_node_id.rsplit(":", 1)[-1]) if ":" in ffn_group.gate_node_id else 0
            for i, (nid, _, _, _) in enumerate(layer_norms):
                nidx = int(nid.rsplit(":", 1)[-1]) if ":" in nid else 0
                if nidx < gate_idx:
                    ffn_norm_idx = i

        if not any_fused:
            # --- Gemma4 non-fused residual pattern ---
            for norm_i, (nid, attr, uses_res, fused) in enumerate(layer_norms):
                if norm_i == 0:
                    add(lines, indent, f"residual = hidden_states")
                    add(lines, indent, f"hidden_states = {attr}(hidden_states)")
                else:
                    add(lines, indent, f"hidden_states = {attr}(hidden_states)")
                    if uses_res and not fused:
                        add(lines, indent, f"hidden_states = hidden_states + residual")
                        add(lines, indent, f"residual = hidden_states")

                if norm_i == 0 and qkv_group:
                    self._emit_attn_block(lines, indent, cls, qkv_group, rope_id, o_proj_id)
                if norm_i == ffn_norm_idx and ffn_group:
                    self._emit_ffn_block(lines, indent, cls, ffn_group, act_name)

            if has_pli:
                self._emit_pli_block(lines, indent, cls)
            if has_per_layer_scalar:
                add(lines, indent, f"hidden_states = hidden_states * self.layer_scalars[i]")
        else:
            # --- Gemma3 fused residual pattern ---
            for norm_i, (nid, attr, uses_res, _) in enumerate(layer_norms):
                if norm_i == 0:
                    add(lines, indent, f"if residual is None:")
                    add(lines, indent + 4, f"residual = hidden_states")
                    add(lines, indent + 4, f"hidden_states = {attr}(hidden_states)")
                    add(lines, indent, f"else:")
                    add(lines, indent + 4, f"hidden_states, residual = {attr}(hidden_states, residual)")
                elif uses_res:
                    add(lines, indent, f"hidden_states, residual = {attr}(hidden_states, residual)")
                else:
                    add(lines, indent, f"hidden_states = {attr}(hidden_states)")

                if norm_i == 0 and qkv_group:
                    self._emit_attn_block(lines, indent, cls, qkv_group, rope_id, o_proj_id)
                if norm_i == ffn_norm_idx and ffn_group:
                    self._emit_ffn_block(lines, indent, cls, ffn_group, act_name)

            if has_pli:
                self._emit_pli_block(lines, indent, cls)
            if has_per_layer_scalar:
                add(lines, indent, f"hidden_states = hidden_states * self.layer_scalars[i]")

        # Final norm
        if final_norm_attr:
            if any_fused:
                add(lines, 8, f"hidden_states, _ = {final_norm_attr}(hidden_states, residual)")
            else:
                add(lines, 8, f"hidden_states = {final_norm_attr}(hidden_states)")

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

        # Fused QKV projection
        add(lines, indent, f"qkv, _ = {q_attr}(hidden_states)")
        add(lines, indent, f"q, k, v = qkv.split([_q_size, _kv_size, _kv_size], dim=-1)")

        # QK norms with reshape
        qk_norms = sorted(cls.qk_norm_node_ids)
        has_var_hd = self._detect_head_dim_expr(self._get_repeated_module() or repeated_mod, "i") is not None if self._get_repeated_module() else False
        head_dim_ref = "_hd" if has_var_hd else "_head_dim"

        q_top = qkv_group.q_node_id.split(":", 1)[0] if ":" in qkv_group.q_node_id else ""
        k_top = qkv_group.k_node_id.split(":", 1)[0] if ":" in qkv_group.k_node_id else ""

        for nid in qk_norms:
            node = self._find_node_by_id(nid)
            attr = self._vllm_attr_access(node)
            norm_top = nid.split(":", 1)[0] if ":" in nid else ""
            if norm_top == k_top and norm_top != q_top:
                add(lines, indent, f"k = k.unflatten(-1, (_num_kv_heads, {head_dim_ref}))")
                add(lines, indent, f"k = {attr}(k)")
                add(lines, indent, f"k = k.flatten(-2, -1)")
            else:
                add(lines, indent, f"q = q.unflatten(-1, (_num_heads, {head_dim_ref}))")
                add(lines, indent, f"q = {attr}(q)")
                add(lines, indent, f"q = q.flatten(-2, -1)")

        # RoPE
        if rope_id:
            add(lines, indent, f"q, k = self.rotary_emb[i](self._positions, q, k)")

        # Attention
        if attn_node:
            attn_attr = self._vllm_attr_access(attn_node)
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
    ) -> None:
        add = self._add

        if ffn_group.gate_node_id and ffn_group.up_node_id:
            gate_node = self._find_node_by_id(ffn_group.gate_node_id)
            up_node = self._find_node_by_id(ffn_group.up_node_id)
            gate_attr = self._vllm_attr_access(gate_node)
            up_attr = self._vllm_attr_access(up_node)
            add(lines, indent, f"gate = {gate_attr}(hidden_states)[0]")
            add(lines, indent, f"up = {up_attr}(hidden_states)[0]")
            if act_name:
                act_code = self._activation_to_code(act_name)
                add(lines, indent, f"hidden_states = {act_code.format(x='gate')} * up")
            else:
                add(lines, indent, f"hidden_states = F.silu(gate) * up")
        elif ffn_group.up_node_id:
            up_node = self._find_node_by_id(ffn_group.up_node_id)
            up_attr = self._vllm_attr_access(up_node)
            add(lines, indent, f"hidden_states = {up_attr}(hidden_states)[0]")
            if act_name:
                act_code = self._activation_to_code(act_name)
                add(lines, indent, f"hidden_states = {act_code.format(x='hidden_states')}")

        if ffn_group.down_node_id:
            down_node = self._find_node_by_id(ffn_group.down_node_id)
            down_attr = self._vllm_attr_access(down_node)
            add(lines, indent, f"hidden_states = {down_attr}(hidden_states)[0]")

    def _emit_forward_legacy(self, lines: list[str]) -> None:
        add = self._add
        main = self.modules_by_name[self.program.main_module]
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
        add(lines, 12, "hidden_states = self.embed_input_ids(input_ids)")
        add(lines, 8, "")
        add(lines, 8, "self._attn_metadata = attn_metadata")
        add(lines, 8, "self._positions = positions")
        add(lines, 8, "")
        add(lines, 8, f"result = self.{self.method_names[main.name]}(input_ids=input_ids)")
        add(lines, 8, "return result")

    def _emit_vllm_layer_inits(
        self,
        lines: list[str],
        classification: VLLMLayerClassification,
        indent: int,
    ) -> None:
        add = self._add
        seen = set()
        num_layers_expr = self._config_expr("num_hidden_layers")
        skip_node_ids: set[str] = set()
        use_clean_forward = False
        if classification.repeated_module_names:
            _repeated_mod_name = max(
                classification.repeated_module_names,
                key=lambda n: len(self.modules_by_name[n].nodes) if n in self.modules_by_name else 0,
            )
            _repeated_mod = self.modules_by_name.get(_repeated_mod_name)
            if _repeated_mod is not None:
                _layer_norms = self._analyze_layer_norms(_repeated_mod, classification)
                if _layer_norms:
                    use_clean_forward = True
        self._use_clean_forward = use_clean_forward
        if use_clean_forward:
            for g in classification.qkv_groups:
                if g.q_node_id != g.k_node_id:
                    skip_node_ids.add(g.k_node_id)
                    skip_node_ids.add(g.v_node_id)
        for module in self.program.modules:
            for node in module.nodes:
                if node.id in seen:
                    continue
                if node.id in skip_node_ids:
                    continue
                layer_type = classification.layer_type(node)
                if layer_type == VLLMLayerType.DEFAULT:
                    continue
                seen.add(node.id)
                attr_name = self._vllm_layer_attr_name(node)
                is_repeated = self._is_repeated_node(node)
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
            vocab = self._config_expr("vocab_size")
            if node.id == self._vllm_classification.pli_embed_node_id:
                num_layers = self._config_expr("num_hidden_layers")
                pli_dim = self._resolve_const_value("PLI") or f"getattr(config, 'per_layer_input_dim', 256)"
                dim = f"({num_layers} * {pli_dim})"
            else:
                dim = self._config_expr("hidden_size")
            add(lines, indent, "VocabParallelEmbedding(")
            add(lines, indent + 1, f"{vocab}, {dim},")
            add(lines, indent + 1, f"prefix={prefix},")
            add(lines, indent, ")")
        elif layer_type == VLLMLayerType.PARALLEL_LM_HEAD:
            vocab = self._config_expr("vocab_size")
            dim = self._config_expr("hidden_size")
            add(lines, indent, "ParallelLMHead(")
            add(lines, indent + 1, f"{vocab}, {dim},")
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
                classification.pli_model_proj_node_id,
            }
            pli_linear_ids.discard(None)
            in_dim = self._node_input_dim_expr(node)
            out_dim = self._node_output_dim_expr(node)
            bias = _bool_arg(node, 3)
            if node.id in pli_linear_ids:
                add(lines, indent, "ReplicatedLinear(")
                add(lines, indent + 1, f"{in_dim}, {out_dim},")
                add(lines, indent + 1, f"bias={bias}, skip_bias_add=True,")
                add(lines, indent + 1, f"prefix={prefix},")
                add(lines, indent + 1, "params_dtype=params_dtype,")
                add(lines, indent, ")")
            elif layer_type == VLLMLayerType.QKV_PARALLEL_LINEAR:
                self._emit_qkv_init(lines, node, classification, indent)
            elif layer_type in (
                VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR,
                VLLMLayerType.COLUMN_PARALLEL_LINEAR,
            ):
                add(lines, indent, "ColumnParallelLinear(")
                add(lines, indent + 1, f"{in_dim}, {out_dim},")
                add(lines, indent + 1, f"bias={bias}, skip_bias_add=True,")
                add(lines, indent + 1, f"prefix={prefix},")
                add(lines, indent + 1, "quant_config=quant_config,")
                add(lines, indent + 1, "params_dtype=params_dtype,")
                add(lines, indent, ")")
            else:
                add(lines, indent, "RowParallelLinear(")
                add(lines, indent + 1, f"{in_dim}, {out_dim},")
                add(lines, indent + 1, f"bias={bias}, skip_bias_add=True,")
                add(lines, indent + 1, f"prefix={prefix},")
                add(lines, indent + 1, "quant_config=quant_config,")
                add(lines, indent + 1, "params_dtype=params_dtype,")
                add(lines, indent, ")")
        elif layer_type == VLLMLayerType.ATTENTION:
            scale = self._config_expr("query_pre_attn_scalar", alt="head_dim")
            head_size = self._config_expr("head_dim", alt="hidden_size")
            if self._is_repeated_node(node):
                repeated_mod = self._get_repeated_module()
                if repeated_mod is not None:
                    loop_var = self._node_loop_index(node)
                    hd_expr = self._detect_head_dim_expr(repeated_mod, loop_var)
                    if hd_expr:
                        head_size = hd_expr
            add(lines, indent, "Attention(")
            add(lines, indent + 1, "_num_heads,")
            add(lines, indent + 1, f"{head_size},")
            add(lines, indent + 1, f"scale=1.0 / (float({scale}) ** 0.5),")
            add(lines, indent + 1, "num_kv_heads=_num_kv_heads,")
            add(lines, indent + 1, "cache_config=cache_config,")
            add(lines, indent + 1, "quant_config=quant_config,")
            add(lines, indent + 1, "logits_soft_cap=getattr(config, 'attn_logit_softcapping', None),")
            add(lines, indent + 1, "per_layer_sliding_window=(")
            add(lines, indent + 2, "getattr(config, 'sliding_window', None)")
            add(lines, indent + 2, "if (getattr(config, 'sliding_window_pattern', 0) and (i + 1) % getattr(config, 'sliding_window_pattern', 0) != 0)")
            add(lines, indent + 2, "else None),")
            add(lines, indent + 1, f"prefix={prefix},")
            add(lines, indent, ")")
        elif layer_type == VLLMLayerType.RMSNORM:
            if node.id in self._vllm_classification.qk_norm_node_ids:
                dim = self._config_expr("head_dim")
                if self._is_repeated_node(node):
                    repeated_mod = self._get_repeated_module()
                    if repeated_mod is not None:
                        loop_var = self._node_loop_index(node)
                        hd_expr = self._detect_head_dim_expr(repeated_mod, loop_var)
                        if hd_expr:
                            dim = hd_expr
            elif node.id == self._vllm_classification.pli_proj_norm_node_id:
                dim = self._resolve_const_value("PLI") or f"getattr(config, 'per_layer_input_dim', 256)"
            else:
                dim = self._config_expr("hidden_size")
            eps = self._node_rmsnorm_eps(node)
            add(lines, indent, "RMSNorm(")
            add(lines, indent + 1, f"{dim}, eps={eps},")
            add(lines, indent, ")")
        elif layer_type == VLLMLayerType.LAYERNORM:
            dim = self._config_expr("hidden_size")
            eps = self._node_layernorm_eps(node)
            add(lines, indent, "LayerNorm(")
            add(lines, indent + 1, f"{dim}, eps={eps},")
            add(lines, indent, ")")

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
            bias = _bool_arg(node, 3)
            prefix = self._layer_prefix(node)
            add(lines, indent, "ColumnParallelLinear(")
            add(lines, indent + 1, f"{in_dim}, {out_dim},")
            add(lines, indent + 1, f"bias={bias}, skip_bias_add=True,")
            add(lines, indent + 1, f"prefix={prefix},")
            add(lines, indent + 1, "quant_config=quant_config,")
            add(lines, indent + 1, "params_dtype=params_dtype,")
            add(lines, indent, ")")
            return

        hidden_size = self._node_input_dim_expr(node)
        head_size = self._config_expr("head_dim", alt="hidden_size")
        if self._is_repeated_node(node):
            repeated_mod = self._get_repeated_module()
            if repeated_mod is not None:
                loop_var = self._node_loop_index(node)
                hd_expr = self._detect_head_dim_expr(repeated_mod, loop_var)
                if hd_expr:
                    head_size = hd_expr
        total_num_heads = self._config_expr("num_attention_heads")
        total_num_kv_heads = self._config_expr("num_key_value_heads", alt="num_attention_heads")
        bias = _bool_arg(node, 3)

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
        add(lines, indent + 1, f"{hidden_size}, {head_size},")
        add(lines, indent + 1, f"{total_num_heads}, {total_num_kv_heads},")
        add(lines, indent + 1, f"bias={bias}, skip_bias_add=True,")
        add(lines, indent + 1, f"prefix={qkv_prefix},")
        add(lines, indent + 1, "quant_config=quant_config,")
        add(lines, indent + 1, "params_dtype=params_dtype,")
        add(lines, indent, ")")

    def _qkv_layer_prefix(self, q_node: GraphNode) -> str:
        """Compute prefix for fused QKV layer from Q node's path."""
        if not self._is_repeated_node(q_node):
            base = _linear_base_key(q_node)
            if base and "." in base:
                return repr(base.rsplit(".", 1)[0] + ".qkv_proj")
            return repr("qkv_proj")
        mod_name = self._node_module_name(q_node)
        scope_parts = self._vllm_classification.module_scope_parts.get(mod_name)
        if scope_parts is None:
            return repr("qkv_proj")
        base = _linear_base_key(q_node)
        if base:
            sub_parts = base.split(".")
            if sub_parts and sub_parts[0] == "{__scope}":
                full_parts = list(scope_parts) + sub_parts[1:]
            else:
                full_parts = sub_parts
        else:
            full_parts = list(scope_parts) + ["self_attn"]
        if full_parts:
            full_parts[-1] = "qkv_proj"
        else:
            full_parts = ["qkv_proj"]
        fparts: list[str] = []
        for p in full_parts:
            if p == "{__scope}":
                fparts.append("{prefix}")
            else:
                fparts.append(p)
        return f'f"{".".join(fparts)}"'

    def _emit_weight_loading_body(
        self,
        lines: list[str],
        classification: VLLMLayerClassification,
        indent: int,
    ) -> None:
        add = self._add
        add(lines, indent, "for _pname, _wname, _sid in stacked_params_mapping:")
        add(lines, indent + 1, "if _wname not in name:")
        add(lines, indent + 2, "continue")
        add(lines, indent + 1, "name = name.replace(_wname, _pname)")
        add(lines, indent + 1, "if name not in params_dict:")
        add(lines, indent + 2, "break")
        add(lines, indent + 1, "param = params_dict[name]")
        add(lines, indent + 1, "weight_loader = getattr(param, 'weight_loader', None)")
        add(lines, indent + 1, "if weight_loader is not None:")
        add(lines, indent + 2, "weight_loader(param, loaded_weight, _sid)")
        add(lines, indent + 1, "else:")
        add(lines, indent + 2, "param.data.copy_(loaded_weight)")
        add(lines, indent + 1, "loaded_params.add(name)")
        add(lines, indent + 1, "break")
        add(lines, indent, "else:")
        add(lines, indent + 1, "if name not in params_dict:")
        add(lines, indent + 2, "continue")
        add(lines, indent + 1, "param = params_dict[name]")
        add(lines, indent + 1, "weight_loader = getattr(param, 'weight_loader', None)")
        add(lines, indent + 1, "if weight_loader is not None:")
        add(lines, indent + 2, "weight_loader(param, loaded_weight)")
        add(lines, indent + 1, "else:")
        add(lines, indent + 2, "param.data.copy_(loaded_weight)")
        add(lines, indent + 1, "loaded_params.add(name)")

    def _vllm_layer_attr_name(self, node: GraphNode) -> str:
        return f"_vllm_{_safe_ident(node.id)}"

    def _node_prefix(self, node: GraphNode) -> str:
        base = _linear_base_key(node)
        if not base:
            return node.id
        return base

    def _layer_prefix(self, node: GraphNode) -> str:
        if not self._is_repeated_node(node):
            return repr(self._node_prefix(node))
        mod_name = self._node_module_name(node)
        scope_parts = self._vllm_classification.module_scope_parts.get(mod_name)
        if scope_parts is None:
            return repr(self._node_prefix(node))
        base = _linear_base_key(node)
        if base:
            sub_parts = base.split(".")
            if sub_parts and sub_parts[0] == "{__scope}":
                full_parts = list(scope_parts) + sub_parts[1:]
            else:
                full_parts = sub_parts
        else:
            full_parts = list(scope_parts) + ["self_attn"]
        fparts: list[str] = []
        for p in full_parts:
            if p == "{__scope}":
                fparts.append("{prefix}")
            else:
                fparts.append(p)
        return f'f"{".".join(fparts)}"'

    def _config_expr(self, config_name: str, alt: str | None = None) -> str:
        if alt is not None:
            inner = self._config_expr(alt)
            return (
                f"getattr(config, {config_name!r}, "
                f"self._model_config.get({config_name!r}, {inner}))"
            )
        return (
            f"getattr(config, {config_name!r}, "
            f"self._model_config.get({config_name!r}, 0))"
        )

    def _node_input_dim_expr(self, node: GraphNode) -> str:
        if node.id in self._ffn_down_node_ids:
            # Try extracting per-layer dim from input tensor type first
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
            head_dim = self._config_expr("head_dim")
            return f"({num_heads} * {head_dim})"
        typed = self._node_input_dim_from_type(node)
        if typed is not None:
            return typed
        return self._config_expr("hidden_size")

    def _node_input_dim_from_type(self, node: GraphNode) -> str | None:
        """Extract input dim from the node's input tensor type (last dim if int)."""
        if len(node.inputs) >= 2:
            inp = node.inputs[1]
            inp_dims = getattr(inp, "dims", None) or getattr(getattr(inp, "type_expr", None), "dims", None)
            if inp_dims and len(inp_dims) > 0:
                last = inp_dims[-1]
                if isinstance(last, int):
                    return str(last)
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
            hd_expr = self._detect_head_dim_expr(repeated_mod, loop_var)
            if hd_expr and dim == "hd":
                return hd_expr
            # Try tracing through the module (including cross-module)
            traced = self._trace_dim_expr(GraphValueRef(name=dim, type_expr=None), repeated_mod, loop_var)
            if traced:
                return traced
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
        if self._is_repeated_node(node):
            # Try tracing with the node's own module first, then the primary repeated module
            for mod in self._get_node_modules(node):
                if mod is not None and len(node.inputs) >= 3:
                    loop_var = self._node_loop_index(node)
                    traced = self._trace_dim_expr(node.inputs[2], mod, loop_var)
                    if traced:
                        return traced
        # Fallback: extract output dim from the node's output type
        if node.outputs:
            out = node.outputs[0]
            out_dims = getattr(out, "dims", None) or getattr(getattr(out, "type_expr", None), "dims", None)
            if out_dims and len(out_dims) > 0:
                last = out_dims[-1]
                if isinstance(last, int):
                    return str(last)
        return self._config_expr("hidden_size")

    def _node_rmsnorm_eps(self, node: GraphNode) -> str:
        eps = _literal_value(node.inputs[1], None) if len(node.inputs) >= 2 else None
        if isinstance(eps, (int, float)):
            return repr(float(eps))
        return "1e-6"

    def _node_layernorm_eps(self, node: GraphNode) -> str:
        eps = _literal_value(node.inputs[2], None) if len(node.inputs) >= 3 else None
        if isinstance(eps, (int, float)):
            return repr(float(eps))
        return "1e-5"

    def _find_node_by_id(self, node_id: str) -> GraphNode | None:
        for module in self.program.modules:
            for node in module.nodes:
                if node.id == node_id:
                    return node
        return None

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

        if layer_type == VLLMLayerType.VOCAB_PARALLEL_EMBEDDING:
            attr = self._vllm_attr_access(node)
            args = self._collect_args(node, local)
            return f"{attr}({args[1]})"

        if layer_type == VLLMLayerType.QKV_PARALLEL_LINEAR:
            attr = self._vllm_attr_access(node)
            args = self._collect_args(node, local)
            return f"{attr}({args[1]})[0]"

        if layer_type in (
            VLLMLayerType.COLUMN_PARALLEL_LINEAR,
            VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR,
        ):
            attr = self._vllm_attr_access(node)
            args = self._collect_args(node, local)
            return f"{attr}({args[1]})[0]"

        if layer_type == VLLMLayerType.ROW_PARALLEL_LINEAR:
            attr = self._vllm_attr_access(node)
            args = self._collect_args(node, local)
            return f"{attr}({args[1]})[0]"

        if layer_type == VLLMLayerType.PARALLEL_LM_HEAD:
            attr = self._vllm_attr_access(node)
            args = self._collect_args(node, local)
            return f"{attr}({args[1]})"

        if layer_type == VLLMLayerType.ATTENTION:
            attr = self._vllm_attr_access(node)
            args = self._collect_args(node, local)
            if len(args) >= 3:
                return (
                    f"{attr}({args[0]}, {args[1]}, {args[2]}, "
                    f"attn_metadata=self._attn_metadata)"
                )
            return f"{attr}(attn_metadata=self._attn_metadata)"

        if layer_type == VLLMLayerType.RMSNORM:
            attr = self._vllm_attr_access(node)
            args = self._collect_args(node, local)
            return f"{attr}({args[0]})"

        if layer_type == VLLMLayerType.LAYERNORM:
            attr = self._vllm_attr_access(node)
            args = self._collect_args(node, local)
            return f"{attr}({args[0]})"

        if primitive == "_vllm_paged_attention":
            attr = self._vllm_attr_access(node)
            args = self._collect_args(node, local)
            if len(args) >= 3:
                return (
                    f"{attr}({args[0]}, {args[1]}, {args[2]}, "
                    f"attn_metadata=self._attn_metadata)"
                )
            return f"{attr}(attn_metadata=self._attn_metadata)"

        return super()._primitive_expr(primitive, node, local=local, symbols_dict=symbols_dict)

    def _collect_args(self, node: GraphNode, local: dict[str, str]) -> list[str]:
        args: list[str] = []
        for inp in node.inputs:
            if isinstance(inp, (GraphValueRef, GraphValue)):
                args.append(local.get(inp.name, inp.name))
            elif isinstance(inp, GraphLiteral):
                val = inp.value
                if isinstance(val, bool):
                    args.append(repr(val))
                elif isinstance(val, (int, float)):
                    args.append(repr(val))
                elif isinstance(val, str):
                    args.append(repr(val))
                elif val is None:
                    args.append("None")
                else:
                    args.append(repr(val))
            elif isinstance(inp, GraphPath):
                args.append(repr(_graph_path_key(inp)))
            else:
                args.append(repr(inp))
        return args

    def _emit_generate(self, lines: list[str]) -> None:
        add = self._add
        add(lines, 4, "def generate(self, *args, **kwargs):")
        add(lines, 8, 'raise NotImplementedError("vLLM handles generation externally")')


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
    emitter = _DirectVLLMEmitter(
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
            "",
            f"_MODEL_CONFIG = {model_config!r}",
            "",
            body,
        ]
    )
    return "\n".join(header)


__all__ = [
    "emit_model_code_from_graph_ir",
]
