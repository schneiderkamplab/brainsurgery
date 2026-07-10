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
    _is_linear_call,
    classify_graph_for_vllm,
)


def _graph_path_key(path: GraphPath) -> str:
    return ".".join(path.parts)


_BINOP_SYMBOLS = {
    "core.binary.*": "*",
    "core.binary.+": "+",
    "core.binary.%": "%",
    "core.binary.==": "==",
    "core.binary.>=": ">=",
    "core.binary.and": "and",
    "core.binary.or": "or",
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

    def __init__(self, *args: Any, model_config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._vllm_classification: VLLMLayerClassification = classify_graph_for_vllm(self.program)
        self._ffn_down_node_ids: set[str] = {
            g.down_node_id for g in self._vllm_classification.ffn_groups
        }
        self._use_clean_forward: bool = False
        self._model_config_data: dict[str, Any] = model_config or {}

    def emit(self) -> str:
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
        self._emit_generate(lines)
        return "\n".join(lines)

    def _resolve_const_value(self, name: str) -> str | None:
        """Resolve a top-level constant name to a config expression."""
        for module in self.program.modules:
            if module.name != name:
                continue
            for node in module.nodes:
                if node.op.name in ("_config_int", "_config_dim"):
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
                return _simplify_binop(op, left, right)
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
                return _simplify_binop(op, left, right)
        elif op == "core.select" and len(expr.inputs) >= 3:
            cond = self._trace_dim_expr(expr.inputs[0], repeated_mod, loop_var, visited)
            true_val = self._trace_dim_expr(expr.inputs[1], repeated_mod, loop_var, visited)
            false_val = self._trace_dim_expr(expr.inputs[2], repeated_mod, loop_var, visited)
            if cond and true_val and false_val:
                return f"({true_val} if {cond} else {false_val})"
            return None

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
            expr = args[1]
        elif layer_type == VLLMLayerType.ATTENTION:
            if len(args) >= 3:
                expr = f"{attr}({args[0]}, {args[1]}, {args[2]}, attn_metadata=self._attn_metadata)"
            else:
                expr = f"{attr}(attn_metadata=self._attn_metadata)"
        elif layer_type == VLLMLayerType.RMSNORM:
            if len(args) < 1:
                return False
            data_idx = 1 if (len(node.inputs) >= 2 and isinstance(node.inputs[0], GraphPath)) else 0
            if data_idx >= len(args):
                return False
            expr = f"{attr}({args[data_idx]})"
        elif layer_type == VLLMLayerType.LAYERNORM:
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
        add(lines, indent * 2, "from vllm.distributed import get_tensor_model_parallel_world_size")
        add(lines, indent * 2, "_tp_size = get_tensor_model_parallel_world_size()")
        add(lines, indent * 2, "self._tp_size = _tp_size")
        add(lines, indent * 2, "_num_heads = config.num_attention_heads // _tp_size")
        add(lines, indent * 2, "_num_kv_heads = max(1, getattr(config, 'num_key_value_heads', config.num_attention_heads) // _tp_size)")
        add(lines, indent * 2, "self._model_config = dict(_MODEL_CONFIG or {})")
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

        add(lines, indent * 2, "self.state_dict_tensors = {}")
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
                add(lines, indent * 4, "rope_parameters=(")
                add(lines, indent * 5, "dict(config.rope_parameters[config.layer_types[i]])")
                add(lines, indent * 5, "if hasattr(config, 'rope_parameters') and hasattr(config, 'layer_types')")
                add(lines, indent * 5, "and isinstance(config.rope_parameters, dict) and i < len(config.layer_types)")
                add(lines, indent * 5, "and config.layer_types[i] in config.rope_parameters")
                add(lines, indent * 5, f"else {{'rope_type': 'default', 'rope_theta': ({full_theta} if ((i + 1) % _rope_period == 0) else {local_theta})}}")
                add(lines, indent * 4, "),")
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

        # Register embedding normalizer buffer (matches native vLLM: float32
        # buffer that gets dtype-converted to bf16 when model is loaded).
        emb_scale = self._model_config_data.get("embedding_scale")
        if emb_scale:
            add(lines, indent * 2, f"self.register_buffer('normalizer', torch.tensor({float(emb_scale)}), persistent=False)")

        # PLI scale factor buffers (matches native vLLM Gemma4Model).
        if classification.pli_gate_node_id is not None:
            hs_expr = self._config_expr("hidden_size")
            pli_dim_expr = self._resolve_const_value("PLI") or "getattr(config, 'per_layer_input_dim', 256)"
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
        add(lines, indent, "def embed_input_ids(self, input_ids):")
        for node_id in sorted(classification.embedding_node_ids):
            if node_id == classification.pli_embed_node_id:
                continue
            node = self._find_node_by_id(node_id)
            if node is not None:
                attr = self._vllm_attr_access(node)
                if hasattr(self, '_model_config_data') and self._model_config_data.get("embedding_scale"):
                    add(lines, indent * 2, f"return {attr}(input_ids) * self.normalizer")
                else:
                    add(lines, indent * 2, f"return {attr}(input_ids)")
                break
        add(lines, indent * 2, "return None")
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
        add(lines, indent * 2, "params_dict = dict(self.named_parameters())")
        add(lines, indent * 2, "params_dict.update(dict(self.named_buffers()))")
        add(lines, indent * 2, "loaded_params = set()")
        add(lines, indent * 2, "")
        add(lines, indent * 2, "# Build checkpoint name -> model param name mapping from module prefixes")
        add(lines, indent * 2, "_ckpt_to_model = {}")
        add(lines, indent * 2, "for _mod_name, _module in self.named_modules():")
        add(lines, indent * 3, "_prefix = getattr(_module, 'prefix', None)")
        add(lines, indent * 3, "if _prefix:")
        add(lines, indent * 4, "for _pname, _ in _module.named_parameters(recurse=False):")
        add(lines, indent * 5, "_ck = f'{_prefix}.{_pname}'")
        add(lines, indent * 5, "if _ck not in _ckpt_to_model:")
        add(lines, indent * 6, "_ckpt_to_model[_ck] = f'{_mod_name}.{_pname}'")
        add(lines, indent * 4, "for _bname, _ in _module.named_buffers(recurse=False):")
        add(lines, indent * 5, "_ck = f'{_prefix}.{_bname}'")
        add(lines, indent * 5, "if _ck not in _ckpt_to_model:")
        add(lines, indent * 6, "_ckpt_to_model[_ck] = f'{_mod_name}.{_bname}'")
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
            add(lines, indent * 4, f'_ckpt_to_model[f"{{_layer_prefix}}.layer_scalar"] = f"layer_scalars.{{_i}}"')
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
        if not self._use_clean_forward:
            add(lines, indent, "")
            add(lines, indent, "def _config(self, path, default=None):")
            add(lines, indent * 2, "key = str(path).lstrip('@').strip('.')")
            add(lines, indent * 2, "found = hasattr(self.config, key)")
            add(lines, indent * 2, "value = getattr(self.config, key, None) if found else None")
            add(lines, indent * 2, "if not found or value is None:")
            add(lines, indent * 3, "value = self._model_config.get(key, default)")
            add(lines, indent * 2, "return value if value is not None else default")
            add(lines, indent, "")
            add(lines, indent, "def _has_config(self, path):")
            add(lines, indent * 2, "key = str(path).lstrip('@').strip('.')")
            add(lines, indent * 2, "if hasattr(self.config, key):")
            add(lines, indent * 3, "return getattr(self.config, key, None) is not None")
            add(lines, indent * 2, "return key in self._model_config")
            add(lines, indent, "")
            self._emit_runtime_helpers(lines)
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

        if cls.mamba_mixer_module_names:
            self._emit_forward_ssm(lines, cls)
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
        add(lines, 8, "**kwargs,")
        add(lines, 8, "):")
        add(lines, 8, "if inputs_embeds is not None:")
        add(lines, 12, "hidden_states = inputs_embeds")
        add(lines, 8, "else:")
        add(lines, 12, "hidden_states = self.embed_input_ids(input_ids)")
        add(lines, 8, "")
        add(lines, 8, "config = self.config")
        add(lines, 8, "_tp_size = self._tp_size")
        add(lines, 8, f"_num_heads = {num_heads_expr} // _tp_size")
        add(lines, 8, f"_num_kv_heads = max(1, {num_kv_heads_expr} // _tp_size)")
        add(lines, 8, f"_head_dim = {head_dim_expr}")
        add(lines, 8, f"_num_layers = {num_layers_expr}")

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
                pli_dim = self._resolve_const_value("PLI") or "getattr(config, 'per_layer_input_dim', 256)"
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
        add(lines, 8, f"_first_kv_shared = getattr(config, 'num_hidden_layers', 0) - getattr(config, 'num_kv_shared_layers', 0)")
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
                    if ffn_group and ffn_group.down_node_id:
                        down_idx = int(ffn_group.down_node_id.rsplit(":", 1)[-1]) if ":" in ffn_group.down_node_id else 0
                        if add_idx > down_idx and not has_standalone_ffn_add:
                            has_standalone_ffn_add = True

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
                add(lines, indent, f"hidden_states = {attr}(hidden_states)")

            if norm_i == 0 and qkv_group:
                self._emit_attn_block(lines, indent, cls, qkv_group, rope_id, o_proj_id)
                if has_standalone_attn_add:
                    add(lines, indent, f"hidden_states = hidden_states + residual")
                    add(lines, indent, f"residual = hidden_states")
            if norm_i == ffn_norm_idx and ffn_group:
                self._emit_ffn_block(lines, indent, cls, ffn_group, act_name)
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

        # Always use packed QKV: k/v graph nodes are skipped in init
        # (they're part of the single QKVParallelLinear)
        add(lines, indent, f"qkv, _ = {q_attr}(hidden_states)")
        add(lines, indent, f"q, k, v = qkv.split([_q_size, _kv_size, _kv_size], dim=-1)")

        # QK norms with reshape
        qk_norms = sorted(cls.qk_norm_node_ids)
        has_var_hd = self._detect_head_dim_expr(self._get_repeated_module() or repeated_mod, "i") is not None if self._get_repeated_module() else False
        head_dim_ref = "_hd" if has_var_hd else "_head_dim"

        q_top = qkv_group.q_node_id.split(":", 1)[0] if ":" in qkv_group.q_node_id else ""
        k_top = qkv_group.k_node_id.split(":", 1)[0] if ":" in qkv_group.k_node_id else ""

        # Separate Q norm (always) from K norm (skip for KV-shared)
        k_norm_attr = None
        for nid in qk_norms:
            node = self._find_node_by_id(nid)
            attr = self._vllm_attr_access(node)
            norm_top = nid.split(":", 1)[0] if ":" in nid else ""
            if norm_top == k_top and norm_top != q_top:
                k_norm_attr = attr
            else:
                add(lines, indent, f"q = q.unflatten(-1, (_num_heads, {head_dim_ref}))")
                add(lines, indent, f"q = {attr}(q)")
                add(lines, indent, f"q = q.flatten(-2, -1)")

        # Match native vLLM ordering: K norm → RoPE → V norm (all inside one
        # if not _is_kv_shared block), with shared layers only applying RoPE to Q
        v_norm_attr = None
        if cls.v_norm_node_ids:
            nid = sorted(cls.v_norm_node_ids)[0]
            node = self._find_node_by_id(nid)
            v_norm_attr = self._vllm_attr_access(node)

        if rope_id:
            add(lines, indent, f"if not _is_kv_shared:")
            if k_norm_attr:
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
            if k_norm_attr:
                add(lines, indent, f"if not _is_kv_shared:")
                add(lines, indent + 1, f"k = k.unflatten(-1, (_num_kv_heads, {head_dim_ref}))")
                add(lines, indent + 1, f"k = {k_norm_attr}(k)")
                add(lines, indent + 1, f"k = k.flatten(-2, -1)")
            if v_norm_attr:
                if not k_norm_attr:
                    add(lines, indent, f"if not _is_kv_shared:")
                add(lines, indent + 1, f"v = v.unflatten(-1, (_num_kv_heads, {head_dim_ref}))")
                add(lines, indent + 1, f"v = {v_norm_attr}(v)")
                add(lines, indent + 1, f"v = v.flatten(-2, -1)")

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
            gate_attr = self._vllm_attr_access(gate_node)
            # Use merged gate_up_proj + GeluAndMul for numerical parity with native vLLM
            add(lines, indent, f"gate_up, _ = {gate_attr}(hidden_states)")
            if act_name and act_name.endswith(("gelu_pytorch_tanh", "gelu_tanh")):
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
            if act_name:
                act_code = self._activation_to_code(act_name)
                add(lines, indent, f"hidden_states = {act_code.format(x='hidden_states')}")

        if ffn_group.down_node_id:
            down_node = self._find_node_by_id(ffn_group.down_node_id)
            down_attr = self._vllm_attr_access(down_node)
            add(lines, indent, f"hidden_states = {down_attr}(hidden_states)[0]")

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

        # Find final norm: non-repeated RMSNorm (not in any repeated module)
        final_norm_attr = None
        for nid in sorted(cls.rmsnorm_node_ids):
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
        add(lines, 12, "hidden_states = self.embed_input_ids(input_ids)")
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
            add(lines, 8, f"hidden_states, _ = {final_norm_attr}(hidden_states, residual)")
        add(lines, 8, "")
        if final_norm_attr is not None:
            add(lines, 8, f"hidden_states = {final_norm_attr}(hidden_states)")
        add(lines, 8, "return hidden_states")
        add(lines, 4, "")

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
        add(lines, 8, "if input_ids is not None and input_ids.dim() == 1:")
        add(lines, 12, "input_ids = input_ids.unsqueeze(0)")
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
        for g in classification.ffn_groups:
            if g.gate_node_id and g.up_node_id:
                ffn_gate_to_up[g.gate_node_id] = g.up_node_id
                ffn_up_to_skip.add(g.up_node_id)
        self._ffn_gate_to_up = ffn_gate_to_up
        # Deduplicate v_norms: keep only one (they are identical RMSNorm has_weight=False).
        # Multiple v_norms arise from alternative KV modules (e.g. gemma4_kv_shared
        # vs gemma4_kv_separate) where only one path is taken at runtime.
        v_norm_ids = sorted(classification.v_norm_node_ids)
        if len(v_norm_ids) > 1:
            for nid in v_norm_ids[1:]:
                skip_node_ids.add(nid)
        # Precompute KV sharing targets: for layers in the last
        # num_kv_shared_layers, map to the earlier layer of the same
        # attention type whose KV cache they reuse.
        add(lines, indent, "_kv_sharing_targets = {}")
        add(lines, indent, "_nkv = getattr(config, 'num_kv_shared_layers', 0)")
        add(lines, indent, "if _nkv > 0:")
        add(lines, indent + 1, "_nl = getattr(config, 'num_hidden_layers', 0)")
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
                layer_type = classification.layer_type(node)
                if layer_type == VLLMLayerType.DEFAULT:
                    continue
                seen.add(node.id)
                attr_name = self._vllm_layer_attr_name(node)
                is_repeated = self._is_repeated_node(node) or '{i}' in self._node_prefix(node)
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
                # Modules that don't store prefix in __init__ need it set after creation.
                # PARALLEL_LM_HEAD is excluded: when tied to embeddings it shares the
                # same prefix, which would overwrite the embedding entry in the
                # checkpoint-to-model mapping.
                if layer_type in (
                    VLLMLayerType.RMSNORM,
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
            eps_expr = "getattr(config, 'layer_norm_epsilon', getattr(config, 'rms_norm_eps', 1e-5))"
            mixer_prefix = self._derive_mamba_mixer_prefix_from_scope(classification)
            add(lines, indent, f"self._vllm_mamba_mixer = nn.ModuleList([")
            add(lines, indent + 1, f"MambaMixer(")
            add(lines, indent + 2, f"hidden_size={hidden_expr},")
            add(lines, indent + 2, f"ssm_state_size={state_expr},")
            add(lines, indent + 2, f"conv_kernel_size={conv_k_expr},")
            add(lines, indent + 2, f"intermediate_size={inter_expr},")
            add(lines, indent + 2, f"time_step_rank={tsr_expr},")
            add(lines, indent + 2, f"use_conv_bias=getattr(config, 'use_conv_bias', True),")
            add(lines, indent + 2, f"use_bias=getattr(config, 'use_bias', False),")
            add(lines, indent + 2, f"use_rms_norm=False,")
            add(lines, indent + 2, f"rms_norm_eps={eps_expr},")
            add(lines, indent + 2, f"activation=getattr(config, 'hidden_act', 'silu'),")
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
                    is_repeated = self._is_repeated_node(node) or '{i}' in self._node_prefix(node)
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
                                        shape_parts.append(f"getattr(config, '{d}', 1)")
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
            }
            pli_linear_ids.discard(None)
            in_dim = self._node_input_dim_expr(node)
            out_dim = self._node_output_dim_expr(node)
            bias = _bool_arg(node, 3)
            if node.id == classification.pli_model_proj_node_id:
                add(lines, indent, "ColumnParallelLinear(")
                add(lines, indent + 1, f"{in_dim}, {out_dim},")
                add(lines, indent + 1, f"bias={bias}, gather_output=True,")
                add(lines, indent + 1, f"prefix={prefix},")
                add(lines, indent + 1, "quant_config=quant_config,")
                add(lines, indent + 1, "params_dtype=params_dtype,")
                add(lines, indent, ")")
            elif node.id in pli_linear_ids:
                add(lines, indent, "ReplicatedLinear(")
                add(lines, indent + 1, f"{in_dim}, {out_dim},")
                add(lines, indent + 1, f"bias={bias}, skip_bias_add=True,")
                add(lines, indent + 1, f"prefix={prefix},")
                add(lines, indent + 1, "params_dtype=params_dtype,")
                add(lines, indent, ")")
            elif layer_type == VLLMLayerType.QKV_PARALLEL_LINEAR:
                self._emit_qkv_init(lines, node, classification, indent)
            elif layer_type == VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR:
                up_node_id = self._ffn_gate_to_up.get(node.id)
                if up_node_id:
                    up_node = self._find_node_by_id(up_node_id)
                    up_out_dim = self._node_output_dim_expr(up_node) if up_node else out_dim
                    merged_prefix = prefix.replace(".gate_proj", ".gate_up_proj")
                    add(lines, indent, "MergedColumnParallelLinear(")
                    add(lines, indent + 1, f"{in_dim}, [{out_dim}, {up_out_dim}],")
                    add(lines, indent + 1, f"bias={bias},")
                    add(lines, indent + 1, f"prefix={merged_prefix},")
                    add(lines, indent + 1, "quant_config=quant_config,")
                    add(lines, indent + 1, "params_dtype=params_dtype,")
                    add(lines, indent, ")")
                else:
                    add(lines, indent, "ColumnParallelLinear(")
                    add(lines, indent + 1, f"{in_dim}, {out_dim},")
                    add(lines, indent + 1, f"bias={bias}, skip_bias_add=True,")
                    add(lines, indent + 1, f"prefix={prefix},")
                    add(lines, indent + 1, "quant_config=quant_config,")
                    add(lines, indent + 1, "params_dtype=params_dtype,")
                    add(lines, indent, ")")
            elif layer_type == VLLMLayerType.COLUMN_PARALLEL_LINEAR:
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
            head_size = self._config_expr("head_dim", alt="hidden_size")
            kv_heads_arg = "_num_kv_heads"
            if self._is_repeated_node(node):
                repeated_mod = self._get_repeated_module()
                if repeated_mod is not None:
                    loop_var = self._node_loop_index(node)
                    hd_expr = self._detect_head_dim_expr(repeated_mod, loop_var)
                    if hd_expr:
                        head_size = hd_expr
                    kvh_expr = self._detect_kv_heads_expr(repeated_mod, loop_var)
                    if kvh_expr:
                        kv_heads_arg = f"max(1, {kvh_expr} // _tp_size)"
            attn_scale = self._model_config_data.get("attention_scale")
            if attn_scale is not None:
                scale_expr = f"{float(attn_scale)}"
            else:
                scale_cfg = self._config_expr("query_pre_attn_scalar", alt="head_dim")
                scale_expr = f"1.0 / (float({scale_cfg}) ** 0.5)"
            add(lines, indent, "Attention(")
            add(lines, indent + 1, "_num_heads,")
            add(lines, indent + 1, f"{head_size},")
            add(lines, indent + 1, f"scale={scale_expr},")
            add(lines, indent + 1, f"num_kv_heads={kv_heads_arg},")
            add(lines, indent + 1, "cache_config=cache_config,")
            add(lines, indent + 1, "quant_config=quant_config,")
            add(lines, indent + 1, "logits_soft_cap=getattr(config, 'attn_logit_softcapping', None),")
            is_rep = self._is_repeated_node(node)
            if is_rep:
                add(lines, indent + 1, "per_layer_sliding_window=(")
                add(lines, indent + 2, "getattr(config, 'sliding_window', None)")
                add(lines, indent + 2, "if ((hasattr(config, 'layer_types') and i < len(config.layer_types) and config.layer_types[i] != 'full_attention')")
                add(lines, indent + 2, "    or (not hasattr(config, 'layer_types') and getattr(config, 'sliding_window_pattern', 0) and (i + 1) % getattr(config, 'sliding_window_pattern', 0) != 0))")
                add(lines, indent + 2, "else None),")
            else:
                add(lines, indent + 1, "per_layer_sliding_window=getattr(config, 'sliding_window', None),")
            # KV sharing: layers in the last num_kv_shared_layers reuse KV cache
            # from earlier layers of the same attention type.
            # Build target prefix by replacing {i} with the target index.
            # Attention prefix must end with .self_attn.attn to match native vLLM
            # convention used by KV cache binding and layer_name lookup.
            if ".self_attn.attn" in prefix:
                attn_prefix = prefix
            elif "{i}" in prefix:
                attn_prefix = prefix.replace("{i}", "{i}.self_attn.attn")
            else:
                attn_prefix = prefix
            if is_rep:
                target_prefix = attn_prefix.replace("{i}", "{_kv_sharing_targets[i]}")
                add(lines, indent + 1, "kv_sharing_target_layer_name=(")
                add(lines, indent + 2, f"{target_prefix}")
                add(lines, indent + 2, "if i in _kv_sharing_targets else None),")
            else:
                add(lines, indent + 1, "kv_sharing_target_layer_name=None,")
            add(lines, indent + 1, f"prefix={attn_prefix},")
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
            elif node.id in self._vllm_classification.v_norm_node_ids:
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
            has_weight = "rmsnorm_noscale" not in node.op.name
            if has_weight:
                add(lines, indent, "RMSNorm(")
                add(lines, indent + 1, f"{dim}, eps={eps},")
                add(lines, indent, ")")
            else:
                add(lines, indent, "RMSNorm(")
                add(lines, indent + 1, f"{dim}, eps={eps}, has_weight=False,")
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
        if self._is_repeated_node(node):
            repeated_mod = self._get_repeated_module()
            if repeated_mod is not None:
                loop_var = self._node_loop_index(node)
                kvh_expr = self._detect_kv_heads_expr(repeated_mod, loop_var)
                if kvh_expr:
                    total_num_kv_heads = kvh_expr
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
            base = _linear_base_key(q_node)
            if base and "{i}" in base:
                return f'f"{base.rsplit(".", 1)[0] + ".qkv_proj"}"'
            return f'f"{{prefix}}.layers.{{i}}.qkv_proj"'
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
        add(lines, indent, "for _pname, _wname, _sid in stacked_params_mapping:")
        add(lines, indent + 1, "if _wname not in name:")
        add(lines, indent + 2, "continue")
        add(lines, indent + 1, "name = name.replace(_wname, _pname)")
        add(lines, indent + 1, "name = _ckpt_to_model.get(name, name)")
        add(lines, indent + 1, "if name not in params_dict:")
        add(lines, indent + 2, "self.state_dict_tensors[_orig_name] = loaded_weight")
        add(lines, indent + 2, "break")
        add(lines, indent + 1, "param = params_dict[name]")
        add(lines, indent + 1, "weight_loader = getattr(param, 'weight_loader', None)")
        add(lines, indent + 1, "if weight_loader is not None:")
        add(lines, indent + 2, "weight_loader(param, loaded_weight, _sid)")
        add(lines, indent + 1, "else:")
        add(lines, indent + 2, "param.data.copy_(loaded_weight)")
        add(lines, indent + 1, "loaded_params.add(name)")
        add(lines, indent + 1, "self.state_dict_tensors[_orig_name] = loaded_weight")
        add(lines, indent + 1, "break")
        add(lines, indent, "else:")
        add(lines, indent + 1, "name = _ckpt_to_model.get(name, name)")
        add(lines, indent + 1, "if name not in params_dict:")
        add(lines, indent + 2, "self.state_dict_tensors[_orig_name] = loaded_weight")
        add(lines, indent + 2, "continue")
        add(lines, indent + 1, "param = params_dict[name]")
        add(lines, indent + 1, "weight_loader = getattr(param, 'weight_loader', None)")
        add(lines, indent + 1, "if weight_loader is not None:")
        add(lines, indent + 2, "weight_loader(param, loaded_weight)")
        add(lines, indent + 1, "else:")
        add(lines, indent + 2, "param.data.copy_(loaded_weight)")
        add(lines, indent + 1, "loaded_params.add(name)")
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
        return fallback

    def _node_prefix(self, node: GraphNode) -> str:
        base = _linear_base_key(node)
        if not base:
            base = self._resolve_prefix_from_called_module(node)
        if not base:
            return node.id
        return base

    def _layer_prefix(self, node: GraphNode) -> str:
        if not self._is_repeated_node(node):
            base = self._node_prefix(node)
            if "{i}" in base:
                return f'f"{base}"'
            return repr(base)
        mod_name = self._node_module_name(node)
        scope_parts = self._vllm_classification.module_scope_parts.get(mod_name)
        if scope_parts is None:
            base = self._node_prefix(node)
            if "{i}" in base or "{prefix}" in base:
                return f'f"{base}"'
            return f'f"{{prefix}}.layers.{{i}}.{_safe_ident(base)}"'
        base = _linear_base_key(node)
        if not base:
            base = self._resolve_prefix_from_called_module(node)
        if not base:
            for inp in node.inputs:
                if isinstance(inp, GraphPath) and inp.parts:
                    base = ".".join(inp.parts)
                    break
        if base:
            sub_parts = base.split(".")
            if sub_parts and sub_parts[0] == "{__scope}":
                full_parts = list(scope_parts) + sub_parts[1:]
            else:
                full_parts = sub_parts
        else:
            full_parts = list(scope_parts)
        fparts: list[str] = []
        for p in full_parts:
            if p == "{__scope}":
                fparts.append("{prefix}")
            else:
                fparts.append(p)
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
        fparts: list[str] = []
        for p in scope_parts:
            if p == "{__scope}":
                fparts.append("{prefix}")
            elif p == "{i}":
                fparts.append("{_i}")
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
        """Extract input dim from the node's input tensor type (last dim)."""
        if len(node.inputs) >= 2:
            inp = node.inputs[1]
            inp_dims = getattr(inp, "dims", None) or getattr(getattr(inp, "type_expr", None), "dims", None)
            if inp_dims and len(inp_dims) > 0:
                last = inp_dims[-1]
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

    def _collect_args(self, node: GraphNode, local: set[str] | dict[str, str]) -> list[str]:
        args: list[str] = []
        for inp in node.inputs:
            if isinstance(inp, (GraphValueRef, GraphValue)):
                name = inp.name
                if isinstance(local, dict):
                    args.append(local.get(name, name))
                else:
                    args.append(name if name in local else name)
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
            "import torch",
            "from torch import nn",
            "from torch.nn import functional as F",
            "",
            f"_MODEL_CONFIG = {model_config!r}",
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


__all__ = [
    "emit_model_code_from_graph_ir",
]
