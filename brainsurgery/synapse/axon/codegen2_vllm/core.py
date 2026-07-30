from __future__ import annotations

import re
from typing import Any

from ..codegen2_torch.core import _DirectTorchEmitter, _py_ident
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
    _is_embedding_call,
    _is_linear_call,
    _value_name,
    _trace_back,
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


def _select_linear_expr(node: GraphNode) -> GraphExpr | None:
    """Extract the first GraphExpr branch from a core.select node."""
    for inp in node.inputs:
        if isinstance(inp, GraphExpr):
            return inp
    return None


def _node_sort_index(node_id: str) -> int:
    """Extract a numeric sort index from a graph node ID.

    Node IDs after optimization may have forms like:
    - ``opt_block:2`` → 2
    - ``opt_block:nested-cse:opt_block:2:1`` → 1 (last segment)
    - ``opt_block:hoist:opt_block:17:_v2`` → 17 (fallback to earlier segment)
    - ``smollm_model__loop_layers_step_1:inl:...:2:smollm_block:17`` → 17 (last segment)

    Uses the last colon-separated segment if it is numeric, otherwise
    searches backwards for the first numeric segment (skipping non-numeric
    variant suffixes like ``_v2``).
    """
    parts = node_id.split(":")
    for part in reversed(parts):
        part = part.strip()
        if part.isdigit():
            return int(part)
    return 0


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
        elif isinstance(inp, GraphExpr):
            for sub in inp.inputs:
                if isinstance(sub, GraphPath) and sub.parts:
                    key = _graph_path_key(sub)
                    if not key.endswith(".weight") and not key.endswith(".bias"):
                        return key
                    if not fallback:
                        fallback = key
    if fallback:
        for suffix in (".weight", ".bias"):
            if fallback.endswith(suffix):
                return fallback[: -len(suffix)]
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
        self._ffn_up_node_ids: set[str] = {
            g.up_node_id for g in self._vllm_classification.ffn_groups
        }
        self._use_clean_forward: bool = False
        self._has_lm_head: bool = False
        self._model_config_data: dict[str, Any] = model_config or {}
        self._node_id_aliases: dict[str, str] = self._build_node_id_aliases()
        self._inline_loop_prefix: dict[str, str] = self._build_inline_loop_prefix()
        self._unit_offset_norm_attrs: list[tuple[str, bool]] = []
        self._pos_emb_info: tuple[str, str] | None = None

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
                elif node.op.name == "_config_bool":
                    path_inp = node.inputs[0] if node.inputs else None
                    default = _literal_value(node.inputs[1], None) if len(node.inputs) >= 2 else None
                    if isinstance(path_inp, GraphPath) and path_inp.parts:
                        field = path_inp.parts[-1]
                        if isinstance(default, bool):
                            return f"getattr(config, '{field}', {repr(default)})"
                        return f"getattr(config, '{field}', False)"
            # Fallback: modules with no config nodes but a literal output
            if module.outputs:
                out = module.outputs[0]
                if isinstance(out, GraphLiteral) and isinstance(out.value, (int, float)):
                    return str(out.value)
        return None

    def _bias_expr(self, node: GraphNode) -> bool | str:
        """Resolve the bias argument (input index 3) of a linear node.

        Returns a ``bool`` for literal values, or a ``str`` config expression
        (e.g. ``"getattr(config, 'use_qkv_bias', False)"``) for config-derived
        booleans.

        After graph optimization, the ``bias`` boolean literal may be stripped
        from the call site's inputs (dead-formal pruning removes it when all
        call sites agree on the same value).  In that case, recover the bias
        value from the ``NN.linear`` module's internal ``_linear`` node, where
        the optimizer has constant-folded it.
        """
        if len(node.inputs) <= 3:
            return False
        inp = node.inputs[3]
        if isinstance(inp, GraphLiteral):
            return bool(inp.value)
        ref_name = getattr(inp, "name", None)
        if ref_name is not None:
            resolved = self._resolve_const_value(ref_name)
            if resolved is not None:
                return resolved
        # Bias literal was stripped from call site — recover from module def
        if node.op.name == "NN.linear":
            inner_mod = self.modules_by_name.get("NN.linear")
            if inner_mod is not None:
                for inner_node in inner_mod.nodes:
                    if inner_node.op.name == "_linear" and len(inner_node.inputs) > 3:
                        bias_inp = inner_node.inputs[3]
                        if isinstance(bias_inp, GraphLiteral):
                            return bool(bias_inp.value)
        return False

    def _trace_module_output(
        self,
        inner_mod: Any,
        repeated_mod: Any,
        loop_var: str,
        visited: set[str],
    ) -> str | None:
        """Trace a module's output node rather than its first non-None node."""
        if inner_mod.outputs:
            for mod_out in inner_mod.outputs:
                if isinstance(mod_out, GraphLiteral) and isinstance(mod_out.value, (int, float)):
                    return str(mod_out.value)
                out_name = getattr(mod_out, "name", None)
                if out_name is not None:
                    for inner_node in inner_mod.nodes:
                        for node_out in inner_node.outputs:
                            if hasattr(node_out, "name") and node_out.name == out_name:
                                result = self._trace_node_dim_expr(
                                    inner_node, inner_mod, loop_var, visited,
                                )
                                if result is not None:
                                    return result
        for inner_node in inner_mod.nodes:
            result = self._trace_node_dim_expr(inner_node, inner_mod, loop_var, visited)
            if result is not None:
                return result
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
            if operand.value is None:
                return None
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
            # Check if name matches a module (constant definition)
            inner_mod = self.modules_by_name.get(name)
            if inner_mod is not None:
                result = self._trace_module_output(inner_mod, repeated_mod, loop_var, visited)
                if result is not None:
                    return result
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
        elif op == "Math.floor" and len(node.inputs) >= 1:
            return self._trace_dim_expr(node.inputs[0], repeated_mod, loop_var, visited)
        # If the op is a module call, trace into the module
        inner_mod = self.modules_by_name.get(op)
        if inner_mod is not None:
            result = self._trace_module_output(inner_mod, repeated_mod, loop_var, visited)
            if result is not None:
                return result
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
        canonical = self._node_id_aliases.get(nid, nid)
        for mod_name in self._vllm_classification.repeated_module_names:
            if nid == mod_name or nid.startswith(mod_name + ":"):
                return True
            if canonical == mod_name or canonical.startswith(mod_name + ":"):
                return True
        return False

    def _get_primary_loop_var(self) -> str:
        """Return the loop variable of the primary (layer) loop."""
        layer_mod = self._get_layer_loop_module()
        if layer_mod is not None:
            lv = self._vllm_classification.loop_index_param.get(layer_mod.name)
            if lv:
                return lv
        return "i"

    def _get_layer_loop_module(self) -> Any:
        """Return the layer loop module — the __loop_*_step_* module whose
        loop variable is 'i' or whose range is num_hidden_layers.
        This is the outermost loop that iterates over layers, not inner loops
        like expert or timestep loops.
        """
        cls = self._vllm_classification
        # Prefer modules with loop var 'i' that have 'layers' in their name
        for mod in self.program.modules:
            if "__loop_" not in mod.name or "_step_" not in mod.name:
                continue
            lv = cls.loop_index_param.get(mod.name)
            if lv == "i" and "layers" in mod.name:
                return mod
        # Fallback: any module with loop var 'i'
        for mod in self.program.modules:
            if "__loop_" not in mod.name or "_step_" not in mod.name:
                continue
            lv = cls.loop_index_param.get(mod.name)
            if lv == "i":
                return mod
        # Last resort: the __loop_*_step_* module with the most nodes
        best = None
        best_count = 0
        for mod in self.program.modules:
            if "__loop_" not in mod.name or "_step_" not in mod.name:
                continue
            if len(mod.nodes) > best_count:
                best = mod
                best_count = len(mod.nodes)
        return best

    def _is_nested_loop_node(self, node: Any) -> bool:
        """Check if a node belongs to a nested (non-primary) loop, e.g. an expert loop."""
        nid = getattr(node, "id", "")
        canonical = self._node_id_aliases.get(nid, nid)
        primary_var = self._get_primary_loop_var()
        for mod_name in self._vllm_classification.repeated_module_names:
            if nid == mod_name or nid.startswith(mod_name + ":"):
                lv = self._vllm_classification.loop_index_param.get(mod_name)
                if lv is not None and lv != primary_var:
                    return True
            if canonical == mod_name or canonical.startswith(mod_name + ":"):
                lv = self._vllm_classification.loop_index_param.get(mod_name)
                if lv is not None and lv != primary_var:
                    return True
        return False

    def _get_nested_loop_var(self, node: Any) -> str | None:
        """Return the loop variable for a nested loop node, or None."""
        nid = getattr(node, "id", "")
        canonical = self._node_id_aliases.get(nid, nid)
        primary_var = self._get_primary_loop_var()
        for mod_name in self._vllm_classification.repeated_module_names:
            if nid == mod_name or nid.startswith(mod_name + ":"):
                lv = self._vllm_classification.loop_index_param.get(mod_name)
                if lv is not None and lv != primary_var:
                    return lv
            if canonical == mod_name or canonical.startswith(mod_name + ":"):
                lv = self._vllm_classification.loop_index_param.get(mod_name)
                if lv is not None and lv != primary_var:
                    return lv
        return None

    def _get_layer_scope_parts(self) -> tuple[str, ...] | None:
        """Return scope_parts for the primary (layer) loop."""
        layer_mod = self._get_layer_loop_module()
        if layer_mod is not None:
            sp = self._vllm_classification.module_scope_parts.get(layer_mod.name)
            if sp:
                return sp
        # Fallback: any scope_parts containing the primary loop var
        primary_var = self._get_primary_loop_var()
        loop_token = "{" + primary_var + "}"
        for sp in self._vllm_classification.module_scope_parts.values():
            if any(loop_token in p for p in sp):
                return sp
        return None

    def _get_expert_count_expr(self) -> str:
        """Return a Python expression for the number of experts from config."""
        for key in ('n_routed_experts', 'num_local_experts', 'num_experts'):
            if key in self._model_config_data:
                return f"getattr(config, '{key}', self._model_config.get('{key}', 0))"
        # Try config keys at runtime even if not in model_config
        return ("getattr(config, 'n_routed_experts', "
                "getattr(config, 'num_local_experts', "
                "getattr(config, 'num_experts', 0)))")

    def _get_repeated_module(self) -> Any:
        cls = self._vllm_classification
        if not cls.repeated_module_names:
            return None
        name = max(
            cls.repeated_module_names,
            key=lambda n: len(self.modules_by_name[n].nodes) if n in self.modules_by_name else 0,
        )
        mod = self.modules_by_name.get(name)
        # When the block is inlined into the loop body, the repeated_module_names
        # may only contain small callee modules (e.g. exaone4_rms with 1 node)
        # instead of the loop body module that has all the per-layer nodes.
        # Fall back to the loop body module (__loop_..._step_) when it has
        # significantly more nodes.
        if mod is not None and len(mod.nodes) < 5:
            for m in self.program.modules:
                if "__loop_" in m.name and "_step_" in m.name and len(m.nodes) > len(mod.nodes):
                    return m
        return mod

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
        nid = getattr(node, "id", "")
        # For rewritten inline-body nodes (id contains ":inline:"),
        # the loop variable is renamed to {inline_prefix}_{loop_var}.
        if ":inline:" in nid:
            canonical = self._node_id_aliases.get(nid, nid)
            prefix_part = nid.rsplit(":inline:", 1)[0]
            # For nested inline nodes, the prefix won't match the repeat node ID,
            # so look up the correct loop prefix in _inline_loop_prefix.
            if prefix_part in self._inline_loop_prefix:
                inline_prefix = self._inline_loop_prefix[prefix_part]
            else:
                inline_prefix = _py_ident(f"__loop_inline_{prefix_part.replace(':', '_')}")
            for mod_name, loop_var in self._vllm_classification.loop_index_param.items():
                if canonical == mod_name or canonical.startswith(mod_name + ":"):
                    return f"{inline_prefix}_{loop_var}"
            return f"{inline_prefix}_i"
        mod = self._node_module_name(node)
        return self._vllm_classification.loop_index_param.get(mod, "i")

    def _vllm_attr_access(self, node: Any) -> str:
        attr = self._vllm_layer_attr_name(node)
        if self._is_nested_loop_node(node):
            layer_idx = self._get_primary_loop_var()
            expert_idx = self._get_nested_loop_var(node) or "e"
            return f"self.{attr}[{layer_idx}][{expert_idx}]"
        if self._is_repeated_node(node) or '{i}' in self._node_prefix(node):
            idx = self._node_loop_index(node)
            return f"self.{attr}[{idx}]"
        return f"self.{attr}"

    def _vllm_attr_access_init(self, node: Any) -> str:
        """Like _vllm_attr_access but for __init__ context where the loop
        variable is not in scope.  Uses [-1] for repeated nodes (last layer).
        """
        attr = self._vllm_layer_attr_name(node)
        if self._is_nested_loop_node(node):
            return f"self.{attr}[-1][-1]"
        if self._is_repeated_node(node) or '{i}' in self._node_prefix(node):
            return f"self.{attr}[-1]"
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

    def _emit_runtime_helpers(self, lines: list[str]) -> None:
        super()._emit_runtime_helpers(lines)
        add = self._add
        # Override _linear: vLLM weight loading already transposes Conv1D
        # weights from [in, out] to [out, in], so always use F.linear
        # (which computes x @ weight.T) regardless of the transpose flag.
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
        add(lines, 12, "out_dim = int(weight_run.shape[-2])")
        add(lines, 12, "return x.new_empty((*x.shape[:-1], out_dim))")
        add(lines, 8, "return F.linear(x, weight_run, bias_run)")
        add(lines, 4, "")
        add(lines, 4, "def _safe_linear(self, x, weight, bias=None):")
        add(lines, 8, "if x.is_floating_point() and weight.is_floating_point() and x.dtype != weight.dtype:")
        add(lines, 12, "x = x.to(weight.dtype)")
        add(lines, 8, "return F.linear(x, weight, bias)")
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
        # In the vLLM backend, Conv1D weights are already transposed from
        # [in, out] to [out, in] during weight loading.  The inline linear
        # code must therefore always use F.linear (x @ weight.T) instead
        # of torch.matmul (x @ weight) for transpose=True nodes.
        if len(node.inputs) > 4:
            import dataclasses
            modified = list(node.inputs)
            modified[4] = GraphLiteral(False, None)
            node_copy = dataclasses.replace(node, inputs=tuple(modified))
            before = len(lines)
            result = super()._emit_linear_node(
                lines, node_copy,
                target=target, indent=indent,
                local=local, symbols_dict=symbols_dict,
            )
            if result:
                for i in range(before, len(lines)):
                    lines[i] = lines[i].replace('F.linear(', 'self._safe_linear(')
            return result
        before = len(lines)
        result = super()._emit_linear_node(
            lines, node,
            target=target, indent=indent,
            local=local, symbols_dict=symbols_dict,
        )
        if result:
            for i in range(before, len(lines)):
                lines[i] = lines[i].replace('F.linear(', 'self._safe_linear(')
        return result

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
        bias_guard = False

        if layer_type == VLLMLayerType.VOCAB_PARALLEL_EMBEDDING:
            if len(args) < 2:
                return False
            expr = f"{attr}({args[1]})"
            if len(node.inputs) >= 4:
                scale = node.inputs[3]
                if not (isinstance(scale, GraphLiteral) and scale.value is None):
                    local_set = local if isinstance(local, set) else set(local.keys())
                    scale_expr = self._operand_expr(scale, local=local_set, symbols_dict=symbols_dict)
                    if scale_expr:
                        expr = f"({expr} * {scale_expr})"
        elif layer_type == VLLMLayerType.QKV_PARALLEL_LINEAR:
            if len(args) < 2:
                return False
            bias = self._bias_expr(node)
            if isinstance(bias, str):
                bias_guard = True
                expr = f"{attr}({args[1]})[0]"
            elif bias:
                expr = f"{attr}({args[1]})[0] + {attr}.bias"
            else:
                expr = f"{attr}({args[1]})[0]"
        elif layer_type in (VLLMLayerType.COLUMN_PARALLEL_LINEAR, VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR):
            if len(args) < 2:
                return False
            bias = self._bias_expr(node)
            if isinstance(bias, str):
                bias_guard = True
                expr = f"{attr}({args[1]})[0]"
            elif bias:
                expr = f"{attr}({args[1]})[0] + {attr}.bias"
            else:
                expr = f"{attr}({args[1]})[0]"
        elif layer_type == VLLMLayerType.ROW_PARALLEL_LINEAR:
            if len(args) < 2:
                return False
            bias = self._bias_expr(node)
            if isinstance(bias, str):
                bias_guard = True
                expr = f"{attr}({args[1]})[0]"
            elif bias:
                expr = f"{attr}({args[1]})[0] + {attr}.bias"
            else:
                expr = f"{attr}({args[1]})[0]"
        elif layer_type == VLLMLayerType.PARALLEL_LM_HEAD:
            lin_expr = _select_linear_expr(node)
            if lin_expr is not None:
                data_inp = lin_expr.inputs[1] if len(lin_expr.inputs) > 1 else None
                if data_inp is None:
                    return False
                data_name = _value_name(data_inp)
                if data_name is None:
                    return False
                local_set = local if isinstance(local, set) else set(local.keys())
                data_expr = _py_ident(data_name) if data_name in local_set else data_name
                expr = data_expr
            else:
                if len(args) < 2:
                    return False
                expr = args[1]
        elif layer_type == VLLMLayerType.ATTENTION:
            if len(args) >= 3:
                expr = f"{attr}({args[0]}.contiguous(), {args[1]}.contiguous(), {args[2]}.contiguous())"
            else:
                expr = f"{attr}()"
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
        if bias_guard:
            add(lines, indent, f"if {attr}.bias is not None:")
            add(lines, indent + 1, f"{targets[0]} = {targets[0]} + {attr}.bias")
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
            # Check if this node calls a sub-module that contains rope
            inner_mod = self.modules_by_name.get(node.op.name)
            if inner_mod is not None:
                for inner_node in inner_mod.nodes:
                    if "rope" in inner_node.op.name.lower():
                        return node.id
                    inner_mod2 = self.modules_by_name.get(inner_node.op.name)
                    if inner_mod2 is not None:
                        for inner_node2 in inner_mod2.nodes:
                            if "rope" in inner_node2.op.name.lower():
                                return node.id
            for inp in node.inputs:
                if isinstance(inp, GraphExpr) and _search_expr(inp):
                    return node.id
        return None

    def _detect_rope_variants(self) -> tuple[str, str, str, str, str] | None:
        """Detect per-layer RoPE variants (local vs full) across all modules.

        Returns (local_hd, local_theta, full_hd, full_theta, rope_period_expr) or None.
        """
        rope_calls: list[tuple[int, float]] = []
        for module in self.program.modules:
            for node in module.nodes:
                op_lower = node.op.name.lower()
                if "rope" not in op_lower:
                    continue
                if len(node.inputs) >= 3:
                    hd = _literal_value(node.inputs[1], None)
                    theta = _literal_value(node.inputs[2], None)
                    if isinstance(hd, int) and isinstance(theta, (int, float)):
                        rope_calls.append((int(hd), float(theta)))
        seen: set[tuple[int, float]] = set()
        unique: list[tuple[int, float]] = []
        for pair in rope_calls:
            if pair not in seen:
                seen.add(pair)
                unique.append(pair)
        if len(unique) < 2:
            return None
        unique.sort()
        local_hd, local_theta = unique[0]
        full_hd, full_theta = unique[-1]
        rope_period = self._resolve_const_value("ROPE_PERIOD")
        if rope_period is None:
            rope_period = "6"
        return (str(local_hd), repr(local_theta), str(full_hd), repr(full_theta), rope_period)

    def _detect_rope_interleaved(self) -> bool:
        """Detect if the model uses interleaved (GPT-J style) RoPE.

        After optimization, the ``interleaved`` boolean is resolved into
        specialized op names: ``rope_base_factors__cond_true_*`` for
        interleaved=true vs ``rope_base_factors__cond_else_*`` for false.
        """
        for module in self.program.modules:
            for node in module.nodes:
                if "rope_base_factors" not in node.op.name.lower():
                    continue
                if "__cond_true" in node.op.name:
                    return True
        return False

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

    def _detect_pos_embedding(self, classification: Any) -> tuple[str, str] | None:
        """Detect a learned position embedding (NN.embedding not classified as
        VOCAB_PARALLEL_EMBEDDING, with path containing 'position' or 'embed_pos').

        Returns (prefix_expr, dim_expr) or None.
        """
        for module in self.program.modules:
            for node in module.nodes:
                if not _is_embedding_call(node, self.modules_by_name):
                    continue
                if node.id in classification.embedding_node_ids:
                    continue
                if node.id == getattr(classification, 'pli_embed_node_id', None):
                    continue
                path_inp = node.inputs[0] if node.inputs else None
                if not isinstance(path_inp, GraphPath) or not path_inp.parts:
                    continue
                path_str = ".".join(path_inp.parts).lower()
                if "position" not in path_str and "embed_pos" not in path_str:
                    continue
                prefix = self._layer_prefix(node)
                dim = self._node_output_dim_expr(node)
                if dim is None:
                    dim = self._config_expr("hidden_size")
                return (prefix, dim)
        return None

    def _detect_pos_offset(self) -> int:
        """Detect a constant offset added to position IDs (e.g. OPT adds +2).

        Searches for a ``core.binary.+`` node in the main module that adds a
        positive integer literal to a value tracing back to position_ids.
        """
        main_mod = self.modules_by_name.get(self.program.main_module)
        if main_mod is None:
            return 0
        for node in main_mod.nodes:
            if node.op.name != "core.binary.+":
                continue
            for i, inp in enumerate(node.inputs):
                if not isinstance(inp, GraphLiteral):
                    continue
                if not isinstance(inp.value, int) or inp.value <= 0:
                    continue
                other = node.inputs[1 - i]
                if isinstance(other, GraphExpr):
                    if "position" in other.op.name.lower() or "select" in other.op.name.lower():
                        return inp.value
                elif hasattr(other, "name") and "position" in (other.name or "").lower():
                    return inp.value
        return 0

    def _detect_activation(self, repeated_mod: Any) -> str | None:
        visited: set[str] = set()
        def _search(mod, depth=0):
            if depth > 4 or mod.name in visited:
                return None
            visited.add(mod.name)
            for node in mod.nodes:
                op_name = node.op.name
                if op_name.startswith("Activations.") or op_name.startswith("_activations_"):
                    return op_name
                # Check GraphExpr inputs for activation calls (e.g. core.select
                # branches that wrap Activations.* calls)
                for inp in node.inputs:
                    if hasattr(inp, "op") and hasattr(inp, "inputs"):
                        sub_op = inp.op.name
                        if sub_op.startswith("Activations.") or sub_op.startswith("_activations_"):
                            return sub_op
                if op_name in self.modules_by_name and op_name not in visited:
                    found = _search(self.modules_by_name[op_name], depth + 1)
                    if found:
                        return found
            return None
        return _search(repeated_mod)

    @staticmethod
    def _activation_to_code(act_name: str) -> str:
        if act_name.endswith("gelu_pytorch_tanh") or act_name.endswith("gelu_new"):
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
    ) -> list[tuple[str, str, bool, bool, bool]]:
        """Return ordered list of (node_id, attr_expr, uses_residual, fused, post_norm) for non-QK norms.

        ``post_norm`` is True when the norm is applied AFTER its subblock
        (attention or FFN) rather than before it (e.g. OLMo-2 post-norm structure).
        """
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
                idx = _node_sort_index(node.id)
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

        # Detect post-norms: norms whose input comes from a subblock output
        # (o_proj for attention, down_proj for FFN) rather than the block input.
        post_norm_set: set[str] = set()
        o_proj_id = self._find_o_proj_id(repeated_mod, classification)
        ffn_down_ids = {g.down_node_id for g in classification.ffn_groups}
        subblock_node_ids: set[str] = set()
        if o_proj_id:
            subblock_node_ids.add(o_proj_id)
        subblock_node_ids.update(ffn_down_ids)
        if subblock_node_ids:
            for _, nid in norms:
                node = self._find_node_by_id(nid)
                if node is None or not node.inputs:
                    continue
                data_inp = node.inputs[1] if isinstance(node.inputs[0], GraphPath) else node.inputs[0]
                src = _trace_back(repeated_mod, data_inp, subblock_node_ids)
                if src is not None:
                    post_norm_set.add(nid)

        result: list[tuple[str, str, bool, bool, bool]] = []
        for _, nid in norms:
            node = self._find_node_by_id(nid)
            attr = self._vllm_attr_access(node)
            result.append((nid, attr, nid in uses_residual_set, nid in fused_set, nid in post_norm_set))
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

    def _detect_alibi(self) -> bool:
        """Check if model uses ALiBi position bias."""
        main_mod = self.modules_by_name.get(self.program.main_module)
        if main_mod is None:
            return False
        _alibi_ops = {
            "Positions.linear_position_bias",
            "Positions.linear_position_bias_for_input",
        }
        for node in main_mod.nodes:
            if node.op.name in _alibi_ops:
                return True
            if node.op.name == "core.select":
                for inp in node.inputs:
                    if hasattr(inp, "op") and inp.op.name in _alibi_ops:
                        return True
        return False

    def _emit_common(self, lines: list[str]) -> None:
        add = self._add
        classification = self._vllm_classification
        indent = 4

        add(lines, indent, '"""Generated vLLM model from Axon Graph IR."""')
        add(lines, indent, "")
        if self._needs_mamba_cache_placeholders(classification) or classification.mamba_mixer_module_names:
            if classification.qkv_groups:
                add(lines, indent, "is_hybrid = True")
            else:
                add(lines, indent, "is_attention_free = True")
            add(lines, indent, "")
            add(lines, indent, "@classmethod")
            add(lines, indent, "def get_mamba_state_shape_from_config(cls, vllm_config):")
            add(lines, indent + 1, "from vllm.model_executor.models.mamba import MambaStateShapeCalculator")
            add(lines, indent + 1, "parallel_config = vllm_config.parallel_config")
            add(lines, indent + 1, "hf_config = vllm_config.model_config.hf_config")
            add(lines, indent + 1, "_tp = parallel_config.tensor_parallel_size")
            add(lines, indent + 1, "_mhd = getattr(hf_config, 'mamba_head_dim', None)")
            add(lines, indent + 1, "_mh = getattr(hf_config, 'mamba_num_heads', None)")
            add(lines, indent + 1, "if _mhd is not None and _mh is not None:")
            add(lines, indent + 2, "_inter = _mh * _mhd")
            add(lines, indent + 2, "_ng = getattr(hf_config, 'n_groups', getattr(hf_config, 'n_group', 1))")
            add(lines, indent + 2, "_ss = getattr(hf_config, 'ssm_state_size', getattr(hf_config, 'state_size', 128))")
            add(lines, indent + 2, "_ck = getattr(hf_config, 'conv_kernel', getattr(hf_config, 'mamba_d_conv', 4))")
            add(lines, indent + 2, "return MambaStateShapeCalculator.mamba2_state_shape(")
            add(lines, indent + 3, "tp_world_size=_tp, intermediate_size=_inter, n_groups=_ng,")
            add(lines, indent + 3, "num_heads=_mh, head_dim=_mhd, state_size=_ss, conv_kernel=_ck,")
            add(lines, indent + 2, ")")
            add(lines, indent + 1, "_inter = getattr(hf_config, 'intermediate_size', None)")
            add(lines, indent + 1, "if _inter is None:")
            add(lines, indent + 2, "_hs = getattr(hf_config, 'hidden_size', 0)")
            add(lines, indent + 2, "_inter = getattr(hf_config, 'mamba_expand', 2) * _hs")
            add(lines, indent + 1, "_ss = getattr(hf_config, 'state_size', getattr(hf_config, 'mamba_d_state', 128))")
            add(lines, indent + 1, "_ck = getattr(hf_config, 'conv_kernel', getattr(hf_config, 'mamba_d_conv', 4))")
            add(lines, indent + 1, "return MambaStateShapeCalculator.mamba1_state_shape(")
            add(lines, indent + 2, "tp_world_size=_tp, intermediate_size=_inter, state_size=_ss, conv_kernel=_ck,")
            add(lines, indent + 1, ")")
            add(lines, indent, "")
            add(lines, indent, "@classmethod")
            add(lines, indent, "def get_mamba_state_dtype_from_config(cls, vllm_config):")
            add(lines, indent + 1, "hf_config = vllm_config.model_config.hf_config")
            add(lines, indent + 1, "_dt = getattr(hf_config, 'mamba_ssm_cache_dtype', None)")
            add(lines, indent + 1, "if _dt:")
            add(lines, indent + 2, "from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE")
            add(lines, indent + 2, "return [STR_DTYPE_TO_TORCH_DTYPE[_dt]]")
            add(lines, indent + 1, "return [vllm_config.model_config.dtype]")
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
        add(lines, indent * 2, "self._params_dtype = params_dtype")
        add(lines, indent * 2, "cache_config = getattr(vllm_config, 'cache_config', None)")
        add(lines, indent * 2, "from vllm.distributed import get_tensor_model_parallel_world_size")
        add(lines, indent * 2, "_tp_size = get_tensor_model_parallel_world_size()")
        add(lines, indent * 2, "self._tp_size = _tp_size")
        add(lines, indent * 2, "self._model_config = dict(_MODEL_CONFIG or {})")
        add(lines, indent * 2, "self.config = config")
        add(lines, indent * 2, "_num_heads = getattr(config, 'num_attention_heads', self._model_config.get('num_attention_heads', 1)) // _tp_size")
        add(lines, indent * 2, "_num_kv_heads = max(1, getattr(config, 'num_key_value_heads', getattr(config, 'num_attention_heads', self._model_config.get('num_attention_heads', 1))) // _tp_size)")
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
            head_dim_expr = self._head_dim_expr()
            _has_partial_rotary = self._detect_partial_rotary()
            add(lines, indent * 2, "from vllm.model_executor.layers.rotary_embedding import get_rope")
            _is_neox = not self._detect_rope_interleaved()
            rope_variants = self._detect_rope_variants()
            if rope_variants:
                local_hd, local_theta, full_hd, full_theta, cond_expr = rope_variants
                add(lines, indent * 2, f"_rope_period = {cond_expr}")
                add(lines, indent * 2, "self.rotary_emb = nn.ModuleList([")
                add(lines, indent * 3, "get_rope(")
                add(lines, indent * 4, f"({full_hd} if ((i + 1) % _rope_period == 0) else {local_hd}),")
                add(lines, indent * 4, "max_position=getattr(config, 'max_position_embeddings', 4096),")
                add(lines, indent * 4, f"is_neox_style={_is_neox},")
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
                add(lines, indent * 2, "_rope_theta = getattr(config, 'rope_theta', None)")
                add(lines, indent * 2, "if _rope_theta is None:")
                add(lines, indent * 3, "_rope_theta = self._model_config.get('rope_theta', 10000.0)")
                add(lines, indent * 2, "_rope_scaling = getattr(config, 'rope_scaling', None)")
                add(lines, indent * 2, "if (hasattr(config, 'rope_parameters') and hasattr(config, 'layer_types')")
                add(lines, indent * 3, "and isinstance(getattr(config, 'rope_parameters', None), dict)")
                add(lines, indent * 3, "and isinstance(getattr(config, 'layer_types', None), (list, tuple))")
                add(lines, indent * 3, "and len(config.layer_types) > 0")
                add(lines, indent * 3, "and config.layer_types[0] in config.rope_parameters):")
                add(lines, indent * 3, "self.rotary_emb = nn.ModuleList([")
                add(lines, indent * 4, "get_rope(")
                add(lines, indent * 5, f"{head_dim_expr},")
                add(lines, indent * 5, "max_position=getattr(config, 'max_position_embeddings', 4096),")
                add(lines, indent * 5, f"is_neox_style={_is_neox},")
                add(lines, indent * 5, "rope_parameters=(")
                add(lines, indent * 6, "dict(config.rope_parameters[config.layer_types[i]])")
                add(lines, indent * 6, "if i < len(config.layer_types) and config.layer_types[i] in config.rope_parameters")
                if _has_partial_rotary:
                    add(lines, indent * 6, "else {**{'rope_type': 'default', 'rope_theta': _rope_theta}, **({'rope_dim': config.rotary_dim} if hasattr(config, 'rotary_dim') and config.rotary_dim is not None else {'partial_rotary_factor': getattr(config, 'partial_rotary_factor', getattr(config, 'rotary_pct', 1.0))})}")
                else:
                    add(lines, indent * 6, "else {'rope_type': 'default', 'rope_theta': _rope_theta}")
                add(lines, indent * 5, "),")
                add(lines, indent * 4, f") for i in range({num_layers_expr})")
                add(lines, indent * 3, "])")
                add(lines, indent * 2, "else:")
                add(lines, indent * 3, "if _rope_scaling is not None:")
                add(lines, indent * 4, "_rope_params = dict(_rope_scaling)")
                add(lines, indent * 4, "if 'rope_theta' not in _rope_params:")
                add(lines, indent * 5, "_rope_params['rope_theta'] = _rope_theta")
                add(lines, indent * 3, "else:")
                add(lines, indent * 4, "_rope_params = {'rope_type': 'default', 'rope_theta': _rope_theta}")
                if _has_partial_rotary:
                    add(lines, indent * 4, "_rd = getattr(config, 'rotary_dim', None)")
                    add(lines, indent * 4, "if _rd is not None:")
                    add(lines, indent * 5, "_rope_params['rope_dim'] = _rd")
                    add(lines, indent * 4, "else:")
                    add(lines, indent * 5, "_prf = getattr(config, 'partial_rotary_factor', getattr(config, 'rotary_pct', 1.0))")
                    add(lines, indent * 5, "_rope_params['partial_rotary_factor'] = _prf")
                add(lines, indent * 3, "self.rotary_emb = nn.ModuleList([")
                add(lines, indent * 4, f"get_rope({head_dim_expr},")
                add(lines, indent * 4, "max_position=getattr(config, 'max_position_embeddings', 4096),")
                add(lines, indent * 4, f"is_neox_style={_is_neox},")
                add(lines, indent * 4, "rope_parameters=_rope_params,")
                add(lines, indent * 4, f") for i in range({num_layers_expr})")
                add(lines, indent * 3, "])")
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
        _has_lm_head = False
        if classification.lm_head_node_id:
            lm_head_node = self._find_node_by_id(classification.lm_head_node_id)
            lm_head_attr = self._vllm_attr_access_init(lm_head_node) if lm_head_node else None
            if lm_head_attr:
                add(lines, indent * 2, f"self.lm_head = {lm_head_attr}")
                _has_lm_head = True
                # Handle tied word embeddings
                add(lines, indent * 2, "if getattr(config, 'tie_word_embeddings', False):")
                for emb_id in classification.embedding_node_ids:
                    if emb_id == classification.pli_embed_node_id:
                        continue
                    emb_node = self._find_node_by_id(emb_id)
                    if emb_node is not None:
                        emb_attr = self._vllm_attr_access_init(emb_node)
                        add(lines, indent * 3, f"self.lm_head = self.lm_head.tie_weights({emb_attr})")
                        break
        else:
            main_module = self.modules_by_name.get(self.program.main_module)
            if main_module is not None:
                for out in main_module.outputs:
                    if not isinstance(out, GraphExpr):
                        continue
                    if out.op.name not in ("_linear", "NN.linear"):
                        continue
                    add(lines, indent * 2, "from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead")
                    add(lines, indent * 2, "self.lm_head = ParallelLMHead(")
                    add(lines, indent * 3, f"{self._config_expr('vocab_size')}, {self._config_expr('embedding_size', alt='hidden_size')},")
                    add(lines, indent * 3, "bias=False,")
                    add(lines, indent * 3, f"prefix=f\"{{prefix}}.lm_head\",")
                    add(lines, indent * 3, "params_dtype=params_dtype,")
                    add(lines, indent * 2, ")")
                    _has_lm_head = True
                    add(lines, indent * 2, "if getattr(config, 'tie_word_embeddings', False):")
                    for emb_id in classification.embedding_node_ids:
                        if emb_id == classification.pli_embed_node_id:
                            continue
                        emb_node = self._find_node_by_id(emb_id)
                        if emb_node is not None:
                            emb_attr = self._vllm_attr_access_init(emb_node)
                            add(lines, indent * 3, f"self.lm_head = self.lm_head.tie_weights({emb_attr})")
                            break
                    break
        self._has_lm_head = _has_lm_head
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

        # Fused activation for merged gate_up_proj
        has_merged_ffn = any(
            g.gate_node_id and g.up_node_id for g in classification.ffn_groups
        )
        if has_merged_ffn:
            add(lines, indent * 2, "from vllm.model_executor.layers.activation import SiluAndMul, GeluAndMul")
            # Determine activation type from config
            add(lines, indent * 2, "_hidden_act = getattr(config, 'hidden_act', self._model_config.get('hidden_act', 'silu'))")
            add(lines, indent * 2, "if _hidden_act in ('gelu_pytorch_tanh', 'gelu_tanh'):")
            add(lines, indent * 3, "self._ffn_act = GeluAndMul(approximate='tanh')")
            add(lines, indent * 2, "elif _hidden_act in ('gelu', 'gelu_approx'):")
            add(lines, indent * 3, "self._ffn_act = GeluAndMul()")
            add(lines, indent * 2, "else:")
            add(lines, indent * 3, "self._ffn_act = SiluAndMul()")

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

        # Detect and create position embeddings (e.g. OPT learned position embeddings)
        pos_emb_info = self._detect_pos_embedding(classification)
        self._pos_emb_info = pos_emb_info
        if pos_emb_info is not None:
            pos_prefix, pos_dim = pos_emb_info
            pos_offset = self._detect_pos_offset()
            add(lines, indent * 2, f"self.pos_emb = VocabParallelEmbedding(")
            if pos_offset:
                add(lines, indent * 3, f"getattr(config, 'max_position_embeddings', 2048) + {pos_offset}, {pos_dim},")
            else:
                add(lines, indent * 3, f"getattr(config, 'max_position_embeddings', 2048), {pos_dim},")
            add(lines, indent * 3, f"prefix={pos_prefix},")
            add(lines, indent * 3, "params_dtype=params_dtype,")
            add(lines, indent * 2, ")")
            add(lines, indent * 2, f"self.pos_emb.prefix = {pos_prefix}")
            if pos_offset:
                add(lines, indent * 2, f"self._pos_offset = {pos_offset}")
            else:
                add(lines, indent * 2, "self._pos_offset = 0")

        add(lines, indent * 2, "self._build_state_dict_tensors()")
        add(lines, indent * 2, "self._eval_symbols()")
        add(lines, indent, "")
        add(lines, indent, "def embed_input_ids(self, input_ids):")
        _embed_candidates: list[tuple[str, str]] = []
        for node_id in sorted(classification.embedding_node_ids):
            if node_id == classification.pli_embed_node_id:
                continue
            node = self._find_node_by_id(node_id)
            if node is not None:
                attr = self._vllm_attr_access(node)
                param_name = attr.replace("self.", "", 1) + ".weight"
                _embed_candidates.append((attr, param_name))
        _has_emb_scale = hasattr(self, '_model_config_data') and self._model_config_data.get("embedding_scale")
        if len(_embed_candidates) > 1:
            for attr, param_name in _embed_candidates:
                add(lines, indent * 2, f"if hasattr(self, '_loaded_params') and {param_name!r} in self._loaded_params:")
                if _has_emb_scale:
                    add(lines, indent * 3, f"return {attr}(input_ids) * self.normalizer")
                else:
                    add(lines, indent * 3, f"return {attr}(input_ids)")
        if _embed_candidates:
            attr, _ = _embed_candidates[0]
            if _has_emb_scale:
                add(lines, indent * 2, f"return {attr}(input_ids) * self.normalizer")
            else:
                add(lines, indent * 2, f"return {attr}(input_ids)")
        else:
            add(lines, indent * 2, "return None")
        add(lines, indent, "")

        add(lines, indent, "def load_weights(self, weights):")
        add(lines, indent * 2, "import re as _re")
        add(lines, indent * 2, "stacked_params_mapping = [")
        _qkv_leaves_seen: set[str] = set()
        for g in classification.qkv_groups:
            if g.q_node_id == g.k_node_id:
                continue
            q_node = self._find_node_by_id(g.q_node_id)
            k_node = self._find_node_by_id(g.k_node_id)
            v_node = self._find_node_by_id(g.v_node_id)
            if q_node is None or k_node is None or v_node is None:
                continue
            for node, shard in [(q_node, "q"), (k_node, "k"), (v_node, "v")]:
                base = _linear_base_key(node)
                if not base:
                    continue
                leaf = base.rsplit(".", 1)[-1] if "." in base else base
                if leaf in ("q_proj", "k_proj", "v_proj") or leaf in _qkv_leaves_seen:
                    continue
                _qkv_leaves_seen.add(leaf)
                add(lines, indent * 3, f'(".qkv_proj", ".{leaf}", "{shard}"),')
        if ".q_proj" not in _qkv_leaves_seen:
            add(lines, indent * 3, '(".qkv_proj", ".q_proj", "q"),')
        if ".k_proj" not in _qkv_leaves_seen:
            add(lines, indent * 3, '(".qkv_proj", ".k_proj", "k"),')
        if ".v_proj" not in _qkv_leaves_seen:
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
        add(lines, indent * 4, "_prefix = _prefix.lstrip('.')")
        add(lines, indent * 4, "_resolved = _prefix")
        add(lines, indent * 4, "if '{i}' in _prefix:")
        add(lines, indent * 5, "_idx = None")
        add(lines, indent * 5, "for _p in reversed(_mod_name.split('.')):")
        add(lines, indent * 6, "if _p.isdigit():")
        add(lines, indent * 7, "_idx = int(_p)")
        add(lines, indent * 7, "break")
        add(lines, indent * 5, "if _idx is not None:")
        add(lines, indent * 6, "_resolved = _prefix.replace('{i}', str(_idx))")
        add(lines, indent * 5, "else:")
        add(lines, indent * 6, "continue")
        add(lines, indent * 4, "for _pname, _ in _module.named_parameters(recurse=False):")
        add(lines, indent * 5, "_ck = f'{_resolved}.{_pname}'")
        add(lines, indent * 5, "if _ck not in _ckpt_to_model:")
        add(lines, indent * 6, "_ckpt_to_model[_ck] = f'{_mod_name}.{_pname}'")
        add(lines, indent * 4, "for _bname, _ in _module.named_buffers(recurse=False):")
        add(lines, indent * 5, "_ck = f'{_resolved}.{_bname}'")
        add(lines, indent * 5, "if _ck not in _ckpt_to_model:")
        add(lines, indent * 6, "_ckpt_to_model[_ck] = f'{_mod_name}.{_bname}'")
        add(lines, indent * 4, "# Alias: modules whose prefix ends with '.decoder' may have")
        add(lines, indent * 4, "# checkpoint bias stored as prefix_without_decoder + '.bias'")
        add(lines, indent * 4, "if _resolved.endswith('.decoder') and 'bias' in {p for p, _ in _module.named_parameters(recurse=False)}:")
        add(lines, indent * 5, "_alt_ck = _resolved[:-8] + '.bias'")
        add(lines, indent * 5, "if _alt_ck not in _ckpt_to_model:")
        add(lines, indent * 6, "_ckpt_to_model[_alt_ck] = f'{_mod_name}.bias'")
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
        add(lines, indent * 2, "# Build set of checkpoint weight names that need transposition (Conv1D)")
        add(lines, indent * 2, "_transposed_ck_weights = set()")
        add(lines, indent * 2, "for _mod_name, _module in self.named_modules():")
        add(lines, indent * 3, "if not getattr(_module, '_bs_transposed', False):")
        add(lines, indent * 4, "continue")
        add(lines, indent * 3, "_prefix = getattr(_module, 'prefix', None)")
        add(lines, indent * 3, "if not _prefix:")
        add(lines, indent * 4, "continue")
        add(lines, indent * 3, "_prefix = _prefix.lstrip('.')")
        add(lines, indent * 3, "_resolved = _prefix")
        add(lines, indent * 3, "if '{i}' in _prefix:")
        add(lines, indent * 4, "_idx = None")
        add(lines, indent * 4, "for _p in reversed(_mod_name.split('.')):")
        add(lines, indent * 5, "if _p.isdigit():")
        add(lines, indent * 6, "_idx = int(_p)")
        add(lines, indent * 6, "break")
        add(lines, indent * 4, "if _idx is not None:")
        add(lines, indent * 5, "_resolved = _prefix.replace('{i}', str(_idx))")
        add(lines, indent * 3, "_transposed_ck_weights.add(f'{_resolved}.weight')")
        add(lines, indent * 2, "for _pname, _wname, _sid in stacked_params_mapping:")
        add(lines, indent * 3, "for _ck in list(_transposed_ck_weights):")
        add(lines, indent * 4, "_individual = _ck.replace(_pname, _wname)")
        add(lines, indent * 4, "if _individual != _ck:")
        add(lines, indent * 5, "_transposed_ck_weights.add(_individual)")
        add(lines, indent * 2, "")
        add(lines, indent * 2, "for name, loaded_weight in weights:")
        self._emit_weight_loading_body(lines, classification, indent * 3)
        add(lines, indent * 2, "self._loaded_params = loaded_params")
        add(lines, indent * 2, "# Re-evaluate symbols after weight loading so has_root")
        add(lines, indent * 2, "# checks reflect actual checkpoint keys")
        add(lines, indent * 2, "self._eval_symbols()")
        add(lines, indent * 2, "# Move state_dict_tensors to the model's device")
        add(lines, indent * 2, "try:")
        add(lines, indent * 3, "_params_device = next(self.parameters()).device")
        add(lines, indent * 3, "for _k, _v in list(self.state_dict_tensors.items()):")
        add(lines, indent * 4, "if torch.is_tensor(_v) and _v.device != _params_device:")
        add(lines, indent * 5, "self.state_dict_tensors[_k] = _v.to(_params_device)")
        add(lines, indent * 2, "except StopIteration:")
        add(lines, indent * 3, "pass")
        if self._unit_offset_norm_attrs:
            add(lines, indent * 2, "# Add 1.0 to unit_offset RMSNorm weights (Gemma-family)")
            for attr_name, is_repeated in self._unit_offset_norm_attrs:
                if is_repeated:
                    add(lines, indent * 2, f"for _mod in self.{attr_name}:")
                    add(lines, indent * 3, "_mod.weight.data.add_(1.0)")
                else:
                    add(lines, indent * 2, f"self.{attr_name}.weight.data.add_(1.0)")
        add(lines, indent, "")

        add(lines, indent, "def compute_logits(self, hidden_states):")
        add(lines, indent * 2, "hidden_states = hidden_states.to(self._params_dtype)")
        add(lines, indent * 2, "_vocab = getattr(self.config, 'vocab_size', self._model_config.get('vocab_size', 0))")
        add(lines, indent * 2, "if _vocab and hidden_states.shape[-1] == _vocab:")
        add(lines, indent * 3, "return hidden_states")
        add(lines, indent * 2, "return self.logits_processor(self.lm_head, hidden_states, getattr(self.lm_head, 'bias', None))")
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
        add(lines, indent * 3, "_prefix = _prefix.lstrip('.')")
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
        add(lines, indent, "def _apply(self, fn, recurse=True):")
        add(lines, indent * 2, "result = super()._apply(fn, recurse=recurse)")
        add(lines, indent * 2, "for _k, _v in list(result.state_dict_tensors.items()):")
        add(lines, indent * 3, "if torch.is_tensor(_v):")
        add(lines, indent * 4, "result.state_dict_tensors[_k] = fn(_v)")
        add(lines, indent * 2, "return result")
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

        repeated_mod = self._get_repeated_module()

        if repeated_mod is None:
            self._emit_forward_legacy(lines)
            return

        # Encoder-decoder models have 2+ repeated module types that each
        # contain QKV/attention (encoder + decoder blocks).  The clean
        # forward path only supports a single repeated module type, so fall
        # back to legacy for encoder-decoder models.
        _qkv_repeated = {
            g.q_node_id.split(":")[0] for g in cls.qkv_groups
            if ":" in g.q_node_id
        } & cls.repeated_module_names
        if len(_qkv_repeated) > 1:
            self._emit_forward_legacy(lines)
            return

        _is_hybrid_mamba_fwd = bool(
            cls.mamba_mixer_module_names and cls.qkv_groups
        )
        if cls.mamba_mixer_module_names and not _is_hybrid_mamba_fwd:
            self._emit_forward_ssm(lines, cls)
            return

        # Hybrid mamba models (attention + mamba/moe per-layer conditionals)
        # need the legacy forward to handle per-layer branching.
        if _is_hybrid_mamba_fwd:
            self._emit_forward_legacy(lines)
            return

        # --- Analyze per-layer structure ---
        layer_norms = self._analyze_layer_norms(repeated_mod, cls)
        if not layer_norms:
            self._emit_forward_legacy(lines)
            return

        # Fall back to legacy forward if there are too many norms — more than
        # 4 indicates core.select branches with alternative norm paths (e.g.
        # Gemma3 sliding vs full attention) that the clean forward can't handle.
        # Count ALL norms (including QK/V norms) for this check, since models
        # with QK norms (e.g. Gemma3) also have the sliding/full attention
        # select branches that the clean forward can't handle yet.
        _total_norm_count = len(layer_norms) + len(cls.qk_norm_node_ids) + len(cls.v_norm_node_ids)
        if _total_norm_count > 4:
            self._emit_forward_legacy(lines)
            return

        # Fall back to legacy forward if there are no QKV groups — the clean
        # forward path can only emit attention when QKV is classified.
        if not cls.qkv_groups:
            self._emit_forward_legacy(lines)
            return

        # Fall back to legacy forward if there are embedding calls outside the
        # repeated module that aren't classified as VOCAB_PARALLEL_EMBEDDING
        # and aren't position embeddings (e.g. learned position embeddings in
        # OPT/GPT-2).  Position embeddings are handled by the clean forward
        # via self.pos_emb.  Other unclassified embeddings require legacy.
        for module in self.program.modules:
            for node in module.nodes:
                if (
                    node.id not in cls.embedding_node_ids
                    and _is_embedding_call(node, self.modules_by_name)
                ):
                    # Check if this is a position embedding
                    path_inp = node.inputs[0] if node.inputs else None
                    if isinstance(path_inp, GraphPath) and path_inp.parts:
                        path_str = ".".join(path_inp.parts).lower()
                        if "position" in path_str or "embed_pos" in path_str:
                            continue  # Will be handled as position embedding
                    self._emit_forward_legacy(lines)
                    return
            # Fall back if the main module has position operations that the
            # clean forward can't handle.  position_ids and rope_* are safe
            # (positions are passed as a parameter; RoPE is handled by
            # self.rotary_emb).  Only sinusoidal_positions and
            # linear_position_bias_for_input require legacy fallback.
            # Also check inside core.select branches for these ops.
            _blocked_ops = {
                "Positions.sinusoidal_positions",
                "Positions.linear_position_bias_for_input",
                "Positions.linear_position_bias",
            }
            main_mod = self.modules_by_name.get(self.program.main_module)
            if main_mod is not None:
                for node in main_mod.nodes:
                    if node.op.name in _blocked_ops:
                        self._emit_forward_legacy(lines)
                        return
                    if node.op.name == "core.select":
                        for inp in node.inputs:
                            if hasattr(inp, "op") and inp.op.name in _blocked_ops:
                                self._emit_forward_legacy(lines)
                                return

        qkv_group = cls.qkv_groups[0] if cls.qkv_groups else None
        ffn_group = cls.ffn_groups[0] if cls.ffn_groups else None
        o_proj_id = self._find_o_proj_id(repeated_mod, cls)
        rope_id = self._detect_rope(repeated_mod)
        act_name = self._detect_activation(repeated_mod)

        # Find final norm (non-repeated norm — RMSNorm or LayerNorm)
        final_norm_attr = None
        final_norm_candidates: list[str] = []
        for nid in sorted(cls.rmsnorm_node_ids | cls.layernorm_node_ids):
            node = self._find_node_by_id(nid)
            if node is None:
                continue
            # Skip norms inside the repeated module (per-layer norms)
            if self._is_repeated_node(node):
                continue
            # Skip norms inside loop body modules (also per-layer)
            mod_name = self._node_module_name(node)
            if "__loop_" in mod_name:
                continue
            attr = self._vllm_attr_access(node)
            final_norm_candidates.append(attr)
        if final_norm_candidates:
            final_norm_attr = final_norm_candidates[0]

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
        add(lines, 8, "attn_metadata=None,")
        add(lines, 8, "**kwargs,")
        add(lines, 8, "):")
        add(lines, 8, "if inputs_embeds is not None:")
        add(lines, 12, "hidden_states = inputs_embeds")
        add(lines, 8, "else:")
        add(lines, 12, "hidden_states = self.embed_input_ids(input_ids)")
        add(lines, 12, "_emb_mult = self._config('embedding_multiplier', 1.0)")
        add(lines, 12, "if _emb_mult != 1.0:")
        add(lines, 16, "hidden_states = hidden_states * _emb_mult")
        # Add learned position embeddings (e.g. OPT)
        if self._pos_emb_info is not None:
            add(lines, 12, "if hasattr(self, 'pos_emb') and positions is not None:")
            add(lines, 16, "hidden_states = hidden_states + self.pos_emb(positions + self._pos_offset)")
        add(lines, 8, "self._attn_metadata = attn_metadata")
        add(lines, 8, "self._positions = positions")
        add(lines, 8, "if getattr(self, '_vllm_native_mode', False) and positions is not None:")
        add(lines, 12, "_np = positions if positions.dim() > 1 else positions.unsqueeze(0)")
        add(lines, 12, "for _a in dir(self):")
        add(lines, 16, "if _a.startswith('_def_Positions_position_ids'):")
        add(lines, 20, "setattr(self, _a, lambda *a, _np=_np, **kw: (_np,))")
        add(lines, 8, "_residual_mult = self._config('residual_multiplier', 1.0)")
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
        any_fused = any(f for _, _, _, f, _ in layer_norms) and not has_pli

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

        add(lines, 8, f"if hidden_states.dim() == 3:")
        add(lines, 8, f"    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])")
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

        # Determine which norm index is the pre-FFN norm by tracing from
        # gate_proj's input back to find which norm feeds into it.
        ffn_norm_idx = len(layer_norms) - 1
        if ffn_group and ffn_group.gate_node_id and repeated_mod is not None:
            norm_nids = {nid for nid, _, _, _, _ in layer_norms}
            gate_node = self._find_node_by_id(ffn_group.gate_node_id)
            if gate_node is not None and gate_node.inputs:
                gate_input = gate_node.inputs[1] if isinstance(gate_node.inputs[0], GraphPath) else gate_node.inputs[0]
                src = _trace_back(repeated_mod, gate_input, norm_nids)
                if src is not None:
                    for i, (nid, _, _, _, _) in enumerate(layer_norms):
                        if nid == src:
                            ffn_norm_idx = i
                            break

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
            norm_ids_set = {nid for nid, _, _, _, _ in layer_norms}
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

        # Detect parallel residual: the module's output is an inline
        # core.binary.+ expression (e.g. x + attn_out + ffn_out), indicating
        # the attention and FFN outputs are added to the original residual
        # simultaneously rather than sequentially (e.g. GPT-NeoX/Pythia with
        # use_parallel_residual=True).  In this pattern the second norm
        # (post_attention_layernorm) takes the original input, not the
        # attention output + residual.
        #
        # Also detect parallel residual when both attention and FFN share a
        # single norm (ffn_norm_idx == 0) — both subblocks consume the same
        # normed input, so the residual add must happen after both, not
        # between them (e.g. GPT-J).
        has_parallel_residual = False
        if repeated_mod is not None:
            if not has_standalone_attn_add and not has_standalone_ffn_add:
                for out in repeated_mod.outputs:
                    if hasattr(out, "op") and out.op and out.op.name == "core.binary.+":
                        has_parallel_residual = True
                        break
            elif ffn_norm_idx == 0 and (has_standalone_attn_add or has_standalone_ffn_add):
                has_parallel_residual = True
                has_standalone_attn_add = False
                has_standalone_ffn_add = False

        # Determine if we can use fused RMSNorm (residual carry across layers).
        # This matches native vLLM's pattern: norm(x, residual) does the add
        # inside the kernel.  Disabled when:
        # - PLI present: needs explicit residual before PLI
        # - per-layer scalar with residual add: needs explicit residual
        # - residual_multiplier != 1.0: fused norm does residual + x without
        #   a multiplier, so we can't fuse (granite uses residual + x * mult)
        _has_resid_mult = self._model_config_data.get("residual_multiplier", 1.0) != 1.0
        _has_layernorm = any(
            cls.node_types.get(nid) == VLLMLayerType.LAYERNORM
            for nid, _, _, _, _ in layer_norms
        )
        # When there are no standalone adds and some non-first norms use the
        # unfused uses_res pattern (norm → add residual → R=H), the residual
        # track equals hidden_states at layer end.  The fused first norm would
        # double-count (norm(H+R) = norm(2*H)), so disable it.
        _has_unfused_resid_norm = (
            not (has_standalone_attn_add or has_standalone_ffn_add or has_parallel_residual)
            and any(
                uses_res and not fused
                for _i, (_, _, uses_res, fused, _) in enumerate(layer_norms)
                if _i > 0
            )
        )
        first_norm_fused = (
            not has_pli
            and not (has_per_layer_scalar and cls.per_layer_scalar_has_residual_add)
            and not _has_resid_mult
            and not _has_layernorm
            and not _has_unfused_resid_norm
        )

        # When first_norm_fused AND there are standalone explicit adds (Llama
        # pattern with 2 norms), fold those adds into fused norms: override all
        # non-first norms to fused and suppress the standalone adds.
        # When first_norm_fused but NO standalone adds (Gemma2 pattern with 4
        # norms), the _analyze_layer_norms heuristic already set the correct
        # fused flags — don't override them.
        if first_norm_fused and (has_standalone_attn_add or has_standalone_ffn_add):
            _layer_norms = []
            for _i, (nid, attr, uses_res, fused, post_norm) in enumerate(layer_norms):
                if _i == 0:
                    _layer_norms.append((nid, attr, False, False, post_norm))
                else:
                    _layer_norms.append((nid, attr, True, True, post_norm))
            layer_norms = _layer_norms
            has_standalone_attn_add = False
            has_standalone_ffn_add = False

        # Unified per-norm pattern: each norm uses its own fused flag.
        _attn_emitted = False
        _ffn_emitted = False
        for norm_i, (nid, attr, uses_res, fused, post_norm) in enumerate(layer_norms):
            if post_norm:
                # Post-norm: the subblock runs before the norm, but only if a
                # pre-norm hasn't already emitted it (e.g. Gemma-2 has both
                # pre-attention and post-attention norms; OLMo-2 has only post).
                if norm_i == 0 and not _attn_emitted:
                    add(lines, indent, f"residual = hidden_states")
                    if qkv_group:
                        self._emit_attn_block(lines, indent, cls, qkv_group, rope_id, o_proj_id)
                    _attn_emitted = True
                if norm_i == ffn_norm_idx and not _ffn_emitted:
                    if ffn_group:
                        self._emit_ffn_block(lines, indent, cls, ffn_group, act_name)
                    _ffn_emitted = True
                add(lines, indent, f"hidden_states = {attr}(hidden_states)")
                add(lines, indent, f"hidden_states = hidden_states + residual")
                add(lines, indent, f"residual = hidden_states")
                continue

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
            elif fused:
                add(lines, indent, f"hidden_states, residual = {attr}(hidden_states, residual)")
            elif uses_res:
                add(lines, indent, f"hidden_states = {attr}(hidden_states)")
                add(lines, indent, f"hidden_states = hidden_states + residual")
                add(lines, indent, f"residual = hidden_states")
            elif not (has_parallel_residual and norm_i == ffn_norm_idx):
                add(lines, indent, f"hidden_states = {attr}(hidden_states)")

            if norm_i == 0 and qkv_group:
                if has_parallel_residual and ffn_norm_idx == 0:
                    add(lines, indent, f"_xn = hidden_states")
                self._emit_attn_block(lines, indent, cls, qkv_group, rope_id, o_proj_id)
                _attn_emitted = True
                if has_parallel_residual:
                    add(lines, indent, f"_attn_out = hidden_states")
                    if ffn_norm_idx == 0:
                        add(lines, indent, f"hidden_states = _xn")
                elif has_standalone_attn_add:
                    add(lines, indent, f"hidden_states = residual + hidden_states * _residual_mult")
                    add(lines, indent, f"residual = hidden_states")
            if norm_i == ffn_norm_idx and ffn_group:
                if has_parallel_residual and ffn_norm_idx != 0:
                    add(lines, indent, f"hidden_states = {attr}(residual)")
                self._emit_ffn_block(lines, indent, cls, ffn_group, act_name)
                _ffn_emitted = True
                if has_parallel_residual:
                    add(lines, indent, f"hidden_states = residual + _attn_out + hidden_states")
                elif has_standalone_ffn_add:
                    add(lines, indent, f"hidden_states = residual + hidden_states * _residual_mult")

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
            if len(final_norm_candidates) > 1:
                for _idx, cand_attr in enumerate(final_norm_candidates):
                    cand_param = cand_attr.replace("self.", "", 1) + ".weight"
                    kw = "if" if _idx == 0 else "elif"
                    add(lines, 8, f"{kw} hasattr(self, '_loaded_params') and {cand_param!r} in self._loaded_params:")
                    if first_norm_fused:
                        add(lines, 12, f"hidden_states, _ = {cand_attr}(hidden_states, residual)")
                    else:
                        add(lines, 12, f"hidden_states = {cand_attr}(hidden_states)")
                add(lines, 8, "else:")
                if first_norm_fused:
                    add(lines, 12, f"hidden_states, _ = {final_norm_attr}(hidden_states, residual)")
                else:
                    add(lines, 12, f"hidden_states = {final_norm_attr}(hidden_states)")
            elif first_norm_fused:
                add(lines, 8, f"hidden_states, _ = {final_norm_attr}(hidden_states, residual)")
            else:
                add(lines, 8, f"hidden_states = {final_norm_attr}(hidden_states)")

        # Apply LM head via compute_logits when the model has a classified LM head.
        # In native vLLM mode, forward() returns hidden states and vLLM calls
        # compute_logits() separately.
        if self._has_lm_head:
            add(lines, 8, "if not getattr(self, '_vllm_native_mode', False):")
            add(lines, 12, "if hidden_states.dim() == 3:")
            add(lines, 16, "hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])")
            add(lines, 12, "_compute_logits = getattr(self, 'compute_logits', None)")
            add(lines, 12, "if callable(_compute_logits) and hidden_states.shape[-1] != getattr(self.config, 'vocab_size', 0):")
            add(lines, 16, "hidden_states = _compute_logits(hidden_states)")
            add(lines, 16, "_logits_scaling = self._config('logits_scaling', 1.0)")
            add(lines, 16, "if _logits_scaling != 1.0:")
            add(lines, 20, "hidden_states = hidden_states / _logits_scaling")
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
        add(lines, indent, f"qkv, _qkv_bias = {q_attr}(hidden_states)")
        add(lines, indent, f"if _qkv_bias is not None: qkv = qkv + _qkv_bias")
        add(lines, indent, f"q, k, v = qkv.split([_q_size, _kv_size, _kv_size], dim=-1)")

        # QK norms with reshape
        qk_norms = sorted(cls.qk_norm_node_ids)
        has_var_hd = self._detect_head_dim_expr(self._get_repeated_module() or repeated_mod, "i") is not None if self._get_repeated_module() else False
        head_dim_ref = "_hd" if has_var_hd else "_head_dim"

        # Separate Q norm (always) from K norm (skip for KV-shared)
        k_norm_attr = None
        k_norm_nid = None
        for nid in qk_norms:
            node = self._find_node_by_id(nid)
            attr = self._vllm_attr_access(node)
            is_k_norm = nid in cls.qk_norm_k_node_ids
            is_pre_reshape = nid in cls.qk_norm_pre_reshape_node_ids
            if is_k_norm:
                k_norm_attr = attr
                k_norm_nid = nid
            else:
                if is_pre_reshape:
                    add(lines, indent, f"q = {attr}(q)")
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
                if k_norm_nid and k_norm_nid in cls.qk_norm_pre_reshape_node_ids:
                    add(lines, indent + 1, f"k = {k_norm_attr}(k)")
                else:
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
                if k_norm_nid and k_norm_nid in cls.qk_norm_pre_reshape_node_ids:
                    add(lines, indent + 1, f"k = {k_norm_attr}(k)")
                else:
                    add(lines, indent + 1, f"k = k.unflatten(-1, (_num_kv_heads, {head_dim_ref}))")
                    add(lines, indent + 1, f"k = {k_norm_attr}(k)")
                    add(lines, indent + 1, f"k = k.flatten(-2, -1)")
            if v_norm_attr:
                if not k_norm_attr:
                    add(lines, indent, f"if not _is_kv_shared:")
                add(lines, indent + 1, f"v = v.unflatten(-1, (_num_kv_heads, {head_dim_ref}))")
                add(lines, indent + 1, f"v = {v_norm_attr}(v)")
                add(lines, indent + 1, f"v = v.flatten(-2, -1)")

        # Attention: use vLLM Attention layer (unified_attention_with_output)
        if attn_node:
            attn_attr = self._vllm_attr_access(attn_node)
            add(lines, indent, f"attn_out = {attn_attr}(q, k, v)")
        else:
            add(lines, indent, f"attn_out = v")

        # O projection
        if o_proj_id:
            o_node = self._find_node_by_id(o_proj_id)
            o_attr = self._vllm_attr_access(o_node)
            add(lines, indent, f"hidden_states, _o_bias = {o_attr}(attn_out.to(self._params_dtype))")
            add(lines, indent, f"if _o_bias is not None: hidden_states = hidden_states + _o_bias")

    def _find_xielu_params(self, repeated_mod: Any) -> tuple[str, str, str, str] | None:
        """Find the xielu activation node in the repeated module and extract
        the parameter paths (alpha_p, alpha_n, beta, eps)."""
        if repeated_mod is None:
            return None
        for node in repeated_mod.nodes:
            if "xielu" not in node.op.name.lower():
                continue
            if len(node.inputs) < 5:
                continue
            paths = []
            for inp in node.inputs[1:5]:
                if isinstance(inp, GraphPath) and inp.parts:
                    path_str = ".".join(inp.parts)
                    path_str = path_str.replace("{i}", "{self._path_template_part(i)}")
                    paths.append(path_str)
                else:
                    return None
            if len(paths) == 4:
                return (paths[0], paths[1], paths[2], paths[3])
        return None

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
            add(lines, indent, f"gate_up, _gu_bias = {gate_attr}(hidden_states)")
            add(lines, indent, f"if _gu_bias is not None: gate_up = gate_up + _gu_bias")
            add(lines, indent, f"hidden_states = self._ffn_act(gate_up)")
        elif ffn_group.up_node_id:
            up_node = self._find_node_by_id(ffn_group.up_node_id)
            up_attr = self._vllm_attr_access(up_node)
            add(lines, indent, f"hidden_states, _up_bias = {up_attr}(hidden_states)")
            add(lines, indent, f"if _up_bias is not None: hidden_states = hidden_states + _up_bias")
            if act_name:
                if act_name.endswith("xielu"):
                    repeated_mod = self._get_repeated_module()
                    xielu_params = self._find_xielu_params(repeated_mod)
                    if xielu_params:
                        p_alpha_p, p_alpha_n, p_beta, p_eps = xielu_params
                        add(lines, indent, f"hidden_states = self._xielu(hidden_states, self._param(f'{p_alpha_p}'), self._param(f'{p_alpha_n}'), self._param(f'{p_beta}'), self._param(f'{p_eps}'))")
                    else:
                        add(lines, indent, f"hidden_states = F.gelu(hidden_states)")
                else:
                    act_code = self._activation_to_code(act_name)
                    add(lines, indent, f"hidden_states = {act_code.format(x='hidden_states')}")

        if ffn_group.down_node_id:
            down_node = self._find_node_by_id(ffn_group.down_node_id)
            down_attr = self._vllm_attr_access(down_node)
            add(lines, indent, f"hidden_states, _down_bias = {down_attr}(hidden_states)")
            add(lines, indent, f"if _down_bias is not None: hidden_states = hidden_states + _down_bias")

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
        add(lines, 12, "_emb_mult = self._config('embedding_multiplier', 1.0)")
        add(lines, 12, "if _emb_mult != 1.0:")
        add(lines, 16, "hidden_states = hidden_states * _emb_mult")
        add(lines, 8, "")
        add(lines, 8, "config = self.config")
        add(lines, 8, f"_num_layers = {num_layers_expr}")
        add(lines, 8, "residual = None")
        add(lines, 8, "_residual_mult = self._config('residual_multiplier', 1.0)")
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
        if self._has_lm_head:
            add(lines, 8, "if not getattr(self, '_vllm_native_mode', False):")
            add(lines, 12, "if hidden_states.dim() == 3:")
            add(lines, 16, "hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])")
            add(lines, 12, "_compute_logits = getattr(self, 'compute_logits', None)")
            add(lines, 12, "if callable(_compute_logits) and hidden_states.shape[-1] != getattr(self.config, 'vocab_size', 0):")
            add(lines, 16, "hidden_states = _compute_logits(hidden_states)")
            add(lines, 16, "_logits_scaling = self._config('logits_scaling', 1.0)")
            add(lines, 16, "if _logits_scaling != 1.0:")
            add(lines, 20, "hidden_states = hidden_states / _logits_scaling")
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
        add(lines, 8, "**kwargs,")
        add(lines, 8, "):")
        add(lines, 8, "if inputs_embeds is not None:")
        add(lines, 12, "hidden_states = inputs_embeds")
        add(lines, 8, "else:")
        add(lines, 12, "hidden_states = self.embed_input_ids(input_ids)")
        add(lines, 8, "")
        add(lines, 8, "self._attn_metadata = attn_metadata")
        add(lines, 8, "self._positions = positions")
        add(lines, 8, "if getattr(self, '_vllm_native_mode', False) and positions is not None:")
        add(lines, 12, "_np = positions if positions.dim() > 1 else positions.unsqueeze(0)")
        add(lines, 12, "for _a in dir(self):")
        add(lines, 16, "if _a.startswith('_def_Positions_position_ids'):")
        add(lines, 20, "setattr(self, _a, lambda *a, _np=_np, **kw: (_np,))")
        add(lines, 12, "if _a.startswith('_def_Positions_linear_position_bias'):")
        add(lines, 16, "setattr(self, _a, lambda *a, **kw: (None,))")
        add(lines, 8, "")
        add(lines, 8, "if not getattr(self, '_state', None) and hasattr(self, 'state_dict_tensors'):")
        add(lines, 12, "self._state = dict(self.state_dict_tensors)")
        add(lines, 8, "if input_ids is not None and input_ids.dim() == 1:")
        add(lines, 12, "input_ids = input_ids.unsqueeze(0)")
        add(lines, 8, "_attn_mask = kwargs.pop('attn_mask', kwargs.pop('attention_mask', None))")
        add(lines, 8, "if _attn_mask is None and input_ids is not None and getattr(self, '_vllm_native_mode', False):")
        add(lines, 12, "_attn_mask = torch.ones(input_ids.shape[0], input_ids.shape[1], device=input_ids.device, dtype=torch.bool)")
        call_args = []
        for value in main.inputs:
            vname = _py_ident(value.name)
            if vname in ("input_ids",):
                call_args.append(f"{vname}=input_ids")
            elif vname in ("attn_mask", "attention_mask"):
                call_args.append(f"{vname}=_attn_mask")
            elif vname in ("decoder_input_ids", "decoder_attention_mask"):
                call_args.append(f"{vname}=kwargs.get({vname!r}, None)")
            else:
                call_args.append(f"{vname}=kwargs.get({vname!r}, None)")
        add(lines, 8, f"result = self.{self.method_names[main.name]}({', '.join(call_args)})")
        add(lines, 8, "if isinstance(result, (tuple, list)):")
        add(lines, 12, "result = result[0]")
        add(lines, 8, "if result.dim() == 3:")
        add(lines, 12, "result = result.reshape(-1, result.shape[-1])")
        add(lines, 8, "if not getattr(self, '_vllm_native_mode', False):")
        add(lines, 12, "_compute_logits = getattr(self, 'compute_logits', None)")
        add(lines, 12, "if callable(_compute_logits) and result.shape[-1] != getattr(self.config, 'vocab_size', 0):")
        add(lines, 16, "result = _compute_logits(result)")
        add(lines, 16, "_logits_scaling = self._config('logits_scaling', 1.0)")
        add(lines, 16, "if _logits_scaling != 1.0:")
        add(lines, 20, "result = result / _logits_scaling")
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
        # Check if the main module has ops that force legacy forward
        # (e.g. ALiBi position bias).  If so, don't use clean forward
        # because the legacy forward calls the full module method which
        # references all node attributes — skipping them in __init__
        # would cause AttributeError at runtime.
        _force_legacy = False
        main_mod = self.modules_by_name.get(self.program.main_module)
        if main_mod is not None:
            _blocked_ops = {
                "Positions.sinusoidal_positions",
                "Positions.linear_position_bias_for_input",
                "Positions.linear_position_bias",
            }
            for node in main_mod.nodes:
                if node.op.name in _blocked_ops:
                    _force_legacy = True
                    break
                if node.op.name == "core.select":
                    for inp in node.inputs:
                        if hasattr(inp, "op") and inp.op.name in _blocked_ops:
                            _force_legacy = True
                            break
                    if _force_legacy:
                        break
        _qkv_repeated = {
            g.q_node_id.split(":")[0] for g in classification.qkv_groups
            if ":" in g.q_node_id
        } & classification.repeated_module_names
        if (
            classification.repeated_module_names
            and len(_qkv_repeated) <= 1
            and not _force_legacy
        ):
            _repeated_mod_name = max(
                classification.repeated_module_names,
                key=lambda n: len(self.modules_by_name[n].nodes) if n in self.modules_by_name else 0,
            )
            _repeated_mod = self.modules_by_name.get(_repeated_mod_name)
            if _repeated_mod is not None:
                _layer_norms = self._analyze_layer_norms(_repeated_mod, classification)
                if _layer_norms:
                    use_clean_forward = True
        _is_hybrid_mamba = bool(
            classification.mamba_mixer_module_names
            and classification.qkv_groups
        )
        if _is_hybrid_mamba:
            use_clean_forward = False
        if classification.mamba_mixer_module_names and not _is_hybrid_mamba:
            use_clean_forward = True
            for mamba_mod_name in classification.mamba_mixer_module_names:
                mamba_mod = self.modules_by_name.get(mamba_mod_name)
                if mamba_mod is not None:
                    for node in mamba_mod.nodes:
                        skip_node_ids.add(node.id)
        self._use_clean_forward = use_clean_forward
        if use_clean_forward:
            if classification.lm_head_node_id:
                lm_head_node = self._find_node_by_id(classification.lm_head_node_id)
                if lm_head_node is not None and lm_head_node.op.name == "core.select":
                    for inp in lm_head_node.inputs:
                        if isinstance(inp, GraphExpr):
                            branch_mod = self.modules_by_name.get(inp.op.name)
                            if branch_mod is not None:
                                for bn in branch_mod.nodes:
                                    # Skip linear nodes (the LM head itself)
                                    # but keep embedding and norm nodes which
                                    # are needed for __init__ and forward.
                                    if _is_linear_call(bn, self.modules_by_name):
                                        skip_node_ids.add(bn.id)
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
                if g.up_node_id != g.gate_node_id:
                    ffn_up_to_skip.add(g.up_node_id)
        self._ffn_gate_to_up = ffn_gate_to_up
        # Deduplicate v_norms: keep only one (they are identical RMSNorm has_weight=False).
        # Multiple v_norms arise from alternative KV modules (e.g. gemma4_kv_shared
        # vs gemma4_kv_separate) where only one path is taken at runtime.
        v_norm_ids = sorted(classification.v_norm_node_ids)
        if len(v_norm_ids) > 1:
            for nid in v_norm_ids[1:]:
                skip_node_ids.add(nid)
        # Deduplicate attention nodes that share the same resolved prefix.
        # Multiple attention nodes arise from conditional branches (e.g. phi3small
        # dense vs blocksparse attention) where only one path is taken at
        # runtime, but both would create vLLM Attention layers with the same
        # prefix, causing "Duplicate layer name" errors.
        _seen_attn_prefixes: dict[str, str] = {}
        for module in self.program.modules:
            for node in module.nodes:
                if classification.layer_type(node) != VLLMLayerType.ATTENTION:
                    continue
                prefix = self._layer_prefix(node)
                if prefix in _seen_attn_prefixes:
                    skip_node_ids.add(node.id)
                    self._node_id_aliases[node.id] = _seen_attn_prefixes[prefix]
                else:
                    _seen_attn_prefixes[prefix] = node.id
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

        # Skip nested expert loop nodes — their weights are handled by
        # state_dict_tensors and the _expert_linear method, not vLLM modules.
        for module in self.program.modules:
            for node in module.nodes:
                if self._is_nested_loop_node(node):
                    skip_node_ids.add(node.id)

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
                is_nested = self._is_nested_loop_node(node)
                inner: list[str] = []
                self._emit_single_layer_init(inner, node, layer_type, classification, indent + 1)
                if is_nested:
                    expert_var = self._get_nested_loop_var(node) or "e"
                    expert_count = self._get_expert_count_expr()
                    add(lines, indent, f"self.{attr_name} = nn.ModuleList([")
                    add(lines, indent + 1, "nn.ModuleList([")
                    lines.extend(inner)
                    add(lines, indent + 2, f"for {expert_var} in range({expert_count})")
                    add(lines, indent + 1, "])")
                    add(lines, indent + 1, f"for i in range({num_layers_expr})")
                    add(lines, indent, "])")
                elif is_repeated:
                    add(lines, indent, f"self.{attr_name} = nn.ModuleList([")
                    lines.extend(inner)
                    add(lines, indent + 1, f"for i in range({num_layers_expr})")
                    add(lines, indent, "])")
                else:
                    add(lines, indent, f"self.{attr_name} = " + inner[0].strip())
                    lines.extend(inner[1:])
                if node.id in classification.transposed_linear_node_ids:
                    if is_nested:
                        add(lines, indent, f"for _mod in self.{attr_name}:")
                        add(lines, indent + 1, f"for _sub in _mod:")
                        add(lines, indent + 2, "_sub._bs_transposed = True")
                    elif is_repeated:
                        add(lines, indent, f"for _mod in self.{attr_name}:")
                        add(lines, indent + 1, "_mod._bs_transposed = True")
                    else:
                        add(lines, indent, f"self.{attr_name}._bs_transposed = True")
                # Modules that don't store prefix in __init__ need it set after creation.
                if layer_type in (
                    VLLMLayerType.RMSNORM,
                    VLLMLayerType.LAYERNORM,
                    VLLMLayerType.VOCAB_PARALLEL_EMBEDDING,
                    VLLMLayerType.PARALLEL_LM_HEAD,
                ):
                    prefix_expr = self._layer_prefix(node)
                    prefix_setters.append((attr_name, prefix_expr, is_repeated, is_nested))
                # Track unit_offset RMSNorms for post-load weight adjustment.
                if (
                    layer_type == VLLMLayerType.RMSNORM
                    and "rmsnorm_noscale" not in node.op.name
                    and self._detect_rmsnorm_unit_offset(node)
                ):
                    self._unit_offset_norm_attrs.append((attr_name, is_repeated))
        # Emit prefix-setting code for modules that need it
        if prefix_setters:
            add(lines, indent, "# Set prefix on modules that don't accept it in __init__")
            for attr_name, prefix_expr, is_repeated, is_nested in prefix_setters:
                if is_nested:
                    fixed_expr = prefix_expr.replace("{i}", "{_i}").replace("{e}", "{_e}")
                    add(lines, indent, f"for _i, _mod in enumerate(self.{attr_name}):")
                    add(lines, indent + 1, f"for _e, _sub in enumerate(_mod):")
                    add(lines, indent + 2, f"_sub.prefix = {fixed_expr}")
                elif is_repeated:
                    fixed_expr = prefix_expr.replace("{i}", "{_i}")
                    add(lines, indent, f"for _i, _mod in enumerate(self.{attr_name}):")
                    add(lines, indent + 1, f"_mod.prefix = {fixed_expr}")
                else:
                    add(lines, indent, f"self.{attr_name}.prefix = {prefix_expr}")

        if classification.mamba_mixer_module_names and not _is_hybrid_mamba:
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
                dim = self._config_expr("embedding_size", alt="hidden_size")
            add(lines, indent, "VocabParallelEmbedding(")
            add(lines, indent + 1, f"{vocab}, {dim},")
            add(lines, indent + 1, f"prefix={prefix},")
            add(lines, indent + 1, "params_dtype=params_dtype,")
            add(lines, indent, ")")
        elif layer_type == VLLMLayerType.PARALLEL_LM_HEAD:
            vocab = self._config_expr("vocab_size")
            dim = self._config_expr("embedding_size", alt="hidden_size")
            lin_expr = _select_linear_expr(node)
            if lin_expr is not None and len(lin_expr.inputs) > 3:
                bias = _literal_value(lin_expr.inputs[3], False)
                if not isinstance(bias, bool):
                    bias = False
            else:
                bias = self._bias_expr(node)
            add(lines, indent, "ParallelLMHead(")
            add(lines, indent + 1, f"{vocab}, {dim},")
            add(lines, indent + 1, f"bias={bias},")
            add(lines, indent + 1, f"prefix={prefix},")
            add(lines, indent + 1, "params_dtype=params_dtype,")
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
            bias = self._bias_expr(node)
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
                if up_node_id and up_node_id != node.id:
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
            head_size = self._head_dim_expr()
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
                _hd = self._head_dim_expr()
                scale_cfg = (
                    f"getattr(config, 'query_pre_attn_scalar', "
                    f"self._model_config.get('query_pre_attn_scalar', {_hd}))"
                )
                scale_expr = f"getattr(config, 'attention_multiplier', None) or (1.0 / (float({scale_cfg}) ** 0.5))"
            add(lines, indent, "Attention(")
            add(lines, indent + 1, "_num_heads,")
            add(lines, indent + 1, f"{head_size},")
            add(lines, indent + 1, f"scale={scale_expr},")
            add(lines, indent + 1, f"num_kv_heads={kv_heads_arg},")
            add(lines, indent + 1, "cache_config=cache_config,")
            add(lines, indent + 1, "quant_config=quant_config,")
            add(lines, indent + 1, "logits_soft_cap=getattr(config, 'attn_logit_softcapping', None),")
            if self._detect_alibi():
                add(lines, indent + 1, "alibi_slopes=_get_alibi_slopes(getattr(config, 'num_attention_heads', 1)).tolist(),")
            is_rep = self._is_repeated_node(node)
            if is_rep:
                add(lines, indent + 1, "per_layer_sliding_window=(")
                add(lines, indent + 2, "getattr(config, 'sliding_window', None)")
                add(lines, indent + 2, "if ((hasattr(config, 'layer_types') and i < len(config.layer_types) and config.layer_types[i] != 'full_attention')")
                add(lines, indent + 2, "    or (not hasattr(config, 'layer_types') and getattr(config, 'sliding_window_pattern', 0) and (i + 1) % getattr(config, 'sliding_window_pattern', 0) != 0)")
                add(lines, indent + 2, "    or (not hasattr(config, 'layer_types') and not getattr(config, 'sliding_window_pattern', 0) and getattr(config, 'sliding_window', None) is not None))")
                add(lines, indent + 2, "else None),")
            else:
                add(lines, indent + 1, "per_layer_sliding_window=getattr(config, 'sliding_window', None),")
            # KV sharing: layers in the last num_kv_shared_layers reuse KV cache
            # from earlier layers of the same attention type.
            # Build target prefix by replacing {i} with the target index.
            # Attention prefix must end with a .attn suffix to match native vLLM
            # convention used by KV cache binding and layer_name lookup.
            # For modules with multiple attention groups (encoder-decoder cross-attn),
            # use self_attn.attn for the first and encoder_attn.attn for the rest.
            _attn_suffix = "self_attn.attn"
            if is_rep:
                _mod_name = self._node_module_name(node)
                _attn_idx = 0
                _attn_idx_in_mod = 0
                for g in self._vllm_classification.qkv_groups:
                    if g.attention_node_id is None:
                        continue
                    g_mod = self._node_module_name(self._find_node_by_id(g.attention_node_id))
                    if g_mod == _mod_name:
                        if g.attention_node_id == node.id:
                            _attn_idx_in_mod = _attn_idx
                        _attn_idx += 1
                if _attn_idx > 1:
                    _attn_suffix = "encoder_attn.attn" if _attn_idx_in_mod > 0 else "self_attn.attn"
            if f".{_attn_suffix}" in prefix:
                attn_prefix = prefix
            elif "{i}" in prefix:
                attn_prefix = prefix.replace("{i}", f"{{i}}.{_attn_suffix}")
            elif is_rep:
                attn_prefix = prefix
            else:
                attn_prefix = f'f"model.layers.0.{_attn_suffix}"'
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
                if node.id in self._vllm_classification.qk_norm_pre_reshape_node_ids:
                    dim = self._config_expr("hidden_size")
                else:
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
                dim = self._resolve_const_value("PLI") or f"getattr(config, 'per_layer_input_dim', 256)"
            else:
                dim = self._config_expr("hidden_size")
            eps = self._node_rmsnorm_eps(node)
            has_weight = "rmsnorm_noscale" not in node.op.name
            if has_weight:
                add(lines, indent, "RMSNorm(")
                add(lines, indent + 1, f"{dim}, eps={eps},")
                add(lines, indent + 1, "dtype=params_dtype,")
                add(lines, indent, ")")
            else:
                add(lines, indent, "RMSNorm(")
                add(lines, indent + 1, f"{dim}, eps={eps}, has_weight=False,")
                add(lines, indent + 1, "dtype=params_dtype,")
                add(lines, indent, ")")
        elif layer_type == VLLMLayerType.LAYERNORM:
            dim = self._node_input_dim_expr(node)
            eps = self._node_layernorm_eps(node)
            add(lines, indent, "LayerNorm(")
            add(lines, indent + 1, f"int({dim}), eps={eps},")
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
            bias = self._bias_expr(node)
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
        bias = self._bias_expr(node)

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
        def _fix_prefix(s: str) -> str:
            return s.replace("{root}", "{prefix}").replace("{path}", "{prefix}").replace("{__scope}", "{prefix}")
        if not self._is_repeated_node(q_node):
            base = _linear_base_key(q_node)
            if base and "." in base:
                fixed = _fix_prefix(base.rsplit(".", 1)[0] + ".qkv_proj")
                if "{prefix}" in fixed or "{i}" in fixed:
                    return f'f"{fixed}"'
                return repr(fixed)
            return repr("qkv_proj")
        mod_name = self._node_module_name(q_node)
        scope_parts = self._vllm_classification.module_scope_parts.get(mod_name)
        if scope_parts is None:
            base = _linear_base_key(q_node)
            if base and "{i}" in base:
                return f'f"{_fix_prefix(base.rsplit(".", 1)[0] + ".qkv_proj")}"'
            return f'f"{{prefix}}.layers.{{i}}.qkv_proj"'
        base = _linear_base_key(q_node)
        if base:
            sub_parts = base.split(".")
            if sub_parts and sub_parts[0] in ("{__scope}", "{path}", "{root}"):
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
        return f'f"{_fix_prefix(".".join(fparts))}"'

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
        add(lines, indent, "# Remap HF LayerNorm parameter names (.gamma/.beta) to PyTorch names (.weight/.bias)")
        add(lines, indent, "if name.endswith('.gamma'):")
        add(lines, indent + 1, "name = name[:-6] + '.weight'")
        add(lines, indent, "elif name.endswith('.beta'):")
        add(lines, indent + 1, "name = name[:-5] + '.bias'")
        add(lines, indent, "if _orig_name in _transposed_ck_weights and loaded_weight.dim() == 2:")
        add(lines, indent + 1, "loaded_weight = loaded_weight.t().contiguous()")
        add(lines, indent, "for _pname, _wname, _sid in stacked_params_mapping:")
        add(lines, indent + 1, "if _wname not in name:")
        add(lines, indent + 2, "continue")
        add(lines, indent + 1, "name = name.replace(_wname, _pname)")
        add(lines, indent + 1, "name = _ckpt_to_model.get(name, name)")
        add(lines, indent + 1, "if name not in params_dict and name.startswith('model.'):")
        add(lines, indent + 2, "name = _ckpt_to_model.get(name[6:], name[6:])")
        add(lines, indent + 1, "if name not in params_dict:")
        add(lines, indent + 2, "_w = loaded_weight.to(dtype=self._params_dtype) if loaded_weight.is_floating_point() else loaded_weight")
        add(lines, indent + 2, "self.state_dict_tensors[_orig_name] = _w")
        add(lines, indent + 2, "if _orig_name.startswith('model.') and _orig_name[6:] not in self.state_dict_tensors:")
        add(lines, indent + 3, "self.state_dict_tensors[_orig_name[6:]] = _w")
        add(lines, indent + 2, "break")
        add(lines, indent + 1, "param = params_dict[name]")
        add(lines, indent + 1, "weight_loader = getattr(param, 'weight_loader', None)")
        add(lines, indent + 1, "if weight_loader is not None:")
        add(lines, indent + 2, "weight_loader(param, loaded_weight, _sid)")
        add(lines, indent + 1, "else:")
        add(lines, indent + 2, "param.data.copy_(loaded_weight)")
        add(lines, indent + 1, "loaded_params.add(name)")
        add(lines, indent + 1, "self.state_dict_tensors[_orig_name] = loaded_weight.to(dtype=self._params_dtype) if loaded_weight.is_floating_point() else loaded_weight")
        add(lines, indent + 1, "break")
        add(lines, indent, "else:")
        add(lines, indent + 1, "name = _ckpt_to_model.get(name, name)")
        add(lines, indent + 1, "if name not in params_dict and name.startswith('model.'):")
        add(lines, indent + 2, "name = _ckpt_to_model.get(name[6:], name[6:])")
        add(lines, indent + 1, "if name not in params_dict:")
        add(lines, indent + 2, "_w = loaded_weight.to(dtype=self._params_dtype) if loaded_weight.is_floating_point() else loaded_weight")
        add(lines, indent + 2, "self.state_dict_tensors[_orig_name] = _w")
        add(lines, indent + 2, "if _orig_name.startswith('model.') and _orig_name[6:] not in self.state_dict_tensors:")
        add(lines, indent + 3, "self.state_dict_tensors[_orig_name[6:]] = _w")
        add(lines, indent + 2, "continue")
        add(lines, indent + 1, "param = params_dict[name]")
        add(lines, indent + 1, "weight_loader = getattr(param, 'weight_loader', None)")
        add(lines, indent + 1, "if weight_loader is not None:")
        add(lines, indent + 2, "weight_loader(param, loaded_weight)")
        add(lines, indent + 1, "else:")
        add(lines, indent + 2, "param.data.copy_(loaded_weight)")
        add(lines, indent + 1, "loaded_params.add(name)")
        add(lines, indent + 1, "self.state_dict_tensors[_orig_name] = loaded_weight.to(dtype=self._params_dtype) if loaded_weight.is_floating_point() else loaded_weight")

    def _vllm_layer_attr_name(self, node: GraphNode) -> str:
        node_id = node.id
        canonical_id = self._node_id_aliases.get(node_id, node_id)
        return f"_vllm_{_safe_ident(canonical_id)}"

    def _build_node_id_aliases(self) -> dict[str, str]:
        """Build a mapping from rewritten inline-body node IDs to the
        original classified node IDs.

        ``_emit_repeat_inline_body`` rewrites node IDs to
        ``f"{repeat_node.id}:inline:{inner_index}"`` but the
        classification and ``__init__`` use the original callee-module
        node IDs.  This map lets ``_vllm_layer_attr_name`` resolve the
        original ID so both sides agree on the attribute name.

        Also handles nested inlining: when a module call inside the
        repeat callee is itself inlined, its nodes get rewritten to
        ``__call_inline_{parent_rewritten_id}:inline:{inner_index}``.
        """
        aliases: dict[str, str] = {}
        for module in self.program.modules:
            for node in module.nodes:
                if node.op.name != "core.repeat":
                    continue
                try:
                    callee = self._repeat_attr_string(node, "callee")
                except (ValueError, KeyError):
                    continue
                if callee not in self.modules_by_name:
                    continue
                callee_mod = self.modules_by_name[callee]
                for inner_index, inner_node in enumerate(callee_mod.nodes, start=1):
                    rewritten_id = f"{node.id}:inline:{inner_index}"
                    aliases[rewritten_id] = inner_node.id
                    # Handle nested module call inlining: if this inner
                    # node is a call to another module, its body nodes
                    # will be inlined with prefix
                    # __call_inline_{parent_rewritten_id.replace(':','_')}.
                    called_mod = self.modules_by_name.get(inner_node.op.name)
                    if called_mod is not None:
                        nested_prefix = _py_ident(f"__call_inline_{rewritten_id.replace(':', '_')}")
                        for nested_index, nested_node in enumerate(called_mod.nodes, start=1):
                            nested_rewritten = f"{nested_prefix}:inline:{nested_index}"
                            aliases[nested_rewritten] = nested_node.id
                    # Handle core.select nodes inside the repeat body
                    # that call modules (e.g. hybrid attention/mamba models).
                    if inner_node.op.name == "core.select":
                        self._add_select_aliases(rewritten_id, inner_node, aliases)
        # Handle top-level core.select nodes that call modules
        for module in self.program.modules:
            for node in module.nodes:
                if node.op.name != "core.select":
                    continue
                self._add_select_aliases(node.id, node, aliases)
        return aliases

    def _add_select_aliases(
        self,
        select_rewritten_id: str,
        select_node: Any,
        aliases: dict[str, str],
    ) -> None:
        """Create aliases for branch-module nodes inlined from a
        ``core.select`` node, recursively handling nested selects.

        ``select_rewritten_id`` is the rewritten ID of the select node
        (e.g. ``"nemotron_h:13:inline:11"`` for a select inside a
        repeat body, or ``"nemotron_h_non_attention:2"`` for a
        top-level select).  The branch modules' nodes are inlined with
        prefix ``_py_ident("__select_inline_{id}:{branch}")`` and their
        rewritten IDs are ``"{prefix}:inline:{index}"``.
        """
        if len(select_node.inputs) < 3:
            return
        branch_inputs = {
            "_then": select_node.inputs[1],
            "_else": select_node.inputs[2],
        }
        for branch_suffix in ("_then", "_else"):
            inp = branch_inputs[branch_suffix]
            if not (hasattr(inp, "op") and hasattr(inp.op, "name")):
                continue
            branch_mod = self.modules_by_name.get(inp.op.name)
            if branch_mod is None:
                continue
            sel_prefix = _py_ident(
                f"__select_inline_{select_rewritten_id.replace(':', '_')}{branch_suffix}"
            )
            for bi, bn in enumerate(branch_mod.nodes, start=1):
                nested_rewritten = f"{sel_prefix}:inline:{bi}"
                aliases[nested_rewritten] = bn.id
                if bn.op.name == "core.select":
                    self._add_select_aliases(nested_rewritten, bn, aliases)

    def _build_inline_loop_prefix(self) -> dict[str, str]:
        """Build a mapping from inline-body node ID prefixes to the
        loop variable prefix used in the codegen.

        ``_emit_repeat_inline_body`` defines the loop variable as
        ``_py_ident(f"__loop_inline_{repeat_node.id.replace(':', '_')}")``.
        For nested inline nodes (e.g. module calls inlined within the
        repeat body), the node ID prefix is ``_call_inline_...`` which
        does not contain the repeat node ID, so ``_node_loop_index``
        cannot derive the loop variable from it.  This map provides
        the correct loop prefix for each inline node ID prefix.
        """
        prefix_map: dict[str, str] = {}
        for module in self.program.modules:
            for node in module.nodes:
                if node.op.name != "core.repeat":
                    continue
                try:
                    callee = self._repeat_attr_string(node, "callee")
                except (ValueError, KeyError):
                    continue
                if callee not in self.modules_by_name:
                    continue
                loop_prefix = _py_ident(f"__loop_inline_{node.id.replace(':', '_')}")
                callee_mod = self.modules_by_name[callee]
                for inner_index, inner_node in enumerate(callee_mod.nodes, start=1):
                    rewritten_id = f"{node.id}:inline:{inner_index}"
                    prefix_map[rewritten_id] = loop_prefix
                    called_mod = self.modules_by_name.get(inner_node.op.name)
                    if called_mod is not None:
                        nested_prefix = _py_ident(f"__call_inline_{rewritten_id.replace(':', '_')}")
                        prefix_map[nested_prefix] = loop_prefix
                    if inner_node.op.name == "core.select":
                        self._add_select_loop_prefix(rewritten_id, inner_node, loop_prefix, prefix_map)
        return prefix_map

    def _add_select_loop_prefix(
        self,
        select_rewritten_id: str,
        select_node: Any,
        loop_prefix: str,
        prefix_map: dict[str, str],
    ) -> None:
        """Add select-branch inline prefixes to the loop-prefix map so
        that ``_node_loop_index`` can resolve the correct repeat-body
        loop variable for nodes inlined from select branches."""
        if len(select_node.inputs) < 3:
            return
        branch_inputs = {
            "_then": select_node.inputs[1],
            "_else": select_node.inputs[2],
        }
        for branch_suffix in ("_then", "_else"):
            inp = branch_inputs[branch_suffix]
            if not (hasattr(inp, "op") and hasattr(inp.op, "name")):
                continue
            branch_mod = self.modules_by_name.get(inp.op.name)
            if branch_mod is None:
                continue
            sel_prefix = _py_ident(
                f"__select_inline_{select_rewritten_id.replace(':', '_')}{branch_suffix}"
            )
            prefix_map[sel_prefix] = loop_prefix
            for bn in branch_mod.nodes:
                if bn.op.name == "core.select":
                    bi = branch_mod.nodes.index(bn) + 1
                    nested_rewritten = f"{sel_prefix}:inline:{bi}"
                    self._add_select_loop_prefix(nested_rewritten, bn, loop_prefix, prefix_map)

    def _resolve_prefix_from_called_module(self, node: GraphNode) -> str:
        """Search called module's internal nodes for a GraphPath prefix.

        The optimizer may move GraphPath inputs from a call node into the
        called module's internal nodes (e.g. NN.embedding, NN.rmsnorm_noscale).
        This searches the called module's nodes for a non-weight/bias GraphPath.

        For core.select nodes (e.g. tied/untied LM head), search all
        GraphExpr branch modules for a GraphPath prefix.
        """
        if node.op.name == "core.select":
            fallback = ""
            for inp in node.inputs:
                if not isinstance(inp, GraphExpr):
                    continue
                mod = self.modules_by_name.get(inp.op.name)
                if mod is None:
                    continue
                for inner_node in mod.nodes:
                    for sub in inner_node.inputs:
                        if isinstance(sub, GraphPath) and sub.parts:
                            key = _graph_path_key(sub)
                            if not key.endswith(".weight") and not key.endswith(".bias"):
                                return key
                            if not fallback:
                                fallback = key
            return fallback
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
        if node.op.name == "core.select":
            base = self._resolve_prefix_from_called_module(node)
            if base:
                return base
        base = _linear_base_key(node)
        if not base:
            base = self._resolve_prefix_from_called_module(node)
        if not base:
            return node.id
        return base

    def _branch_benefits_from_control_inline(self, operand: GraphOperand, *, module_name: str) -> bool:
        if (
            isinstance(operand, GraphExpr)
            and operand.op.name.startswith("Positions.position_ids")
        ):
            return False
        return super()._branch_benefits_from_control_inline(operand, module_name=module_name)

    def _layer_prefix(self, node: GraphNode) -> str:
        def _fix_prefix(s: str) -> str:
            return s.replace("{root}", "{prefix}").replace("{path}", "{prefix}").replace("{__scope}", "{prefix}")
        if not self._is_repeated_node(node):
            base = _fix_prefix(self._node_prefix(node))
            if "{i}" in base or "{prefix}" in base:
                return f'f"{base}"'
            return repr(base)
        # For nested loop nodes (e.g. expert loops inside layer loops),
        # combine the layer-level scope with the nested-level path.
        if self._is_nested_loop_node(node):
            layer_scope = self._get_layer_scope_parts()
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
                # Strip the {path}/{__scope} prefix — it represents the parent
                # (layer-level) scope, which we replace with layer_scope.
                if sub_parts and sub_parts[0] in ("{path}", "{__scope}", "{root}"):
                    sub_parts = sub_parts[1:]
            else:
                sub_parts = []
            if layer_scope:
                full_parts = list(layer_scope) + sub_parts
            else:
                full_parts = sub_parts
            fparts: list[str] = []
            for p in full_parts:
                if p == "{__scope}":
                    fparts.append("{prefix}")
                else:
                    fparts.append(p)
            return f'f"{_fix_prefix(".".join(fparts))}"'
        mod_name = self._node_module_name(node)
        scope_parts = self._vllm_classification.module_scope_parts.get(mod_name)
        if scope_parts is None:
            for parts in self._vllm_classification.module_scope_parts.values():
                if "{i}" in parts:
                    scope_parts = parts
                    break
        if scope_parts is None:
            base = _fix_prefix(self._node_prefix(node))
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
            if sub_parts and sub_parts[0] in ("{__scope}", "{path}", "{root}"):
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
        return f'f"{_fix_prefix(".".join(fparts))}"'

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

    def _head_dim_expr(self) -> str:
        """Generate a config expression that derives head_dim from
        hidden_size // num_attention_heads when head_dim is not present."""
        hs = self._config_expr("hidden_size")
        nah = self._config_expr("num_attention_heads")
        return (
            f"getattr(config, 'head_dim', "
            f"self._model_config.get('head_dim', "
            f"({hs} // {nah}) if {nah} else {hs}))"
        )

    def _detect_partial_rotary(self) -> bool:
        """Detect whether the model uses partial rotary (rotary_dim < head_dim)."""
        for module in self.program.modules:
            for node in module.nodes:
                if "rope_base_factors" not in node.op.name:
                    continue
                if "cond_false" in node.op.name:
                    continue
                if len(node.inputs) < 2:
                    continue
                rd_inp = node.inputs[1]
                rd_name = getattr(rd_inp, "name", None)
                if rd_name is None:
                    continue
                if rd_name in ("MODEL_HEAD_DIM", "HEAD_DIM", "head_dim"):
                    continue
                # rotary dim is not head_dim → partial rotary
                return True
        # Fallback: check config for partial_rotary_factor or rotary_dim
        partial_rotary = self._model_config_data.get("partial_rotary_factor", 1.0)
        if isinstance(partial_rotary, (int, float)) and partial_rotary < 1.0:
            return True
        rotary_dim = self._model_config_data.get("rotary_dim", None)
        if rotary_dim is not None:
            head_dim = self._model_config_data.get("head_dim", None)
            if head_dim is None:
                hs = self._model_config_data.get("hidden_size", 0)
                nah = self._model_config_data.get("num_attention_heads", 0)
                head_dim = hs // nah if nah else None
            if head_dim and rotary_dim < head_dim:
                return True
        rotary_pct = self._model_config_data.get("rotary_pct", None)
        if rotary_pct is not None and rotary_pct < 1.0:
            return True
        return False

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
            hs = self._config_expr("hidden_size")
            return f"getattr(config, 'intermediate_size', self._model_config.get('intermediate_size', 4 * {hs}))"
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

    def _unwrap_expr_value(self, expr: Any) -> int | float | None:
        """Recursively unwrap GraphExpr (core.ascribe, core.binary) to find a literal."""
        if isinstance(expr, GraphLiteral) and isinstance(expr.value, (int, float)) and not isinstance(expr.value, bool):
            return expr.value
        if isinstance(expr, GraphExpr):
            if expr.op.name == "core.ascribe" and len(expr.inputs) >= 1:
                return self._unwrap_expr_value(expr.inputs[0])
            if expr.op.name in ("core.binary.*", "core.binary.+", "core.binary./") and len(expr.inputs) >= 2:
                left = self._unwrap_expr_value(expr.inputs[0])
                right = self._unwrap_expr_value(expr.inputs[1])
                if left is not None and right is not None:
                    if expr.op.name == "core.binary.*":
                        return int(left * right)
                    elif expr.op.name == "core.binary.+":
                        return int(left + right)
                    elif expr.op.name == "core.binary./" and right != 0:
                        return int(left / right)
        if isinstance(expr, GraphValueRef) and expr.name:
            return self._resolve_const_literal(expr.name)
        return None

    def _resolve_const_literal(self, name: str) -> int | float | None:
        """Resolve a constant name to its literal value from the graph IR.

        Handles simple constants (GraphLiteral outputs) and derived constants
        (core.binary.* nodes with resolvable operands).
        """
        for module in self.program.modules:
            if module.name != name:
                continue
            if module.outputs:
                out = module.outputs[0]
                val = self._unwrap_expr_value(out)
                if val is not None:
                    return val
            for node in module.nodes:
                if node.op.name in ("core.binary.*", "core.binary.+", "core.binary./") and len(node.inputs) >= 2:
                    left = self._resolve_const_operand(node.inputs[0])
                    right = self._resolve_const_operand(node.inputs[1])
                    if left is not None and right is not None:
                        if node.op.name == "core.binary.*":
                            return int(left * right)
                        elif node.op.name == "core.binary.+":
                            return int(left + right)
                        elif node.op.name == "core.binary./" and right != 0:
                            return int(left / right)
                elif node.op.name == "Math.floor" and len(node.inputs) >= 1:
                    inner = self._resolve_const_operand(node.inputs[0])
                    if inner is not None:
                        return int(inner)
                elif node.op.name == "core.select" and len(node.inputs) >= 3:
                    val = self._unwrap_expr_value(node.inputs[1])
                    if val is not None:
                        return val
            return None
        return None

    def _resolve_const_operand(self, operand: GraphOperand) -> int | float | None:
        """Resolve a graph operand to a literal numeric value."""
        if isinstance(operand, GraphLiteral) and isinstance(operand.value, (int, float)) and not isinstance(operand.value, bool):
            return operand.value
        if isinstance(operand, GraphExpr):
            return self._unwrap_expr_value(operand)
        if isinstance(operand, GraphValueRef) and operand.name:
            return self._resolve_const_literal(operand.name)
        return None

    def _node_output_dim_expr(self, node: GraphNode) -> str:
        dim = _int_arg(node, 2)
        if dim is not None:
            return str(dim)
        # Try tracing with the node's own module first, then the primary repeated module
        for mod in self._get_node_modules(node):
            if mod is not None and len(node.inputs) >= 3:
                loop_var = self._node_loop_index(node) if self._is_repeated_node(node) else "i"
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
        # Fallback: resolve the dim parameter as a derived constant
        if len(node.inputs) >= 3 and isinstance(node.inputs[2], GraphValueRef) and node.inputs[2].name:
            literal = self._resolve_const_literal(node.inputs[2].name)
            if literal is not None:
                return str(int(literal))
        if node.id in self._ffn_up_node_ids:
            hs = self._config_expr("hidden_size")
            return f"getattr(config, 'intermediate_size', self._model_config.get('intermediate_size', 4 * {hs}))"
        return self._config_expr("hidden_size")

    def _node_rmsnorm_eps(self, node: GraphNode) -> str:
        for i in range(1, min(len(node.inputs), 4)):
            eps = _literal_value(node.inputs[i], None)
            if isinstance(eps, (int, float)) and not isinstance(eps, bool):
                return repr(float(eps))
            if isinstance(node.inputs[i], GraphValueRef) and node.inputs[i].name:
                resolved = self._resolve_value_ref(node.inputs[i].name)
                if isinstance(resolved, (int, float)) and not isinstance(resolved, bool):
                    return repr(float(resolved))
        eps = self._find_rmsnorm_eps_in_module(node, set())
        if eps is not None:
            return repr(float(eps))
        return "1e-6"

    def _find_rmsnorm_eps_in_module(
        self, node: GraphNode, visited: set[str]
    ) -> float | None:
        mod_name = getattr(node.op, "name", None)
        if mod_name is None or mod_name in visited:
            return None
        visited.add(mod_name)
        mod = self.modules_by_name.get(mod_name)
        if mod is None:
            return None
        for n in mod.nodes:
            if n.op.name == "_rmsnorm" and len(n.inputs) >= 2:
                eps = _literal_value(n.inputs[1], None)
                if isinstance(eps, (int, float)) and not isinstance(eps, bool):
                    return float(eps)
                if isinstance(n.inputs[1], GraphValueRef) and n.inputs[1].name:
                    resolved = self._resolve_value_ref(n.inputs[1].name)
                    if isinstance(resolved, (int, float)) and not isinstance(resolved, bool):
                        return float(resolved)
            else:
                called_mod = self.modules_by_name.get(n.op.name)
                if called_mod is not None:
                    eps_idx = None
                    for idx, inp in enumerate(called_mod.inputs):
                        if inp.name == "eps":
                            eps_idx = idx
                            break
                    if eps_idx is not None and eps_idx < len(n.inputs):
                        eps = _literal_value(n.inputs[eps_idx], None)
                        if isinstance(eps, (int, float)) and not isinstance(eps, bool):
                            return float(eps)
                        if isinstance(n.inputs[eps_idx], GraphValueRef) and n.inputs[eps_idx].name:
                            resolved = self._resolve_value_ref(n.inputs[eps_idx].name)
                            if isinstance(resolved, (int, float)) and not isinstance(resolved, bool):
                                return float(resolved)
            inner = self._find_rmsnorm_eps_in_module(n, visited)
            if inner is not None:
                return inner
        return None

    def _detect_rmsnorm_unit_offset(self, node: GraphNode) -> bool:
        """Check if a RMSNorm node uses unit_offset (1 + weight) scaling.

        The ``unit_offset=true`` flag in Axon's ``NN.rmsnorm`` builtin causes
        the module output to be ``y + y0`` (i.e. ``x*w + x`` = ``x*(1+w)``).
        This is used by Gemma-family models.  We detect it by checking if the
        RMSNorm module's output is a ``core.binary.+`` expression.
        """
        mod_name = getattr(node.op, "name", None)
        if mod_name is None:
            return False
        mod = self.modules_by_name.get(mod_name)
        if mod is None:
            return False
        for output in mod.outputs:
            if isinstance(output, GraphExpr) and output.op.name == "core.binary.+":
                return True
        return False

    def _resolve_value_ref(self, ref_name: str, module_name: str | None = None) -> Any:
        """Resolve a ValueRef name to a literal value by tracing through module calls.

        Searches for the node that produces ``ref_name`` in the given module (or
        all modules if ``module_name`` is None).  If the producing node's op is a
        module name whose output is a GraphLiteral, returns that literal value.

        Also checks global binding modules whose name matches ``ref_name`` directly.
        """
        for module in self.program.modules:
            if module_name is not None and module.name != module_name:
                continue
            if module.is_global_binding and module.name == ref_name:
                if module.outputs and isinstance(module.outputs[0], GraphLiteral):
                    return module.outputs[0].value
            for node in module.nodes:
                for out in node.outputs:
                    if hasattr(out, "name") and out.name == ref_name:
                        mod = self.modules_by_name.get(node.op.name)
                        if mod is not None and mod.outputs:
                            first_out = mod.outputs[0]
                            if isinstance(first_out, GraphLiteral):
                                return first_out.value
                        # Continue searching other modules instead of returning None
        return None

    def _find_module_for_node(self, node: GraphNode) -> str | None:
        """Find the name of the module that contains the given node."""
        for module in self.program.modules:
            for n in module.nodes:
                if n is node:
                    return module.name
        return None

    def _node_layernorm_eps(self, node: GraphNode) -> str:
        module_name = self._find_module_for_node(node)
        for i in range(2, min(len(node.inputs), 4)):
            eps = _literal_value(node.inputs[i], None)
            if isinstance(eps, (int, float)) and not isinstance(eps, bool):
                return repr(float(eps))
            if isinstance(node.inputs[i], GraphValueRef) and node.inputs[i].name:
                resolved = self._resolve_value_ref(node.inputs[i].name, module_name=module_name)
                if isinstance(resolved, (int, float)) and not isinstance(resolved, bool):
                    return repr(float(resolved))
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
            bias = self._bias_expr(node)
            if isinstance(bias, str):
                return f"{attr}({args[1]})[0]"
            if bias:
                return f"{attr}({args[1]})[0] + {attr}.bias"
            return f"{attr}({args[1]})[0]"

        if layer_type in (
            VLLMLayerType.COLUMN_PARALLEL_LINEAR,
            VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR,
        ):
            attr = self._vllm_attr_access(node)
            args = self._collect_args(node, local)
            bias = self._bias_expr(node)
            if isinstance(bias, str):
                return f"{attr}({args[1]})[0]"
            if bias:
                return f"{attr}({args[1]})[0] + {attr}.bias"
            return f"{attr}({args[1]})[0]"

        if layer_type == VLLMLayerType.ROW_PARALLEL_LINEAR:
            attr = self._vllm_attr_access(node)
            args = self._collect_args(node, local)
            bias = self._bias_expr(node)
            if isinstance(bias, str):
                return f"{attr}({args[1]})[0]"
            if bias:
                return f"{attr}({args[1]})[0] + {attr}.bias"
            return f"{attr}({args[1]})[0]"

        if layer_type == VLLMLayerType.PARALLEL_LM_HEAD:
            attr = self._vllm_attr_access(node)
            args = self._collect_args(node, local)
            return f"{attr}({args[1]})"

        if layer_type == VLLMLayerType.ATTENTION:
            attr = self._vllm_attr_access(node)
            args = self._collect_args(node, local)
            if len(args) >= 3:
                return f"{attr}({args[0]}.contiguous(), {args[1]}.contiguous(), {args[2]}.contiguous())"
            return f"{attr}()"

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
                q_var = args[0]
                k_var = args[1]
                v_var = args[2]
                # vLLM Attention expects 3D (B*S, H, DH) and returns 2D
                # (B*S, H*DH).  Graph has 4D Q/K/V (B, H, S, DH).  Permute
                # to (B, S, H, DH) and flatten to 3D for the attention call,
                # then reshape output back to (B, H, S, DH) for the
                # subsequent merge_heads operation.
                return (
                    f"({attr}("
                    f"{q_var}.permute(0, 2, 1, 3).reshape(-1, {q_var}.shape[1], {q_var}.shape[3]).contiguous(), "
                    f"{k_var}.permute(0, 2, 1, 3).reshape(-1, {k_var}.shape[1], {k_var}.shape[3]).contiguous(), "
                    f"{v_var}.permute(0, 2, 1, 3).reshape(-1, {v_var}.shape[1], {v_var}.shape[3]).contiguous()"
                    f"))"
                    f".reshape({q_var}.shape[0], {q_var}.shape[2], {q_var}.shape[1], {q_var}.shape[3])"
                    f".permute(0, 2, 1, 3)"
                )
            return f"{attr}()"

        return super()._primitive_expr(primitive, node, local=local, symbols_dict=symbols_dict)

    def _collect_args(self, node: GraphNode, local: set[str] | dict[str, str]) -> list[str]:
        args: list[str] = []
        for inp in node.inputs:
            if isinstance(inp, (GraphValueRef, GraphValue)):
                name = inp.name
                if isinstance(local, dict):
                    mapped = local.get(name, name)
                    args.append(_py_ident(mapped) if mapped == name else mapped)
                else:
                    args.append(_py_ident(name))
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
            elif isinstance(inp, GraphExpr):
                args.append(self._eval_graph_expr(inp, local))
            else:
                args.append(repr(inp))
        return args

    def _eval_graph_expr(self, expr: GraphExpr, local: set[str] | dict[str, str]) -> str:
        """Evaluate a GraphExpr to a Python expression string."""
        op_name = expr.op.name
        if op_name.startswith("core.binary."):
            op_map = {"+": "+", "-": "-", "*": "*", "/": "/"}
            sym = op_map.get(op_name.rsplit(".", 1)[-1])
            if sym is None:
                return repr(expr)
            parts = []
            for inp in expr.inputs:
                parts.append(self._eval_graph_operand(inp, local))
            return f"({sym.join(parts)})"
        return repr(expr)

    def _eval_graph_operand(self, operand: Any, local: set[str] | dict[str, str]) -> str:
        if isinstance(operand, (GraphValueRef, GraphValue)):
            name = operand.name
            if isinstance(local, dict):
                mapped = local.get(name, name)
                return _py_ident(mapped) if mapped == name else mapped
            return _py_ident(name)
        elif isinstance(operand, GraphLiteral):
            return repr(operand.value)
        elif isinstance(operand, GraphPath):
            return repr(_graph_path_key(operand))
        elif isinstance(operand, GraphExpr):
            return self._eval_graph_expr(operand, local)
        return repr(operand)

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
    if True:
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
    if emitter._detect_alibi():
        header.extend(
            [
                "",
                "import math as _math",
                "def _get_alibi_slopes(total_num_heads):",
                "    closest_power_of_2 = 2 ** _math.floor(_math.log2(total_num_heads))",
                "    base = torch.tensor(2 ** (-(2 ** -(_math.log2(closest_power_of_2) - 3))), dtype=torch.float32)",
                "    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)",
                "    slopes = torch.pow(base, powers)",
                "    if closest_power_of_2 != total_num_heads:",
                "        extra_base = torch.tensor(2 ** (-(2 ** -(_math.log2(2 * closest_power_of_2) - 3))), dtype=torch.float32)",
                "        num_remaining_heads = min(closest_power_of_2, total_num_heads - closest_power_of_2)",
                "        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=torch.int32)",
                "        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)",
                "    return slopes",
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
