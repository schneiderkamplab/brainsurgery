from __future__ import annotations

from typing import Any

from ..codegen2_torch.core import _DirectTorchEmitter
from ..graph_ir.core import (
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


def _linear_weight_key(node: GraphNode) -> str:
    if len(node.inputs) < 1:
        return ""
    base = node.inputs[0]
    if isinstance(base, GraphPath):
        base_key = _graph_path_key(base)
    else:
        base_key = str(base)
    weight_leaf = "weight"
    if len(node.inputs) >= 7:
        leaf = _literal_value(node.inputs[6], "weight")
        if isinstance(leaf, str) and leaf:
            weight_leaf = leaf
    return f"{base_key}.{weight_leaf}"


def _linear_bias_key(node: GraphNode) -> str | None:
    if not _bool_arg(node, 3):
        return None
    if len(node.inputs) < 1:
        return None
    base = node.inputs[0]
    if isinstance(base, GraphPath):
        base_key = _graph_path_key(base)
    else:
        base_key = str(base)
    bias_leaf = "bias"
    if len(node.inputs) >= 8:
        leaf = _literal_value(node.inputs[7], "bias")
        if isinstance(leaf, str) and leaf:
            bias_leaf = leaf
    return f"{base_key}.{bias_leaf}"


def _linear_base_key(node: GraphNode) -> str:
    if len(node.inputs) < 1:
        return ""
    base = node.inputs[0]
    if isinstance(base, GraphPath):
        return _graph_path_key(base)
    return str(base)


def _embedding_weight_key(node: GraphNode) -> str:
    if len(node.inputs) < 1:
        return ""
    base = node.inputs[0]
    if isinstance(base, GraphPath):
        return f"{_graph_path_key(base)}.weight"
    return ""


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
        attr = self._vllm_layer_attr_name(node)
        args = self._collect_args(node, {k: k for k in local} if isinstance(local, set) else local)

        if layer_type == VLLMLayerType.VOCAB_PARALLEL_EMBEDDING:
            if len(args) < 2:
                return False
            expr = f"self.{attr}({args[1]})"
        elif layer_type == VLLMLayerType.QKV_PARALLEL_LINEAR:
            if len(args) < 2:
                return False
            bias = _bool_arg(node, 3)
            if bias:
                expr = f"self.{attr}({args[1]})[0]"
            else:
                expr = f"self.{attr}({args[1]})"
        elif layer_type in (VLLMLayerType.COLUMN_PARALLEL_LINEAR, VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR):
            if len(args) < 2:
                return False
            expr = f"self.{attr}({args[1]})[0]"
        elif layer_type == VLLMLayerType.ROW_PARALLEL_LINEAR:
            if len(args) < 2:
                return False
            expr = f"self.{attr}({args[1]})[0]"
        elif layer_type == VLLMLayerType.PARALLEL_LM_HEAD:
            if len(args) < 2:
                return False
            expr = f"self.{attr}({args[1]})"
        elif layer_type == VLLMLayerType.ATTENTION:
            if len(args) >= 3:
                expr = f"self.{attr}({args[0]}, {args[1]}, {args[2]}, attn_metadata=self._attn_metadata)"
            else:
                expr = f"self.{attr}(attn_metadata=self._attn_metadata)"
        elif layer_type == VLLMLayerType.RMSNORM:
            if len(args) < 1:
                return False
            expr = f"self.{attr}({args[0]})"
        elif layer_type == VLLMLayerType.LAYERNORM:
            if len(args) < 1:
                return False
            expr = f"self.{attr}({args[0]})"
        else:
            return False

        joined = ", ".join(targets)
        if len(targets) == 1:
            add(lines, indent, f"{joined} = {expr}")
        else:
            add(lines, indent, f"{joined} = {expr}")
        return True

    def _emit_common(self, lines: list[str]) -> None:
        add = self._add
        cls = self.class_name
        classification = self._vllm_classification
        indent = 4

        add(lines, 0, f"class {cls}(nn.Module):")
        add(lines, indent, '"""Generated vLLM model from Axon Graph IR."""')
        add(lines, indent, "")
        add(lines, indent, "def __init__(self, vllm_config, prefix: str = ''):")
        add(lines, indent * 2, "super().__init__()")
        add(lines, indent * 2, "config = vllm_config.model_config.hf_config")
        add(lines, indent * 2, "self.vllm_config = vllm_config")
        add(lines, indent * 2, "self._prefix = prefix")
        add(lines, indent * 2, "quant_config = getattr(vllm_config, 'quant_config', None)")
        add(lines, indent * 2, "params_dtype = vllm_config.model_config.dtype")
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
        add(lines, indent * 2, "from vllm.model_executor.layers.norm import RMSNorm, LayerNorm")
        add(lines, indent * 2, "")

        self._emit_vllm_layer_inits(lines, classification, indent * 2)

        add(lines, indent * 2, "")
        add(lines, indent * 2, "# --- Config helpers ---")
        add(lines, indent * 2, "self._model_config = dict(_MODEL_CONFIG)")
        add(lines, indent * 2, "")

        add(lines, indent, "")
        add(lines, indent, "def embed_input_ids(self, input_ids):")
        for node_id in classification.embedding_node_ids:
            node = self._find_node_by_id(node_id)
            if node is not None:
                layer_name = self._vllm_layer_attr_name(node)
                add(lines, indent * 2, f"return self.{layer_name}(input_ids)")
                break
        add(lines, indent * 2, "return None")
        add(lines, indent, "")

        add(lines, indent, "def load_weights(self, weights):")
        add(lines, indent * 2, "# Map checkpoint keys to vLLM layer parameters")
        add(lines, indent * 2, "params_dict = dict(self.named_parameters())")
        add(lines, indent * 2, "params_dict.update(dict(self.named_buffers()))")
        add(lines, indent * 2, "loaded_params = set()")
        add(lines, indent * 2, "")
        add(lines, indent * 2, "for name, loaded_weight in weights:")
        self._emit_weight_loading_body(lines, classification, indent * 3)
        add(lines, indent, "")

        add(lines, indent, "def forward(")
        add(lines, indent * 2, "self,")
        add(lines, indent * 2, "input_ids: torch.Tensor | None = None,")
        add(lines, indent * 2, "positions: torch.Tensor | None = None,")
        add(lines, indent * 2, "intermediate_tensors=None,")
        add(lines, indent * 2, "inputs_embeds: torch.Tensor | None = None,")
        add(lines, indent * 2, "attn_metadata=None,")
        add(lines, indent * 2, "):")
        add(lines, indent * 2, "if inputs_embeds is not None:")
        add(lines, indent * 3, "hidden_states = inputs_embeds")
        add(lines, indent * 2, "else:")
        add(lines, indent * 3, "hidden_states = self.embed_input_ids(input_ids)")
        add(lines, indent * 2, "")
        add(lines, indent * 2, "# Store attn_metadata for use by attention layers")
        add(lines, indent * 2, "self._attn_metadata = attn_metadata")
        add(lines, indent * 2, "")

    def _emit_vllm_layer_inits(
        self,
        lines: list[str],
        classification: VLLMLayerClassification,
        indent: int,
    ) -> None:
        add = self._add
        seen = set()
        for module in self.program.modules:
            for node in module.nodes:
                if node.id in seen:
                    continue
                layer_type = classification.layer_type(node)
                if layer_type == VLLMLayerType.DEFAULT:
                    continue
                seen.add(node.id)
                attr_name = self._vllm_layer_attr_name(node)
                prefix = self._node_prefix(node)
                if layer_type == VLLMLayerType.VOCAB_PARALLEL_EMBEDDING:
                    vocab = self._config_expr("vocab_size", "V")
                    dim = self._config_expr("hidden_size", "D")
                    add(lines, indent, f"self.{attr_name} = VocabParallelEmbedding(")
                    add(lines, indent + 1, f"{vocab}, {dim},")
                    add(lines, indent + 1, f"prefix={prefix!r},")
                    add(lines, indent, ")")
                elif layer_type == VLLMLayerType.PARALLEL_LM_HEAD:
                    vocab = self._config_expr("vocab_size", "V")
                    dim = self._config_expr("hidden_size", "D")
                    add(lines, indent, f"self.{attr_name} = ParallelLMHead(")
                    add(lines, indent + 1, f"{vocab}, {dim},")
                    add(lines, indent + 1, f"prefix={prefix!r},")
                    add(lines, indent, ")")
                elif layer_type == VLLMLayerType.QKV_PARALLEL_LINEAR:
                    in_dim = self._node_input_dim_expr(node)
                    q_out = self._node_output_dim_expr(node)
                    bias = _bool_arg(node, 3)
                    add(lines, indent, f"self.{attr_name} = QKVParallelLinear(")
                    add(lines, indent + 1, f"{in_dim}, {q_out},")
                    add(lines, indent + 1, f"bias={bias},")
                    add(lines, indent + 1, f"prefix={prefix!r},")
                    add(lines, indent + 1, "quant_config=quant_config,")
                    add(lines, indent + 1, "params_dtype=params_dtype,")
                    add(lines, indent, ")")
                elif layer_type == VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR:
                    in_dim = self._node_input_dim_expr(node)
                    out_dim = self._node_output_dim_expr(node)
                    bias = _bool_arg(node, 3)
                    add(lines, indent, f"self.{attr_name} = MergedColumnParallelLinear(")
                    add(lines, indent + 1, f"{in_dim}, [{out_dim}, {out_dim}],")
                    add(lines, indent + 1, f"bias={bias},")
                    add(lines, indent + 1, f"prefix={prefix!r},")
                    add(lines, indent + 1, "quant_config=quant_config,")
                    add(lines, indent + 1, "params_dtype=params_dtype,")
                    add(lines, indent, ")")
                elif layer_type == VLLMLayerType.COLUMN_PARALLEL_LINEAR:
                    in_dim = self._node_input_dim_expr(node)
                    out_dim = self._node_output_dim_expr(node)
                    bias = _bool_arg(node, 3)
                    add(lines, indent, f"self.{attr_name} = ColumnParallelLinear(")
                    add(lines, indent + 1, f"{in_dim}, {out_dim},")
                    add(lines, indent + 1, f"bias={bias},")
                    add(lines, indent + 1, f"prefix={prefix!r},")
                    add(lines, indent + 1, "quant_config=quant_config,")
                    add(lines, indent + 1, "params_dtype=params_dtype,")
                    add(lines, indent, ")")
                elif layer_type == VLLMLayerType.ROW_PARALLEL_LINEAR:
                    in_dim = self._node_input_dim_expr(node)
                    out_dim = self._node_output_dim_expr(node)
                    bias = _bool_arg(node, 3)
                    add(lines, indent, f"self.{attr_name} = RowParallelLinear(")
                    add(lines, indent + 1, f"{in_dim}, {out_dim},")
                    add(lines, indent + 1, f"bias={bias},")
                    add(lines, indent + 1, f"prefix={prefix!r},")
                    add(lines, indent + 1, "quant_config=quant_config,")
                    add(lines, indent + 1, "params_dtype=params_dtype,")
                    add(lines, indent, ")")
                elif layer_type == VLLMLayerType.ATTENTION:
                    add(lines, indent, f"self.{attr_name} = Attention(")
                    add(lines, indent + 1, f"prefix={prefix!r},")
                    add(lines, indent, ")")
                elif layer_type == VLLMLayerType.RMSNORM:
                    dim = self._node_input_dim_expr(node)
                    eps = self._node_rmsnorm_eps(node)
                    add(lines, indent, f"self.{attr_name} = RMSNorm(")
                    add(lines, indent + 1, f"{dim}, eps={eps},")
                    add(lines, indent, ")")
                elif layer_type == VLLMLayerType.LAYERNORM:
                    dim = self._node_input_dim_expr(node)
                    eps = self._node_layernorm_eps(node)
                    add(lines, indent, f"self.{attr_name} = LayerNorm(")
                    add(lines, indent + 1, f"{dim}, eps={eps},")
                    add(lines, indent, ")")

    def _emit_weight_loading_body(
        self,
        lines: list[str],
        classification: VLLMLayerClassification,
        indent: int,
    ) -> None:
        add = self._add
        add(lines, indent, "if name not in params_dict:")
        add(lines, indent + 1, "continue")
        add(lines, indent, "param = params_dict[name]")
        add(lines, indent, "weight_loader = getattr(param, 'weight_loader', None)")
        add(lines, indent, "if weight_loader is not None:")
        add(lines, indent + 1, "weight_loader(param, loaded_weight)")
        add(lines, indent, "else:")
        add(lines, indent + 1, "param.data.copy_(loaded_weight)")
        add(lines, indent, "loaded_params.add(name)")

    def _vllm_layer_attr_name(self, node: GraphNode) -> str:
        base = _linear_base_key(node)
        if not base:
            base = node.id
        safe = base.replace(".", "_").replace("-", "_").replace("{", "").replace("}", "")
        return f"_vllm_{safe}"

    def _node_prefix(self, node: GraphNode) -> str:
        base = _linear_base_key(node)
        if not base:
            return node.id
        return base

    def _config_expr(self, config_name: str, fallback_dim: str) -> str:
        return f"getattr(config, {config_name!r}, self._model_config.get({config_name!r}, 0))"

    def _node_input_dim_expr(self, node: GraphNode) -> str:
        if len(node.inputs) >= 3:
            dim = _literal_value(node.inputs[2], None)
            if isinstance(dim, int):
                return str(dim)
        return "0"

    def _node_output_dim_expr(self, node: GraphNode) -> str:
        if len(node.outputs) >= 1 and hasattr(node.outputs[0], "type_expr"):
            te = node.outputs[0].typeExpr if hasattr(node.outputs[0], "typeExpr") else getattr(node.outputs[0], "type_expr", None)
            if te is not None:
                return "0"
        return "0"

    def _node_rmsnorm_eps(self, node: GraphNode) -> str:
        if len(node.inputs) >= 2:
            eps = _literal_value(node.inputs[1], None)
            if isinstance(eps, float):
                return repr(eps)
        return "1e-6"

    def _node_layernorm_eps(self, node: GraphNode) -> str:
        if len(node.inputs) >= 3:
            eps = _literal_value(node.inputs[2], None)
            if isinstance(eps, float):
                return repr(eps)
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
            attr = self._vllm_layer_attr_name(node)
            args = self._collect_args(node, local)
            return f"self.{attr}({args[1]})"

        if layer_type == VLLMLayerType.QKV_PARALLEL_LINEAR:
            attr = self._vllm_layer_attr_name(node)
            args = self._collect_args(node, local)
            bias = _bool_arg(node, 3)
            if bias:
                return f"self.{attr}({args[1]})[0]"
            return f"self.{attr}({args[1]})"

        if layer_type in (
            VLLMLayerType.COLUMN_PARALLEL_LINEAR,
            VLLMLayerType.MERGED_COLUMN_PARALLEL_LINEAR,
        ):
            attr = self._vllm_layer_attr_name(node)
            args = self._collect_args(node, local)
            return f"self.{attr}({args[1]})[0]"

        if layer_type == VLLMLayerType.ROW_PARALLEL_LINEAR:
            attr = self._vllm_layer_attr_name(node)
            args = self._collect_args(node, local)
            return f"self.{attr}({args[1]})[0]"

        if layer_type == VLLMLayerType.PARALLEL_LM_HEAD:
            attr = self._vllm_layer_attr_name(node)
            args = self._collect_args(node, local)
            return f"self.{attr}({args[1]})"

        if layer_type == VLLMLayerType.ATTENTION:
            attr = self._vllm_layer_attr_name(node)
            args = self._collect_args(node, local)
            if len(args) >= 3:
                return (
                    f"self.{attr}({args[0]}, {args[1]}, {args[2]}, "
                    f"attn_metadata=self._attn_metadata)"
                )
            return f"self.{attr}(attn_metadata=self._attn_metadata)"

        if layer_type == VLLMLayerType.RMSNORM:
            attr = self._vllm_layer_attr_name(node)
            args = self._collect_args(node, local)
            return f"self.{attr}({args[0]})"

        if layer_type == VLLMLayerType.LAYERNORM:
            attr = self._vllm_layer_attr_name(node)
            args = self._collect_args(node, local)
            return f"self.{attr}({args[0]})"

        if primitive == "_vllm_paged_attention":
            attr = self._vllm_layer_attr_name(node)
            args = self._collect_args(node, local)
            if len(args) >= 3:
                return (
                    f"self.{attr}({args[0]}, {args[1]}, {args[2]}, "
                    f"attn_metadata=self._attn_metadata)"
                )
            return f"self.{attr}(attn_metadata=self._attn_metadata)"

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


__all__ = [
    "emit_model_code_from_graph_ir",
]
