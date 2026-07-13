from __future__ import annotations

from pathlib import Path

import pytest

from brainsurgery.synapse.axon import (
    GraphOptimizeConfig,
    elaborate_closed_axon_file,
    flatten_closed_axon_file,
    lower_axon_program_to_graph_ir,
    normalize_closed_axon_file,
    optimize_graph_program,
    optimize_safe_flat_typed_axon_file,
    resolve_axon_program_from_path,
    typecheck2_flat_axon_file,
)
from brainsurgery.synapse.axon.codegen2_vllm.classify import classify_graph_for_vllm
from brainsurgery.synapse.axon.codegen2_vllm import emit_model_code_from_graph_ir
from brainsurgery.synapse.axon.ast import TypeDim, TypeInt, TypeTensor
from brainsurgery.synapse.axon.graph_ir import (
    GraphLiteral,
    GraphModule,
    GraphNode,
    GraphOp,
    GraphPath,
    GraphProgram,
    GraphValue,
    GraphValueRef,
)


def _vllm_optimized_graph(axon_path: str):
    program = resolve_axon_program_from_path(Path(axon_path)).ast
    program = normalize_closed_axon_file(program)
    program = elaborate_closed_axon_file(program)
    program = flatten_closed_axon_file(program)
    program = typecheck2_flat_axon_file(program)
    program = optimize_safe_flat_typed_axon_file(program)
    graph = lower_axon_program_to_graph_ir(program)
    return optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(backend_intrinsics="codegen2-vllm", max_iterations=8),
    )


def _ops(graph) -> set[str]:
    return {node.op.name for module in graph.modules for node in module.nodes}


def test_vllm_codegen_renders_dim_operands_with_dim_identifiers() -> None:
    input_ids = GraphValue("input_ids", TypeTensor("Tensor", ("B", "S")), ("B", "S"))
    end_dim = GraphValue("_v1", TypeDim(), None)
    out = GraphValue("_v2", TypeTensor("Tensor", ("S",)), ("S",))
    program = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(input_ids,),
                outputs=(GraphValueRef(out.name, out.type_expr, out.dims),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("core.binary.+"),
                        inputs=(GraphLiteral(1, TypeInt()), GraphLiteral(2, TypeInt())),
                        attrs={},
                        outputs=(end_dim,),
                        source_module="main",
                        type_expr=TypeDim(),
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_arange"),
                        inputs=(
                            GraphValueRef(input_ids.name, input_ids.type_expr, input_ids.dims),
                            GraphLiteral(0, TypeInt()),
                            GraphValueRef(end_dim.name, end_dim.type_expr, end_dim.dims),
                        ),
                        attrs={},
                        outputs=(out,),
                        source_module="main",
                        type_expr=out.type_expr,
                        dims=out.dims,
                    ),
                ),
                return_type_expr=out.type_expr,
            ),
        ),
        main_module="main",
        pragmas={},
    )

    code = emit_model_code_from_graph_ir(program)

    assert "_dim_v1 = (1 + 2)" in code
    assert "torch.arange(0, _dim_v1" in code
    assert "torch.arange(0, _v1" not in code


def test_vllm_selected_expert_rewrite_uses_primitive_provenance_for_packed_moe() -> None:
    graph = _vllm_optimized_graph("brainsurgery/synapse/models/qwen3-moe/generic-qwen3-moe.axon")

    assert "__vllm_selected_expert_packed_swiglu_ffn" in _ops(graph)


def test_vllm_selected_expert_rewrite_uses_primitive_provenance_for_unpacked_moe() -> None:
    graph = _vllm_optimized_graph("brainsurgery/synapse/models/mixtral/generic-mixtral.axon")

    assert "__vllm_selected_expert_swiglu_ffn" in _ops(graph)


def test_vllm_codegen_emits_selected_expert_scatter_helper() -> None:
    graph = _vllm_optimized_graph("brainsurgery/synapse/models/phimoe/Phi-MoE-Test.axon")
    code = emit_model_code_from_graph_ir(
        graph,
        model_config={"vocab_size": 65536, "hidden_size": 128, "num_hidden_layers": 4},
    )

    assert "def _scatter(self, x, index, src, dim):" in code
    assert "self._scatter(" in code
    assert "self._selected_expert_swiglu_ffn(" in code


def test_vllm_codegen_loads_helper_wrapped_vocab_projection_as_lm_head() -> None:
    graph = _vllm_optimized_graph("brainsurgery/synapse/models/phimoe/Phi-MoE-Test.axon")
    code = emit_model_code_from_graph_ir(
        graph,
        model_config={"vocab_size": 65536, "hidden_size": 128, "num_hidden_layers": 4},
    )

    assert "_ckpt_to_model.setdefault('lm_head.weight', []).append('lm_head.weight')" in code


def test_vllm_codegen_preserves_preloop_position_embedding_addend() -> None:
    graph = _vllm_optimized_graph("brainsurgery/synapse/models/xglm/generic-xglm.axon")
    code = emit_model_code_from_graph_ir(
        graph,
        model_config={
            "vocab_size": 256008,
            "d_model": 1024,
            "num_layers": 24,
            "attention_heads": 16,
        },
    )

    assert "model.embed_positions.weight" in code
    assert "hidden_states = hidden_states + (" in code


def test_vllm_classifier_propagates_nested_path_formal_scopes() -> None:
    graph = _vllm_optimized_graph("brainsurgery/synapse/models/deepseekv4/generic-deepseek-v4.axon")
    classification = classify_graph_for_vllm(graph)

    missing: list[tuple[str, str, str]] = []
    for module in graph.modules:
        for node in module.nodes:
            if node.op.name != "Params.param" or not node.inputs:
                continue
            path = node.inputs[0]
            if not isinstance(path, GraphPath):
                continue
            if not any("{path}" in part for part in path.parts):
                continue
            in_repeated_module = any(
                node.id == repeated or node.id.startswith(repeated + ":")
                for repeated in classification.repeated_module_names
            )
            if not in_repeated_module or module.name not in classification.module_scope_parts:
                missing.append((module.name, node.id, "@@" + ".".join(path.parts)))

    assert missing == []


def test_vllm_codegen_rejects_rank4_clean_forward_carry() -> None:
    graph = _vllm_optimized_graph("brainsurgery/synapse/models/deepseekv4/generic-deepseek-v4.axon")

    with pytest.raises(NotImplementedError, match="rank-3 tensor layer carries"):
        emit_model_code_from_graph_ir(
            graph,
            model_config={"vocab_size": 65536, "hidden_size": 128, "num_hidden_layers": 6},
        )
