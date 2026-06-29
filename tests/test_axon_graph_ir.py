from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from brainsurgery.synapse import lower_axon_program_to_graph_ir, parse_axon_program
from brainsurgery.synapse.axon import (
    elaborate_closed_axon_file,
    flatten_closed_axon_file,
    normalize_closed_axon_file,
    resolve_axon_program_from_path,
    typecheck2_flat_axon_file,
)
from brainsurgery.synapse.axon.ast import (
    Constraint,
    DimExprBinary,
    TypeAny,
    TypeBool,
    TypeDim,
    TypeFloat,
    TypeInt,
    TypeList,
    TypeNull,
    TypeOptional,
    TypePath,
    TypeString,
    TypeTensor,
    TypeTuple,
    TypeVar,
    render_axon_file,
)
from brainsurgery.synapse.axon.codegen2_torch import Codegen2GraphModel, emit_model_code_from_graph_ir
from brainsurgery.synapse.axon.codegen2_tinygrad import (
    emit_model_code_from_graph_ir as emit_tinygrad_model_code_from_graph_ir,
    tinygrad_op_table_markdown,
)
from brainsurgery.synapse.axon.codegen2_mlx import (
    emit_model_code_from_graph_ir as emit_mlx_model_code_from_graph_ir,
    mlx_op_table_markdown,
    torch_state_dict_to_mlx,
)
from brainsurgery.synapse.axon.codegen2_triton import (
    emit_model_code_from_graph_ir as emit_triton_model_code_from_graph_ir,
)
from brainsurgery.synapse.axon.analysis import infer_axon_definition_effects, op_effect
from brainsurgery.synapse.axon.graph_ir import (
    GraphDomainAnalysis,
    GraphDomainFact,
    GraphLiteral,
    GraphExpr,
    GraphEffect,
    UsageClass,
    GraphOptimizeConfig,
    GraphDomainInterval,
    GraphDomainKind,
    GraphModule,
    GraphNode,
    GraphOp,
    GraphPath,
    GraphProgram,
    GraphValue,
    GraphValueRef,
    graph_program_to_axon_file,
    graph_module_effect,
    graph_module_usage,
    graph_domain_definition_comments,
    infer_graph_provenance,
    infer_graph_module_effects,
    infer_graph_module_usages,
    infer_graph_ownership,
    infer_main_module_domain_facts,
    optimize_graph_program,
    prune_graph_to_main,
    render_graph_program_to_dot,
    validate_graph_domain_analysis,
    validate_graph_program,
)
from brainsurgery.synapse.axon.graph_ir.substitute import (
    substitute_dim_token,
    substitute_graph_operand_dims,
)


def _typed(program, *, main_module: str):
    normalized = normalize_closed_axon_file(program, main_module=main_module)
    elaborated = elaborate_closed_axon_file(normalized, main_module=main_module)
    flat = flatten_closed_axon_file(elaborated, main_module=main_module)
    return typecheck2_flat_axon_file(flat, main_module=main_module)


def test_graph_ir_lowers_flat_typed_axon_without_synapse_spec() -> None:
    program = parse_axon_program(
        """
main :: Int -> Int
main x = do
  y <- x + 1
  y <- y + x
  return y
"""
    )

    graph = lower_axon_program_to_graph_ir(_typed(program, main_module="main"), main_module="main")
    module = graph.modules[-1]

    assert graph.main_module == "main"
    assert [node.op.name for node in module.nodes] == ["core.binary.+", "core.binary.+"]
    assert len({output.name for node in module.nodes for output in node.outputs}) == 2
    assert module.nodes[0].outputs[0].name == "y"
    assert module.nodes[1].outputs[0].name != "y"
    assert isinstance(module.outputs[0], GraphValueRef)
    assert module.outputs[0].name == module.nodes[1].outputs[0].name


def test_global_initialization_binding_survives_to_graph_ir() -> None:
    program = parse_axon_program(
        """
MODEL_DIM <- (768 :: Dim)

main :: Tensor[B,S,MODEL_DIM] -> Tensor[B,S,MODEL_DIM]
main x = x
"""
    )

    assert program.pragmas["main"] == "main"
    assert program.modules[0].name == "MODEL_DIM"
    assert program.modules[0].is_global_binding
    assert "MODEL_DIM <- (768 :: Dim)" in render_axon_file(program)

    graph = lower_axon_program_to_graph_ir(_typed(program, main_module="main"), main_module="main")
    global_module = next(module for module in graph.modules if module.name == "MODEL_DIM")

    assert global_module.is_global_binding
    assert not global_module.inputs
    assert len(global_module.outputs) == 1
    rendered = render_axon_file(graph_program_to_axon_file(graph), show_types=True)
    assert "MODEL_DIM <- do" in rendered
    assert "main :: Tensor[B,S,MODEL_DIM] -> Tensor[B,S,MODEL_DIM]" in rendered


def test_graph_ir_renderer_uses_global_value_refs_directly_without_shadowing() -> None:
    dim_t = TypeDim()
    tensor_t = TypeTensor("Tensor", ("B", "S"))
    path_t = TypePath()
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="MODEL_DIM",
                inputs=(),
                outputs=(GraphLiteral(768, dim_t),),
                output_names=("out",),
                nodes=(),
                return_type_expr=dim_t,
                is_global_binding=True,
            ),
            GraphModule(
                name="main",
                inputs=(GraphValue("x", TypeTensor("Tensor", ("B", "S")), ("B", "S")),),
                outputs=(GraphValueRef("tok", tensor_t, tensor_t.dims),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:0",
                        op=GraphOp("core.ascribe"),
                        inputs=(GraphValueRef("MODEL_DIM", dim_t),),
                        attrs={},
                        outputs=(GraphValue("dim_arg", dim_t),),
                        source_module="main",
                        type_expr=dim_t,
                    ),
                    GraphNode(
                        id="main:1",
                        op=GraphOp("embedding"),
                        inputs=(
                            GraphPath(True, ("wte",)),
                            GraphValueRef("x", TypeTensor("Tensor", ("B", "S")), ("B", "S")),
                            GraphValueRef("dim_arg", dim_t),
                            GraphLiteral(None, TypeNull()),
                        ),
                        attrs={},
                        outputs=(GraphValue("tok", tensor_t, tensor_t.dims),),
                        source_module="main",
                        type_expr=tensor_t,
                        dims=tensor_t.dims,
                    ),
                ),
                return_type_expr=tensor_t,
            ),
            GraphModule(
                name="embedding",
                inputs=(
                    GraphValue("path", path_t),
                    GraphValue("x", TypeTensor("Tensor", ("B", "S")), ("B", "S")),
                    GraphValue("dim", dim_t),
                    GraphValue("scale", TypeNull()),
                ),
                outputs=(GraphValueRef("x", tensor_t, tensor_t.dims),),
                output_names=("out",),
                nodes=(),
                return_type_expr=tensor_t,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    rendered = render_axon_file(graph_program_to_axon_file(graph), show_types=True)

    assert "__flat_1 <- ((MODEL_DIM)" not in rendered
    assert "dim_arg <-" not in rendered
    assert "((MODEL_DIM :: Dim))" in rendered
    assert "MODEL_DIM <- do" in rendered
    assert path_t == TypePath()


def test_graph_optimizer_prunes_dead_formal_and_rewrites_all_callsites() -> None:
    int_t = TypeInt()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", int_t), GraphValue("unused", int_t)),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("a", int_t), GraphValue("b", int_t)),
        outputs=(GraphValueRef("out", TypeTuple((int_t, int_t))),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphValueRef("a", int_t), GraphLiteral(1024, int_t)),
                attrs={},
                outputs=(GraphValue("left", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("helper"),
                inputs=(GraphValueRef("b", int_t), GraphLiteral(1024, int_t)),
                attrs={},
                outputs=(GraphValue("right", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
            GraphNode(
                id="main:3",
                op=GraphOp("core.tuple"),
                inputs=(GraphValueRef("left", int_t), GraphValueRef("right", int_t)),
                attrs={},
                outputs=(GraphValue("out", TypeTuple((int_t, int_t))),),
                source_module="main",
                type_expr=TypeTuple((int_t, int_t)),
            ),
        ),
        return_type_expr=TypeTuple((int_t, int_t)),
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            specialize_definitions="off",
            inline_safe=False,
            common_subexpression_elimination=False,
        ),
    )
    optimized_helper = next(module for module in optimized.modules if module.name == "helper")
    optimized_main = next(module for module in optimized.modules if module.name == "main")

    assert [value.name for value in optimized_helper.inputs] == ["x"]
    helper_calls = [node for node in optimized_main.nodes if node.op.name == "helper"]
    assert [len(node.inputs) for node in helper_calls] == [1, 1]


def test_graph_specialization_rejects_unbound_term_dim_clone() -> None:
    dim_t = TypeDim()
    float_t = TypeFloat()
    source_t = TypeTensor("Tensor", ("B", "K"))
    result_t = TypeTensor("Tensor", ("B",))
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="callee",
                inputs=(
                    GraphValue("x", source_t, source_t.dims),
                    GraphValue("scale", float_t),
                ),
                outputs=(GraphValueRef("y", result_t, result_t.dims),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="callee:1",
                        op=GraphOp("_slice"),
                        inputs=(
                            GraphValueRef("x", source_t, source_t.dims),
                            GraphLiteral(0, TypeInt()),
                            GraphValueRef("K", dim_t),
                            GraphValueRef("scale", float_t),
                        ),
                        attrs={},
                        outputs=(GraphValue("y", result_t, result_t.dims),),
                        source_module="callee",
                        type_expr=result_t,
                        dims=result_t.dims,
                    ),
                ),
                return_type_expr=result_t,
            ),
            GraphModule(
                name="main",
                inputs=(GraphValue("x", source_t, source_t.dims),),
                outputs=(GraphValueRef("y", result_t, result_t.dims),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("callee"),
                        inputs=(
                            GraphValueRef("x", source_t, source_t.dims),
                            GraphLiteral(0.5, float_t),
                        ),
                        attrs={},
                        outputs=(GraphValue("y", result_t, result_t.dims),),
                        source_module="main",
                        type_expr=result_t,
                        dims=result_t.dims,
                    ),
                ),
                return_type_expr=result_t,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(
            constant_dim_substitution=False,
            common_subexpression_elimination=False,
            inline_safe=False,
        ),
    )

    assert {module.name for module in optimized.modules} == {"callee", "main"}


def test_graph_ir_renderer_uses_zero_arg_global_exprs_directly_in_arguments() -> None:
    bool_t = TypeBool()
    dim_t = TypeDim()
    int_t = TypeInt()
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="NUM_LAYERS",
                inputs=(),
                outputs=(GraphLiteral(12, dim_t),),
                output_names=("out",),
                nodes=(),
                return_type_expr=dim_t,
                is_global_binding=True,
            ),
            GraphModule(
                name="main",
                inputs=(GraphValue("i", int_t),),
                outputs=(GraphValueRef("done", bool_t),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("core.binary.>="),
                        inputs=(
                            GraphValueRef("i", int_t),
                            GraphExpr(
                                op=GraphOp("core.ascribe"),
                                inputs=(
                                    GraphExpr(
                                        op=GraphOp("NUM_LAYERS"),
                                        inputs=(),
                                        attrs={},
                                        type_expr=dim_t,
                                    ),
                                ),
                                attrs={},
                                type_expr=dim_t,
                            ),
                        ),
                        attrs={},
                        outputs=(GraphValue("done", bool_t),),
                        source_module="main",
                        type_expr=bool_t,
                    ),
                ),
                return_type_expr=bool_t,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    rendered = render_axon_file(graph_program_to_axon_file(graph), show_types=True)

    assert "_arg2 <-" not in rendered
    assert "done <- (((i :: Int) >= (NUM_LAYERS :: Dim)) :: Bool)" in rendered


def test_graph_ir_renderer_uses_compact_names_for_generated_atomization_temps() -> None:
    int_t = TypeInt()
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("x", int_t),),
                outputs=(GraphValueRef("_v1", int_t),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_consumer"),
                        inputs=(
                            GraphExpr(
                                op=GraphOp("_producer"),
                                inputs=(GraphValueRef("x", int_t),),
                                attrs={},
                                type_expr=int_t,
                            ),
                        ),
                        attrs={},
                        outputs=(GraphValue("_v1", int_t),),
                        source_module="main",
                        type_expr=int_t,
                    ),
                ),
                return_type_expr=int_t,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    rendered = render_axon_file(graph_program_to_axon_file(graph), show_types=True)

    assert "_v1_arg1" not in rendered
    assert "_v1 <- ((_producer ((x :: Int))) :: Int)" in rendered
    assert "_v2 <- ((_consumer ((_v1 :: Int))) :: Int)" in rendered


def test_graph_ir_renderer_preserves_shadowed_global_names_as_locals() -> None:
    dim_t = TypeDim()
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="MODEL_DIM",
                inputs=(),
                outputs=(GraphLiteral(768, dim_t),),
                output_names=("out",),
                nodes=(),
                return_type_expr=dim_t,
                is_global_binding=True,
            ),
            GraphModule(
                name="main",
                inputs=(GraphValue("MODEL_DIM", dim_t),),
                outputs=(GraphValueRef("out", dim_t),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("core.alias"),
                        inputs=(GraphValueRef("MODEL_DIM", dim_t),),
                        attrs={},
                        outputs=(GraphValue("out", dim_t),),
                        source_module="main",
                        type_expr=dim_t,
                    ),
                ),
                return_type_expr=dim_t,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    rendered = render_axon_file(graph_program_to_axon_file(graph), show_types=True)

    assert "main :: Dim -> Dim" in rendered
    assert "main MODEL_DIM = do" in rendered
    assert "return (out :: Dim)" in rendered


def test_graph_ir_prunes_unreachable_modules_from_main() -> None:
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphLiteral(1, TypeInt()),),
        output_names=("out",),
        nodes=(),
        return_type_expr=TypeInt(),
    )
    unused = GraphModule(
        name="unused",
        inputs=(),
        outputs=(GraphLiteral(2, TypeInt()),),
        output_names=("out",),
        nodes=(),
        return_type_expr=TypeInt(),
    )
    graph = GraphProgram(modules=(unused, main), main_module="main", pragmas={})

    pruned = prune_graph_to_main(graph)

    assert [module.name for module in pruned.modules] == ["main"]


def test_graph_ir_prune_preserves_path_template_value_dependencies() -> None:
    int_t = TypeInt()
    string_t = TypeString()
    cfg = GraphModule(
        name="CFG",
        inputs=(),
        outputs=(GraphLiteral("text_config", string_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=string_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("Params.param"),
                inputs=(GraphPath(True, ("{CFG}.weight",)),),
                attrs={},
                outputs=(GraphValue("unused", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    pruned = prune_graph_to_main(GraphProgram(modules=(cfg, main), main_module="main", pragmas={}))

    assert [module.name for module in pruned.modules] == ["CFG", "main"]


def test_graph_ir_validator_rejects_unbound_path_template_symbol() -> None:
    int_t = TypeInt()
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("x", int_t),),
                outputs=(GraphValueRef("x", int_t),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("Params.param"),
                        inputs=(GraphPath(True, ("{CFG}.weight",)),),
                        attrs={},
                        outputs=(GraphValue("unused", int_t),),
                        source_module="main",
                        type_expr=int_t,
                    ),
                ),
                return_type_expr=int_t,
            ),
        ),
        main_module="main",
        pragmas={},
    )

    with pytest.raises(ValueError, match="path template uses undefined value 'CFG'"):
        validate_graph_program(graph)


def test_topk_variadic_wrapper_does_not_collapse_prefix_shape() -> None:
    from brainsurgery.synapse.ops.topk import type_rule

    class Helpers:
        @staticmethod
        def type_dims(tp):
            return tp.dims if isinstance(tp, TypeTensor) else None

        @staticmethod
        def expr_to_dim_token(_value):
            return "k"

    inferred = type_rule(
        arg_types=(TypeTensor("Tensor", ("..S",)), TypeDim(), TypeInt(), TypeBool(), TypeBool()),
        kwarg_types={},
        args=(object(), object(), object(), object(), object()),
        kwargs={},
        helpers=Helpers(),
    )

    assert inferred is None


def test_graph_ir_optimizer_folds_local_total_core_nodes() -> None:
    int_t = TypeInt()
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(),
                outputs=(GraphValueRef("z", int_t),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="n1",
                        op=GraphOp("core.binary.+"),
                        inputs=(GraphLiteral(1, int_t), GraphLiteral(2, int_t)),
                        attrs={},
                        outputs=(GraphValue("x", int_t),),
                        source_module="main",
                        type_expr=int_t,
                    ),
                    GraphNode(
                        id="n2",
                        op=GraphOp("core.alias"),
                        inputs=(GraphValueRef("x", int_t),),
                        attrs={},
                        outputs=(GraphValue("z", int_t),),
                        source_module="main",
                        type_expr=int_t,
                    ),
                    GraphNode(
                        id="n3",
                        op=GraphOp("core.binary.+"),
                        inputs=(GraphLiteral(3, int_t), GraphLiteral(4, int_t)),
                        attrs={},
                        outputs=(GraphValue("unused", int_t),),
                        source_module="main",
                        type_expr=int_t,
                    ),
                ),
                return_type_expr=int_t,
            ),
        ),
        main_module="main",
        pragmas={},
    )

    optimized = optimize_graph_program(graph)

    module = optimized.modules[0]
    assert module.nodes == ()
    assert module.outputs == (GraphLiteral(3, int_t),)


def test_graph_ir_optimizer_keeps_unused_partial_calls() -> None:
    int_t = TypeInt()
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("x", int_t),),
                outputs=(GraphValueRef("x", int_t),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="n1",
                        op=GraphOp("Params.param"),
                        inputs=(GraphPath(True, ("missing",)),),
                        attrs={},
                        outputs=(GraphValue("unused", int_t),),
                        source_module="main",
                        type_expr=int_t,
                    ),
                ),
                return_type_expr=int_t,
            ),
        ),
        main_module="main",
        pragmas={},
    )

    optimized = optimize_graph_program(graph)

    assert [node.op.name for node in optimized.modules[0].nodes] == ["Params.param"]


def test_graph_ir_effect_model_keeps_unknown_calls_partial() -> None:
    int_t = TypeInt()
    module = GraphModule(
        name="main",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("Params.param"),
                inputs=(GraphPath(True, ("missing",)),),
                attrs={},
                outputs=(GraphValue("unused", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    assert graph_module_effect(module) == GraphEffect.PARTIAL_PURE


def test_purity_effects_track_config_defaults_and_pure_primitives() -> None:
    int_t = TypeInt()
    assert op_effect("core.binary.+") == GraphEffect.TOTAL_PURE
    assert op_effect("_add") == GraphEffect.TOTAL_PURE
    assert op_effect("_list_init") == GraphEffect.TOTAL_PURE
    assert op_effect("_config_dim") == GraphEffect.PARTIAL_PURE
    assert (
        op_effect("_config_dim", attrs={"default": GraphLiteral(1024, int_t)})
        == GraphEffect.TOTAL_PURE
    )
    node = GraphNode(
        id="n",
        op=GraphOp("_config_dim"),
        inputs=(GraphPath(True, ("n_layer",)), GraphLiteral(12, int_t)),
        attrs={},
        outputs=(GraphValue("x", int_t),),
        source_module="main",
        type_expr=int_t,
    )
    assert graph_module_effect(
        GraphModule(
            name="main",
            inputs=(),
            outputs=(GraphValueRef("x", int_t),),
            output_names=("out",),
            nodes=(node,),
            return_type_expr=int_t,
        )
    ) == GraphEffect.TOTAL_PURE


def test_graph_purity_treats_fully_overwritten_empty_like_as_total_pure() -> None:
    tensor_t = TypeTensor("Tensor", ("B", "S"))
    float_t = TypeFloat()
    filled = GraphModule(
        name="filled",
        inputs=(GraphValue("x", tensor_t, tensor_t.dims),),
        outputs=(GraphValueRef("out", tensor_t, tensor_t.dims),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="filled:1",
                op=GraphOp("_empty_like"),
                inputs=(GraphValueRef("x", tensor_t, tensor_t.dims), GraphLiteral(None, TypeNull())),
                attrs={},
                outputs=(GraphValue("tmp", tensor_t, tensor_t.dims),),
                source_module="filled",
                type_expr=tensor_t,
                dims=tensor_t.dims,
            ),
            GraphNode(
                id="filled:2",
                op=GraphOp("_fill"),
                inputs=(
                    GraphValueRef("tmp", tensor_t, tensor_t.dims),
                    GraphLiteral(0.0, float_t),
                    GraphLiteral(None, TypeNull()),
                ),
                attrs={},
                outputs=(GraphValue("out", tensor_t, tensor_t.dims),),
                source_module="filled",
                type_expr=tensor_t,
                dims=tensor_t.dims,
            ),
        ),
        return_type_expr=tensor_t,
    )
    leaked = GraphModule(
        name="leaked",
        inputs=(GraphValue("x", tensor_t, tensor_t.dims),),
        outputs=(GraphValueRef("tmp", tensor_t, tensor_t.dims),),
        output_names=("out",),
        nodes=(filled.nodes[0],),
        return_type_expr=tensor_t,
    )

    assert graph_module_effect(filled) == GraphEffect.TOTAL_PURE
    assert graph_module_effect(leaked) == GraphEffect.PARTIAL_PURE


def test_graph_usage_marks_fresh_zero_arg_global_refs_affine() -> None:
    list_t = TypeList(TypeAny())
    list_init = GraphModule(
        name="List.init",
        inputs=(),
        outputs=(GraphValueRef("__flat_1", list_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="List.init:1",
                op=GraphOp("_list_init"),
                inputs=(),
                attrs={},
                outputs=(GraphValue("__flat_1", list_t),),
                source_module="List.init",
                type_expr=list_t,
            ),
        ),
        return_type_expr=list_t,
    )
    cache_init = GraphModule(
        name="Cache.init",
        inputs=(),
        outputs=(GraphExpr(GraphOp("List.init"), (), {}, list_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=list_t,
    )

    effects = infer_graph_module_effects((list_init, cache_init))
    usages = infer_graph_module_usages((list_init, cache_init))

    assert effects["List.init"] == GraphEffect.TOTAL_PURE
    assert effects["Cache.init"] == GraphEffect.TOTAL_PURE
    assert usages["List.init"] == UsageClass.AFFINE
    assert usages["Cache.init"] == UsageClass.AFFINE


def test_graph_usage_marks_fresh_tensor_allocations_affine() -> None:
    tensor_t = TypeTensor("Tensor", (2, 3))
    ref = GraphValue("ref", tensor_t, tensor_t.dims)
    shape = GraphExpr(
        GraphOp("core.list"),
        (GraphLiteral(2, TypeDim()), GraphLiteral(3, TypeDim())),
        {},
        TypeList(TypeDim()),
    )
    module = GraphModule(
        name="main",
        inputs=(ref,),
        outputs=(GraphValueRef("z", tensor_t, tensor_t.dims),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("_zeros"),
                inputs=(GraphValueRef("ref", tensor_t, tensor_t.dims), shape, GraphLiteral(None, TypeNull())),
                attrs={},
                outputs=(GraphValue("z", tensor_t, tensor_t.dims),),
                source_module="main",
                type_expr=tensor_t,
                dims=tensor_t.dims,
            ),
        ),
        return_type_expr=tensor_t,
    )

    effects = infer_graph_module_effects((module,))
    usages = infer_graph_module_usages((module,))

    assert effects["main"] == GraphEffect.TOTAL_PURE
    assert usages["main"] == UsageClass.AFFINE


def test_graph_ir_optimizer_does_not_cse_fresh_tensor_allocations() -> None:
    tensor_t = TypeTensor("Tensor", (2, 3))
    ref = GraphValue("ref", tensor_t, tensor_t.dims)
    shape = GraphExpr(
        GraphOp("core.list"),
        (GraphLiteral(2, TypeDim()), GraphLiteral(3, TypeDim())),
        {},
        TypeList(TypeDim()),
    )
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(ref,),
                outputs=(
                    GraphValueRef("keys", tensor_t, tensor_t.dims),
                    GraphValueRef("values", tensor_t, tensor_t.dims),
                ),
                output_names=("keys", "values"),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_zeros"),
                        inputs=(GraphValueRef("ref", tensor_t, tensor_t.dims), shape, GraphLiteral(None, TypeNull())),
                        attrs={},
                        outputs=(GraphValue("keys", tensor_t, tensor_t.dims),),
                        source_module="main",
                        type_expr=tensor_t,
                        dims=tensor_t.dims,
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_zeros"),
                        inputs=(GraphValueRef("ref", tensor_t, tensor_t.dims), shape, GraphLiteral(None, TypeNull())),
                        attrs={},
                        outputs=(GraphValue("values", tensor_t, tensor_t.dims),),
                        source_module="main",
                        type_expr=tensor_t,
                        dims=tensor_t.dims,
                    ),
                ),
                return_type_expr=TypeTuple((tensor_t, tensor_t)),
            ),
        ),
        main_module="main",
        pragmas={},
    )

    optimized = optimize_graph_program(graph)
    nodes = optimized.modules[0].nodes

    assert [node.op.name for node in nodes] == ["_zeros", "_zeros"]
    assert optimized.modules[0].outputs == (
        GraphValueRef("keys", tensor_t, tensor_t.dims),
        GraphValueRef("values", tensor_t, tensor_t.dims),
    )


def test_graph_ownership_allows_inplace_assign_slice_for_owned_base() -> None:
    tensor_t = TypeTensor("Tensor", (2, 3))
    ref = GraphValue("ref", tensor_t, tensor_t.dims)
    shape = GraphExpr(
        GraphOp("core.list"),
        (GraphLiteral(2, TypeDim()), GraphLiteral(3, TypeDim())),
        {},
        TypeList(TypeDim()),
    )
    src = GraphValue("src", tensor_t, tensor_t.dims)
    assign_node = GraphNode(
        id="main:2",
        op=GraphOp("_assign_slice"),
        inputs=(
            GraphValueRef("base", tensor_t, tensor_t.dims),
            GraphValueRef("src", tensor_t, tensor_t.dims),
            GraphLiteral(0, TypeInt()),
            GraphLiteral(0, TypeDim()),
            GraphLiteral(1, TypeDim()),
        ),
        attrs={},
        outputs=(GraphValue("updated", tensor_t, tensor_t.dims),),
        source_module="main",
        type_expr=tensor_t,
        dims=tensor_t.dims,
    )
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(ref, src),
                outputs=(GraphValueRef("updated", tensor_t, tensor_t.dims),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_zeros"),
                        inputs=(GraphValueRef("ref", tensor_t, tensor_t.dims), shape, GraphLiteral(None, TypeNull())),
                        attrs={},
                        outputs=(GraphValue("base", tensor_t, tensor_t.dims),),
                        source_module="main",
                        type_expr=tensor_t,
                        dims=tensor_t.dims,
                    ),
                    assign_node,
                ),
                return_type_expr=tensor_t,
            ),
        ),
        main_module="main",
        pragmas={},
    )

    ownership = infer_graph_ownership(graph)

    assert "main:2" in ownership.inplace_assign_slice_node_ids


def test_graph_ownership_rejects_inplace_assign_slice_when_old_base_is_live() -> None:
    tensor_t = TypeTensor("Tensor", (2, 3))
    ref = GraphValue("ref", tensor_t, tensor_t.dims)
    shape = GraphExpr(
        GraphOp("core.list"),
        (GraphLiteral(2, TypeDim()), GraphLiteral(3, TypeDim())),
        {},
        TypeList(TypeDim()),
    )
    src = GraphValue("src", tensor_t, tensor_t.dims)
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(ref, src),
                outputs=(
                    GraphValueRef("updated", tensor_t, tensor_t.dims),
                    GraphValueRef("base", tensor_t, tensor_t.dims),
                ),
                output_names=("updated", "base"),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_zeros"),
                        inputs=(GraphValueRef("ref", tensor_t, tensor_t.dims), shape, GraphLiteral(None, TypeNull())),
                        attrs={},
                        outputs=(GraphValue("base", tensor_t, tensor_t.dims),),
                        source_module="main",
                        type_expr=tensor_t,
                        dims=tensor_t.dims,
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_assign_slice"),
                        inputs=(
                            GraphValueRef("base", tensor_t, tensor_t.dims),
                            GraphValueRef("src", tensor_t, tensor_t.dims),
                            GraphLiteral(0, TypeInt()),
                            GraphLiteral(0, TypeDim()),
                            GraphLiteral(1, TypeDim()),
                        ),
                        attrs={},
                        outputs=(GraphValue("updated", tensor_t, tensor_t.dims),),
                        source_module="main",
                        type_expr=tensor_t,
                        dims=tensor_t.dims,
                    ),
                ),
                return_type_expr=TypeTuple((tensor_t, tensor_t)),
            ),
        ),
        main_module="main",
        pragmas={},
    )

    ownership = infer_graph_ownership(graph)

    assert "main:2" not in ownership.inplace_assign_slice_node_ids


def test_graph_ir_optimizer_inlines_effectful_zero_arg_forwarder_without_sharing() -> None:
    list_t = TypeList(TypeAny())
    list_init = GraphModule(
        name="List.init",
        inputs=(),
        outputs=(GraphValueRef("__flat_1", list_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="List.init:1",
                op=GraphOp("_list_init"),
                inputs=(),
                attrs={},
                outputs=(GraphValue("__flat_1", list_t),),
                source_module="List.init",
                type_expr=list_t,
            ),
        ),
        return_type_expr=list_t,
    )
    cache_init = GraphModule(
        name="Cache.init",
        inputs=(),
        outputs=(GraphExpr(GraphOp("List.init"), (), {}, list_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=list_t,
    )
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("out", list_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("Cache.init"),
                inputs=(),
                attrs={},
                outputs=(GraphValue("out", list_t),),
                source_module="main",
                type_expr=list_t,
            ),
        ),
        return_type_expr=list_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(list_init, cache_init, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off"),
    )

    assert [module.name for module in optimized.modules] == ["main"]
    assert optimized.modules[0].nodes[0].op.name == "_list_init"


def test_graph_ir_optimizer_refreshes_primitive_shape_rules() -> None:
    q_t = TypeTensor("Tensor", ("B", "H", "S", "DH"))
    k_t = TypeTensor("Tensor", ("B", "H", "K", "DH"))
    kt_t = TypeTensor("Tensor", ("B", "H", "DH", "K"))
    scores_t = TypeTensor("Tensor", ("B", "H", "S", "K"))
    stale_t = TypeTensor("Tensor", ("B", "H", "S", "..R"))
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("q", q_t), GraphValue("k", k_t)),
                outputs=(GraphValueRef("scores", stale_t),),
                output_names=("scores",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_transpose"),
                        inputs=(
                            GraphValueRef("k", k_t),
                            GraphLiteral(2, TypeInt()),
                            GraphLiteral(3, TypeInt()),
                        ),
                        attrs={},
                        outputs=(GraphValue("kt", TypeTensor("Tensor", ("B", "H", "DH", "..RB"))),),
                        source_module="main",
                        type_expr=TypeTensor("Tensor", ("B", "H", "DH", "..RB")),
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_matmul"),
                        inputs=(GraphValueRef("q", q_t), GraphValueRef("kt", kt_t)),
                        attrs={},
                        outputs=(GraphValue("scores", stale_t),),
                        source_module="main",
                        type_expr=stale_t,
                    ),
                ),
                return_type_expr=stale_t,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(
            specialize_definitions="off",
            inline_safe=False,
            constant_folding=False,
        ),
    )

    nodes = optimized.modules[0].nodes
    assert nodes[0].outputs[0].type_expr == kt_t
    assert nodes[1].outputs[0].type_expr == scores_t
    assert optimized.modules[0].outputs[0].type_expr == scores_t


def test_graph_ir_optimizer_select_uses_broadcast_branch_shape() -> None:
    bool_t = TypeBool()
    broad_t = TypeTensor("Tensor", ("B", 1, "Q", "K"))
    narrow_t = TypeTensor("Tensor", ("B", 1, 1, "K"))
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(
                    GraphValue("cond", bool_t),
                    GraphValue("keep", broad_t),
                    GraphValue("pad", narrow_t),
                ),
                outputs=(GraphValueRef("out", narrow_t),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("core.select"),
                        inputs=(
                            GraphValueRef("cond", bool_t),
                            GraphValueRef("keep", broad_t),
                            GraphValueRef("pad", narrow_t),
                        ),
                        attrs={},
                        outputs=(GraphValue("out", narrow_t),),
                        source_module="main",
                        type_expr=narrow_t,
                    ),
                ),
                return_type_expr=narrow_t,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(
            specialize_definitions="off",
            inline_safe=False,
            constant_folding=False,
        ),
    )

    assert optimized.modules[0].nodes[0].type_expr == broad_t
    assert optimized.modules[0].nodes[0].outputs[0].type_expr == broad_t
    assert optimized.modules[0].outputs[0].type_expr == broad_t


def test_graph_ir_optimizer_refreshes_broadcast_and_slice_shape_rules() -> None:
    cond_t = TypeTensor("Tensor", ("B", 1, "S", "K"))
    value_t = TypeTensor("Tensor", ("B", "H", "S", "K"))
    out_t = TypeTensor("Tensor", ("B", "H", "S", "K"))
    stale_t = TypeTensor("Tensor", ("B", 1, "S", "K"))
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("cond", cond_t), GraphValue("value", value_t)),
                outputs=(GraphValueRef("out", stale_t),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_where"),
                        inputs=(
                            GraphValueRef("cond", cond_t),
                            GraphValueRef("value", value_t),
                            GraphLiteral(0, TypeInt()),
                        ),
                        attrs={},
                        outputs=(GraphValue("masked", stale_t),),
                        source_module="main",
                        type_expr=stale_t,
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_slice"),
                        inputs=(
                            GraphValueRef("masked", out_t),
                            GraphLiteral(-1, TypeInt()),
                            GraphLiteral(0, TypeDim()),
                            GraphLiteral("K", TypeDim()),
                        ),
                        attrs={},
                        outputs=(GraphValue("out", TypeTensor("Tensor", ("..R",))),),
                        source_module="main",
                        type_expr=TypeTensor("Tensor", ("..R",)),
                    ),
                ),
                return_type_expr=stale_t,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(
            specialize_definitions="off",
            inline_safe=False,
            constant_folding=False,
        ),
    )

    nodes = optimized.modules[0].nodes
    assert nodes[0].outputs[0].type_expr == out_t
    assert nodes[1].outputs[0].type_expr == out_t
    assert optimized.modules[0].outputs[0].type_expr == out_t


def test_graph_ir_optimizer_forwarder_inlining_refreshes_broadcast_shape() -> None:
    broad_t = TypeTensor("Tensor", ("B", 1, "Q", "K"))
    narrow_t = TypeTensor("Tensor", ("B", 1, 1, "K"))
    wrapper_result_t = TypeTensor("Tensor", ("..R",))
    wrapper = GraphModule(
        name="Tensor.and",
        inputs=(GraphValue("x", broad_t), GraphValue("y", narrow_t)),
        outputs=(GraphValueRef("out", wrapper_result_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="Tensor.and:1",
                op=GraphOp("_and"),
                inputs=(GraphValueRef("x", broad_t), GraphValueRef("y", narrow_t)),
                attrs={},
                outputs=(GraphValue("out", wrapper_result_t),),
                source_module="Tensor.and",
                type_expr=wrapper_result_t,
            ),
        ),
        return_type_expr=wrapper_result_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("keep", broad_t), GraphValue("pad", narrow_t)),
        outputs=(GraphValueRef("out", narrow_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("Tensor.and"),
                inputs=(GraphValueRef("keep", broad_t), GraphValueRef("pad", narrow_t)),
                attrs={},
                outputs=(GraphValue("out", narrow_t),),
                source_module="main",
                type_expr=narrow_t,
            ),
        ),
        return_type_expr=narrow_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(wrapper, main), main_module="main", pragmas={"main": "main"}),
        config=GraphOptimizeConfig(
            specialize_definitions="off",
            constant_folding=False,
        ),
    )

    optimized_main = {module.name: module for module in optimized.modules}["main"]
    assert optimized_main.nodes[0].op.name == "_and"
    assert optimized_main.nodes[0].outputs[0].type_expr == broad_t
    assert optimized_main.outputs[0].type_expr == broad_t


def test_graph_ir_optimizer_does_not_collapse_distinct_variadic_broadcast_rows() -> None:
    wrapper_t = TypeTensor("Tensor", ("..R",))
    wrapper = GraphModule(
        name="Tensor.and",
        inputs=(
            GraphValue("x", TypeTensor("Tensor", ("..S",))),
            GraphValue("y", TypeTensor("Tensor", ("..T",))),
        ),
        outputs=(GraphValueRef("out", wrapper_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="Tensor.and:1",
                op=GraphOp("_and"),
                inputs=(
                    GraphValueRef("x", TypeTensor("Tensor", ("..S",))),
                    GraphValueRef("y", TypeTensor("Tensor", ("..T",))),
                ),
                attrs={},
                outputs=(GraphValue("out", wrapper_t),),
                source_module="Tensor.and",
                type_expr=wrapper_t,
            ),
        ),
        return_type_expr=wrapper_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(wrapper,), main_module="Tensor.and", pragmas={"main": "Tensor.and"}),
        config=GraphOptimizeConfig(inline_safe=False, specialize_definitions="off"),
    )

    optimized_wrapper = optimized.modules[0]
    assert optimized_wrapper.nodes[0].outputs[0].type_expr == wrapper_t
    assert optimized_wrapper.outputs[0].type_expr == wrapper_t
    assert optimized_wrapper.return_type_expr == wrapper_t


def test_graph_ir_optimizer_does_not_inline_row_polymorphic_forwarder() -> None:
    broad_t = TypeTensor("Tensor", ("B", 1, "Q", "K"))
    narrow_t = TypeTensor("Tensor", ("B", 1, 1, "K"))
    wrapper_t = TypeTensor("Tensor", ("..R",))
    wrapper = GraphModule(
        name="Tensor.and",
        inputs=(
            GraphValue("x", TypeTensor("Tensor", ("..S",))),
            GraphValue("y", TypeTensor("Tensor", ("..T",))),
        ),
        outputs=(GraphValueRef("out", wrapper_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="Tensor.and:1",
                op=GraphOp("_and"),
                inputs=(
                    GraphValueRef("x", TypeTensor("Tensor", ("..S",))),
                    GraphValueRef("y", TypeTensor("Tensor", ("..T",))),
                ),
                attrs={},
                outputs=(GraphValue("out", wrapper_t),),
                source_module="Tensor.and",
                type_expr=wrapper_t,
            ),
        ),
        return_type_expr=wrapper_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("keep", broad_t), GraphValue("pad", narrow_t)),
        outputs=(GraphValueRef("out", broad_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("Tensor.and"),
                inputs=(GraphValueRef("keep", broad_t), GraphValueRef("pad", narrow_t)),
                attrs={},
                outputs=(GraphValue("out", broad_t),),
                source_module="main",
                type_expr=broad_t,
            ),
        ),
        return_type_expr=broad_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(wrapper, main), main_module="main", pragmas={"main": "main"}),
        config=GraphOptimizeConfig(specialize_definitions="off", constant_folding=False),
    )

    optimized_main = {module.name: module for module in optimized.modules}["main"]
    assert optimized_main.nodes[0].op.name == "Tensor.and"
    assert optimized_main.nodes[0].outputs[0].type_expr == broad_t


def test_axon_purity_treats_fresh_zero_arg_name_refs_as_total_pure() -> None:
    program = parse_axon_program(
        """
List.init :: List[_T]
List.init = _list_init

Cache.init :: List[_T]
Cache.init = List.init
"""
    )

    effects = infer_axon_definition_effects(program.modules)

    assert effects["List.init"] == GraphEffect.TOTAL_PURE
    assert effects["Cache.init"] == GraphEffect.TOTAL_PURE


def test_render_axon_file_can_annotate_definition_purity() -> None:
    program = parse_axon_program(
        """
pure_total :: Int
pure_total = 1 + (_config_dim @@n_positions default=1024)

pure_partial :: Tensor[..S]
pure_partial = _params_param @@wte.weight
"""
    )

    text = render_axon_file(program, show_purity=True)

    assert "-- purity: total_pure\npure_total :: Int" in text
    assert "-- purity: partial_pure\npure_partial :: Tensor[..S]" in text


def test_callsite_default_makes_config_wrapper_total_pure() -> None:
    program = parse_axon_program(
        """
Config.dim :: Path -> ?Dim -> Dim
Config.dim key ?default=null = _config_dim key default

NUM_LAYERS :: Dim
NUM_LAYERS = Config.dim @@n_layer default=12
"""
    )

    text = render_axon_file(program, show_purity=True)

    assert "-- purity: partial_pure\nConfig.dim :: Path -> ?Dim -> Dim" in text
    assert "-- purity: total_pure\nNUM_LAYERS :: Dim" in text


def test_graph_ir_optimizer_drops_dead_select_branch_without_dropping_partial_effect() -> None:
    int_t = TypeInt()
    bool_t = TypeBool()
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(),
                outputs=(GraphLiteral(0, int_t),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="n1",
                        op=GraphOp("core.select"),
                        inputs=(
                            GraphLiteral(True, bool_t),
                            GraphExpr(
                                op=GraphOp("Params.param"),
                                inputs=(GraphPath(True, ("missing",)),),
                                attrs={},
                                type_expr=int_t,
                            ),
                            GraphLiteral(1, int_t),
                        ),
                        attrs={},
                        outputs=(GraphValue("unused", int_t),),
                        source_module="main",
                        type_expr=int_t,
                    ),
                ),
                return_type_expr=int_t,
            ),
        ),
        main_module="main",
        pragmas={},
    )

    optimized = optimize_graph_program(graph)

    assert [node.op.name for node in optimized.modules[0].nodes] == ["Params.param"]


def test_graph_domain_analysis_infers_all_reachable_calls_pass_null() -> None:
    tensor_t = TypeTensor("Tensor", ("B", "S"))
    opt_tensor_t = TypeOptional(tensor_t)
    int_t = TypeInt()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", int_t), GraphValue("mask", opt_tensor_t)),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("y", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="n1",
                op=GraphOp("helper"),
                inputs=(GraphValueRef("x", int_t), GraphLiteral(None, TypeNull())),
                attrs={},
                outputs=(GraphValue("y", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    unused = GraphModule(
        name="unused",
        inputs=(),
        outputs=(GraphLiteral(True, TypeBool()),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="u1",
                op=GraphOp("helper"),
                inputs=(GraphLiteral(1, int_t), GraphLiteral(False, TypeBool())),
                attrs={},
                outputs=(GraphValue("u", int_t),),
                source_module="unused",
                type_expr=int_t,
            ),
        ),
        return_type_expr=TypeBool(),
    )
    graph = GraphProgram(modules=(helper, main, unused), main_module="main", pragmas={})

    analysis = infer_main_module_domain_facts(graph)

    assert analysis.module_input_facts["helper"]["mask"].kind == GraphDomainKind.NULL
    assert "unused" not in analysis.module_input_facts


def test_graph_domain_analysis_propagates_facts_through_local_and_interprocedural_refs() -> None:
    int_t = TypeInt()
    null_t = TypeNull()
    maybe_int_t = TypeOptional(int_t)
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("value", maybe_int_t, optional=True),),
        outputs=(GraphValueRef("value", maybe_int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=maybe_int_t,
    )
    mid = GraphModule(
        name="mid",
        inputs=(GraphValue("mid_value", maybe_int_t, optional=True),),
        outputs=(GraphValueRef("out", maybe_int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="mid:1",
                op=GraphOp("helper"),
                inputs=(GraphValueRef("mid_value", maybe_int_t),),
                attrs={},
                outputs=(GraphValue("out", maybe_int_t),),
                source_module="mid",
                type_expr=maybe_int_t,
            ),
        ),
        return_type_expr=maybe_int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("out", maybe_int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.alias"),
                inputs=(GraphLiteral(None, null_t),),
                attrs={},
                outputs=(GraphValue("tmp", maybe_int_t, optional=True),),
                source_module="main",
                type_expr=maybe_int_t,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("mid"),
                inputs=(GraphValueRef("tmp", maybe_int_t),),
                attrs={},
                outputs=(GraphValue("out", maybe_int_t, optional=True),),
                source_module="main",
                type_expr=maybe_int_t,
            ),
        ),
        return_type_expr=maybe_int_t,
    )
    graph = GraphProgram(modules=(helper, mid, main), main_module="main", pragmas={})

    analysis = infer_main_module_domain_facts(graph)

    assert analysis.module_input_facts["mid"]["mid_value"].kind == GraphDomainKind.NULL
    assert analysis.module_input_facts["helper"]["value"].kind == GraphDomainKind.NULL


def test_graph_domain_analysis_disagreed_calls_are_unknown() -> None:
    bool_t = TypeBool()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("flag", bool_t),),
        outputs=(GraphValueRef("flag", bool_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=bool_t,
    )
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("b", bool_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="n1",
                op=GraphOp("helper"),
                inputs=(GraphLiteral(True, bool_t),),
                attrs={},
                outputs=(GraphValue("a", bool_t),),
                source_module="main",
                type_expr=bool_t,
            ),
            GraphNode(
                id="n2",
                op=GraphOp("helper"),
                inputs=(GraphLiteral(False, bool_t),),
                attrs={},
                outputs=(GraphValue("b", bool_t),),
                source_module="main",
                type_expr=bool_t,
            ),
        ),
        return_type_expr=bool_t,
    )
    graph = GraphProgram(modules=(helper, main), main_module="main", pragmas={})

    analysis = infer_main_module_domain_facts(graph)

    assert analysis.module_input_facts["helper"]["flag"].kind == GraphDomainKind.NOT_NULL


def test_graph_domain_analysis_includes_repeat_body_calls_for_shared_facts() -> None:
    bool_t = TypeBool()
    int_t = TypeInt()
    string_t = TypeString()
    where = GraphModule(
        name="Tensor.where",
        inputs=(
            GraphValue("cond", bool_t),
            GraphValue("x", int_t),
            GraphValue("y", int_t),
        ),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="where:1",
                op=GraphOp("_where"),
                inputs=(
                    GraphValueRef("cond", bool_t),
                    GraphValueRef("x", int_t),
                    GraphValueRef("y", int_t),
                ),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="Tensor.where",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    loop_step = GraphModule(
        name="loop_step",
        inputs=(GraphValue("i", int_t), GraphValue("value", int_t)),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("value",),
        nodes=(
            GraphNode(
                id="loop_step:1",
                op=GraphOp("Tensor.where"),
                inputs=(
                    GraphLiteral(True, bool_t),
                    GraphValueRef("value", int_t),
                    GraphLiteral(0, int_t),
                ),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="loop_step",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("value", int_t),),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("Tensor.where"),
                inputs=(
                    GraphLiteral(True, bool_t),
                    GraphLiteral(1, int_t),
                    GraphLiteral(2, int_t),
                ),
                attrs={},
                outputs=(GraphValue("direct", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("core.repeat"),
                inputs=(
                    GraphLiteral(0, int_t),
                    GraphLiteral(2, int_t),
                    GraphLiteral(1, int_t),
                    GraphValueRef("value", int_t),
                ),
                attrs={
                    "callee": GraphLiteral("loop_step", string_t),
                    "var": GraphLiteral("i", string_t),
                    "arg_count": GraphLiteral(2, int_t),
                    "carry_count": GraphLiteral(1, int_t),
                    "carry_0": GraphLiteral("out", string_t),
                    "arg_0": GraphLiteral("iter", string_t),
                    "arg_1": GraphLiteral("carry:0", string_t),
                },
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    graph = GraphProgram(modules=(where, loop_step, main), main_module="main", pragmas={})

    analysis = infer_main_module_domain_facts(graph)

    assert "loop_step" in analysis.module_input_facts
    assert analysis.module_input_facts["Tensor.where"]["x"].kind == GraphDomainKind.UNKNOWN


def test_graph_domain_analysis_propagates_local_null_comparison() -> None:
    bool_t = TypeBool()
    null_t = TypeNull()
    module = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("is_null", bool_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="n1",
                op=GraphOp("core.alias"),
                inputs=(GraphLiteral(None, null_t),),
                attrs={},
                outputs=(GraphValue("x", null_t),),
                source_module="main",
                type_expr=null_t,
            ),
            GraphNode(
                id="n2",
                op=GraphOp("core.binary.=="),
                inputs=(GraphValueRef("x", null_t), GraphLiteral(None, null_t)),
                attrs={},
                outputs=(GraphValue("is_null", bool_t),),
                source_module="main",
                type_expr=bool_t,
            ),
        ),
        return_type_expr=bool_t,
    )
    graph = GraphProgram(modules=(module,), main_module="main", pragmas={})

    analysis = infer_main_module_domain_facts(graph)

    fact = analysis.module_local_facts["main"]["is_null"]
    assert fact.kind == GraphDomainKind.LITERAL
    assert fact.value is True


def test_graph_domain_analysis_refines_named_null_guard_select_branches() -> None:
    int_t = TypeInt()
    bool_t = TypeBool()
    null_t = TypeNull()
    maybe_int_t = TypeOptional(int_t)
    module = GraphModule(
        name="main",
        inputs=(GraphValue("maybe", maybe_int_t, optional=True),),
        outputs=(GraphValueRef("out", maybe_int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.binary.=="),
                inputs=(GraphValueRef("maybe", maybe_int_t), GraphLiteral(None, null_t)),
                attrs={},
                outputs=(GraphValue("is_null", bool_t),),
                source_module="main",
                type_expr=bool_t,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("core.select"),
                inputs=(
                    GraphValueRef("is_null", bool_t),
                    GraphLiteral(0, int_t),
                    GraphValueRef("maybe", maybe_int_t),
                ),
                attrs={},
                outputs=(GraphValue("out", maybe_int_t),),
                source_module="main",
                type_expr=maybe_int_t,
            ),
        ),
        return_type_expr=maybe_int_t,
    )
    graph = GraphProgram(modules=(module,), main_module="main", pragmas={})

    analysis = infer_main_module_domain_facts(graph)

    assert analysis.module_local_facts["main"]["out"].kind == GraphDomainKind.NOT_NULL
    assert analysis.module_output_facts["main"][0].kind == GraphDomainKind.NOT_NULL


def test_graph_domain_analysis_joins_numeric_literals_to_interval() -> None:
    int_t = TypeInt()
    bool_t = TypeBool()
    module = GraphModule(
        name="main",
        inputs=(GraphValue("flag", bool_t),),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.select"),
                inputs=(
                    GraphValueRef("flag", bool_t),
                    GraphLiteral(1, int_t),
                    GraphLiteral(4, int_t),
                ),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    graph = GraphProgram(modules=(module,), main_module="main", pragmas={})

    analysis = infer_main_module_domain_facts(graph)

    fact = analysis.module_local_facts["main"]["out"]
    assert fact.kind == GraphDomainKind.INTERVAL
    assert fact.value == GraphDomainInterval(lower=1, upper=4)
    rendered = render_axon_file(
        graph_program_to_axon_file(graph),
        definition_comments=graph_domain_definition_comments(graph),
    )
    assert "-- domain: outputs out0=[1,4]" in rendered


def test_graph_domain_analysis_literal_equality_branch_refinement_can_form_interval() -> None:
    int_t = TypeInt()
    bool_t = TypeBool()
    module = GraphModule(
        name="main",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.binary.=="),
                inputs=(GraphValueRef("x", int_t), GraphLiteral(3, int_t)),
                attrs={},
                outputs=(GraphValue("is_three", bool_t),),
                source_module="main",
                type_expr=bool_t,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("core.select"),
                inputs=(
                    GraphValueRef("is_three", bool_t),
                    GraphValueRef("x", int_t),
                    GraphLiteral(0, int_t),
                ),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    graph = GraphProgram(modules=(module,), main_module="main", pragmas={})

    analysis = infer_main_module_domain_facts(graph)

    fact = analysis.module_local_facts["main"]["out"]
    assert fact.kind == GraphDomainKind.INTERVAL
    assert fact.value == GraphDomainInterval(lower=0, upper=3)


def test_graph_domain_comments_render_as_definition_comments() -> None:
    int_t = TypeInt()
    bool_t = TypeBool()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("flag", bool_t),),
        outputs=(GraphValueRef("flag", bool_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=bool_t,
    )
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("out", bool_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="n1",
                op=GraphOp("helper"),
                inputs=(GraphLiteral(True, bool_t),),
                attrs={},
                outputs=(GraphValue("out", bool_t),),
                source_module="main",
                type_expr=bool_t,
            ),
        ),
        return_type_expr=bool_t,
    )
    graph = GraphProgram(modules=(helper, main), main_module="main", pragmas={})
    axon = graph_program_to_axon_file(graph)

    rendered = render_axon_file(
        axon,
        definition_comments=graph_domain_definition_comments(graph),
    )

    assert "-- domain: inputs flag=True\n-- domain: outputs out0=True\nhelper :: Bool -> Bool" in rendered


def test_graph_domain_analysis_validation_rejects_invalid_interval() -> None:
    int_t = TypeInt()
    module = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.alias"),
                inputs=(GraphLiteral(1, int_t),),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    graph = GraphProgram(modules=(module,), main_module="main", pragmas={})
    analysis = GraphDomainAnalysis(
        module_input_facts={"main": {}},
        module_local_facts={
            "main": {
                "out": GraphDomainFact(
                    GraphDomainKind.INTERVAL,
                    GraphDomainInterval(lower=4, upper=1),
                )
            }
        },
        module_output_facts={"main": (GraphDomainFact(GraphDomainKind.UNKNOWN),)},
    )

    with pytest.raises(ValueError, match="lower bound greater than upper bound"):
        validate_graph_domain_analysis(graph, analysis)


def test_graph_domain_analysis_validation_rejects_wrong_output_arity() -> None:
    int_t = TypeInt()
    module = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphLiteral(1, int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=int_t,
    )
    graph = GraphProgram(modules=(module,), main_module="main", pragmas={})
    analysis = GraphDomainAnalysis(
        module_input_facts={"main": {}},
        module_local_facts={"main": {}},
        module_output_facts={"main": ()},
    )

    with pytest.raises(ValueError, match="0 output facts; expected 1"):
        validate_graph_domain_analysis(graph, analysis)


def test_graph_domain_analysis_validation_accepts_value_level_optional_null() -> None:
    tensor_t = TypeTensor("Tensor", ("B", "S"))
    module = GraphModule(
        name="main",
        inputs=(GraphValue("rel_bias", tensor_t, optional=True),),
        outputs=(GraphLiteral(0, TypeInt()),),
        output_names=("out",),
        nodes=(),
        return_type_expr=TypeInt(),
    )
    graph = GraphProgram(modules=(module,), main_module="main", pragmas={})
    analysis = GraphDomainAnalysis(
        module_input_facts={"main": {"rel_bias": GraphDomainFact(GraphDomainKind.NULL)}},
        module_local_facts={"main": {"rel_bias": GraphDomainFact(GraphDomainKind.NULL)}},
        module_output_facts={"main": (GraphDomainFact(GraphDomainKind.LITERAL, 0),)},
    )

    validate_graph_domain_analysis(graph, analysis)


def test_graph_domain_analysis_validation_rejects_obvious_type_mismatch() -> None:
    bool_t = TypeBool()
    module = GraphModule(
        name="main",
        inputs=(GraphValue("flag", bool_t),),
        outputs=(GraphValueRef("flag", bool_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=bool_t,
    )
    graph = GraphProgram(modules=(module,), main_module="main", pragmas={})
    analysis = GraphDomainAnalysis(
        module_input_facts={"main": {"flag": GraphDomainFact(GraphDomainKind.LITERAL, 1)}},
        module_local_facts={"main": {"flag": GraphDomainFact(GraphDomainKind.LITERAL, 1)}},
        module_output_facts={"main": (GraphDomainFact(GraphDomainKind.LITERAL, 1),)},
    )

    with pytest.raises(ValueError, match="incompatible with type"):
        validate_graph_domain_analysis(graph, analysis)


def test_graph_optimizer_refines_optional_refs_in_non_null_select_branch() -> None:
    int_t = TypeInt()
    null_t = TypeNull()
    bool_t = TypeBool()
    maybe_int_t = TypeOptional(int_t)
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("maybe", int_t, optional=True),),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.binary.=="),
                inputs=(GraphValueRef("maybe", maybe_int_t), GraphLiteral(None, null_t)),
                attrs={},
                outputs=(GraphValue("is_null", bool_t),),
                source_module="main",
                type_expr=bool_t,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("core.select"),
                inputs=(
                    GraphValueRef("is_null", bool_t),
                    GraphLiteral(0, int_t),
                    GraphExpr(
                        op=GraphOp("helper"),
                        inputs=(GraphValueRef("maybe", maybe_int_t),),
                        attrs={},
                        type_expr=int_t,
                    ),
                ),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    graph = GraphProgram(modules=(helper, main), main_module="main", pragmas={})

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(
            inline_safe=False,
            specialize_definitions="off",
            constant_dim_substitution=False,
            common_subexpression_elimination=False,
            dead_temp_elimination=False,
        ),
    )

    main_module = next(module for module in optimized.modules if module.name == "main")
    select = main_module.nodes[0]
    false_call = select.inputs[2]
    assert isinstance(false_call, GraphExpr)
    assert isinstance(false_call.inputs[0], GraphValueRef)
    assert false_call.inputs[0].type_expr == int_t


def test_graph_optimizer_inlines_forwarder_in_non_null_select_branch() -> None:
    int_t = TypeInt()
    null_t = TypeNull()
    bool_t = TypeBool()
    list_int_t = TypeList(int_t)
    maybe_list_int_t = TypeOptional(list_int_t)
    helper = GraphModule(
        name="append_value",
        inputs=(GraphValue("values", list_int_t), GraphValue("value", int_t)),
        outputs=(
            GraphExpr(
                op=GraphOp("_list_append"),
                inputs=(GraphValueRef("values", list_int_t), GraphValueRef("value", int_t)),
                attrs={},
                type_expr=list_int_t,
            ),
        ),
        output_names=("out",),
        nodes=(),
        return_type_expr=list_int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("maybe_values", list_int_t, optional=True), GraphValue("value", int_t)),
        outputs=(GraphValueRef("out", maybe_list_int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.binary.=="),
                inputs=(GraphValueRef("maybe_values", maybe_list_int_t), GraphLiteral(None, null_t)),
                attrs={},
                outputs=(GraphValue("is_null", bool_t),),
                source_module="main",
                type_expr=bool_t,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("core.select"),
                inputs=(
                    GraphValueRef("is_null", bool_t),
                    GraphLiteral(None, null_t),
                    GraphExpr(
                        op=GraphOp("append_value"),
                        inputs=(GraphValueRef("maybe_values", maybe_list_int_t), GraphValueRef("value", int_t)),
                        attrs={},
                        type_expr=list_int_t,
                    ),
                ),
                attrs={},
                outputs=(GraphValue("out", maybe_list_int_t),),
                source_module="main",
                type_expr=maybe_list_int_t,
            ),
        ),
        return_type_expr=maybe_list_int_t,
    )
    graph = GraphProgram(modules=(helper, main), main_module="main", pragmas={})

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(
            specialize_definitions="off",
            constant_dim_substitution=False,
            common_subexpression_elimination=False,
            dead_temp_elimination=False,
        ),
    )

    main_module = next(module for module in optimized.modules if module.name == "main")
    select = main_module.nodes[0]
    false_branch = select.inputs[2]
    assert isinstance(false_branch, GraphExpr)
    assert false_branch.op.name == "_list_append"
    values_operand = false_branch.inputs[0]
    assert isinstance(values_operand, GraphValueRef)
    assert values_operand.type_expr == list_int_t


def test_graph_domain_does_not_treat_non_optional_expression_type_as_non_null_fact() -> None:
    int_t = TypeInt()
    null_t = TypeNull()
    bool_t = TypeBool()
    list_int_t = TypeList(int_t)
    maybe_list_int_t = TypeOptional(list_int_t)
    module = GraphModule(
        name="main",
        inputs=(GraphValue("values", list_int_t), GraphValue("value", int_t)),
        outputs=(GraphValueRef("is_null", bool_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.binary.=="),
                inputs=(
                    GraphExpr(
                        op=GraphOp("_list_append"),
                        inputs=(GraphValueRef("values", list_int_t), GraphValueRef("value", int_t)),
                        attrs={},
                        type_expr=list_int_t,
                    ),
                    GraphLiteral(None, null_t),
                ),
                attrs={},
                outputs=(GraphValue("is_null", bool_t),),
                source_module="main",
                type_expr=bool_t,
            ),
        ),
        return_type_expr=bool_t,
    )
    graph = GraphProgram(modules=(module,), main_module="main", pragmas={})

    analysis = infer_main_module_domain_facts(graph)

    fact = analysis.module_local_facts["main"]["is_null"]
    assert fact.kind == GraphDomainKind.UNKNOWN


def test_graph_dim_simplifies_add_then_subtract_same_symbol() -> None:
    dim = DimExprBinary(
        op="-",
        left=DimExprBinary(op="+", left="past_length", right="S"),
        right="past_length",
    )

    assert substitute_dim_token(dim, {}) == "S"


def test_graph_dim_simplifies_symbol_minus_symbol_minus_dim() -> None:
    dim = DimExprBinary(
        op="-",
        left="K",
        right=DimExprBinary(op="-", left="K", right="S"),
    )

    assert substitute_dim_token(dim, {}) == "S"


def test_graph_dim_substitution_is_hygienic_for_same_named_caller_dims() -> None:
    dim_t = TypeDim()
    operand = GraphValueRef("S", dim_t)
    rewritten = substitute_graph_operand_dims(
        operand,
        {"S": DimExprBinary(op="+", left="S", right="S")},
    )

    assert isinstance(rewritten, GraphExpr)
    assert rewritten.op == GraphOp("core.binary.+")
    assert rewritten.inputs == (GraphValueRef("S", dim_t), GraphValueRef("S", dim_t))


def test_graph_domain_analysis_validation_accepts_literal_for_type_var() -> None:
    var_t = TypeVar("_T")
    module = GraphModule(
        name="main",
        inputs=(GraphValue("x", var_t),),
        outputs=(GraphValueRef("x", var_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=var_t,
    )
    graph = GraphProgram(modules=(module,), main_module="main", pragmas={})
    analysis = GraphDomainAnalysis(
        module_input_facts={"main": {"x": GraphDomainFact(GraphDomainKind.LITERAL, 0.0)}},
        module_local_facts={"main": {"x": GraphDomainFact(GraphDomainKind.LITERAL, 0.0)}},
        module_output_facts={"main": (GraphDomainFact(GraphDomainKind.LITERAL, 0.0),)},
    )

    validate_graph_domain_analysis(graph, analysis)


def test_graph_ir_optimizer_specializes_single_callsite_literal_argument() -> None:
    int_t = TypeInt()
    path_t = TypePath()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("p", path_t), GraphValue("x", int_t)),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("y", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphPath(True, ("wte",)), GraphValueRef("x", int_t)),
                attrs={},
                outputs=(GraphValue("y", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    graph = GraphProgram(modules=(helper, main), main_module="main", pragmas={})

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(inline_safe=False),
    )

    module_names = {module.name for module in optimized.modules}
    assert "helper" in module_names
    specialized = next(module for module in optimized.modules if module.name == "helper")
    assert [value.name for value in specialized.inputs] == ["x"]
    main_node = next(module for module in optimized.modules if module.name == "main").nodes[0]
    assert main_node.op.name == "helper"
    assert len(main_node.inputs) == 1


def test_graph_ir_optimizer_keeps_compact_suffix_when_original_module_remains() -> None:
    int_t = TypeInt()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("scale", int_t), GraphValue("x", int_t)),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("scale", int_t), GraphValue("x", int_t)),
        outputs=(GraphValueRef("a", int_t), GraphValueRef("b", int_t)),
        output_names=("a", "b"),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphLiteral(2, int_t), GraphValueRef("x", int_t)),
                attrs={},
                outputs=(GraphValue("a", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("helper"),
                inputs=(GraphValueRef("scale", int_t), GraphValueRef("x", int_t)),
                attrs={},
                outputs=(GraphValue("b", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=TypeTuple((int_t, int_t)),
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="monomorphize", inline_safe=False),
    )

    assert {module.name for module in optimized.modules} == {"helper", "main"}
    optimized_helper = next(module for module in optimized.modules if module.name == "helper")
    assert [value.name for value in optimized_helper.inputs] == ["x"]
    optimized_main = next(module for module in optimized.modules if module.name == "main")
    helper_calls = [node for node in optimized_main.nodes if node.op.name == "helper"]
    assert helper_calls
    assert all(len(node.inputs) == 1 for node in helper_calls)


def test_graph_ir_optimizer_specialization_substitutes_dim_metadata() -> None:
    dim_t = TypeDim()
    tensor_formal = TypeTensor("Tensor", ("B", "S", "d"))
    tensor_actual = TypeTensor("Tensor", ("B", "S", 8))
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("d", dim_t), GraphValue("x", tensor_formal)),
        outputs=(GraphValueRef("x", tensor_formal),),
        output_names=("out",),
        nodes=(),
        return_type_expr=tensor_formal,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", tensor_actual),),
        outputs=(GraphValueRef("out", tensor_actual),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphLiteral(8, dim_t), GraphValueRef("x", tensor_actual)),
                attrs={},
                outputs=(GraphValue("out", tensor_actual),),
                source_module="main",
                type_expr=tensor_actual,
            ),
        ),
        return_type_expr=tensor_actual,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False),
    )

    specialized = next(module for module in optimized.modules if module.name == "helper")
    assert specialized.inputs == (GraphValue("x", tensor_actual),)
    assert specialized.outputs == (GraphValueRef("x", tensor_actual),)
    assert specialized.return_type_expr == tensor_actual


def test_graph_ir_optimizer_specialization_substitutes_and_prunes_true_constraints() -> None:
    dim_t = TypeDim()
    tensor_formal = TypeTensor("Tensor", ("B", "d"))
    tensor_actual = TypeTensor("Tensor", ("B", 8))
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("d", dim_t), GraphValue("x", tensor_formal)),
        outputs=(GraphValueRef("x", tensor_formal),),
        output_names=("out",),
        nodes=(),
        return_type_expr=tensor_formal,
        constraints=(Constraint("=", "d", 8),),
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", tensor_actual),),
        outputs=(GraphValueRef("out", tensor_actual),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphLiteral(8, dim_t), GraphValueRef("x", tensor_actual)),
                attrs={},
                outputs=(GraphValue("out", tensor_actual),),
                source_module="main",
                type_expr=tensor_actual,
            ),
        ),
        return_type_expr=tensor_actual,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False),
    )

    specialized = next(module for module in optimized.modules if module.name == "helper")
    assert [value.name for value in specialized.inputs] == ["x"]
    assert specialized.constraints == ()


def test_graph_ir_optimizer_specialization_keeps_remaining_substituted_constraints() -> None:
    dim_t = TypeDim()
    tensor_formal = TypeTensor("Tensor", ("B", "S", "d"))
    tensor_actual = TypeTensor("Tensor", ("B", "S", 8))
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("d", dim_t), GraphValue("x", tensor_formal)),
        outputs=(GraphValueRef("x", tensor_formal),),
        output_names=("out",),
        nodes=(),
        return_type_expr=tensor_formal,
        constraints=(Constraint("=", "d", "S"),),
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", tensor_actual),),
        outputs=(GraphValueRef("out", tensor_actual),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphLiteral(8, dim_t), GraphValueRef("x", tensor_actual)),
                attrs={},
                outputs=(GraphValue("out", tensor_actual),),
                source_module="main",
                type_expr=tensor_actual,
            ),
        ),
        return_type_expr=tensor_actual,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False),
    )

    specialized = next(module for module in optimized.modules if module.name == "helper")
    assert specialized.constraints == (Constraint("=", 8, "S"),)


def test_graph_ir_optimizer_specialization_skips_false_constraints() -> None:
    dim_t = TypeDim()
    tensor_formal = TypeTensor("Tensor", ("B", "d"))
    tensor_actual = TypeTensor("Tensor", ("B", 9))
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("d", dim_t), GraphValue("x", tensor_formal)),
        outputs=(GraphValueRef("x", tensor_formal),),
        output_names=("out",),
        nodes=(),
        return_type_expr=tensor_formal,
        constraints=(Constraint("=", "d", 8),),
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", tensor_actual),),
        outputs=(GraphValueRef("out", tensor_actual),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphLiteral(9, dim_t), GraphValueRef("x", tensor_actual)),
                attrs={},
                outputs=(GraphValue("out", tensor_actual),),
                source_module="main",
                type_expr=tensor_actual,
            ),
        ),
        return_type_expr=tensor_actual,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False),
    )

    assert {module.name for module in optimized.modules} == {"helper", "main"}
    assert next(module for module in optimized.modules if module.name == "main").nodes[0].op.name == "helper"


def test_graph_ir_optimizer_specialization_skips_unrepresentable_path_constraint_substitution() -> None:
    path_t = TypePath()
    int_t = TypeInt()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("p", path_t), GraphValue("x", int_t)),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=int_t,
        constraints=(Constraint("not_null", "p"),),
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphPath(True, ("wte",)), GraphValueRef("x", int_t)),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False),
    )

    assert {module.name for module in optimized.modules} == {"helper", "main"}


def test_graph_ir_optimizer_specializes_single_callsite_null_argument() -> None:
    int_t = TypeInt()
    null_t = TypeNull()
    maybe_int_t = TypeOptional(int_t)
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", int_t), GraphValue("optional", maybe_int_t, optional=True)),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("value", int_t),),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphValueRef("value", int_t), GraphLiteral(None, null_t)),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False),
    )

    optimized_main = next(module for module in optimized.modules if module.name == "main")
    assert len(optimized_main.nodes[0].inputs) == 1
    spec = next(module for module in optimized.modules if module.name == "helper")
    assert [value.name for value in spec.inputs] == ["x"]


def test_graph_ir_optimizer_uses_domain_null_fact_to_fold_callee_branch() -> None:
    bool_t = TypeBool()
    int_t = TypeInt()
    null_t = TypeNull()
    maybe_int_t = TypeOptional(int_t)
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", int_t), GraphValue("maybe", maybe_int_t, optional=True)),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="helper:1",
                op=GraphOp("core.binary.=="),
                inputs=(GraphValueRef("maybe", maybe_int_t), GraphLiteral(None, null_t)),
                attrs={},
                outputs=(GraphValue("is_null", bool_t),),
                source_module="helper",
                type_expr=bool_t,
            ),
            GraphNode(
                id="helper:2",
                op=GraphOp("core.select"),
                inputs=(
                    GraphValueRef("is_null", bool_t),
                    GraphValueRef("x", int_t),
                    GraphExpr(
                        op=GraphOp("core.binary.+"),
                        inputs=(GraphValueRef("x", int_t), GraphValueRef("maybe", maybe_int_t)),
                        attrs={},
                        type_expr=int_t,
                    ),
                ),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="helper",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("value", int_t),),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphValueRef("value", int_t), GraphLiteral(None, null_t)),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False, specialize_definitions="off"),
    )

    optimized_helper = next(module for module in optimized.modules if module.name == "helper")
    assert optimized_helper.nodes == ()
    assert optimized_helper.outputs == (GraphValueRef("x", int_t),)


def test_graph_ir_optimizer_folds_float_comparisons_and_bool_ops_to_bool() -> None:
    any_t = TypeAny()
    bool_t = TypeBool()
    float_t = TypeFloat()
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("out", bool_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.binary.=="),
                inputs=(GraphLiteral(0.0, float_t), GraphLiteral(0.0, float_t)),
                attrs={},
                outputs=(GraphValue("cmp", bool_t),),
                source_module="main",
                type_expr=bool_t,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("core.binary.or"),
                inputs=(GraphLiteral(True, bool_t), GraphValueRef("cmp", bool_t)),
                attrs={},
                outputs=(GraphValue("out", any_t),),
                source_module="main",
                type_expr=any_t,
            ),
        ),
        return_type_expr=bool_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(main,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False, specialize_definitions="off"),
    )

    optimized_main = optimized.modules[0]
    assert optimized_main.nodes == ()
    assert optimized_main.outputs == (GraphLiteral(True, bool_t),)


def test_graph_ir_optimizer_folds_total_pure_numeric_primitive_literals() -> None:
    dim_t = TypeDim()
    float_t = TypeFloat()
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("scale", float_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("_sqrt"),
                inputs=(GraphLiteral(64, dim_t),),
                attrs={},
                outputs=(GraphValue("root", float_t),),
                source_module="main",
                type_expr=float_t,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("core.binary./"),
                inputs=(GraphLiteral(1.0, float_t), GraphValueRef("root", float_t)),
                attrs={},
                outputs=(GraphValue("scale", float_t),),
                source_module="main",
                type_expr=float_t,
            ),
        ),
        return_type_expr=float_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(main,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False, specialize_definitions="off"),
    )

    optimized_main = optimized.modules[0]
    assert optimized_main.nodes == ()
    assert optimized_main.outputs == (GraphLiteral(0.125, float_t),)


def test_graph_ir_optimizer_inlines_chunk_forwarder_with_fresh_variadic_result() -> None:
    dim_t = TypeDim()
    tensor_t = TypeTensor("Tensor", ("..S",))
    item_t = TypeTensor("Tensor", ("B", "S", "D"))
    list_t = TypeList(TypeTensor("Tensor", ("..R",)))
    tuple_t = TypeTuple((item_t, item_t, item_t))
    chunk = GraphModule(
        name="Tensor.chunk",
        inputs=(GraphValue("x", tensor_t), GraphValue("dim", TypeInt()), GraphValue("parts", dim_t)),
        outputs=(GraphValueRef("__flat_1", list_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="Tensor.chunk:1",
                op=GraphOp("_chunk"),
                inputs=(GraphValueRef("x", tensor_t), GraphValueRef("dim", TypeInt()), GraphValueRef("parts", dim_t)),
                attrs={},
                outputs=(GraphValue("__flat_1", list_t),),
                source_module="Tensor.chunk",
                type_expr=list_t,
            ),
        ),
        return_type_expr=list_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", TypeTensor("Tensor", ("B", "S", "D"))),),
        outputs=(GraphValueRef("q", item_t), GraphValueRef("k", item_t), GraphValueRef("v", item_t)),
        output_names=("q", "k", "v"),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("Tensor.chunk"),
                inputs=(GraphValueRef("x", TypeTensor("Tensor", ("B", "S", "D"))), GraphLiteral(-1, TypeInt()), GraphLiteral(3, dim_t)),
                attrs={},
                outputs=(GraphValue("q", item_t), GraphValue("k", item_t), GraphValue("v", item_t)),
                source_module="main",
                type_expr=tuple_t,
            ),
        ),
        return_type_expr=tuple_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(chunk, main), main_module="main", pragmas={}),
    )

    assert {module.name for module in optimized.modules} == {"main"}
    optimized_main = next(module for module in optimized.modules if module.name == "main")
    assert optimized_main.nodes[0].op.name == "_chunk"
    assert len(optimized_main.nodes[0].outputs) == 3


def test_graph_ir_optimizer_refreshes_destructured_chunk_shapes() -> None:
    dim_t = TypeDim()
    input_t = TypeTensor("Tensor", ("B", "S", 2304))
    stale_item_t = TypeTensor("Tensor", ("B", "S", 2304))
    expected_item_t = TypeTensor("Tensor", ("B", "S", 768))
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", input_t),),
        outputs=(
            GraphValueRef("q", stale_item_t),
            GraphValueRef("k", stale_item_t),
            GraphValueRef("v", stale_item_t),
        ),
        output_names=("q", "k", "v"),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("_chunk"),
                inputs=(
                    GraphValueRef("x", input_t),
                    GraphLiteral(-1, TypeInt()),
                    GraphLiteral(3, dim_t),
                ),
                attrs={},
                outputs=(
                    GraphValue("q", stale_item_t),
                    GraphValue("k", stale_item_t),
                    GraphValue("v", stale_item_t),
                ),
                source_module="main",
                type_expr=TypeTuple((stale_item_t, stale_item_t, stale_item_t)),
            ),
        ),
        return_type_expr=TypeTuple((stale_item_t, stale_item_t, stale_item_t)),
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(main,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            inline_safe=False,
            specialize_definitions="off",
            constant_folding=False,
            constant_dim_substitution=False,
            common_subexpression_elimination=False,
            dead_temp_elimination=False,
            atomic_alias_cleanup=False,
        ),
    )

    optimized_main = optimized.modules[0]
    node = optimized_main.nodes[0]
    assert [output.type_expr for output in node.outputs] == [
        expected_item_t,
        expected_item_t,
        expected_item_t,
    ]
    assert optimized_main.return_type_expr == TypeTuple(
        (expected_item_t, expected_item_t, expected_item_t)
    )


def test_graph_ir_optimizer_refreshes_binary_output_from_operands() -> None:
    left_t = TypeTensor("Tensor", ("B", "S", 768))
    right_t = TypeTensor("Tensor", ("B", "S", 768))
    stale_t = TypeTensor("Tensor", ("B", "K", 768))
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", left_t), GraphValue("a", right_t)),
        outputs=(GraphValueRef("out", stale_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.binary.+"),
                inputs=(GraphValueRef("x", left_t), GraphValueRef("a", right_t)),
                attrs={},
                outputs=(GraphValue("out", stale_t),),
                source_module="main",
                type_expr=stale_t,
            ),
        ),
        return_type_expr=stale_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(main,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            inline_safe=False,
            specialize_definitions="off",
            constant_folding=False,
            constant_dim_substitution=False,
            common_subexpression_elimination=False,
            dead_temp_elimination=False,
            atomic_alias_cleanup=False,
        ),
    )

    optimized_main = optimized.modules[0]
    assert optimized_main.nodes[0].outputs[0].type_expr == left_t
    assert optimized_main.return_type_expr == left_t


def test_graph_ir_call_substitution_unifies_repeated_formal_dims() -> None:
    helper = GraphModule(
        name="helper",
        inputs=(
            GraphValue("past", TypeTensor("Tensor", ("B", "H", "P", "DH"))),
            GraphValue("k", TypeTensor("Tensor", ("B", "H", "S", "DH"))),
        ),
        outputs=(GraphValueRef("k", TypeTensor("Tensor", ("B", "H", "S", "DH"))),),
        output_names=("out",),
        nodes=(),
        return_type_expr=TypeTensor("Tensor", ("B", "H", "S", "DH")),
    )
    main = GraphModule(
        name="main",
        inputs=(
            GraphValue("past", TypeTensor("Tensor", ("B", "H", "P", "DH"))),
            GraphValue("k", TypeTensor("Tensor", ("B", 12, "S", 64))),
        ),
        outputs=(GraphValueRef("out", TypeTensor("Tensor", ("B", "H", "S", "DH"))),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(
                    GraphValueRef("past", TypeTensor("Tensor", ("B", "H", "P", "DH"))),
                    GraphValueRef("k", TypeTensor("Tensor", ("B", 12, "S", 64))),
                ),
                attrs={},
                outputs=(GraphValue("out", TypeTensor("Tensor", ("B", "H", "S", "DH"))),),
                source_module="main",
                type_expr=TypeTensor("Tensor", ("B", "H", "S", "DH")),
            ),
        ),
        return_type_expr=TypeTensor("Tensor", ("B", "H", "S", "DH")),
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            inline_safe=False,
            specialize_definitions="off",
            constant_folding=False,
            constant_dim_substitution=False,
            common_subexpression_elimination=False,
            dead_temp_elimination=False,
            atomic_alias_cleanup=False,
        ),
    )

    optimized_main = next(module for module in optimized.modules if module.name == "main")
    assert optimized_main.nodes[0].outputs[0].type_expr == TypeTensor("Tensor", ("B", 12, "S", 64))


def test_graph_ir_optimizer_keeps_partial_one_op_forwarders_inside_select_branches() -> None:
    bool_t = TypeBool()
    int_t = TypeInt()
    list_t = TypeList(int_t)
    index = GraphModule(
        name="List.index",
        inputs=(GraphValue("values", list_t), GraphValue("i", int_t)),
        outputs=(GraphValueRef("__flat_1", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="List.index:1",
                op=GraphOp("_list_index"),
                inputs=(GraphValueRef("values", list_t), GraphValueRef("i", int_t)),
                attrs={},
                outputs=(GraphValue("__flat_1", int_t),),
                source_module="List.index",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("cond", bool_t), GraphValue("values", list_t), GraphValue("i", int_t)),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.select"),
                inputs=(
                    GraphValueRef("cond", bool_t),
                    GraphLiteral(0, int_t),
                    GraphExpr(
                        op=GraphOp("List.index"),
                        inputs=(GraphValueRef("values", list_t), GraphValueRef("i", int_t)),
                        attrs={},
                        type_expr=int_t,
                    ),
                ),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(index, main), main_module="main", pragmas={}),
    )

    optimized_main = next(module for module in optimized.modules if module.name == "main")
    select = optimized_main.nodes[0].inputs[2]
    assert isinstance(select, GraphExpr)
    assert select.op.name == "_list_index"
    assert {module.name for module in optimized.modules} == {"main"}


def test_graph_ir_optimizer_folds_shape_index_from_tensor_type() -> None:
    dim_t = TypeDim()
    list_dim_t = TypeList(dim_t)
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", TypeTensor("Tensor", ("B", "S", 64))),),
        outputs=(GraphValueRef("d", dim_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("_shape"),
                inputs=(GraphValueRef("x", TypeTensor("Tensor", ("B", "S", 64))),),
                attrs={},
                outputs=(GraphValue("shape", list_dim_t),),
                source_module="main",
                type_expr=list_dim_t,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("_list_index"),
                inputs=(GraphValueRef("shape", list_dim_t), GraphLiteral(-1, TypeInt())),
                attrs={},
                outputs=(GraphValue("d", dim_t),),
                source_module="main",
                type_expr=dim_t,
            ),
        ),
        return_type_expr=dim_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(main,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False, specialize_definitions="off"),
    )

    optimized_main = optimized.modules[0]
    assert optimized_main.nodes == ()
    assert optimized_main.outputs == (GraphLiteral(64, dim_t),)


def test_graph_ir_optimizer_folds_generic_shape_index_helper_without_name_special_case() -> None:
    dim_t = TypeDim()
    int_t = TypeInt()
    list_dim_t = TypeList(dim_t)
    shape_dim = GraphModule(
        name="Shape.dim",
        inputs=(
            GraphValue("x", TypeTensor("Tensor", ("..S",))),
            GraphValue("dim", int_t),
        ),
        outputs=(GraphValueRef("__flat_2", dim_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="Shape.dim:1",
                op=GraphOp("_shape"),
                inputs=(GraphValueRef("x", TypeTensor("Tensor", ("..S",))),),
                attrs={},
                outputs=(GraphValue("__flat_1", list_dim_t),),
                source_module="Shape.dim",
                type_expr=list_dim_t,
            ),
            GraphNode(
                id="Shape.dim:2",
                op=GraphOp("_list_index"),
                inputs=(GraphValueRef("__flat_1", list_dim_t), GraphValueRef("dim", int_t)),
                attrs={},
                outputs=(GraphValue("__flat_2", dim_t),),
                source_module="Shape.dim",
                type_expr=dim_t,
            ),
        ),
        return_type_expr=dim_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", TypeTensor("Tensor", ("B", "K", 64))),),
        outputs=(GraphValueRef("K", dim_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("Shape.dim"),
                inputs=(GraphValueRef("x", TypeTensor("Tensor", ("B", "K", 64))), GraphLiteral(-2, int_t)),
                attrs={},
                outputs=(GraphValue("K", dim_t),),
                source_module="main",
                type_expr=dim_t,
            ),
        ),
        return_type_expr=dim_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(shape_dim, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False, specialize_definitions="off"),
    )

    optimized_main = next(module for module in optimized.modules if module.name == "main")
    assert optimized_main.nodes == ()
    assert optimized_main.outputs == (GraphValueRef("K", dim_t),)
    assert {module.name for module in optimized.modules} == {"main"}


def test_graph_ir_optimizer_drops_unbound_unguarded_constraint_refs() -> None:
    int_t = TypeInt()
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=int_t,
        constraints=(Constraint("=", "missing", 1),),
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(main,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            atomic_alias_cleanup=False,
            dead_temp_elimination=False,
            constant_folding=False,
            specialize_definitions="off",
            inline_safe=False,
        ),
    )

    assert optimized.modules[0].constraints == ()


def test_graph_ir_optimizer_allows_callsite_guarded_external_constraint_refs() -> None:
    int_t = TypeInt()
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=int_t,
        constraints=(
            Constraint(
                "=",
                "x",
                "caller_value",
                guards=(Constraint("callsite", "caller->main#1"),),
            ),
        ),
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(main,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            atomic_alias_cleanup=False,
            dead_temp_elimination=False,
            constant_folding=False,
            specialize_definitions="off",
            inline_safe=False,
        ),
    )

    assert optimized.modules[0].constraints == main.constraints


def test_graph_ir_optimizer_specializes_recursive_invariant_literal_argument() -> None:
    int_t = TypeInt()
    recur = GraphModule(
        name="recur",
        inputs=(GraphValue("step", int_t), GraphValue("x", int_t)),
        outputs=(GraphValueRef("y", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="recur:1",
                op=GraphOp("recur"),
                inputs=(GraphValueRef("step", int_t), GraphValueRef("x", int_t)),
                attrs={},
                outputs=(GraphValue("y", int_t),),
                source_module="recur",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("y", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("recur"),
                inputs=(GraphLiteral(1, int_t), GraphValueRef("x", int_t)),
                attrs={},
                outputs=(GraphValue("y", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    graph = GraphProgram(modules=(recur, main), main_module="main", pragmas={})

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(inline_safe=False),
    )

    assert {module.name for module in optimized.modules} == {"recur", "main"}
    specialized = next(module for module in optimized.modules if module.name == "recur")
    assert [value.name for value in specialized.inputs] == ["x"]
    assert specialized.nodes[0].op.name == "recur"
    assert next(module for module in optimized.modules if module.name == "main").nodes[0].op.name == "recur"


def test_graph_ir_optimizer_does_not_specialize_changing_recursive_argument() -> None:
    int_t = TypeInt()
    recur = GraphModule(
        name="recur",
        inputs=(GraphValue("i", int_t), GraphValue("limit", int_t), GraphValue("step", int_t)),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="recur:1",
                op=GraphOp("continue"),
                inputs=(
                    GraphValueRef("i", int_t),
                    GraphValueRef("limit", int_t),
                    GraphValueRef("step", int_t),
                ),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="recur",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    cont = GraphModule(
        name="continue",
        inputs=(GraphValue("i", int_t), GraphValue("limit", int_t), GraphValue("step", int_t)),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="continue:1",
                op=GraphOp("core.binary.+"),
                inputs=(GraphValueRef("i", int_t), GraphValueRef("step", int_t)),
                attrs={},
                outputs=(GraphValue("next_i", int_t),),
                source_module="continue",
                type_expr=int_t,
            ),
            GraphNode(
                id="continue:2",
                op=GraphOp("recur"),
                inputs=(
                    GraphValueRef("next_i", int_t),
                    GraphValueRef("limit", int_t),
                    GraphValueRef("step", int_t),
                ),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="continue",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("recur"),
                inputs=(GraphLiteral(0, int_t), GraphLiteral(4, int_t), GraphLiteral(1, int_t)),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(recur, cont, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False),
    )

    spec_recur = next(module for module in optimized.modules if module.name == "recur")
    spec_continue = next(module for module in optimized.modules if module.name == "continue")
    assert [value.name for value in spec_recur.inputs] == ["i"]
    assert [value.name for value in spec_continue.inputs] == ["i"]
    assert spec_continue.nodes[0].op.name == "recur"
    assert len(spec_continue.nodes[0].inputs) == 1
    assert isinstance(spec_continue.nodes[0].inputs[0], GraphExpr)
    assert spec_continue.nodes[0].inputs[0].op.name == "core.binary.+"


def test_graph_ir_optimizer_constraints_do_not_keep_dead_temps_alive() -> None:
    bool_t = TypeBool()
    int_t = TypeInt()
    module = GraphModule(
        name="main",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.binary.<="),
                inputs=(GraphValueRef("x", int_t), GraphLiteral(12, int_t)),
                attrs={},
                outputs=(GraphValue("unused_cmp", bool_t),),
                source_module="main",
                type_expr=bool_t,
            ),
        ),
        return_type_expr=int_t,
        constraints=(Constraint("is_true", "unused_cmp"),),
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(module,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False, specialize_definitions="off"),
    )

    main = optimized.modules[0]
    assert main.nodes == ()
    assert main.constraints == ()


def test_graph_ir_optimizer_inlines_total_top_level_helper() -> None:
    int_t = TypeInt()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", int_t), GraphValue("y", int_t)),
        outputs=(GraphValueRef("sum", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="helper:1",
                op=GraphOp("core.binary.+"),
                inputs=(GraphValueRef("x", int_t), GraphValueRef("y", int_t)),
                attrs={},
                outputs=(GraphValue("sum", int_t),),
                source_module="helper",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("a", int_t), GraphValue("b", int_t)),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphValueRef("a", int_t), GraphValueRef("b", int_t)),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off"),
    )

    assert [module.name for module in optimized.modules] == ["main"]
    main_module = optimized.modules[0]
    assert [node.op.name for node in main_module.nodes] == ["core.binary.+"]
    assert main_module.outputs == (GraphValueRef(main_module.nodes[0].outputs[0].name, int_t),)


def test_graph_ir_optimizer_inlines_single_callsite_helper_with_select() -> None:
    bool_t = TypeBool()
    int_t = TypeInt()
    helper = GraphModule(
        name="helper",
        inputs=(
            GraphValue("cond", bool_t),
            GraphValue("x", int_t),
            GraphValue("y", int_t),
        ),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="helper:1",
                op=GraphOp("core.select"),
                inputs=(
                    GraphValueRef("cond", bool_t),
                    GraphValueRef("x", int_t),
                    GraphValueRef("y", int_t),
                ),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="helper",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(
            GraphValue("flag", bool_t),
            GraphValue("a", int_t),
            GraphValue("b", int_t),
        ),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(
                    GraphValueRef("flag", bool_t),
                    GraphValueRef("a", int_t),
                    GraphValueRef("b", int_t),
                ),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off"),
    )

    assert [module.name for module in optimized.modules] == ["main"]
    optimized_main = optimized.modules[0]
    assert [node.op.name for node in optimized_main.nodes] == ["core.select"]
    assert optimized_main.nodes[0].inputs == (
        GraphValueRef("flag", bool_t),
        GraphValueRef("a", int_t),
        GraphValueRef("b", int_t),
    )
    assert optimized_main.outputs == (GraphValueRef("out", int_t),)


def test_graph_ir_optimizer_inlines_select_helper_with_local_branch_work_at_callsite() -> None:
    bool_t = TypeBool()
    int_t = TypeInt()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("cond", bool_t), GraphValue("x", int_t)),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="helper:1",
                op=GraphOp("core.binary.+"),
                inputs=(GraphValueRef("x", int_t), GraphLiteral(1, int_t)),
                attrs={},
                outputs=(GraphValue("computed", int_t),),
                source_module="helper",
                type_expr=int_t,
            ),
            GraphNode(
                id="helper:2",
                op=GraphOp("core.select"),
                inputs=(
                    GraphValueRef("cond", bool_t),
                    GraphLiteral(0, int_t),
                    GraphValueRef("computed", int_t),
                ),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="helper",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("flag", bool_t), GraphValue("a", int_t)),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphValueRef("flag", bool_t), GraphValueRef("a", int_t)),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off"),
    )

    assert [module.name for module in optimized.modules] == ["main"]
    optimized_main = optimized.modules[0]
    assert [node.op.name for node in optimized_main.nodes] == ["core.select"]
    assert optimized_main.nodes[0].inputs == (
        GraphValueRef("flag", bool_t),
        GraphLiteral(0, int_t),
        GraphExpr(
            op=GraphOp("core.binary.+"),
            inputs=(GraphValueRef("a", int_t), GraphLiteral(1, int_t)),
            attrs={},
            type_expr=int_t,
        ),
    )


def test_graph_ir_optimizer_preserves_lazy_branch_for_inlined_general_call() -> None:
    bool_t = TypeBool()
    int_t = TypeInt()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="helper:1",
                op=GraphOp("core.binary.+"),
                inputs=(GraphValueRef("x", int_t), GraphLiteral(1, int_t)),
                attrs={},
                outputs=(GraphValue("tmp", int_t),),
                source_module="helper",
                type_expr=int_t,
            ),
            GraphNode(
                id="helper:2",
                op=GraphOp("core.binary.+"),
                inputs=(GraphValueRef("tmp", int_t), GraphLiteral(1, int_t)),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="helper",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("flag", bool_t), GraphValue("a", int_t), GraphValue("b", int_t)),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.select"),
                inputs=(
                    GraphValueRef("flag", bool_t),
                    GraphExpr(
                        op=GraphOp("helper"),
                        inputs=(GraphValueRef("a", int_t),),
                        attrs={},
                        type_expr=int_t,
                    ),
                    GraphValueRef("b", int_t),
                ),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off"),
    )

    assert [module.name for module in optimized.modules] == ["main"]
    optimized_main = next(module for module in optimized.modules if module.name == "main")
    assert [node.op.name for node in optimized_main.nodes] == ["core.select"]
    branch = optimized_main.nodes[0].inputs[1]
    assert isinstance(branch, GraphExpr)
    assert branch.op.name == "core.binary.+"
    assert isinstance(branch.inputs[0], GraphExpr)
    assert branch.inputs[0].op.name == "core.binary.+"
    assert branch.inputs[1] == GraphLiteral(1, int_t)


def test_graph_ir_optimizer_inlining_substitutes_callsite_dim_metadata() -> None:
    tensor_formal = TypeTensor("Tensor", ("B", "S", "d"))
    tensor_actual = TypeTensor("Tensor", ("B", "S", 8))
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", tensor_formal),),
        outputs=(GraphValueRef("sum", tensor_formal),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="helper:1",
                op=GraphOp("core.binary.+"),
                inputs=(GraphValueRef("x", tensor_formal), GraphValueRef("x", tensor_formal)),
                attrs={},
                outputs=(GraphValue("sum", tensor_formal),),
                source_module="helper",
                type_expr=tensor_formal,
                dims=tensor_formal.dims,
            ),
        ),
        return_type_expr=tensor_formal,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("a", tensor_actual),),
        outputs=(GraphValueRef("out", tensor_actual),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphValueRef("a", tensor_actual),),
                attrs={},
                outputs=(GraphValue("out", tensor_actual),),
                source_module="main",
                type_expr=tensor_actual,
                dims=tensor_actual.dims,
            ),
        ),
        return_type_expr=tensor_actual,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off"),
    )

    main_module = optimized.modules[0]
    assert [module.name for module in optimized.modules] == ["main"]
    assert [node.op.name for node in main_module.nodes] == ["core.binary.+"]
    assert main_module.nodes[0].outputs[0].type_expr == tensor_actual
    assert main_module.nodes[0].type_expr == tensor_actual
    assert main_module.outputs == (
        GraphValueRef(main_module.nodes[0].outputs[0].name, tensor_actual, tensor_actual.dims),
    )


def test_graph_ir_optimizer_inlines_multi_output_total_helper() -> None:
    int_t = TypeInt()
    helper = GraphModule(
        name="pair",
        inputs=(GraphValue("x", int_t), GraphValue("y", int_t)),
        outputs=(GraphValueRef("x", int_t), GraphValueRef("y", int_t)),
        output_names=("left", "right"),
        nodes=(),
        return_type_expr=TypeTuple((int_t, int_t)),
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("a", int_t), GraphValue("b", int_t)),
        outputs=(GraphValueRef("left", int_t), GraphValueRef("right", int_t)),
        output_names=("left", "right"),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("pair"),
                inputs=(GraphValueRef("a", int_t), GraphValueRef("b", int_t)),
                attrs={},
                outputs=(GraphValue("left", int_t), GraphValue("right", int_t)),
                source_module="main",
                type_expr=TypeTuple((int_t, int_t)),
            ),
        ),
        return_type_expr=TypeTuple((int_t, int_t)),
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off"),
    )

    assert [module.name for module in optimized.modules] == ["main"]
    assert optimized.modules[0].nodes == ()
    assert optimized.modules[0].outputs == (
        GraphValueRef("a", int_t),
        GraphValueRef("b", int_t),
    )


def test_graph_ir_optimizer_inlines_literal_atomic_constant_by_default() -> None:
    dim_t = TypeDim()
    const = GraphModule(
        name="VOCAB_SIZE",
        inputs=(),
        outputs=(GraphLiteral(151936, dim_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=dim_t,
    )
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("vocab", dim_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("VOCAB_SIZE"),
                inputs=(),
                attrs={},
                outputs=(GraphValue("vocab", dim_t),),
                source_module="main",
                type_expr=dim_t,
            ),
        ),
        return_type_expr=dim_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(const, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off"),
    )

    assert [module.name for module in optimized.modules] == ["main"]
    optimized_main = optimized.modules[0]
    assert optimized_main.nodes == ()
    assert optimized_main.outputs == (GraphLiteral(151936, dim_t),)


def test_graph_ir_optimizer_promotes_total_pure_zero_arg_defs_to_globals() -> None:
    dim_t = TypeDim()
    const = GraphModule(
        name="CONTEXT_SIZE",
        inputs=(),
        outputs=(GraphValueRef("value", dim_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="CONTEXT_SIZE:1",
                op=GraphOp("_config_dim"),
                inputs=(GraphPath(True, ("n_positions",)), GraphLiteral(1024, dim_t)),
                attrs={},
                outputs=(GraphValue("value", dim_t),),
                source_module="CONTEXT_SIZE",
                type_expr=dim_t,
            ),
        ),
        return_type_expr=dim_t,
    )
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("ctx", dim_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("CONTEXT_SIZE"),
                inputs=(),
                attrs={},
                outputs=(GraphValue("ctx", dim_t),),
                source_module="main",
                type_expr=dim_t,
            ),
        ),
        return_type_expr=dim_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(const, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off"),
    )

    assert {module.name for module in optimized.modules} == {"CONTEXT_SIZE", "main"}
    optimized_const = next(module for module in optimized.modules if module.name == "CONTEXT_SIZE")
    assert optimized_const.is_global_binding
    optimized_main = next(module for module in optimized.modules if module.name == "main")
    assert optimized_main.nodes == ()
    assert optimized_main.outputs == (GraphValueRef("CONTEXT_SIZE", dim_t),)


def test_graph_ir_optimizer_replaces_literal_global_dims_without_local_constraint() -> None:
    dim_t = TypeDim()
    tensor_with_symbol = TypeTensor(base="Tensor", dims=("B", "VOCAB_SIZE"))
    tensor_with_literal = TypeTensor(base="Tensor", dims=("B", 151936))
    const = GraphModule(
        name="VOCAB_SIZE",
        inputs=(),
        outputs=(GraphLiteral(151936, dim_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=dim_t,
    )
    unconstrained = GraphModule(
        name="unconstrained",
        inputs=(GraphValue("x", tensor_with_symbol),),
        outputs=(GraphValueRef("x", tensor_with_symbol),),
        output_names=("out",),
        nodes=(),
        return_type_expr=tensor_with_symbol,
    )
    constrained = GraphModule(
        name="constrained",
        inputs=(GraphValue("x", tensor_with_symbol),),
        outputs=(GraphValueRef("x", tensor_with_symbol),),
        output_names=("out",),
        nodes=(),
        return_type_expr=tensor_with_symbol,
        constraints=(Constraint("=", "VOCAB_SIZE", 151936),),
    )

    optimized = optimize_graph_program(
        GraphProgram(
            modules=(const, unconstrained, constrained),
            main_module="constrained",
            pragmas={},
        ),
        config=GraphOptimizeConfig(
            constant_dim_substitution=True,
            specialize_definitions="off",
            inline_safe=False,
            prune_to_main=False,
        ),
    )

    unchanged = next(module for module in optimized.modules if module.name == "unconstrained")
    changed = next(module for module in optimized.modules if module.name == "constrained")
    assert unchanged.inputs[0].type_expr == tensor_with_literal
    assert changed.inputs[0].type_expr == tensor_with_literal
    assert changed.return_type_expr == tensor_with_literal


def test_graph_ir_optimizer_replaces_literal_global_value_refs_and_zero_arg_exprs() -> None:
    int_t = TypeInt()
    const = GraphModule(
        name="NUM_LAYERS",
        inputs=(),
        outputs=(GraphLiteral(12, int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("i", int_t),),
        outputs=(GraphValueRef("done", TypeBool()),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.binary.>="),
                inputs=(
                    GraphValueRef("i", int_t),
                    GraphExpr(
                        op=GraphOp("NUM_LAYERS"),
                        inputs=(),
                        attrs={},
                        type_expr=int_t,
                    ),
                ),
                attrs={},
                outputs=(GraphValue("done", TypeBool()),),
                source_module="main",
                type_expr=TypeBool(),
            ),
        ),
        return_type_expr=TypeBool(),
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(const, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            specialize_definitions="off",
            inline_safe=False,
        ),
    )

    optimized_main = optimized.modules[0]
    assert [module.name for module in optimized.modules] == ["main"]
    assert optimized_main.nodes[0].inputs == (
        GraphValueRef("i", int_t),
        GraphLiteral(12, int_t),
    )


def test_graph_ir_optimizer_cses_duplicate_total_pure_nodes() -> None:
    int_t = TypeInt()
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("b", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.binary.+"),
                inputs=(GraphValueRef("x", int_t), GraphLiteral(1, int_t)),
                attrs={},
                outputs=(GraphValue("a", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("core.binary.+"),
                inputs=(GraphValueRef("x", int_t), GraphLiteral(1, int_t)),
                attrs={},
                outputs=(GraphValue("b", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(main,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            atomic_alias_cleanup=False,
            dead_temp_elimination=False,
            constant_folding=False,
            specialize_definitions="off",
            inline_safe=False,
        ),
    )

    assert len(optimized.modules[0].nodes) == 1
    assert optimized.modules[0].outputs == (GraphValueRef("a", int_t),)


def test_graph_ir_optimizer_hoists_duplicate_nested_total_pure_expressions() -> None:
    int_t = TypeInt()
    repeated = GraphExpr(
        op=GraphOp("core.binary.+"),
        inputs=(GraphValueRef("x", int_t), GraphLiteral(1, int_t)),
        attrs={},
        type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("_partial_sink"),
                inputs=(repeated, repeated),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(main,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            atomic_alias_cleanup=False,
            dead_temp_elimination=False,
            constant_folding=False,
            specialize_definitions="off",
            inline_safe=False,
        ),
    )

    nodes = optimized.modules[0].nodes
    assert [node.op.name for node in nodes] == ["core.binary.+", "_partial_sink"]
    assert isinstance(nodes[1].inputs[0], GraphValueRef)
    assert isinstance(nodes[1].inputs[1], GraphValueRef)
    assert nodes[1].inputs[0] == nodes[1].inputs[1]


def test_graph_ir_optimizer_does_not_hoist_duplicate_nested_partial_expressions() -> None:
    int_t = TypeInt()
    repeated = GraphExpr(
        op=GraphOp("Params.param"),
        inputs=(GraphPath(True, ("missing",)),),
        attrs={},
        type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("_partial_sink"),
                inputs=(repeated, repeated),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(main,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            atomic_alias_cleanup=False,
            dead_temp_elimination=False,
            constant_folding=False,
            specialize_definitions="off",
            inline_safe=False,
        ),
    )

    nodes = optimized.modules[0].nodes
    assert [node.op.name for node in nodes] == ["_partial_sink"]
    assert isinstance(nodes[0].inputs[0], GraphExpr)
    assert isinstance(nodes[0].inputs[1], GraphExpr)


def test_graph_ir_optimizer_rejects_stale_tensor_dims_metadata() -> None:
    tensor_type = TypeTensor("Tensor", ("B", "S", 8))
    stale_dims = ("B", "S", "d")
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", tensor_type, tensor_type.dims),),
        outputs=(GraphValueRef("out", tensor_type, tensor_type.dims),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.alias"),
                inputs=(GraphValueRef("x", tensor_type, tensor_type.dims),),
                attrs={},
                outputs=(GraphValue("out", tensor_type, stale_dims),),
                source_module="main",
                type_expr=tensor_type,
                dims=stale_dims,
            ),
        ),
        return_type_expr=tensor_type,
    )

    with pytest.raises(ValueError, match="graph optimizer phase 'input'.*stale dims metadata"):
        optimize_graph_program(
            GraphProgram(modules=(main,), main_module="main", pragmas={}),
            config=GraphOptimizeConfig(
                atomic_alias_cleanup=False,
                dead_temp_elimination=False,
                constant_folding=False,
                specialize_definitions="off",
                inline_safe=False,
            ),
        )


def test_graph_ir_optimizer_rejects_stale_module_call_result_after_dim_binding() -> None:
    dim_t = TypeDim()
    tensor_formal = TypeTensor("Tensor", ("B", "d"))
    tensor_actual = TypeTensor("Tensor", ("B", 8))
    tensor_stale = TypeTensor("Tensor", ("B", 9))
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("d", dim_t), GraphValue("x", tensor_formal),),
        outputs=(GraphValueRef("x", tensor_formal),),
        output_names=("out",),
        nodes=(),
        return_type_expr=tensor_formal,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", tensor_actual, tensor_actual.dims),),
        outputs=(GraphValueRef("out", tensor_stale, tensor_stale.dims),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphLiteral(8, dim_t), GraphValueRef("x", tensor_actual, tensor_actual.dims)),
                attrs={},
                outputs=(GraphValue("out", tensor_stale, tensor_stale.dims),),
                source_module="main",
                type_expr=tensor_stale,
                dims=tensor_stale.dims,
            ),
        ),
        return_type_expr=tensor_stale,
    )

    with pytest.raises(ValueError, match="graph optimizer phase 'input'.*stale type"):
        optimize_graph_program(
            GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
            config=GraphOptimizeConfig(
                atomic_alias_cleanup=False,
                dead_temp_elimination=False,
                constant_folding=False,
                specialize_definitions="off",
                inline_safe=False,
            ),
        )


def test_graph_ir_optimizer_inlines_path_operand_helper() -> None:
    path_t = TypePath()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("p", path_t),),
        outputs=(GraphValueRef("p", path_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=path_t,
    )
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("out", path_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphPath(True, ("wte",)),),
                attrs={},
                outputs=(GraphValue("out", path_t),),
                source_module="main",
                type_expr=path_t,
            ),
        ),
        return_type_expr=path_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off"),
    )

    assert [module.name for module in optimized.modules] == ["main"]
    assert optimized.modules[0].nodes == ()
    assert optimized.modules[0].outputs == (GraphPath(True, ("wte",)),)


def test_graph_ir_optimizer_inlines_constrained_helper_when_constraints_transfer() -> None:
    int_t = TypeInt()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=int_t,
        constraints=(Constraint("not_null", "x"),),
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("a", int_t),),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphValueRef("a", int_t),),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off"),
    )

    assert [module.name for module in optimized.modules] == ["main"]
    optimized_main = optimized.modules[0]
    assert optimized_main.nodes == ()
    assert optimized_main.outputs == (GraphValueRef("a", int_t),)
    assert optimized_main.constraints == (Constraint("not_null", "a"),)


def test_graph_ir_optimizer_does_not_inline_constrained_helper_when_constraint_is_false() -> None:
    int_t = TypeInt()
    maybe_int_t = TypeOptional(int_t)
    null_t = TypeNull()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", maybe_int_t),),
        outputs=(GraphValueRef("x", maybe_int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=maybe_int_t,
        constraints=(Constraint("not_null", "x"),),
    )
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("out", maybe_int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphLiteral(None, null_t),),
                attrs={},
                outputs=(GraphValue("out", maybe_int_t),),
                source_module="main",
                type_expr=maybe_int_t,
            ),
        ),
        return_type_expr=maybe_int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off"),
    )

    assert {module.name for module in optimized.modules} == {"helper", "main"}
    assert next(module for module in optimized.modules if module.name == "main").nodes[0].op.name == "helper"


def test_graph_ir_optimizer_does_not_inline_constrained_helper_when_constraint_substitution_is_unrepresentable() -> None:
    int_t = TypeInt()
    path_t = TypePath()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("p", path_t), GraphValue("x", int_t)),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=int_t,
        constraints=(Constraint("not_null", "p"),),
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("a", int_t),),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphPath(True, ("wte",)), GraphValueRef("a", int_t)),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off"),
    )

    assert {module.name for module in optimized.modules} == {"helper", "main"}
    assert next(module for module in optimized.modules if module.name == "main").nodes[0].op.name == "helper"


def test_graph_ir_optimizer_inlines_dim_value_refs_with_callsite_substitution() -> None:
    dim_t = TypeDim()
    formal_t = TypeTensor("Tensor", ("B", "T"))
    actual_t = TypeTensor("Tensor", ("B", "S"))
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", formal_t, formal_t.dims),),
        outputs=(GraphValueRef("t", dim_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="helper:1",
                op=GraphOp("core.binary.+"),
                inputs=(GraphValueRef("T", dim_t), GraphLiteral(0, dim_t)),
                attrs={},
                outputs=(GraphValue("t", dim_t),),
                source_module="helper",
                type_expr=dim_t,
            ),
        ),
        return_type_expr=dim_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("a", actual_t, actual_t.dims),),
        outputs=(GraphValueRef("out", dim_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphValueRef("a", actual_t, actual_t.dims),),
                attrs={},
                outputs=(GraphValue("out", dim_t),),
                source_module="main",
                type_expr=dim_t,
            ),
        ),
        return_type_expr=dim_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off", constant_folding=False),
    )

    optimized_main = optimized.modules[0]
    assert [module.name for module in optimized.modules] == ["main"]
    assert optimized_main.nodes
    assert optimized_main.nodes[0].inputs[0] == GraphValueRef("S", dim_t)
    assert optimized_main.outputs == (
        GraphValueRef(optimized_main.nodes[0].outputs[0].name, dim_t),
    )


def test_graph_ir_optimizer_inlines_dim_expr_value_refs_with_callsite_substitution() -> None:
    dim_t = TypeDim()
    formal_t = TypeTensor("Tensor", ("B", "HD"))
    head_dim = DimExprBinary("/", "MODEL_DIM", "NUM_HEADS")
    actual_t = TypeTensor("Tensor", ("B", head_dim))
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", formal_t, formal_t.dims),),
        outputs=(GraphValueRef("out", dim_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="helper:1",
                op=GraphOp("core.binary.+"),
                inputs=(GraphValueRef("HD", dim_t), GraphLiteral(0, dim_t)),
                attrs={},
                outputs=(GraphValue("out", dim_t),),
                source_module="helper",
                type_expr=dim_t,
            ),
        ),
        return_type_expr=dim_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("a", actual_t, actual_t.dims),),
        outputs=(GraphValueRef("out", dim_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphValueRef("a", actual_t, actual_t.dims),),
                attrs={},
                outputs=(GraphValue("out", dim_t),),
                source_module="main",
                type_expr=dim_t,
            ),
        ),
        return_type_expr=dim_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off", constant_folding=False),
    )

    optimized_main = optimized.modules[0]
    assert [module.name for module in optimized.modules] == ["main"]
    dim_expr = optimized_main.nodes[0].inputs[0]
    assert isinstance(dim_expr, GraphExpr)
    assert dim_expr.op.name == "core.binary./"
    assert dim_expr.inputs == (
        GraphValueRef("MODEL_DIM", dim_t),
        GraphValueRef("NUM_HEADS", dim_t),
    )


def test_graph_ir_optimizer_does_not_substitute_scalar_dim_with_variadic_row() -> None:
    dim_t = TypeDim()
    formal_t = TypeTensor("Tensor", ("B", "S"))
    actual_t = TypeTensor("Tensor", ("..R", "S"))
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", formal_t, formal_t.dims),),
        outputs=(GraphValueRef("b", dim_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="helper:1",
                op=GraphOp("core.binary.+"),
                inputs=(GraphValueRef("B", dim_t), GraphLiteral(0, dim_t)),
                attrs={},
                outputs=(GraphValue("b", dim_t),),
                source_module="helper",
                type_expr=dim_t,
            ),
        ),
        return_type_expr=dim_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("a", actual_t, actual_t.dims), GraphValue("B", dim_t)),
        outputs=(GraphValueRef("out", dim_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphValueRef("a", actual_t, actual_t.dims),),
                attrs={},
                outputs=(GraphValue("out", dim_t),),
                source_module="main",
                type_expr=dim_t,
            ),
        ),
        return_type_expr=dim_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off", constant_folding=False),
    )

    optimized_main = optimized.modules[0]
    assert [module.name for module in optimized.modules] == ["main"]
    assert optimized_main.nodes[0].inputs[0] == GraphValueRef("B", dim_t)


def test_graph_ir_optimizer_instantiates_variadic_return_rows_at_callsite() -> None:
    dim_t = TypeDim()
    path_t = TypePath()
    formal_in = TypeTensor("Tensor", ("..S",))
    formal_out = TypeTensor("Tensor", ("..S", "dim"))
    actual_in = TypeTensor("Tensor", ("B", "S"))
    actual_out = TypeTensor("Tensor", ("B", "S", "D"))
    helper = GraphModule(
        name="embedding_like",
        inputs=(
            GraphValue("path", path_t),
            GraphValue("x", formal_in, formal_in.dims),
            GraphValue("dim", dim_t),
        ),
        outputs=(GraphValueRef("y", formal_out, formal_out.dims),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="embedding_like:1",
                op=GraphOp("_embedding"),
                inputs=(
                    GraphValueRef("path", path_t),
                    GraphValueRef("x", formal_in, formal_in.dims),
                    GraphValueRef("dim", dim_t),
                ),
                attrs={},
                outputs=(GraphValue("y", formal_out, formal_out.dims),),
                source_module="embedding_like",
                type_expr=formal_out,
                dims=formal_out.dims,
            ),
        ),
        return_type_expr=formal_out,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", actual_in, actual_in.dims), GraphValue("D", dim_t)),
        outputs=(GraphValueRef("tok", actual_out, actual_out.dims),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("embedding_like"),
                inputs=(
                    GraphPath(True, ("wte",)),
                    GraphValueRef("x", actual_in, actual_in.dims),
                    GraphValueRef("D", dim_t),
                ),
                attrs={},
                outputs=(GraphValue("tok", actual_out, actual_out.dims),),
                source_module="main",
                type_expr=actual_out,
                dims=actual_out.dims,
            ),
        ),
        return_type_expr=actual_out,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            atomic_alias_cleanup=False,
            dead_temp_elimination=False,
            constant_folding=False,
            constant_dim_substitution=False,
            common_subexpression_elimination=False,
            specialize_definitions="off",
            inline_safe=False,
        ),
    )

    optimized_main = next(module for module in optimized.modules if module.name == "main")
    assert optimized_main.nodes[0].type_expr == actual_out
    assert optimized_main.nodes[0].outputs[0].type_expr == actual_out
    assert optimized_main.nodes[0].outputs[0].dims == actual_out.dims


def test_graph_ir_optimizer_preserves_more_specific_compatible_callsite_type() -> None:
    dim_t = TypeDim()
    path_t = TypePath()
    null_t = TypeNull()
    formal_in = TypeTensor("Tensor", ("..S",))
    formal_out = TypeTensor("Tensor", ("..S", "dim"))
    actual_in = TypeTensor("Tensor", ("B", "S"))
    refined_out = TypeTensor("Tensor", ("B", "S", "MODEL_DIM"))
    model_dim = GraphModule(
        name="MODEL_DIM",
        inputs=(),
        outputs=(GraphLiteral(768, dim_t),),
        output_names=("value",),
        nodes=(),
        return_type_expr=dim_t,
        is_global_binding=True,
    )
    helper = GraphModule(
        name="embedding_like",
        inputs=(
            GraphValue("path", path_t),
            GraphValue("x", formal_in, formal_in.dims),
            GraphValue("dim", TypeOptional(dim_t), optional=True),
        ),
        outputs=(GraphValueRef("y", formal_out, formal_out.dims),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="embedding_like:1",
                op=GraphOp("_embedding"),
                inputs=(
                    GraphValueRef("path", path_t),
                    GraphValueRef("x", formal_in, formal_in.dims),
                    GraphValueRef("dim", dim_t),
                ),
                attrs={},
                outputs=(GraphValue("y", formal_out, formal_out.dims),),
                source_module="embedding_like",
                type_expr=formal_out,
                dims=formal_out.dims,
            ),
        ),
        return_type_expr=formal_out,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", actual_in, actual_in.dims),),
        outputs=(GraphValueRef("pos", refined_out, refined_out.dims),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("embedding_like"),
                inputs=(
                    GraphPath(True, ("wpe",)),
                    GraphValueRef("x", actual_in, actual_in.dims),
                    GraphLiteral(None, null_t),
                ),
                attrs={},
                outputs=(GraphValue("pos", refined_out, refined_out.dims),),
                source_module="main",
                type_expr=refined_out,
                dims=refined_out.dims,
            ),
        ),
        return_type_expr=refined_out,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(model_dim, helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            atomic_alias_cleanup=False,
            dead_temp_elimination=False,
            constant_folding=False,
            constant_dim_substitution=False,
            common_subexpression_elimination=False,
            specialize_definitions="off",
            inline_safe=False,
        ),
    )

    optimized_main = next(module for module in optimized.modules if module.name == "main")
    assert optimized_main.nodes[0].type_expr == refined_out
    assert optimized_main.nodes[0].outputs[0].type_expr == refined_out
    assert optimized_main.nodes[0].outputs[0].dims == refined_out.dims


def test_graph_ir_optimizer_specializes_shared_null_argument_across_call_sites() -> None:
    tensor_t = TypeTensor("Tensor", ("B", "S", "D"))
    bool_t = TypeBool()
    null_t = TypeNull()
    helper = GraphModule(
        name="embedding_like",
        inputs=(
            GraphValue("x", tensor_t, tensor_t.dims),
            GraphValue("scale", TypeOptional(TypeFloat()), optional=True),
        ),
        outputs=(GraphValueRef("out", tensor_t, tensor_t.dims),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="embedding_like:1",
                op=GraphOp("core.binary.=="),
                inputs=(GraphValueRef("scale", TypeOptional(TypeFloat())), GraphLiteral(None, null_t)),
                attrs={},
                outputs=(GraphValue("is_null", bool_t),),
                source_module="embedding_like",
                type_expr=bool_t,
            ),
            GraphNode(
                id="embedding_like:2",
                op=GraphOp("core.select"),
                inputs=(
                    GraphValueRef("is_null", bool_t),
                    GraphValueRef("x", tensor_t, tensor_t.dims),
                    GraphExpr(
                        op=GraphOp("core.binary.*"),
                        inputs=(GraphValueRef("x", tensor_t, tensor_t.dims), GraphValueRef("scale", TypeOptional(TypeFloat()))),
                        attrs={},
                        type_expr=tensor_t,
                        dims=tensor_t.dims,
                    ),
                ),
                attrs={},
                outputs=(GraphValue("out", tensor_t, tensor_t.dims),),
                source_module="embedding_like",
                type_expr=tensor_t,
                dims=tensor_t.dims,
            ),
        ),
        return_type_expr=tensor_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", tensor_t, tensor_t.dims), GraphValue("y", tensor_t, tensor_t.dims)),
        outputs=(GraphValueRef("a", tensor_t, tensor_t.dims), GraphValueRef("b", tensor_t, tensor_t.dims)),
        output_names=("a", "b"),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("embedding_like"),
                inputs=(GraphValueRef("x", tensor_t, tensor_t.dims), GraphLiteral(None, null_t)),
                attrs={},
                outputs=(GraphValue("a", tensor_t, tensor_t.dims),),
                source_module="main",
                type_expr=tensor_t,
                dims=tensor_t.dims,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("embedding_like"),
                inputs=(GraphValueRef("y", tensor_t, tensor_t.dims), GraphLiteral(None, null_t)),
                attrs={},
                outputs=(GraphValue("b", tensor_t, tensor_t.dims),),
                source_module="main",
                type_expr=tensor_t,
                dims=tensor_t.dims,
            ),
        ),
        return_type_expr=TypeTuple((tensor_t, tensor_t)),
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            specialize_definitions="monomorphize",
            inline_safe=False,
        ),
    )

    spec = next(module for module in optimized.modules if module.name == "embedding_like")
    optimized_main = next(module for module in optimized.modules if module.name == "main")
    assert [value.name for value in spec.inputs] == ["x"]
    assert [node.op.name for node in optimized_main.nodes] == [spec.name, spec.name]
    assert all(len(node.inputs) == 1 for node in optimized_main.nodes)
    assert not any(value.name == "scale" for value in spec.inputs)


def test_graph_ir_optimizer_specializes_single_callsite_global_optional_argument() -> None:
    dim_t = TypeDim()
    bool_t = TypeBool()
    null_t = TypeNull()
    global_dim = GraphModule(
        name="CONTEXT_SIZE",
        inputs=(),
        outputs=(GraphLiteral(1024, dim_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=dim_t,
        is_global_binding=True,
    )
    helper = GraphModule(
        name="windowed",
        inputs=(GraphValue("window", TypeOptional(dim_t), optional=True),),
        outputs=(GraphValueRef("out", dim_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="windowed:1",
                op=GraphOp("core.binary.=="),
                inputs=(GraphValueRef("window", TypeOptional(dim_t)), GraphLiteral(None, null_t)),
                attrs={},
                outputs=(GraphValue("is_null", bool_t),),
                source_module="windowed",
                type_expr=bool_t,
            ),
            GraphNode(
                id="windowed:2",
                op=GraphOp("core.select"),
                inputs=(
                    GraphValueRef("is_null", bool_t),
                    GraphLiteral(0, dim_t),
                    GraphValueRef("window", TypeOptional(dim_t)),
                ),
                attrs={},
                outputs=(GraphValue("out", dim_t),),
                source_module="windowed",
                type_expr=dim_t,
            ),
        ),
        return_type_expr=dim_t,
    )
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("out", dim_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("windowed"),
                inputs=(GraphExpr(op=GraphOp("CONTEXT_SIZE"), inputs=(), attrs={}, type_expr=dim_t),),
                attrs={},
                outputs=(GraphValue("out", dim_t),),
                source_module="main",
                type_expr=dim_t,
            ),
        ),
        return_type_expr=dim_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(global_dim, helper, main), main_module="main", pragmas={}),
    )

    optimized_main = next(module for module in optimized.modules if module.name == "main")
    assert optimized_main.nodes == ()
    assert optimized_main.outputs == (GraphLiteral(1024, dim_t),)
    assert {module.name for module in optimized.modules} == {"main"}


def test_graph_ir_optimizer_folds_global_value_null_comparison() -> None:
    dim_t = TypeDim()
    bool_t = TypeBool()
    global_dim = GraphModule(
        name="CONTEXT_SIZE",
        inputs=(),
        outputs=(GraphLiteral(1024, dim_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=dim_t,
        is_global_binding=True,
    )
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("out", bool_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.binary.=="),
                inputs=(GraphExpr(op=GraphOp("CONTEXT_SIZE"), inputs=(), attrs={}, type_expr=dim_t), GraphLiteral(None, TypeNull())),
                attrs={},
                outputs=(GraphValue("out", bool_t),),
                source_module="main",
                type_expr=bool_t,
            ),
        ),
        return_type_expr=bool_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(global_dim, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False, specialize_definitions="off"),
    )

    optimized_main = optimized.modules[0]
    assert optimized_main.nodes == ()
    assert optimized_main.outputs == (GraphLiteral(False, bool_t),)


def test_graph_ir_optimizer_canonicalizes_generated_value_names() -> None:
    generated_dim = "__flat_8__inl_6_mask__inl_6___flat_2"
    renamed_dim = f"{generated_dim}__dim"
    tensor_t = TypeTensor(base="Tensor", dims=(generated_dim,))
    renamed_tensor_t = TypeTensor(base="Tensor", dims=(renamed_dim,))
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", tensor_t),),
        outputs=(
            GraphValueRef(
                "__flat_8__inl_6_mask__inl_6___flat_2",
                tensor_t,
                tensor_t.dims,
            ),
        ),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("_effectful_read"),
                inputs=(GraphValueRef("x", tensor_t),),
                attrs={},
                outputs=(
                    GraphValue(
                        "__flat_8__inl_6_mask__inl_6___flat_2",
                        tensor_t,
                        tensor_t.dims,
                    ),
                ),
                source_module="main",
                type_expr=tensor_t,
                dims=tensor_t.dims,
            ),
        ),
        return_type_expr=tensor_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(main,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            atomic_alias_cleanup=False,
            dead_temp_elimination=False,
            constant_folding=False,
            common_subexpression_elimination=False,
            specialize_definitions="off",
            inline_safe=False,
        ),
    )

    optimized_main = optimized.modules[0]
    assert optimized_main.nodes[0].outputs == (GraphValue("_v1", renamed_tensor_t, renamed_tensor_t.dims),)
    assert optimized_main.outputs == (GraphValueRef("_v1", renamed_tensor_t, renamed_tensor_t.dims),)


def test_graph_ir_optimizer_preserves_source_value_names_when_canonicalizing() -> None:
    int_t = TypeInt()
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("logits", int_t),),
        output_names=("logits",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("_effectful_read"),
                inputs=(GraphValueRef("x", int_t),),
                attrs={},
                outputs=(GraphValue("logits", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(main,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            atomic_alias_cleanup=False,
            dead_temp_elimination=False,
            constant_folding=False,
            common_subexpression_elimination=False,
            specialize_definitions="off",
            inline_safe=False,
        ),
    )

    optimized_main = optimized.modules[0]
    assert optimized_main.nodes[0].outputs == (GraphValue("logits", int_t),)
    assert optimized_main.outputs == (GraphValueRef("logits", int_t),)


def test_graph_ir_optimizer_canonicalizes_path_template_value_refs() -> None:
    dim_t = TypeDim()
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("param", dim_t),),
        output_names=("param",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("_effectful_dim"),
                inputs=(),
                attrs={},
                outputs=(GraphValue("__flat_1", dim_t),),
                source_module="main",
                type_expr=dim_t,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("_effectful_param"),
                inputs=(GraphPath(absolute=True, parts=("h.{__flat_1}.w",)),),
                attrs={},
                outputs=(GraphValue("param", dim_t),),
                source_module="main",
                type_expr=dim_t,
            ),
        ),
        return_type_expr=dim_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(main,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            atomic_alias_cleanup=False,
            dead_temp_elimination=False,
            constant_folding=False,
            common_subexpression_elimination=False,
            specialize_definitions="off",
            inline_safe=False,
        ),
    )

    optimized_main = optimized.modules[0]
    assert optimized_main.nodes[0].outputs == (GraphValue("_v1", dim_t),)
    assert optimized_main.nodes[1].inputs == (GraphPath(absolute=True, parts=("h.{_v1}.w",)),)


def test_graph_ir_optimizer_removes_tuple_return_repackaging_scaffold() -> None:
    int_t = TypeInt()
    pair_t = TypeTuple((int_t, int_t))
    triple_t = TypeTuple((int_t, int_t, pair_t))
    main = GraphModule(
        name="main",
        inputs=(GraphValue("a", int_t), GraphValue("b", int_t)),
        outputs=(
            GraphValueRef("x", int_t),
            GraphValueRef("y", int_t),
            GraphValueRef("z", pair_t),
        ),
        output_names=("x", "y", "z"),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.tuple"),
                inputs=(GraphValueRef("a", int_t), GraphValueRef("b", int_t)),
                attrs={},
                outputs=(GraphValue("pair", pair_t),),
                source_module="main",
                type_expr=pair_t,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("core.tuple"),
                inputs=(GraphValueRef("a", int_t), GraphValueRef("b", int_t), GraphValueRef("pair", pair_t)),
                attrs={},
                outputs=(GraphValue("x", int_t), GraphValue("y", int_t), GraphValue("z", pair_t)),
                source_module="main",
                type_expr=triple_t,
            ),
        ),
        return_type_expr=triple_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(main,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False, specialize_definitions="off"),
    )

    optimized_main = optimized.modules[0]
    assert optimized_main.nodes == ()
    assert optimized_main.outputs == (
        GraphValueRef("a", int_t),
        GraphValueRef("b", int_t),
        GraphExpr(
            op=GraphOp("core.tuple"),
            inputs=(GraphValueRef("a", int_t), GraphValueRef("b", int_t)),
            attrs={},
            type_expr=pair_t,
        ),
    )


def test_graph_ir_optimizer_inlines_one_node_forwarder_with_argument_reordering() -> None:
    int_t = TypeInt()
    pair_t = TypeTuple((int_t, int_t))
    f = GraphModule(
        name="f",
        inputs=(GraphValue("x", int_t), GraphValue("y", int_t)),
        outputs=(GraphValueRef("out", pair_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="f:1",
                op=GraphOp("_g"),
                inputs=(GraphValueRef("y", int_t), GraphValueRef("x", int_t)),
                attrs={},
                outputs=(GraphValue("out", pair_t),),
                source_module="f",
                type_expr=pair_t,
            ),
        ),
        return_type_expr=pair_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("a", int_t), GraphValue("b", int_t)),
        outputs=(GraphValueRef("z", pair_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("f"),
                inputs=(GraphValueRef("a", int_t), GraphValueRef("b", int_t)),
                attrs={},
                outputs=(GraphValue("z", pair_t),),
                source_module="main",
                type_expr=pair_t,
            ),
        ),
        return_type_expr=pair_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(f, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            atomic_alias_cleanup=False,
            common_subexpression_elimination=False,
            specialize_definitions="off",
        ),
    )

    optimized_main = next(module for module in optimized.modules if module.name == "main")
    assert len(optimized_main.nodes) == 1
    assert optimized_main.nodes[0].op.name == "_g"
    assert optimized_main.nodes[0].inputs == (
        GraphValueRef("b", int_t),
        GraphValueRef("a", int_t),
    )
    assert optimized_main.nodes[0].outputs == (GraphValue("z", pair_t),)


def test_graph_ir_optimizer_inlines_total_pure_nested_expression_callsite() -> None:
    int_t = TypeInt()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("a", int_t),),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.alias"),
                inputs=(
                    GraphExpr(
                        op=GraphOp("helper"),
                        inputs=(GraphValueRef("a", int_t),),
                        attrs={},
                        type_expr=int_t,
                    ),
                ),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off"),
    )

    validate_graph_program(optimized)
    assert {module.name for module in optimized.modules} == {"main"}
    assert optimized.modules[0].outputs == (GraphValueRef("a", int_t),)


def test_graph_ir_optimizer_inlines_single_nested_expression_only_callsite() -> None:
    float_t = TypeFloat()
    dim_t = TypeDim()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", dim_t),),
        outputs=(GraphValueRef("out", float_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="helper:1",
                op=GraphOp("_sqrt"),
                inputs=(GraphValueRef("x", dim_t),),
                attrs={},
                outputs=(GraphValue("root", float_t),),
                source_module="helper",
                type_expr=float_t,
            ),
            GraphNode(
                id="helper:2",
                op=GraphOp("core.binary./"),
                inputs=(GraphLiteral(1.0, float_t), GraphValueRef("root", float_t)),
                attrs={},
                outputs=(GraphValue("out", float_t),),
                source_module="helper",
                type_expr=float_t,
            ),
        ),
        return_type_expr=float_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("D", dim_t),),
        outputs=(GraphValueRef("out", float_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.ascribe"),
                inputs=(
                    GraphExpr(
                        op=GraphOp("helper"),
                        inputs=(GraphValueRef("D", dim_t),),
                        attrs={},
                        type_expr=float_t,
                    ),
                ),
                attrs={},
                outputs=(GraphValue("out", float_t),),
                source_module="main",
                type_expr=float_t,
            ),
        ),
        return_type_expr=float_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off"),
    )

    assert [module.name for module in optimized.modules] == ["main"]
    optimized_main = optimized.modules[0]
    assert [node.op.name for node in optimized_main.nodes] == ["_sqrt", "core.binary./"]
    assert optimized_main.outputs == (
        GraphValueRef(optimized_main.nodes[-1].outputs[0].name, float_t),
    )


def test_graph_ir_optimizer_does_not_pull_lazy_select_branch_calls_out() -> None:
    bool_t = TypeBool()
    int_t = TypeInt()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="helper:1",
                op=GraphOp("_unsafe_if_eager"),
                inputs=(GraphValueRef("x", int_t),),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="helper",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("flag", bool_t), GraphValue("a", int_t)),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.alias"),
                inputs=(
                    GraphExpr(
                        op=GraphOp("core.select"),
                        inputs=(
                            GraphValueRef("flag", bool_t),
                            GraphLiteral(0, int_t),
                            GraphExpr(
                                op=GraphOp("helper"),
                                inputs=(GraphValueRef("a", int_t),),
                                attrs={},
                                type_expr=int_t,
                            ),
                        ),
                        attrs={},
                        type_expr=int_t,
                    ),
                ),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(helper, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(specialize_definitions="off"),
    )

    validate_graph_program(optimized)
    assert {module.name for module in optimized.modules} == {"main"}
    optimized_main = next(module for module in optimized.modules if module.name == "main")
    assert optimized_main.nodes == ()
    select = optimized_main.outputs[0]
    assert isinstance(select, GraphExpr)
    assert select.op.name == "core.select"
    assert isinstance(select.inputs[2], GraphExpr)
    assert select.inputs[2].op.name == "_unsafe_if_eager"


def test_graph_ir_validator_rejects_module_call_type_mismatch() -> None:
    int_t = TypeInt()
    bool_t = TypeBool()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(),
        outputs=(GraphValueRef("y", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphLiteral(True, bool_t),),
                attrs={},
                outputs=(GraphValue("y", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    with pytest.raises(ValueError, match="arg 'x'"):
        validate_graph_program(GraphProgram(modules=(helper, main), main_module="main", pragmas={}))


def test_graph_ir_validator_rejects_stale_node_output_type() -> None:
    int_t = TypeInt()
    bool_t = TypeBool()
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(),
                outputs=(GraphValueRef("y", bool_t),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("core.binary.+"),
                        inputs=(GraphLiteral(1, int_t), GraphLiteral(2, int_t)),
                        attrs={},
                        outputs=(GraphValue("y", bool_t),),
                        source_module="main",
                        type_expr=int_t,
                    ),
                ),
                return_type_expr=bool_t,
            ),
        ),
        main_module="main",
        pragmas={},
    )

    with pytest.raises(ValueError, match="output 'y'"):
        validate_graph_program(graph)


def test_graph_ir_validator_rejects_unbound_dim_in_node_metadata() -> None:
    tensor_t = TypeTensor("Tensor", ("B",))
    stale_t = TypeTensor("Tensor", ("B", "K"))
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("x", tensor_t, tensor_t.dims),),
                outputs=(GraphValueRef("y", tensor_t, tensor_t.dims),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("core.alias"),
                        inputs=(GraphValueRef("x", tensor_t, tensor_t.dims),),
                        attrs={},
                        outputs=(GraphValue("y", stale_t, stale_t.dims),),
                        source_module="main",
                        type_expr=stale_t,
                        dims=stale_t.dims,
                    ),
                ),
                return_type_expr=tensor_t,
            ),
        ),
        main_module="main",
        pragmas={},
    )

    with pytest.raises(ValueError, match="expected Tensor\\[B,K\\], got Tensor\\[B\\]"):
        validate_graph_program(graph)


def test_graph_ir_validator_rejects_stale_module_call_result_after_dim_substitution() -> None:
    dim_t = TypeDim()
    tensor_formal = TypeTensor("Tensor", ("B", "d"))
    tensor_actual = TypeTensor("Tensor", ("B", 8))
    tensor_stale = TypeTensor("Tensor", ("B", 9))
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("d", dim_t), GraphValue("x", tensor_formal, tensor_formal.dims)),
        outputs=(GraphValueRef("x", tensor_formal, tensor_formal.dims),),
        output_names=("out",),
        nodes=(),
        return_type_expr=tensor_formal,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", tensor_actual, tensor_actual.dims),),
        outputs=(GraphValueRef("out", tensor_stale, tensor_stale.dims),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphLiteral(8, dim_t), GraphValueRef("x", tensor_actual, tensor_actual.dims)),
                attrs={},
                outputs=(GraphValue("out", tensor_stale, tensor_stale.dims),),
                source_module="main",
                type_expr=tensor_stale,
                dims=tensor_stale.dims,
            ),
        ),
        return_type_expr=tensor_stale,
    )

    with pytest.raises(ValueError, match="call result 0"):
        validate_graph_program(GraphProgram(modules=(helper, main), main_module="main", pragmas={}))


def test_graph_ir_validator_rejects_select_branch_type_mismatch() -> None:
    bool_t = TypeBool()
    int_t = TypeInt()
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("flag", bool_t),),
                outputs=(GraphValueRef("out", int_t),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("core.select"),
                        inputs=(
                            GraphValueRef("flag", bool_t),
                            GraphLiteral(1, int_t),
                            GraphLiteral(True, bool_t),
                        ),
                        attrs={},
                        outputs=(GraphValue("out", int_t),),
                        source_module="main",
                        type_expr=int_t,
                    ),
                ),
                return_type_expr=int_t,
            ),
        ),
        main_module="main",
        pragmas={},
    )

    with pytest.raises(ValueError, match="false branch"):
        validate_graph_program(graph)


def test_graph_ir_validator_allows_literal_selected_branch_only() -> None:
    bool_t = TypeBool()
    int_t = TypeInt()
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(),
                outputs=(GraphValueRef("out", int_t),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("core.select"),
                        inputs=(
                            GraphLiteral(True, bool_t),
                            GraphLiteral(1, int_t),
                            GraphLiteral(False, bool_t),
                        ),
                        attrs={},
                        outputs=(GraphValue("out", int_t),),
                        source_module="main",
                        type_expr=int_t,
                    ),
                ),
                return_type_expr=int_t,
            ),
        ),
        main_module="main",
        pragmas={},
    )

    validate_graph_program(graph)


def test_graph_ir_validator_rejects_multi_output_call_in_expression_position() -> None:
    int_t = TypeInt()
    pair_t = TypeTuple((int_t, int_t))
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("x", int_t), GraphValueRef("x", int_t)),
        output_names=("a", "b"),
        nodes=(),
        return_type_expr=pair_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("out", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.alias"),
                inputs=(
                    GraphExpr(
                        op=GraphOp("helper"),
                        inputs=(GraphValueRef("x", int_t),),
                        attrs={},
                        type_expr=int_t,
                    ),
                ),
                attrs={},
                outputs=(GraphValue("out", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    with pytest.raises(ValueError, match="single expression"):
        validate_graph_program(GraphProgram(modules=(helper, main), main_module="main", pragmas={}))


def test_graph_ir_renders_back_to_flat_typed_axon() -> None:
    program = parse_axon_program(
        """
{-# MAIN "main" #-}

main :: Int -> Int
main x = do
  y <- x + 1
  return y
"""
    )

    typed = _typed(program, main_module="main")
    graph = lower_axon_program_to_graph_ir(typed, main_module="main")
    assert graph.modules[-1].nodes[0].type_expr == TypeInt()
    assert graph.modules[-1].return_type_expr == TypeInt()
    rendered_axon = graph_program_to_axon_file(graph)
    rendered = render_axon_file(
        rendered_axon,
        show_types=True,
        show_inferred_expr_types=False,
    )

    assert "main :: Int -> Int" in rendered
    assert "y <- x + 1" in rendered
    assert "return y" in rendered


def test_graph_ir_renderer_orders_main_reachable_modules_by_dependency() -> None:
    int_t = TypeInt()
    const = GraphModule(
        name="CONST",
        inputs=(),
        outputs=(GraphLiteral(1, int_t),),
        output_names=("CONST",),
        nodes=(),
        return_type_expr=int_t,
        is_global_binding=True,
    )
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("y", int_t),),
        output_names=("y",),
        nodes=(
            GraphNode(
                id="helper:1",
                op=GraphOp("core.binary.+"),
                inputs=(GraphValueRef("x", int_t), GraphValueRef("CONST", int_t)),
                attrs={},
                outputs=(GraphValue("y", int_t),),
                source_module="helper",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("y", int_t),),
        output_names=("y",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphValueRef("x", int_t),),
                attrs={},
                outputs=(GraphValue("y", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    unused = GraphModule(
        name="unused",
        inputs=(),
        outputs=(GraphLiteral(0, int_t),),
        output_names=("unused",),
        nodes=(),
        return_type_expr=int_t,
    )

    axon = graph_program_to_axon_file(
        GraphProgram(
            modules=(main, unused, helper, const),
            main_module="main",
            pragmas={"main": "main", "checkpoints": "toy"},
        )
    )

    assert [module.name for module in axon.modules] == ["CONST", "helper", "main"]
    assert axon.modules[0].is_global_binding
    assert axon.pragmas == {"main": "main", "checkpoints": "toy"}


def test_graph_ir_renderer_orders_dependency_sccs_deterministically() -> None:
    int_t = TypeInt()
    a = GraphModule(
        name="a",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("y", int_t),),
        output_names=("y",),
        nodes=(
            GraphNode(
                id="a:1",
                op=GraphOp("b"),
                inputs=(GraphValueRef("x", int_t),),
                attrs={},
                outputs=(GraphValue("y", int_t),),
                source_module="a",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    b = GraphModule(
        name="b",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("y", int_t),),
        output_names=("y",),
        nodes=(
            GraphNode(
                id="b:1",
                op=GraphOp("a"),
                inputs=(GraphValueRef("x", int_t),),
                attrs={},
                outputs=(GraphValue("y", int_t),),
                source_module="b",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("y", int_t),),
        output_names=("y",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("b"),
                inputs=(GraphValueRef("x", int_t),),
                attrs={},
                outputs=(GraphValue("y", int_t),),
                source_module="main",
                type_expr=int_t,
            ),
        ),
        return_type_expr=int_t,
    )

    axon = graph_program_to_axon_file(
        GraphProgram(modules=(main, b, a), main_module="main", pragmas={})
    )

    assert [module.name for module in axon.modules] == ["a", "b", "main"]


def test_graph_ir_renders_rich_dot_directly() -> None:
    program = parse_axon_program(
        """
{-# MAIN "main" #-}

helper :: Tensor[B,S] -> Tensor[B,S]
helper x = do
  y <- x + x
  return y

main :: Tensor[B,S] -> Tensor[B,S]
main x = do
  y <- helper x
  z <- y + x
  return z
"""
    )

    graph = lower_axon_program_to_graph_ir(_typed(program, main_module="main"), main_module="main")
    rendered = render_graph_program_to_dot(graph, direction="left-right")

    assert rendered.startswith("digraph GraphIR")
    assert "rankdir=LR" in rendered
    assert "subgraph cluster_helper" in rendered
    assert "subgraph cluster_main" in rendered
    assert "helper" in rendered
    assert "Tensor[B,S]" in rendered
    assert "style=\"dotted\"" in rendered
    assert "lhead=cluster_helper" in rendered
    assert "core.binary.+" in rendered


def test_graph_ir_dot_shows_nested_expression_calls_and_real_label_newlines() -> None:
    program = parse_axon_program(
        """
{-# MAIN "main" #-}

helper :: Tensor[B,S] -> Tensor[B,S]
helper x = do
  y <- x + x
  return y

main :: Bool -> Tensor[B,S] -> Tensor[B,S]
main cond x = do
  y <- cond ? (helper x) : x
  return y
"""
    )

    graph = lower_axon_program_to_graph_ir(_typed(program, main_module="main"), main_module="main")
    rendered = render_graph_program_to_dot(graph)

    assert 'label="module helper\\nreturns Tensor[B,S]' in rendered
    assert "\\nconstraints " in rendered
    assert 'label="module helper\\\\nreturns Tensor[B,S]"' not in rendered
    assert "style=\"dotted\"" in rendered
    assert "lhead=cluster_helper" in rendered
    assert 'label="arg1"' in rendered


def test_graph_ir_lowers_generic_gpt2_kv_as_alternative_lowering_target() -> None:
    program = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2.axon")
    ).ast

    graph = lower_axon_program_to_graph_ir(_typed(program, main_module="gpt2"), main_module="gpt2")

    assert graph.main_module == "gpt2"
    assert len(graph.modules) > 1
    assert graph.modules[-1].name == "gpt2"
    assert all(node.outputs for module in graph.modules for node in module.nodes)


def test_codegen2_tinygrad_emits_gpt2_source() -> None:
    program = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2.axon")
    ).ast

    graph = lower_axon_program_to_graph_ir(_typed(program, main_module="gpt2"), main_module="gpt2")
    table = tinygrad_op_table_markdown(graph)
    code = emit_tinygrad_model_code_from_graph_ir(graph, class_name="TinyGPT2")

    assert "| Op | Count | Reason |" in table
    assert "`embedding`" not in table
    assert "from tinygrad import Tensor, TinyJit, dtypes" in code
    assert "class TinyGPT2" in code
    assert "class TinyGPT2(nn.Module)" not in code
    assert "def load_state_dict(self, state_dict, strict=True):" in code
    assert "def setup(self):" in code
    assert "def after_to(self):" in code
    assert "def _forward(self, input_ids=None, **inputs):" in code
    assert "def _forward_tiny" not in code


def test_codegen2_triton_emits_gpt2_source() -> None:
    program = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2.axon")
    ).ast

    graph = lower_axon_program_to_graph_ir(_typed(program, main_module="gpt2"), main_module="gpt2")
    code = emit_triton_model_code_from_graph_ir(graph, class_name="TritonGPT2")

    assert "import triton" in code
    assert "import triton.language as tl" in code
    assert "def _axon_triton_rope_apply_kernel" in code
    assert "_axon_triton_rope_apply_kernel[grid]" in code
    assert "class TritonGPT2(nn.Module)" in code
    assert "def load_state_dict(self, state_dict, strict=True):" in code
    assert "def setup(self):" in code
    assert "def _forward(self, input_ids=None, **inputs):" in code


def test_codegen2_tinygrad_accepts_where_indices_primitive() -> None:
    tensor_type = _tensor("B", "S")
    idx_type = _tensor("N")
    where_module = GraphModule(
        name="Tensor.where_indices",
        inputs=(GraphValue("x", tensor_type),),
        outputs=(GraphValueRef("idx0", idx_type), GraphValueRef("idx1", idx_type)),
        output_names=("idx0", "idx1"),
        nodes=(
            GraphNode(
                id="Tensor.where_indices:1",
                op=GraphOp("_where_indices"),
                inputs=(GraphValueRef("x", tensor_type),),
                attrs={},
                outputs=(GraphValue("idx0", idx_type), GraphValue("idx1", idx_type)),
                source_module="Tensor.where_indices",
                type_expr=TypeTuple((idx_type, idx_type)),
            ),
        ),
        return_type_expr=TypeTuple((idx_type, idx_type)),
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", tensor_type),),
        outputs=(GraphValueRef("idx0", idx_type), GraphValueRef("idx1", idx_type)),
        output_names=("idx0", "idx1"),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("Tensor.where_indices"),
                inputs=(GraphValueRef("x", tensor_type),),
                attrs={},
                outputs=(GraphValue("idx0", idx_type), GraphValue("idx1", idx_type)),
                source_module="main",
                type_expr=TypeTuple((idx_type, idx_type)),
            ),
        ),
        return_type_expr=TypeTuple((idx_type, idx_type)),
    )
    graph = GraphProgram(
        modules=(where_module, main),
        main_module="main",
        pragmas={"main": "main"},
    )

    table = tinygrad_op_table_markdown(graph)
    code = emit_tinygrad_model_code_from_graph_ir(graph)

    assert "`where_indices`" not in table
    assert "def _where_indices(self, x):" in code


def _expert_linear_graph(*, bias: bool = False, transpose: bool = False) -> GraphProgram:
    x_type = _tensor("B", "T", "K", "D")
    idx_type = TypeTensor("IdxTensor", ("B", "T", "K"))
    y_type = _tensor("B", "T", "K", "O")
    main = GraphModule(
        name="main",
        inputs=(
            GraphValue("x", x_type),
            GraphValue("expert_idx", idx_type),
        ),
        outputs=(GraphValueRef("y", y_type),),
        output_names=("y",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("_expert_linear"),
                inputs=(
                    GraphPath(absolute=True, parts=("experts",)),
                    GraphValueRef("x", x_type),
                    GraphValueRef("expert_idx", idx_type),
                    GraphLiteral(5, TypeDim()),
                    GraphLiteral(bias, TypeBool()),
                    GraphLiteral(transpose, TypeBool()),
                    GraphLiteral("@weight", TypePath()),
                    GraphLiteral("@bias", TypePath()),
                ),
                attrs={},
                outputs=(GraphValue("y", y_type),),
                source_module="main",
                type_expr=y_type,
            ),
        ),
        return_type_expr=y_type,
    )
    return GraphProgram(modules=(main,), main_module="main", pragmas={"main": "main"})


def test_codegen2_torch_expert_linear_selects_per_token_expert_weights() -> None:
    graph = _expert_linear_graph()
    weight = torch.arange(3 * 5 * 4, dtype=torch.float32).reshape(3, 5, 4) / 10.0
    x = torch.arange(2 * 2 * 2 * 4, dtype=torch.float32).reshape(2, 2, 2, 4) / 7.0
    expert_idx = torch.tensor([[[0, 1], [2, 0]], [[1, 2], [0, 1]]], dtype=torch.long)
    model = Codegen2GraphModel.from_state_dict(
        {"experts.weight": weight},
        graph=graph,
    )

    actual = model.forward(x=x, expert_idx=expert_idx)
    expected = torch.matmul(
        x.unsqueeze(-2),
        weight[expert_idx].transpose(-1, -2),
    ).squeeze(-2)

    assert torch.allclose(actual, expected)


def test_codegen2_torch_generated_expert_linear_matches_interpreter() -> None:
    graph = _expert_linear_graph()
    weight = torch.arange(3 * 5 * 4, dtype=torch.float32).reshape(3, 5, 4) / 10.0
    x = torch.arange(2 * 2 * 2 * 4, dtype=torch.float32).reshape(2, 2, 2, 4) / 7.0
    expert_idx = torch.tensor([[[0, 1], [2, 0]], [[1, 2], [0, 1]]], dtype=torch.long)
    namespace: dict[str, object] = {}
    exec(emit_model_code_from_graph_ir(graph), namespace)
    model_cls = namespace["GeneratedAxonModel"]
    model = model_cls.from_state_dict({"experts.weight": weight})

    actual = model.forward(x=x, expert_idx=expert_idx)
    expected = torch.matmul(
        x.unsqueeze(-2),
        weight[expert_idx].transpose(-1, -2),
    ).squeeze(-2)

    assert torch.allclose(actual, expected)


def test_codegen2_torch_generated_expert_linear_groups_with_bias_and_empty_experts() -> None:
    graph = _expert_linear_graph(bias=True)
    weight = torch.arange(4 * 5 * 4, dtype=torch.float32).reshape(4, 5, 4) / 10.0
    bias = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5) / 17.0
    x = torch.arange(1 * 2 * 2 * 4, dtype=torch.float32).reshape(1, 2, 2, 4) / 7.0
    expert_idx = torch.tensor([[[0, 2], [2, 0]]], dtype=torch.long)
    namespace: dict[str, object] = {}
    exec(emit_model_code_from_graph_ir(graph), namespace)
    model_cls = namespace["GeneratedAxonModel"]
    model = model_cls.from_state_dict({"experts.weight": weight, "experts.bias": bias})

    actual = model.forward(x=x, expert_idx=expert_idx)
    expected = torch.matmul(
        x.unsqueeze(-2),
        weight[expert_idx].transpose(-1, -2),
    ).squeeze(-2) + bias[expert_idx]

    assert torch.allclose(actual, expected)


def test_codegen2_torch_generated_expert_linear_groups_transpose() -> None:
    graph = _expert_linear_graph(transpose=True)
    weight = torch.arange(4 * 4 * 5, dtype=torch.float32).reshape(4, 4, 5) / 10.0
    x = torch.arange(1 * 2 * 2 * 4, dtype=torch.float32).reshape(1, 2, 2, 4) / 7.0
    expert_idx = torch.tensor([[[0, 2], [2, 0]]], dtype=torch.long)
    namespace: dict[str, object] = {}
    exec(emit_model_code_from_graph_ir(graph), namespace)
    model_cls = namespace["GeneratedAxonModel"]
    model = model_cls.from_state_dict({"experts.weight": weight})

    actual = model.forward(x=x, expert_idx=expert_idx)
    expected = torch.matmul(
        x.unsqueeze(-2),
        weight[expert_idx],
    ).squeeze(-2)

    assert torch.allclose(actual, expected)


def test_codegen2_torch_generated_list_append_does_not_mutate_input() -> None:
    int_t = TypeInt()
    list_t = TypeList(int_t)
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("values", list_t), GraphValue("value", int_t)),
                outputs=(GraphValueRef("out", list_t),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_list_append"),
                        inputs=(GraphValueRef("values", list_t), GraphValueRef("value", int_t)),
                        attrs={},
                        outputs=(GraphValue("out", list_t),),
                        source_module="main",
                        type_expr=list_t,
                    ),
                ),
                return_type_expr=list_t,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )
    namespace: dict[str, object] = {}
    code = emit_model_code_from_graph_ir(graph)
    assert "out = ([*(values or []), value])" in code
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]
    model = model_cls.from_state_dict({})

    values = [1]
    actual = model.forward(values=values, value=2)

    assert actual == [1, 2]
    assert values == [1]


def test_graph_optimizer_rewrites_dense_gate_up_linear_pair_for_torch() -> None:
    x_type = _tensor("B", "S", "D")
    out_type = _tensor("B", "S", 5)
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("x", x_type, dims=x_type.dims),),
                outputs=(
                    GraphValueRef("gate", out_type, dims=out_type.dims),
                    GraphValueRef("up", out_type, dims=out_type.dims),
                ),
                output_names=("gate", "up"),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("gate",)),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(5, TypeDim()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("gate", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("up",)),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(5, TypeDim()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("up", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                ),
                return_type_expr=TypeTuple((out_type, out_type)),
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(
            backend_intrinsics="codegen2-torch",
            common_subexpression_elimination=False,
        ),
    )
    render_axon_file(graph_program_to_axon_file(optimized), show_types=True)
    main = next(module for module in optimized.modules if module.name == "main")
    assert [node.op.name for node in main.nodes] == ["_linear", "_split"]
    assert len(optimized.packed_parameters) == 1

    code = emit_model_code_from_graph_ir(optimized)
    namespace: dict[str, object] = {}
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]
    gate_weight = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3) / 11.0
    up_weight = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3) / 7.0
    model = model_cls.from_state_dict(
        {
            "gate.weight": gate_weight,
            "up.weight": up_weight,
        }
    )
    x = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3) / 5.0
    actual = model.forward(x=x)
    gate = actual["gate"]
    up = actual["up"]

    assert torch.allclose(gate, torch.nn.functional.linear(x, gate_weight))
    assert torch.allclose(up, torch.nn.functional.linear(x, up_weight))
    assert "gate.weight" not in model.state_dict_tensors
    assert "up.weight" not in model.state_dict_tensors
    fused_keys = [
        key for key in model.state_dict_tensors
        if str(key).startswith("__packed.linear_pack.")
    ]
    assert len(fused_keys) == 1


def test_graph_optimizer_rewrites_non_adjacent_qkv_linear_pack_with_provenance_safety() -> None:
    x_type = _tensor("B", "S", 3)
    q_type = _tensor("B", "S", 5)
    k_type = _tensor("B", "S", 2)
    v_type = _tensor("B", "S", 2)
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("x", x_type, dims=x_type.dims),),
                outputs=(
                    GraphValueRef("q", q_type, dims=q_type.dims),
                    GraphValueRef("k", k_type, dims=k_type.dims),
                    GraphValueRef("v", v_type, dims=v_type.dims),
                ),
                output_names=("q", "k", "v"),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("q_proj",)),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(5, TypeDim()),
                            GraphLiteral(True, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("q", q_type, dims=q_type.dims),),
                        source_module="main",
                        type_expr=q_type,
                        dims=q_type.dims,
                    ),
                    GraphNode(
                        id="main:between",
                        op=GraphOp("core.alias"),
                        inputs=(GraphValueRef("q", q_type, dims=q_type.dims),),
                        attrs={},
                        outputs=(GraphValue("q_alias", q_type, dims=q_type.dims),),
                        source_module="main",
                        type_expr=q_type,
                        dims=q_type.dims,
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("k_proj",)),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(2, TypeDim()),
                            GraphLiteral(True, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("k", k_type, dims=k_type.dims),),
                        source_module="main",
                        type_expr=k_type,
                        dims=k_type.dims,
                    ),
                    GraphNode(
                        id="main:3",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("v_proj",)),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(2, TypeDim()),
                            GraphLiteral(True, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("v", v_type, dims=v_type.dims),),
                        source_module="main",
                        type_expr=v_type,
                        dims=v_type.dims,
                    ),
                ),
                return_type_expr=TypeTuple((q_type, k_type, v_type)),
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(graph, config=GraphOptimizeConfig())
    main = next(module for module in optimized.modules if module.name == "main")
    assert [node.op.name for node in main.nodes[:2]] == ["_linear", "_split"]
    assert len(optimized.packed_parameters) == 2

    code = emit_model_code_from_graph_ir(optimized)
    namespace: dict[str, object] = {}
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]
    q_weight = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3) / 11.0
    k_weight = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3) / 7.0
    v_weight = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3) / 5.0
    q_bias = torch.arange(5, dtype=torch.float32) / 13.0
    k_bias = torch.arange(2, dtype=torch.float32) / 17.0
    v_bias = torch.arange(2, dtype=torch.float32) / 19.0
    model = model_cls.from_state_dict(
        {
            "q_proj.weight": q_weight,
            "k_proj.weight": k_weight,
            "v_proj.weight": v_weight,
            "q_proj.bias": q_bias,
            "k_proj.bias": k_bias,
            "v_proj.bias": v_bias,
        }
    )
    x = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3) / 5.0
    actual = model.forward(x=x)

    assert torch.allclose(actual["q"], torch.nn.functional.linear(x, q_weight, q_bias))
    assert torch.allclose(actual["k"], torch.nn.functional.linear(x, k_weight, k_bias))
    assert torch.allclose(actual["v"], torch.nn.functional.linear(x, v_weight, v_bias))
    assert "q_proj.weight" not in model.state_dict_tensors
    assert "k_proj.weight" not in model.state_dict_tensors
    assert "v_proj.weight" not in model.state_dict_tensors


def test_graph_optimizer_does_not_pack_qkv_when_parameter_used_elsewhere() -> None:
    x_type = _tensor("B", "S", 3)
    q_type = _tensor("B", "S", 5)
    k_type = _tensor("B", "S", 2)
    v_type = _tensor("B", "S", 2)
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("x", x_type, dims=x_type.dims),),
                outputs=(
                    GraphValueRef("q", q_type, dims=q_type.dims),
                    GraphValueRef("k", k_type, dims=k_type.dims),
                    GraphValueRef("v", v_type, dims=v_type.dims),
                    GraphValueRef("k_weight", TypeAny()),
                ),
                output_names=("q", "k", "v", "k_weight"),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("q_proj",)),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(5, TypeDim()),
                            GraphLiteral(True, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("q", q_type, dims=q_type.dims),),
                        source_module="main",
                        type_expr=q_type,
                        dims=q_type.dims,
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("k_proj",)),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(2, TypeDim()),
                            GraphLiteral(True, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("k", k_type, dims=k_type.dims),),
                        source_module="main",
                        type_expr=k_type,
                        dims=k_type.dims,
                    ),
                    GraphNode(
                        id="main:3",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("v_proj",)),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(2, TypeDim()),
                            GraphLiteral(True, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("v", v_type, dims=v_type.dims),),
                        source_module="main",
                        type_expr=v_type,
                        dims=v_type.dims,
                    ),
                    GraphNode(
                        id="main:4",
                        op=GraphOp("_params_param"),
                        inputs=(GraphPath(True, ("k_proj", "weight")),),
                        attrs={},
                        outputs=(GraphValue("k_weight", TypeAny()),),
                        source_module="main",
                        type_expr=TypeAny(),
                    ),
                ),
                return_type_expr=TypeTuple((q_type, k_type, v_type, TypeAny())),
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(graph, config=GraphOptimizeConfig())
    main = next(module for module in optimized.modules if module.name == "main")
    assert [node.op.name for node in main.nodes[:3]] == ["_linear", "_linear", "_linear"]
    assert not optimized.packed_parameters


def test_graph_optimizer_materializes_templated_qkv_pack_at_load_time() -> None:
    int_t = TypeInt()
    x_type = _tensor("B", "S", 3)
    q_type = _tensor("B", "S", 5)
    k_type = _tensor("B", "S", 2)
    v_type = _tensor("B", "S", 2)
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(
                    GraphValue("i", int_t),
                    GraphValue("x", x_type, dims=x_type.dims),
                ),
                outputs=(
                    GraphValueRef("q", q_type, dims=q_type.dims),
                    GraphValueRef("k", k_type, dims=k_type.dims),
                    GraphValueRef("v", v_type, dims=v_type.dims),
                ),
                output_names=("q", "k", "v"),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("layers", "{i}", "q_proj")),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(5, TypeDim()),
                            GraphLiteral(True, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("q", q_type, dims=q_type.dims),),
                        source_module="main",
                        type_expr=q_type,
                        dims=q_type.dims,
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("layers", "{i}", "k_proj")),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(2, TypeDim()),
                            GraphLiteral(True, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("k", k_type, dims=k_type.dims),),
                        source_module="main",
                        type_expr=k_type,
                        dims=k_type.dims,
                    ),
                    GraphNode(
                        id="main:3",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("layers", "{i}", "v_proj")),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(2, TypeDim()),
                            GraphLiteral(True, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("v", v_type, dims=v_type.dims),),
                        source_module="main",
                        type_expr=v_type,
                        dims=v_type.dims,
                    ),
                ),
                return_type_expr=TypeTuple((q_type, k_type, v_type)),
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(graph, config=GraphOptimizeConfig())
    assert len(optimized.packed_parameters) == 2
    code = emit_model_code_from_graph_ir(optimized)
    namespace: dict[str, object] = {}
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]
    q_weight = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3) / 11.0
    k_weight = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3) / 7.0
    v_weight = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3) / 5.0
    q_bias = torch.arange(5, dtype=torch.float32) / 13.0
    k_bias = torch.arange(2, dtype=torch.float32) / 17.0
    v_bias = torch.arange(2, dtype=torch.float32) / 19.0

    model = model_cls.from_state_dict(
        {
            "layers.0.q_proj.weight": q_weight,
            "layers.0.k_proj.weight": k_weight,
            "layers.0.v_proj.weight": v_weight,
            "layers.0.q_proj.bias": q_bias,
            "layers.0.k_proj.bias": k_bias,
            "layers.0.v_proj.bias": v_bias,
        }
    )

    assert "layers.0.q_proj.weight" not in model.state_dict_tensors
    assert "layers.0.k_proj.weight" not in model.state_dict_tensors
    assert "layers.0.v_proj.weight" not in model.state_dict_tensors
    assert any(str(key).startswith("__packed.linear_pack.") for key in model.state_dict_tensors)

    x = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3) / 5.0
    actual = model.forward(i=0, x=x)

    assert torch.allclose(actual["q"], torch.nn.functional.linear(x, q_weight, q_bias))
    assert torch.allclose(actual["k"], torch.nn.functional.linear(x, k_weight, k_bias))
    assert torch.allclose(actual["v"], torch.nn.functional.linear(x, v_weight, v_bias))


def test_graph_optimizer_does_not_rewrite_gate_up_pair_when_weight_reused() -> None:
    x_type = _tensor("B", "S", "D")
    out_type = _tensor("B", "S", 5)
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("x", x_type, dims=x_type.dims),),
                outputs=(
                    GraphValueRef("gate", out_type, dims=out_type.dims),
                    GraphValueRef("up", out_type, dims=out_type.dims),
                    GraphValueRef("gate_again", out_type, dims=out_type.dims),
                ),
                output_names=("gate", "up", "gate_again"),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("gate",)),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(5, TypeDim()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("gate", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("up",)),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(5, TypeDim()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("up", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                    GraphNode(
                        id="main:3",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("gate",)),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(5, TypeDim()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("gate_again", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                ),
                return_type_expr=TypeTuple((out_type, out_type, out_type)),
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(
            backend_intrinsics="codegen2-torch",
            common_subexpression_elimination=False,
        ),
    )
    main = next(module for module in optimized.modules if module.name == "main")
    assert "__torch_gate_up_linear_pair" not in [node.op.name for node in main.nodes]


def test_graph_optimizer_does_not_rewrite_gate_up_pair_when_weight_used_by_other_primitive() -> None:
    x_type = _tensor("B", "S", "D")
    out_type = _tensor("B", "S", 5)
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("x", x_type, dims=x_type.dims),),
                outputs=(
                    GraphValueRef("gate", out_type, dims=out_type.dims),
                    GraphValueRef("up", out_type, dims=out_type.dims),
                    GraphValueRef("gate_weight", TypeAny()),
                ),
                output_names=("gate", "up", "gate_weight"),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("gate",)),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(5, TypeDim()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("gate", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("up",)),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(5, TypeDim()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("up", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                    GraphNode(
                        id="main:3",
                        op=GraphOp("_params_param"),
                        inputs=(GraphPath(True, ("gate", "weight")),),
                        attrs={},
                        outputs=(GraphValue("gate_weight", TypeAny()),),
                        source_module="main",
                        type_expr=TypeAny(),
                    ),
                ),
                return_type_expr=TypeTuple((out_type, out_type, TypeAny())),
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    not_selected = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(backend_intrinsics="codegen2-torch:sdpa"),
    )
    not_selected_main = next(module for module in not_selected.modules if module.name == "main")
    assert "__torch_selected_expert_packed_swiglu_ffn" not in [node.op.name for node in not_selected_main.nodes]

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(backend_intrinsics="codegen2-torch:selected_expert_packed_swiglu_ffn"),
    )
    main = next(module for module in optimized.modules if module.name == "main")
    assert "__torch_gate_up_linear_pair" not in [node.op.name for node in main.nodes]


def test_graph_optimizer_rewrites_dense_swiglu_ffn_for_torch() -> None:
    x_type = _tensor("B", "S", 3)
    hidden_type = _tensor("B", "S", 5)
    out_type = _tensor("B", "S", 2)
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("x", x_type, dims=x_type.dims),),
                outputs=(GraphValueRef("y", out_type, dims=out_type.dims),),
                output_names=("y",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("__torch_gate_up_linear_pair"),
                        inputs=(
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphPath(True, ("gate", "weight")),
                            GraphPath(True, ("up", "weight")),
                            GraphPath(True, ("gate", "bias")),
                            GraphPath(True, ("up", "bias")),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                        ),
                        attrs={},
                        outputs=(
                            GraphValue("gate", hidden_type, dims=hidden_type.dims),
                            GraphValue("up", hidden_type, dims=hidden_type.dims),
                        ),
                        source_module="main",
                        type_expr=TypeTuple((hidden_type, hidden_type)),
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_activations_silu"),
                        inputs=(GraphValueRef("gate", hidden_type, dims=hidden_type.dims),),
                        attrs={},
                        outputs=(GraphValue("gate_act", hidden_type, dims=hidden_type.dims),),
                        source_module="main",
                        type_expr=hidden_type,
                        dims=hidden_type.dims,
                    ),
                    GraphNode(
                        id="main:3",
                        op=GraphOp("_mul"),
                        inputs=(
                            GraphValueRef("gate_act", hidden_type, dims=hidden_type.dims),
                            GraphValueRef("up", hidden_type, dims=hidden_type.dims),
                        ),
                        attrs={},
                        outputs=(GraphValue("hidden", hidden_type, dims=hidden_type.dims),),
                        source_module="main",
                        type_expr=hidden_type,
                        dims=hidden_type.dims,
                    ),
                    GraphNode(
                        id="main:4",
                        op=GraphOp("_linear"),
                        inputs=(
                            GraphPath(True, ("down",)),
                            GraphValueRef("hidden", hidden_type, dims=hidden_type.dims),
                            GraphLiteral(2, TypeDim()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(None, TypeNull()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("y", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                ),
                return_type_expr=out_type,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(backend_intrinsics="codegen2-torch"),
    )
    main = next(module for module in optimized.modules if module.name == "main")
    assert [node.op.name for node in main.nodes] == ["__torch_swiglu_ffn"]

    code = emit_model_code_from_graph_ir(optimized)
    namespace: dict[str, object] = {}
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]
    gate_weight = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3) / 11.0
    up_weight = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3) / 7.0
    down_weight = torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5) / 13.0
    model = model_cls.from_state_dict(
        {
            "gate.weight": gate_weight,
            "up.weight": up_weight,
            "down.weight": down_weight,
        }
    )
    x = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3) / 5.0
    actual = model.forward(x=x)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(x, gate_weight))
        * torch.nn.functional.linear(x, up_weight),
        down_weight,
    )

    assert torch.allclose(actual, expected)
    assert "gate.weight" not in model.state_dict_tensors
    assert "up.weight" not in model.state_dict_tensors
    assert "down.weight" in model.state_dict_tensors


def test_graph_optimizer_rewrites_expert_swiglu_ffn_for_torch() -> None:
    x_type = _tensor("B", "T", "K", 3)
    idx_type = TypeTensor("IdxTensor", ("B", "T", "K"))
    hidden_type = _tensor("B", "T", "K", 5)
    out_type = _tensor("B", "T", "K", 2)
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(
                    GraphValue("x", x_type, dims=x_type.dims),
                    GraphValue("expert_idx", idx_type, dims=idx_type.dims),
                ),
                outputs=(GraphValueRef("y", out_type, dims=out_type.dims),),
                output_names=("y",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_expert_linear"),
                        inputs=(
                            GraphPath(True, ("experts", "w1")),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphValueRef("expert_idx", idx_type, dims=idx_type.dims),
                            GraphLiteral(5, TypeDim()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("gate", hidden_type, dims=hidden_type.dims),),
                        source_module="main",
                        type_expr=hidden_type,
                        dims=hidden_type.dims,
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_expert_linear"),
                        inputs=(
                            GraphPath(True, ("experts", "w3")),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphValueRef("expert_idx", idx_type, dims=idx_type.dims),
                            GraphLiteral(5, TypeDim()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("up", hidden_type, dims=hidden_type.dims),),
                        source_module="main",
                        type_expr=hidden_type,
                        dims=hidden_type.dims,
                    ),
                    GraphNode(
                        id="main:3",
                        op=GraphOp("_activations_silu"),
                        inputs=(GraphValueRef("gate", hidden_type, dims=hidden_type.dims),),
                        attrs={},
                        outputs=(GraphValue("gate_act", hidden_type, dims=hidden_type.dims),),
                        source_module="main",
                        type_expr=hidden_type,
                        dims=hidden_type.dims,
                    ),
                    GraphNode(
                        id="main:4",
                        op=GraphOp("_mul"),
                        inputs=(
                            GraphValueRef("gate_act", hidden_type, dims=hidden_type.dims),
                            GraphValueRef("up", hidden_type, dims=hidden_type.dims),
                        ),
                        attrs={},
                        outputs=(GraphValue("hidden", hidden_type, dims=hidden_type.dims),),
                        source_module="main",
                        type_expr=hidden_type,
                        dims=hidden_type.dims,
                    ),
                    GraphNode(
                        id="main:5",
                        op=GraphOp("_expert_linear"),
                        inputs=(
                            GraphPath(True, ("experts", "w2")),
                            GraphValueRef("hidden", hidden_type, dims=hidden_type.dims),
                            GraphValueRef("expert_idx", idx_type, dims=idx_type.dims),
                            GraphLiteral(2, TypeDim()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("y", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                ),
                return_type_expr=out_type,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(backend_intrinsics="codegen2-torch"),
    )
    main = next(module for module in optimized.modules if module.name == "main")
    assert [node.op.name for node in main.nodes] == ["__torch_expert_swiglu_ffn"]

    code = emit_model_code_from_graph_ir(optimized)
    assert "_expert_swiglu_ffn" in code
    namespace: dict[str, object] = {}
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]
    gate_weight = torch.arange(3 * 5 * 3, dtype=torch.float32).reshape(3, 5, 3) / 11.0
    up_weight = torch.arange(3 * 5 * 3, dtype=torch.float32).reshape(3, 5, 3) / 7.0
    down_weight = torch.arange(3 * 2 * 5, dtype=torch.float32).reshape(3, 2, 5) / 13.0
    model = model_cls.from_state_dict(
        {
            "experts.w1.weight": gate_weight,
            "experts.w3.weight": up_weight,
            "experts.w2.weight": down_weight,
        }
    )
    x = torch.arange(2 * 2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 2, 3) / 5.0
    expert_idx = torch.tensor([[[0, 1], [2, 0]], [[1, 2], [0, 1]]], dtype=torch.long)

    actual = model.forward(x=x, expert_idx=expert_idx)
    gate = torch.matmul(x.unsqueeze(-2), gate_weight[expert_idx].transpose(-1, -2)).squeeze(-2)
    up = torch.matmul(x.unsqueeze(-2), up_weight[expert_idx].transpose(-1, -2)).squeeze(-2)
    expected = torch.matmul(
        (torch.nn.functional.silu(gate) * up).unsqueeze(-2),
        down_weight[expert_idx].transpose(-1, -2),
    ).squeeze(-2)

    assert torch.allclose(actual, expected)


def test_graph_optimizer_rewrites_expert_packed_swiglu_ffn_for_torch() -> None:
    x_type = _tensor("B", "T", "K", 3)
    idx_type = TypeTensor("IdxTensor", ("B", "T", "K"))
    packed_type = _tensor("B", "T", "K", 10)
    hidden_type = _tensor("B", "T", "K", 5)
    out_type = _tensor("B", "T", "K", 2)
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(
                    GraphValue("x", x_type, dims=x_type.dims),
                    GraphValue("expert_idx", idx_type, dims=idx_type.dims),
                ),
                outputs=(GraphValueRef("y", out_type, dims=out_type.dims),),
                output_names=("y",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_expert_linear"),
                        inputs=(
                            GraphPath(True, ("experts", "gate_up")),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphValueRef("expert_idx", idx_type, dims=idx_type.dims),
                            GraphLiteral(10, TypeDim()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("gate_up", packed_type, dims=packed_type.dims),),
                        source_module="main",
                        type_expr=packed_type,
                        dims=packed_type.dims,
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_chunk"),
                        inputs=(
                            GraphValueRef("gate_up", packed_type, dims=packed_type.dims),
                            GraphLiteral(-1, TypeInt()),
                            GraphLiteral(2, TypeDim()),
                        ),
                        attrs={},
                        outputs=(
                            GraphValue("gate", hidden_type, dims=hidden_type.dims),
                            GraphValue("up", hidden_type, dims=hidden_type.dims),
                        ),
                        source_module="main",
                        type_expr=TypeTuple((hidden_type, hidden_type)),
                    ),
                    GraphNode(
                        id="main:3",
                        op=GraphOp("_activations_silu"),
                        inputs=(GraphValueRef("gate", hidden_type, dims=hidden_type.dims),),
                        attrs={},
                        outputs=(GraphValue("gate_act", hidden_type, dims=hidden_type.dims),),
                        source_module="main",
                        type_expr=hidden_type,
                        dims=hidden_type.dims,
                    ),
                    GraphNode(
                        id="main:4",
                        op=GraphOp("_mul"),
                        inputs=(
                            GraphValueRef("gate_act", hidden_type, dims=hidden_type.dims),
                            GraphValueRef("up", hidden_type, dims=hidden_type.dims),
                        ),
                        attrs={},
                        outputs=(GraphValue("hidden", hidden_type, dims=hidden_type.dims),),
                        source_module="main",
                        type_expr=hidden_type,
                        dims=hidden_type.dims,
                    ),
                    GraphNode(
                        id="main:5",
                        op=GraphOp("_expert_linear"),
                        inputs=(
                            GraphPath(True, ("experts", "down")),
                            GraphValueRef("hidden", hidden_type, dims=hidden_type.dims),
                            GraphValueRef("expert_idx", idx_type, dims=idx_type.dims),
                            GraphLiteral(2, TypeDim()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("y", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                ),
                return_type_expr=out_type,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(backend_intrinsics="codegen2-torch"),
    )
    main = next(module for module in optimized.modules if module.name == "main")
    assert [node.op.name for node in main.nodes] == ["__torch_expert_packed_swiglu_ffn"]

    code = emit_model_code_from_graph_ir(optimized)
    assert "_expert_packed_swiglu_ffn" in code
    namespace: dict[str, object] = {}
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]
    gate_up_weight = torch.arange(3 * 10 * 3, dtype=torch.float32).reshape(3, 10, 3) / 11.0
    down_weight = torch.arange(3 * 2 * 5, dtype=torch.float32).reshape(3, 2, 5) / 13.0
    model = model_cls.from_state_dict(
        {
            "experts.gate_up.weight": gate_up_weight,
            "experts.down.weight": down_weight,
        }
    )
    x = torch.arange(2 * 2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 2, 3) / 5.0
    expert_idx = torch.tensor([[[0, 1], [2, 0]], [[1, 2], [0, 1]]], dtype=torch.long)

    actual = model.forward(x=x, expert_idx=expert_idx)
    gate_up = torch.matmul(
        x.unsqueeze(-2),
        gate_up_weight[expert_idx].transpose(-1, -2),
    ).squeeze(-2)
    gate, up = torch.chunk(gate_up, 2, dim=-1)
    hidden = torch.nn.functional.silu(gate) * up
    expected = torch.matmul(
        hidden.unsqueeze(-2),
        down_weight[expert_idx].transpose(-1, -2),
    ).squeeze(-2)

    assert torch.allclose(actual, expected)


def test_graph_optimizer_rewrites_direct_selected_expert_packed_swiglu_ffn_for_torch() -> None:
    x_type = _tensor("B", "T", "D")
    scores_type = _tensor("B", "T", "K")
    idx_type = _tensor("B", "T", "K")
    expanded_type = _tensor("B", "T", "K", "D")
    gate_up_type = _tensor("B", "T", "K", "F")
    hidden_type = _tensor("B", "T", "K", "H")
    values_type = _tensor("B", "T", "K", "D")
    output_type = _tensor("B", "T", "D")
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(
                    GraphValue("x", x_type, dims=x_type.dims),
                    GraphValue("topk_scores", scores_type, dims=scores_type.dims),
                    GraphValue("topk_indices", idx_type, dims=idx_type.dims),
                ),
                outputs=(GraphValueRef("out", output_type, dims=output_type.dims),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_unsqueeze"),
                        inputs=(GraphValueRef("x", x_type, dims=x_type.dims), GraphLiteral(2, TypeInt())),
                        attrs={},
                        outputs=(GraphValue("x_unsq", _tensor("B", "T", "1", "D"), dims=("B", "T", "1", "D")),),
                        source_module="main",
                        type_expr=_tensor("B", "T", "1", "D"),
                        dims=("B", "T", "1", "D"),
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_expand"),
                        inputs=(
                            GraphValueRef("x_unsq", _tensor("B", "T", "1", "D"), dims=("B", "T", "1", "D")),
                            GraphExpr(
                                op=GraphOp("core.list"),
                                inputs=(
                                    GraphLiteral("B", TypeDim()),
                                    GraphLiteral("T", TypeDim()),
                                    GraphLiteral("K", TypeDim()),
                                    GraphLiteral("D", TypeDim()),
                                ),
                                attrs={},
                                type_expr=TypeList(TypeDim()),
                            ),
                        ),
                        attrs={},
                        outputs=(GraphValue("x_sel", expanded_type, dims=expanded_type.dims),),
                        source_module="main",
                        type_expr=expanded_type,
                        dims=expanded_type.dims,
                    ),
                    GraphNode(
                        id="main:3",
                        op=GraphOp("_expert_linear"),
                        inputs=(
                            GraphPath(True, ("experts", "gate_up")),
                            GraphValueRef("x_sel", expanded_type, dims=expanded_type.dims),
                            GraphValueRef("topk_indices", idx_type, dims=idx_type.dims),
                            GraphLiteral(10, TypeDim()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("gate_up", gate_up_type, dims=gate_up_type.dims),),
                        source_module="main",
                        type_expr=gate_up_type,
                        dims=gate_up_type.dims,
                    ),
                    GraphNode(
                        id="main:4",
                        op=GraphOp("_chunk"),
                        inputs=(
                            GraphValueRef("gate_up", gate_up_type, dims=gate_up_type.dims),
                            GraphLiteral(-1, TypeInt()),
                            GraphLiteral(2, TypeDim()),
                        ),
                        attrs={},
                        outputs=(
                            GraphValue("gate", hidden_type, dims=hidden_type.dims),
                            GraphValue("up", hidden_type, dims=hidden_type.dims),
                        ),
                        source_module="main",
                        type_expr=TypeTuple((hidden_type, hidden_type)),
                    ),
                    GraphNode(
                        id="main:5",
                        op=GraphOp("_activations_silu"),
                        inputs=(GraphValueRef("gate", hidden_type, dims=hidden_type.dims),),
                        attrs={},
                        outputs=(GraphValue("gate_act", hidden_type, dims=hidden_type.dims),),
                        source_module="main",
                        type_expr=hidden_type,
                        dims=hidden_type.dims,
                    ),
                    GraphNode(
                        id="main:6",
                        op=GraphOp("_mul"),
                        inputs=(
                            GraphValueRef("gate_act", hidden_type, dims=hidden_type.dims),
                            GraphValueRef("up", hidden_type, dims=hidden_type.dims),
                        ),
                        attrs={},
                        outputs=(GraphValue("hidden", hidden_type, dims=hidden_type.dims),),
                        source_module="main",
                        type_expr=hidden_type,
                        dims=hidden_type.dims,
                    ),
                    GraphNode(
                        id="main:7",
                        op=GraphOp("_expert_linear"),
                        inputs=(
                            GraphPath(True, ("experts", "down")),
                            GraphValueRef("hidden", hidden_type, dims=hidden_type.dims),
                            GraphValueRef("topk_indices", idx_type, dims=idx_type.dims),
                            GraphLiteral(3, TypeDim()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                            GraphLiteral("weight", TypeString()),
                            GraphLiteral("bias", TypeString()),
                        ),
                        attrs={},
                        outputs=(GraphValue("values", values_type, dims=values_type.dims),),
                        source_module="main",
                        type_expr=values_type,
                        dims=values_type.dims,
                    ),
                    GraphNode(
                        id="main:8",
                        op=GraphOp("_unsqueeze"),
                        inputs=(GraphValueRef("topk_scores", scores_type, dims=scores_type.dims), GraphLiteral(-1, TypeInt())),
                        attrs={},
                        outputs=(GraphValue("scale", _tensor("B", "T", "K", "1"), dims=("B", "T", "K", "1")),),
                        source_module="main",
                        type_expr=_tensor("B", "T", "K", "1"),
                        dims=("B", "T", "K", "1"),
                    ),
                    GraphNode(
                        id="main:9",
                        op=GraphOp("_mul"),
                        inputs=(
                            GraphValueRef("values", values_type, dims=values_type.dims),
                            GraphValueRef("scale", _tensor("B", "T", "K", "1"), dims=("B", "T", "K", "1")),
                        ),
                        attrs={},
                        outputs=(GraphValue("weighted", values_type, dims=values_type.dims),),
                        source_module="main",
                        type_expr=values_type,
                        dims=values_type.dims,
                    ),
                    GraphNode(
                        id="main:10",
                        op=GraphOp("_sum"),
                        inputs=(
                            GraphValueRef("weighted", values_type, dims=values_type.dims),
                            GraphLiteral(2, TypeInt()),
                            GraphLiteral(False, TypeBool()),
                        ),
                        attrs={},
                        outputs=(GraphValue("out", output_type, dims=output_type.dims),),
                        source_module="main",
                        type_expr=output_type,
                        dims=output_type.dims,
                    ),
                ),
                return_type_expr=output_type,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(backend_intrinsics="codegen2-torch"),
    )
    main = next(module for module in optimized.modules if module.name == "main")
    assert [node.op.name for node in main.nodes] == ["__torch_selected_expert_packed_swiglu_ffn"]

    code = emit_model_code_from_graph_ir(optimized)
    assert "_selected_expert_packed_swiglu_ffn" in code
    namespace: dict[str, object] = {}
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]
    gate_up_weight = torch.arange(3 * 10 * 3, dtype=torch.float32).reshape(3, 10, 3) / 11.0
    down_weight = torch.arange(3 * 3 * 5, dtype=torch.float32).reshape(3, 3, 5) / 13.0
    model = model_cls.from_state_dict(
        {
            "experts.gate_up.weight": gate_up_weight,
            "experts.down.weight": down_weight,
        }
    )
    x = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3) / 5.0
    topk_scores = torch.tensor([[[0.75, 0.25], [0.4, 0.6]], [[0.2, 0.8], [0.55, 0.45]]], dtype=torch.float32)
    topk_indices = torch.tensor([[[0, 1], [2, 0]], [[1, 2], [0, 1]]], dtype=torch.long)

    actual = model.forward(x=x, topk_scores=topk_scores, topk_indices=topk_indices)
    expanded = x.unsqueeze(2).expand((*topk_indices.shape, x.shape[-1]))
    gate_up = torch.matmul(
        expanded.unsqueeze(-2),
        gate_up_weight[topk_indices].transpose(-1, -2),
    ).squeeze(-2)
    gate, up = torch.chunk(gate_up, 2, dim=-1)
    hidden = torch.nn.functional.silu(gate) * up
    values = torch.matmul(
        hidden.unsqueeze(-2),
        down_weight[topk_indices].transpose(-1, -2),
    ).squeeze(-2)
    expected = torch.sum(values * topk_scores.unsqueeze(-1), dim=2, keepdim=False)

    assert torch.allclose(actual, expected)


def test_graph_optimizer_rewrites_weighted_topk_sum_for_torch() -> None:
    values_type = _tensor("B", "T", "K", "D")
    scores_type = _tensor("B", "T", "K")
    output_type = _tensor("B", "T", "D")
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(
                    GraphValue("values", values_type, dims=values_type.dims),
                    GraphValue("scores", scores_type, dims=scores_type.dims),
                ),
                outputs=(GraphValueRef("out", output_type, dims=output_type.dims),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_unsqueeze"),
                        inputs=(
                            GraphValueRef("scores", scores_type, dims=scores_type.dims),
                            GraphLiteral(-1, TypeInt()),
                        ),
                        attrs={},
                        outputs=(GraphValue("scale", _tensor("B", "T", "K", 1), dims=("B", "T", "K", 1)),),
                        source_module="main",
                        type_expr=_tensor("B", "T", "K", 1),
                        dims=("B", "T", "K", 1),
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_mul"),
                        inputs=(
                            GraphValueRef("values", values_type, dims=values_type.dims),
                            GraphValueRef("scale", _tensor("B", "T", "K", 1), dims=("B", "T", "K", 1)),
                        ),
                        attrs={},
                        outputs=(GraphValue("weighted", values_type, dims=values_type.dims),),
                        source_module="main",
                        type_expr=values_type,
                        dims=values_type.dims,
                    ),
                    GraphNode(
                        id="main:3",
                        op=GraphOp("_sum"),
                        inputs=(
                            GraphValueRef("weighted", values_type, dims=values_type.dims),
                            GraphLiteral(2, TypeInt()),
                            GraphLiteral(False, TypeBool()),
                        ),
                        attrs={},
                        outputs=(GraphValue("out", output_type, dims=output_type.dims),),
                        source_module="main",
                        type_expr=output_type,
                        dims=output_type.dims,
                    ),
                ),
                return_type_expr=output_type,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(backend_intrinsics="codegen2-torch"),
    )
    main = next(module for module in optimized.modules if module.name == "main")
    assert [node.op.name for node in main.nodes] == ["__torch_weighted_topk_sum"]

    code = emit_model_code_from_graph_ir(optimized)
    assert "__torch_weighted_topk_sum" not in code
    namespace: dict[str, object] = {}
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]
    model = model_cls.from_state_dict({})
    values = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5) / 11.0
    scores = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) / 7.0
    actual = model.forward(values=values, scores=scores)
    expected = torch.sum(values * scores.unsqueeze(-1), dim=2, keepdim=False)
    assert torch.allclose(actual, expected)


def test_codegen2_torch_selected_expert_packed_swiglu_ffn() -> None:
    x_type = _tensor("B", "T", 3)
    scores_type = _tensor("B", "T", 2)
    idx_type = TypeTensor("IdxTensor", ("B", "T", 2))
    out_type = _tensor("B", "T", 4)
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(
                    GraphValue("x", x_type, dims=x_type.dims),
                    GraphValue("scores", scores_type, dims=scores_type.dims),
                    GraphValue("indices", idx_type, dims=idx_type.dims),
                ),
                outputs=(GraphValueRef("out", out_type, dims=out_type.dims),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("__torch_selected_expert_packed_swiglu_ffn"),
                        inputs=(
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphValueRef("scores", scores_type, dims=scores_type.dims),
                            GraphValueRef("indices", idx_type, dims=idx_type.dims),
                            GraphPath(True, ("experts", "gate_up", "weight")),
                            GraphPath(True, ("experts", "down", "weight")),
                            GraphLiteral(False, TypeBool()),
                        ),
                        attrs={},
                        outputs=(GraphValue("out", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                ),
                return_type_expr=out_type,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    code = emit_model_code_from_graph_ir(graph)
    namespace: dict[str, object] = {}
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]
    gate_up_weight = torch.arange(3 * 10 * 3, dtype=torch.float32).reshape(3, 10, 3) / 11.0
    down_weight = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5) / 13.0
    model = model_cls.from_state_dict(
        {
            "experts.gate_up.weight": gate_up_weight,
            "experts.down.weight": down_weight,
        }
    )
    x = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3) / 5.0
    scores = torch.tensor(
        [[[0.8, 0.2], [0.3, 0.7], [1.0, 0.0]], [[0.4, 0.6], [0.5, 0.5], [0.9, 0.1]]],
        dtype=torch.float32,
    )
    indices = torch.tensor([[[0, 1], [2, 0], [1, 2]], [[2, 1], [0, 2], [1, 0]]], dtype=torch.long)

    actual = model.forward(x=x, scores=scores, indices=indices)
    expanded = x.unsqueeze(2).expand((*indices.shape, x.shape[-1]))
    gate_up = torch.matmul(expanded.unsqueeze(-2), gate_up_weight[indices].transpose(-1, -2)).squeeze(-2)
    gate, up = torch.chunk(gate_up, 2, dim=-1)
    hidden = torch.nn.functional.silu(gate) * up
    values = torch.matmul(hidden.unsqueeze(-2), down_weight[indices].transpose(-1, -2)).squeeze(-2)
    expected = torch.sum(values * scores.unsqueeze(-1), dim=2)

    assert torch.allclose(actual, expected)


def test_codegen2_torch_selected_expert_swiglu_ffn() -> None:
    x_type = _tensor("B", "T", 3)
    scores_type = _tensor("B", "T", 2)
    idx_type = TypeTensor("IdxTensor", ("B", "T", 2))
    out_type = _tensor("B", "T", 4)
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(
                    GraphValue("x", x_type, dims=x_type.dims),
                    GraphValue("scores", scores_type, dims=scores_type.dims),
                    GraphValue("indices", idx_type, dims=idx_type.dims),
                ),
                outputs=(GraphValueRef("out", out_type, dims=out_type.dims),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("__torch_selected_expert_swiglu_ffn"),
                        inputs=(
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphValueRef("scores", scores_type, dims=scores_type.dims),
                            GraphValueRef("indices", idx_type, dims=idx_type.dims),
                            GraphPath(True, ("experts", "gate", "weight")),
                            GraphPath(True, ("experts", "up", "weight")),
                            GraphPath(True, ("experts", "down", "weight")),
                            GraphLiteral(False, TypeBool()),
                        ),
                        attrs={},
                        outputs=(GraphValue("out", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                ),
                return_type_expr=out_type,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    code = emit_model_code_from_graph_ir(graph)
    namespace: dict[str, object] = {}
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]
    gate_weight = torch.arange(3 * 5 * 3, dtype=torch.float32).reshape(3, 5, 3) / 11.0
    up_weight = torch.arange(3 * 5 * 3, dtype=torch.float32).reshape(3, 5, 3) / 17.0
    down_weight = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5) / 13.0
    model = model_cls.from_state_dict(
        {
            "experts.gate.weight": gate_weight,
            "experts.up.weight": up_weight,
            "experts.down.weight": down_weight,
        }
    )
    x = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3) / 5.0
    scores = torch.tensor(
        [[[0.8, 0.2], [0.3, 0.7], [1.0, 0.0]], [[0.4, 0.6], [0.5, 0.5], [0.9, 0.1]]],
        dtype=torch.float32,
    )
    indices = torch.tensor([[[0, 1], [2, 0], [1, 2]], [[2, 1], [0, 2], [1, 0]]], dtype=torch.long)

    actual = model.forward(x=x, scores=scores, indices=indices)
    expanded = x.unsqueeze(2).expand((*indices.shape, x.shape[-1]))
    gate = torch.matmul(expanded.unsqueeze(-2), gate_weight[indices].transpose(-1, -2)).squeeze(-2)
    up = torch.matmul(expanded.unsqueeze(-2), up_weight[indices].transpose(-1, -2)).squeeze(-2)
    hidden = torch.nn.functional.silu(gate) * up
    values = torch.matmul(hidden.unsqueeze(-2), down_weight[indices].transpose(-1, -2)).squeeze(-2)
    expected = torch.sum(values * scores.unsqueeze(-1), dim=2)

    assert torch.allclose(actual, expected)


def test_codegen2_torch_selected_expert_relu2_ffn() -> None:
    x_type = _tensor("B", "T", 3)
    scores_type = _tensor("B", "T", 2)
    idx_type = TypeTensor("IdxTensor", ("B", "T", 2))
    out_type = _tensor("B", "T", 4)
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(
                    GraphValue("x", x_type, dims=x_type.dims),
                    GraphValue("scores", scores_type, dims=scores_type.dims),
                    GraphValue("indices", idx_type, dims=idx_type.dims),
                ),
                outputs=(GraphValueRef("out", out_type, dims=out_type.dims),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("__torch_selected_expert_relu2_ffn"),
                        inputs=(
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphValueRef("scores", scores_type, dims=scores_type.dims),
                            GraphValueRef("indices", idx_type, dims=idx_type.dims),
                            GraphPath(True, ("experts", "up", "weight")),
                            GraphPath(True, ("experts", "down", "weight")),
                            GraphLiteral(False, TypeBool()),
                        ),
                        attrs={},
                        outputs=(GraphValue("out", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                ),
                return_type_expr=out_type,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    code = emit_model_code_from_graph_ir(graph)
    namespace: dict[str, object] = {}
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]
    up_weight = torch.arange(3 * 5 * 3, dtype=torch.float32).reshape(3, 5, 3) / 11.0
    down_weight = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5) / 13.0
    model = model_cls.from_state_dict(
        {
            "experts.up.weight": up_weight,
            "experts.down.weight": down_weight,
        }
    )
    x = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3) / 5.0
    scores = torch.tensor(
        [[[0.8, 0.2], [0.3, 0.7], [1.0, 0.0]], [[0.4, 0.6], [0.5, 0.5], [0.9, 0.1]]],
        dtype=torch.float32,
    )
    indices = torch.tensor([[[0, 1], [2, 0], [1, 2]], [[2, 1], [0, 2], [1, 0]]], dtype=torch.long)

    actual = model.forward(x=x, scores=scores, indices=indices)
    expanded = x.unsqueeze(2).expand((*indices.shape, x.shape[-1]))
    up = torch.matmul(expanded.unsqueeze(-2), up_weight[indices].transpose(-1, -2)).squeeze(-2)
    hidden = torch.relu(up) * torch.relu(up)
    values = torch.matmul(hidden.unsqueeze(-2), down_weight[indices].transpose(-1, -2)).squeeze(-2)
    expected = torch.sum(values * scores.unsqueeze(-1), dim=2)

    assert torch.allclose(actual, expected)


def test_codegen2_torch_selected_expert_packed_gegelu_ffn() -> None:
    x_type = _tensor("B", "T", 3)
    scores_type = _tensor("B", "T", 2)
    idx_type = TypeTensor("IdxTensor", ("B", "T", 2))
    out_type = _tensor("B", "T", 4)
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(
                    GraphValue("x", x_type, dims=x_type.dims),
                    GraphValue("scores", scores_type, dims=scores_type.dims),
                    GraphValue("indices", idx_type, dims=idx_type.dims),
                ),
                outputs=(GraphValueRef("out", out_type, dims=out_type.dims),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("__torch_selected_expert_packed_gegelu_ffn"),
                        inputs=(
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphValueRef("scores", scores_type, dims=scores_type.dims),
                            GraphValueRef("indices", idx_type, dims=idx_type.dims),
                            GraphPath(True, ("experts", "gate_up", "weight")),
                            GraphPath(True, ("experts", "gate_up", "bias")),
                            GraphPath(True, ("experts", "down", "weight")),
                            GraphPath(True, ("experts", "down", "bias")),
                            GraphLiteral(7.0, TypeFloat()),
                            GraphLiteral(1.3, TypeFloat()),
                            GraphLiteral(True, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                        ),
                        attrs={},
                        outputs=(GraphValue("out", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                ),
                return_type_expr=out_type,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    code = emit_model_code_from_graph_ir(graph)
    namespace: dict[str, object] = {}
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]
    gate_up_weight = torch.arange(3 * 10 * 3, dtype=torch.float32).reshape(3, 10, 3) / 11.0
    gate_up_bias = torch.arange(3 * 10, dtype=torch.float32).reshape(3, 10) / 17.0
    down_weight = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5) / 13.0
    down_bias = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4) / 19.0
    model = model_cls.from_state_dict(
        {
            "experts.gate_up.weight": gate_up_weight,
            "experts.gate_up.bias": gate_up_bias,
            "experts.down.weight": down_weight,
            "experts.down.bias": down_bias,
        }
    )
    x = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3) / 5.0
    scores = torch.tensor(
        [[[0.8, 0.2], [0.3, 0.7], [1.0, 0.0]], [[0.4, 0.6], [0.5, 0.5], [0.9, 0.1]]],
        dtype=torch.float32,
    )
    indices = torch.tensor([[[0, 1], [2, 0], [1, 2]], [[2, 1], [0, 2], [1, 0]]], dtype=torch.long)

    actual = model.forward(x=x, scores=scores, indices=indices)
    expanded = x.unsqueeze(2).expand((*indices.shape, x.shape[-1]))
    gate_up = torch.matmul(expanded.unsqueeze(-2), gate_up_weight[indices].transpose(-1, -2)).squeeze(-2)
    gate_up = gate_up + gate_up_bias[indices]
    gate = gate_up[..., ::2]
    up = gate_up[..., 1::2]
    gate = torch.clamp(gate, max=7.0)
    up = torch.clamp(up, min=-7.0, max=7.0)
    hidden = gate * torch.sigmoid(1.3 * gate) * (up + 1)
    values = torch.matmul(hidden.unsqueeze(-2), down_weight[indices].transpose(-1, -2)).squeeze(-2)
    values = values + down_bias[indices]
    expected = torch.sum(values * scores.unsqueeze(-1), dim=2)

    assert torch.allclose(actual, expected)


def test_codegen2_torch_selected_expert_clamped_packed_swiglu_ffn() -> None:
    x_type = _tensor("B", "T", 3)
    scores_type = _tensor("B", "T", 2)
    idx_type = TypeTensor("IdxTensor", ("B", "T", 2))
    out_type = _tensor("B", "T", 4)
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(
                    GraphValue("x", x_type, dims=x_type.dims),
                    GraphValue("scores", scores_type, dims=scores_type.dims),
                    GraphValue("indices", idx_type, dims=idx_type.dims),
                ),
                outputs=(GraphValueRef("out", out_type, dims=out_type.dims),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("__torch_selected_expert_clamped_packed_swiglu_ffn"),
                        inputs=(
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphValueRef("scores", scores_type, dims=scores_type.dims),
                            GraphValueRef("indices", idx_type, dims=idx_type.dims),
                            GraphPath(True, ("experts", "gate_up", "weight")),
                            GraphPath(True, ("experts", "down", "weight")),
                            GraphLiteral(7.0, TypeFloat()),
                            GraphLiteral(False, TypeBool()),
                        ),
                        attrs={},
                        outputs=(GraphValue("out", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                ),
                return_type_expr=out_type,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    code = emit_model_code_from_graph_ir(graph)
    namespace: dict[str, object] = {}
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]
    gate_up_weight = torch.arange(3 * 10 * 3, dtype=torch.float32).reshape(3, 10, 3) / 11.0
    down_weight = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5) / 13.0
    model = model_cls.from_state_dict(
        {
            "experts.gate_up.weight": gate_up_weight,
            "experts.down.weight": down_weight,
        }
    )
    x = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3) / 5.0
    scores = torch.tensor(
        [[[0.8, 0.2], [0.3, 0.7], [1.0, 0.0]], [[0.4, 0.6], [0.5, 0.5], [0.9, 0.1]]],
        dtype=torch.float32,
    )
    indices = torch.tensor([[[0, 1], [2, 0], [1, 2]], [[2, 1], [0, 2], [1, 0]]], dtype=torch.long)

    actual = model.forward(x=x, scores=scores, indices=indices)
    expanded = x.unsqueeze(2).expand((*indices.shape, x.shape[-1]))
    gate_up = torch.matmul(expanded.unsqueeze(-2), gate_up_weight[indices].transpose(-1, -2)).squeeze(-2)
    gate, up = torch.chunk(gate_up, 2, dim=-1)
    gate = torch.clamp(gate, max=7.0)
    up = torch.clamp(up, min=-7.0, max=7.0)
    hidden = torch.nn.functional.silu(gate) * up
    values = torch.matmul(hidden.unsqueeze(-2), down_weight[indices].transpose(-1, -2)).squeeze(-2)
    expected = torch.sum(values * scores.unsqueeze(-1), dim=2)

    assert torch.allclose(actual, expected)


def test_graph_optimizer_rewrites_topk_normalize_for_torch() -> None:
    probs_type = _tensor("B", "T", "E")
    weights_type = _tensor("B", "T", 4)
    denom_type = _tensor("B", "T", 1)
    idx_type = TypeTensor("IdxTensor", ("B", "T", 4))
    x_type = _tensor("B", "T", "D")
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(
                    GraphValue("probs", probs_type, dims=probs_type.dims),
                    GraphValue("x", x_type, dims=x_type.dims),
                ),
                outputs=(
                    GraphValueRef("out", weights_type, dims=weights_type.dims),
                    GraphValueRef("topk_indices", idx_type, dims=idx_type.dims),
                ),
                output_names=("out", "topk_indices"),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_topk"),
                        inputs=(
                            GraphValueRef("probs", probs_type, dims=probs_type.dims),
                            GraphLiteral(4, TypeDim()),
                            GraphLiteral(-1, TypeInt()),
                            GraphLiteral(True, TypeBool()),
                            GraphLiteral(False, TypeBool()),
                        ),
                        attrs={},
                        outputs=(
                            GraphValue("topk_weights", weights_type, dims=weights_type.dims),
                            GraphValue("topk_indices", idx_type, dims=idx_type.dims),
                        ),
                        source_module="main",
                        type_expr=TypeTuple((weights_type, idx_type)),
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("core.binary.-"),
                        inputs=(GraphLiteral(4, TypeDim()), GraphLiteral(1, TypeInt())),
                        attrs={},
                        outputs=(GraphValue("last_start", TypeDim()),),
                        source_module="main",
                        type_expr=TypeDim(),
                    ),
                    GraphNode(
                        id="main:3",
                        op=GraphOp("_cumsum"),
                        inputs=(
                            GraphValueRef("topk_weights", weights_type, dims=weights_type.dims),
                            GraphLiteral(-1, TypeInt()),
                        ),
                        attrs={},
                        outputs=(GraphValue("running", weights_type, dims=weights_type.dims),),
                        source_module="main",
                        type_expr=weights_type,
                        dims=weights_type.dims,
                    ),
                    GraphNode(
                        id="main:4",
                        op=GraphOp("_slice"),
                        inputs=(
                            GraphValueRef("running", weights_type, dims=weights_type.dims),
                            GraphLiteral(-1, TypeInt()),
                            GraphValueRef("last_start", TypeDim()),
                            GraphLiteral(4, TypeDim()),
                        ),
                        attrs={},
                        outputs=(GraphValue("denom", denom_type, dims=denom_type.dims),),
                        source_module="main",
                        type_expr=denom_type,
                        dims=denom_type.dims,
                    ),
                    GraphNode(
                        id="main:5",
                        op=GraphOp("core.binary./"),
                        inputs=(
                            GraphValueRef("topk_weights", weights_type, dims=weights_type.dims),
                            GraphValueRef("denom", denom_type, dims=denom_type.dims),
                        ),
                        attrs={},
                        outputs=(GraphValue("normalized", weights_type, dims=weights_type.dims),),
                        source_module="main",
                        type_expr=weights_type,
                        dims=weights_type.dims,
                    ),
                    GraphNode(
                        id="main:6",
                        op=GraphOp("_cast_like"),
                        inputs=(
                            GraphValueRef("normalized", weights_type, dims=weights_type.dims),
                            GraphValueRef("x", x_type, dims=x_type.dims),
                        ),
                        attrs={},
                        outputs=(GraphValue("out", weights_type, dims=weights_type.dims),),
                        source_module="main",
                        type_expr=weights_type,
                        dims=weights_type.dims,
                    ),
                ),
                return_type_expr=TypeTuple((weights_type, idx_type)),
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(
        graph,
        config=GraphOptimizeConfig(backend_intrinsics="codegen2-torch"),
    )
    main = next(module for module in optimized.modules if module.name == "main")
    assert any(node.op.name == "__torch_topk_normalize" for node in main.nodes)
    assert not any(node.op.name in {"_cumsum", "_slice"} for node in main.nodes)

    code = emit_model_code_from_graph_ir(optimized)
    assert "__torch_topk_normalize" not in code
    namespace: dict[str, object] = {}
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]
    model = model_cls.from_state_dict({})
    probs = torch.softmax(torch.arange(2 * 3 * 7, dtype=torch.float32).reshape(2, 3, 7) / 13.0, dim=-1)
    x = torch.zeros((2, 3, 5), dtype=torch.float16)
    result = model.forward(probs=probs, x=x)
    actual = result["out"] if isinstance(result, dict) else result[0]
    indices = result["topk_indices"] if isinstance(result, dict) else result[1]
    topk_weights, expected_indices = torch.topk(probs, 4, dim=-1, largest=True, sorted=False)
    expected = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(dtype=x.dtype)
    assert torch.equal(indices, expected_indices)
    assert torch.allclose(actual, expected)


def test_graph_provenance_preserves_destructured_tuple_wrapper_outputs() -> None:
    tensor_t = _tensor("B", "S", "E")
    weights_t = _tensor("B", "S", "K")
    indices_t = TypeTensor("IdxTensor", ("B", "S", "K"))
    wrapper = GraphModule(
        name="Wrapper.topk",
        inputs=(GraphValue("x", tensor_t, dims=tensor_t.dims),),
        outputs=(GraphValueRef("out", TypeTuple((weights_t, indices_t))),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="Wrapper.topk:1",
                op=GraphOp("_topk"),
                inputs=(
                    GraphValueRef("x", tensor_t, dims=tensor_t.dims),
                    GraphLiteral("K", TypeDim()),
                    GraphLiteral(-1, TypeInt()),
                    GraphLiteral(True, TypeBool()),
                    GraphLiteral(False, TypeBool()),
                ),
                attrs={},
                outputs=(GraphValue("out", TypeTuple((weights_t, indices_t))),),
                source_module="Wrapper.topk",
                type_expr=TypeTuple((weights_t, indices_t)),
            ),
        ),
        return_type_expr=TypeTuple((weights_t, indices_t)),
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", tensor_t, dims=tensor_t.dims),),
        outputs=(GraphValueRef("weights", weights_t, dims=weights_t.dims),),
        output_names=("weights",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("Wrapper.topk"),
                inputs=(GraphValueRef("x", tensor_t, dims=tensor_t.dims),),
                attrs={},
                outputs=(
                    GraphValue("weights", weights_t, dims=weights_t.dims),
                    GraphValue("indices", indices_t, dims=indices_t.dims),
                ),
                source_module="main",
                type_expr=TypeTuple((weights_t, indices_t)),
            ),
        ),
        return_type_expr=weights_t,
    )

    provenance = infer_graph_provenance(
        GraphProgram(modules=(wrapper, main), main_module="main", pragmas={"main": "main"})
    )
    local = provenance.module_local_provenance["main"]
    assert local["weights"].kind == "op"
    assert local["weights"].op == "_topk[0]"
    assert local["indices"].kind == "op"
    assert local["indices"].op == "_topk[1]"


def test_graph_optimizer_preserves_expanded_packed_gegelu_internals() -> None:
    x_type = _tensor("B", "T", "K", 10)
    pair_type = _tensor("B", "T", "K", 5, 2)
    one_type = _tensor("B", "T", "K", 5, 1)
    out_type = _tensor("B", "T", "K", 5)
    shape_pair = GraphLiteral((2, 3, 4, 5, 2), TypeList(TypeDim()))
    shape_out = GraphLiteral((2, 3, 4, 5), TypeList(TypeDim()))
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(
                    GraphValue("x", x_type, dims=x_type.dims),
                    GraphValue("limit", TypeFloat()),
                ),
                outputs=(GraphValueRef("y", out_type, dims=out_type.dims),),
                output_names=("y",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_reshape"),
                        inputs=(GraphValueRef("x", x_type, dims=x_type.dims), shape_pair),
                        attrs={},
                        outputs=(GraphValue("pair", pair_type, dims=pair_type.dims),),
                        source_module="main",
                        type_expr=pair_type,
                        dims=pair_type.dims,
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_slice"),
                        inputs=(
                            GraphValueRef("pair", pair_type, dims=pair_type.dims),
                            GraphLiteral(-1, TypeInt()),
                            GraphLiteral(0, TypeDim()),
                            GraphLiteral(1, TypeDim()),
                        ),
                        attrs={},
                        outputs=(GraphValue("gate_slice", one_type, dims=one_type.dims),),
                        source_module="main",
                        type_expr=one_type,
                        dims=one_type.dims,
                    ),
                    GraphNode(
                        id="main:3",
                        op=GraphOp("_reshape"),
                        inputs=(GraphValueRef("gate_slice", one_type, dims=one_type.dims), shape_out),
                        attrs={},
                        outputs=(GraphValue("gate", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                    GraphNode(
                        id="main:4",
                        op=GraphOp("_slice"),
                        inputs=(
                            GraphValueRef("pair", pair_type, dims=pair_type.dims),
                            GraphLiteral(-1, TypeInt()),
                            GraphLiteral(1, TypeDim()),
                            GraphLiteral(2, TypeDim()),
                        ),
                        attrs={},
                        outputs=(GraphValue("up_slice", one_type, dims=one_type.dims),),
                        source_module="main",
                        type_expr=one_type,
                        dims=one_type.dims,
                    ),
                    GraphNode(
                        id="main:5",
                        op=GraphOp("_reshape"),
                        inputs=(GraphValueRef("up_slice", one_type, dims=one_type.dims), shape_out),
                        attrs={},
                        outputs=(GraphValue("up", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                    GraphNode(
                        id="main:6",
                        op=GraphOp("_clamp"),
                        inputs=(
                            GraphValueRef("gate", out_type, dims=out_type.dims),
                            GraphLiteral(None, TypeNull()),
                            GraphValueRef("limit", TypeFloat()),
                        ),
                        attrs={},
                        outputs=(GraphValue("gate_c", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                    GraphNode(
                        id="main:7",
                        op=GraphOp("core.binary.-"),
                        inputs=(GraphLiteral(0.0, TypeFloat()), GraphValueRef("limit", TypeFloat())),
                        attrs={},
                        outputs=(GraphValue("neg_limit", TypeFloat()),),
                        source_module="main",
                        type_expr=TypeFloat(),
                    ),
                    GraphNode(
                        id="main:8",
                        op=GraphOp("_clamp"),
                        inputs=(
                            GraphValueRef("up", out_type, dims=out_type.dims),
                            GraphValueRef("neg_limit", TypeFloat()),
                            GraphValueRef("limit", TypeFloat()),
                        ),
                        attrs={},
                        outputs=(GraphValue("up_c", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                    GraphNode(
                        id="main:9",
                        op=GraphOp("core.binary.*"),
                        inputs=(GraphValueRef("gate_c", out_type, dims=out_type.dims), GraphLiteral(1.702, TypeFloat())),
                        attrs={},
                        outputs=(GraphValue("gate_alpha", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                    GraphNode(
                        id="main:10",
                        op=GraphOp("_activations_sigmoid"),
                        inputs=(GraphValueRef("gate_alpha", out_type, dims=out_type.dims),),
                        attrs={},
                        outputs=(GraphValue("sigmoid", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                    GraphNode(
                        id="main:11",
                        op=GraphOp("core.binary.*"),
                        inputs=(
                            GraphValueRef("gate_c", out_type, dims=out_type.dims),
                            GraphValueRef("sigmoid", out_type, dims=out_type.dims),
                        ),
                        attrs={},
                        outputs=(GraphValue("gate_term", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                    GraphNode(
                        id="main:12",
                        op=GraphOp("core.binary.+"),
                        inputs=(GraphValueRef("up_c", out_type, dims=out_type.dims), GraphLiteral(1.0, TypeFloat())),
                        attrs={},
                        outputs=(GraphValue("up_plus", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                    GraphNode(
                        id="main:13",
                        op=GraphOp("core.binary.*"),
                        inputs=(
                            GraphValueRef("up_plus", out_type, dims=out_type.dims),
                            GraphValueRef("gate_term", out_type, dims=out_type.dims),
                        ),
                        attrs={},
                        outputs=(GraphValue("y", out_type, dims=out_type.dims),),
                        source_module="main",
                        type_expr=out_type,
                        dims=out_type.dims,
                    ),
                ),
                return_type_expr=out_type,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(graph, config=GraphOptimizeConfig())
    main = next(module for module in optimized.modules if module.name == "main")
    assert not any(node.op.name == "_activations_gegelu" for node in main.nodes)
    assert any(node.op.name == "_activations_sigmoid" for node in main.nodes)
    assert any(node.op.name == "_clamp" for node in main.nodes)

    code = emit_model_code_from_graph_ir(optimized)
    namespace: dict[str, object] = {}
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]
    model = model_cls.from_state_dict({})
    x = torch.linspace(-2.0, 2.0, steps=2 * 3 * 4 * 10).reshape(2, 3, 4, 10)
    actual = model.forward(x=x, limit=1.5)
    pair = x.reshape(2, 3, 4, 5, 2)
    gate = pair[..., 0].clamp(max=1.5)
    up = pair[..., 1].clamp(min=-1.5, max=1.5)
    expected = (up + 1.0) * (gate * torch.sigmoid(gate * 1.702))
    assert torch.allclose(actual, expected)


def test_codegen2_static_embedding_param_key_uses_graph_path_metadata_only() -> None:
    x_type = _tensor("B", "S")
    y_type = _tensor("B", "S", "D")
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("x", x_type, dims=x_type.dims),),
                outputs=(GraphValueRef("y", y_type, dims=y_type.dims),),
                output_names=("y",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_embedding"),
                        inputs=(GraphPath(True, ("wte",)), GraphValueRef("x", x_type, dims=x_type.dims), GraphLiteral(None, TypeNull())),
                        attrs={},
                        outputs=(GraphValue("y", y_type, dims=y_type.dims),),
                        source_module="main",
                        type_expr=y_type,
                        dims=y_type.dims,
                    ),
                ),
                return_type_expr=y_type,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    code = emit_model_code_from_graph_ir(graph)

    assert "y = F.embedding(x, self.state_dict_tensors['wte.weight'])" in code


def test_codegen2_templated_embedding_param_key_keeps_dynamic_path_resolution() -> None:
    int_t = TypeInt()
    x_type = _tensor("B", "S")
    y_type = _tensor("B", "S", "D")
    graph = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("i", int_t), GraphValue("x", x_type, dims=x_type.dims)),
                outputs=(GraphValueRef("y", y_type, dims=y_type.dims),),
                output_names=("y",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_embedding"),
                        inputs=(GraphPath(True, ("h.{i}",)), GraphValueRef("x", x_type, dims=x_type.dims), GraphLiteral(None, TypeNull())),
                        attrs={},
                        outputs=(GraphValue("y", y_type, dims=y_type.dims),),
                        source_module="main",
                        type_expr=y_type,
                        dims=y_type.dims,
                    ),
                ),
                return_type_expr=y_type,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    code = emit_model_code_from_graph_ir(graph)

    assert "self.state_dict_tensors['h.{i}.weight']" not in code
    assert "y = F.embedding(x, self._param(" in code
    assert "_path_template_part(i)" in code


def test_codegen2_tinygrad_generated_expert_linear_matches_torch() -> None:
    pytest.importorskip("tinygrad")
    if not torch.cuda.is_available() and shutil.which("clang") is None:
        pytest.skip("tinygrad CPU backend needs clang")
    graph = _expert_linear_graph()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight = torch.arange(3 * 5 * 4, dtype=torch.float32, device=device).reshape(3, 5, 4) / 10.0
    x = torch.arange(2 * 2 * 2 * 4, dtype=torch.float32, device=device).reshape(2, 2, 2, 4) / 7.0
    expert_idx = torch.tensor([[[0, 1], [2, 0]], [[1, 2], [0, 1]]], dtype=torch.long, device=device)
    namespace: dict[str, object] = {}
    exec(emit_tinygrad_model_code_from_graph_ir(graph), namespace)
    model_cls = namespace["AxonTinygradModel"]
    model = model_cls.from_state_dict({"experts.weight": weight})

    actual = model.forward(x=x, expert_idx=expert_idx)
    expected = torch.matmul(
        x.unsqueeze(-2),
        weight[expert_idx].transpose(-1, -2),
    ).squeeze(-2)

    assert torch.allclose(actual.to(expected.device), expected, atol=1e-5, rtol=1e-5)


def test_codegen2_tinygrad_setup_stacks_individual_expert_weights() -> None:
    pytest.importorskip("tinygrad")
    if not torch.cuda.is_available() and shutil.which("clang") is None:
        pytest.skip("tinygrad CPU backend needs clang")
    graph = _expert_linear_graph()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight = torch.arange(3 * 5 * 4, dtype=torch.float32, device=device).reshape(3, 5, 4) / 10.0
    x = torch.arange(2 * 2 * 2 * 4, dtype=torch.float32, device=device).reshape(2, 2, 2, 4) / 7.0
    expert_idx = torch.tensor([[[0, 1], [2, 0]], [[1, 2], [0, 1]]], dtype=torch.long, device=device)
    namespace: dict[str, object] = {}
    exec(emit_tinygrad_model_code_from_graph_ir(graph), namespace)
    model_cls = namespace["AxonTinygradModel"]
    model = model_cls.from_state_dict(
        {f"experts.{idx}.weight": expert_weight for idx, expert_weight in enumerate(weight)}
    )

    actual = model.forward(x=x, expert_idx=expert_idx)
    expected = torch.matmul(
        x.unsqueeze(-2),
        weight[expert_idx].transpose(-1, -2),
    ).squeeze(-2)

    assert torch.allclose(actual.to(expected.device), expected, atol=1e-5, rtol=1e-5)


def _single_primitive_graph(
    primitive: str,
    inputs: tuple[GraphValue, ...],
    node_inputs: tuple[object, ...],
    output_type: TypeTensor,
) -> GraphProgram:
    main = GraphModule(
        name="main",
        inputs=inputs,
        outputs=(GraphValueRef("y", output_type),),
        output_names=("y",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp(primitive),
                inputs=node_inputs,
                attrs={},
                outputs=(GraphValue("y", output_type),),
                source_module="main",
                type_expr=output_type,
            ),
        ),
        return_type_expr=output_type,
    )
    return GraphProgram(modules=(main,), main_module="main", pragmas={"main": "main"})


def test_codegen2_tinygrad_gelu_matches_torch_exact_gelu() -> None:
    pytest.importorskip("tinygrad")
    if not torch.cuda.is_available() and shutil.which("clang") is None:
        pytest.skip("tinygrad CPU backend needs clang")
    tensor_type = _tensor("B", "S")
    graph = _single_primitive_graph(
        "_activations_gelu",
        (GraphValue("x", tensor_type),),
        (GraphValueRef("x", tensor_type),),
        tensor_type,
    )
    namespace: dict[str, object] = {}
    exec(emit_tinygrad_model_code_from_graph_ir(graph), namespace)
    model = namespace["AxonTinygradModel"].from_state_dict({})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.linspace(-6, 6, 97, dtype=torch.float32, device=device).reshape(1, 97)

    actual = model.forward(x=x)
    expected = torch.nn.functional.gelu(x)

    assert torch.allclose(actual.to(expected.device), expected, atol=1e-6, rtol=1e-6)


def test_codegen2_tinygrad_softmax_honors_dtype_argument() -> None:
    pytest.importorskip("tinygrad")
    if not torch.cuda.is_available() and shutil.which("clang") is None:
        pytest.skip("tinygrad CPU backend needs clang")
    tensor_type = _tensor("B", "S")
    graph = _single_primitive_graph(
        "_softmax",
        (GraphValue("x", tensor_type),),
        (
            GraphValueRef("x", tensor_type),
            GraphLiteral(-1, TypeInt()),
            GraphLiteral("float32", TypeString()),
        ),
        tensor_type,
    )
    namespace: dict[str, object] = {}
    exec(emit_tinygrad_model_code_from_graph_ir(graph), namespace)
    model = namespace["AxonTinygradModel"].from_state_dict({})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    x = (torch.randn(3, 11, dtype=torch.float16, device=device) * 8).contiguous()

    actual = model.forward(x=x)
    expected = torch.softmax(x.float(), dim=-1)

    assert actual.dtype == torch.float32
    assert torch.allclose(actual.to(expected.device), expected, atol=1e-4, rtol=1e-4)


def test_codegen2_tinygrad_materializes_final_logits_bias_flat_alias() -> None:
    pytest.importorskip("tinygrad")
    if not torch.cuda.is_available() and shutil.which("clang") is None:
        pytest.skip("tinygrad CPU backend needs clang")
    x_type = _tensor("B", "D")
    y_type = _tensor("B", "V")
    graph = _single_primitive_graph(
        "_linear",
        (GraphValue("x", x_type),),
        (
            GraphPath(absolute=True, parts=("shared",)),
            GraphValueRef("x", x_type),
            GraphLiteral(5, TypeDim()),
            GraphLiteral(True, TypeBool()),
            GraphLiteral(False, TypeBool()),
            GraphLiteral(None, TypeNull()),
            GraphLiteral("@weight", TypePath()),
            GraphLiteral("@@final_logits_bias_flat", TypePath()),
        ),
        y_type,
    )
    namespace: dict[str, object] = {}
    exec(emit_tinygrad_model_code_from_graph_ir(graph), namespace)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight = torch.arange(5 * 4, dtype=torch.float32, device=device).reshape(5, 4) / 13.0
    bias = torch.arange(5, dtype=torch.float32, device=device).reshape(1, 5) / 7.0
    x = torch.arange(3 * 4, dtype=torch.float32, device=device).reshape(3, 4) / 11.0
    model = namespace["AxonTinygradModel"].from_state_dict(
        {
            "shared.weight": weight,
            "final_logits_bias": bias,
        }
    )

    actual = model.forward(x=x)
    expected = torch.nn.functional.linear(x, weight, bias.reshape(5))

    assert torch.allclose(actual.to(expected.device), expected, atol=1e-6, rtol=1e-6)


def test_codegen2_mlx_emits_gpt2_source() -> None:
    pytest.importorskip("mlx")
    program = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2.axon")
    ).ast

    graph = lower_axon_program_to_graph_ir(_typed(program, main_module="gpt2"), main_module="gpt2")
    table = mlx_op_table_markdown(graph)
    code = emit_mlx_model_code_from_graph_ir(graph, class_name="MlxGPT2")

    assert "| Op | Count | Reason |" in table
    assert "`embedding`" not in table
    assert "import mlx.core as mx" in code
    assert "import mlx.nn as nn" in code
    assert "class MlxGPT2(nn.Module):" in code
    assert "def _forward(self, input_ids=None, **inputs):" in code
    assert "def forward(self, input_ids=None, **inputs):" in code


def test_codegen2_mlx_generated_expert_linear_matches_torch() -> None:
    pytest.importorskip("mlx")
    graph = _expert_linear_graph()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight = torch.arange(3 * 5 * 4, dtype=torch.float32, device=device).reshape(3, 5, 4) / 10.0
    x = torch.arange(2 * 2 * 2 * 4, dtype=torch.float32, device=device).reshape(2, 2, 2, 4) / 7.0
    expert_idx_t = torch.tensor([[[0, 1], [2, 0]], [[1, 2], [0, 1]]], dtype=torch.long, device=device)
    namespace: dict[str, object] = {}
    exec(emit_mlx_model_code_from_graph_ir(graph), namespace)
    model_cls = namespace["AxonMlxModel"]
    mlx_weight = torch_state_dict_to_mlx({"experts.weight": weight})
    model = model_cls.from_state_dict(mlx_weight)

    expert_idx = torch_state_dict_to_mlx({"idx": expert_idx_t})["idx"]
    actual_np = model.forward(x=torch_state_dict_to_mlx({"x": x})["x"], expert_idx=expert_idx)
    actual = torch.from_numpy(np.asarray(actual_np))

    expected = torch.matmul(
        x.unsqueeze(-2),
        weight[expert_idx_t].transpose(-1, -2),
    ).squeeze(-2)

    assert torch.allclose(actual.to(expected.device), expected, atol=1e-5, rtol=1e-5)


def test_codegen2_mlx_setup_stacks_individual_expert_weights() -> None:
    pytest.importorskip("mlx")
    graph = _expert_linear_graph()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight = torch.arange(3 * 5 * 4, dtype=torch.float32, device=device).reshape(3, 5, 4) / 10.0
    x = torch.arange(2 * 2 * 2 * 4, dtype=torch.float32, device=device).reshape(2, 2, 2, 4) / 7.0
    expert_idx_t = torch.tensor([[[0, 1], [2, 0]], [[1, 2], [0, 1]]], dtype=torch.long, device=device)
    namespace: dict[str, object] = {}
    exec(emit_mlx_model_code_from_graph_ir(graph), namespace)
    model_cls = namespace["AxonMlxModel"]
    mlx_state = torch_state_dict_to_mlx(
        {f"experts.{idx}.weight": expert_weight for idx, expert_weight in enumerate(weight)}
    )
    model = model_cls.from_state_dict(mlx_state)

    expert_idx = torch_state_dict_to_mlx({"idx": expert_idx_t})["idx"]
    actual_np = model.forward(x=torch_state_dict_to_mlx({"x": x})["x"], expert_idx=expert_idx)
    actual = torch.from_numpy(np.asarray(actual_np))

    expected = torch.matmul(
        x.unsqueeze(-2),
        weight[expert_idx_t].transpose(-1, -2),
    ).squeeze(-2)

    assert torch.allclose(actual.to(expected.device), expected, atol=1e-5, rtol=1e-5)


def test_codegen2_mlx_gelu_matches_torch_exact_gelu() -> None:
    pytest.importorskip("mlx")
    tensor_type = _tensor("B", "S")
    graph = _single_primitive_graph(
        "_activations_gelu",
        (GraphValue("x", tensor_type),),
        (GraphValueRef("x", tensor_type),),
        tensor_type,
    )
    namespace: dict[str, object] = {}
    exec(emit_mlx_model_code_from_graph_ir(graph), namespace)
    model = namespace["AxonMlxModel"].from_state_dict({})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_t = torch.linspace(-6, 6, 97, dtype=torch.float32, device=device).reshape(1, 97)

    actual_np = model.forward(x=torch_state_dict_to_mlx({"x": x_t})["x"])
    actual = torch.from_numpy(np.asarray(actual_np))

    expected = torch.nn.functional.gelu(x_t)

    assert torch.allclose(actual.to(expected.device), expected, atol=1e-6, rtol=1e-6)


def test_codegen2_mlx_compiled_forward_matches_uncompiled() -> None:
    pytest.importorskip("mlx")
    import mlx.core as mx
    tensor_type = _tensor("B", "S")
    graph = _single_primitive_graph(
        "_activations_gelu",
        (GraphValue("x", tensor_type),),
        (GraphValueRef("x", tensor_type),),
        tensor_type,
    )
    namespace: dict[str, object] = {}
    exec(emit_mlx_model_code_from_graph_ir(graph), namespace)
    model = namespace["AxonMlxModel"].from_state_dict({})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_t = torch.linspace(-6, 6, 97, dtype=torch.float32, device=device).reshape(1, 97)
    x = torch_state_dict_to_mlx({"x": x_t})["x"]

    actual_np = model.forward(x)
    actual = torch.from_numpy(np.asarray(actual_np))

    model.forward = mx.compile(model.forward)
    compiled_np = model.forward(x)
    compiled = torch.from_numpy(np.asarray(compiled_np))

    assert torch.allclose(compiled.to(actual.device), actual, atol=1e-6, rtol=1e-6)


def test_codegen2_mlx_softmax_honors_dtype_argument() -> None:
    pytest.importorskip("mlx")
    tensor_type = _tensor("B", "S")
    graph = _single_primitive_graph(
        "_softmax",
        (GraphValue("x", tensor_type),),
        (
            GraphValueRef("x", tensor_type),
            GraphLiteral(-1, TypeInt()),
            GraphLiteral("float32", TypeString()),
        ),
        tensor_type,
    )
    namespace: dict[str, object] = {}
    exec(emit_mlx_model_code_from_graph_ir(graph), namespace)
    model = namespace["AxonMlxModel"].from_state_dict({})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    x_t = (torch.randn(3, 11, dtype=torch.float16, device=device) * 8).contiguous()

    actual_np = model.forward(x=torch_state_dict_to_mlx({"x": x_t})["x"])
    actual = torch.from_numpy(np.asarray(actual_np))

    expected = torch.softmax(x_t.float(), dim=-1)

    assert actual.dtype == torch.float32
    assert torch.allclose(actual.to(expected.device), expected, atol=1e-4, rtol=1e-4)


def test_codegen2_mlx_materializes_final_logits_bias_flat_alias() -> None:
    pytest.importorskip("mlx")
    x_type = _tensor("B", "D")
    y_type = _tensor("B", "V")
    graph = _single_primitive_graph(
        "_linear",
        (GraphValue("x", x_type),),
        (
            GraphPath(absolute=True, parts=("shared",)),
            GraphValueRef("x", x_type),
            GraphLiteral(5, TypeDim()),
            GraphLiteral(True, TypeBool()),
            GraphLiteral(False, TypeBool()),
            GraphLiteral(None, TypeNull()),
            GraphLiteral("@weight", TypePath()),
            GraphLiteral("@@final_logits_bias_flat", TypePath()),
        ),
        y_type,
    )
    namespace: dict[str, object] = {}
    exec(emit_mlx_model_code_from_graph_ir(graph), namespace)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_t = torch.arange(5 * 4, dtype=torch.float32, device=device).reshape(5, 4) / 13.0
    bias_t = torch.arange(5, dtype=torch.float32, device=device).reshape(1, 5) / 7.0
    x_t = torch.arange(3 * 4, dtype=torch.float32, device=device).reshape(3, 4) / 11.0
    state = torch_state_dict_to_mlx({
        "shared.weight": weight_t,
        "final_logits_bias": bias_t,
    })
    model = namespace["AxonMlxModel"].from_state_dict(state)

    actual_np = model.forward(x=torch_state_dict_to_mlx({"x": x_t})["x"])
    actual = torch.from_numpy(np.asarray(actual_np))

    expected = torch.nn.functional.linear(x_t, weight_t, bias_t.reshape(5))

    assert torch.allclose(actual.to(expected.device), expected, atol=1e-6, rtol=1e-6)


def test_codegen2_mlx_torch_state_dict_conversion() -> None:
    pytest.importorskip("mlx")
    import mlx.core as mx

    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    result = torch_state_dict_to_mlx({"w": t})
    assert "w" in result
    assert isinstance(result["w"], mx.array)
    assert result["w"].shape == (3, 4)
    assert result["w"].dtype == mx.float32


def test_graph_ir_preserves_list_destructuring_outputs() -> None:
    program = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/gpt2.axon")
    ).ast
    typed = _typed(program, main_module="gpt2")

    graph = lower_axon_program_to_graph_ir(typed, main_module="gpt2")
    chunk_nodes = [
        node
        for module in graph.modules
        for node in module.nodes
        if node.op.name in {"Tensor.chunk", "_chunk"}
    ]

    assert chunk_nodes
    assert any(len(node.outputs) == 3 for node in chunk_nodes)


def _tensor(*dims: str) -> TypeTensor:
    return TypeTensor("Tensor", dims)


def _toy_program(inputs: tuple[GraphValue, ...], output_names: tuple[str, ...]) -> GraphProgram:
    outputs = (GraphValueRef(inputs[0].name, inputs[0].type_expr),)
    if len(output_names) > 1:
        outputs = (*outputs, GraphValueRef(inputs[-2].name, inputs[-2].type_expr))
    return GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=inputs,
                outputs=outputs,
                output_names=output_names,
                nodes=(),
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )


def test_codegen2_tensor_size_uses_runtime_shape_without_graph_optimization() -> None:
    x_type = _tensor("B", "S")
    program = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("x", x_type, dims=x_type.dims),),
                outputs=(GraphValueRef("s", TypeDim()),),
                output_names=("s",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_tensor_size"),
                        inputs=(
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(1, TypeInt()),
                        ),
                        attrs={},
                        outputs=(GraphValue("s", TypeDim()),),
                        source_module="main",
                        type_expr=TypeDim(),
                    ),
                ),
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    code = emit_model_code_from_graph_ir(program)

    assert "s = x.shape[1]" in code


def test_codegen2_tensor_size_uses_runtime_shape_for_unbound_result_dim() -> None:
    x_type = _tensor("B", "S")
    token_idx_type = _tensor("N")
    program = GraphProgram(
        modules=(
            GraphModule(
                name="N",
                inputs=(),
                outputs=(GraphLiteral(16, TypeInt()),),
                output_names=("N",),
                nodes=(),
                return_type_expr=TypeInt(),
                is_global_binding=True,
            ),
            GraphModule(
                name="main",
                inputs=(GraphValue("x", x_type, dims=x_type.dims),),
                outputs=(GraphValueRef("n", TypeDim()),),
                output_names=("n",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_where_indices"),
                        inputs=(GraphValueRef("x", x_type, dims=x_type.dims),),
                        attrs={},
                        outputs=(GraphValue("token_idx", token_idx_type, dims=token_idx_type.dims),),
                        source_module="main",
                        type_expr=token_idx_type,
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_tensor_size"),
                        inputs=(
                            GraphValueRef("token_idx", token_idx_type, dims=token_idx_type.dims),
                            GraphLiteral(0, TypeInt()),
                        ),
                        attrs={},
                        outputs=(GraphValue("n", TypeDim()),),
                        source_module="main",
                        type_expr=TypeDim(),
                    ),
                ),
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    code = emit_model_code_from_graph_ir(program)

    assert "n = self._symbols['N']" not in code
    assert "n = token_idx.shape[0]" in code


def test_graph_optimizer_folds_tensor_size_of_stable_module_input() -> None:
    x_type = _tensor("B", "S")
    program = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("x", x_type, dims=x_type.dims),),
                outputs=(GraphValueRef("s", TypeDim()),),
                output_names=("s",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_tensor_size"),
                        inputs=(
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(1, TypeInt()),
                        ),
                        attrs={},
                        outputs=(GraphValue("s", TypeDim()),),
                        source_module="main",
                        type_expr=TypeDim(),
                    ),
                ),
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(
        program,
        config=GraphOptimizeConfig(
            atomic_alias_cleanup=True,
            dead_temp_elimination=True,
            constant_folding=True,
            constant_dim_substitution=False,
            common_subexpression_elimination=False,
            specialize_definitions="off",
            inline_safe=False,
        ),
    )

    main = optimized.modules[0]
    assert not main.nodes
    assert main.outputs == (GraphValueRef("S", TypeDim()),)


def test_graph_optimizer_folds_structural_tensor_size_forwarder_without_name_special_case() -> None:
    x_type = _tensor("B", "S")
    program = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("x", x_type, dims=x_type.dims),),
                outputs=(GraphValueRef("s", TypeDim()),),
                output_names=("s",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("Arbitrary.forwarder"),
                        inputs=(
                            GraphValueRef("x", x_type, dims=x_type.dims),
                            GraphLiteral(1, TypeInt()),
                        ),
                        attrs={},
                        outputs=(GraphValue("s", TypeDim()),),
                        source_module="main",
                        type_expr=TypeDim(),
                    ),
                ),
            ),
            GraphModule(
                name="Arbitrary.forwarder",
                inputs=(GraphValue("value", x_type, dims=x_type.dims), GraphValue("axis", TypeInt())),
                outputs=(GraphValueRef("out", TypeDim()),),
                output_names=("out",),
                nodes=(
                    GraphNode(
                        id="Arbitrary.forwarder:1",
                        op=GraphOp("_tensor_size"),
                        inputs=(GraphValueRef("value", x_type, dims=x_type.dims), GraphValueRef("axis", TypeInt())),
                        attrs={},
                        outputs=(GraphValue("out", TypeDim()),),
                        source_module="Arbitrary.forwarder",
                        type_expr=TypeDim(),
                    ),
                ),
                return_type_expr=TypeDim(),
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(
        program,
        config=GraphOptimizeConfig(
            atomic_alias_cleanup=True,
            dead_temp_elimination=True,
            constant_folding=True,
            constant_dim_substitution=False,
            common_subexpression_elimination=False,
            specialize_definitions="off",
            inline_safe=False,
        ),
    )

    main = next(module for module in optimized.modules if module.name == "main")
    assert not main.nodes
    assert main.outputs == (GraphValueRef("S", TypeDim()),)


def test_graph_optimizer_does_not_fold_tensor_size_of_value_dependent_intermediate() -> None:
    mask_type = _tensor(2, "BT")
    stale_idx_type = _tensor(16)
    program = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(GraphValue("mask", mask_type, dims=mask_type.dims),),
                outputs=(GraphValueRef("n", TypeDim()),),
                output_names=("n",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_where_indices"),
                        inputs=(GraphValueRef("mask", mask_type, dims=mask_type.dims),),
                        attrs={},
                        outputs=(
                            GraphValue("topk_pos", stale_idx_type, dims=stale_idx_type.dims),
                            GraphValue("token_idx", stale_idx_type, dims=stale_idx_type.dims),
                        ),
                        source_module="main",
                        type_expr=TypeTuple((stale_idx_type, stale_idx_type)),
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_tensor_size"),
                        inputs=(
                            GraphValueRef("token_idx", stale_idx_type, dims=stale_idx_type.dims),
                            GraphLiteral(0, TypeInt()),
                        ),
                        attrs={},
                        outputs=(GraphValue("n", TypeDim()),),
                        source_module="main",
                        type_expr=TypeDim(),
                    ),
                ),
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(
        program,
        config=GraphOptimizeConfig(
            atomic_alias_cleanup=True,
            dead_temp_elimination=True,
            constant_folding=True,
            constant_dim_substitution=False,
            common_subexpression_elimination=False,
            specialize_definitions="off",
            inline_safe=False,
        ),
    )

    main = optimized.modules[0]
    assert any(node.op.name == "_tensor_size" for node in main.nodes)
    assert main.outputs == (GraphValueRef("n", TypeDim()),)


def test_graph_optimizer_preserves_value_dependent_primitive_dims() -> None:
    mask_type = _tensor(2, "BT")
    stale_idx_type = _tensor(16)
    x_type = _tensor("BT", "D")
    stale_selected_type = _tensor(16, "D")
    program = GraphProgram(
        modules=(
            GraphModule(
                name="main",
                inputs=(
                    GraphValue("mask", mask_type, dims=mask_type.dims),
                    GraphValue("x", x_type, dims=x_type.dims),
                ),
                outputs=(GraphValueRef("selected", stale_selected_type, dims=stale_selected_type.dims),),
                output_names=("selected",),
                nodes=(
                    GraphNode(
                        id="main:1",
                        op=GraphOp("_where_indices"),
                        inputs=(GraphValueRef("mask", mask_type, dims=mask_type.dims),),
                        attrs={},
                        outputs=(
                            GraphValue("topk_pos", stale_idx_type, dims=stale_idx_type.dims),
                            GraphValue("token_idx", stale_idx_type, dims=stale_idx_type.dims),
                        ),
                        source_module="main",
                        type_expr=TypeTuple((stale_idx_type, stale_idx_type)),
                    ),
                    GraphNode(
                        id="main:2",
                        op=GraphOp("_tensor_size"),
                        inputs=(
                            GraphValueRef("token_idx", stale_idx_type, dims=stale_idx_type.dims),
                            GraphLiteral(0, TypeInt()),
                        ),
                        attrs={},
                        outputs=(GraphValue("n", TypeDim()),),
                        source_module="main",
                        type_expr=TypeDim(),
                    ),
                    GraphNode(
                        id="main:3",
                        op=GraphOp("_expand"),
                        inputs=(
                            GraphValueRef("token_idx", stale_idx_type, dims=stale_idx_type.dims),
                            GraphExpr(
                                op=GraphOp("core.list"),
                                inputs=(GraphValueRef("n", TypeDim()), GraphValueRef("D", TypeDim())),
                                attrs={},
                                type_expr=TypeList(TypeDim()),
                            ),
                        ),
                        attrs={},
                        outputs=(GraphValue("selected", stale_selected_type, dims=stale_selected_type.dims),),
                        source_module="main",
                        type_expr=stale_selected_type,
                        dims=stale_selected_type.dims,
                    ),
                ),
                return_type_expr=stale_selected_type,
            ),
        ),
        main_module="main",
        pragmas={"main": "main"},
    )

    optimized = optimize_graph_program(
        program,
        config=GraphOptimizeConfig(
            atomic_alias_cleanup=False,
            dead_temp_elimination=False,
            constant_folding=False,
            constant_dim_substitution=False,
            common_subexpression_elimination=False,
            specialize_definitions="off",
            inline_safe=False,
        ),
    )

    main = optimized.modules[0]
    assert main.nodes[0].outputs[0].type_expr == _tensor("topk_pos_dim")
    assert main.nodes[0].outputs[1].type_expr == _tensor("topk_pos_dim")
    assert main.nodes[2].outputs[0].type_expr == _tensor("topk_pos_dim", "D")
    assert main.return_type_expr == _tensor("topk_pos_dim", "D")


def test_codegen2_generate_uses_cached_decoder_contract_from_signature() -> None:
    cache_type = TypeOptional(
        TypeList(TypeTuple((_tensor("B", "H", "P", "DH"), _tensor("B", "H", "P", "DH"))))
    )
    code = emit_model_code_from_graph_ir(
        _toy_program(
            (
                GraphValue("input_ids", _tensor("B", "S")),
                GraphValue("attn_mask", TypeOptional(_tensor("B", "K")), optional=True),
                GraphValue("past_kv", cache_type, optional=True),
                GraphValue("use_cache", TypeOptional(TypeBool()), optional=True),
            ),
            ("logits", "new_kv"),
        )
    )

    assert "step_input = out[:, -1:] if cache is not None else out" in code
    assert "forward_kwargs['past_kv'] = cache" in code
    assert "forward_kwargs['use_cache'] = True" in code
    assert "attention_mask = torch.cat([attention_mask, _ones_like_ids(next_id)], dim=1)" in code


def test_codegen2_generate_uses_uncached_decoder_contract_from_signature() -> None:
    code = emit_model_code_from_graph_ir(
        _toy_program(
            (
                GraphValue("input_ids", _tensor("B", "S")),
                GraphValue("attn_mask", TypeOptional(_tensor("B", "S")), optional=True),
            ),
            ("logits",),
        )
    )

    assert "result = self._forward(out, **forward_kwargs)" in code
    assert "step_input = out[:, -1:] if cache is not None else out" not in code
    assert "forward_kwargs['attn_mask'] = attention_mask" in code


def test_codegen2_generate_uses_encoder_decoder_contract_from_signature() -> None:
    code = emit_model_code_from_graph_ir(
        _toy_program(
            (
                GraphValue("input_ids", _tensor("B", "S")),
                GraphValue("attention_mask", TypeOptional(_tensor("B", "S")), optional=True),
                GraphValue("decoder_input_ids", _tensor("B", "T")),
                GraphValue(
                    "decoder_attention_mask",
                    TypeOptional(_tensor("B", "T")),
                    optional=True,
                ),
            ),
            ("logits",),
        )
    )

    assert "decoder_input_ids = kwargs.pop('decoder_input_ids', None)" in code
    assert "forward_kwargs['decoder_input_ids'] = decoder_input_ids" in code
    assert "result = self._forward(input_ids, **forward_kwargs)" in code
    assert "return decoder_input_ids" in code


def test_codegen2_nested_config_lookup_uses_dotted_path() -> None:
    found, value = Codegen2GraphModel._lookup_config(
        {"decoder": {"hidden_size": 1152}}, "decoder.hidden_size"
    )

    assert found
    assert value == 1152
    assert Codegen2GraphModel._lookup_config({"decoder": {}}, "decoder.hidden_size") == (
        False,
        None,
    )


def test_codegen2_generated_state_placement_preserves_tensor_aliases() -> None:
    code = emit_model_code_from_graph_ir(
        _toy_program((GraphValue("input_ids", _tensor("B", "S")),), ("logits",))
    )
    namespace: dict[str, object] = {}
    exec(code, namespace)
    model_cls = namespace["GeneratedAxonModel"]

    tensor = torch.ones(2, 2)
    model = model_cls.from_state_dict(
        {
            "model.layers.0.mlp.experts.gate_up_proj.weight": tensor,
            "model.layers.0.mlp.experts.0.gate_up_proj.weight": tensor,
        },
        param_devices=["cpu"],
    )

    assert (
        model.state_dict_tensors["model.layers.0.mlp.experts.gate_up_proj.weight"]
        is model.state_dict_tensors["model.layers.0.mlp.experts.0.gate_up_proj.weight"]
    )


def test_codegen2_torch_generated_model_uses_shared_public_interface() -> None:
    code = emit_model_code_from_graph_ir(
        _toy_program((GraphValue("input_ids", _tensor("B", "S")),), ("logits",))
    )

    assert "def load_state_dict(self, state_dict, strict=True):" in code
    assert "def setup(self):" in code
    assert "def _forward(self, input_ids=None, **inputs):" in code
    assert "def forward(self, input_ids=None, **inputs):" in code


def test_graph_optimizer_simplifies_symbolic_dim_factor_roundtrip() -> None:
    dim_t = TypeDim()
    main = GraphModule(
        name="main",
        inputs=(GraphValue("D", dim_t), GraphValue("H", dim_t)),
        outputs=(GraphValueRef("out", dim_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.binary./"),
                inputs=(GraphValueRef("D", dim_t), GraphValueRef("H", dim_t)),
                attrs={},
                outputs=(GraphValue("hd", dim_t),),
                source_module="main",
                type_expr=dim_t,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("core.binary.*"),
                inputs=(GraphValueRef("H", dim_t), GraphValueRef("hd", dim_t)),
                attrs={},
                outputs=(GraphValue("out", dim_t),),
                source_module="main",
                type_expr=dim_t,
            ),
        ),
        return_type_expr=dim_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(main,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False, specialize_definitions="off"),
    )

    optimized_main = optimized.modules[0]
    assert optimized_main.nodes == ()
    assert optimized_main.outputs == (GraphValueRef("D", dim_t),)


def test_graph_optimizer_simplifies_symbolic_dim_type_metadata() -> None:
    dim_t = TypeDim()
    simplified = TypeTensor(
        "Tensor",
        ("B", "S", DimExprBinary("*", "H", DimExprBinary("/", "D", "H"))),
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", TypeTensor("Tensor", ("B", "S", "D")), ("B", "S", "D")),),
        outputs=(GraphValueRef("out", simplified, simplified.dims),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.alias"),
                inputs=(GraphValueRef("x", TypeTensor("Tensor", ("B", "S", "D")), ("B", "S", "D")),),
                attrs={},
                outputs=(GraphValue("out", simplified, simplified.dims),),
                source_module="main",
                type_expr=simplified,
                dims=simplified.dims,
            ),
        ),
        return_type_expr=simplified,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(main,), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False, specialize_definitions="off"),
    )

    assert optimized.modules[0].return_type_expr == TypeTensor("Tensor", ("B", "S", "D"))


def test_graph_optimizer_does_not_substitute_later_dim_value_into_prior_rhs() -> None:
    dim_t = TypeDim()
    bool_t = TypeBool()
    mask_t = TypeTensor("Tensor", ("B", 1, "S", "K"))
    x_t = TypeTensor("Tensor", ("B", "S"))
    attn_t = TypeTensor("Tensor", ("B", "K"))
    masker = GraphModule(
        name="masker",
        inputs=(
            GraphValue("x", x_t, x_t.dims),
            GraphValue("K", dim_t),
            GraphValue("seed", mask_t, mask_t.dims),
        ),
        outputs=(GraphValueRef("out", mask_t, mask_t.dims),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="masker:1",
                op=GraphOp("core.alias"),
                inputs=(GraphValueRef("seed", mask_t, mask_t.dims),),
                attrs={},
                outputs=(GraphValue("out", mask_t, mask_t.dims),),
                source_module="masker",
                type_expr=mask_t,
                dims=mask_t.dims,
            ),
        ),
        return_type_expr=mask_t,
    )
    main = GraphModule(
        name="main",
        inputs=(
            GraphValue("x", x_t, x_t.dims),
            GraphValue("attn", attn_t, attn_t.dims),
            GraphValue("seed", mask_t, mask_t.dims),
            GraphValue("cond", bool_t),
            GraphValue("past_length", dim_t),
        ),
        outputs=(GraphValueRef("mask", mask_t, mask_t.dims),),
        output_names=("mask",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("core.select"),
                inputs=(
                    GraphValueRef("cond", bool_t),
                    GraphExpr(
                        op=GraphOp("core.binary.+"),
                        inputs=(GraphValueRef("past_length", dim_t), GraphValueRef("S", dim_t)),
                        attrs={},
                        type_expr=dim_t,
                    ),
                    GraphValueRef("K", dim_t),
                ),
                attrs={},
                outputs=(GraphValue("key_len", dim_t),),
                source_module="main",
                type_expr=dim_t,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("masker"),
                inputs=(
                    GraphValueRef("x", x_t, x_t.dims),
                    GraphValueRef("key_len", dim_t),
                    GraphValueRef("seed", mask_t, mask_t.dims),
                ),
                attrs={},
                outputs=(GraphValue("mask", mask_t, mask_t.dims),),
                source_module="main",
                type_expr=mask_t,
                dims=mask_t.dims,
            ),
        ),
        return_type_expr=mask_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(masker, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(inline_safe=False, specialize_definitions="off"),
    )

    optimized_main = next(module for module in optimized.modules if module.name == "main")
    key_len_node = next(node for node in optimized_main.nodes if node.outputs and node.outputs[0].name == "key_len")
    assert key_len_node.op.name == "core.select"
    assert key_len_node.inputs[2] == GraphValueRef("K", dim_t)


def test_graph_ir_renderer_keeps_atomic_list_arguments_inline() -> None:
    dim_t = TypeDim()
    tensor_t = TypeTensor("Tensor", ("B", "S"))
    shape_t = TypeList(dim_t)
    main = GraphModule(
        name="main",
        inputs=(GraphValue("x", tensor_t, tensor_t.dims),),
        outputs=(GraphValueRef("y", tensor_t, tensor_t.dims),),
        output_names=("y",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("_reshape"),
                inputs=(
                    GraphValueRef("x", tensor_t, tensor_t.dims),
                    GraphExpr(
                        op=GraphOp("core.list"),
                        inputs=(GraphValueRef("B", dim_t), GraphValueRef("S", dim_t)),
                        attrs={},
                        type_expr=shape_t,
                    ),
                ),
                attrs={},
                outputs=(GraphValue("y", tensor_t, tensor_t.dims),),
                source_module="main",
                type_expr=tensor_t,
                dims=tensor_t.dims,
            ),
        ),
        return_type_expr=tensor_t,
    )

    rendered = render_axon_file(
        graph_program_to_axon_file(GraphProgram(modules=(main,), main_module="main", pragmas={}))
    )

    assert "_reshape x [B, S]" in rendered
    assert "<- [B, S]" not in rendered


def test_graph_ir_renderer_keeps_atomic_return_tuple_inline() -> None:
    int_t = TypeInt()
    pair_t = TypeTuple((int_t, int_t))
    main = GraphModule(
        name="main",
        inputs=(GraphValue("a", int_t), GraphValue("b", int_t)),
        outputs=(
            GraphExpr(
                op=GraphOp("core.tuple"),
                inputs=(GraphValueRef("a", int_t), GraphValueRef("b", int_t)),
                attrs={},
                type_expr=pair_t,
            ),
        ),
        output_names=("pair",),
        nodes=(),
        return_type_expr=pair_t,
    )

    rendered = render_axon_file(
        graph_program_to_axon_file(GraphProgram(modules=(main,), main_module="main", pragmas={}))
    )

    assert "return (a, b)" in rendered
    assert "<- (a, b)" not in rendered


def test_graph_ir_optimizer_rewrites_fill_scatter_unit_slice_for_torch_backend() -> None:
    dim_t = TypeDim()
    int_t = TypeInt()
    string_t = TypeString()
    tensor_t = TypeTensor("Tensor", ("B", "S", "D"))
    step_t = TypeTensor("Tensor", ("B", 1, "D"))
    fill = GraphModule(
        name="Tensor.fill",
        inputs=(
            GraphValue("x", step_t),
            GraphValue("value", int_t),
            GraphValue("dtype", string_t),
        ),
        outputs=(GraphValueRef("out", step_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="Tensor.fill:1",
                op=GraphOp("_fill"),
                inputs=(
                    GraphValueRef("x", step_t),
                    GraphValueRef("value", int_t),
                    GraphValueRef("dtype", string_t),
                ),
                attrs={},
                outputs=(GraphValue("out", step_t),),
                source_module="Tensor.fill",
                type_expr=step_t,
                dims=step_t.dims,
            ),
        ),
        return_type_expr=step_t,
    )
    scatter = GraphModule(
        name="Tensor.scatter",
        inputs=(
            GraphValue("x", tensor_t),
            GraphValue("index", step_t),
            GraphValue("src", step_t),
            GraphValue("dim", dim_t),
        ),
        outputs=(GraphValueRef("out", tensor_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="Tensor.scatter:1",
                op=GraphOp("_scatter"),
                inputs=(
                    GraphValueRef("x", tensor_t),
                    GraphValueRef("index", step_t),
                    GraphValueRef("src", step_t),
                    GraphValueRef("dim", dim_t),
                ),
                attrs={},
                outputs=(GraphValue("out", tensor_t),),
                source_module="Tensor.scatter",
                type_expr=tensor_t,
                dims=tensor_t.dims,
            ),
        ),
        return_type_expr=tensor_t,
    )
    main = GraphModule(
        name="main",
        inputs=(GraphValue("y_all", tensor_t), GraphValue("y_step", step_t), GraphValue("t", TypeInt())),
        outputs=(GraphValueRef("out", tensor_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("Tensor.fill"),
                inputs=(
                    GraphValueRef("y_step", step_t),
                    GraphValueRef("t", TypeInt()),
                    GraphLiteral("long", TypeString()),
                ),
                attrs={},
                outputs=(GraphValue("idx", step_t),),
                source_module="main",
                type_expr=step_t,
                dims=step_t.dims,
            ),
            GraphNode(
                id="main:2",
                op=GraphOp("Tensor.scatter"),
                inputs=(
                    GraphValueRef("y_all", tensor_t),
                    GraphValueRef("idx", step_t),
                    GraphValueRef("y_step", step_t),
                    GraphLiteral(1, dim_t),
                ),
                attrs={},
                outputs=(GraphValue("out", tensor_t),),
                source_module="main",
                type_expr=tensor_t,
                dims=tensor_t.dims,
            ),
        ),
        return_type_expr=tensor_t,
    )

    optimized = optimize_graph_program(
        GraphProgram(modules=(fill, scatter, main), main_module="main", pragmas={}),
        config=GraphOptimizeConfig(
            backend_intrinsics="codegen2-torch",
            specialize_definitions="off",
            inline_safe=False,
            common_subexpression_elimination=False,
        ),
    )

    optimized_main = next(module for module in optimized.modules if module.name == "main")
    ops = [node.op.name for node in optimized_main.nodes]
    assert "_fill" not in ops
    assert "Tensor.scatter" not in ops
    assert "_assign_slice" in ops
    assign = next(node for node in optimized_main.nodes if node.op.name == "_assign_slice")
    assert assign.inputs == (
        GraphValueRef("y_all", tensor_t),
        GraphValueRef("y_step", step_t),
        GraphLiteral(1, dim_t),
        GraphValueRef("t", TypeInt()),
        GraphExpr(
            op=GraphOp("core.binary.+"),
            inputs=(GraphValueRef("t", TypeInt()), GraphLiteral(1, TypeInt())),
            attrs={},
            type_expr=TypeInt(),
        ),
    )
