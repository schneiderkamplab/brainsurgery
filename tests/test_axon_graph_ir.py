from __future__ import annotations

import shutil
from pathlib import Path

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
    TypeBool,
    TypeDim,
    TypeInt,
    TypeList,
    TypeNull,
    TypeOptional,
    TypePath,
    TypeString,
    TypeTensor,
    TypeTuple,
    render_axon_file,
)
from brainsurgery.synapse.axon.codegen2_torch import Codegen2GraphModel, emit_model_code_from_graph_ir
from brainsurgery.synapse.axon.codegen2_tinygrad import (
    emit_model_code_from_graph_ir as emit_tinygrad_model_code_from_graph_ir,
    tinygrad_op_table_markdown,
)
from brainsurgery.synapse.axon.graph_ir import (
    GraphLiteral,
    GraphExpr,
    GraphEffect,
    GraphOptimizeConfig,
    GraphModule,
    GraphNode,
    GraphOp,
    GraphPath,
    GraphProgram,
    GraphValue,
    GraphValueRef,
    graph_program_to_axon_file,
    graph_module_effect,
    optimize_graph_program,
    prune_graph_to_main,
    render_graph_program_to_dot,
    validate_graph_program,
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


def test_graph_ir_optimizer_does_not_fold_select_to_partial_expression() -> None:
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

    assert [node.op.name for node in optimized.modules[0].nodes] == ["core.select"]


def test_graph_ir_optimizer_specializes_single_callsite_literal_argument() -> None:
    int_t = TypeInt()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("scale", int_t), GraphValue("x", int_t)),
        outputs=(GraphValueRef("y", int_t),),
        output_names=("out",),
        nodes=(
            GraphNode(
                id="helper:1",
                op=GraphOp("core.binary.+"),
                inputs=(GraphValueRef("x", int_t), GraphValueRef("scale", int_t)),
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
        output_names=("out",),
        nodes=(
            GraphNode(
                id="main:1",
                op=GraphOp("helper"),
                inputs=(GraphLiteral(3, int_t), GraphValueRef("x", int_t)),
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
    assert "helper__spec_1" in module_names
    specialized = next(module for module in optimized.modules if module.name == "helper__spec_1")
    assert [value.name for value in specialized.inputs] == ["x"]
    main_node = next(module for module in optimized.modules if module.name == "main").nodes[0]
    assert main_node.op.name == "helper__spec_1"
    assert len(main_node.inputs) == 1


def test_graph_ir_optimizer_does_not_specialize_recursive_helper() -> None:
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
    assert next(module for module in optimized.modules if module.name == "main").nodes[0].op.name == "recur"


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


def test_graph_ir_optimizer_does_not_inline_atomic_constant_by_default() -> None:
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

    assert {module.name for module in optimized.modules} == {"VOCAB_SIZE", "main"}
    optimized_main = next(module for module in optimized.modules if module.name == "main")
    assert optimized_main.nodes[0].op.name == "VOCAB_SIZE"


def test_graph_ir_optimizer_constant_dim_substitution_requires_local_constraint() -> None:
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
    assert unchanged.inputs[0].type_expr == tensor_with_symbol
    assert changed.inputs[0].type_expr == tensor_with_literal
    assert changed.return_type_expr == tensor_with_literal


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


def test_graph_ir_optimizer_does_not_inline_constrained_helper() -> None:
    int_t = TypeInt()
    helper = GraphModule(
        name="helper",
        inputs=(GraphValue("x", int_t),),
        outputs=(GraphValueRef("x", int_t),),
        output_names=("out",),
        nodes=(),
        return_type_expr=int_t,
        constraints=(Constraint("=", "x", "x"),),
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

    assert {module.name for module in optimized.modules} == {"helper", "main"}
    assert next(module for module in optimized.modules if module.name == "main").nodes[0].op.name == "helper"


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
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2-kv.axon")
    ).ast

    graph = lower_axon_program_to_graph_ir(_typed(program, main_module="gpt2"), main_module="gpt2")

    assert graph.main_module == "gpt2"
    assert len(graph.modules) > 1
    assert graph.modules[-1].name == "gpt2"
    assert all(node.outputs for module in graph.modules for node in module.nodes)


def test_codegen2_tinygrad_emits_gpt2_source() -> None:
    program = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2-kv.axon")
    ).ast

    graph = lower_axon_program_to_graph_ir(_typed(program, main_module="gpt2"), main_module="gpt2")
    table = tinygrad_op_table_markdown(graph)
    code = emit_tinygrad_model_code_from_graph_ir(graph, class_name="TinyGPT2")

    assert "| Op | Count | Reason |" in table
    assert "`embedding`" not in table
    assert "from tinygrad import Tensor, dtypes" in code
    assert "class TinyGPT2" in code
    assert "class TinyGPT2(nn.Module)" not in code
    assert "def load_state_dict(self, state_dict, strict=True):" in code
    assert "def setup(self):" in code
    assert "def after_to(self):" in code
    assert "def _forward(self, input_ids=None, **inputs):" in code
    assert "def _forward_tiny" not in code


def test_codegen2_tinygrad_reports_where_indices_as_unsupported_primitive() -> None:
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

    assert "`where_indices`" in table
    with pytest.raises(NotImplementedError, match="where_indices"):
        emit_tinygrad_model_code_from_graph_ir(graph)


def _expert_linear_graph() -> GraphProgram:
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
                    GraphLiteral(False, TypeBool()),
                    GraphLiteral(False, TypeBool()),
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


def test_graph_ir_preserves_list_destructuring_outputs() -> None:
    program = resolve_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/gpt2-kv.axon")
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
