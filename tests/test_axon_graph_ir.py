from __future__ import annotations

from pathlib import Path

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
    TypeBool,
    TypeInt,
    TypeList,
    TypeOptional,
    TypeTensor,
    TypeTuple,
    render_axon_file,
)
from brainsurgery.synapse.axon.codegen2 import Codegen2GraphModel, emit_model_code_from_graph_ir
from brainsurgery.synapse.axon.graph_ir import (
    GraphModule,
    GraphProgram,
    GraphValue,
    GraphValueRef,
    graph_program_to_axon_file,
    render_graph_program_to_dot,
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

    assert 'label="module helper\\nreturns Tensor[B,S]"' in rendered
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

    assert "result = self.forward(out, **forward_kwargs)" in code
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
    assert "result = self.forward(input_ids, **forward_kwargs)" in code
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
