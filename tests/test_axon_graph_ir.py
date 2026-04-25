from __future__ import annotations

from pathlib import Path

from brainsurgery.synapse import lower_axon_program_to_graph_ir, parse_axon_program
from brainsurgery.synapse.axon import parse_axon_program_from_path
from brainsurgery.synapse.axon.graph_ir import GraphValueRef


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

    graph = lower_axon_program_to_graph_ir(program, main_module="main")
    module = graph.modules[-1]

    assert graph.main_module == "main"
    assert [node.op.name for node in module.nodes] == ["core.binary.+", "core.binary.+"]
    assert len({output.name for node in module.nodes for output in node.outputs}) == 2
    assert module.nodes[0].outputs[0].name == "y"
    assert module.nodes[1].outputs[0].name != "y"
    assert isinstance(module.outputs[0], GraphValueRef)
    assert module.outputs[0].name == module.nodes[1].outputs[0].name


def test_graph_ir_lowers_generic_gpt2_kv_as_alternative_lowering_target() -> None:
    program = parse_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2-kv.axon")
    )

    graph = lower_axon_program_to_graph_ir(program, main_module="gpt2", optimize=False)

    assert graph.main_module == "gpt2"
    assert len(graph.modules) > 1
    assert len(graph.constant_nodes) > 0
    assert graph.modules[-1].name == "gpt2"
    assert all(node.outputs for module in graph.modules for node in module.nodes)
