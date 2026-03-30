from __future__ import annotations

from pathlib import Path

import pytest
import torch
import typer
from omegaconf import OmegaConf

from brainsurgery.cli.synapse import axon_to_synapse, axon_visualize, emit_generic, synapse_to_axon
from brainsurgery.engine.state_dicts import _InMemoryStateDict
from brainsurgery.synapse import (
    emit_model_code_from_synapse_spec,
    infer_output_types_for_node,
    render_synapse_spec_to_dot,
)


def _spec_dict() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "symbols": {"D": 16, "V": 32, "C": 12, "L": 2, "H": 4, "M": 64},
            "params": {
                "activation": "gelu_new",
                "layer_norm_epsilon": 1e-5,
                "attn_backend": "sdpa",
            },
            "graph": [],
        },
    }


def _spec_yaml() -> str:
    return """synapse: 1
model:
  symbols:
    D: 16
    V: 32
    C: 12
    L: 2
    H: 4
    M: 64
  params:
    activation: gelu_new
    layer_norm_epsilon: 1.0e-5
    attn_backend: sdpa
  graph: []
"""


def test_cli_emit_synapse_writes_python_file(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    out_path = tmp_path / "generated_model.py"
    spec_path.write_text(_spec_yaml(), encoding="utf-8")

    emit_generic(spec_path=spec_path, output_path=out_path, class_name="FromCli", force=False)

    assert out_path.exists()
    contents = out_path.read_text(encoding="utf-8")
    assert "class FromCli(nn.Module):" in contents
    assert "def from_state_dict(cls, state_dict" in contents


def test_cli_emit_requires_force_for_existing_output(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    out_path = tmp_path / "generated_model.py"
    spec_path.write_text(_spec_yaml(), encoding="utf-8")
    out_path.write_text("# existing\n", encoding="utf-8")

    with pytest.raises(typer.BadParameter) as exc_info:
        emit_generic(spec_path=spec_path, output_path=out_path, class_name="FromCli", force=False)

    assert "overwrite" in str(exc_info.value)


def test_emit_accepts_minimal_op_map() -> None:
    bad_op_map = {
        "ops": {
            "embedding": {"target": "torch.nn.Embedding"},
            "linear": {"target": "torch.nn.Linear"},
            "layernorm": {"target": "torch.nn.LayerNorm"},
            "attention": {"target": "torch.nn.MultiheadAttention"},
        }
    }
    source = emit_model_code_from_synapse_spec(_spec_dict(), class_name="BadMap", op_map=bad_op_map)
    assert "class BadMap(nn.Module):" in source


def test_emit_generic_from_gemma3_spec(tmp_path: Path) -> None:
    spec_path = Path(__file__).resolve().parents[1] / "examples" / "gemma3_270m_synapse.yaml"
    out_path = tmp_path / "gemma_model.py"
    emit_generic(spec_path=spec_path, output_path=out_path, class_name="Gemma3Synapse", force=False)
    contents = out_path.read_text(encoding="utf-8")
    assert "class Gemma3Synapse(nn.Module):" in contents
    assert "'D': 640" in contents


def test_emit_model_code_from_synapse_spec_generic() -> None:
    source = emit_model_code_from_synapse_spec(_spec_dict(), class_name="GenericSynapse")
    assert "class GenericSynapse(nn.Module):" in source
    assert "def generate(self, input_ids: torch.Tensor" in source


def test_infer_output_types_for_node_linear_uses_input_shape_and_dim() -> None:
    inferred = infer_output_types_for_node(
        op_name="linear",
        node_spec={"dim": 64},
        input_slots=[("x", {"x"})],
        output_vars=["y"],
        var_types={"x": "Tensor[B,S,D]"},
    )
    assert inferred == {"y": "Tensor[B,S,64]"}


def test_cli_synapse_to_axon_and_back_roundtrip(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    axon_path = tmp_path / "spec.axon"
    lowered_path = tmp_path / "lowered.yaml"
    spec_path.write_text(_spec_yaml(), encoding="utf-8")

    synapse_to_axon(
        spec_path=spec_path,
        output_path=axon_path,
        module_name="tiny",
        force=False,
    )
    assert axon_path.exists()
    axon_text = axon_path.read_text(encoding="utf-8")
    assert axon_text.startswith("tiny :: ")
    assert "meta __inputs" not in axon_text
    assert "meta __outputs" not in axon_text

    axon_to_synapse(axon_path=axon_path, output_path=lowered_path, force=False)
    assert lowered_path.exists()
    lowered = OmegaConf.to_container(OmegaConf.load(lowered_path), resolve=True)
    assert isinstance(lowered, dict)
    assert lowered.get("synapse") == 1
    assert lowered.get("model", {}).get("symbols") is None


def test_cli_synapse_to_axon_requires_force_for_existing_output(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    axon_path = tmp_path / "spec.axon"
    spec_path.write_text(_spec_yaml(), encoding="utf-8")
    axon_path.write_text("# existing\n", encoding="utf-8")

    with pytest.raises(typer.BadParameter) as exc_info:
        synapse_to_axon(
            spec_path=spec_path,
            output_path=axon_path,
            module_name="tiny",
            force=False,
        )
    assert "overwrite" in str(exc_info.value)


def test_cli_axon_to_synapse_requires_yaml_output(tmp_path: Path) -> None:
    axon_path = tmp_path / "spec.axon"
    bad_output = tmp_path / "lowered.txt"
    axon_path.write_text(
        "tiny :: Tensor -> Tensor\ntiny x = do\n  y <- x\n  return y\n", encoding="utf-8"
    )

    with pytest.raises(typer.BadParameter) as exc_info:
        axon_to_synapse(axon_path=axon_path, output_path=bad_output, force=False)
    assert ".yaml" in str(exc_info.value)


def test_render_synapse_spec_to_dot_has_variable_labeled_edges_and_block_calls() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}, "y": {"shape": []}},
            "graph": [
                {"sum_xy": {"_op": "add", "_args": ["x", "y"], "_bind": "z"}},
                {
                    "call_blk": {
                        "_op": "call",
                        "_target": "blk",
                        "_args": "z",
                        "b": "y",
                        "_bind": "o",
                    }
                },
            ],
            "blocks": {
                "blk": {
                    "inputs": {"a": {"shape": []}, "b": {"shape": []}},
                    "graph": [{"sum_ab": {"_op": "add", "_args": ["a", "b"], "_bind": "r"}}],
                    "outputs": {"r": "r"},
                }
            },
            "outputs": {"o": "o"},
        },
    }
    dot = render_synapse_spec_to_dot(spec, block_label_by_block={"main": "demo.main"})
    assert "digraph synapse" in dot
    assert "rankdir=TB;" in dot
    assert "subgraph cluster_blk" in dot
    assert 'label="block demo.main"' in dot
    assert "sum_xy" in dot
    assert ">add<" in dot
    assert 'style=dashed, color="gray65"' in dot
    assert '"n_block::main":"p_x":s -> "n_op::main::graph::0000_sum_xy":"p_arg_x";' in dot
    assert '"n_block::main":"p_y":s -> "n_op::main::graph::0000_sum_xy":"p_arg_y";' in dot
    assert '"n_outputs::blk":"p_r":s -> "n_op::main::graph::0001_call_blk":"p_out_o"' in dot
    assert 'label="call"' in dot
    assert "block blk" in dot


def test_render_synapse_spec_to_dot_supports_horizontal_directions() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}, "y": {"shape": []}},
            "graph": [{"sum_xy": {"_op": "add", "_args": ["x", "y"], "_bind": "z"}}],
            "outputs": {"z": "z"},
        },
    }
    dot_lr = render_synapse_spec_to_dot(spec, direction="left-right")
    assert "rankdir=LR;" in dot_lr
    assert '"n_block::main":"p_x":e -> "n_op::main::graph::0000_sum_xy":"p_arg_x";' in dot_lr
    assert '"n_op::main::graph::0000_sum_xy":"p_out_z" -> "n_outputs::main":"p_z":w;' in dot_lr
    assert (
        '<TR><TD BGCOLOR="gray90" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>IN</B></FONT></TD>'
        '<TD BGCOLOR="gray90" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>OP</B></FONT></TD>'
        '<TD BGCOLOR="gray90" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>OUT</B></FONT></TD></TR>'
        in dot_lr
    )
    assert 'ROWSPAN="2"' in dot_lr

    dot_rl = render_synapse_spec_to_dot(spec, direction="right-left")
    assert "rankdir=RL;" in dot_rl
    assert '"n_block::main":"p_x":w -> "n_op::main::graph::0000_sum_xy":"p_arg_x";' in dot_rl
    assert '"n_op::main::graph::0000_sum_xy":"p_out_z" -> "n_outputs::main":"p_z":e;' in dot_rl


def test_cli_axon_visualize_writes_dot_file(tmp_path: Path) -> None:
    axon_path = tmp_path / "tiny.axon"
    dot_path = tmp_path / "tiny.dot"
    axon_path.write_text(
        """
tiny :: Tensor -> Tensor -> Tensor
tiny x y = do
  z <- x + y
  return z
""".strip()
        + "\n",
        encoding="utf-8",
    )

    axon_visualize(axon_path=axon_path, output_path=dot_path, force=False, main_module=None)

    assert dot_path.exists()
    dot_text = dot_path.read_text(encoding="utf-8")
    assert "digraph synapse" in dot_text
    assert "add" in dot_text
    assert '"p_arg_x"' in dot_text


def test_render_synapse_spec_to_dot_vertical_directions_keep_non_transposed_tables() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}, "y": {"shape": []}},
            "graph": [{"sum_xy": {"_op": "add", "_args": ["x", "y"], "_bind": "z"}}],
            "outputs": {"z": "z"},
        },
    }
    dot_td = render_synapse_spec_to_dot(spec, direction="top-down")
    assert "rankdir=TB;" in dot_td
    assert (
        '<TR><TD BGCOLOR="gray90" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>IN</B></FONT></TD>'
        '<TD PORT="p_arg_x" BGCOLOR="lemonchiffon" ALIGN="CENTER"><FONT POINT-SIZE="7">x</FONT></TD>'
        '<TD PORT="p_arg_y" BGCOLOR="lemonchiffon" ALIGN="CENTER"><FONT POINT-SIZE="7">y</FONT></TD></TR>'
        in dot_td
    )

    dot_bu = render_synapse_spec_to_dot(spec, direction="bottom-up")
    assert "rankdir=BT;" in dot_bu
    assert (
        '<TR><TD BGCOLOR="gray90" ALIGN="CENTER"><FONT POINT-SIZE="7"><B>OUT</B></FONT></TD>'
        '<TD PORT="p_out_z" BGCOLOR="honeydew" ALIGN="CENTER"><FONT POINT-SIZE="7">z</FONT></TD>'
        '<TD BGCOLOR="honeydew"></TD></TR>' in dot_bu
    )


def test_cli_axon_visualize_requires_dot_output(tmp_path: Path) -> None:
    axon_path = tmp_path / "tiny.axon"
    bad_output = tmp_path / "tiny.txt"
    axon_path.write_text(
        "tiny :: Tensor -> Tensor\ntiny x = do\n  y <- x\n  return y\n", encoding="utf-8"
    )

    with pytest.raises(typer.BadParameter) as exc_info:
        axon_visualize(axon_path=axon_path, output_path=bad_output, force=False, main_module=None)
    assert ".dot" in str(exc_info.value)


def test_render_synapse_spec_to_dot_includes_parameter_annotations() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}},
            "graph": [
                {"proj": {"_op": "linear", "_args": "x", "_bind": "y", "bias": True}},
            ],
            "outputs": {"y": "y"},
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert "proj.weight" in dot


def test_render_synapse_spec_to_dot_connects_when_condition_dependencies() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"use_cache": {"shape": []}},
            "graph": [
                {
                    "init_cache": {
                        "_op": "list_init",
                        "_bind": "new_kv",
                        "when": "use_cache",
                    }
                },
                {
                    "null_cache": {
                        "_op": "_ir_expr",
                        "value": None,
                        "_bind": "new_kv",
                        "when": "not (use_cache)",
                    }
                },
            ],
            "outputs": {"new_kv": "new_kv"},
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert 'PORT="p_arg_cond"' in dot
    assert 'PORT="p_arg_then"' in dot
    assert 'PORT="p_arg_else"' in dot
    assert (
        '"n_block::main":"p_use_cache":s -> "n_ternary::main::graph::0001_new_kv":"p_arg_cond"'
        in dot
    )
    assert (
        '"n_op::main::graph::0000_init_cache":"p_out_new_kv" -> "n_ternary::main::graph::0001_new_kv":"p_arg_then"'
        in dot
    )
    assert (
        '"n_op::main::graph::0001_null_cache":"p_out_new_kv" -> "n_ternary::main::graph::0001_new_kv":"p_arg_else"'
        in dot
    )
    assert (
        '"n_block::main":"p_use_cache" -> "n_op::main::graph::0000_init_cache" [label="use_cache"]'
        not in dot
    )
    assert (
        '"n_block::main":"p_use_cache" -> "n_op::main::graph::0001_null_cache" [label="use_cache"]'
        not in dot
    )


def test_render_synapse_spec_to_dot_uses_canonical_cache_seq_len_slot_name() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"cache": {"shape": []}},
            "graph": [
                {"len": {"_op": "cache_seq_len", "_args": "cache", "_bind": "n"}},
            ],
            "outputs": {"n": "n"},
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert "p_arg_entry" in dot
    assert "p_arg_cache" not in dot


def test_render_synapse_spec_to_dot_renders_for_as_flowchart_loop() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}},
            "graph": [
                {
                    "loop_main": {
                        "_op": "for",
                        "_var": "i",
                        "_from": 0,
                        "_to": 3,
                        "_step": 1,
                        "_body": [
                            {"step_a": {"_op": "_ir_alias", "_args": "x", "_bind": "x"}},
                            {"step_b": {"_op": "_ir_alias", "_args": "x", "_bind": "x"}},
                        ],
                    }
                }
            ],
            "outputs": {"x": "x"},
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert "<B>FOR</B>" in dot
    assert 'label="for i <- [0..3)"' in dot
    assert 'COLOR="deepskyblue4"' in dot
    assert 'PORT="p_arg_i"' in dot
    assert 'PORT="p_arg_from"' not in dot
    assert 'PORT="p_arg_to"' not in dot
    assert 'PORT="p_arg_step"' not in dot
    assert 'PORT="p_arg_x"' in dot
    assert 'label="loop"' in dot
    assert 'label="next"' in dot
    assert "subgraph cluster_loop_" in dot


def test_render_synapse_spec_to_dot_wires_loop_variable_into_body_ops() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"xs": {"shape": []}},
            "graph": [
                {
                    "loop": {
                        "_op": "for",
                        "_var": "i",
                        "_from": 0,
                        "_to": 3,
                        "_step": 1,
                        "_body": [
                            {"pick": {"_op": "list_index", "_args": ["xs", "i"], "_bind": "x"}},
                        ],
                    }
                }
            ],
            "outputs": {"x": "x"},
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert (
        '"n_op::main::graph::0000_loop":"p_arg_i":s -> "n_op::main::graph.loop._body::0000_pick":"p_arg_index";'
        in dot
    )


def test_render_synapse_spec_to_dot_routes_loop_carried_values_via_for_outputs() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}},
            "graph": [
                {
                    "loop": {
                        "_op": "for",
                        "_var": "i",
                        "_from": 0,
                        "_to": 2,
                        "_step": 1,
                        "_body": [
                            {"upd": {"_op": "_ir_alias", "_args": "x", "_bind": "x"}},
                            {"scratch": {"_op": "_ir_alias", "_args": "x", "_bind": "tmp"}},
                        ],
                    }
                },
                {"after": {"_op": "_ir_alias", "_args": "x", "_bind": "y"}},
            ],
            "outputs": {"y": "y"},
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert (
        '"n_op::main::graph.loop._body::0000_upd":"p_out_x" -> "n_for_outputs::main::graph::0000_loop":"p_out_x":n;'
        in dot
    )
    assert (
        '"n_for_outputs::main::graph::0000_loop":"p_out_x":s -> "n_op::main::graph::0001_after":"p_arg_x";'
        in dot
    )
    assert (
        '"n_op::main::graph.loop._body::0000_upd":"p_out_x" -> "n_op::main::graph::0001_after":"p_arg_x";'
        not in dot
    )
    assert '"n_for_outputs::main::graph::0000_loop":"p_out_tmp"' not in dot


def test_render_synapse_spec_to_dot_routes_loop_inputs_via_for_header() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x0": {"shape": []}},
            "graph": [
                {"seed": {"_op": "_ir_alias", "_args": "x0", "_bind": "x"}},
                {
                    "loop": {
                        "_op": "for",
                        "_var": "i",
                        "_from": 0,
                        "_to": 2,
                        "_step": 1,
                        "_body": [
                            {"upd": {"_op": "_ir_alias", "_args": "x", "_bind": "x"}},
                        ],
                    }
                },
                {"after": {"_op": "_ir_alias", "_args": "x", "_bind": "y"}},
            ],
            "outputs": {"y": "y"},
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert (
        '"n_op::main::graph::0000_seed":"p_out_x" -> "n_op::main::graph::0001_loop":"p_arg_x":n;'
        in dot
    )
    assert (
        '"n_op::main::graph::0001_loop":"p_arg_x":s -> "n_op::main::graph.loop._body::0000_upd":"p_arg_x";'
        in dot
    )
    assert (
        '"n_op::main::graph::0000_seed":"p_out_x" -> "n_op::main::graph.loop._body::0000_upd":"p_arg_x";'
        not in dot
    )


def test_render_synapse_spec_to_dot_uses_block_label_mapping() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}},
            "graph": [{"id": {"_op": "_ir_alias", "_args": "x", "_bind": "y"}}],
            "outputs": {"y": "y"},
        },
    }
    dot = render_synapse_spec_to_dot(spec, block_label_by_block={"main": "pkg.main"})
    assert 'label="block pkg.main"' in dot


def test_render_synapse_spec_to_dot_shows_unique_call_scope_prefix_in_subblock_params() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}},
            "blocks": {
                "sub": {
                    "inputs": {"x": {"shape": []}},
                    "graph": [
                        {
                            "proj": {
                                "_op": "linear",
                                "_args": "x",
                                "_bind": "y",
                                "_params": {"weight": "w"},
                            }
                        }
                    ],
                    "outputs": {"out_0": "y"},
                }
            },
            "graph": [
                {
                    "c": {
                        "_op": "call",
                        "_target": "sub",
                        "_scope": "foo",
                        "_args": "x",
                        "_bind": "y",
                    }
                }
            ],
            "outputs": {"y": "y"},
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert "foo.w" in dot


def test_render_synapse_spec_to_dot_infers_loop_scope_prefix_for_subblock_params() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}},
            "blocks": {
                "sub": {
                    "inputs": {"x": {"shape": []}},
                    "graph": [
                        {
                            "proj": {
                                "_op": "linear",
                                "_args": "x",
                                "_bind": "y",
                                "_params": {"weight": "w"},
                            }
                        }
                    ],
                    "outputs": {"out_0": "y"},
                }
            },
            "graph": [
                {
                    "lp": {
                        "_op": "for",
                        "_scope": "model.layers",
                        "_var": "i",
                        "_from": 0,
                        "_to": 2,
                        "_body": [
                            {
                                "c": {
                                    "_op": "call",
                                    "_target": "sub",
                                    "_scope": "model",
                                    "_args": "x",
                                    "_bind": "x",
                                }
                            },
                        ],
                    }
                }
            ],
            "outputs": {"x": "x"},
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert "model.layers.{i}.w" in dot


def test_render_synapse_spec_to_dot_infers_nested_loop_scope_prefix_for_subblock_params() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}},
            "blocks": {
                "sub": {
                    "inputs": {"x": {"shape": []}},
                    "graph": [
                        {
                            "proj": {
                                "_op": "linear",
                                "_args": "x",
                                "_bind": "y",
                                "_params": {"weight": "w"},
                            }
                        }
                    ],
                    "outputs": {"out_0": "y"},
                }
            },
            "graph": [
                {
                    "outer": {
                        "_op": "for",
                        "_scope": "model.layers",
                        "_var": "i",
                        "_from": 0,
                        "_to": 2,
                        "_body": [
                            {
                                "inner": {
                                    "_op": "for",
                                    "_scope": "model.layers.experts",
                                    "_var": "e",
                                    "_from": 0,
                                    "_to": 2,
                                    "_body": [
                                        {
                                            "c": {
                                                "_op": "call",
                                                "_target": "sub",
                                                "_scope": "model",
                                                "_args": "x",
                                                "_bind": "x",
                                            }
                                        }
                                    ],
                                }
                            }
                        ],
                    }
                }
            ],
            "outputs": {"x": "x"},
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert "model.layers.{i}.experts.{e}.w" in dot


def test_render_synapse_spec_to_dot_infers_loop_scope_prefix_for_calls_without_scope() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}},
            "blocks": {
                "sub": {
                    "inputs": {"x": {"shape": []}},
                    "graph": [
                        {
                            "proj": {
                                "_op": "linear",
                                "_args": "x",
                                "_bind": "y",
                                "_params": {"weight": "w"},
                            }
                        }
                    ],
                    "outputs": {"out_0": "y"},
                }
            },
            "graph": [
                {
                    "lp": {
                        "_op": "for",
                        "_scope": "h",
                        "_var": "i",
                        "_from": 0,
                        "_to": 2,
                        "_body": [
                            {"c": {"_op": "call", "_target": "sub", "_args": "x", "_bind": "x"}},
                        ],
                    }
                }
            ],
            "outputs": {"x": "x"},
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert "h.{i}.w" in dot


def test_render_synapse_spec_to_dot_renders_scope_subgraph_with_input_output_gateways() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}},
            "blocks": {
                "blk": {
                    "inputs": {"x": {"shape": []}},
                    "graph": [{"id": {"_op": "_ir_alias", "_args": "x", "_bind": "out_0"}}],
                    "outputs": {"out_0": "out_0"},
                }
            },
            "graph": [
                {
                    "c": {
                        "_op": "call",
                        "_target": "blk",
                        "_scope": "model.layer",
                        "_args": "x",
                        "_bind": "h",
                    }
                },
                {"out": {"_op": "_ir_alias", "_args": "h", "_bind": "y"}},
            ],
            "outputs": {"y": "y"},
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert "subgraph cluster_scope_main__model_layer" in dot
    assert 'label="scope model.layer"' in dot
    assert '"n_block::main":"p_x":s -> "n_scope_in::main::model.layer":"p_in_x":n' in dot
    assert (
        '"n_scope_in::main::model.layer":"p_in_x":s -> "n_op::main::graph::0000_c":"p_arg_x"' in dot
    )
    assert (
        '"n_op::main::graph::0000_c":"p_out_h" -> "n_scope_out::main::model.layer":"p_out_h"' in dot
    )
    assert (
        '"n_scope_out::main::model.layer":"p_out_h":s -> "n_op::main::graph::0001_out":"p_arg_x"'
        in dot
    )


def test_render_synapse_spec_to_dot_prefers_latest_outer_rebind_over_scope_out_for_block_outputs() -> (
    None
):
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}},
            "graph": [
                {"a": {"_op": "_ir_alias", "_scope": "model.inner", "_args": "x", "_bind": "h"}},
                {"b": {"_op": "add", "_args": ["h", "x"], "_bind": "h"}},
            ],
            "outputs": {"h": "h"},
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert '"n_op::main::graph::0001_b":"p_out_h" -> "n_outputs::main":"p_h":n;' in dot
    assert '"n_scope_out::main::model.inner":"p_out_h":s -> "n_outputs::main":"p_h":n;' not in dot


def test_render_synapse_spec_to_dot_nests_loop_cluster_under_parent_scope() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}},
            "graph": [
                {"a": {"_op": "_ir_alias", "_scope": "model", "_args": "x", "_bind": "x"}},
                {
                    "lp": {
                        "_op": "for",
                        "_scope": "model.layers",
                        "_var": "i",
                        "_from": 0,
                        "_to": 2,
                        "_body": [
                            {
                                "b": {
                                    "_op": "_ir_alias",
                                    "_scope": "model",
                                    "_args": "x",
                                    "_bind": "x",
                                }
                            }
                        ],
                    }
                },
            ],
            "outputs": {"x": "x"},
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert "subgraph cluster_scope_main__model" in dot
    assert 'label="scope model"' in dot
    assert "subgraph cluster_loop_main__graph__0001_lp" in dot
    scope_start = dot.index("subgraph cluster_scope_main__model")
    scope_end = dot.index("\n  }", scope_start)
    scope_body = dot[scope_start:scope_end]
    assert "subgraph cluster_loop_main__graph__0001_lp" in scope_body


def test_render_synapse_spec_to_dot_does_not_duplicate_scope_cluster_names() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}},
            "graph": [
                {"a": {"_op": "_ir_alias", "_scope": "self_attn", "_args": "x", "_bind": "y"}}
            ],
            "outputs": {"y": "y"},
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert dot.count("subgraph cluster_scope_main__self_attn") == 1


def test_render_synapse_spec_to_dot_propagates_transitive_call_scope_prefixes() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}},
            "blocks": {
                "decoder": {
                    "inputs": {"x": {"shape": []}},
                    "graph": [
                        {
                            "experts": {
                                "_op": "for",
                                "_scope": "mlp.experts",
                                "_var": "e",
                                "_from": 0,
                                "_to": 2,
                                "_body": [
                                    {
                                        "call_expert": {
                                            "_op": "call",
                                            "_target": "expert_ffn",
                                            "_scope": "mlp",
                                            "_args": "x",
                                            "_bind": "x",
                                        }
                                    }
                                ],
                            }
                        }
                    ],
                    "outputs": {"out_0": "x"},
                },
                "expert_ffn": {
                    "inputs": {"x": {"shape": []}},
                    "graph": [
                        {
                            "proj": {
                                "_op": "linear",
                                "_args": "x",
                                "_bind": "y",
                                "_params": {"weight": "w"},
                            }
                        }
                    ],
                    "outputs": {"out_0": "y"},
                },
            },
            "graph": [
                {
                    "layers": {
                        "_op": "for",
                        "_scope": "model.layers",
                        "_var": "i",
                        "_from": 0,
                        "_to": 2,
                        "_body": [
                            {
                                "call_decoder": {
                                    "_op": "call",
                                    "_target": "decoder",
                                    "_scope": "model",
                                    "_args": "x",
                                    "_bind": "x",
                                }
                            }
                        ],
                    }
                }
            ],
            "outputs": {"x": "x"},
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert "model.layers.{i}.mlp.experts.{e}.w" in dot


def test_render_synapse_spec_to_dot_renders_slot_type_hints_when_provided() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}},
            "graph": [{"id": {"_op": "_ir_alias", "_args": "x", "_bind": "y"}}],
            "outputs": {"y": "y"},
        },
    }
    dot = render_synapse_spec_to_dot(
        spec,
        block_io_types={
            "main": {
                "inputs": {"x": "Tensor[B,S,D]"},
                "outputs": {"y": "Tensor[B,S,D]"},
            }
        },
    )
    assert "Tensor[B,S,D]" in dot
    assert 'POINT-SIZE="6" COLOR="gray50"' in dot


def test_render_synapse_spec_to_dot_reads_persisted_block_io_types_from_spec() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}},
            "graph": [{"id": {"_op": "_ir_alias", "_args": "x", "_bind": "y"}}],
            "outputs": {"y": "y"},
            "types": {
                "block_io": {
                    "main": {
                        "inputs": {"x": "Tensor[B,S,D]"},
                        "outputs": {"y": "Tensor[B,S,D]"},
                    }
                }
            },
        },
    }
    dot = render_synapse_spec_to_dot(spec)
    assert "Tensor[B,S,D]" in dot


def test_render_synapse_spec_to_dot_infers_linear_output_type_from_input_and_dim() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"shape": []}},
            "graph": [{"proj": {"_op": "linear", "_args": "x", "_bind": "y", "dim": 64}}],
            "outputs": {"y": "y"},
        },
    }
    dot = render_synapse_spec_to_dot(
        spec,
        block_io_types={
            "main": {
                "inputs": {"x": "Tensor[B,S,D]"},
                "outputs": {},
            }
        },
    )
    assert "Tensor[B,S,64]" in dot


def test_optional_input_defaults_to_none_in_emitted_code() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "symbols": {},
            "inputs": {
                "x": {"shape": [], "optional": True},
            },
            "graph": [],
            "outputs": {"x_out": "x"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="OptionalModel")
    assert "def _prepare_env(" in source
    assert "x = env.get('x')" in source


def test_index_on_none_collection_is_none_safe() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "symbols": {},
            "inputs": {"collection": {"shape": [], "optional": True}},
            "graph": [{"at0": {"_op": "list_index", "_args": ["collection", 0], "_bind": "x0"}}],
            "outputs": {"x0": "x0"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="IndexSafeModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["IndexSafeModel"]()
    out = model()
    assert out["x0"] is None


def test_emit_repeat_block_single_output_loop_carry() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "symbols": {"L": 3},
            "inputs": {"zero": {"shape": []}, "one_seed": {"shape": []}},
            "blocks": {
                "step": {
                    "inputs": {"x": {"shape": []}, "one": {"shape": []}},
                    "graph": [{"inc": {"_op": "add", "_args": ["x", "one"], "_bind": "y"}}],
                    "outputs": {"y": "y"},
                }
            },
            "graph": [
                {"init": {"_op": "add", "_args": ["zero", "zero"], "_bind": "x"}},
                {"one_make": {"_op": "add", "_args": ["zero", "one_seed"], "_bind": "one"}},
                {
                    "loop": {
                        "_op": "for",
                        "_scope": "loop",
                        "_var": "i",
                        "_to": "L",
                        "_body": [
                            {
                                "blk": {
                                    "_op": "call",
                                    "_target": "step",
                                    "_args": "x",
                                    "one": "one",
                                    "_bind": "x",
                                }
                            }
                        ],
                    }
                },
            ],
            "outputs": {"result": "x"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="LoopModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["LoopModel"]()
    out = model(zero=torch.tensor(0.0), one_seed=torch.tensor(1.0))
    assert torch.is_tensor(out["result"])
    assert float(out["result"]) == 3.0


def test_emit_for_block_with_step_single_output_loop_carry() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"zero": {"shape": []}, "one_seed": {"shape": []}},
            "blocks": {
                "step": {
                    "inputs": {"x": {"shape": []}, "one": {"shape": []}},
                    "graph": [{"inc": {"_op": "add", "_args": ["x", "one"], "_bind": "y"}}],
                    "outputs": {"y": "y"},
                }
            },
            "graph": [
                {"init": {"_op": "add", "_args": ["zero", "zero"], "_bind": "x"}},
                {"one_make": {"_op": "add", "_args": ["zero", "one_seed"], "_bind": "one"}},
                {
                    "loop": {
                        "_op": "for",
                        "_scope": "loop",
                        "_var": "i",
                        "_from": 0,
                        "_to": 6,
                        "_step": 2,
                        "_body": [
                            {
                                "blk": {
                                    "_op": "call",
                                    "_target": "step",
                                    "_args": "x",
                                    "one": "one",
                                    "_bind": "x",
                                }
                            }
                        ],
                    }
                },
            ],
            "outputs": {"result": "x"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="LoopStepModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["LoopStepModel"]()
    out = model(zero=torch.tensor(0.0), one_seed=torch.tensor(1.0))
    assert torch.is_tensor(out["result"])
    assert float(out["result"]) == 3.0


def test_generated_linear_handles_empty_batch() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "symbols": {},
            "inputs": {"x": {}},
            "graph": [{"n": {"_op": "linear", "_args": "x", "_bind": "y", "bias": False}}],
            "outputs": {"y": "y"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="LinearEmptyModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["LinearEmptyModel"]()
    model.load_state_dict_tensors({"n.weight": torch.randn(8, 4)})
    out = model(x=torch.empty((0, 4), dtype=torch.float32))
    assert out["y"].shape == (0, 8)


def test_generated_linear_expert_materializes_mxfp4_aliases() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [
                {
                    "n": {
                        "_op": "linear",
                        "_args": "x",
                        "_bind": "y",
                        "bias": True,
                        "expert": 1,
                        "transpose": True,
                    }
                }
            ],
            "outputs": {"y": "y"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="LinearExpertMXFP4Model")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["LinearExpertMXFP4Model"]()
    model.load_state_dict_tensors(
        {
            "n_blocks": torch.tensor(
                [
                    [
                        [[0x00, 0x00]],
                        [[0x00, 0x00]],
                    ],
                    [
                        [[0x21, 0x43]],
                        [[0x65, 0x87]],
                    ],
                ],
                dtype=torch.uint8,
            ),
            "n_scales": torch.full((2, 2, 1), 127, dtype=torch.uint8),
            "n_bias": torch.tensor(
                [
                    [0.0, 0.0],
                    [0.25, -0.75],
                ],
                dtype=torch.float32,
            ),
        }
    )
    out = model(x=torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32))
    expected = torch.tensor([[15.25, 28.25]], dtype=torch.float32)
    assert torch.allclose(out["y"], expected, atol=1e-6, rtol=0.0)


def test_generated_split_interleave_mode() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [
                {
                    "s": {
                        "_op": "split",
                        "_args": "x",
                        "_bind": ["even", "odd"],
                        "parts": 2,
                        "interleave": True,
                    }
                }
            ],
            "outputs": {"even": "even", "odd": "odd"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="SplitInterleaveModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["SplitInterleaveModel"]()
    out = model(x=torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32))
    assert torch.equal(out["even"], torch.tensor([[0.0, 2.0]], dtype=torch.float32))
    assert torch.equal(out["odd"], torch.tensor([[1.0, 3.0]], dtype=torch.float32))


def test_generated_clamp_and_sigmoid_ops() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [
                {"c": {"_op": "clamp", "_args": "x", "_bind": "xc", "min": -1.0, "max": 1.0}},
                {"s": {"_op": "activations_sigmoid", "_args": "xc", "_bind": "y"}},
            ],
            "outputs": {"y": "y"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="ClampSigmoidModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["ClampSigmoidModel"]()
    out = model(x=torch.tensor([[-2.0, 0.0, 2.0]], dtype=torch.float32))
    expected = torch.sigmoid(torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float32))
    assert torch.allclose(out["y"], expected, atol=1e-6, rtol=0.0)


def test_generated_model_records_intermediates_to_runtime_state_dict() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}, "y": {}},
            "graph": [{"sum_xy": {"_op": "add", "_args": ["x", "y"], "_bind": "z"}}],
            "outputs": {"z": "z"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="RuntimeStateDictModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    runtime_state_dict = _InMemoryStateDict()
    model = namespace["RuntimeStateDictModel"](runtime_state_dict=runtime_state_dict)
    out = model(x=torch.tensor([1.0, 2.0]), y=torch.tensor([3.0, 4.0]))
    expected = torch.tensor([4.0, 6.0])
    assert torch.equal(out["z"], expected)
    assert "sum_xy::z" in runtime_state_dict
    assert torch.equal(runtime_state_dict["sum_xy::z"], expected)
