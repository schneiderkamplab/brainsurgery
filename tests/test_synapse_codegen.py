from __future__ import annotations

import math
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
    lower_axon_program_to_synapse_spec,
    parse_axon_program,
    render_synapse_spec_to_dot,
)
from brainsurgery.synapse.ops import linear as linear_op
from brainsurgery.synapse.runtime import SynapseProgramModel


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


def test_generated_model_keeps_double_at_params_absolute_under_current_param_root() -> None:
    source = emit_model_code_from_synapse_spec(_spec_dict(), class_name="RootedDoubleAt")
    namespace: dict[str, object] = {}
    exec(source, namespace)
    model_cls = namespace["RootedDoubleAt"]
    model = model_cls(  # type: ignore[operator]
        state_dict={"mlp.experts.0.gate_proj.weight": torch.ones(1, 1, dtype=torch.float32)}
    )
    model._param_roots_stack.append(["mlp.experts.0"])

    assert model._pick_param_from_single("ignored.scope", "@@gate_proj.weight") == "gate_proj.weight"
    assert model._pick_param_path("ignored.scope", ["@@gate_proj.weight"]) == "gate_proj.weight"


def test_generated_model_keeps_double_at_params_absolute_under_scope() -> None:
    source = emit_model_code_from_synapse_spec(_spec_dict(), class_name="ScopedDoubleAt")
    namespace: dict[str, object] = {}
    exec(source, namespace)
    model_cls = namespace["ScopedDoubleAt"]
    model = model_cls(  # type: ignore[operator]
        state_dict={"mlp.experts.0.gate_proj.weight": torch.ones(1, 1, dtype=torch.float32)}
    )

    assert model._pick_param_from_single("mlp.experts.0", "@@gate_proj.weight") == "gate_proj.weight"
    assert model._pick_param_path("mlp.experts.0", ["@@gate_proj.weight"]) == "gate_proj.weight"


def test_generated_rope_pair_hf_yarn_matches_runtime() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {
                "q": {"optional": False},
                "k": {"optional": False},
                "pos_ids": {"optional": False},
            },
            "graph": [
                {
                    "rope": {
                        "_op": "rope_pair",
                        "_bind": ["q_out", "k_out"],
                        "_args": ["q", "k"],
                        "position_ids": "pos_ids",
                        "theta": 150000.0,
                        "scale_factor": 32.0,
                        "low_freq_factor": 1.0,
                        "high_freq_factor": 32.0,
                        "original_context": 4096,
                        "attention_factor": 1.3465735902799727,
                        "rope_mode": "hf_yarn",
                        "truncate": False,
                    }
                }
            ],
            "outputs": {"q_out": "q_out", "k_out": "k_out"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="GeneratedRopePairHfYarn")
    namespace: dict[str, object] = {}
    exec(source, namespace)
    model_cls = namespace["GeneratedRopePairHfYarn"]

    runtime = SynapseProgramModel.from_spec(spec).eval()
    generated = model_cls.from_state_dict({}).eval()  # type: ignore[operator]

    torch.manual_seed(0)
    q = torch.randn(1, 64, 5, 64, dtype=torch.float32)
    k = torch.randn(1, 8, 5, 64, dtype=torch.float32)
    pos_ids = torch.arange(5, dtype=torch.int64).unsqueeze(0)

    with torch.no_grad():
        runtime_out = runtime(q=q, k=k, pos_ids=pos_ids)
        generated_out = generated(q=q, k=k, pos_ids=pos_ids)

    assert torch.allclose(runtime_out["q_out"], generated_out["q_out"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(runtime_out["k_out"], generated_out["k_out"], atol=1e-5, rtol=1e-5)


def test_generated_rope_pair_plain_scale_factor_matches_runtime() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {
                "q": {"optional": False},
                "k": {"optional": False},
                "pos_ids": {"optional": False},
            },
            "graph": [
                {
                    "rope": {
                        "_op": "rope_pair",
                        "_bind": ["q_out", "k_out"],
                        "_args": ["q", "k"],
                        "position_ids": "pos_ids",
                        "theta": 1_000_000.0,
                        "scale_factor": 8.0,
                    }
                }
            ],
            "outputs": {"q_out": "q_out", "k_out": "k_out"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="GeneratedRopePairScaleOnly")
    namespace: dict[str, object] = {}
    exec(source, namespace)
    model_cls = namespace["GeneratedRopePairScaleOnly"]

    runtime = SynapseProgramModel.from_spec(spec).eval()
    generated = model_cls.from_state_dict({}).eval()  # type: ignore[operator]

    torch.manual_seed(0)
    q = torch.randn(1, 4, 5, 64, dtype=torch.float32)
    k = torch.randn(1, 4, 5, 64, dtype=torch.float32)
    pos_ids = torch.arange(5, dtype=torch.int64).unsqueeze(0)

    with torch.no_grad():
        runtime_out = runtime(q=q, k=k, pos_ids=pos_ids)
        generated_out = generated(q=q, k=k, pos_ids=pos_ids)

    assert torch.allclose(runtime_out["q_out"], generated_out["q_out"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(runtime_out["k_out"], generated_out["k_out"], atol=1e-5, rtol=1e-5)


def test_linear_op_keeps_node_path_for_double_at_weights() -> None:
    assert linear_op.uses_node_path(None, {"bias": False, "weight": "@@gate_proj.weight"}) is True


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


def test_render_synapse_spec_to_dot_connects_select_condition_dependencies() -> None:
    spec: dict[str, object] = {
        "synapse": 1,
        "model": {
            "inputs": {"use_cache": {"shape": []}},
            "graph": [
                {
                    "init_or_null_cache": {
                        "_op": "select",
                        "_bind": "new_kv",
                        "cond": "use_cache",
                        "_then_bind": "then_new_kv",
                        "_else_bind": "else_new_kv",
                        "_then": [{"then_node": {"_op": "list_init", "_bind": "then_new_kv"}}],
                        "_else": [
                            {
                                "else_node": {
                                    "_op": "_ir_expr",
                                    "value": None,
                                    "_bind": "else_new_kv",
                                }
                            }
                        ],
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
        '"n_block::main":"p_use_cache":s -> "n_op::main::graph::0000_init_or_null_cache":"p_arg_cond"'
        in dot
    )
    assert (
        '"n_op::main::graph.init_or_null_cache._then::0000_then_node":"p_out_then_new_kv"'
        ' -> "n_op::main::graph::0000_init_or_null_cache":"p_arg_then"' in dot
    )
    assert (
        '"n_op::main::graph.init_or_null_cache._else::0000_else_node":"p_out_else_new_kv"'
        ' -> "n_op::main::graph::0000_init_or_null_cache":"p_arg_else"' in dot
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


def test_generated_xielu_reads_learned_activation_parameters() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [
                {
                    "xielu": {
                        "_op": "activations_xielu",
                        "_args": "x",
                        "_bind": "y",
                        "_params": {
                            "alpha_p": "act.alpha_p",
                            "alpha_n": "act.alpha_n",
                            "beta": "act.beta",
                            "eps": "act.eps",
                        },
                    }
                }
            ],
            "outputs": {"y": "y"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="XieluModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["XieluModel"]()
    state_dict = {
        "act.alpha_p": torch.tensor([0.2], dtype=torch.float32),
        "act.alpha_n": torch.tensor([-0.4], dtype=torch.float32),
        "act.beta": torch.tensor([0.6], dtype=torch.float32),
        "act.eps": torch.tensor([-1.0e-4], dtype=torch.float32),
    }
    model.load_state_dict_tensors(state_dict)
    x = torch.tensor([[-1.25, -0.25, 0.5, 1.75]], dtype=torch.float32)
    out = model(x=x)

    alpha_p = torch.nn.functional.softplus(state_dict["act.alpha_p"])
    beta = state_dict["act.beta"]
    alpha_n = beta + torch.nn.functional.softplus(state_dict["act.alpha_n"])
    eps = state_dict["act.eps"]
    expected = torch.where(
        x > 0,
        alpha_p * x * x + beta * x,
        (torch.expm1(torch.minimum(x, eps)) - x) * alpha_n + beta * x,
    )
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


def test_generated_config_primitives_read_model_config_and_defaults() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "config": {
                "hidden_size": 640,
                "name": "gemma3",
                "text_config": {"sliding_window": 512},
            },
            "graph": [
                {"k": {"_op": "_ir_expr", "_bind": "k", "value": "'text_config.sliding_window'"}},
                {"h": {"_op": "config_has", "_args": "k", "_bind": "has_key"}},
                {"s": {"_op": "config_int", "_args": "k", "_bind": "window"}},
                {
                    "d": {
                        "_op": "config_int",
                        "_args": "missing.value",
                        "_bind": "defaulted",
                        "default": 9,
                    }
                },
                {"f": {"_op": "config_float", "_args": "hidden_size", "_bind": "hidden_f"}},
                {"n": {"_op": "config_str", "_args": "name", "_bind": "name"}},
            ],
            "outputs": {
                "has_key": "has_key",
                "window": "window",
                "defaulted": "defaulted",
                "hidden_f": "hidden_f",
                "name": "name",
            },
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="ConfigPrimModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["ConfigPrimModel"]()
    out = model()
    assert out["has_key"] is True
    assert out["window"] == 512
    assert out["defaulted"] == 9
    assert out["hidden_f"] == 640.0
    assert out["name"] == "gemma3"


def test_generated_config_value_and_expression_default() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "config": {
                "hidden_size": 640,
                "rope_scaling": {"long_factor": [1.0, 2.0, 3.0]},
            },
            "graph": [
                {
                    "v": {
                        "_op": "config_value",
                        "_args": "rope_scaling.long_factor",
                        "_bind": "vals",
                    }
                },
                {
                    "d": {
                        "_op": "config_int",
                        "_args": "missing.hidden_size",
                        "_bind": "fallback_hidden",
                        "default": {
                            "_expr": "call",
                            "callee": "Config.int",
                            "args": ["hidden_size"],
                            "kwargs": {},
                        },
                    }
                },
            ],
            "outputs": {"vals": "vals", "fallback_hidden": "fallback_hidden"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="ConfigValueModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["ConfigValueModel"]()
    out = model()
    assert out["vals"] == [1.0, 2.0, 3.0]
    assert out["fallback_hidden"] == 640


def test_generated_config_primitives_support_root_kwarg() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "config": {"text_config": {"hidden_size": 4096}},
            "graph": [
                {
                    "h": {
                        "_op": "config_has",
                        "_args": "hidden_size",
                        "_bind": "has_h",
                        "root": "text_config",
                    }
                },
                {
                    "i": {
                        "_op": "config_int",
                        "_args": "hidden_size",
                        "_bind": "h",
                        "root": "text_config",
                    }
                },
            ],
            "outputs": {"has_h": "has_h", "h": "h"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="ConfigRootModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["ConfigRootModel"]()
    out = model()
    assert out["has_h"] is True
    assert out["h"] == 4096


def test_generated_params_primitives_detect_and_select_param_root() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {
                    "h": {
                        "_op": "params_has_root",
                        "_args": "language_model",
                        "_bind": "has_lm",
                    }
                },
                {
                    "r": {
                        "_op": "params_root",
                        "_args": "language_model",
                        "_bind": "root",
                        "default": "",
                    }
                },
            ],
            "outputs": {"has_lm": "has_lm", "root": "root"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="ParamsRootModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["ParamsRootModel"](
        state_dict={"language_model.embed_tokens.weight": torch.randn(8, 4)}
    )
    out = model()
    assert out["has_lm"] is True
    assert out["root"] == "language_model"


def test_generated_param_root_guides_parameter_resolution() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [
                {"r": {"_op": "_ir_expr", "_bind": "root", "value": "'language_model'"}},
                {
                    "n": {
                        "_op": "linear",
                        "_args": "x",
                        "_bind": "y",
                        "_params": {"weight": "proj.weight", "bias": "proj.bias"},
                        "_param_root": {"_expr": "name", "id": "root"},
                    }
                },
            ],
            "outputs": {"y": "y"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="ParamRootExprModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["ParamRootExprModel"](
        state_dict={
            "language_model.proj.weight": torch.eye(2, dtype=torch.float32),
            "language_model.proj.bias": torch.zeros(2, dtype=torch.float32),
        }
    )
    out = model(x=torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    assert torch.equal(out["y"], torch.tensor([[1.0, 2.0]], dtype=torch.float32))


def test_generated_param_root_from_select_supports_linear_trace_codegen() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [
                {
                    "has_root": {
                        "_op": "params_has_root",
                        "_args": "language_model",
                        "_bind": "has_lm",
                    }
                },
                {
                    "root_select": {
                        "_op": "select",
                        "_bind": "root",
                        "cond": "has_lm",
                        "_then_bind": "root_then",
                        "_else_bind": "root_else",
                        "_then": [
                            {
                                "then_value": {
                                    "_op": "_ir_expr",
                                    "_bind": "root_then",
                                    "value": "'language_model'",
                                }
                            }
                        ],
                        "_else": [
                            {
                                "else_value": {
                                    "_op": "_ir_expr",
                                    "_bind": "root_else",
                                    "value": "''",
                                }
                            }
                        ],
                    }
                },
                {
                    "proj": {
                        "_op": "linear",
                        "_args": "x",
                        "_bind": "y",
                        "_params": {"weight": "proj.weight", "bias": "proj.bias"},
                        "_param_root": {"_expr": "name", "id": "root"},
                        "bias": True,
                    }
                },
            ],
            "outputs": {"y": "y"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="ParamRootSelectTraceModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["ParamRootSelectTraceModel"](
        state_dict={
            "language_model.proj.weight": torch.eye(2, dtype=torch.float32),
            "language_model.proj.bias": torch.zeros(2, dtype=torch.float32),
        }
    )
    model._trace_enabled = True
    out = model(x=torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    assert torch.equal(out["y"], torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    assert model.trace_ops[-1]["weight_path"] == "language_model.proj.weight"


def test_generated_model_prefers_existing_param_candidate_path() -> None:
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
                        "_params": {
                            "weight": ["missing.weight", "live.weight"],
                            "bias": ["missing.bias", "live.bias"],
                        },
                    }
                }
            ],
            "outputs": {"y": "y"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="ParamCandidateModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["ParamCandidateModel"](
        state_dict={
            "live.weight": torch.eye(2, dtype=torch.float32),
            "live.bias": torch.zeros(2, dtype=torch.float32),
        }
    )
    out = model(x=torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    assert torch.equal(out["y"], torch.tensor([[1.0, 2.0]], dtype=torch.float32))


def test_generated_scope_path_bound_calls_do_not_double_apply_local_scope() -> None:
    modules = parse_axon_program(
        """
D = 2
lin :: @Path -> Tensor[B,T,D] -> Tensor[B,T,D]
lin@path x = linear@path x dim=D bias=true
tiny :: Tensor[B,T,D] -> Tensor[B,T,D]
tiny x = do
  y <- scope@attn do
    return lin@proj x
  return y
"""
    )
    spec = lower_axon_program_to_synapse_spec(modules)
    source = emit_model_code_from_synapse_spec(spec, class_name="ScopedExplicitParamModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["ScopedExplicitParamModel"](
        state_dict={
            "attn.proj.weight": torch.eye(2, dtype=torch.float32),
            "attn.proj.bias": torch.zeros(2, dtype=torch.float32),
        }
    )
    out = model(x=torch.tensor([[[1.0, 2.0]]], dtype=torch.float32))
    assert torch.equal(out["y"], torch.tensor([[[1.0, 2.0]]], dtype=torch.float32))


def test_generated_layernorm_allows_explicit_weight_and_bias_names() -> None:
    modules = parse_axon_program(
        """
D = 2
tiny :: Tensor[B,T,D] -> Tensor[B,T,D]
tiny x = do
  y <- layernorm@norm x eps=1e-05 weight=gamma bias=beta
  return y
"""
    )
    spec = lower_axon_program_to_synapse_spec(modules)
    source = emit_model_code_from_synapse_spec(spec, class_name="ExplicitLayerNormParamModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    gamma = torch.tensor([1.5, 0.5], dtype=torch.float32)
    beta = torch.tensor([-0.25, 0.75], dtype=torch.float32)
    model = namespace["ExplicitLayerNormParamModel"](
        state_dict={
            "norm.gamma": gamma,
            "norm.beta": beta,
        }
    )
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    out = model(x=x)["y"]
    expected = torch.nn.functional.layer_norm(x, (2,), weight=gamma, bias=beta, eps=1.0e-5)
    assert torch.allclose(out, expected, atol=1.0e-6, rtol=0.0)


@pytest.mark.parametrize("bias_literal", ["false", "null"])
def test_generated_layernorm_allows_disabling_bias_via_literal(bias_literal: str) -> None:
    modules = parse_axon_program(
        f"""
D = 2
tiny :: Tensor[B,T,D] -> Tensor[B,T,D]
tiny x = do
  y <- layernorm@norm x eps=1e-05 weight=gamma bias={bias_literal}
  return y
"""
    )
    spec = lower_axon_program_to_synapse_spec(modules)
    source = emit_model_code_from_synapse_spec(spec, class_name="NoBiasLayerNormModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    gamma = torch.tensor([1.5, 0.5], dtype=torch.float32)
    model = namespace["NoBiasLayerNormModel"](state_dict={"norm.gamma": gamma})
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    out = model(x=x)["y"]
    expected = torch.nn.functional.layer_norm(x, (2,), weight=gamma, bias=None, eps=1.0e-5)
    assert torch.allclose(out, expected, atol=1.0e-6, rtol=0.0)


def test_generated_select_is_lazy_and_value_producing() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"flag": {"optional": False}, "zero": {"optional": True}},
            "graph": [
                {
                    "pick": {
                        "_op": "select",
                        "cond": "flag",
                        "_bind": "y",
                        "_then_bind": "then_v",
                        "_else_bind": "else_v",
                        "_then": [
                            {"then_node": {"_op": "_ir_expr", "value": 1, "_bind": "then_v"}}
                        ],
                        "_else": [
                            {
                                "else_node": {
                                    "_op": "_ir_expr",
                                    "value": "1 / zero",
                                    "_bind": "else_v",
                                }
                            }
                        ],
                    }
                }
            ],
            "outputs": {"y": "y"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="SelectModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["SelectModel"]()
    out = model(flag=True)
    assert isinstance(out, dict)
    assert out["y"] == 1


def test_generated_ir_expr_supports_inline_sqrt_with_config_call() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "config": {"text_config": {"query_pre_attn_scalar": 256}},
            "graph": [
                {
                    "a": {
                        "_op": "_ir_expr",
                        "_bind": "attn_scale",
                        "value": '1.0 / sqrt (Config.float "query_pre_attn_scalar" root="text_config" default=256.0)',
                    }
                }
            ],
            "outputs": {"attn_scale": "attn_scale"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="IRExprConfigCallModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["IRExprConfigCallModel"]()
    out = model()
    assert out["attn_scale"] == pytest.approx(0.0625)


def test_generated_ir_expr_supports_inline_params_root_call() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {
                    "x": {
                        "_op": "_ir_expr",
                        "_bind": "chosen",
                        "value": 'Params.root "language_model" default=""',
                    }
                },
                {
                    "h": {
                        "_op": "_ir_expr",
                        "_bind": "has",
                        "value": 'Params.has_root "language_model"',
                    }
                },
            ],
            "outputs": {"chosen": "chosen", "has": "has"},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="IRExprParamsCallModel")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - generated test code
    model = namespace["IRExprParamsCallModel"](
        state_dict={"language_model.embed_tokens.weight": torch.randn(8, 4)}
    )
    out = model()
    assert out["chosen"] == "language_model"
    assert out["has"] is True
