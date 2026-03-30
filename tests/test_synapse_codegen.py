from __future__ import annotations

from pathlib import Path

import pytest
import torch
import typer
from omegaconf import OmegaConf

from brainsurgery.cli.synapse import axon_to_synapse, axon_visualize, emit_generic, synapse_to_axon
from brainsurgery.synapse import emit_model_code_from_synapse_spec, render_synapse_spec_to_dot


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
    dot = render_synapse_spec_to_dot(spec)
    assert "digraph synapse" in dot
    assert "rankdir=TB;" in dot
    assert "subgraph cluster_blk" in dot
    assert "sum_xy\\nadd" in dot
    assert "sum_xy\\\\nadd" not in dot
    assert 'style=dashed, color="gray65"' in dot
    assert 'label="x"' in dot
    assert 'label="y"' in dot
    assert 'label="z"' in dot
    assert 'label="calls"' in dot
    assert "block blk" in dot


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
    assert 'label="x"' in dot_text


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
    assert "params: proj.weight" in dot


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
                        "_op": "_ir_const",
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
    assert 'label="use_cache"' in dot


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
