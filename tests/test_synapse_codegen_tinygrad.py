from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import typer
from omegaconf import OmegaConf

try:
    from tinygrad import Tensor, dtypes
except ImportError:
    Tensor = None  # type: ignore[assignment,misc]
    dtypes = None  # type: ignore[assignment]

from brainsurgery.cli.synapse import axon_to_synapse, emit_generic, synapse_to_axon
from brainsurgery.synapse import emit_model_code_from_synapse_spec


requires_tinygrad = pytest.mark.skipif(
    Tensor is None, reason="tinygrad not installed"
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
    return """\
synapse: 1
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

    emit_generic(spec_path=spec_path, output_path=out_path, class_name="FromCli", force=False, backend="tinygrad")

    assert out_path.exists()
    contents = out_path.read_text(encoding="utf-8")
    assert "class FromCli:" in contents
    assert "def from_state_dict(cls, state_dict" in contents


def test_cli_emit_requires_force_for_existing_output(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    out_path = tmp_path / "generated_model.py"
    spec_path.write_text(_spec_yaml(), encoding="utf-8")
    out_path.write_text("# existing\n", encoding="utf-8")

    with pytest.raises(typer.BadParameter) as exc_info:
        emit_generic(spec_path=spec_path, output_path=out_path, class_name="FromCli", force=False, backend="tinygrad")

    assert "overwrite" in str(exc_info.value)


def test_emit_accepts_minimal_op_map() -> None:
    bad_op_map = {
        "ops": {
            "embedding": {"target": "tinygrad.nn.Embedding"},
            "linear": {"target": "tinygrad.nn.Linear"},
            "layernorm": {"target": "tinygrad.nn.LayerNorm"},
            "attention": {"target": "tinygrad.nn.Attention"},
        }
    }
    source = emit_model_code_from_synapse_spec(
        _spec_dict(), class_name="BadMap", op_map=bad_op_map, backend="tinygrad",
    )
    assert "class BadMap:" in source
    assert "nn.Module" not in source


def test_emit_generic_from_gemma3_spec(tmp_path: Path) -> None:
    spec_path = Path(__file__).resolve().parents[1] / "examples" / "gemma3_270m_synapse.yaml"
    out_path = tmp_path / "gemma_model.py"
    emit_generic(spec_path=spec_path, output_path=out_path, class_name="Gemma3Synapse", force=False, backend="tinygrad")
    contents = out_path.read_text(encoding="utf-8")
    assert "class Gemma3Synapse:" in contents
    assert "'D': 640" in contents


def test_emit_model_code_from_synapse_spec_generic() -> None:
    source = emit_model_code_from_synapse_spec(
        _spec_dict(), class_name="GenericSynapse", backend="tinygrad",
    )
    assert "class GenericSynapse:" in source
    assert "def generate(self, input_ids:" in source
    assert "Tensor" in source


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
    source = emit_model_code_from_synapse_spec(spec, class_name="OptionalModel", backend="tinygrad")
    assert "def _prepare_env(" in source
    assert "x = env.get('x')" in source


@requires_tinygrad
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
    source = emit_model_code_from_synapse_spec(spec, class_name="IndexSafeModel", backend="tinygrad")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102
    model = namespace["IndexSafeModel"]()
    out = model()
    assert out["x0"] is None


@requires_tinygrad
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
    source = emit_model_code_from_synapse_spec(spec, class_name="LoopModel", backend="tinygrad")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102
    model = namespace["LoopModel"]()
    out = model(zero=Tensor(0.0), one_seed=Tensor(1.0))
    result = out["result"].realize()
    assert float(result.numpy()) == 3.0


@requires_tinygrad
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
    source = emit_model_code_from_synapse_spec(spec, class_name="LoopStepModel", backend="tinygrad")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102
    model = namespace["LoopStepModel"]()
    out = model(zero=Tensor(0.0), one_seed=Tensor(1.0))
    result = out["result"].realize()
    assert float(result.numpy()) == 3.0


@requires_tinygrad
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
    source = emit_model_code_from_synapse_spec(spec, class_name="LinearEmptyModel", backend="tinygrad")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102
    model = namespace["LinearEmptyModel"]()
    model.load_state_dict_tensors({"n.weight": Tensor.randn(8, 4).realize()})
    out = model(x=Tensor.zeros(0, 4, dtype=dtypes.float32).realize())
    assert out["y"].shape == (0, 8)


@requires_tinygrad
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
    source = emit_model_code_from_synapse_spec(spec, class_name="SplitInterleaveModel", backend="tinygrad")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102
    model = namespace["SplitInterleaveModel"]()
    out = model(x=Tensor([[0.0, 1.0, 2.0, 3.0]]))
    assert np.allclose(out["even"].realize().numpy(), np.array([[0.0, 2.0]]))
    assert np.allclose(out["odd"].realize().numpy(), np.array([[1.0, 3.0]]))


@requires_tinygrad
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
    source = emit_model_code_from_synapse_spec(spec, class_name="ClampSigmoidModel", backend="tinygrad")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102
    model = namespace["ClampSigmoidModel"]()
    out = model(x=Tensor([[-2.0, 0.0, 2.0]]))
    expected = Tensor([-1.0, 0.0, 1.0]).sigmoid().realize()
    assert np.allclose(out["y"].realize().numpy(), expected.numpy(), atol=1e-6, rtol=0.0)
