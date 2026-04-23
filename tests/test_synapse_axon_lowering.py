from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brainsurgery.synapse import (
    SynapseProgramModel,
    emit_model_code_from_synapse_spec,
    lower_axon_program_to_synapse_spec,
    parse_axon_program,
    parse_axon_program_from_path,
)


def _spec_text(spec: object) -> str:
    return repr(spec)


def test_lowering_preprocesses_simple_surface_program() -> None:
    program = parse_axon_program(
        """
main :: Int -> Int
main x = do
  y <- x + 1
  return y
"""
    )
    spec = lower_axon_program_to_synapse_spec(program, main_module="main")
    assert spec["synapse"] == 1
    assert "main" not in spec["model"].get("blocks", {})


def test_lowering_preprocesses_surface_generic_gpt2_without_scope_or_root() -> None:
    program = parse_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2-kv.axon")
    )
    spec = lower_axon_program_to_synapse_spec(program, main_module="gpt2")
    text = _spec_text(spec)
    assert "_scope" not in text
    assert "_param_root" not in text
    assert "'_op': 'for'" not in text
    assert "'_op': 'repeat'" not in text
    assert "_repeat" not in text
    assert "'_abs_path'" in text
    assert "'_op': 'embedding'" in text
    assert "param_base" not in text
    assert spec["synapse"] == 1
    assert len(spec["model"].get("blocks", {})) > 0


def test_backends_pruned_for_lowered_generic_gpt2_contract() -> None:
    program = parse_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2-kv.axon")
    )
    spec = lower_axon_program_to_synapse_spec(program, main_module="gpt2")

    runtime_model = SynapseProgramModel.from_spec(spec, state_dict={})
    assert not hasattr(runtime_model, "_param_roots_stack")

    simple_program = parse_axon_program(
        """
main :: Int -> Int
main x = do
  y <- x + 1
  return y
"""
    )
    simple_spec = lower_axon_program_to_synapse_spec(simple_program, main_module="main")
    source = emit_model_code_from_synapse_spec(simple_spec, class_name="Generated")
    assert "_param_roots_stack" not in source
    assert "_node_scope(" not in source
    assert "_current_param_roots(" not in source
    assert "['_scope']" not in source
    assert '"_scope"' not in source
    assert "_param_root" not in source


def test_lowering_rejects_repeat_primitive() -> None:
    program = parse_axon_program(
        """
main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- _repeat x 2 -1
  return y
"""
    )
    with pytest.raises(ValueError, match="_repeat must be flattened away before lowering"):
        lower_axon_program_to_synapse_spec(program, main_module="main")


def test_codegen_resolves_shape_symbol_strings_in_flat_ops() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [{"n_op_1": {"_op": "arange", "_args": ["q", "0", "Q"], "_bind": "out"}}],
            "inputs": {"q": {"optional": False}},
            "outputs": {"out": "out"},
            "symbols": {},
            "types": {"block_io": {"main": {"inputs": {"q": "Tensor[B,Q]"}}}},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="Generated")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - test-controlled generated source
    model_cls = namespace["Generated"]
    model = model_cls.from_state_dict({})  # type: ignore[attr-defined]

    out = model(q=torch.zeros((2, 5), dtype=torch.long))

    assert torch.equal(out["out"], torch.arange(5, dtype=torch.long))


def test_codegen_refreshes_shape_aliases_from_block_outputs() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {
                    "n_call_1": {
                        "_op": "call",
                        "_target": "mk",
                        "_args": ["cache", "x"],
                        "_bind": "y",
                    }
                },
                {"n_op_2": {"_op": "sqrt", "_args": "DH", "_bind": "out"}},
            ],
            "blocks": {
                "mk": {
                    "graph": [],
                    "inputs": {"cache": {"optional": True}, "x": {"optional": False}},
                    "outputs": {"y": "x"},
                }
            },
            "inputs": {"cache": {"optional": True}, "x": {"optional": False}},
            "outputs": {"out": "out"},
            "symbols": {},
            "types": {
                "block_io": {
                    "main": {
                        "inputs": {
                            "cache": "?Tensor[B,H,S,DH]",
                            "x": "Tensor[B,H,S,DH]",
                        }
                    },
                    "mk": {
                        "inputs": {
                            "cache": "?Tensor[B,H,S,DH]",
                            "x": "Tensor[B,H,S,DH]",
                        },
                        "outputs": {"y": "Tensor[B,H,S,DH]"},
                    },
                }
            },
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="Generated")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - test-controlled generated source
    model_cls = namespace["Generated"]
    model = model_cls.from_state_dict({})  # type: ignore[attr-defined]

    out = model(cache=None, x=torch.zeros((2, 3, 4, 5), dtype=torch.float32))

    assert out["out"] == pytest.approx(5**0.5)


def test_codegen_does_not_overwrite_existing_shape_alias_with_nullable_output() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {
                    "n_call_1": {
                        "_op": "call",
                        "_target": "mk",
                        "_args": ["cache", "x"],
                        "_bind": "cache2",
                    }
                },
                {"n_op_2": {"_op": "sqrt", "_args": "DH", "_bind": "out"}},
            ],
            "blocks": {
                "mk": {
                    "graph": [],
                    "inputs": {"cache": {"optional": True}, "x": {"optional": False}},
                    "outputs": {"cache2": "cache"},
                }
            },
            "inputs": {"cache": {"optional": True}, "x": {"optional": False}},
            "outputs": {"out": "out"},
            "symbols": {},
            "types": {
                "block_io": {
                    "main": {
                        "inputs": {
                            "cache": "?List[(Tensor[B,H,S,DH], Tensor[B,H,S,DH])]",
                            "x": "Tensor[B,H,S,DH]",
                        }
                    },
                    "mk": {
                        "inputs": {
                            "cache": "?List[(Tensor[B,H,S,DH], Tensor[B,H,S,DH])]",
                            "x": "Tensor[B,H,S,DH]",
                        },
                        "outputs": {"cache2": "?List[(Tensor[B,H,S,DH], Tensor[B,H,S,DH])]"},
                    },
                }
            },
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="Generated")
    assert "cache2[0][0].shape[3]" not in source
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - test-controlled generated source
    model_cls = namespace["Generated"]
    model = model_cls.from_state_dict({})  # type: ignore[attr-defined]

    out = model(cache=None, x=torch.zeros((2, 3, 4, 5), dtype=torch.float32))

    assert out["out"] == pytest.approx(5**0.5)


def test_codegen_parses_null_scalar_tokens_in_flat_ops() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [{"n_op_1": {"_op": "softmax", "_args": ["x", "-1", "null"], "_bind": "out"}}],
            "inputs": {"x": {"optional": False}},
            "outputs": {"out": "out"},
            "symbols": {},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="Generated")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - test-controlled generated source
    model_cls = namespace["Generated"]
    model = model_cls.from_state_dict({})  # type: ignore[attr-defined]

    x = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    out = model(x=x)

    assert torch.allclose(out["out"], torch.softmax(x, dim=-1))
