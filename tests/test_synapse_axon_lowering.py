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


def test_lowering_preserves_path_typed_block_inputs_for_gemma3_scope() -> None:
    program = parse_axon_program_from_path(
        Path("brainsurgery/synapse/models/gemma3/gemma-3-270m.axon")
    )
    spec = lower_axon_program_to_synapse_spec(program, main_module="gemma3")

    block_io = spec["model"]["types"]["block_io"]
    assert block_io["gemma3_body"]["inputs"]["__scope"] == "Path"
    assert block_io["NN.embedding"]["inputs"]["path"] == "Path"

    body_graph = spec["model"]["blocks"]["gemma3_body"]["graph"]
    embedding_call = next(
        node["n_call_13"] for node in body_graph if "n_call_13" in node
    )
    assert embedding_call["_args"][0] == "@@'{__scope}.embed_tokens'"

    embedding_graph = spec["model"]["blocks"]["NN.embedding"]["graph"]
    embedding_node = next(node["n_op_1"] for node in embedding_graph if "n_op_1" in node)
    assert embedding_node["_abs_path"] == {
        "_expr": "path",
        "absolute": True,
        "parts": ["{path}"],
    }


def test_lowering_keeps_config_symbols_as_path_payloads_for_gpt2() -> None:
    program = parse_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2-kv.axon")
    )
    spec = lower_axon_program_to_synapse_spec(program, main_module="gpt2")

    symbols = spec["model"]["symbols"]
    num_heads = symbols["NUM_HEADS"]
    assert num_heads["callee"] == "Config.dim"
    assert num_heads["args"][0] == {
        "_expr": "path",
        "absolute": True,
        "parts": ["n_head"],
    }


def test_lowering_canonicalizes_generated_dim_terms_before_codegen() -> None:
    program = parse_axon_program_from_path(
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2-kv.axon")
    )
    spec = lower_axon_program_to_synapse_spec(program, main_module="gpt2")

    h1_graph = spec["model"]["blocks"]["gpt2_h1"]["graph"]
    sqrt_node = next(node for raw in h1_graph for node in raw.values() if node.get("_op") == "sqrt")
    assert sqrt_node["_args"] == "DH"

    shape_node = next(
        node
        for raw in h1_graph
        for node in raw.values()
        if node.get("_op") == "_ir_expr"
        and isinstance(node.get("value"), list)
        and node.get("_bind") == "_v16"
    )
    assert shape_node["value"] == [
        {"_expr": "name", "id": "B"},
        {"_expr": "name", "id": "S"},
        {
            "_expr": "binary",
            "op": "*",
            "left": {"_expr": "name", "id": "H"},
            "right": {"_expr": "name", "id": "DH"},
        },
    ]

    emit_model_code_from_synapse_spec(spec, class_name="Generated")


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


def test_lowering_supports_repeat_primitive_via_repeat_op() -> None:
    program = parse_axon_program(
        """
main :: Tensor[B,KVH,T,HD] -> Tensor[B,H,T,HD]
main x = do
  y <- _repeat x 2 1
  return y
"""
    )
    spec = lower_axon_program_to_synapse_spec(program, main_module="main")
    text = _spec_text(spec)
    assert "'_op': 'repeat'" in text
    assert "_repeat" not in text


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


def test_codegen_sanitizes_internal_dunder_block_params() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {
                    "n_call_1": {
                        "_op": "call",
                        "_target": "wrap",
                        "_args": ["@@root", "x"],
                        "_bind": "out",
                    }
                }
            ],
            "blocks": {
                "wrap": {
                    "graph": [],
                    "inputs": {"__scope": {"optional": False}, "x": {"optional": False}},
                    "outputs": {"out": "x"},
                }
            },
            "inputs": {"x": {"optional": False}},
            "outputs": {"out": "out"},
            "symbols": {},
            "types": {"block_io": {"wrap": {"inputs": {"path": "Path"}}}},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="Generated")

    assert "def _block_wrap(self, __scope" not in source
    assert "(__scope=" not in source
    assert ", __scope=" not in source
    assert "def _block_wrap(self, v__scope" in source
    assert "v__scope='@@root'" in source


def test_codegen_resolves_nested_path_param_scopes_for_params() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {
                    "n_call_1": {
                        "_op": "call",
                        "_target": "wrap",
                        "_args": ["@@root", "x"],
                        "_bind": "out",
                    }
                }
            ],
            "blocks": {
                "wrap": {
                    "graph": [
                        {
                            "n_op_1": {
                                "_op": "embedding",
                                "_bind": "y",
                                "_args": ["x", "2"],
                                "_abs_path": {
                                    "_expr": "path",
                                    "absolute": True,
                                    "parts": ["{path}"],
                                },
                                "weight": "weight",
                                "_params": {"weight": "weight"},
                            }
                        }
                    ],
                    "inputs": {"path": {"optional": False}, "x": {"optional": False}},
                    "outputs": {"out": "y"},
                }
            },
            "inputs": {"x": {"optional": False}},
            "outputs": {"out": "out"},
            "symbols": {},
            "types": {"block_io": {"wrap": {"inputs": {"path": "Path"}}}},
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="Generated")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - test-controlled generated source
    model_cls = namespace["Generated"]
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    model = model_cls.from_state_dict({"root.weight": weight})  # type: ignore[attr-defined]

    out = model(x=torch.tensor([[1]], dtype=torch.long))

    assert torch.equal(out["out"], weight[torch.tensor([[1]])])


def test_codegen_resolves_source_named_path_template_args() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {
                    "n_call_1": {
                        "_op": "call",
                        "_target": "body",
                        "_args": ["@@model", "x"],
                        "_bind": "out",
                    }
                }
            ],
            "blocks": {
                "body": {
                    "graph": [
                        {
                            "n_call_1": {
                                "_op": "call",
                                "_target": "NN.embedding",
                                "_args": ["@@'{__scope}.embed_tokens'", "x"],
                                "_bind": "y",
                            }
                        }
                    ],
                    "inputs": {"__scope": {"optional": False}, "x": {"optional": False}},
                    "outputs": {"out": "y"},
                },
                "NN.embedding": {
                    "graph": [
                        {
                            "n_op_1": {
                                "_op": "embedding",
                                "_bind": "y",
                                "_args": ["x", "2"],
                                "_abs_path": {
                                    "_expr": "path",
                                    "absolute": True,
                                    "parts": ["{path}"],
                                },
                                "weight": "weight",
                                "_params": {"weight": "weight"},
                            }
                        }
                    ],
                    "inputs": {"path": {"optional": False}, "x": {"optional": False}},
                    "outputs": {"out": "y"},
                },
            },
            "inputs": {"x": {"optional": False}},
            "outputs": {"out": "out"},
            "symbols": {},
            "types": {
                "block_io": {
                    "body": {"inputs": {"__scope": "Path"}},
                    "NN.embedding": {"inputs": {"path": "Path"}},
                }
            },
        },
    }
    source = emit_model_code_from_synapse_spec(spec, class_name="Generated")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - test-controlled generated source
    model_cls = namespace["Generated"]
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    model = model_cls.from_state_dict({"model.embed_tokens.weight": weight})  # type: ignore[attr-defined]

    out = model(x=torch.tensor([[1]], dtype=torch.long))

    assert torch.equal(out["out"], weight[torch.tensor([[1]])])
