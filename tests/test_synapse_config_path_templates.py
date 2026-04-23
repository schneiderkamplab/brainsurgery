from __future__ import annotations

import torch

from brainsurgery.synapse import SynapseProgramModel, emit_model_code_from_synapse_spec


def _minimal_spec() -> dict:
    return {
        "synapse": 1,
        "model": {
            "config": {"text_config": {"hidden_size": 5376}},
            "graph": [],
            "inputs": {},
            "outputs": {},
            "symbols": {},
        },
    }


def test_runtime_resolves_single_quoted_templated_config_path() -> None:
    model = SynapseProgramModel.from_spec(_minimal_spec(), state_dict={})

    key = model._resolve_config_path_key(  # noqa: SLF001 - regression for runtime helper
        "@'{CFG}.hidden_size'",
        {"CFG": "text_config"},
        "Config",
    )

    assert key == "text_config.hidden_size"


def test_codegen_resolves_single_quoted_templated_config_path() -> None:
    source = emit_model_code_from_synapse_spec(_minimal_spec(), class_name="Generated")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - test-controlled generated source
    model_cls = namespace["Generated"]
    model = model_cls.from_state_dict({})  # type: ignore[attr-defined]

    key = model._resolve_config_path_key(  # noqa: SLF001 - regression for generated helper
        "@'{CFG}.hidden_size'",
        {"CFG": "text_config"},
        "Config",
    )

    assert key == "text_config.hidden_size"


def test_runtime_resolves_structured_templated_config_path() -> None:
    model = SynapseProgramModel.from_spec(_minimal_spec(), state_dict={})

    key = model._resolve_config_path_key(  # noqa: SLF001 - regression for runtime helper
        {"_expr": "path", "absolute": False, "parts": ["{CFG}", "hidden_size"]},
        {"CFG": "text_config"},
        "Config",
    )

    assert key == "text_config.hidden_size"


def test_codegen_resolves_structured_templated_config_path() -> None:
    source = emit_model_code_from_synapse_spec(_minimal_spec(), class_name="Generated")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - test-controlled generated source
    model_cls = namespace["Generated"]
    model = model_cls.from_state_dict({})  # type: ignore[attr-defined]

    key = model._resolve_config_path_key(  # noqa: SLF001 - regression for generated helper
        {"_expr": "path", "absolute": False, "parts": ["{CFG}", "hidden_size"]},
        {"CFG": "text_config"},
        "Config",
    )

    assert key == "text_config.hidden_size"


def test_runtime_eval_expr_call_supports_config_dim() -> None:
    model = SynapseProgramModel.from_spec(_minimal_spec(), state_dict={})

    value = model._eval_expr_call(  # noqa: SLF001 - regression for runtime helper
        "Config.dim",
        ["@'{CFG}.hidden_size'"],
        {"default": 16},
        {"CFG": "text_config"},
        {},
    )

    assert value == 5376


def test_codegen_eval_expr_call_supports_config_dim() -> None:
    source = emit_model_code_from_synapse_spec(_minimal_spec(), class_name="Generated")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - test-controlled generated source
    model_cls = namespace["Generated"]
    model = model_cls.from_state_dict({})  # type: ignore[attr-defined]

    value = model._eval_expr_call(  # noqa: SLF001 - regression for generated helper
        "Config.dim",
        ["@'{CFG}.hidden_size'"],
        {"default": 16},
        {"CFG": "text_config"},
        {},
    )

    assert value == 5376


def test_runtime_pick_param_from_single_resolves_template_env() -> None:
    model = SynapseProgramModel.from_spec(
        _minimal_spec(), state_dict={"h.0.weight": torch.zeros(1)}
    )

    key = model._infer_param_path(  # noqa: SLF001 - regression for runtime helper
        {"_abs_path": {"_expr": "path", "absolute": True, "parts": ["h", "{i}"]}},
        node_path="block",
        param_name="weight",
        env={"i": 0},
    )

    assert key == "h.0.weight"


def test_codegen_pick_param_from_single_resolves_template_env() -> None:
    source = emit_model_code_from_synapse_spec(_minimal_spec(), class_name="Generated")
    namespace: dict[str, object] = {}
    exec(source, namespace)  # noqa: S102 - test-controlled generated source
    model_cls = namespace["Generated"]
    model = model_cls.from_state_dict({"h.0.weight": torch.zeros(1)})  # type: ignore[attr-defined]

    key = model._pick_param_from_single(  # noqa: SLF001 - regression for generated helper
        "h.0",
        "@@'h.{i}.weight'",
        {"i": 0},
    )

    assert key == "h.0.weight"
