from __future__ import annotations

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
