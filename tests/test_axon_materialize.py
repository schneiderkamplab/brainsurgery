from __future__ import annotations

import json
from pathlib import Path

from brainsurgery.cli.synapse_materialize import run_axon_materialize_workflow
from brainsurgery.synapse import (
    ast_equal,
    load_materialize_context,
    materialize_axon_file,
    parse_axon_program_from_path,
    render_axon_file,
)


def test_gemma3_materialize_reads_checkpoint_pragma_and_groups_variants(tmp_path: Path) -> None:
    axon_dir = tmp_path / "brainsurgery" / "synapse" / "models" / "gemma" / "gemma3"
    axon_dir.mkdir(parents=True)
    axon_path = axon_dir / "gemma3.axon"
    source = Path("brainsurgery/synapse/models/gemma3/generic-gemma-3.axon").read_text(
        encoding="utf-8"
    )
    source = source.replace(
        '{-# CHECKPOINTS ["google/gemma-3-270m", "google/gemma-3-270m-it", "google/gemma-3-1b-pt", "google/gemma-3-1b-it", "google/gemma-3-4b-pt", "google/gemma-3-4b-it", "google/gemma-3-12b-pt", "google/gemma-3-12b-it", "google/gemma-3-27b-pt", "google/gemma-3-27b-it"] #-}',
        '{-# CHECKPOINTS ["google/gemma-3-270m", "google/gemma-3-270m-it", "google/gemma-3-1b-pt", "google/gemma-3-1b-it"] #-}',
    )
    axon_path.write_text(source, encoding="utf-8")

    model_root = tmp_path / "models" / "google"
    model_root.mkdir(parents=True)
    payload_270m = {
        "model_type": "gemma3_text",
        "hidden_size": 640,
        "vocab_size": 262144,
        "num_hidden_layers": 18,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "intermediate_size": 2048,
        "_sliding_window_pattern": 6,
        "rope_local_base_freq": 10000.0,
        "rope_theta": 1000000.0,
        "sliding_window": 512,
        "max_position_embeddings": 32768,
        "query_pre_attn_scalar": 256,
    }
    payload_1b = {
        "model_type": "gemma3_text",
        "hidden_size": 1152,
        "vocab_size": 262144,
        "num_hidden_layers": 26,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "intermediate_size": 6912,
        "sliding_window_pattern": 6,
        "rope_local_base_freq": 10000.0,
        "rope_theta": 1000000.0,
        "sliding_window": 512,
        "max_position_embeddings": 32768,
        "query_pre_attn_scalar": 256,
    }
    for name, payload in (
        ("gemma-3-270m", payload_270m),
        ("gemma-3-270m-it", payload_270m),
        ("gemma-3-1b-pt", payload_1b),
        ("gemma-3-1b-it", payload_1b),
    ):
        d = model_root / name
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps(payload), encoding="utf-8")
        if "270m" in name:
            embed_shape = [262144, 640]
            q_shape = [1024, 640]
            k_shape = [256, 640]
        else:
            embed_shape = [262144, 1152]
            q_shape = [1024, 1152]
            k_shape = [256, 1152]
        import torch
        from safetensors.torch import save_file

        save_file(
            {
                "model.embed_tokens.weight": torch.zeros(embed_shape),
                "model.layers.0.self_attn.q_proj.weight": torch.zeros(q_shape),
                "model.layers.0.self_attn.k_proj.weight": torch.zeros(k_shape),
            },
            str(d / "model.safetensors"),
        )

    written = run_axon_materialize_workflow(axon_path=axon_path, models_root=tmp_path / "models")

    assert written == [axon_dir / "gemma-3-270m.axon", axon_dir / "gemma-3-1b.axon"]
    rendered_270m = (axon_dir / "gemma-3-270m.axon").read_text(encoding="utf-8")
    rendered_1b = (axon_dir / "gemma-3-1b.axon").read_text(encoding="utf-8")
    assert '{-# CHECKPOINTS ["google/gemma-3-270m", "google/gemma-3-270m-it"] #-}' in rendered_270m
    assert '{-# CHECKPOINTS ["google/gemma-3-1b-pt", "google/gemma-3-1b-it"] #-}' in rendered_1b
    assert "Config.int" not in rendered_270m
    assert "Config.float" not in rendered_270m
    assert "Config.int" not in rendered_1b
    assert "Config.float" not in rendered_1b
    assert "Params." not in rendered_270m
    assert "Params." not in rendered_1b
    assert not (axon_dir / "gemma-3-270m-it.axon").exists()
    assert not (axon_dir / "gemma-3-1b-it.axon").exists()


def test_generic_materialize_works_for_non_gemma3_with_config_and_params(tmp_path: Path) -> None:
    axon_dir = tmp_path / "modelsrc"
    axon_dir.mkdir(parents=True)
    axon_path = axon_dir / "toy.axon"
    axon_path.write_text(
        "\n".join(
            [
                '{-# CHECKPOINTS ["org/toy-pt", "org/toy-it"] #-}',
                "",
                "import Config",
                "import Params",
                "",
                'CFG = Config.has_key@text_config ? "text_config" : ""',
                'ROOT = Params.root "language_model" default=""',
                "D = Config.int @'{CFG}.hidden_size' default=16",
                "",
                "toy :: TokenIds[B,S] -> Tensor[B,S,D]",
                "toy input_ids = do",
                "  x <- scope@model root=ROOT do",
                "    return embedding@embed_tokens input_ids dim=D",
                "  return x",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model_root = tmp_path / "weights" / "org"
    model_root.mkdir(parents=True)
    payload = {"text_config": {"hidden_size": 32}}
    for name in ("toy-pt", "toy-it"):
        d = model_root / name
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps(payload), encoding="utf-8")
        import torch
        from safetensors.torch import save_file

        save_file(
            {"language_model.embed_tokens.weight": torch.zeros([128, 32])},
            str(d / "model.safetensors"),
        )

    written = run_axon_materialize_workflow(axon_path=axon_path, models_root=tmp_path / "weights")

    assert written == [axon_dir / "toy.axon"]
    rendered = (axon_dir / "toy.axon").read_text(encoding="utf-8")
    assert '{-# CHECKPOINTS ["org/toy-pt", "org/toy-it"] #-}' in rendered
    assert "import Config" in rendered
    assert "import Params" in rendered
    assert 'CFG = true ? "text_config" : ""' in rendered
    assert 'ROOT = Params.root "language_model" default=""' in rendered
    assert "D = 32" in rendered
    assert "root=ROOT" in rendered
    assert "dim=D" in rendered


def test_materialize_groups_identical_instruct_variant_by_body(tmp_path: Path) -> None:
    axon_dir = tmp_path / "modelsrc"
    axon_dir.mkdir(parents=True)
    axon_path = axon_dir / "toy.axon"
    axon_path.write_text(
        "\n".join(
            [
                '{-# CHECKPOINTS ["org/toy", "org/toy-Instruct"] #-}',
                "",
                "import Config",
                "",
                "D = Config.int@hidden_size default=16",
                "",
                "toy :: TokenIds[B,S] -> Tensor[B,S,D]",
                "toy input_ids = do",
                "  x <- embedding@embed_tokens input_ids dim=D",
                "  return x",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model_root = tmp_path / "weights" / "org"
    model_root.mkdir(parents=True)
    payload = {"hidden_size": 32}
    for name in ("toy", "toy-Instruct"):
        d = model_root / name
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps(payload), encoding="utf-8")
        import torch
        from safetensors.torch import save_file

        save_file({"embed_tokens.weight": torch.zeros([128, 32])}, str(d / "model.safetensors"))

    written = run_axon_materialize_workflow(axon_path=axon_path, models_root=tmp_path / "weights")

    assert written == [axon_dir / "toy.axon"]
    rendered = (axon_dir / "toy.axon").read_text(encoding="utf-8")
    assert '{-# CHECKPOINTS ["org/toy", "org/toy-Instruct"] #-}' in rendered
    assert "import Config" in rendered
    assert "D = 32" in rendered
    assert not (axon_dir / "toy-Instruct.axon").exists()


def test_materialize_resolves_config_calls_inside_module_body_using_constant_env(
    tmp_path: Path,
) -> None:
    axon_dir = tmp_path / "modelsrc"
    axon_dir.mkdir(parents=True)
    axon_path = axon_dir / "toy.axon"
    axon_path.write_text(
        "\n".join(
            [
                '{-# CHECKPOINTS ["org/toy"] #-}',
                "",
                "import Config",
                "",
                'CFG = Config.has_key@text_config ? "text_config" : ""',
                "",
                "toy :: Tensor[B,S,D] -> Tensor[B,S,D]",
                "toy x = do",
                "  eps <- Config.float @'{CFG}.rms_norm_eps' default=1e-05",
                "  y <- rmsnorm@norm x eps=eps",
                "  return y",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model_root = tmp_path / "weights" / "org" / "toy"
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text(
        json.dumps({"text_config": {"rms_norm_eps": 2.5e-5}}),
        encoding="utf-8",
    )
    import torch
    from safetensors.torch import save_file

    save_file({"norm.weight": torch.zeros([16])}, str(model_root / "model.safetensors"))

    written = run_axon_materialize_workflow(axon_path=axon_path, models_root=tmp_path / "weights")
    assert written == [axon_dir / "toy.axon"]
    rendered = (axon_dir / "toy.axon").read_text(encoding="utf-8")
    assert "import Config" in rendered
    assert "Config.float" not in rendered
    assert "eps <- 2.5e-05" in rendered or "eps <- 2.5e-5" in rendered


def test_materialize_instantiates_scope_prefix_templates_from_env(tmp_path: Path) -> None:
    axon_dir = tmp_path / "modelsrc"
    axon_dir.mkdir(parents=True)
    axon_path = axon_dir / "toy.axon"
    axon_path.write_text(
        "\n".join(
            [
                '{-# CHECKPOINTS ["org/toy"] #-}',
                "",
                "import Params (has_root)",
                "",
                'PARAM_ROOT = (has_root "language_model") ? "language_model" : ""',
                "",
                "toy :: TokenIds[B,S] -> Tensor[B,S,D]",
                "toy input_ids = do",
                "  x <- scope@'{PARAM_ROOT}.model' do",
                "    return embedding@embed_tokens input_ids dim=16",
                "  return x",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model_root = tmp_path / "weights" / "org" / "toy"
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    import torch
    from safetensors.torch import save_file

    save_file(
        {"language_model.model.embed_tokens.weight": torch.zeros([128, 16])},
        str(model_root / "model.safetensors"),
    )

    written = run_axon_materialize_workflow(axon_path=axon_path, models_root=tmp_path / "weights")
    assert written == [axon_dir / "toy.axon"]

    rendered = (axon_dir / "toy.axon").read_text(encoding="utf-8")
    assert 'PARAM_ROOT = (has_root "language_model") ? "language_model" : ""' in rendered
    assert "scope@'{PARAM_ROOT}.model' do" in rendered


def test_materialize_workflow_written_file_roundtrips_to_expected_ast(tmp_path: Path) -> None:
    axon_dir = tmp_path / "modelsrc"
    axon_dir.mkdir(parents=True)
    axon_path = axon_dir / "toy.axon"
    axon_path.write_text(
        "\n".join(
            [
                '{-# CHECKPOINTS ["org/toy"] #-}',
                "",
                "import Config",
                "",
                "D = Config.int@hidden_size default=16",
                "",
                "toy :: Tensor[B,S,D] -> Tensor[B,S,D]",
                "toy x = do",
                "  y <- linear@proj x dim=D",
                "  return y",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model_root = tmp_path / "weights" / "org" / "toy"
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text(json.dumps({"hidden_size": 24}), encoding="utf-8")
    import torch
    from safetensors.torch import save_file

    save_file({"proj.weight": torch.zeros([24, 24])}, str(model_root / "model.safetensors"))

    written = run_axon_materialize_workflow(axon_path=axon_path, models_root=tmp_path / "weights")
    assert written == [axon_dir / "toy.axon"]

    original = parse_axon_program_from_path(axon_path)
    context = load_materialize_context(checkpoint="org/toy", models_root=tmp_path / "weights")
    expected = materialize_axon_file(original, context=context)
    expected_path = written[0].with_suffix(".expected.axon")
    expected_path.unlink(missing_ok=True)
    expected_path.write_text(
        render_axon_file(
            expected.__class__(
                modules=expected.modules,
                imports=expected.imports,
                imported_members=dict(expected.imported_members),
                exports=expected.exports,
                pragmas={"checkpoints": "org/toy"},
                constants=dict(expected.constants),
                type_aliases=dict(expected.type_aliases),
                origin_path=expected.origin_path,
            )
        ),
        encoding="utf-8",
    )
    expected = parse_axon_program_from_path(expected_path)
    reparsed = parse_axon_program_from_path(written[0])
    assert ast_equal(reparsed, expected)
