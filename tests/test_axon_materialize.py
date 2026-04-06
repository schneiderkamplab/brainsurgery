from __future__ import annotations

import json
from pathlib import Path

from brainsurgery.synapse.axon_materialize import run_axon_materialize


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

    written = run_axon_materialize(axon_path=axon_path, models_root=tmp_path / "models")

    assert written == [axon_dir / "gemma-3-270m.axon", axon_dir / "gemma-3-1b.axon"]
    rendered_270m = (axon_dir / "gemma-3-270m.axon").read_text(encoding="utf-8")
    rendered_1b = (axon_dir / "gemma-3-1b.axon").read_text(encoding="utf-8")
    assert 'CHECKPOINTS ["google/gemma-3-270m", "google/gemma-3-270m-it"]' in rendered_270m
    assert 'CHECKPOINTS ["google/gemma-3-1b-pt", "google/gemma-3-1b-it"]' in rendered_1b
    assert "Config." not in rendered_270m
    assert "Params." not in rendered_270m
    assert "Config." not in rendered_1b
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
                'CFG = (Config.has "text_config") ? "text_config" : ""',
                'ROOT = Params.root "language_model" default=""',
                'D = Config.int "hidden_size" root=CFG default=16',
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

    written = run_axon_materialize(axon_path=axon_path, models_root=tmp_path / "weights")

    assert written == [axon_dir / "toy.axon"]
    rendered = (axon_dir / "toy.axon").read_text(encoding="utf-8")
    assert 'CHECKPOINTS ["org/toy-pt", "org/toy-it"]' in rendered
    assert "Config." not in rendered
    assert "Params." not in rendered
    assert 'ROOT = "language_model"' in rendered
    assert "D = 32" in rendered
    assert "scope@model root=ROOT" in rendered
