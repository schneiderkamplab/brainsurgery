from __future__ import annotations

from types import SimpleNamespace

import safetensors.torch
import torch

from brainsurgery.synapse import axon_test as axon_test_mod


def test_extract_logits_accepts_namespace_logits() -> None:
    logits = torch.randn(2, 3, 5)

    output = SimpleNamespace(logits=logits)

    assert axon_test_mod._extract_logits(output) is logits


def test_should_trust_remote_code_for_local_custom_code_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "modeling_phi3_small.py").write_text("# local custom model\n", encoding="utf-8")

    assert axon_test_mod._should_trust_remote_code(tmp_path, model_config=None) is True


def test_should_trust_remote_code_for_auto_map_config(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "configuration_phi3_small.py").write_text("# local config\n", encoding="utf-8")
    (tmp_path / "modeling_phi3_small.py").write_text("# local model\n", encoding="utf-8")
    model_config = {
        "auto_map": {
            "AutoConfig": "configuration_phi3_small.Phi3SmallConfig",
            "AutoModelForCausalLM": "modeling_phi3_small.Phi3SmallForCausalLM",
        }
    }

    assert axon_test_mod._should_trust_remote_code(tmp_path, model_config=model_config) is True


def test_should_not_trust_remote_code_for_incomplete_local_auto_map(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "modeling_deepseek.py").write_text("# local model\n", encoding="utf-8")
    model_config = {
        "auto_map": {
            "AutoConfig": "configuration_deepseek.DeepseekV2Config",
            "AutoModelForCausalLM": "modeling_deepseek.DeepseekV2ForCausalLM",
        }
    }

    assert axon_test_mod._should_trust_remote_code(tmp_path, model_config=model_config) is False


def test_should_not_trust_remote_code_without_local_signals(tmp_path) -> None:  # type: ignore[no-untyped-def]
    model_config = {"model_type": "llama"}

    assert axon_test_mod._should_trust_remote_code(tmp_path, model_config=model_config) is False


def test_augment_model_config_from_checkpoint_infers_black_mamba_time_step_rank(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    config_path = tmp_path / "config.json"
    config_path.write_text(
        (
            '{"hidden_size": 1152, "num_layers": 28, "state_size": 16, '
            '"expansion_factor": 2, "ffn_hidden_size": 3072, "mamba_moe_layers": ["8", "r"]}'
        ),
        encoding="utf-8",
    )
    weights_path = tmp_path / "model.safetensors"
    safetensors.torch.save_file(
        {"decoder.layers.0.mixer.dt_proj.weight": torch.zeros((2304, 72), dtype=torch.float32)},
        str(weights_path),
    )

    enriched = axon_test_mod._augment_model_config_from_checkpoint(
        model_dir=tmp_path,
        safetensors_files=[weights_path],
        model_config=axon_test_mod._load_model_config(tmp_path),
    )

    assert isinstance(enriched, dict)
    assert enriched["time_step_rank"] == 72


class _FakePhi3LongropeModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.longrope_config = SimpleNamespace(
            short_factor=[1.0, 1.5, 2.0, 2.5],
            long_factor=[3.0, 3.5, 4.0, 4.5],
        )
        self.is_longrope = True
        self.max_seq_len = 8
        self.dim_model = 8
        self.register_buffer("range_vector", torch.full((8,), float("nan")), persistent=False)
        self.register_buffer(
            "short_factors", torch.zeros((4,), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "long_factors", torch.zeros((4,), dtype=torch.float32), persistent=False
        )


class _FakePhi3Wrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rotary = _FakePhi3LongropeModule()


def test_rebuild_hf_phi3small_longrope_buffers_repairs_corrupted_buffers() -> None:
    model = _FakePhi3Wrapper()

    rebuilt = axon_test_mod._rebuild_hf_phi3small_longrope_buffers(model)

    assert rebuilt == 1
    assert torch.equal(model.rotary.range_vector, torch.arange(8, dtype=torch.float32))
    assert torch.equal(
        model.rotary.short_factors,
        torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.float32),
    )
    assert torch.equal(
        model.rotary.long_factors,
        torch.tensor([3.0, 3.5, 4.0, 4.5], dtype=torch.float32),
    )


def test_rebuild_hf_phi3small_longrope_buffers_skips_valid_buffers() -> None:
    model = _FakePhi3Wrapper()
    model.rotary.range_vector = torch.arange(8, dtype=torch.float32)
    model.rotary.short_factors = torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.float32)
    model.rotary.long_factors = torch.tensor([3.0, 3.5, 4.0, 4.5], dtype=torch.float32)

    rebuilt = axon_test_mod._rebuild_hf_phi3small_longrope_buffers(model)

    assert rebuilt == 0
