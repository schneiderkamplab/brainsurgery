from __future__ import annotations

from types import SimpleNamespace

import safetensors.torch
import torch

from brainsurgery.synapse import axon_test as axon_test_mod


def test_infer_model_type_from_unambiguous_architecture() -> None:
    assert (
        axon_test_mod._infer_model_type_from_architectures(
            {"architectures": ["UMT5ForConditionalGeneration"]}
        )
        == "umt5"
    )


def test_explicit_serialized_tie_setting_overrides_config_default() -> None:
    config = SimpleNamespace(tie_word_embeddings=True, hash_seed=None)

    axon_test_mod._ensure_legacy_hf_config_attributes(
        config,
        {"tie_word_embeddings": False},
    )

    assert config.tie_word_embeddings is False
    assert config.hash_seed == 0


def test_hf_cpu_reference_disables_unavailable_mamba_kernels() -> None:
    config = SimpleNamespace(use_mamba_kernels=True)

    result = axon_test_mod._disable_unavailable_hf_cpu_kernels(
        config,
        torch.device("cpu"),
    )

    assert result is config
    assert config.use_mamba_kernels is False


def test_hf_cuda_reference_preserves_mamba_kernel_setting() -> None:
    config = SimpleNamespace(use_mamba_kernels=True)

    axon_test_mod._disable_unavailable_hf_cpu_kernels(
        config,
        torch.device("cuda"),
    )

    assert config.use_mamba_kernels is True


def test_hf_config_compat_replaces_unsupported_kernel_literals() -> None:
    payload = {
        "model_type": "xlstm",
        "chunkwise_kernel": "chunkwise--triton_xl_chunk",
        "sequence_kernel": "native_sequence__triton",
        "step_kernel": "triton",
    }

    assert axon_test_mod._patch_unsupported_kernel_payload_for_compat(payload)
    assert payload["chunkwise_kernel"] == "chunkwise--native_autograd"
    assert payload["sequence_kernel"] == "native_sequence__native"
    assert payload["step_kernel"] == "native"


class _OutputEmbeddingProbe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = torch.nn.Module()
        self.lm_head.decoder = torch.nn.Linear(4, 8, bias=False)
        self.entity_predictions = torch.nn.Module()
        self.entity_predictions.decoder = torch.nn.Linear(4, 8, bias=False)

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head.decoder


def test_explicit_output_head_check_ignores_unrelated_decoder(tmp_path) -> None:  # type: ignore[no-untyped-def]
    safetensors.torch.save_file(
        {
            "entity_predictions.decoder.weight": torch.zeros(8, 4),
        },
        tmp_path / "model.safetensors",
    )

    assert not axon_test_mod._checkpoint_has_explicit_output_head_weight(
        tmp_path,
        _OutputEmbeddingProbe(),
    )


def test_extract_logits_accepts_namespace_logits() -> None:
    logits = torch.randn(2, 3, 5)

    output = SimpleNamespace(logits=logits)

    assert axon_test_mod._extract_logits(output) is logits


class _MissingInputEmbeddingAccessor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.word_embeddings = torch.nn.Embedding(8, 4)

    def get_input_embeddings(self) -> torch.nn.Module:
        raise NotImplementedError


def test_hf_input_embedding_device_falls_back_to_named_embedding() -> None:
    model = _MissingInputEmbeddingAccessor()

    assert axon_test_mod._hf_input_embedding_device(model) == model.backbone.word_embeddings.weight.device


def test_transformers_compat_restores_default_rope_initializer() -> None:
    from transformers import modeling_rope_utils

    modeling_rope_utils.ROPE_INIT_FUNCTIONS.pop("default", None)
    axon_test_mod._ensure_transformers_import_compat()

    config = SimpleNamespace(
        rope_parameters={"rope_theta": 10000.0},
        hidden_size=8,
        num_attention_heads=2,
        standardize_rope_params=lambda: None,
    )
    inv_freq, scale = modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"](config)
    assert torch.equal(inv_freq, torch.tensor([1.0, 0.01]))
    assert scale == 1.0


class _RotaryBufferProbe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace()
        self.rope_init_fn = lambda config, device=None: (
            torch.tensor([1.0, 0.01], device=device),
            1.0,
        )
        self.register_buffer(
            "inv_freq",
            torch.tensor([float("nan"), 0.0]),
            persistent=False,
        )
        self.original_inv_freq = self.inv_freq
        self.attention_scaling = 7.0


def test_hf_rotary_refresh_rebuilds_nonpersistent_inv_freq() -> None:
    model = _RotaryBufferProbe()

    refreshed = axon_test_mod._refresh_hf_rotary_caches_if_needed(
        model,
        dtype=torch.float32,
    )

    assert refreshed == 1
    assert torch.equal(model.inv_freq, torch.tensor([1.0, 0.01]))
    assert model.original_inv_freq is model.inv_freq
    assert model.attention_scaling == 1.0


def test_hf_reference_restores_checkpoint_when_from_pretrained_left_random_weights(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    weights_path = tmp_path / "model.safetensors"
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    safetensors.torch.save_file({"weight": expected}, weights_path)
    model = torch.nn.Linear(4, 3, bias=False)
    torch.nn.init.zeros_(model.weight)

    restored = axon_test_mod._restore_hf_reference_weights_if_needed(
        model,
        safetensors_files=[weights_path],
        dtype=torch.float32,
    )

    assert restored is True
    assert torch.equal(model.weight, expected)


def test_legacy_hf_config_defaults_are_added_without_overwriting_values() -> None:
    config = SimpleNamespace(is_decoder=True)

    result = axon_test_mod._ensure_legacy_hf_config_attributes(config)

    assert result is config
    assert config.is_decoder is True
    assert config.add_cross_attention is False


def test_deberta_legacy_mlm_head_is_detected_for_both_model_generations(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    weights_path = tmp_path / "model.safetensors"
    safetensors.torch.save_file(
        {"lm_predictions.lm_head.bias": torch.zeros(7)},
        weights_path,
    )

    for model_type in ("deberta", "deberta-v2"):
        assert axon_test_mod._is_deberta_modern_mlm_checkpoint(
            model_dir=tmp_path,
            model_config={"model_type": model_type},
            safetensors_files=[weights_path],
        )


class _GenerateCompatibilityProbe:
    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return None

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return None

    def _update_model_kwargs_for_generation(  # type: ignore[no-untyped-def]
        self, outputs, *args, **kwargs
    ):
        del outputs, args, kwargs
        return {}

    def generate(self, input_ids=None) -> str:  # type: ignore[no-untyped-def]
        del input_ids
        patched = (
            type(self).__call__ is not _GenerateCompatibilityProbe.__call__
            or type(self).forward is not _GenerateCompatibilityProbe.forward
            or "_update_model_kwargs_for_generation" in self.__dict__
        )
        return "patched" if patched else "unpatched"


def test_call_generate_compatible_does_not_patch_successful_generate() -> None:
    model = _GenerateCompatibilityProbe()

    result = axon_test_mod._call_generate_compatible(
        model,
        input_ids=torch.tensor([[1, 2, 3]]),
        unsupported_kwarg="filtered",
    )

    assert result == "unpatched"


def test_hf_reference_generate_disables_known_broken_cache_paths() -> None:
    assert not axon_test_mod._hf_reference_generate_uses_cache("blt")
    assert not axon_test_mod._hf_reference_generate_uses_cache("cpmant")
    assert not axon_test_mod._hf_reference_generate_uses_cache("xLSTM")
    assert axon_test_mod._hf_reference_generate_uses_cache("gpt2")


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


def test_rebuild_hf_phi3small_longrope_buffers_repairs_finite_garbage() -> None:
    model = _FakePhi3Wrapper()
    model.rotary.range_vector = torch.full((8,), 123.0, dtype=torch.float32)
    model.rotary.short_factors = torch.tensor([9.0, 9.0, 9.0, 9.0], dtype=torch.float32)
    model.rotary.long_factors = torch.tensor([8.0, 8.0, 8.0, 8.0], dtype=torch.float32)

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


def test_mra_checkpoint_uses_declared_external_tokenizer() -> None:
    assert not axon_test_mod._checkpoint_requires_local_tokenizer("uw-madison/mra-base-512-4")


def test_portable_mra_attention_matches_dense_attention_for_one_block() -> None:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn((1, 2, 32, 4), generator=generator)
    key = torch.randn((1, 2, 32, 4), generator=generator)
    value = torch.randn((1, 2, 32, 4), generator=generator)
    mask = torch.ones((2, 32), dtype=torch.float32)

    actual = axon_test_mod._portable_mra_attention(
        query,
        key,
        value,
        mask,
        num_blocks=1,
        approx_mode="full",
    )
    logits = torch.matmul(query, key.transpose(-1, -2)) / 2.0
    expected = torch.matmul(torch.softmax(logits, dim=-1), value)

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
