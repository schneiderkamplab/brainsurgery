from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from brainsurgery.synapse import (
    SynapseProgramModel,
    lower_axon_program_to_synapse_spec,
    parse_axon_program_from_path,
)
from tests.synapse_test_utils import build_codegen_model, extract_logits, masked_logits_diff


def _load_axon_spec(path: Path) -> dict[str, Any]:
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


def _nest_language_model_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    nested: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            nested[f"model.language_model.{key.removeprefix('model.')}"] = value
            continue
        nested[key] = value
    return nested


def _run_against_hf(
    *,
    repo_root: Path,
    axon_name: str,
    config: Any,
    check_codegen: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    device = torch.device("cpu")
    transformers = pytest.importorskip("transformers")
    hf_model = transformers.Gemma4ForCausalLM(config).to(device=device, dtype=torch.float32).eval()

    hf_state_dict = {
        key: value.detach().to(device=device, dtype=torch.float32)
        if torch.is_tensor(value) and value.is_floating_point()
        else value.detach().to(device=device)
        for key, value in hf_model.state_dict().items()
        if torch.is_tensor(value)
    }
    state_dict = _nest_language_model_state_dict(hf_state_dict)

    spec = _load_axon_spec(repo_root / "examples" / axon_name)
    model_section = spec.get("model")
    assert isinstance(model_section, dict)
    model_section["config"] = {"text_config": hf_model.config.to_dict()}

    runtime_model = SynapseProgramModel.from_spec(spec, state_dict=state_dict).to(device).eval()
    codegen_model = (
        build_codegen_model(spec, f"Tmp{axon_name.replace('.', '_')}", state_dict).eval()
        if check_codegen
        else None
    )

    input_ids = torch.tensor(
        [[11, 7, 19, 23, 17, 9], [5, 13, 29, 31, 2, 6]],
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        hf_logits = hf_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        runtime_logits = extract_logits(runtime_model(input_ids, attn_mask=attention_mask))
        codegen_logits = (
            extract_logits(codegen_model(input_ids, attn_mask=attention_mask))
            if codegen_model is not None
            else None
        )
    return hf_logits, runtime_logits, codegen_logits


def test_gemma4_dense_runtime_tracks_hf_use_cache_float32(repo_root: Path) -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=12,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_global_key_value_heads=4,
        head_dim=16,
        global_head_dim=16,
        sliding_window=32,
        sliding_window_pattern=1024,
        max_position_embeddings=128,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=128,
        num_kv_shared_layers=4,
        use_double_wide_mlp=True,
        attention_k_eq_v=False,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
    )
    device = torch.device("cpu")
    hf_model = transformers.Gemma4ForCausalLM(config).to(device=device, dtype=torch.float32).eval()

    hf_state_dict = {
        key: value.detach().to(device=device, dtype=torch.float32)
        if torch.is_tensor(value) and value.is_floating_point()
        else value.detach().to(device=device)
        for key, value in hf_model.state_dict().items()
        if torch.is_tensor(value)
    }
    state_dict = _nest_language_model_state_dict(hf_state_dict)

    spec = _load_axon_spec(repo_root / "examples" / "gemma4_e.axon")
    model_section = spec.get("model")
    assert isinstance(model_section, dict)
    model_section["config"] = {"text_config": hf_model.config.to_dict()}
    runtime_model = SynapseProgramModel.from_spec(spec, state_dict=state_dict).to(device).eval()

    input_ids = torch.tensor(
        [[11, 7, 19, 23, 17, 9], [5, 13, 29, 31, 2, 6]], device=device, dtype=torch.long
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        hf_logits = hf_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=True
        ).logits
        runtime_out = runtime_model(input_ids, attn_mask=attention_mask, use_cache=True)
        runtime_logits = extract_logits(runtime_out)

    diff = masked_logits_diff(runtime_logits, hf_logits, attention_mask)
    assert float(diff.mean()) < 4.0e-2
    assert float(diff.max()) < 1.7e-1


def test_gemma4_plain_dense_runtime_tracks_hf_float32(repo_root: Path) -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.Gemma4TextConfig(
        vocab_size=128,
        hidden_size=96,
        intermediate_size=192,
        num_hidden_layers=6,
        num_attention_heads=6,
        num_key_value_heads=6,
        num_global_key_value_heads=6,
        head_dim=16,
        global_head_dim=16,
        sliding_window=32,
        sliding_window_pattern=1024,
        max_position_embeddings=128,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        attention_k_eq_v=False,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
    )
    hf_logits, runtime_logits, _ = _run_against_hf(
        repo_root=repo_root,
        axon_name="gemma4_dense.axon",
        config=config,
        check_codegen=False,
    )
    attention_mask = torch.ones(runtime_logits.shape[:2], dtype=torch.long)
    diff = masked_logits_diff(runtime_logits, hf_logits, attention_mask)
    assert float(diff.mean()) < 2.0e-2
    assert float(diff.max()) < 1.6e-1


def test_gemma4_moe_runtime_tracks_hf_float32(repo_root: Path) -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_global_key_value_heads=4,
        head_dim=16,
        global_head_dim=16,
        sliding_window=32,
        sliding_window_pattern=1024,
        max_position_embeddings=128,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        attention_k_eq_v=False,
        enable_moe_block=True,
        num_experts=8,
        top_k_experts=2,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
    )
    hf_logits, runtime_logits, _ = _run_against_hf(
        repo_root=repo_root,
        axon_name="gemma4_moe.axon",
        config=config,
        check_codegen=False,
    )
    attention_mask = torch.ones(runtime_logits.shape[:2], dtype=torch.long)
    diff = masked_logits_diff(runtime_logits, hf_logits, attention_mask)
    assert float(diff.mean()) < 1.5e-2
    assert float(diff.max()) < 2.0e-1


def test_gemma4_moe_runtime_tracks_hf_use_cache_float32(repo_root: Path) -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_global_key_value_heads=4,
        head_dim=16,
        global_head_dim=16,
        sliding_window=32,
        sliding_window_pattern=1024,
        max_position_embeddings=128,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        attention_k_eq_v=False,
        enable_moe_block=True,
        num_experts=8,
        top_k_experts=2,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
    )
    device = torch.device("cpu")
    hf_model = transformers.Gemma4ForCausalLM(config).to(device=device, dtype=torch.float32).eval()

    hf_state_dict = {
        key: value.detach().to(device=device, dtype=torch.float32)
        if torch.is_tensor(value) and value.is_floating_point()
        else value.detach().to(device=device)
        for key, value in hf_model.state_dict().items()
        if torch.is_tensor(value)
    }
    state_dict = _nest_language_model_state_dict(hf_state_dict)

    spec = _load_axon_spec(repo_root / "examples" / "gemma4_moe.axon")
    model_section = spec.get("model")
    assert isinstance(model_section, dict)
    model_section["config"] = {"text_config": hf_model.config.to_dict()}
    runtime_model = SynapseProgramModel.from_spec(spec, state_dict=state_dict).to(device).eval()

    input_ids = torch.tensor(
        [[11, 7, 19, 23, 17, 9], [5, 13, 29, 31, 2, 6]], device=device, dtype=torch.long
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        hf_logits = hf_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=True
        ).logits
        runtime_out = runtime_model(input_ids, attn_mask=attention_mask, use_cache=True)
        runtime_logits = extract_logits(runtime_out)

    diff = masked_logits_diff(runtime_logits, hf_logits, attention_mask)
    assert float(diff.mean()) < 1.5e-2
    assert float(diff.max()) < 2.0e-1
