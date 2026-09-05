#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
import transformers
import transformers.utils.import_utils as transformers_import_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from create_deepseek_v4_random import create_deepseek_v4_test_checkpoint

if not hasattr(transformers_import_utils, "is_torch_fx_available"):
    transformers_import_utils.is_torch_fx_available = lambda: False


@dataclass(frozen=True)
class TestSpec:
    name: str
    source: str


TEST_SPECS: tuple[TestSpec, ...] = (
    TestSpec("Apertus-Test", "swiss-ai/Apertus-8B-2509"),
    TestSpec("Cohere-Test", "CohereLabs/aya-expanse-8b"),
    TestSpec("Comma-Test", "danish-foundation-models/dfm-decoder-open-v0-7b-pt"),
    TestSpec("DeepSeek-V1-Test", "deepseek-ai/deepseek-moe-16b-base"),
    TestSpec("DeepSeek-V2-Test", "deepseek-ai/DeepSeek-V2-Lite"),
    TestSpec("DeepSeek-V3-Test", "deepseek-ai/DeepSeek-V3-Base"),
    TestSpec("DeepSeek-V4-Test", "deepseek-ai/DeepSeek-V4-Flash"),
    TestSpec("FlexOlmo-Test", "allenai/Flex-math-2x7B-1T"),
    TestSpec("Gemma4-Dense-Test", "google/gemma-4-31B"),
    TestSpec("Gemma4-E-Test", "google/gemma-4-E2B"),
    TestSpec("Gemma4-MoE-Test", "google/gemma-4-26B-A4B"),
    TestSpec("GLM4-Dense-Test", "zai-org/GLM-4-9B-0414"),
    TestSpec("GLM4-MoE-Test", "glm_4_5_air"),
    TestSpec("GPT-J-Test", "EleutherAI/gpt-j-6b"),
    TestSpec("GPT-NeoX-Test", "EleutherAI/pythia-6.9b"),
    TestSpec("GPT-OSS-Test", "openai/gpt-oss-20b"),
    TestSpec("Llama2-Test", "meta-llama/Llama-2-7b-hf"),
    TestSpec("Llama4-Test", "meta-llama/Llama-4-Scout-17B-16E"),
    TestSpec("Mistral-Test", "mistralai/Mistral-7B-v0.1"),
    TestSpec("Magistral-Test", "mistralai/Magistral-Small-2509"),
    TestSpec("Mistral3-Test", "mistralai/Ministral-3-8B-Base-2512"),
    TestSpec("Mistral4-Test", "mistralai/Mistral-Small-4-119B-2603"),
    TestSpec("Mixtral-Test", "mistralai/Mixtral-8x7B-v0.1"),
    TestSpec("Nemotron-H-Test", "nemotron3"),
    TestSpec("Olmo3-Test", "allenai/Olmo-3-1025-7B"),
    TestSpec("OLMoE-Test", "allenai/OLMoE-1B-7B-0924"),
    TestSpec("Phi3-Small-Test", "microsoft/Phi-3-small-8k-instruct"),
    TestSpec("Phi-MoE-Test", "microsoft/Phi-3.5-MoE-instruct"),
    TestSpec("OCRonos-Test", "PleIAs/OCRonos"),
    TestSpec("Qwen3-Test", "Qwen/Qwen3-14B"),
    TestSpec("Qwen3-MoE-Test", "Qwen/Qwen3-30B-A3B"),
)


def _set_if_present(config: Any, name: str, value: Any) -> None:
    if hasattr(config, name):
        try:
            setattr(config, name, value)
        except AttributeError:
            pass


def _cycle_like(values: Any, length: int, fallback: str) -> list[str]:
    if not isinstance(values, list) or not values:
        return [fallback for _ in range(length)]
    return [str(values[index % len(values)]) for index in range(length)]


def _mutate_config(config: Any, *, vocab_size: int) -> None:
    # Keep every test model non-trivial: at least several decoder blocks, at
    # least several heads, and at least two experts where the family supports it.
    hidden = 128
    heads = 4
    kv_heads = 2
    layers = 4
    intermediate = 256
    expert_intermediate = 64

    for target in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "vision_config", None),
        getattr(config, "audio_config", None),
    ):
        if target is None:
            continue
        for name in ("vocab_size",):
            _set_if_present(target, name, vocab_size)
        for name in ("hidden_size", "d_model", "n_embd", "model_dim"):
            _set_if_present(target, name, hidden)
        for name in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            _set_if_present(target, name, layers)
        _set_if_present(target, "num_kv_shared_layers", 0)
        for name in ("num_attention_heads", "n_head", "num_heads"):
            _set_if_present(target, name, heads)
        for name in ("num_key_value_heads", "num_kv_heads", "n_kv_heads"):
            _set_if_present(target, name, kv_heads)
        for name in ("head_dim", "kv_channels"):
            _set_if_present(target, name, hidden // heads)
        if hasattr(target, "qk_nope_head_dim") or hasattr(target, "qk_rope_head_dim"):
            setattr(target, "head_dim", hidden // heads)
            _set_if_present(target, "qk_nope_head_dim", 24)
            _set_if_present(target, "qk_rope_head_dim", 8)
            _set_if_present(target, "v_head_dim", hidden // heads)
            _set_if_present(target, "q_lora_rank", 32)
            _set_if_present(target, "kv_lora_rank", 32)
        for name in ("intermediate_size", "ffn_dim", "mlp_hidden_size"):
            _set_if_present(target, name, intermediate)
        for name in ("max_position_embeddings", "n_positions", "seq_length"):
            _set_if_present(target, name, 1024)
        for name in ("image_size", "input_size"):
            _set_if_present(target, name, 16)
        for name in ("patch_size",):
            _set_if_present(target, name, 8)
        for name in ("sliding_window", "sliding_window_pattern"):
            _set_if_present(target, name, 64)
        for name in ("num_experts", "num_local_experts", "n_routed_experts"):
            _set_if_present(target, name, 4)
        for name in ("num_experts_per_tok", "num_experts_per_token", "num_selected_experts"):
            _set_if_present(target, name, 2)
        for name in ("n_shared_experts", "num_shared_experts"):
            _set_if_present(target, name, 1)
        for name in ("moe_intermediate_size", "expert_intermediate_size"):
            _set_if_present(target, name, expert_intermediate)
        for name in ("first_k_dense_replace", "num_experts_per_tok"):
            if hasattr(target, name):
                setattr(target, name, min(int(getattr(target, name)), 2))
        if hasattr(target, "layer_types"):
            target.layer_types = _cycle_like(target.layer_types, layers, "full_attention")
        if hasattr(target, "mlp_layer_types"):
            target.mlp_layer_types = _cycle_like(target.mlp_layer_types, layers, "moe")
        if hasattr(target, "hybrid_override_pattern"):
            target.hybrid_override_pattern = "MEME"[:layers]
        if hasattr(target, "rope_scaling") and isinstance(target.rope_scaling, dict):
            target.rope_scaling = dict(target.rope_scaling)
            if "rope_type" in target.rope_scaling and "type" not in target.rope_scaling:
                target.rope_scaling["type"] = target.rope_scaling["rope_type"]
            if target.rope_scaling.get("type") == "default":
                target.rope_scaling["type"] = "linear"
            target.rope_scaling.setdefault("factor", 1.0)
        if getattr(target, "model_type", None) == "apertus":
            # Apertus configs may omit head_dim even though the model Axon and
            # weights use it explicitly for q/k normalization and RoPE.
            setattr(target, "head_dim", hidden // heads)
            if target.rope_scaling.get("type") not in {None, "default"}:
                target.rope_scaling["original_max_position_embeddings"] = min(
                    int(target.rope_scaling.get("original_max_position_embeddings", 512)),
                    256,
                )
        if hasattr(target, "rope_parameters") and isinstance(target.rope_parameters, dict):
            target.rope_parameters = dict(target.rope_parameters)
            rope_type = target.rope_parameters.get("rope_type", target.rope_parameters.get("type"))
            if rope_type not in {None, "default"}:
                target.rope_parameters["original_max_position_embeddings"] = min(
                    int(target.rope_parameters.get("original_max_position_embeddings", 512)),
                    256,
                )


def _apply_family_adjustments(config: Any, spec_name: str) -> None:
    def targets() -> tuple[Any, ...]:
        return (config, getattr(config, "text_config", None))

    def set_all(name: str, value: Any) -> None:
        for target in targets():
            if target is not None:
                _set_if_present(target, name, value)

    mha_tests = {
        "Comma-Test",
        "DeepSeek-V1-Test",
        "FlexOlmo-Test",
        "Mixtral-Test",
        "OCRonos-Test",
        "OLMoE-Test",
    }
    if spec_name in mha_tests:
        set_all("num_key_value_heads", 4)
        set_all("num_kv_heads", 4)
        set_all("n_kv_heads", 4)

    if spec_name == "GPT-J-Test":
        set_all("rotary_dim", 32)
        set_all("n_positions", 2048)

    if spec_name.startswith("Gemma4-"):
        set_all("global_head_dim", 32)
        set_all("num_global_key_value_heads", 2)
        set_all("top_k_experts", 2)
        for target in targets():
            if target is None or not hasattr(target, "layer_types"):
                continue
            target.layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
            setattr(target, "sliding_window_pattern", 2)

    if spec_name in {"DeepSeek-V2-Test", "DeepSeek-V3-Test"}:
        set_all("num_key_value_heads", 4)
    if spec_name == "DeepSeek-V2-Test":
        set_all("q_lora_rank", None)
        set_all("num_hidden_layers", 3)

    if spec_name in {"DeepSeek-V3-Test", "Mistral4-Test"}:
        set_all("qk_nope_head_dim", 24)
        set_all("qk_rope_head_dim", 8)
        set_all("v_head_dim", 32)
        set_all("qk_head_dim", 32)
    if spec_name == "DeepSeek-V3-Test":
        set_all("head_dim", 8)
        set_all("n_group", 2)
        set_all("topk_group", 1)
        set_all("use_cache", False)
        for target in targets():
            if target is not None and hasattr(target, "quantization_config"):
                setattr(target, "quantization_config", None)
            if target is not None and hasattr(target, "dtype"):
                setattr(target, "dtype", "float32")
            if target is not None and hasattr(target, "torch_dtype"):
                setattr(target, "torch_dtype", "float32")
    if spec_name == "Mistral4-Test":
        set_all("head_dim", 32)
        set_all("num_key_value_heads", 4)
    if spec_name == "Nemotron-H-Test":
        set_all("use_cache", False)
        set_all("mamba_num_heads", 4)
        set_all("mamba_head_dim", 32)
        set_all("mamba_n_groups", 2)
        set_all("n_groups", 2)
        set_all("ssm_state_size", 16)

    if spec_name in {"Magistral-Test", "Mistral3-Test"}:
        set_all("vocab_size", 150000)

    if spec_name in {"Qwen3-Test", "Qwen3-MoE-Test"}:
        set_all("num_key_value_heads", 4)
    if spec_name == "Qwen3-MoE-Test":
        for target in targets():
            if target is not None:
                setattr(target, "num_experts", 4)
    if spec_name == "Phi-MoE-Test":
        for target in targets():
            if target is None:
                continue
            head_dim = int(getattr(target, "head_dim", 0) or (getattr(target, "hidden_size") // getattr(target, "num_attention_heads")))
            factor_len = max(1, head_dim // 2)
            for attr in ("rope_scaling", "rope_parameters"):
                rope = getattr(target, attr, None)
                if not isinstance(rope, dict):
                    continue
                rope = dict(rope)
                rope["type"] = rope.get("type", rope.get("rope_type", "longrope"))
                rope["rope_type"] = rope.get("rope_type", rope["type"])
                for key in ("long_factor", "short_factor"):
                    values = list(rope.get(key) or [1.0])
                    if len(values) < factor_len:
                        values.extend([values[-1]] * (factor_len - len(values)))
                    rope[key] = values[:factor_len]
                setattr(target, attr, rope)


def _apply_storage_dtype(config: Any, dtype: torch.dtype) -> None:
    dtype_name = str(dtype).removeprefix("torch.")
    for target in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "vision_config", None),
        getattr(config, "audio_config", None),
    ):
        if target is None:
            continue
        for name in ("dtype", "torch_dtype"):
            if hasattr(target, name):
                try:
                    setattr(target, name, dtype_name)
                except AttributeError:
                    pass


def _replace_packed_experts_with_real_checkpoint_keys(
    state_dict: dict[str, torch.Tensor], *, spec_name: str
) -> None:
    """Use real checkpoint expert key layouts for tiny models too."""

    if spec_name == "Mixtral-Test":
        packed_prefix = ".mlp."
        alias_prefix = ".block_sparse_moe."
        gate_alias = "gate.weight"
        packed_gate_name = "gate.weight"
        expert_names = ("w1.weight", "w3.weight", "w2.weight")
    elif spec_name == "Phi-MoE-Test":
        packed_prefix = ".mlp."
        alias_prefix = ".block_sparse_moe."
        gate_alias = "gate.weight"
        packed_gate_name = "router.weight"
        expert_names = ("w1.weight", "w3.weight", "w2.weight")
    elif spec_name in {"FlexOlmo-Test", "GLM4-MoE-Test", "Qwen3-MoE-Test", "OLMoE-Test"}:
        packed_prefix = ".mlp."
        alias_prefix = ".mlp."
        gate_alias = None
        packed_gate_name = "gate.weight"
        expert_names = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
    else:
        return

    additions: dict[str, torch.Tensor] = {}
    removals: set[str] = set()
    for key, gate_up in list(state_dict.items()):
        if not key.endswith(f"{packed_prefix}experts.gate_up_proj"):
            continue
        prefix = key[: -len(f"{packed_prefix}experts.gate_up_proj")]
        down_key = f"{prefix}{packed_prefix}experts.down_proj"
        down = state_dict.get(down_key)
        if down is None:
            continue
        if gate_up.ndim != 3 or down.ndim != 3:
            continue
        if gate_up.shape[0] != down.shape[0] or gate_up.shape[1] % 2 != 0:
            continue
        removals.update({key, down_key})
        hidden = gate_up.shape[1] // 2
        gate = gate_up[:, :hidden, :].contiguous()
        up = gate_up[:, hidden:, :].contiguous()
        for expert_idx in range(int(gate_up.shape[0])):
            expert_prefix = f"{prefix}{alias_prefix}experts.{expert_idx}"
            additions.setdefault(f"{expert_prefix}.{expert_names[0]}", gate[expert_idx].clone())
            additions.setdefault(f"{expert_prefix}.{expert_names[1]}", up[expert_idx].clone())
            additions.setdefault(f"{expert_prefix}.{expert_names[2]}", down[expert_idx].clone())
        if gate_alias is not None:
            packed_gate_key = f"{prefix}{packed_prefix}{packed_gate_name}"
            packed_gate = state_dict.get(packed_gate_key)
            if packed_gate is not None:
                removals.add(packed_gate_key)
                additions.setdefault(f"{prefix}{alias_prefix}{gate_alias}", packed_gate.clone())
    for key in removals:
        state_dict.pop(key, None)
    state_dict.update(additions)


def _rewrite_test_checkpoint_keys(state_dict: dict[str, torch.Tensor], *, spec_name: str) -> None:
    if spec_name not in {"GPT-OSS-Test", "Magistral-Test", "Mistral3-Test", "Mistral4-Test"}:
        return
    additions: dict[str, torch.Tensor] = {}
    removals: set[str] = set()
    if spec_name == "GPT-OSS-Test":
        suffix_map = {
            ".mlp.experts.gate_up_proj": ".mlp.experts.gate_up_proj.weight",
            ".mlp.experts.down_proj": ".mlp.experts.down_proj.weight",
            ".mlp.experts.gate_up_proj_bias": ".mlp.experts.gate_up_proj.bias",
            ".mlp.experts.down_proj_bias": ".mlp.experts.down_proj.bias",
        }
        for key, value in list(state_dict.items()):
            for old_suffix, new_suffix in suffix_map.items():
                if key.endswith(old_suffix):
                    additions.setdefault(f"{key[: -len(old_suffix)]}{new_suffix}", value)
                    removals.add(key)
                    break
    if spec_name in {"Magistral-Test", "Mistral3-Test"}:
        old_prefix = "model.language_model."
        new_prefix = "language_model.model."
        for key, value in list(state_dict.items()):
            if not key.startswith(old_prefix):
                continue
            additions.setdefault(f"{new_prefix}{key.removeprefix(old_prefix)}", value)
            removals.add(key)
    if spec_name == "Mistral4-Test":
        old_prefix = "model."
        new_prefix = "language_model.model."
        for key, value in list(state_dict.items()):
            if not key.startswith(old_prefix):
                continue
            additions.setdefault(f"{new_prefix}{key.removeprefix(old_prefix)}", value)
            removals.add(key)
    for key in removals:
        state_dict.pop(key, None)
    state_dict.update(additions)
    if "lm_head.weight" in state_dict:
        state_dict.setdefault("language_model.lm_head.weight", state_dict["lm_head.weight"].clone())


def _copy_remote_code_files(source: Path, output: Path) -> None:
    for path in source.glob("*.py"):
        shutil.copy2(path, output / path.name)
    phi3_small_config = output / "configuration_phi3_small.py"
    if phi3_small_config.exists():
        text = phi3_small_config.read_text(encoding="utf-8")
        old = (
            "    @cached_property\n"
            "    def dummy_token_indices(self) -> List[int]:\n"
            "        # Importing here to avoid circular imports\n"
            "        from .tokenization_phi3_small import Phi3SmallTokenizer\n"
            "        tokenizer = Phi3SmallTokenizer()\n"
            "        return tokenizer.dummy_token_indices\n"
        )
        new = (
            "    @cached_property\n"
            "    def dummy_token_indices(self) -> List[int]:\n"
            "        return []\n"
        )
        if old in text:
            phi3_small_config.write_text(text.replace(old, new), encoding="utf-8")


def _patch_saved_config(output: Path, *, spec_name: str) -> None:
    config_path = output / "config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if spec_name == "DeepSeek-V2-Test":
        config.pop("auto_map", None)
    if spec_name == "Qwen3-MoE-Test":
        config["num_experts"] = int(config.get("num_experts", config.get("num_local_experts", 4)))
    if spec_name == "DeepSeek-V3-Test":
        config["qk_nope_head_dim"] = 24
        config["qk_rope_head_dim"] = 8
        config["qk_head_dim"] = 32
        config["v_head_dim"] = 32
        config["head_dim"] = 8
        config["n_group"] = 2
        config["topk_group"] = 1
        config["use_cache"] = False
        config["dtype"] = "float32"
        config["torch_dtype"] = "float32"
        config.pop("quantization_config", None)
    if spec_name == "Mistral4-Test":
        config["qk_nope_head_dim"] = 24
        config["qk_rope_head_dim"] = 8
        config["qk_head_dim"] = 32
        config["v_head_dim"] = 32
        config["head_dim"] = 32
        config["num_key_value_heads"] = 4
        rope_parameters = config.get("rope_parameters")
        if isinstance(rope_parameters, dict):
            rope_parameters["partial_rotary_factor"] = 0.25
    if spec_name.startswith("Gemma4-"):
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            text_config["top_k_experts"] = 2
            text_config["sliding_window_pattern"] = 2
    if spec_name == "Llama4-Test":
        text_config = config.get("text_config")
        if isinstance(text_config, dict) and "rope_scaling" not in text_config:
            rope_parameters = text_config.get("rope_parameters")
            if isinstance(rope_parameters, dict):
                text_config["rope_scaling"] = dict(rope_parameters)
    if spec_name == "Nemotron-H-Test":
        config["use_cache"] = False
        config["mamba_num_heads"] = 4
        config["mamba_head_dim"] = 32
        config["mamba_n_groups"] = 2
        config["n_groups"] = 2
        config["ssm_state_size"] = 16
    if spec_name == "Phi-MoE-Test":
        rope = config.get("rope_parameters")
        if isinstance(rope, dict):
            rope_scaling = dict(rope)
            rope_scaling["type"] = rope_scaling.get("type", rope_scaling.get("rope_type", "longrope"))
            rope_scaling["rope_type"] = rope_scaling.get("rope_type", rope_scaling["type"])
            rope_scaling["long_mscale"] = 1.0
            rope_scaling["short_mscale"] = 1.0
            config["rope_scaling"] = rope_scaling
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _local_source(root: Path, source: str) -> Path:
    path = root / source
    if path.exists():
        return path
    raise FileNotFoundError(f"missing local source checkpoint: {path}")


def _tokenizer_size(source: Path) -> int:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=True,
        )
        return max(int(len(tokenizer)), 1024)
    except Exception:
        return 65536


def create_test_checkpoint(
    spec: TestSpec,
    *,
    repo_root: Path,
    output_root: Path,
    dtype: torch.dtype,
    seed: int,
    max_shard_size: str,
) -> dict[str, Any]:
    if spec.name == "DeepSeek-V4-Test":
        return create_deepseek_v4_test_checkpoint(
            output=output_root / spec.name,
            dtype=dtype,
            seed=seed,
            max_shard_size=max_shard_size,
            tokenizer="openai-community/gpt2",
        )

    source = _local_source(repo_root / "models", spec.source)
    output = output_root / spec.name
    torch.manual_seed(seed)
    try:
        config = AutoConfig.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=True,
        )
    except OSError:
        try:
            config = AutoConfig.from_pretrained(
                source,
                local_files_only=True,
                trust_remote_code=False,
            )
        except OSError:
            config = AutoConfig.from_pretrained(
                spec.source,
                local_files_only=False,
                trust_remote_code=True,
            )
    vocab_size = _tokenizer_size(source)
    _mutate_config(config, vocab_size=vocab_size)
    _apply_family_adjustments(config, spec.name)
    _apply_storage_dtype(config, dtype)
    if type(config).__name__ == "Phi3SmallConfig":
        config.rope_scaling = None
    architectures = getattr(config, "architectures", None)
    model_cls = None
    if (
        isinstance(architectures, list)
        and architectures
        and type(config).__module__.startswith("transformers.models.")
    ):
        model_cls = getattr(transformers, str(architectures[0]), None)
    trust_model_remote_code = not type(config).__module__.startswith("transformers.models.")
    model = (
        model_cls(config)
        if model_cls is not None
        else AutoModelForCausalLM.from_config(config, trust_remote_code=trust_model_remote_code)
    )
    model = model.to(dtype=dtype)
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(output)
    _patch_saved_config(output, spec_name=spec.name)
    if spec.name != "DeepSeek-V2-Test":
        _copy_remote_code_files(source, output)
    else:
        for stale_remote_code in output.glob("*.py"):
            stale_remote_code.unlink()
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(output)
    for path in output.glob("*.safetensors"):
        path.unlink()
    for path in output.glob("*.safetensors.index.json"):
        path.unlink()
    state_dict = {
        name: tensor.detach().contiguous().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    _replace_packed_experts_with_real_checkpoint_keys(state_dict, spec_name=spec.name)
    _rewrite_test_checkpoint_keys(state_dict, spec_name=spec.name)
    save_file(state_dict, output / "model.safetensors", metadata={"format": "pt"})
    try:
        AutoTokenizer.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=True,
        ).save_pretrained(output)
    except Exception:
        AutoTokenizer.from_pretrained("openai-community/gpt2").save_pretrained(output)
    parameter_count = int(sum(param.numel() for param in model.parameters()))
    summary = {
        "name": spec.name,
        "source": spec.source,
        "parameter_count": parameter_count,
        "storage_dtype": str(dtype).replace("torch.", ""),
        "seed": seed,
        "config_class": type(config).__name__,
        "model_class": type(model).__name__,
    }
    (output / "random_checkpoint_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"output": str(output), **summary}


def _dtype(name: str) -> torch.dtype:
    normalized = name.lower().strip()
    if normalized == "float32":
        return torch.float32
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create tiny random test checkpoints for generic Axon families with no real <=4B checkpoint."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("models/test"))
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-shard-size", default="2GB")
    args = parser.parse_args()

    wanted = {item for item in args.only if item}
    specs = [spec for spec in TEST_SPECS if not wanted or spec.name in wanted]
    output_root = args.repo_root / args.output_root
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for spec in specs:
        try:
            results.append(
                create_test_checkpoint(
                    spec,
                    repo_root=args.repo_root,
                    output_root=output_root,
                    dtype=_dtype(args.dtype),
                    seed=args.seed,
                    max_shard_size=args.max_shard_size,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "name": spec.name,
                    "source": spec.source,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    print(json.dumps({"created": results, "errors": errors}, indent=2, sort_keys=True))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
