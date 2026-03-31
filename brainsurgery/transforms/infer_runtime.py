from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from ..core import StateDictLike


def _load_infer_runtime_model(
    *,
    runtime: str,
    program: str,
    state_dict: dict[str, torch.Tensor],
    runtime_state_dict: StateDictLike | None = None,
) -> Any:
    runtime_name = runtime.strip().lower()
    if runtime_name == "synapse":
        return _load_synapse_runtime(
            program=program,
            state_dict=state_dict,
            runtime_state_dict=runtime_state_dict,
        )
    if runtime_name == "codegen":
        return _load_codegen_runtime(
            program=program,
            state_dict=state_dict,
            runtime_state_dict=runtime_state_dict,
        )
    if runtime_name == "hf":
        return _load_hf_runtime(program=program, state_dict=state_dict)
    raise ValueError(f"unsupported infer runtime: {runtime_name}")


def _load_synapse_runtime(
    *,
    program: str,
    state_dict: dict[str, torch.Tensor],
    runtime_state_dict: StateDictLike | None = None,
) -> Any:
    synapse = import_module("brainsurgery.synapse")
    program_path = Path(program)
    suffix = program_path.suffix.lower()
    if suffix == ".axon":
        modules = synapse.parse_axon_program_from_path(program_path)
        spec = synapse.lower_axon_program_to_synapse_spec(modules)
        return synapse.SynapseProgramModel.from_spec(
            spec,
            state_dict=state_dict,
            runtime_state_dict=runtime_state_dict,
        )
    if suffix in {".yaml", ".yml"}:
        return synapse.SynapseProgramModel.from_yaml(
            program_path,
            state_dict=state_dict,
            runtime_state_dict=runtime_state_dict,
        )
    raise ValueError("infer.program must point to a .axon or .yaml/.yml file")


def _load_codegen_runtime(
    *,
    program: str,
    state_dict: dict[str, torch.Tensor],
    runtime_state_dict: StateDictLike | None = None,
) -> Any:
    synapse = import_module("brainsurgery.synapse")
    spec = _load_synapse_spec(program=program)
    class_name = "InferCodegenModel"
    source = synapse.emit_model_code_from_synapse_spec(spec, class_name=class_name)
    module = ModuleType("infer_codegen_runtime")
    exec(source, module.__dict__)  # noqa: S102
    runtime_cls = getattr(module, class_name, None)
    if runtime_cls is None:
        raise ValueError("infer codegen failed to produce runtime class")
    return runtime_cls.from_state_dict(state_dict, runtime_state_dict=runtime_state_dict)


def _load_hf_runtime(
    *,
    program: str,
    state_dict: dict[str, torch.Tensor],
) -> Any:
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover - dependency/runtime dependent
        raise ValueError("infer runtime=hf requires transformers to be installed") from exc

    config = AutoConfig.from_pretrained(program, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    hf_state_dict = _adapt_hf_state_dict_keys(state_dict)
    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
    missing_filtered = [key for key in missing if key not in {"lm_head.weight"}]
    if unexpected:
        raise ValueError(f"infer runtime=hf got unexpected state_dict keys: {unexpected[:3]}")
    if missing_filtered:
        raise ValueError(
            f"infer runtime=hf missing required state_dict keys: {missing_filtered[:3]}"
        )
    model.tie_weights()
    return model


def _adapt_hf_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # The checkpoint loader can normalize GPT-2 names to bare keys (e.g. h.0..., wte...).
    # HF GPT-2 expects these under transformer.* and does not load attn bias buffers.
    keys = tuple(state_dict.keys())
    if not _looks_like_gpt2_stripped_state_dict(keys):
        return state_dict
    adapted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        next_key = key
        if not key.startswith("transformer.") and not key.startswith("lm_head."):
            next_key = f"transformer.{key}"
        if next_key.endswith(".attn.bias") or next_key.endswith(".attn.masked_bias"):
            continue
        adapted[next_key] = value
    return adapted


def _looks_like_gpt2_stripped_state_dict(keys: tuple[str, ...]) -> bool:
    has_h_blocks = any(key.startswith("h.") for key in keys)
    has_embed = any(key.startswith("wte.") for key in keys)
    has_already_prefixed = any(key.startswith("transformer.") for key in keys)
    return has_h_blocks and has_embed and not has_already_prefixed


def _load_synapse_spec(*, program: str) -> dict[str, Any]:
    synapse = import_module("brainsurgery.synapse")
    program_path = Path(program)
    suffix = program_path.suffix.lower()
    if suffix == ".axon":
        modules = synapse.parse_axon_program_from_path(program_path)
        return synapse.lower_axon_program_to_synapse_spec(modules)
    if suffix in {".yaml", ".yml"}:
        model = synapse.SynapseProgramModel.from_yaml(program_path)
        return model.spec
    raise ValueError("infer.program must point to a .axon or .yaml/.yml file")
