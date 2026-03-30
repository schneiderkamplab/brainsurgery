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
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"infer runtime=hf got unexpected state_dict keys: {unexpected[:3]}")
    if missing:
        raise ValueError(f"infer runtime=hf missing required state_dict keys: {missing[:3]}")
    return model


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
