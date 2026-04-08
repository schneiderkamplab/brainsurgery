from __future__ import annotations

import re
from collections.abc import Iterator, KeysView
from typing import Any

import torch
from torch import nn

from .pipeline_backend import (
    PipelinePlan,
    PipelineStage,
    build_pipeline_stage_spec,
    build_pipeline_stage_specs,
)
from .runtime import SynapseProgramModel

_LAYER_KEY_RE = re.compile(r"(^|\.)layers\.(\d+)(\.|$)")


def _hf_device_map_prefix_for_key(key: str) -> str | None:
    match = _LAYER_KEY_RE.search(key)
    if match is not None:
        layer_index = int(match.group(2))
        prefix = key[: match.end(2)]
        return prefix
    if "." not in key:
        return None
    return key.rsplit(".", 1)[0]


def _prefix_covers_name(prefix: str, name: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _candidate_hf_prefixes(prefix: str) -> tuple[str, ...]:
    out = [
        prefix,
        f"model.{prefix}",
        prefix.replace("language_model.model.", "language_model.", 1),
        f"model.{prefix.replace('language_model.model.', 'language_model.', 1)}",
        prefix.replace("language_model.", "model.language_model.", 1),
        prefix.replace("vision_tower.", "model.vision_tower.", 1),
        prefix.replace("multi_modal_projector.", "model.multi_modal_projector.", 1),
    ]
    dedup: list[str] = []
    seen: set[str] = set()
    for item in out:
        item = item.strip(".")
        if not item or item in seen:
            continue
        seen.add(item)
        dedup.append(item)
    return tuple(dedup)


def _match_prefix_to_hf_names(prefix: str, hf_param_names: set[str]) -> str:
    if not hf_param_names:
        return prefix
    candidates = _candidate_hf_prefixes(prefix)
    best = prefix
    best_score = -1
    for candidate in candidates:
        score = sum(1 for name in hf_param_names if _prefix_covers_name(candidate, name))
        if score > best_score:
            best = candidate
            best_score = score
    return best


def _add_coverage_roots_for_unmapped_params(
    device_map: dict[str, str],
    hf_param_names: set[str],
    *,
    default_device: str,
) -> None:
    if not hf_param_names:
        return

    def _is_covered(name: str) -> bool:
        for prefix in device_map:
            if _prefix_covers_name(prefix, name):
                return True
        return False

    uncovered = [name for name in hf_param_names if not _is_covered(name)]
    if not uncovered:
        return
    roots: set[str] = set()
    for name in uncovered:
        # Add a precise module root instead of broad family roots like
        # "model.language_model", which can cause parent hooks to run on a
        # different device than child-layer parameters.
        if "." in name:
            roots.add(name.rsplit(".", 1)[0])
        else:
            roots.add(name)
    for root in sorted(roots):
        device_map.setdefault(root, default_device)


def _move_value_to_device(value: Any, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_value_to_device(item, device) for key, item in value.items()}
    return value


class _PipelineStageModel(SynapseProgramModel):
    def __init__(
        self,
        *,
        spec: dict[str, Any],
        state_dict: dict[str, torch.Tensor],
        stage: PipelineStage,
        runtime_state_dict: Any | None = None,
    ) -> None:
        super().__init__(spec=spec, state_dict=state_dict, runtime_state_dict=runtime_state_dict)
        self._pipeline_stage = stage
        self._state = _PipelineStateView(self._state, stage)


class _PipelineStateView:
    def __init__(self, base: dict[str, torch.Tensor], stage: PipelineStage) -> None:
        self._base = base
        self._stage = stage
        self._cache: dict[str, torch.Tensor] = {}
        self._accessed_keys: set[str] = set()

    def _validate_key(self, key: str) -> None:
        match = _LAYER_KEY_RE.search(key)
        if match is None:
            return
        layer_index = int(match.group(2))
        if self._stage.layer_start <= layer_index < self._stage.layer_stop:
            return
        raise ValueError(
            "pipeline stage tried to access out-of-range layer tensor "
            f"{key!r} for stage [{self._stage.layer_start},{self._stage.layer_stop})"
        )

    def __getitem__(self, key: str) -> torch.Tensor:
        self._validate_key(key)
        cached = self._cache.get(key)
        if cached is not None:
            self._accessed_keys.add(key)
            return cached
        tensor = self._base[key].to(self._stage.device)
        self._cache[key] = tensor
        self._accessed_keys.add(key)
        return tensor

    def __contains__(self, key: object) -> bool:
        return key in self._base

    def keys(self) -> KeysView[str]:
        return self._base.keys()

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._base:
            return default
        return self[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._base)

    def __len__(self) -> int:
        return len(self._base)

    @property
    def accessed_keys(self) -> set[str]:
        return set(self._accessed_keys)


class SynapsePipelineModel(nn.Module):
    def __init__(
        self,
        *,
        plan: PipelinePlan,
        stages: list[nn.Module],
        stage_specs: tuple[dict[str, Any], ...],
        original_spec: dict[str, Any],
    ) -> None:
        super().__init__()
        if len(stages) != len(plan.stages):
            raise ValueError("number of stage modules must match pipeline plan")
        self.plan = plan
        self.stage_specs = stage_specs
        self.original_spec = original_spec
        self.stages = nn.ModuleList(stages)

    @classmethod
    def from_spec(
        cls,
        spec: dict[str, Any],
        *,
        state_dict: dict[str, torch.Tensor],
        requested_device: str | torch.device = "cuda",
        runtime_state_dict: Any | None = None,
    ) -> "SynapsePipelineModel":
        plan, stage_specs = build_pipeline_stage_specs(spec, requested_device=requested_device)
        stages: list[nn.Module] = []
        for stage, stage_spec in zip(plan.stages, stage_specs, strict=True):
            model = _PipelineStageModel(
                spec=stage_spec,
                state_dict=state_dict,
                stage=stage,
                runtime_state_dict=runtime_state_dict,
            )
            model.eval()
            stages.append(model)
        return cls(plan=plan, stages=stages, stage_specs=stage_specs, original_spec=spec)

    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:
        current_x: torch.Tensor | None = None
        aggregated_new_kv: list[Any] = []
        final_output: Any = None

        for stage, stage_model, stage_spec in zip(
            self.plan.stages,
            self.stages,
            self.stage_specs,
            strict=True,
        ):
            stage_inputs_spec = stage_spec["model"].get("inputs", {})
            if not isinstance(stage_inputs_spec, dict):
                raise ValueError("pipeline stage inputs must be a mapping")

            stage_kwargs: dict[str, Any] = {}
            for input_name in stage_inputs_spec:
                if input_name == "input_ids":
                    continue
                if input_name == "x":
                    if current_x is not None:
                        stage_kwargs["x"] = current_x.to(stage.device)
                        continue
                    if "x" in inputs:
                        stage_kwargs["x"] = _move_value_to_device(inputs["x"], stage.device)
                        continue
                    raise ValueError("pipeline stage requires x but no prior stage produced it")
                    continue
                if input_name in inputs:
                    stage_kwargs[input_name] = _move_value_to_device(
                        inputs[input_name], stage.device
                    )

            stage_input_ids = input_ids
            if isinstance(stage_input_ids, torch.Tensor):
                stage_input_ids = stage_input_ids.to(stage.device)

            stage_output = stage_model(stage_input_ids, **stage_kwargs)
            final_output = stage_output

            if stage.layer_stop == self.plan.total_layers:
                if isinstance(stage_output, dict) and "new_kv" in stage_output:
                    local_new_kv = stage_output["new_kv"]
                    if isinstance(local_new_kv, list):
                        stage_output = dict(stage_output)
                        stage_output["new_kv"] = [*aggregated_new_kv, *local_new_kv]
                        final_output = stage_output
                break

            if isinstance(stage_output, dict):
                if "x" not in stage_output:
                    raise ValueError("non-final pipeline stage must output x")
                current_x = stage_output["x"]
                local_new_kv = stage_output.get("new_kv")
                if isinstance(local_new_kv, list):
                    aggregated_new_kv.extend(local_new_kv)
            else:
                current_x = stage_output

        return final_output

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        eos_token_id: int,
        max_len: int,
        attention_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be rank-2 [batch, seq]")
        if max_len <= 0:
            raise ValueError("max_len must be > 0")
        if attention_mask is not None and attn_mask is not None:
            raise ValueError("pass at most one of attention_mask or attn_mask")
        mask = attention_mask if attention_mask is not None else attn_mask
        if mask is not None:
            if mask.ndim != 2:
                raise ValueError("attention_mask must be rank-2 [batch, seq]")
            if mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have same shape as input_ids")
        if input_ids.size(1) >= max_len:
            return input_ids[:, :max_len]

        model = self.original_spec.get("model", {})
        input_specs = model.get("inputs", {})
        output_specs = model.get("outputs", {})
        if not isinstance(input_specs, dict):
            input_specs = {}
        if not isinstance(output_specs, dict):
            output_specs = {}

        state_input_name: str | None = None
        for candidate in (
            "past_key_values",
            "past_kv",
            "past",
            "cache_params",
            "cache_state",
            "state",
        ):
            if candidate in input_specs:
                state_input_name = candidate
                break

        use_cache_name = "use_cache" if "use_cache" in input_specs else None
        state_output_names = [
            name
            for name in (
                "past_key_values",
                "present_key_values",
                "new_kv",
                "past_kv",
                "cache_params",
                "cache_state",
                "state",
            )
            if name in output_specs
        ]

        batch, start_len = input_ids.shape
        generated = input_ids.new_empty((batch, max_len))
        generated[:, :start_len] = input_ids
        generated_mask = None
        if mask is not None:
            generated_mask = mask.new_zeros((batch, max_len))
            generated_mask[:, :start_len] = mask

        cache_state = None
        generated_device = input_ids.device
        finished = torch.zeros(batch, dtype=torch.bool, device=generated_device)
        cur_len = start_len
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                while cur_len < max_len and not torch.all(finished):
                    step_input = (
                        generated[:, :cur_len]
                        if cache_state is None
                        else generated[:, cur_len - 1 : cur_len]
                    )
                    call_kwargs: dict[str, Any] = {}
                    if generated_mask is not None:
                        if "attention_mask" in input_specs:
                            call_kwargs["attention_mask"] = generated_mask[:, :cur_len]
                        if "attn_mask" in input_specs:
                            call_kwargs["attn_mask"] = generated_mask[:, :cur_len]
                    if state_input_name is not None:
                        call_kwargs[state_input_name] = cache_state
                    if use_cache_name is not None:
                        call_kwargs[use_cache_name] = True
                    model_out = self.forward(step_input, **call_kwargs)
                    if isinstance(model_out, dict):
                        if "logits" in model_out:
                            logits = model_out["logits"]
                        elif len(model_out) == 1:
                            logits = next(iter(model_out.values()))
                        else:
                            raise KeyError(
                                "Expected 'logits' in model outputs or a single unnamed output"
                            )
                        for out_name in state_output_names:
                            if out_name in model_out:
                                cache_state = model_out[out_name]
                                break
                    else:
                        logits = model_out
                    next_token = torch.argmax(logits[:, -1, :], dim=-1).to(generated_device)
                    next_token = torch.where(
                        finished,
                        torch.full_like(next_token, eos_token_id),
                        next_token,
                    )
                    generated[:, cur_len] = next_token
                    finished = torch.logical_or(finished, next_token == eos_token_id)
                    if generated_mask is not None:
                        generated_mask[:, cur_len] = 1
                    cur_len += 1
        finally:
            if was_training:
                self.train()
        return generated[:, :cur_len]


def build_hf_device_map_from_pipeline_usage(
    spec: dict[str, Any],
    *,
    state_dict: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    requested_device: str | torch.device = "cuda",
    hf_param_names: set[str] | None = None,
) -> tuple[PipelinePlan, dict[str, str]]:
    plan = build_pipeline_stage_specs(spec, requested_device=requested_device)[0]
    cpu_stages = tuple(
        PipelineStage(
            index=stage.index,
            device="cpu",
            layer_start=stage.layer_start,
            layer_stop=stage.layer_stop,
        )
        for stage in plan.stages
    )
    cpu_stage_specs = tuple(build_pipeline_stage_spec(spec, stage) for stage in cpu_stages)
    cpu_stage_models: list[nn.Module] = []
    for stage, stage_spec in zip(cpu_stages, cpu_stage_specs, strict=True):
        model = _PipelineStageModel(
            spec=stage_spec,
            state_dict=state_dict,
            stage=stage,
        )
        model.eval()
        cpu_stage_models.append(model)
    cpu_pipeline = SynapsePipelineModel(
        plan=PipelinePlan(
            devices=tuple(stage.device for stage in cpu_stages),
            layers_var=plan.layers_var,
            layers_scope=plan.layers_scope,
            total_layers=plan.total_layers,
            stages=cpu_stages,
        ),
        stages=cpu_stage_models,
        stage_specs=cpu_stage_specs,
        original_spec=spec,
    ).eval()
    call_kwargs: dict[str, Any] = {}
    model_inputs = spec.get("model", {}).get("inputs", {})
    if not isinstance(model_inputs, dict):
        model_inputs = {}
    if attention_mask is not None:
        if "attention_mask" in model_inputs:
            call_kwargs["attention_mask"] = attention_mask
        if "attn_mask" in model_inputs:
            call_kwargs["attn_mask"] = attention_mask
    if "input_ids" in model_inputs:
        forward_args: tuple[Any, ...] = (input_ids,)
    else:
        call_kwargs["x"] = input_ids
        forward_args = (None,)
    with torch.inference_mode():
        cpu_pipeline(*forward_args, **call_kwargs)

    stage_key_sets: list[set[str]] = []
    for stage_model in cpu_stage_models:
        stage_state = getattr(stage_model, "_state", None)
        keys = stage_state.accessed_keys if isinstance(stage_state, _PipelineStateView) else set()
        stage_key_sets.append(set(keys))

    device_map: dict[str, str] = {}
    for stage, keys in zip(plan.stages, stage_key_sets, strict=True):
        for key in sorted(keys):
            prefix = _hf_device_map_prefix_for_key(key)
            if prefix is None:
                continue
            if hf_param_names is not None:
                prefix = _match_prefix_to_hf_names(prefix, hf_param_names)
            existing = device_map.get(prefix)
            if existing is None:
                device_map[prefix] = stage.device
            elif existing != stage.device:
                continue
    last_device = plan.stages[-1].device
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        prefix = _hf_device_map_prefix_for_key(key)
        if prefix is None:
            continue
        if hf_param_names is not None:
            prefix = _match_prefix_to_hf_names(prefix, hf_param_names)
        device_map.setdefault(prefix, last_device)
    embed_device = next(
        (device for prefix, device in device_map.items() if prefix.endswith("embed_tokens")),
        None,
    )
    if embed_device is not None:
        lm_head_device = embed_device if "lm_head.weight" not in state_dict else last_device
        device_map.setdefault("lm_head", lm_head_device)
        device_map.setdefault("language_model.lm_head", lm_head_device)
        device_map.setdefault("model.lm_head", lm_head_device)
    if hf_param_names is not None:
        _add_coverage_roots_for_unmapped_params(
            device_map,
            hf_param_names,
            default_device=last_device,
        )
    return plan, device_map


__all__ = ["SynapsePipelineModel", "build_hf_device_map_from_pipeline_usage"]
