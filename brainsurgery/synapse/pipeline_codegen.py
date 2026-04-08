from __future__ import annotations

from typing import Any

import torch

from .codegen import emit_model_code_from_synapse_spec
from .pipeline_backend import (
    PipelinePlan,
    build_pipeline_plan,
    build_pipeline_stage_spec,
)


def emit_pipeline_stage_code_from_synapse_spec(
    spec: dict[str, Any],
    *,
    stage_index: int,
    requested_device: str | torch.device = "cuda",
    class_name: str = "GeneratedPipelineStage",
) -> tuple[PipelinePlan, str]:
    plan = build_pipeline_plan(spec, requested_device=requested_device)
    if stage_index < 0 or stage_index >= len(plan.stages):
        raise ValueError(f"invalid stage_index {stage_index}; expected 0..{len(plan.stages) - 1}")
    stage = plan.stages[stage_index]
    stage_spec = build_pipeline_stage_spec(spec, stage)
    code = emit_model_code_from_synapse_spec(
        stage_spec,
        class_name=f"{class_name}{stage_index}",
    )
    return plan, code


def emit_pipeline_stage_codes_from_synapse_spec(
    spec: dict[str, Any],
    *,
    requested_device: str | torch.device = "cuda",
    class_name_prefix: str = "GeneratedPipelineStage",
) -> tuple[PipelinePlan, tuple[str, ...]]:
    plan = build_pipeline_plan(spec, requested_device=requested_device)
    codes = []
    for stage in plan.stages:
        stage_spec = build_pipeline_stage_spec(spec, stage)
        codes.append(
            emit_model_code_from_synapse_spec(
                stage_spec,
                class_name=f"{class_name_prefix}{stage.index}",
            )
        )
    return plan, tuple(codes)


__all__ = [
    "emit_pipeline_stage_code_from_synapse_spec",
    "emit_pipeline_stage_codes_from_synapse_spec",
]
