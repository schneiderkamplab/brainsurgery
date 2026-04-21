from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brainsurgery.synapse.axon import (
    lower_axon_program_to_synapse_spec,
    parse_axon_program_from_path,
)
from brainsurgery.synapse.pipeline_backend import (
    PipelinePlan,
    PipelineStage,
    build_pipeline_plan,
    build_pipeline_stage_spec,
    build_pipeline_stage_specs,
    partition_layer_ranges,
)
from brainsurgery.synapse.pipeline_runtime import (
    SynapsePipelineModel,
    build_hf_device_map_from_pipeline_usage,
)
from brainsurgery.synapse.runtime import SynapseProgramModel


def _load_spec(rel_path: str) -> dict[str, object]:
    path = Path("/work/training/brainsurgery") / rel_path
    modules = parse_axon_program_from_path(path)
    return lower_axon_program_to_synapse_spec(modules)


def _normalize_pipeline_layer_loop_bounds(spec: dict[str, object], *, total_layers: int) -> None:
    model = spec.get("model")
    if not isinstance(model, dict):
        return
    symbols = model.get("symbols")
    if not isinstance(symbols, dict):
        symbols = {}
    symbols["L"] = total_layers
    model["symbols"] = symbols
    graph = model.get("graph")
    if not isinstance(graph, list):
        return
    for item in graph:
        if not isinstance(item, dict):
            continue
        for node_spec in item.values():
            if not isinstance(node_spec, dict):
                continue
            if node_spec.get("_op") != "for":
                continue
            if node_spec.get("_from") is None:
                node_spec["_from"] = 0


def test_partition_layer_ranges_even_split() -> None:
    assert partition_layer_ranges(12, 3) == ((0, 4), (4, 8), (8, 12))


def test_partition_layer_ranges_uses_available_devices_up_to_layer_count() -> None:
    assert partition_layer_ranges(3, 6) == ((0, 1), (1, 2), (2, 3))


def test_build_pipeline_plan_for_smollm_uses_model_layers_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _load_spec("brainsurgery/synapse/models/smollm/generic-smollm.axon")
    _normalize_pipeline_layer_loop_bounds(spec, total_layers=30)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 4)
    plan = build_pipeline_plan(spec, requested_device="cuda")
    assert plan.layers_scope == "layers"
    assert plan.layers_var == "i"
    assert plan.total_layers == 30
    assert plan.stages[0].device == "cuda:0"


def test_build_pipeline_plan_evenly_splits_layers_for_visible_gpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _load_spec("brainsurgery/synapse/models/smollm/generic-smollm.axon")
    _normalize_pipeline_layer_loop_bounds(spec, total_layers=30)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 6)
    plan = build_pipeline_plan(spec, requested_device="cuda")
    assert [stage.device for stage in plan.stages] == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
        "cuda:4",
        "cuda:5",
    ]
    assert [(stage.layer_start, stage.layer_stop) for stage in plan.stages] == [
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 20),
        (20, 25),
        (25, 30),
    ]


def test_build_pipeline_plan_uses_model_symbols_for_materialized_layer_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _load_spec("brainsurgery/synapse/models/smollm/SmolLM-1.7B.axon")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    plan = build_pipeline_plan(spec, requested_device="cuda")
    assert plan.total_layers == 24
    assert [(stage.layer_start, stage.layer_stop) for stage in plan.stages] == [
        (0, 12),
        (12, 24),
    ]


def test_build_pipeline_plan_uses_model_config_for_generic_layer_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {
                    "n_cfg": {
                        "_op": "config_int",
                        "_bind": "L",
                        "_args": "num_hidden_layers",
                        "default": 32,
                    }
                },
                {
                    "n_for_0": {
                        "_op": "for",
                        "_scope": "model.layers",
                        "_var": "i",
                        "_from": 0,
                        "_to": {"_expr": "name", "id": "L"},
                        "_body": [],
                    }
                },
            ],
            "symbols": {},
            "config": {"num_hidden_layers": 80},
        },
    }
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 4)
    plan = build_pipeline_plan(spec, requested_device="cuda")
    assert plan.total_layers == 80
    assert [(stage.layer_start, stage.layer_stop) for stage in plan.stages] == [
        (0, 20),
        (20, 40),
        (40, 60),
        (60, 80),
    ]


def test_build_pipeline_plan_uses_model_config_for_lowered_at_prefixed_config_int(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {
                    "n_cfg": {
                        "_op": "config_int",
                        "_bind": "L",
                        "_args": ["@num_hidden_layers", 30],
                    }
                },
                {
                    "n_for_0": {
                        "_op": "for",
                        "_scope": "model.layers",
                        "_var": "i",
                        "_from": 0,
                        "_to": {"_expr": "name", "id": "L"},
                        "_body": [],
                    }
                },
            ],
            "symbols": {},
            "config": {"num_hidden_layers": 64},
        },
    }
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    plan = build_pipeline_plan(spec, requested_device="cuda")
    assert plan.total_layers == 64
    assert [(stage.layer_start, stage.layer_stop) for stage in plan.stages] == [
        (0, 32),
        (32, 64),
    ]


def test_build_pipeline_plan_prefers_model_symbols_over_config_int_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {
                    "n_cfg": {
                        "_op": "config_int",
                        "_bind": "L",
                        "_args": "num_hidden_layers",
                        "default": 32,
                    }
                },
                {
                    "n_for_0": {
                        "_op": "for",
                        "_scope": "model.layers",
                        "_var": "i",
                        "_from": 0,
                        "_to": {"_expr": "name", "id": "L"},
                        "_body": [],
                    }
                },
            ],
            "symbols": {"L": 24},
        },
    }
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    plan = build_pipeline_plan(spec, requested_device="cuda")
    assert plan.total_layers == 24
    assert [(stage.layer_start, stage.layer_stop) for stage in plan.stages] == [
        (0, 12),
        (12, 24),
    ]


def test_build_pipeline_plan_resolves_config_int_root_selected_by_config_has(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {"n_cfg_has": {"_op": "config_has", "_bind": "cond_cfg", "_args": "text_config"}},
                {
                    "n_cfg_root": {
                        "_op": "select",
                        "_bind": "CFG",
                        "cond": "cond_cfg",
                        "_then": [
                            {
                                "n_then": {
                                    "_op": "_ir_expr",
                                    "_bind": "then_cfg",
                                    "value": {"_expr": "string", "value": "text_config"},
                                }
                            }
                        ],
                        "_then_bind": "then_cfg",
                        "_else": [
                            {
                                "n_else": {
                                    "_op": "_ir_expr",
                                    "_bind": "else_cfg",
                                    "value": {"_expr": "string", "value": ""},
                                }
                            }
                        ],
                        "_else_bind": "else_cfg",
                    }
                },
                {
                    "n_cfg": {
                        "_op": "config_int",
                        "_bind": "L",
                        "_args": "num_hidden_layers",
                        "root": "CFG",
                        "default": 18,
                    }
                },
                {
                    "n_for_0": {
                        "_op": "for",
                        "_scope": "model.layers",
                        "_var": "i",
                        "_from": 0,
                        "_to": {"_expr": "name", "id": "L"},
                        "_body": [],
                    }
                },
            ],
            "symbols": {},
            "config": {"text_config": {"num_hidden_layers": 62}},
        },
    }
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 4)
    plan = build_pipeline_plan(spec, requested_device="cuda")
    assert plan.total_layers == 62
    assert [(stage.layer_start, stage.layer_stop) for stage in plan.stages] == [
        (0, 16),
        (16, 32),
        (32, 47),
        (47, 62),
    ]


def test_build_pipeline_plan_resolves_computed_layer_bound_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {
                    "n_cfg": {
                        "_op": "config_int",
                        "_bind": "L_BASE",
                        "_args": "num_hidden_layers",
                        "root": "text_config",
                        "default": 18,
                    }
                },
                {
                    "n_const_two": {
                        "_op": "_ir_expr",
                        "_bind": "TWO",
                        "value": {"_expr": "int", "value": 2},
                    }
                },
                {
                    "n_div": {
                        "_op": "div",
                        "_bind": "L_HALF",
                        "_args": ["L_BASE", "TWO"],
                    }
                },
                {
                    "n_floor": {
                        "_op": "floor",
                        "_bind": "L",
                        "_args": "L_HALF",
                    }
                },
                {
                    "n_for_0": {
                        "_op": "for",
                        "_scope": "model.layers",
                        "_var": "i",
                        "_from": 0,
                        "_to": {"_expr": "name", "id": "L"},
                        "_body": [],
                    }
                },
            ],
            "symbols": {},
            "config": {"text_config": {"num_hidden_layers": 62}},
        },
    }
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 4)
    plan = build_pipeline_plan(spec, requested_device="cuda")
    assert plan.total_layers == 31
    assert [(stage.layer_start, stage.layer_stop) for stage in plan.stages] == [
        (0, 8),
        (8, 16),
        (16, 24),
        (24, 31),
    ]


def test_build_pipeline_plan_requires_top_level_layer_loop() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [{"foo": {"_op": "add", "_bind": "x"}}],
        },
    }
    with pytest.raises(ValueError, match="for@\\*\\.layers loop"):
        build_pipeline_plan(spec, requested_device="cuda:0")


def test_build_pipeline_stage_specs_for_smollm_split_prefix_loop_and_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _load_spec("brainsurgery/synapse/models/smollm/generic-smollm.axon")
    _normalize_pipeline_layer_loop_bounds(spec, total_layers=30)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    plan, stage_specs = build_pipeline_stage_specs(spec, requested_device="cuda")
    assert len(plan.stages) == 2
    stage0, stage1 = stage_specs

    assert "x" not in stage0["model"]["inputs"]
    assert stage0["model"]["outputs"] == {"x": "x", "new_kv": "new_kv"}
    loop0 = next(
        next(iter(node.values()))
        for node in stage0["model"]["graph"]
        if isinstance(next(iter(node.values())), dict)
        and next(iter(node.values())).get("_op") == "for"
    )
    assert loop0["_from"] == 0
    assert loop0["_to"] == 15

    assert "x" in stage1["model"]["inputs"]
    assert stage1["model"]["outputs"] == {"logits": "logits", "new_kv": "new_kv"}
    loop1 = next(
        next(iter(node.values()))
        for node in stage1["model"]["graph"]
        if isinstance(next(iter(node.values())), dict)
        and next(iter(node.values())).get("_op") == "for"
    )
    assert loop1["_from"] == 15
    assert loop1["_to"] == 30


def test_pipeline_runtime_matches_single_runtime_for_simple_loop_model() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"optional": False}},
            "graph": [
                {
                    "n_const_0": {
                        "_op": "_ir_expr",
                        "_bind": "one",
                        "value": 1.0,
                    }
                },
                {
                    "n_for_0": {
                        "_op": "for",
                        "_scope": "model.layers",
                        "_var": "i",
                        "_from": 0,
                        "_to": 4,
                        "_body": [
                            {
                                "n_add_1": {
                                    "_op": "add",
                                    "_bind": "x",
                                    "_args": ["x", "one"],
                                }
                            }
                        ],
                    }
                },
            ],
            "outputs": {"x": "x"},
            "symbols": {},
            "blocks": {},
            "types": {},
        },
    }
    plan = PipelinePlan(
        devices=("cpu", "cpu"),
        layers_var="i",
        layers_scope="layers",
        total_layers=4,
        stages=(
            PipelineStage(index=0, device="cpu", layer_start=0, layer_stop=2),
            PipelineStage(index=1, device="cpu", layer_start=2, layer_stop=4),
        ),
    )
    stage_specs = tuple(build_pipeline_stage_spec(spec, stage) for stage in plan.stages)
    stages = [SynapseProgramModel.from_spec(stage_spec).eval() for stage_spec in stage_specs]
    pipeline = SynapsePipelineModel(
        plan=plan,
        stages=stages,
        stage_specs=stage_specs,
        original_spec=spec,
    ).eval()
    single = SynapseProgramModel.from_spec(spec).eval()

    x = torch.zeros((2, 3, 5), dtype=torch.float32)
    single_out = single(None, x=x)
    pipe_out = pipeline(None, x=x)

    assert isinstance(single_out, dict)
    assert isinstance(pipe_out, dict)
    torch.testing.assert_close(pipe_out["x"], single_out["x"])


def test_pipeline_runtime_rejects_out_of_range_layer_access() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"optional": False}},
            "graph": [
                {
                    "n_op_0": {
                        "_op": "linear",
                        "_bind": "y",
                        "_args": "x",
                        "_scope": "model.layers.3",
                        "dim": 4,
                    }
                }
            ],
            "outputs": {"y": "y"},
            "symbols": {},
            "blocks": {},
            "types": {},
        },
    }
    state_dict = {
        "model.layers.3.weight": torch.randn(4, 4),
        "model.layers.3.bias": torch.randn(4),
    }
    plan = PipelinePlan(
        devices=("cpu",),
        layers_var="i",
        layers_scope="layers",
        total_layers=4,
        stages=(PipelineStage(index=0, device="cpu", layer_start=0, layer_stop=2),),
    )
    from brainsurgery.synapse.pipeline_runtime import _PipelineStageModel

    stage_model = _PipelineStageModel(
        spec=spec,
        state_dict=state_dict,
        stage=plan.stages[0],
    )
    with pytest.raises(ValueError, match="out-of-range layer tensor"):
        _ = stage_model._state["model.layers.3.weight"]


def test_build_hf_device_map_from_pipeline_usage_uses_stage_accesses() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"optional": False}},
            "graph": [
                {
                    "n_for_0": {
                        "_op": "for",
                        "_scope": "model.layers",
                        "_var": "i",
                        "_from": 0,
                        "_to": 4,
                        "_body": [
                            {
                                "n_lin": {
                                    "_op": "linear",
                                    "_bind": "x",
                                    "_args": "x",
                                    "_scope": "n_lin",
                                    "dim": 4,
                                }
                            }
                        ],
                    }
                }
            ],
            "outputs": {"x": "x"},
            "symbols": {},
            "blocks": {},
            "types": {},
        },
    }
    state_dict = {}
    for i in range(4):
        state_dict[f"model.layers.{i}.n_lin.weight"] = torch.randn(4, 4)
        state_dict[f"model.layers.{i}.n_lin.bias"] = torch.randn(4)
    plan, device_map = build_hf_device_map_from_pipeline_usage(
        spec,
        state_dict=state_dict,
        input_ids=torch.randn(2, 3, 4),
        requested_device="cuda:0",
    )
    assert plan.total_layers == 4
    assert device_map["model.layers.0"] == "cuda:0"
