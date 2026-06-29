from __future__ import annotations

from pathlib import Path

import brainsurgery  # noqa: F401
import brainsurgery.web.ui.backend as webui_backend
from brainsurgery.engine.plan import PlanStep, SurgeryPlan
from brainsurgery.web.ui.backend import _render_execution_summary, _transform_items


def test_transform_items_include_help_exit_dump() -> None:
    names = {item["name"] for item in _transform_items()}
    assert {"help", "exit", "dump"}.issubset(names)


def test_help_transform_metadata_contains_dropdown_options() -> None:
    help_item = next(item for item in _transform_items() if item["name"] == "help")
    assert "copy" in help_item["help_commands"]
    assert "assert" in help_item["help_subcommands"]
    assert help_item["help_subcommands"]["assert"]


def test_assert_transform_metadata_contains_expression_options() -> None:
    assert_item = next(item for item in _transform_items() if item["name"] == "assert")
    assert "equal" in assert_item["assert_expressions"]
    assert "equal" in assert_item["assert_expression_meta"]
    equal_meta = assert_item["assert_expression_meta"]["equal"]
    assert "left" in equal_meta["allowed_keys"]
    assert "right" in equal_meta["allowed_keys"]


def test_set_transform_metadata_contains_boolean_keys() -> None:
    set_item = next(item for item in _transform_items() if item["name"] == "set")
    assert "dry-run" in set_item["boolean_keys"]
    assert "preview" in set_item["boolean_keys"]
    assert "verbose" in set_item["boolean_keys"]


def test_load_transform_metadata_contains_mode_schema() -> None:
    load_item = next(item for item in _transform_items() if item["name"] == "load")
    assert load_item["mode_key"] == "backend"
    assert load_item["default_mode"] == "auto"
    assert "axon" in load_item["modes"]
    assert "weights" in load_item["mode_allowed_keys"]["axon"]


def test_transform_items_include_iterating_metadata() -> None:
    copy_item = next(item for item in _transform_items() if item["name"] == "copy")
    help_item = next(item for item in _transform_items() if item["name"] == "help")
    assert copy_item["iterating"] is True
    assert help_item["iterating"] is False


def test_render_execution_summary_renders_yaml_plan() -> None:
    plan = SurgeryPlan(
        inputs={},
        output=None,
        steps=[
            PlanStep(
                raw={"load": {"path": "/tmp/model.safetensors", "alias": "m1"}}, status="done"
            ),
            PlanStep(raw={"copy": {"from": "m1::.*", "to": "m1::\\1"}}, status="done"),
            PlanStep(raw={"exit": None}, status="done"),
        ],
    )
    summary = _render_execution_summary(plan=plan)
    assert "transforms:" in summary
    assert "- load:" in summary
    assert "- copy:" in summary
    assert "- exit: {}" in summary
    assert "inputs:" not in summary
    assert "output:" not in summary


def test_render_execution_summary_resolve_mode_includes_inputs() -> None:
    plan = SurgeryPlan(
        inputs={"model": Path("/tmp/model.safetensors")},
        output=None,
        steps=[PlanStep(raw={"exit": {}}, status="done")],
    )
    summary = _render_execution_summary(plan=plan, mode="resolve")
    assert "inputs:" in summary
    assert "- model::/tmp/model.safetensors" in summary


def test_serialize_models_is_lightweight_and_does_not_render_dumps(monkeypatch) -> None:
    class _Provider:
        def list_model_aliases(self) -> list[str]:
            return ["m"]

        def get_state_dict(self, alias: str):
            assert alias == "m"
            return {"w": object(), "b": object()}

    monkeypatch.setattr(
        webui_backend,
        "_render_dump_for_alias",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )
    models = webui_backend._serialize_models(_Provider())
    assert models == [{"alias": "m", "tensor_count": 2, "matched_count": 2, "total_count": 2}]


def test_api_error_payload_for_assert_includes_location_and_context() -> None:
    payload = {"equal": {"left": "a", "right": "b"}}
    error = webui_backend._api_error_payload(
        RuntimeError("assert.equal failed"),
        endpoint="/api/_apply_transform",
        transform_name="assert",
        payload=payload,
    )
    assert error["ok"] is False
    assert error["error"] == "assert.equal failed"
    info = error["error_info"]
    assert info["code"] == "assert_error"
    assert info["endpoint"] == "/api/_apply_transform"
    assert info["transform"] == "assert"
    assert info["location"] == {"transform": "assert", "field": "payload"}
    assert info["context"]["expression"] == "equal"
    assert info["context"]["expression_keys"] == ["left", "right"]


def test_api_error_payload_for_assert_with_non_mapping_payload_has_no_context() -> None:
    error = webui_backend._api_error_payload(
        RuntimeError("assert failed"),
        endpoint="/api/_apply_transform",
        transform_name="assert",
        payload=123,
    )
    assert error["ok"] is False
    assert error["error_info"]["context"] is None
