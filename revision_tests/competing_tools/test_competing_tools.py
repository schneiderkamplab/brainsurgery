"""Independent-oracle and protocol negative controls."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from revision_tests.competing_tools.export_anonymous import sanitize_value
from revision_tests.competing_tools.oracle import (
    compare_output,
    expected_state,
    validate_comparison_record,
)
from revision_tests.competing_tools.prepare import canonicalize_gpt2_merge_state, prepare
from revision_tests.competing_tools.validate_protocol import validate


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "fixtures"
    prepare(root)
    return root


def _write_output(root: Path, state: dict[str, torch.Tensor]) -> Path:
    output = root / "output"
    output.mkdir()
    save_file(state, str(output / "model.safetensors"))
    return output


@pytest.mark.parametrize("case_id", ["R01", "M01", "M02"])
def test_independent_expected_outputs_pass(
    fixture_root: Path, tmp_path: Path, case_id: str
) -> None:
    output = _write_output(tmp_path, expected_state(case_id, fixture_root))
    result = compare_output(case_id, output, fixture_root)
    assert result["passed"] is True
    assert result["tensors_passed"] == result["expected_tensor_count"]


def test_changed_exact_value_is_detected(fixture_root: Path, tmp_path: Path) -> None:
    state = expected_state("R01", fixture_root)
    state["block.0.weight"] = state["block.0.weight"].clone()
    state["block.0.weight"][0, 0] += 1
    result = compare_output("R01", _write_output(tmp_path, state), fixture_root)
    assert result["passed"] is False
    assert result["tensors_passed"] == result["expected_tensor_count"] - 1


def test_missing_and_renamed_tensor_is_detected(fixture_root: Path, tmp_path: Path) -> None:
    state = expected_state("R01", fixture_root)
    state["wrong.name"] = state.pop("block.1.bias")
    result = compare_output("R01", _write_output(tmp_path, state), fixture_root)
    assert result["passed"] is False
    assert result["missing_tensors"] == ["block.1.bias"]
    assert result["unexpected_tensors"] == ["wrong.name"]


def test_dtype_change_is_detected(fixture_root: Path, tmp_path: Path) -> None:
    state = expected_state("M01", fixture_root)
    first_name = sorted(state)[0]
    state[first_name] = state[first_name].to(torch.float64)
    result = compare_output("M01", _write_output(tmp_path, state), fixture_root)
    assert result["passed"] is False
    failed = [item for item in result["tensor_results"] if item["name"] == first_name][0]
    assert failed["dtype_equal"] is False


def test_arithmetic_outside_tolerance_is_detected(fixture_root: Path, tmp_path: Path) -> None:
    state = expected_state("M02", fixture_root)
    first_name = sorted(state)[0]
    state[first_name] = state[first_name] + 1e-3
    result = compare_output("M02", _write_output(tmp_path, state), fixture_root)
    assert result["passed"] is False
    assert result["maximum_absolute_difference"] > 1e-4


def test_incompatible_record_metadata_is_rejected(fixture_root: Path, tmp_path: Path) -> None:
    output = _write_output(tmp_path, expected_state("R01", fixture_root))
    result = compare_output("R01", output, fixture_root)
    wrong_protocol = deepcopy(result)
    wrong_protocol["protocol_id"] = "other_protocol"
    with pytest.raises(ValueError, match="protocol_id"):
        validate_comparison_record(wrong_protocol, "R01")
    with pytest.raises(ValueError, match="case_id"):
        validate_comparison_record(result, "M01")


def test_frozen_protocol_validates() -> None:
    result = validate()
    assert result == {
        "protocol_id": "eacl2027_competing_tools_v1",
        "case_count": 3,
        "pair_count": 6,
    }


def test_gpt2_aliases_are_canonicalized_before_comparison() -> None:
    source = {
        "h.0.ln_1.weight": torch.tensor([1.0]),
        "wte.weight": torch.tensor([[2.0]]),
        "lm_head.weight": torch.tensor([[3.0]]),
    }
    canonical, mapping = canonicalize_gpt2_merge_state(source, set(source))
    assert set(canonical) == {
        "transformer.h.0.ln_1.weight",
        "transformer.wte.weight",
        "lm_head.weight",
    }
    assert mapping == {
        "h.0.ln_1.weight": "transformer.h.0.ln_1.weight",
        "lm_head.weight": "lm_head.weight",
        "wte.weight": "transformer.wte.weight",
    }


def test_anonymous_export_redacts_nested_paths_and_hostname() -> None:
    value = {
        "hostname": "named-laptop",
        "command": ["/Users/person/repo/.venv/bin/tool", "/tmp/run/input"],
        "nested": {"path": "/Users/person/repo/model"},
    }
    sanitized = sanitize_value(
        value,
        {
            "/Users/person/repo": "<REPOSITORY_ROOT>",
            "/Users/person": "<USER_HOME>",
        },
    )
    assert sanitized == {
        "hostname": "<REDACTED_HOSTNAME>",
        "command": ["<REPOSITORY_ROOT>/.venv/bin/tool", "/tmp/run/input"],
        "nested": {"path": "<REPOSITORY_ROOT>/model"},
    }
