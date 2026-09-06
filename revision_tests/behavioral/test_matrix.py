from __future__ import annotations

from pathlib import Path

import yaml

from revision_tests.behavioral.run_cuda_matrix import (
    EXPECTED_IDS,
    copy_model_sidecars,
    load_protocol,
    write_plan,
)


def test_matrix_protocol_is_frozen() -> None:
    protocol = load_protocol()
    assert protocol["expected_model_ids"] == EXPECTED_IDS
    assert protocol["operation"]["factor"] == 1.0
    assert protocol["operation"]["output_shard_size_bytes"] == 256 * 1024 * 1024


def test_generated_plan_is_model_neutral(tmp_path: Path) -> None:
    protocol = load_protocol()
    plan_path = tmp_path / "plan.yaml"
    write_plan(plan_path, Path("models/example"), Path("models/output"), protocol["operation"])
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    assert plan["inputs"] == ["model::models/example"]
    assert plan["transforms"] == [{"scale_": {"target": r".*\.weight", "by": 1.0}}]
    assert plan["output"]["shard"] == "256MB"


def test_copy_model_sidecars_requires_config_and_copies_optional_generation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    transformed = tmp_path / "transformed"
    source.mkdir()
    transformed.mkdir()
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    assert copy_model_sidecars(source, transformed) == ["config.json"]
    (source / "generation_config.json").write_text("{}\n", encoding="utf-8")
    assert copy_model_sidecars(source, transformed) == ["config.json", "generation_config.json"]
    assert (transformed / "config.json").read_text(encoding="utf-8") == "{}\n"
