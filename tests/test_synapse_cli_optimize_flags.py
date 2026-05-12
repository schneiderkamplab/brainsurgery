from __future__ import annotations

from typer.testing import CliRunner

from brainsurgery import app


def _help(command: str) -> str:
    result = CliRunner().invoke(app, ["synapse", command, "--help"])
    assert result.exit_code == 0, result.output
    return result.output


def test_synapse_benchmark_exposes_ast_and_graph_optimizer_flags_only() -> None:
    help_text = _help("axon-benchmark")
    assert "--optimize-ast" in help_text
    assert "--optimize-graph" in help_text
    assert "--optimize-safe" not in help_text
    assert "--optimize/--no-optimize" not in help_text


def test_synapse_test_exposes_ast_and_graph_optimizer_flags_only() -> None:
    help_text = _help("axon-test")
    assert "--optimize-ast" in help_text
    assert "--optimize-graph" in help_text
    assert "--optimize-safe" not in help_text
    assert "--optimize/--no-optimize" not in help_text


def test_synapse_stage_dump_uses_optimize_ast_stage() -> None:
    help_text = _help("axon-stage-dump")
    assert "optimize-ast" in help_text
    assert "--optimize-ast" in help_text
    assert "--optimize-safe" not in help_text
    assert "--optimize/--no-optimize" not in help_text
