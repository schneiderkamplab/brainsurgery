from __future__ import annotations

from pathlib import Path

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
    assert "--canonicalize" not in help_text
    assert "--backend-required" not in help_text
    assert "--optimize-safe" not in help_text
    assert "--optimize/--no-optimize" not in help_text


def test_synapse_stage_dump_accepts_graph_optimizer_flag(tmp_path: Path) -> None:
    source = tmp_path / "main.axon"
    source.write_text(
        """
{-# MAIN "main" #-}

main :: Int -> Int
main x = do
  y <- x + 1
  return y
"""
    )
    output = tmp_path / "dump.axon"

    result = CliRunner().invoke(
        app,
        [
            "synapse",
            "axon-stage-dump",
            str(source),
            str(output),
            "--stage",
            "graph-ir-axon",
            "--optimize-ast",
            "--optimize-graph",
            "--show-types",
            "--show-purity",
        ],
    )

    assert result.exit_code == 0, result.output
    text = output.read_text()
    assert "-- purity: total_pure" in text
    assert "main :: Int -> Int" in text
    assert "return" in text


def test_synapse_codegen_dump_emits_torch_python_without_profile_branches(tmp_path: Path) -> None:
    source = tmp_path / "main.axon"
    source.write_text(
        """
{-# MAIN "main" #-}

main :: Int -> Int
main x = do
  y <- x + 1
  return y
"""
    )
    output = tmp_path / "generated.py"

    result = CliRunner().invoke(
        app,
        [
            "synapse",
            "axon-codegen-dump",
            str(source),
            str(output),
            "--main-module",
            "main",
            "--optimize-ast",
            "--optimize-graph",
        ],
    )

    assert result.exit_code == 0, result.output
    text = output.read_text()
    assert "class AxonGeneratedModel" in text
    assert "_profile_call" not in text
    assert "profile_summary" not in text
    assert "import time" not in text


def test_synapse_codegen_dump_can_emit_profile_torch_python(tmp_path: Path) -> None:
    source = tmp_path / "main.axon"
    source.write_text(
        """
{-# MAIN "main" #-}

main :: Int -> Int
main x = do
  y <- x + 1
  return y
"""
    )
    output = tmp_path / "generated.py"

    result = CliRunner().invoke(
        app,
        [
            "synapse",
            "axon-codegen-dump",
            str(source),
            str(output),
            "--main-module",
            "main",
            "--profile-code",
        ],
    )

    assert result.exit_code == 0, result.output
    text = output.read_text()
    assert "_profile_call" in text
    assert "profile_summary" in text
    assert "import time" in text
    assert "if self._profile_enabled" not in text


def test_synapse_test_exposes_ast_and_graph_optimizer_flags_only() -> None:
    help_text = _help("axon-test")
    assert "--optimize-ast" in help_text
    assert "--optimize-graph" in help_text
    assert "--canonicalize" not in help_text
    assert "--backend-required" not in help_text
    assert "--optimize-safe" not in help_text
    assert "--optimize/--no-optimize" not in help_text


def test_synapse_stage_dump_uses_current_codegen2_stage_flags() -> None:
    help_text = _help("axon-stage-dump")
    assert "optimize-ast" in help_text
    assert "graph-ir-axon" in help_text
    assert "--optimize-ast" in help_text
    assert "--optimize-graph" in help_text
    assert "--show-purity" in help_text
    assert "--canonicalize" not in help_text
    assert "--backend-required" not in help_text
    assert "--optimize-safe" not in help_text
    assert "--optimize/--no-optimize" not in help_text


def test_synapse_codegen_dump_exposes_codegen2_flags() -> None:
    help_text = _help("axon-codegen-dump")
    assert "--optimize-ast" in help_text
    assert "--optimize-graph" in help_text
    assert "--profile-code" in help_text
    assert "codegen2-torch" in help_text
