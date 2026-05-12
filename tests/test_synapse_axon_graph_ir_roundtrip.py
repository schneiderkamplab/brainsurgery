from __future__ import annotations

from pathlib import Path

import pytest

from scripts.axon_graph_ir_strong_roundtrip import graph_ir_strong_roundtrip_path
from scripts.axon_graph_ir_weak_roundtrip import graph_ir_weak_roundtrip_path


def _graph_ir_roundtrip_paths() -> list[Path]:
    return sorted(Path("brainsurgery/synapse/models").glob("**/*.axon"))


def _graph_optimize_roundtrip_paths() -> list[Path]:
    return [
        Path("brainsurgery/synapse/models/gpt2/generic-gpt2-kv.axon"),
        Path("brainsurgery/synapse/models/bert/bert-base-uncased.axon"),
        Path("brainsurgery/synapse/models/llama4/generic-llama4.axon"),
        Path("brainsurgery/synapse/models/olmoe/generic-olmoe.axon"),
        Path("brainsurgery/synapse/models/phi3small/generic-phi3small.axon"),
    ]


@pytest.mark.parametrize("axon_path", _graph_ir_roundtrip_paths(), ids=lambda path: path.as_posix())
def test_graph_ir_weak_roundtrip_is_canonical(axon_path: Path, tmp_path: Path) -> None:
    assert graph_ir_weak_roundtrip_path(axon_path, tmp_path / "weak"), (
        f"Graph IR weak roundtrip changed rendered output for {axon_path}; "
        f"artifacts: {tmp_path}"
    )


@pytest.mark.parametrize("axon_path", _graph_ir_roundtrip_paths(), ids=lambda path: path.as_posix())
def test_graph_ir_strong_roundtrip_is_canonical(axon_path: Path, tmp_path: Path) -> None:
    assert graph_ir_strong_roundtrip_path(axon_path, tmp_path / "strong"), (
        f"Graph IR strong roundtrip changed rendered output for {axon_path}; "
        f"artifacts: {tmp_path}"
    )


@pytest.mark.parametrize("axon_path", _graph_ir_roundtrip_paths(), ids=lambda path: path.as_posix())
def test_ast_optimized_graph_ir_weak_roundtrip_is_canonical(
    axon_path: Path, tmp_path: Path
) -> None:
    assert graph_ir_weak_roundtrip_path(
        axon_path,
        tmp_path / "weak-opt-ast",
        optimize_ast=True,
        optimize_graph=False,
    ), (
        f"AST-optimized Graph IR weak roundtrip changed rendered output for {axon_path}; "
        f"artifacts: {tmp_path}"
    )


@pytest.mark.parametrize("axon_path", _graph_ir_roundtrip_paths(), ids=lambda path: path.as_posix())
def test_ast_optimized_graph_ir_strong_roundtrip_is_canonical(
    axon_path: Path, tmp_path: Path
) -> None:
    assert graph_ir_strong_roundtrip_path(
        axon_path,
        tmp_path / "strong-opt-ast",
        optimize_ast=True,
        optimize_graph=False,
    ), (
        f"AST-optimized Graph IR strong roundtrip changed rendered output for {axon_path}; "
        f"artifacts: {tmp_path}"
    )


@pytest.mark.parametrize("axon_path", _graph_optimize_roundtrip_paths(), ids=lambda path: path.as_posix())
def test_graph_optimize_weak_roundtrip_is_canonical(axon_path: Path, tmp_path: Path) -> None:
    assert graph_ir_weak_roundtrip_path(axon_path, tmp_path / "weak-opt", optimize_graph=True), (
        f"Optimized Graph IR weak roundtrip changed rendered output for {axon_path}; "
        f"artifacts: {tmp_path}"
    )


@pytest.mark.parametrize("axon_path", _graph_ir_roundtrip_paths(), ids=lambda path: path.as_posix())
def test_graph_optimized_graph_ir_weak_roundtrip_is_canonical(
    axon_path: Path, tmp_path: Path
) -> None:
    assert graph_ir_weak_roundtrip_path(
        axon_path,
        tmp_path / "weak-opt-graph",
        optimize_ast=False,
        optimize_graph=True,
    ), (
        f"Graph-optimized Graph IR weak roundtrip changed rendered output for {axon_path}; "
        f"artifacts: {tmp_path}"
    )


@pytest.mark.parametrize("axon_path", _graph_ir_roundtrip_paths(), ids=lambda path: path.as_posix())
def test_graph_optimized_graph_ir_strong_roundtrip_is_canonical(
    axon_path: Path, tmp_path: Path
) -> None:
    assert graph_ir_strong_roundtrip_path(
        axon_path,
        tmp_path / "strong-opt-graph",
        optimize_ast=False,
        optimize_graph=True,
    ), (
        f"Graph-optimized Graph IR strong roundtrip changed rendered output for {axon_path}; "
        f"artifacts: {tmp_path}"
    )


@pytest.mark.xfail(
    reason=(
        "all-file optimized Graph IR weak roundtrip is intentionally broad and "
        "currently tracks remaining optimizer/render canonicalization instability"
    ),
    strict=False,
)
@pytest.mark.parametrize("axon_path", _graph_ir_roundtrip_paths(), ids=lambda path: path.as_posix())
def test_safe_optimized_graph_ir_weak_roundtrip_is_canonical(
    axon_path: Path, tmp_path: Path
) -> None:
    assert graph_ir_weak_roundtrip_path(
        axon_path,
        tmp_path / "weak-opt-safe",
        optimize_ast=True,
        optimize_graph=True,
    ), (
        f"Safe optimized Graph IR weak roundtrip changed rendered output for {axon_path}; "
        f"artifacts: {tmp_path}"
    )


@pytest.mark.xfail(
    reason=(
        "optimized strong graph-rendered files currently expose a resolver/closure "
        "bug where a present MAIN definition can be pruned as unknown"
    ),
    strict=False,
)
@pytest.mark.parametrize("axon_path", _graph_optimize_roundtrip_paths(), ids=lambda path: path.as_posix())
def test_safe_optimized_graph_ir_strong_roundtrip_is_canonical(
    axon_path: Path, tmp_path: Path
) -> None:
    assert graph_ir_strong_roundtrip_path(
        axon_path,
        tmp_path / "strong-opt-safe",
        optimize_ast=True,
        optimize_graph=True,
    ), (
        f"Safe optimized Graph IR strong roundtrip changed rendered output for {axon_path}; "
        f"artifacts: {tmp_path}"
    )
