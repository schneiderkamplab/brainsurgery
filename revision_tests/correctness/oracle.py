"""Independent fixture and expected states for the correctness evaluation.

This module intentionally does not import BrainSurgery. The expected results
are expressed only with ordinary Python and PyTorch operations.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def fixture_state() -> dict[str, torch.Tensor]:
    """Return a deterministic, hand-verifiable state dictionary."""

    return {
        "embedding.weight": _sequence((3, 4), start=-6, step=1),
        "layer.0.weight": _sequence((4, 4), start=-8, step=1, scale=0.25),
        "layer.0.bias": torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32),
        "copy.source": torch.tensor([1.25, -2.5, 5.0], dtype=torch.float64),
        "move.source": torch.tensor([3, 1, 4, 1, 5], dtype=torch.int64),
        "delete.target": torch.tensor([True, False, True, True], dtype=torch.bool),
        "matrix.weight": _sequence((4, 4), start=0, step=1),
        "math.a": torch.tensor([[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]]),
        "math.b": torch.tensor([[0.5, 4.0, -3.0], [-2.0, 1.5, 2.0]]),
        "math.add": torch.zeros((2, 3), dtype=torch.float32),
        "math.subtract": torch.zeros((2, 3), dtype=torch.float32),
        "math.multiply": torch.zeros((2, 3), dtype=torch.float32),
        "cast.same": torch.tensor([0.0, -1.25, 2.5, 100.0], dtype=torch.float32),
        "cast.lossy": torch.tensor(
            [0.1, -1.234567, 3.1415927, 65504.0], dtype=torch.float32
        ),
        "unchanged.sentinel": torch.tensor(
            [[-9.0, 8.0], [7.0, -6.0]], dtype=torch.float32
        ),
        # Larger than the 1 KiB shard budget by itself, guaranteeing that C10
        # exercises oversized-tensor sharding as well as ordinary packing.
        "unchanged.large": _sequence((32, 32), start=-512, step=1, scale=0.03125),
    }


def expected_state(case_id: str) -> dict[str, torch.Tensor]:
    """Compute a case's expected state without calling the tested system."""

    state = clone_state(fixture_state())

    if case_id in {"C01", "C02", "C06", "C08", "C10"}:
        # C02 and C06 are inverse compositions. C08 converts float32 to float32.
        return state
    if case_id == "C03":
        state["copy.clone"] = state["copy.source"].clone()
        return state
    if case_id == "C04":
        state["move.destination"] = state.pop("move.source")
        return state
    if case_id == "C05":
        del state["delete.target"]
        return state
    if case_id == "C07":
        left = state["math.a"]
        right = state["math.b"]
        state["math.add"] = left + right
        state["math.subtract"] = left - right
        state["math.multiply"] = left * right
        state["math.scaled"] = left * 0.5
        return state
    if case_id == "C09":
        state["cast.lossy"] = state["cast.lossy"].to(torch.bfloat16)
        return state
    raise ValueError(f"unknown correctness case: {case_id}")


def clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in state.items()}


def _sequence(
    shape: tuple[int, ...],
    *,
    start: int,
    step: int,
    scale: float = 1.0,
) -> torch.Tensor:
    count = 1
    for dimension in shape:
        count *= dimension
    values = torch.arange(start, start + count * step, step, dtype=torch.float32)
    return (values * scale).reshape(shape)
