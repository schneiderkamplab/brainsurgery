from __future__ import annotations

from ..ast import AxonFile
from .backend_required import validate_backend_required_flat_typed_axon_file


def validate_optimized_flat_typed_axon_file(
    ast: AxonFile, *, main_module: str | None = None
) -> None:
    """Validate canonical properties guaranteed by the optimizer.

    This deliberately validates optimizer promises, not maximal optimization.
    Today the optimizer guarantees the backend-required flat typed shape.
    More canonical optimizer invariants should be added here as they become
    contractual rather than best-effort cleanups.
    """

    validate_backend_required_flat_typed_axon_file(ast, main_module=main_module)


__all__ = ["validate_optimized_flat_typed_axon_file"]
