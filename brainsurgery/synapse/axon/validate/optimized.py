from __future__ import annotations

from ..ast import AxonFile
from .typed import validate_typed_axon_file


def validate_optimized_flat_typed_axon_file(
    ast: AxonFile, *, main_module: str | None = None
) -> None:
    """Validate canonical properties guaranteed by the optimizer.

    This deliberately validates optimizer promises, not maximal optimization.
    Backend-specific lowering requirements are validated by the lowering target,
    not by this optimizer contract.
    """

    validate_typed_axon_file(ast, main_module=main_module)


__all__ = ["validate_optimized_flat_typed_axon_file"]
