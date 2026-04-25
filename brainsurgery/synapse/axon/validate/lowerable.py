from __future__ import annotations

from ..ast import AxonFile
from .backend_required import validate_backend_required_flat_typed_axon_file


def validate_lowerable_axon_file(ast: AxonFile, *, main_module: str | None = None) -> None:
    """Validate the Axon AST contract immediately before lowering.

    Lowering consumes a flat, typed, backend-required AST. Canonical naming is
    enforced by the canonicalization pass that should run before this validator.
    """

    validate_backend_required_flat_typed_axon_file(ast, main_module=main_module)


__all__ = ["validate_lowerable_axon_file"]
