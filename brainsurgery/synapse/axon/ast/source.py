from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .nodes import AxonDefinition
from .types import TypeAliasDef


@dataclass(frozen=True)
class AxonFile:
    modules: tuple[AxonDefinition, ...]
    imports: tuple[str, ...]
    imported_members: dict[str, tuple[str, ...]]
    exports: tuple[str, ...]
    pragmas: dict[str, object]
    type_aliases: dict[str, TypeAliasDef]
    origin_path: Path | None = None


__all__ = [
    "AxonFile",
]
