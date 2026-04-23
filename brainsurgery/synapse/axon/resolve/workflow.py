from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..ast import AxonFile, render_axon_file
from ..load import load_axon_files_from_path
from ..validate import ValidationDiagnostic, raise_on_warnings
from .core import resolve_loaded_axon_files


@dataclass(frozen=True)
class ResolveReport:
    ast: AxonFile
    diagnostics: tuple[ValidationDiagnostic, ...]

    @property
    def modules(self):
        return self.ast.modules

    @property
    def constants(self):
        return self.ast.constants

    @property
    def type_aliases(self):
        return self.ast.type_aliases

    @property
    def pragmas(self):
        return self.ast.pragmas


def resolve_axon_program_from_path(path: Path, *, strict: bool = False) -> ResolveReport:
    loaded = load_axon_files_from_path(path)
    ast, diagnostics = resolve_loaded_axon_files(loaded)
    if strict:
        raise_on_warnings(stage_name="resolver", diagnostics=diagnostics)
    return ResolveReport(ast=ast, diagnostics=diagnostics)


def resolve_axon_program_to_source(path: Path, *, strict: bool = False) -> str:
    return render_axon_file(resolve_axon_program_from_path(path, strict=strict).ast)


__all__ = ["ResolveReport", "resolve_axon_program_from_path", "resolve_axon_program_to_source"]
