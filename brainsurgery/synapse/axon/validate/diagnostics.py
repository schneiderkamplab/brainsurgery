from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValidationDiagnostic:
    level: str
    message: str
    file_path: Path | None = None


def raise_on_warnings(*, stage_name: str, diagnostics: tuple[ValidationDiagnostic, ...]) -> None:
    if not any(item.level == "warning" for item in diagnostics):
        return
    warning_text = "\n".join(
        f"{item.file_path}: {item.message}" if item.file_path else item.message
        for item in diagnostics
        if item.level == "warning"
    )
    raise ValueError(f"{stage_name} strict mode failed on warnings:\n{warning_text}")


__all__ = ["ValidationDiagnostic", "raise_on_warnings"]
