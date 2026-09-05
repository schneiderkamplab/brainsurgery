from __future__ import annotations

from .ast import AxonFile


def pragma_main_module(program: AxonFile) -> str | None:
    raw = program.pragmas.get("main")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise ValueError("MAIN pragma must be a non-empty string")
    return raw


def resolve_main_module(program: AxonFile, *, main_module: str | None = None) -> str:
    names = {module.name for module in program.modules}
    selected = main_module if main_module is not None else pragma_main_module(program)
    if selected is None:
        if not program.modules:
            raise ValueError("Axon program contains no definitions")
        selected = program.modules[-1].name
    if selected not in names:
        raise ValueError(f"Axon main definition not found: {selected!r}")
    return selected


__all__ = [
    "pragma_main_module",
    "resolve_main_module",
]
