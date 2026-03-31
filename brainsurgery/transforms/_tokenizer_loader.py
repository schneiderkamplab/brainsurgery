from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def _load_tokenizer(source: str) -> Any:
    candidate = Path(source).expanduser()
    if candidate.exists():
        resolved = str(candidate.resolve())
        try:
            return AutoTokenizer.from_pretrained(resolved, local_files_only=True)
        except Exception:
            return AutoTokenizer.from_pretrained(resolved, local_files_only=True, use_fast=False)
    try:
        return AutoTokenizer.from_pretrained(source, local_files_only=False)
    except Exception:
        return AutoTokenizer.from_pretrained(source, local_files_only=False, use_fast=False)


__all__: list[str] = []
