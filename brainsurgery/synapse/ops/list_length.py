from __future__ import annotations

from typing import Any

OP_NAME = "list_length"


LOWERING_TYPE_SIGNATURE = {
    "args": ("List[_T]",),
    "kwargs": {},
    "returns": ("Dim",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
