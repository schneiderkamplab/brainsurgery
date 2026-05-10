from __future__ import annotations

from typing import Any

OP_NAME = "list_append"


LOWERING_TYPE_SIGNATURE = {
    "args": ("List[_T]", "_T"),
    "kwargs": {},
    "returns": ("List[_T]",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
