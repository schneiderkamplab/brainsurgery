from __future__ import annotations

from typing import Any


OP_NAME = "dtype_value"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "String"),
    "kwargs": {},
    "returns": ("Float",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
