from __future__ import annotations

from typing import Any


OP_NAME = "sqrt"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Float",),
    "kwargs": {},
    "returns": ("Float",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
