from __future__ import annotations

from typing import Any


OP_NAME = "cos"


LOWERING_TYPE_SIGNATURE = {
    "args": ("T",),
    "kwargs": {},
    "returns": "T",
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
