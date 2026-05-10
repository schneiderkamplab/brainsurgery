from __future__ import annotations

from typing import Any


OP_NAME = "clamp"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any", "Any"),
    "kwargs": {},
    "returns": "dynamic",
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
