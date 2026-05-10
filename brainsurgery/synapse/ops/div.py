from __future__ import annotations

from typing import Any

from ._broadcast import broadcast_last_dim, broadcast_shape

OP_NAME = "div"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
    "kwargs": {},
    "returns": "dynamic",
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
