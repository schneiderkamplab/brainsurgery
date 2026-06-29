from __future__ import annotations

from typing import Any


OP_NAME = "shape"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]",),
    "kwargs": {},
    "returns": ("List[Dim]",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
