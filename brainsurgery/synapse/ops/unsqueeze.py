from __future__ import annotations

from typing import Any


OP_NAME = "unsqueeze"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "Int"),
    "kwargs": {},
    "returns": ("Tensor[..R]",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
