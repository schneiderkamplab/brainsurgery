from __future__ import annotations

from typing import Any


OP_NAME = "scatter"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "IdxTensor[..I]", "Any", "Any"),
    "kwargs": {},
    "returns": ("Tensor[..S]",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
