from __future__ import annotations

from typing import Any


OP_NAME = "index_add"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "IdxTensor[..I]", "Tensor[..T]", "Any"),
    "kwargs": {},
    "returns": ("Tensor[..S]",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
