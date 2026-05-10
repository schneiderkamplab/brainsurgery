from __future__ import annotations

from typing import Any


OP_NAME = "where_indices"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": {},
    "returns": ("IdxTensor[..I]", "IdxTensor[..J]"),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
