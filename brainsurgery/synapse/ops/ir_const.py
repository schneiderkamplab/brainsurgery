from __future__ import annotations

from typing import Any

OP_NAME = "_ir_expr"


LOWERING_TYPE_SIGNATURE = {
    "args": (),
    "kwargs": {},
    "returns": ("Any",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
