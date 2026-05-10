from __future__ import annotations

from typing import Any

OP_NAME = "select"


LOWERING_TYPE_SIGNATURE = {
    "args": (),
    "kwargs": {"cond": "Bool"},
    "returns": "dynamic",
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
