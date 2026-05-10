from __future__ import annotations

import math
from typing import Any


OP_NAME = "floor"


LOWERING_TYPE_SIGNATURE = {
    "args": ("T",),
    "kwargs": {},
    "returns": "T",
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
