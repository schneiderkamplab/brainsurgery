from __future__ import annotations

from typing import Any


OP_NAME = "params_param"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Path",),
    "kwargs": {},
    "returns": ("Tensor[..S]",),
}

__all__ = [
    "LOWERING_TYPE_SIGNATURE",
    "OP_NAME",
]
