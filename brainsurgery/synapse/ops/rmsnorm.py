from __future__ import annotations

from typing import Any


OP_NAME = "rmsnorm"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Tensor[..S]", "?Float", "?Dim", "?Bool"),
    "kwargs": {},
    "returns": ("Tensor[..S]",),
}
LOWERING_PARAM_NAMES = ("x", "eps", "dim", "cast_float")
LOWERING_PARAM_DEFAULTS = {"eps": 1e-6, "dim": None, "cast_float": False}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "LOWERING_PARAM_DEFAULTS",
    "LOWERING_PARAM_NAMES",
]
