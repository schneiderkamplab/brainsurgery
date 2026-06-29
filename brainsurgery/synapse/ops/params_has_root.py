from __future__ import annotations

from typing import Any

from ._params_common import (
    _param_root_exists,
    _resolve_root_arg,
)

OP_NAME = "params_has_root"


LOWERING_TYPE_SIGNATURE = {
    "args": ("String",),
    "kwargs": {},
    "returns": ("Bool",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
