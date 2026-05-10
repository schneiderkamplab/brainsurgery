from __future__ import annotations

from typing import Any

from ._config_common import (
    _coerce_int,
    _resolve_config_value,
)

OP_NAME = "config_int"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Path", "Int"),
    "kwargs": {},
    "returns": ("Int",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
