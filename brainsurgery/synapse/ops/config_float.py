from __future__ import annotations

from typing import Any

from ._config_common import (
    _coerce_float,
    _resolve_config_value,
)

OP_NAME = "config_float"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Path", "Float"),
    "kwargs": {},
    "returns": ("Float",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
