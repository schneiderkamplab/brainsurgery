from __future__ import annotations

from typing import Any

from ._config_common import (
    _coerce_str,
    _resolve_config_value,
)

OP_NAME = "config_str"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Path", "String"),
    "kwargs": {},
    "returns": ("String",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
