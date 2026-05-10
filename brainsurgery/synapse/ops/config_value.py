from __future__ import annotations

from typing import Any

from ._config_common import (
    _resolve_config_value,
)

OP_NAME = "config_value"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Path", "Any"),
    "kwargs": {},
    "returns": ("Any",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
