from __future__ import annotations

from typing import Any

from ._config_common import (
    _resolve_config_value,
)

OP_NAME = "config_bool"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Path", "?Bool"),
    "kwargs": {},
    "returns": ("Bool",),
}
LOWERING_PARAM_NAMES = ("key", "default")
LOWERING_PARAM_DEFAULTS = {"default": None}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "LOWERING_PARAM_DEFAULTS",
    "LOWERING_PARAM_NAMES",
]
