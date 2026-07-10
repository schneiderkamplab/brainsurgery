from __future__ import annotations

from typing import Any

from ._config_common import (
    _config_lookup,
    _config_root,
    _resolve_key,
)

OP_NAME = "config_has"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Path",),
    "kwargs": {},
    "returns": ("Bool",),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
]
