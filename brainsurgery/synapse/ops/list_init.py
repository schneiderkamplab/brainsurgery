from __future__ import annotations

from typing import Any

OP_NAME = "list_init"


LOWERING_TYPE_SIGNATURE = {
    "args": (),
    "kwargs": {},
    "returns": ("List[_T]",),
}

PRIMITIVE_SEMANTICS = {
    # Fresh containers are pure to construct at the call site, but their
    # identity must not be CSE'd, hoisted into a global, or otherwise shared.
    "usage": "affine",
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "PRIMITIVE_SEMANTICS",
]
