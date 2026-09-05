from __future__ import annotations

from typing import Any


OP_NAME = "where_indices"


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any",),
    "kwargs": {},
    "returns": ("IdxTensor[..I]", "IdxTensor[..J]"),
}


PRIMITIVE_SEMANTICS = {
    # Both outputs are 1D index tensors whose length is determined by the
    # runtime number of true elements, not by any input shape dimension.
    "value_dependent_output_dim_groups": ((((0, 0), (1, 0)),)),
}

__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "PRIMITIVE_SEMANTICS",
]
