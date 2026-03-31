from __future__ import annotations

from typing import Any

from .backends import emit_model_code_from_synapse_spec as _emit_from_backends
from .backends import load_synapse_torch_op_map as _load_op_map


def load_synapse_torch_op_map() -> dict[str, Any]:
    return _load_op_map()


def emit_model_code_from_synapse_spec(
    spec: dict[str, Any],
    *,
    class_name: str = "GeneratedSynapseModel",
    op_map: dict[str, Any] | None = None,
    backend: str = "pytorch",
) -> str:
    return _emit_from_backends(
        spec, class_name=class_name, op_map=op_map, backend=backend
    )
