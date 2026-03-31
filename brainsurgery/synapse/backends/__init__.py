from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from typing import Any

from ..ops import OP_MODULES, get_op_module


def load_synapse_torch_op_map() -> dict[str, Any]:
    """Load the default PyTorch op map from the bundled YAML config."""
    from .pytorch.emitter import load_synapse_torch_op_map as _load_pytorch_op_map

    return _load_pytorch_op_map()


# ---------------------------------------------------------------------------
# Backend registry: auto-discover backend emitter classes
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def _discover_backends() -> dict[str, type]:
    """Auto-discover backend emitter classes exposed in sub-modules."""
    package_path = globals().get("__path__", [])
    discovered: dict[str, type] = {}

    for module_info in iter_modules(package_path):
        name = module_info.name
        if name.startswith("_") or name == "ops":
            continue

        qualified = f"{__name__}.{name}"
        try:
            module = import_module(qualified)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to import backend module: {qualified}"
            ) from exc

        emitter_cls = getattr(module, "EMITTER_CLASS", None)
        if emitter_cls is None:
            continue
        if not isinstance(emitter_cls, type):
            raise RuntimeError(
                f"Backend module {qualified!r} EMITTER_CLASS must be a class, "
                f"got {type(emitter_cls).__name__}"
            )
        backend_name = getattr(module, "BACKEND_NAME", None)
        if not isinstance(backend_name, str) or not backend_name:
            backend_name = name

        if backend_name in discovered:
            raise RuntimeError(
                f"Duplicate backend name {backend_name!r} discovered in "
                f"{discovered[backend_name].__module__!r} and {qualified!r}"
            )
        discovered[backend_name] = emitter_cls

    return discovered


_REGISTRY = _discover_backends()


def get_backend(name: str) -> type | None:
    """Return the emitter class for the named backend, or None."""
    return _REGISTRY.get(name)


def list_backends() -> list[str]:
    """Return sorted list of registered backend names."""
    return sorted(_REGISTRY.keys())


def emit_model_code_from_synapse_spec(
    spec: dict[str, Any],
    *,
    class_name: str = "GeneratedSynapseModel",
    op_map: dict[str, Any] | None = None,
    backend: str = "pytorch",
) -> str:
    """Emit model source code from a synapse spec using the named backend.

    For backward compatibility, when *backend* is ``"pytorch"`` (default) the
    function delegates to the existing PyTorch codegen path.  Other backends
    are looked up in the registry.
    """
    if not class_name.isidentifier():
        raise ValueError(f"Invalid class name: {class_name!r}")
    if spec.get("synapse") != 1:
        raise ValueError("Only synapse: 1 specs are supported")

    # Resolve op_map for backward compat
    resolved_op_map: dict[str, Any] | None
    if op_map is None:
        resolved_op_map = load_synapse_torch_op_map()
    else:
        if not isinstance(op_map, dict):
            raise ValueError("op_map must be a mapping")
        resolved_op_map = op_map

    _validate_spec_ops(spec, resolved_op_map)

    model = spec.get("model")
    if not isinstance(model, dict):
        raise ValueError("spec.model must be a mapping")

    symbols_raw = model.get("symbols", {})
    symbols = {k: v for k, v in symbols_raw.items() if isinstance(v, (int, float, bool))}

    emitter_cls = _REGISTRY.get(backend)
    if emitter_cls is None:
        raise ValueError(f"Unknown backend: {backend!r}")
    emitter = emitter_cls(
        class_name=class_name,
        spec=spec,
        symbols=symbols,
        op_map=resolved_op_map,
    )
    return emitter.render()


# ---------------------------------------------------------------------------
# Spec validation helpers (re-used from codegen.py)
# ---------------------------------------------------------------------------

def _validate_spec_ops(spec: dict[str, Any], op_map: dict[str, Any]) -> None:
    ops = op_map.get("ops")
    if not isinstance(ops, dict):
        raise ValueError("op map must contain mapping key 'ops'")

    known_control_ops = {"for", "call"}
    known_runtime_builtin_ops = set(OP_MODULES.keys())

    def _walk_graph(graph: list[Any]) -> None:
        for item in graph:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f"Invalid graph item: {item!r}")
            _, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                raise ValueError(f"Invalid node spec: {node_spec!r}")

            op = node_spec.get("_op")
            if isinstance(op, str):
                is_dynamic_activation = op.startswith("activations_")
                if (
                    op not in known_control_ops
                    and op not in known_runtime_builtin_ops
                    and not is_dynamic_activation
                    and op not in ops
                ):
                    raise ValueError(f"Unsupported op in spec: {op!r}")

            if "graph" in node_spec:
                nested = node_spec["graph"]
                if not isinstance(nested, list):
                    raise ValueError("node 'graph' must be a list")
                _walk_graph(nested)

            if op == "for":
                body = node_spec.get("_body")
                if not isinstance(body, list):
                    raise ValueError("for node requires list '_body'")
                _walk_graph(body)

    model = spec.get("model")
    if not isinstance(model, dict):
        raise ValueError("spec.model must be a mapping")

    graph = model.get("graph")
    if not isinstance(graph, list):
        raise ValueError("model.graph must be a list")
    _walk_graph(graph)

    blocks = model.get("blocks", {})
    if not isinstance(blocks, dict):
        raise ValueError("model.blocks must be a mapping when present")
    for block in blocks.values():
        if not isinstance(block, dict):
            raise ValueError("block spec must be mapping")
        block_graph = block.get("graph")
        if not isinstance(block_graph, list):
            raise ValueError("block.graph must be list")
        _walk_graph(block_graph)


__all__ = [
    "OP_MODULES",
    "emit_model_code_from_synapse_spec",
    "get_backend",
    "get_op_module",
    "list_backends",
    "load_synapse_torch_op_map",
]
