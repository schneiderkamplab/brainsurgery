from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..codegen2_common import normalize_primitive_op
from ..graph_ir import GraphProgram


# These primitives need a policy decision or a non-trivial tinygrad implementation.
# The table is intentionally explicit so the backend fails before silently emitting
# incorrect code.
NON_OBVIOUS_TINYGRAD_OPS: dict[str, str] = {
    "embedding": "can be expressed as gather/indexing, but requires correct tinygrad integer indexing semantics and weight placement",
    "linear": "matmul is obvious; expert slicing, transpose convention, bias/path leaves, and dtype policy need backend-specific handling",
    "layernorm": "can be composed, but weight/bias path handling and epsilon/dtype behavior need parity checks",
    "rmsnorm": "composable, but dtype/cast_float behavior needs parity checks",
    "activations_gelu": "tinygrad exact/approx GELU parity must be chosen",
    "activations_gelu_new": "composable tanh approximation; needs fidelity validation",
    "activations_gelu_pytorch_tanh": "composable tanh approximation; needs fidelity validation",
    "activations_gegelu": "compound gated GELU with clipping rules; needs explicit implementation",
    "activations_xielu": "custom activation; no obvious tinygrad primitive",
    "l2norm": "composable, but dtype preservation and epsilon behavior need parity checks",
    "reshape": "obvious if shape values are Python ints; symbolic shape evaluation must be shared",
    "arange": "tinygrad arange/device/dtype behavior must match torch path",
    "slice": "basic slicing likely works; dynamic dim/start/end needs codegen policy",
    "chunk": "needs split/chunk return semantics matching Axon destructuring/list usage",
    "split": "needs list-of-tensors return semantics and dynamic sizes",
    "concat": "tinygrad cat equivalent likely exists; dynamic list flattening needs policy",
    "repeat": "repeat_interleave equivalent is not obviously direct",
    "expand": "broadcast/expand semantics differ across tensor libraries; needs parity checks",
    "permute": "obvious if tinygrad Tensor.permute is available; dynamic axis list needs policy",
    "transpose": "obvious if swapaxes/permute policy is chosen",
    "unsqueeze": "obvious if reshape/expand_dims policy is chosen",
    "softmax": "likely available; dtype override/fp32 compute policy needs parity checks",
    "sum": "likely available; keepdim/dynamic dim behavior needs parity checks",
    "where": "tensor where may exist; scalar eager ternary behavior must stay Axon-correct",
    "gather": "indexing/gather semantics are non-obvious in tinygrad",
    "scatter": "scatter with tensor/scalar source is non-obvious in tinygrad",
    "index_add": "indexed accumulation semantics are non-obvious in tinygrad",
    "topk": "topk sorted/largest/indices semantics need tinygrad support validation",
    "where_indices": "torch.where(condition) index tuple equivalent is non-obvious",
    "tensor_like": "scalar/list-to-tensor creation with device/dtype matching needs backend runtime helpers",
    "dtype_value": "dtype finfo lookup must map tinygrad dtypes to scalar constants",
    "cast": "dtype-name mapping to tinygrad dtypes needed",
    "cast_like": "dtype/device matching helper needed",
    "cumsum": "tinygrad cumsum support and dtype behavior need validation",
    "empty_like": "uninitialized tensor creation semantics may not exist or may be unsafe",
    "empty": "uninitialized tensor creation semantics may not exist or may be unsafe",
    "fill": "full_like equivalent plus dtype override needed",
    "zeros": "zeros creation with reference device/dtype needed",
    "full": "full creation with reference device/dtype needed",
    "zeros_like": "likely available; dtype/device behavior needs validation",
}

SHARED_COMMON_PRIMITIVES: frozenset[str] = frozenset({
    "params_param",
    "params_has_root",
    "config_int",
    "config_dim",
    "config_float",
    "config_bool",
    "config_str",
    "config_value",
    "config_list",
    "config_has",
    "config_has_value",
    "shape",
    "list_init",
    "list_append",
    "list_index",
    "require",
})

OBVIOUS_TINYGRAD_PRIMITIVES: frozenset[str] = frozenset({
    "add",
    "mul",
    "div",
    "pow",
    "floor",
    "sqrt",
    "sin",
    "cos",
    "exp",
    "log",
    "matmul",
    "le",
    "eq",
    "and",
    "activations_tanh",
    "activations_silu",
    "activations_sigmoid",
    "activations_relu",
    "activations_relu2",
    "clamp",
})

@dataclass(frozen=True)
class TinygradUnsupportedOp:
    op: str
    count: int
    reason: str


def _normalize_primitive_op(name: str) -> str:
    return normalize_primitive_op(name)


def non_obvious_tinygrad_ops(graph: GraphProgram) -> tuple[TinygradUnsupportedOp, ...]:
    counts: Counter[str] = Counter()
    module_names = {module.name for module in graph.modules}
    for module in graph.modules:
        for node in module.nodes:
            op = node.op.name
            if op.startswith("core.") or op.startswith("core.binary.") or op in module_names:
                continue
            primitive = _normalize_primitive_op(op)
            if primitive in SHARED_COMMON_PRIMITIVES or primitive in OBVIOUS_TINYGRAD_PRIMITIVES:
                continue
            counts[primitive] += 1
    return tuple(
        TinygradUnsupportedOp(
            op=op,
            count=count,
            reason=NON_OBVIOUS_TINYGRAD_OPS.get(op, "no tinygrad lowering classified yet"),
        )
        for op, count in sorted(counts.items())
    )


def tinygrad_op_table_markdown(graph: GraphProgram) -> str:
    rows = non_obvious_tinygrad_ops(graph)
    if not rows:
        return "| Op | Count | Reason |\n|---|---:|---|\n"
    lines = ["| Op | Count | Reason |", "|---|---:|---|"]
    for row in rows:
        lines.append(f"| `{row.op}` | {row.count} | {row.reason} |")
    return "\n".join(lines)


def emit_model_code_from_graph_ir(
    graph: GraphProgram,
    *,
    class_name: str = "AxonTinygradModel",
    model_config: dict[str, Any] | None = None,
) -> str:
    del class_name, model_config
    table = tinygrad_op_table_markdown(graph)
    raise NotImplementedError(
        "codegen2-tinygrad is scaffolded but op emission is not implemented yet.\n"
        "Non-obvious or unimplemented Graph IR ops for this model:\n"
        f"{table}"
    )


__all__ = [
    "NON_OBVIOUS_TINYGRAD_OPS",
    "OBVIOUS_TINYGRAD_PRIMITIVES",
    "SHARED_COMMON_PRIMITIVES",
    "TinygradUnsupportedOp",
    "emit_model_code_from_graph_ir",
    "non_obvious_tinygrad_ops",
    "tinygrad_op_table_markdown",
]
