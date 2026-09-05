from __future__ import annotations

from typing import Any


from ..axon.ast import AxonExprAscribe, AxonExprInt, AxonExprParen, DimExprBinary, TypeTensor

OP_NAME = "concat"


def _dim_add(left: Any, right: Any) -> Any:
    if isinstance(left, int) and isinstance(right, int):
        return left + right
    if (
        isinstance(left, DimExprBinary)
        and left.op == "/"
        and isinstance(right, DimExprBinary)
        and right.op == "/"
        and left.left == right.left
        and left.right == right.right == 2
    ):
        return left.left
    if isinstance(right, DimExprBinary) and right.op == "-" and right.right == left:
        return right.left
    if isinstance(left, DimExprBinary) and left.op == "-" and left.right == right:
        return left.left
    return DimExprBinary(op="+", left=left, right=right)


def _resolve_dim_alias(dim: Any, helpers: Any) -> Any:
    if not isinstance(dim, str):
        return dim
    resolved = helpers.resolve_name_expr(dim)
    if resolved is None:
        return dim
    token = helpers.expr_to_dim_token(resolved)
    return dim if token is None else token


LOWERING_TYPE_SIGNATURE = {
    "args": ("Any", "Any"),
    "kwargs": {'dim': 'int'},
    "returns": "dynamic",
}
LOWERING_PARAM_NAMES = ("x", "y", "dim")
LOWERING_PARAM_DEFAULTS = {"dim": -1}


def type_rule(
    *,
    arg_types: tuple[Any, ...],
    kwarg_types: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    helpers: Any,
) -> Any | None:
    del kwarg_types
    if len(arg_types) not in {2, 3}:
        return None
    left_dims = helpers.type_dims(arg_types[0])
    right_dims = helpers.type_dims(arg_types[1])
    if left_dims is None or right_dims is None:
        return None
    output_base = "Tensor"
    if (
        isinstance(arg_types[0], TypeTensor)
        and isinstance(arg_types[1], TypeTensor)
        and arg_types[0].base == arg_types[1].base
    ):
        output_base = arg_types[0].base
    if len(left_dims) != len(right_dims):
        if any(isinstance(dim, str) and dim.startswith("..") for dim in left_dims):
            return TypeTensor(base=output_base, dims=tuple(left_dims))
        if any(isinstance(dim, str) and dim.startswith("..") for dim in right_dims):
            return TypeTensor(base=output_base, dims=tuple(right_dims))
        return None
    raw_dim = args[2] if len(args) > 2 else kwargs.get("dim", -1)
    while isinstance(raw_dim, AxonExprAscribe | AxonExprParen):
        raw_dim = raw_dim.expr if isinstance(raw_dim, AxonExprAscribe) else raw_dim.inner
    if isinstance(raw_dim, AxonExprInt):
        raw_dim = raw_dim.value
    else:
        resolved_dim = helpers.expr_to_dim_token(raw_dim)
        if isinstance(resolved_dim, int):
            raw_dim = resolved_dim
    if isinstance(raw_dim, bool) or not isinstance(raw_dim, int):
        return None
    rank = len(left_dims)
    dim = raw_dim if raw_dim >= 0 else rank + raw_dim
    if dim < 0 or dim >= rank:
        return None
    out_dims: list[Any] = []
    for idx, (left_dim, right_dim) in enumerate(zip(left_dims, right_dims, strict=True)):
        if idx == dim:
            out_dims.append(
                _dim_add(
                    _resolve_dim_alias(left_dim, helpers),
                    _resolve_dim_alias(right_dim, helpers),
                )
            )
            continue
        dim_equivalent = getattr(helpers, "dim_equivalent", None)
        if left_dim != right_dim and not (
            callable(dim_equivalent) and dim_equivalent(left_dim, right_dim)
        ):
            return None
        out_dims.append(left_dim)
    return TypeTensor(base=output_base, dims=tuple(out_dims))


__all__ = [
    "OP_NAME",
    "LOWERING_TYPE_SIGNATURE",
    "LOWERING_PARAM_DEFAULTS",
    "LOWERING_PARAM_NAMES",
    "type_rule",
]
