from dataclasses import dataclass
from typing import Any

import torch

from ..core import (
    StateDictProvider,
    TensorRef,
    TransformError,
    compile_tensor_ref_expr,
    format_ref,
    must_model,
    register_assert_expr,
    require_mapping_assert_payload,
    require_same_shape_dtype_device,
    resolve_tensor_mappings,
)


@dataclass(frozen=True)
class EqualExpr:
    left: TensorRef
    right: TensorRef
    eps: float | None = None

    def evaluate(self, provider: StateDictProvider) -> None:
        mappings = resolve_tensor_mappings(self.left, self.right, provider, op_name="equal")
        for left_ref, left, right_ref, right in mappings:
            require_same_shape_dtype_device(
                left,
                right,
                op_name="comparing",
                left_name=format_ref(left_ref),
                right_name=format_ref(right_ref),
            )

            if self.eps is None:
                is_equal = torch.equal(left, right)
            else:
                if left.is_complex():
                    diff = torch.abs(left.to(torch.complex128) - right.to(torch.complex128))
                else:
                    diff = torch.abs(left.to(torch.float64) - right.to(torch.float64))
                is_equal = bool(torch.all(diff <= self.eps).item())

            if not is_equal:
                raise TransformError(
                    f"equal failed: {format_ref(left_ref)} != {format_ref(right_ref)}"
                )

    def collect_models(self) -> set[str]:
        return {must_model(self.left), must_model(self.right)}


@register_assert_expr(
    "equal",
    payload_kind="mapping",
    allowed_keys={"left", "right", "eps"},
    required_keys={"left", "right"},
    description=(
        "Succeeds if two tensors have the same shape and dtype, and their values are equal "
        "(or within 'eps' absolute tolerance when provided). 'right' is resolved as a rewrite "
        "of each 'left' match, like 'to' in copy/move: capture groups from 'left' may be used "
        "in 'right' (\\1, \\g<0>) and aliases may differ, e.g. "
        "equal: { left: 'a::(.+)', right: 'b::\\1' } compares every tensor of alias a with "
        "the same name in alias b."
    ),
)
def compile_equal_expr(payload: Any, default_model: str | None) -> EqualExpr:
    payload = require_mapping_assert_payload(
        payload,
        op_name="equal",
        allowed_keys={"left", "right", "eps"},
        required_keys={"left", "right"},
    )
    left = compile_tensor_ref_expr(payload["left"], default_model, "equal.left")
    right = compile_tensor_ref_expr(payload["right"], default_model, "equal.right")
    raw_eps = payload.get("eps")
    if raw_eps is None:
        eps = None
    else:
        if isinstance(raw_eps, bool) or not isinstance(raw_eps, (int, float)):
            raise TransformError("equal.eps must be a non-negative number")
        eps = float(raw_eps)
        if eps < 0:
            raise TransformError("equal.eps must be a non-negative number")
    return EqualExpr(left=left, right=right, eps=eps)
