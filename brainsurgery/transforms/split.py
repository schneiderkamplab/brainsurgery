from dataclasses import dataclass
from typing import Any

import torch

from ..core import (
    BaseTransform,
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformPayloadSchema,
    TransformResult,
    ensure_mapping_payload,
    match_expr_names,
    must_model,
    parse_model_expr,
    parse_slice,
    register_transform,
    state_dict_for_ref,
    validate_payload_schema,
    view_for_ref_name,
)
from ..engine import emit_verbose_event


class SplitTransformError(TransformError):
    pass


@dataclass(frozen=True)
class SplitSpec:
    from_ref: TensorRef
    to_refs: list[TensorRef]
    sizes: list[int]
    dim: int

    def collect_models(self) -> set[str]:
        models = {must_model(self.from_ref)}
        models.update(must_model(ref) for ref in self.to_refs)
        return models


class SplitTransform(BaseTransform):
    name = "split"
    error_type = SplitTransformError
    spec_type = SplitSpec
    allowed_keys = {"from", "to", "sizes", "dim"}
    required_keys = {"from", "to", "sizes"}
    help_text = (
        "Splits one source tensor into multiple destination tensors.\n"
        "\n"
        "'sizes' must sum to the source size along 'dim'. Destination refs must be\n"
        "single unsliced tensor names and must not already exist.\n"
        "\n"
        "Example:\n"
        "  split: { from: x, to: [x0, x1], sizes: [32, 32], dim: 0 }"
    )

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key=None,
            default_mode="default",
            common_required=set(self.required_keys),
            common_allowed=set(self.allowed_keys),
        )

    def completion_reference_keys(self) -> list[str]:
        return ["from", "to"]

    def compile(self, payload: Any, default_model: str | None) -> SplitSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_schema(
            payload,
            op_name=self.name,
            schema=self.payload_schema(),
            error_type=self.error_type,
        )
        from_ref = parse_model_expr(payload.get("from"), default_model=default_model)
        if from_ref.slice_spec is not None:
            parse_slice(from_ref.slice_spec)
        if from_ref.model is None:
            raise SplitTransformError("split.from missing model alias")

        raw_to = payload.get("to")
        if not isinstance(raw_to, list) or len(raw_to) < 2:
            raise SplitTransformError("split.to must be a list of at least two references")
        to_refs: list[TensorRef] = []
        for idx, item in enumerate(raw_to):
            ref = parse_model_expr(item, default_model=default_model)
            if ref.model is None:
                raise SplitTransformError(f"split.to[{idx}] missing model alias")
            if ref.slice_spec is not None:
                raise SplitTransformError("split destination references must not be sliced")
            if not isinstance(ref.expr, str):
                raise SplitTransformError(
                    "split destination references must resolve to single names"
                )
            to_refs.append(ref)

        sizes = _parse_sizes(payload.get("sizes"))
        if len(sizes) != len(to_refs):
            raise SplitTransformError("split.sizes length must match split.to length")

        raw_dim = payload.get("dim", 0)
        if not isinstance(raw_dim, int):
            raise SplitTransformError("split.dim must be an integer")

        return SplitSpec(from_ref=from_ref, to_refs=to_refs, sizes=sizes, dim=raw_dim)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        src_sd = state_dict_for_ref(provider, typed.from_ref)
        matches = match_expr_names(
            expr=typed.from_ref.expr,
            names=src_sd.keys(),
            op_name=self.name,
            role="source",
        )
        if not matches:
            raise SplitTransformError("split source matched zero tensors")
        if len(matches) != 1:
            raise SplitTransformError(
                f"split source must match exactly one tensor, got {len(matches)}"
            )
        src_name = matches[0]
        _src_sd, src_view = view_for_ref_name(provider, typed.from_ref, src_name)

        rank = src_view.dim()
        dim = typed.dim if typed.dim >= 0 else typed.dim + rank
        if dim < 0 or dim >= rank:
            raise SplitTransformError(f"split.dim {typed.dim} out of range for rank {rank} tensor")
        if sum(typed.sizes) != int(src_view.shape[dim]):
            raise SplitTransformError(
                f"split.sizes must sum to source size along dim {dim}: "
                f"{sum(typed.sizes)} != {int(src_view.shape[dim])}"
            )

        parts = torch.split(src_view, typed.sizes, dim=dim)
        for ref, part in zip(typed.to_refs, parts, strict=True):
            dst_model = must_model(ref)
            assert isinstance(ref.expr, str)
            dst_name = ref.expr
            dst_sd = provider.get_state_dict(dst_model)
            if dst_name in dst_sd:
                raise SplitTransformError(
                    f"split destination already exists: {dst_model}::{dst_name}"
                )
            dst_sd[dst_name] = part.clone()
            emit_verbose_event(self.name, f"{src_name} -> {dst_name}")

        return TransformResult(name=self.name, count=len(parts))

    def _infer_output_model(self, spec: object) -> str:
        typed = self.require_spec(spec)
        return must_model(typed.to_refs[0])

    def require_spec(self, spec: object) -> SplitSpec:
        if not isinstance(spec, SplitSpec):
            raise SplitTransformError(f"split received wrong spec type: {type(spec).__name__}")
        return spec


def _parse_sizes(raw: object) -> list[int]:
    if not isinstance(raw, list) or not raw:
        raise SplitTransformError("split.sizes must be a non-empty list of positive integers")
    if not all(isinstance(x, int) and x > 0 for x in raw):
        raise SplitTransformError("split.sizes must be a non-empty list of positive integers")
    return list(raw)


register_transform(SplitTransform())
