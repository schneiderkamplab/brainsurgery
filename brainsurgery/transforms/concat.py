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


class ConcatTransformError(TransformError):
    pass


@dataclass(frozen=True)
class ConcatSpec:
    from_refs: list[TensorRef]
    to_ref: TensorRef
    dim: int

    def collect_models(self) -> set[str]:
        models = {must_model(self.to_ref)}
        models.update(must_model(ref) for ref in self.from_refs)
        return models


class ConcatTransform(BaseTransform):
    name = "concat"
    error_type = ConcatTransformError
    spec_type = ConcatSpec
    help_text = (
        "Concatenates multiple source tensors into one destination tensor.\n"
        "\n"
        "'from' must be a list of at least two references. Each source reference must resolve\n"
        "to exactly one tensor. 'to' must be a single destination tensor reference.\n"
        "\n"
        "Examples:\n"
        "  concat: { from: [a::x, a::y], to: a::xy, dim: 0 }\n"
        "  concat: { from: ['a::x::[:, :4]', 'a::x::[:, 4:]'], to: a::x_rebuilt, dim: 1 }"
    )

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key=None,
            default_mode="default",
            common_required={"from", "to"},
            common_allowed={"from", "to", "dim"},
        )

    def completion_reference_keys(self) -> list[str]:
        return ["from", "to"]

    def compile(self, payload: Any, default_model: str | None) -> ConcatSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_schema(
            payload,
            op_name=self.name,
            schema=self.payload_schema(),
            error_type=self.error_type,
        )

        raw_from = payload.get("from")
        if not isinstance(raw_from, list) or len(raw_from) < 2:
            raise ConcatTransformError("concat.from must be a list of at least two references")

        from_refs: list[TensorRef] = []
        for idx, item in enumerate(raw_from):
            ref = parse_model_expr(item, default_model=default_model)
            if ref.slice_spec is not None:
                parse_slice(ref.slice_spec)
            if ref.model is None:
                raise ConcatTransformError(f"concat.from[{idx}] missing model alias")
            from_refs.append(ref)

        to_ref = parse_model_expr(payload.get("to"), default_model=default_model)
        if to_ref.model is None:
            raise ConcatTransformError("concat.to missing model alias")
        if to_ref.slice_spec is not None:
            raise ConcatTransformError("concat.to must not be sliced")
        if not isinstance(to_ref.expr, str):
            raise ConcatTransformError("concat.to must resolve to a single tensor name")

        raw_dim = payload.get("dim", 0)
        if not isinstance(raw_dim, int):
            raise ConcatTransformError("concat.dim must be an integer")

        return ConcatSpec(from_refs=from_refs, to_ref=to_ref, dim=raw_dim)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        dst_model = must_model(typed.to_ref)
        assert isinstance(typed.to_ref.expr, str)
        dst_name = typed.to_ref.expr
        dst_sd = provider.get_state_dict(dst_model)
        if dst_name in dst_sd:
            raise ConcatTransformError(
                f"concat destination already exists: {dst_model}::{dst_name}"
            )

        source_tensors = [self._resolve_source_tensor(ref, provider) for ref in typed.from_refs]
        self._validate_sources(source_tensors, dim=typed.dim)
        rank = source_tensors[0].dim()
        cat_dim = typed.dim if typed.dim >= 0 else typed.dim + rank

        dst_sd[dst_name] = torch.cat(source_tensors, dim=cat_dim).clone()
        emit_verbose_event(self.name, f"{len(source_tensors)} tensors -> {dst_name}")
        return TransformResult(name=self.name, count=1)

    def _infer_output_model(self, spec: object) -> str:
        typed = self.require_spec(spec)
        return must_model(typed.to_ref)

    def require_spec(self, spec: object) -> ConcatSpec:
        if not isinstance(spec, ConcatSpec):
            raise ConcatTransformError(f"concat received wrong spec type: {type(spec).__name__}")
        return spec

    def _resolve_source_tensor(self, ref: TensorRef, provider: StateDictProvider) -> torch.Tensor:
        src_model = must_model(ref)
        src_sd = state_dict_for_ref(provider, ref)
        matches = match_expr_names(
            expr=ref.expr,
            names=src_sd.keys(),
            op_name=self.name,
            role="source",
        )
        if not matches:
            raise ConcatTransformError(
                f"concat source matched zero tensors: {src_model}::{ref.expr}"
            )
        if len(matches) != 1:
            raise ConcatTransformError(
                "concat source must match exactly one tensor, "
                f"got {len(matches)}: {src_model}::{ref.expr}"
            )
        src_name = matches[0]
        _sd, src_view = view_for_ref_name(provider, ref, src_name)
        return src_view

    def _validate_sources(self, tensors: list[torch.Tensor], *, dim: int) -> None:
        if not tensors:
            raise ConcatTransformError("concat requires at least one source tensor")

        first = tensors[0]
        rank = first.dim()
        cat_dim = dim if dim >= 0 else dim + rank
        if cat_dim < 0 or cat_dim >= rank:
            raise ConcatTransformError(f"concat.dim {dim} out of range for rank {rank} tensor")

        for idx, tensor in enumerate(tensors[1:], start=1):
            if tensor.dim() != rank:
                raise ConcatTransformError(
                    f"concat source rank mismatch at index {idx}: {tensor.dim()} != {rank}"
                )
            if tensor.dtype != first.dtype:
                raise ConcatTransformError(
                    f"concat source dtype mismatch at index {idx}: {tensor.dtype} != {first.dtype}"
                )
            if tensor.device != first.device:
                raise ConcatTransformError(
                    "concat source device mismatch at index "
                    f"{idx}: {tensor.device} != {first.device}"
                )
            for axis in range(rank):
                if axis == cat_dim:
                    continue
                if tensor.shape[axis] != first.shape[axis]:
                    raise ConcatTransformError(
                        "concat source shape mismatch outside concat dimension "
                        f"at index {idx}, axis {axis}: {tensor.shape[axis]} != {first.shape[axis]}"
                    )


register_transform(ConcatTransform())
