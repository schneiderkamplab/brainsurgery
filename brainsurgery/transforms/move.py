from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..transform import (
    BaseTransform,
    ResolvedMapping,
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    parse_model_expr,
    register_transform,
    require_dest_missing,
    require_nonempty_string,
    resolve_name_mappings,
    validate_payload_keys,
)


class MoveTransformError(TransformError):
    pass


@dataclass(frozen=True)
class MoveSpec:
    from_ref: TensorRef
    to_ref: TensorRef


class MoveTransform(BaseTransform):
    name = "move"

    def compile(self, payload: dict, default_model: str | None) -> MoveSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys={"from", "to"},
            required_keys={"from", "to"},
        )

        raw_from = require_nonempty_string(payload, op_name=self.name, key="from")
        raw_to = require_nonempty_string(payload, op_name=self.name, key="to")

        from_ref = parse_model_expr(raw_from, default_model=default_model)
        to_ref = parse_model_expr(raw_to, default_model=default_model)

        if from_ref.slice_spec is not None:
            raise MoveTransformError("move source must not be sliced")
        if to_ref.slice_spec is not None:
            raise MoveTransformError("move destination must not be sliced")

        assert from_ref.model is not None
        assert to_ref.model is not None
        return MoveSpec(from_ref=from_ref, to_ref=to_ref)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        if not isinstance(spec, MoveSpec):
            raise MoveTransformError(f"move received wrong spec type: {type(spec).__name__}")

        mappings = resolve_move_mappings(spec, provider)
        apply_move_mappings(mappings, provider)
        return TransformResult(name=self.name, count=len(mappings))

    def infer_output_model(self, spec: object) -> str:
        if not isinstance(spec, MoveSpec):
            raise MoveTransformError(f"move received wrong spec type: {type(spec).__name__}")

        model = spec.to_ref.model
        if model is None:
            raise MoveTransformError("move output model missing")
        return model


def resolve_move_mappings(spec: MoveSpec, provider: StateDictProvider) -> List[ResolvedMapping]:
    mappings = resolve_name_mappings(
        from_ref=spec.from_ref,
        to_ref=spec.to_ref,
        provider=provider,
        op_name="move",
    )
    require_dest_missing(
        mappings=mappings,
        provider=provider,
        op_name="move",
    )
    return mappings


def apply_move_mappings(mappings, provider) -> None:
    for item in mappings:
        src_sd = provider.get_state_dict(item.src_model)
        dst_sd = provider.get_state_dict(item.dst_model)

        slot = src_sd.slot(item.src_name)

        if item.dst_name in dst_sd:
            raise MoveTransformError(
                f"move destination already exists during apply: {item.dst_model}::{item.dst_name}"
            )

        dst_sd.bind_slot(item.dst_name, slot)
        del src_sd[item.src_name]


register_transform(MoveTransform())
