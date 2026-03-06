from __future__ import annotations

from dataclasses import dataclass
from typing import List

from brainsurgery.model import tqdm

from ..model import tqdm
from ..transform import (
    BaseTransform,
    ResolvedMapping,
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    must_model,
    parse_model_expr,
    parse_slice,
    register_transform,
    require_nonempty_string,
    require_dest_missing,
    resolve_name_mappings,
    select_tensor,
    validate_payload_keys,
)


class CopyTransformError(TransformError):
    pass


@dataclass(frozen=True)
class CopySpec:
    from_ref: TensorRef
    to_ref: TensorRef


class CopyTransform(BaseTransform):
    name = "copy"

    def compile(self, payload: dict, default_model: str | None) -> CopySpec:
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
            parse_slice(from_ref.slice_spec)
        if to_ref.slice_spec is not None:
            raise CopyTransformError("copy destination must not be sliced")

        assert from_ref.model is not None
        assert to_ref.model is not None
        return CopySpec(from_ref=from_ref, to_ref=to_ref)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        if not isinstance(spec, CopySpec):
            raise CopyTransformError(f"copy received wrong spec type: {type(spec).__name__}")

        resolved = resolve_copy_mappings(spec, provider)
        apply_copy_mappings(resolved, provider)
        return TransformResult(name=self.name, count=len(resolved))

    def infer_output_model(self, spec: object) -> str:
        if not isinstance(spec, CopySpec):
            raise CopyTransformError(f"copy received wrong spec type: {type(spec).__name__}")

        model = spec.to_ref.model
        if model is None:
            raise CopyTransformError("copy output model is missing")
        return model


def resolve_copy_mappings(spec: CopySpec, provider: StateDictProvider) -> List[ResolvedMapping]:
    mappings = resolve_name_mappings(
        from_ref=spec.from_ref,
        to_ref=spec.to_ref,
        provider=provider,
        op_name="copy",
    )
    require_dest_missing(
        mappings=mappings,
        provider=provider,
        op_name="copy",
    )
    return mappings


def apply_copy_mappings(mappings: List[ResolvedMapping], provider: StateDictProvider) -> None:
    for item in tqdm(mappings, desc="Applying copy transforms", unit="tensor"):
        src_sd = provider.get_state_dict(item.src_model)
        dst_sd = provider.get_state_dict(item.dst_model)

        src_tensor = src_sd[item.src_name]
        copied = select_tensor(src_tensor, item.src_slice).clone()

        if item.dst_name in dst_sd:
            raise CopyTransformError(
                f"copy destination already exists during apply: {item.dst_model}::{item.dst_name}"
            )

        dst_sd[item.dst_name] = copied


register_transform(CopyTransform())
