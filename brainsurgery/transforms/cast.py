from __future__ import annotations

from dataclasses import dataclass
from typing import List

import re
import torch

from ..transform import (
    BaseTransform,
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    must_model,
    parse_model_expr,
    register_transform,
    require_nonempty_string,
    validate_payload_keys,
)


class CastTransformError(TransformError):
    pass


@dataclass(frozen=True)
class CastSpec:
    target_ref: TensorRef
    dtype: torch.dtype


def parse_dtype(raw: str) -> torch.dtype:
    value = raw.strip().lower()

    aliases = {
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "fp64": torch.float64,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "short": torch.int16,
        "int32": torch.int32,
        "int": torch.int32,
        "int64": torch.int64,
        "long": torch.int64,
        "bool": torch.bool,
    }

    try:
        return aliases[value]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise CastTransformError(f"unsupported dtype {raw!r}; expected one of: {allowed}") from exc


class CastTransform(BaseTransform):
    name = "cast"

    def compile(self, payload: dict, default_model: str | None) -> CastSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys={"target", "to"},
            required_keys={"target", "to"},
        )

        raw_target = require_nonempty_string(payload, op_name=self.name, key="target")
        raw_dtype = require_nonempty_string(payload, op_name=self.name, key="to")

        target_ref = parse_model_expr(raw_target, default_model=default_model)
        if target_ref.slice_spec is not None:
            raise CastTransformError("cast does not support tensor slices; cast the whole tensor")

        dtype = parse_dtype(raw_dtype)

        assert target_ref.model is not None
        return CastSpec(target_ref=target_ref, dtype=dtype)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        if not isinstance(spec, CastSpec):
            raise CastTransformError(f"cast received wrong spec type: {type(spec).__name__}")

        targets = resolve_cast_targets(spec, provider)
        apply_cast_targets(spec, targets, provider)
        return TransformResult(name=self.name, count=len(targets))

    def infer_output_model(self, spec: object) -> str:
        if not isinstance(spec, CastSpec):
            raise CastTransformError(f"cast received wrong spec type: {type(spec).__name__}")

        model = spec.target_ref.model
        if model is None:
            raise CastTransformError("cast output model missing")
        return model


def resolve_cast_targets(spec: CastSpec, provider: StateDictProvider) -> List[str]:
    model = must_model(spec.target_ref)
    sd = provider.get_state_dict(model)

    matches = sorted(name for name in sd.keys() if re.fullmatch(spec.target_ref.expr, name))
    if not matches:
        raise CastTransformError(f"cast matched zero tensors: {model}::{spec.target_ref.expr}")

    return matches


def apply_cast_targets(
    spec: CastSpec,
    targets: List[str],
    provider: StateDictProvider,
) -> None:
    model = must_model(spec.target_ref)
    sd = provider.get_state_dict(model)

    for name in targets:
        tensor = sd[name]
        sd[name] = tensor.to(dtype=spec.dtype)


register_transform(CastTransform())

