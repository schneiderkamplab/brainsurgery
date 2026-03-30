from dataclasses import dataclass, field

import torch

from .refs import _Expr
from .types import TransformError

_DTYPE_ALIASES: dict[str, torch.dtype] = {
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


@dataclass(frozen=True)
class TransformPayloadSchema:
    mode_key: str | None = None
    default_mode: str = "default"
    common_required: set[str] = field(default_factory=set)
    common_allowed: set[str] = field(default_factory=set)
    mode_required: dict[str, set[str]] = field(default_factory=dict)
    mode_allowed_extra: dict[str, set[str]] = field(default_factory=dict)

    def _ordered_modes(self) -> list[str]:
        ordered: list[str] = []
        for name in (
            str(self.default_mode),
            *(str(item) for item in self.mode_required.keys()),
            *(str(item) for item in self.mode_allowed_extra.keys()),
        ):
            normalized = name.strip().lower()
            if normalized and normalized not in ordered:
                ordered.append(normalized)
        return ordered

    def modes(self) -> set[str]:
        return set(self._ordered_modes())

    def resolve_mode(
        self,
        payload: dict,
        *,
        op_name: str,
        error_type: type[TransformError] = TransformError,
    ) -> str:
        if self.mode_key is None:
            return str(self.default_mode)
        raw_mode = payload.get(self.mode_key, self.default_mode)
        if not isinstance(raw_mode, str) or not raw_mode.strip():
            raise error_type(f"{op_name}.{self.mode_key} must be a non-empty string")
        mode = raw_mode.strip().lower()
        known_ordered = self._ordered_modes()
        known = set(known_ordered)
        if mode not in known:
            raise error_type(
                f"{op_name}.{self.mode_key} must be one of: {', '.join(known_ordered)}"
            )
        return mode

    def required_keys_for_mode(self, mode: str) -> set[str]:
        normalized = str(mode).lower()
        required = set(self.common_required)
        mode_required = {
            str(name).lower(): set(values) for name, values in self.mode_required.items()
        }
        required.update(mode_required.get(normalized, set()))
        return required

    def allowed_keys_for_mode(self, mode: str) -> set[str]:
        normalized = str(mode).lower()
        allowed = set(self.common_allowed)
        mode_allowed_extra = {
            str(name).lower(): set(values) for name, values in self.mode_allowed_extra.items()
        }
        allowed.update(mode_allowed_extra.get(normalized, set()))
        allowed.update(self.required_keys_for_mode(normalized))
        if self.mode_key is not None:
            allowed.add(self.mode_key)
        return allowed


def parse_torch_dtype(
    raw: str,
    *,
    error_type: type[TransformError],
    op_name: str,
    field_name: str,
) -> torch.dtype:
    value = raw.strip().lower()
    try:
        return _DTYPE_ALIASES[value]
    except KeyError as exc:
        allowed = ", ".join(sorted(_DTYPE_ALIASES))
        raise error_type(
            f"{op_name}.{field_name} unsupported dtype {raw!r}; expected one of: {allowed}"
        ) from exc


def ensure_mapping_payload(payload: object, op_name: str) -> dict:
    if not isinstance(payload, dict):
        raise TransformError(f"{op_name} payload must be a mapping")
    return payload


def validate_payload_keys(
    payload: dict,
    *,
    op_name: str,
    allowed_keys: set[str],
    required_keys: set[str] | None = None,
) -> None:
    unknown = set(payload) - allowed_keys
    if unknown:
        raise TransformError(f"{op_name} received unknown keys: {sorted(unknown)}")

    if required_keys is None:
        required_keys = set()

    missing = required_keys - set(payload)
    if missing:
        missing_list = sorted(missing)
        if len(missing_list) == 1:
            raise TransformError(f"{op_name}.{missing_list[0]} is required")
        raise TransformError(f"{op_name} is missing required keys: {missing_list}")


def validate_payload_schema(
    payload: dict,
    *,
    op_name: str,
    schema: TransformPayloadSchema,
    mode: str | None = None,
    error_type: type[TransformError] = TransformError,
) -> str:
    resolved_mode = (
        schema.resolve_mode(payload, op_name=op_name, error_type=error_type)
        if mode is None
        else str(mode).lower()
    )
    unknown = set(payload) - schema.allowed_keys_for_mode(resolved_mode)
    if unknown:
        if schema.mode_key is not None:
            raise error_type(
                f"{op_name} received unknown keys for this mode ({resolved_mode}): {sorted(unknown)}"
            )
        raise error_type(f"{op_name} received unknown keys: {sorted(unknown)}")

    missing = schema.required_keys_for_mode(resolved_mode) - set(payload)
    if missing:
        missing_list = sorted(missing)
        if len(missing_list) == 1:
            raise error_type(f"{op_name}.{missing_list[0]} is required")
        raise error_type(f"{op_name} is missing required keys: {missing_list}")
    return resolved_mode


def require_nonempty_string(payload: dict, *, op_name: str, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise TransformError(f"{op_name}.{key} must be a non-empty string")
    return value


def require_expr(payload: dict, *, op_name: str, key: str) -> _Expr:
    value = payload.get(key)

    if isinstance(value, str):
        if not value:
            raise TransformError(
                f"{op_name}.{key} must be a non-empty string or a non-empty list of non-empty strings"
            )
        return value

    if isinstance(value, list):
        if not value:
            raise TransformError(
                f"{op_name}.{key} must be a non-empty string or a non-empty list of non-empty strings"
            )
        if not all(isinstance(item, str) and item for item in value):
            raise TransformError(
                f"{op_name}.{key} must be a non-empty string or a non-empty list of non-empty strings"
            )
        return value

    raise TransformError(
        f"{op_name}.{key} must be a non-empty string or a non-empty list of non-empty strings"
    )


def require_numeric(payload: dict, *, op_name: str, key: str) -> float:
    value = payload.get(key)
    if value is None:
        raise TransformError(f"{op_name}.{key} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TransformError(f"{op_name}.{key} must be numeric") from exc


def require_same_shape_dtype_device(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    op_name: str,
    left_name: str,
    right_name: str,
) -> None:
    if left.shape != right.shape:
        raise TransformError(
            f"shape mismatch {op_name} {left_name} -> {right_name}: "
            f"{tuple(left.shape)} != {tuple(right.shape)}"
        )
    if left.dtype != right.dtype:
        raise TransformError(
            f"dtype mismatch {op_name} {left_name} -> {right_name}: {left.dtype} != {right.dtype}"
        )
    if left.device != right.device:
        raise TransformError(
            f"device mismatch {op_name} {left_name} -> {right_name}: "
            f"{left.device} != {right.device}"
        )


def require_same_shape_dtype_device3(
    first: torch.Tensor,
    second: torch.Tensor,
    dest: torch.Tensor,
    *,
    op_name: str,
    first_name: str,
    second_name: str,
    dest_name: str,
    symbol: str,
) -> None:
    if first.shape != second.shape or first.shape != dest.shape:
        raise TransformError(
            f"shape mismatch {op_name} {first_name} {symbol} {second_name} -> {dest_name}: "
            f"{tuple(first.shape)} {symbol} {tuple(second.shape)} -> {tuple(dest.shape)}"
        )
    if first.dtype != second.dtype or first.dtype != dest.dtype:
        raise TransformError(
            f"dtype mismatch {op_name} {first_name} {symbol} {second_name} -> {dest_name}: "
            f"{first.dtype} {symbol} {second.dtype} -> {dest.dtype}"
        )
    if first.device != second.device or first.device != dest.device:
        raise TransformError(
            f"device mismatch {op_name} {first_name} {symbol} {second_name} -> {dest_name}: "
            f"{first.device} {symbol} {second.device} -> {dest.device}"
        )


__all__ = [
    "ensure_mapping_payload",
    "parse_torch_dtype",
    "TransformPayloadSchema",
    "require_expr",
    "require_nonempty_string",
    "require_numeric",
    "require_same_shape_dtype_device",
    "require_same_shape_dtype_device3",
    "validate_payload_schema",
    "validate_payload_keys",
]
