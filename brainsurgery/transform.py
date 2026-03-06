from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple

import re
import torch

if TYPE_CHECKING:
    from .plan import SurgeryPlan


class TransformError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorRef:
    model: Optional[str]
    expr: str
    slice_spec: Optional[str] = None


@dataclass(frozen=True)
class TransformResult:
    name: str
    count: int


@dataclass(frozen=True)
class CompiledTransform:
    transform: "BaseTransform"
    spec: object


@dataclass(frozen=True)
class ResolvedMapping:
    src_model: str
    src_name: str
    src_slice: tuple[object, ...] | None
    dst_model: str
    dst_name: str
    dst_slice: tuple[object, ...] | None


class StateDictLike(MutableMapping[str, torch.Tensor]):
    @abstractmethod
    def slot(self, key: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def bind_slot(self, key: str, slot: Any) -> None:
        raise NotImplementedError


class StateDictProvider(Protocol):
    def get_state_dict(self, model: str) -> StateDictLike:
        ...


class BaseTransform(ABC):
    name: str

    @abstractmethod
    def compile(self, payload: dict, default_model: str | None) -> object:
        raise NotImplementedError

    @abstractmethod
    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        raise NotImplementedError

    @abstractmethod
    def infer_output_model(self, spec: object) -> str:
        raise NotImplementedError


_REGISTRY: Dict[str, BaseTransform] = {}


def register_transform(transform: BaseTransform) -> None:
    name = getattr(transform, "name", None)
    if not isinstance(name, str) or not name:
        raise TransformError("transform must define a non-empty string 'name'")
    if name in _REGISTRY:
        raise TransformError(f"transform already registered: {name}")
    _REGISTRY[name] = transform


def get_transform(name: str) -> BaseTransform:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise TransformError(f"unknown transform: {name}") from exc


def list_transforms() -> List[str]:
    return sorted(_REGISTRY.keys())


def apply_transform(compiled: CompiledTransform, provider: StateDictProvider) -> TransformResult:
    return compiled.transform.apply(compiled.spec, provider)


def infer_output_model(plan: SurgeryPlan) -> str:
    destination_models = set()

    for compiled in plan.transforms:
        destination_models.add(compiled.transform.infer_output_model(compiled.spec))

    if len(destination_models) != 1:
        raise TransformError(
            "cannot infer output model uniquely; expected exactly one destination model across all transforms"
        )

    return next(iter(destination_models))


def parse_model_expr(raw: str, default_model: Optional[str] = None) -> TensorRef:
    if not isinstance(raw, str) or not raw:
        raise TransformError("reference must be a non-empty string")

    parts = raw.split("::")
    if len(parts) == 1:
        if default_model is None:
            raise TransformError(f"missing model alias in reference: {raw!r}")
        return TensorRef(model=default_model, expr=parts[0], slice_spec=None)

    if len(parts) == 2:
        head, tail = parts
        if default_model is not None and looks_like_slice(tail):
            return TensorRef(model=default_model, expr=head, slice_spec=tail)
        return TensorRef(model=head or default_model, expr=tail, slice_spec=None)

    if len(parts) == 3:
        head, expr, slice_spec = parts
        if not looks_like_slice(slice_spec):
            raise TransformError(f"invalid slice syntax in reference: {raw!r}")
        model = head or default_model
        if model is None:
            raise TransformError(f"missing model alias in reference: {raw!r}")
        return TensorRef(model=model, expr=expr, slice_spec=slice_spec)

    raise TransformError(f"invalid reference syntax: {raw!r}")


def parse_slice(raw: str) -> Tuple[object, ...]:
    if not looks_like_slice(raw):
        raise TransformError(f"invalid slice syntax: {raw!r}")

    inner = raw[1:-1].strip()
    if not inner:
        return tuple()

    parts = [part.strip() for part in inner.split(",")]
    if any(part == "" for part in parts):
        raise TransformError(f"invalid empty slice component in {raw!r}")

    return tuple(parse_slice_component(part) for part in parts)


def parse_slice_component(raw: str) -> object:
    if raw == ":":
        return slice(None, None, None)

    if ":" not in raw:
        return parse_int(raw)

    parts = raw.split(":")
    if len(parts) not in (2, 3):
        raise TransformError(f"invalid slice component: {raw!r}")

    start = parse_optional_int(parts[0])
    stop = parse_optional_int(parts[1])
    step = parse_optional_int(parts[2]) if len(parts) == 3 else None
    return slice(start, stop, step)


def select_tensor(tensor: torch.Tensor, slice_spec: Optional[Tuple[object, ...]]) -> torch.Tensor:
    if slice_spec is None:
        return tensor
    try:
        return tensor[slice_spec]
    except Exception as exc:  # pragma: no cover
        raise TransformError(
            f"failed to apply slice {slice_spec!r} to tensor with shape {tuple(tensor.shape)}"
        ) from exc


def parse_int(raw: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise TransformError(f"invalid integer in slice component: {raw!r}") from exc


def parse_optional_int(raw: str) -> Optional[int]:
    return None if raw == "" else parse_int(raw)


def looks_like_slice(raw: str) -> bool:
    return raw.startswith("[") and raw.endswith("]")


def must_model(ref: TensorRef) -> str:
    if ref.model is None:
        raise TransformError(f"reference is missing model alias: {ref}")
    return ref.model


def resolve_name_mappings(
    *,
    from_ref: TensorRef,
    to_ref: TensorRef,
    provider: StateDictProvider,
    op_name: str,
) -> List[ResolvedMapping]:
    src_model = must_model(from_ref)
    dst_model = must_model(to_ref)

    src_sd = provider.get_state_dict(src_model)

    src_names = sorted(name for name in src_sd.keys() if re.fullmatch(from_ref.expr, name))
    if not src_names:
        raise TransformError(f"{op_name} source matched zero tensors: {src_model}::{from_ref.expr}")

    src_slice = parse_slice(from_ref.slice_spec) if from_ref.slice_spec else None
    dst_slice = parse_slice(to_ref.slice_spec) if to_ref.slice_spec else None

    dst_names_seen: set[str] = set()
    resolved: List[ResolvedMapping] = []

    for src_name in src_names:
        dst_name = re.sub(from_ref.expr, to_ref.expr, src_name)

        if dst_name in dst_names_seen:
            raise TransformError(f"{op_name} destination collision: {dst_model}::{dst_name}")

        dst_names_seen.add(dst_name)
        resolved.append(
            ResolvedMapping(
                src_model=src_model,
                src_name=src_name,
                src_slice=src_slice,
                dst_model=dst_model,
                dst_name=dst_name,
                dst_slice=dst_slice,
            )
        )

    return resolved


def require_dest_missing(
    *,
    mappings: List[ResolvedMapping],
    provider: StateDictProvider,
    op_name: str,
) -> None:
    for item in mappings:
        dst_sd = provider.get_state_dict(item.dst_model)
        if item.dst_name in dst_sd:
            raise TransformError(f"{op_name} destination already exists: {item.dst_model}::{item.dst_name}")


def require_dest_present(
    *,
    mappings: List[ResolvedMapping],
    provider: StateDictProvider,
    op_name: str,
) -> None:
    for item in mappings:
        dst_sd = provider.get_state_dict(item.dst_model)
        if item.dst_name not in dst_sd:
            raise TransformError(f"{op_name} destination missing: {item.dst_model}::{item.dst_name}")


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


def require_nonempty_string(payload: dict, *, op_name: str, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise TransformError(f"{op_name}.{key} must be a non-empty string")
    return value


def require_numeric(payload: dict, *, op_name: str, key: str) -> float:
    value = payload.get(key)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TransformError(f"{op_name}.{key} must be numeric") from exc
