import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Literal, TypeVar

from tqdm import tqdm

from ..compile import (
    _match_structured_expr,
    _rewrite_structured_expr,
    match_expr_names,
    resolve_name_mappings,
)
from ..compile import _resolve_target_names as resolve_target_names_generic
from ..specs import (
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformPayloadSchema,
    ensure_mapping_payload,
    format_tensor_ref,
    must_model,
    parse_model_expr,
    parse_slice,
    require_expr,
    validate_payload_schema,
)


class TransformControl(Enum):
    CONTINUE = "continue"
    EXIT = "exit"


@dataclass(frozen=True)
class TransformResult:
    name: str
    count: int
    control: TransformControl = TransformControl.CONTINUE


@dataclass(frozen=True)
class CompiledTransform:
    transform: "BaseTransform"
    spec: object


class BaseTransform(ABC):
    name: str
    completion_requires_payload: bool = True

    @abstractmethod
    def compile(self, payload: dict, default_model: str | None) -> object:
        raise NotImplementedError

    @abstractmethod
    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        raise NotImplementedError

    @abstractmethod
    def _infer_output_model(self, spec: object) -> str:
        raise NotImplementedError

    def contributes_output_model(self, spec: object) -> bool:
        del spec
        return True

    def completion_reference_keys(self) -> list[str]:
        return []

    def completion_payload_start_candidates(self, prefix_text: str) -> list[str] | None:
        del prefix_text
        return None

    def completion_key_candidates(self, before_cursor: str, prefix_text: str) -> list[str] | None:
        del before_cursor, prefix_text
        return None

    def completion_value_candidates(
        self,
        value_key: str | None,
        prefix_text: str,
        model_aliases: list[str],
    ) -> list[str] | None:
        del value_key, prefix_text, model_aliases
        return None

    def completion_committed_next_candidates(self, value_key: str | None) -> list[str] | None:
        del value_key
        return None

    @abstractmethod
    def payload_schema(self) -> TransformPayloadSchema:
        raise NotImplementedError

    def resolve_payload_mode(self, payload: dict) -> str:
        return self.payload_schema().resolve_mode(payload, op_name=self.name)

    def payload_required_keys(
        self,
        *,
        payload: dict | None = None,
        mode: str | None = None,
    ) -> set[str]:
        schema = self.payload_schema()
        if mode is None:
            if payload is None:
                mode = schema.default_mode
            else:
                mode = schema.resolve_mode(payload, op_name=self.name)
        return schema.required_keys_for_mode(mode)

    def payload_allowed_keys(
        self,
        *,
        payload: dict | None = None,
        mode: str | None = None,
    ) -> set[str]:
        schema = self.payload_schema()
        if mode is None:
            if payload is None:
                mode = schema.default_mode
            else:
                mode = schema.resolve_mode(payload, op_name=self.name)
        return schema.allowed_keys_for_mode(mode)


SpecT = TypeVar("SpecT")


@dataclass(frozen=True)
class IterationProgress:
    desc: str
    unit: str
    completed: int
    total: int | None


ProgressCallback = Callable[[IterationProgress], None]
_progress_callback: ContextVar[ProgressCallback | None] = ContextVar(
    "brainsurgery_iterating_progress_callback",
    default=None,
)


@contextmanager
def use_progress_callback(callback: ProgressCallback | None):
    token = _progress_callback.set(callback)
    try:
        yield
    finally:
        _progress_callback.reset(token)


class TypedTransform(BaseTransform, ABC, Generic[SpecT]):
    error_type: type[TransformError] = TransformError
    spec_type: type[SpecT]

    def require_spec(self, spec: object) -> SpecT:
        if not isinstance(spec, self.spec_type):
            raise self.error_type(
                f"{self.name} expected {self.spec_type.__name__}, got {type(spec).__name__}"
            )
        return spec


REGISTRY: dict[str, BaseTransform] = {}


def register_transform(transform: BaseTransform) -> None:
    name = getattr(transform, "name", None)
    if not isinstance(name, str) or not name:
        raise TransformError("transform must define a non-empty string 'name'")
    if name in REGISTRY:
        raise TransformError(f"transform already registered: {name}")
    REGISTRY[name] = transform


def get_transform(name: str) -> BaseTransform:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise TransformError(f"unknown transform: {name}") from exc


def list_transforms() -> list[str]:
    return sorted(REGISTRY.keys())


def apply_transform(compiled: CompiledTransform, provider: StateDictProvider) -> TransformResult:
    return compiled.transform.apply(compiled.spec, provider)


class DestinationPolicy(Enum):
    ANY = "any"
    MUST_EXIST = "must_exist"
    MUST_NOT_EXIST = "must_not_exist"


ItemT = TypeVar("ItemT")


class IteratingTransform(TypedTransform[SpecT], ABC, Generic[SpecT, ItemT]):
    error_type: type[TransformError] = TransformError
    spec_type: type[SpecT]
    progress_desc: str | None = None
    progress_unit: str = "tensor"

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        items = self.resolve_items(typed, provider)
        self.validate_resolved_items(typed, items, provider)

        for item in self.iter_items(items):
            self.apply_item(typed, item, provider)

        return TransformResult(name=self.name, count=len(items))

    def iter_items(self, items: list[ItemT]) -> Iterable[ItemT]:
        callback = _progress_callback.get()
        if callback is None and self.progress_desc is None:
            return items
        if callback is None:
            return tqdm(items, desc=self.progress_desc, unit=self.progress_unit)
        total = len(items)
        desc = self.progress_desc or self.name
        callback(
            IterationProgress(
                desc=desc,
                unit=self.progress_unit,
                completed=0,
                total=total,
            )
        )

        def _iter() -> Iterator[ItemT]:
            completed = 0
            for item in items:
                yield item
                completed += 1
                callback(
                    IterationProgress(
                        desc=desc,
                        unit=self.progress_unit,
                        completed=completed,
                        total=total,
                    )
                )

        return _iter()

    @abstractmethod
    def resolve_items(
        self,
        spec: SpecT,
        provider: StateDictProvider,
    ) -> list[ItemT]: ...

    def validate_resolved_items(
        self,
        spec: SpecT,
        items: list[ItemT],
        provider: StateDictProvider,
    ) -> None:
        del spec, items, provider

    @abstractmethod
    def apply_item(
        self,
        spec: SpecT,
        item: ItemT,
        provider: StateDictProvider,
    ) -> None: ...


@dataclass(frozen=True)
class UnarySpec:
    target_ref: TensorRef

    def collect_models(self) -> set[str]:
        return {must_model(self.target_ref)}


UnarySpecT = TypeVar("UnarySpecT", bound=UnarySpec)


def _resolve_unary_target_names(
    *,
    target_ref: TensorRef,
    provider: StateDictProvider,
    op_name: str,
    error_type: type[TransformError],
) -> list[str]:
    return resolve_target_names_generic(
        target_ref=target_ref,
        provider=provider,
        op_name=op_name,
        match_names=match_expr_names,
        error_type=error_type,
    )


class UnaryTransform(IteratingTransform[UnarySpecT, str], ABC, Generic[UnarySpecT]):
    error_type: type[TransformError] = TransformError
    spec_type: type[UnarySpecT]

    allowed_keys: set[str] = {"target"}
    required_keys: set[str] = {"target"}
    target_key: str = "target"
    slice_policy: Literal["allow", "forbid"] = "forbid"

    def completion_reference_keys(self) -> list[str]:
        return [self.target_key]

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key=None,
            default_mode="default",
            common_required=set(self.required_keys),
            common_allowed=set(self.allowed_keys),
        )

    def compile(self, payload: dict, default_model: str | None) -> UnarySpecT:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_schema(
            payload,
            op_name=self.name,
            schema=self.payload_schema(),
            error_type=self.error_type,
        )

        raw_target = self.require_target_expr(payload)
        target_ref = parse_model_expr(raw_target, default_model=default_model)

        self.validate_target_ref(target_ref)

        assert target_ref.model is not None
        return self.build_spec(target_ref=target_ref, payload=payload)

    def _infer_output_model(self, spec: object) -> str:
        typed = self.require_spec(spec)
        model = typed.target_ref.model
        if model is None:
            raise self.error_type(f"{self.name} output model missing")
        return model

    def require_target_expr(self, payload: dict) -> str | list[str]:
        return require_expr(payload, op_name=self.name, key=self.target_key)

    def build_spec(self, target_ref: TensorRef, payload: dict) -> UnarySpecT:
        return self.spec_type(target_ref=target_ref)

    def validate_target_ref(self, target_ref: TensorRef) -> None:
        if target_ref.slice_spec is None:
            return

        if self.slice_policy == "forbid":
            raise self.error_type(f"{self.name} target must not be sliced")

        parse_slice(target_ref.slice_spec)

    def resolve_items(self, spec: UnarySpecT, provider: StateDictProvider) -> list[str]:
        return _resolve_unary_target_names(
            target_ref=spec.target_ref,
            provider=provider,
            op_name=self.name,
            error_type=self.error_type,
        )

    def resolve_targets(self, spec: UnarySpecT, provider: StateDictProvider) -> list[str]:
        return self.resolve_items(spec, provider)

    @abstractmethod
    def apply_to_target(self, spec: UnarySpecT, name: str, provider: StateDictProvider) -> None: ...

    def apply_item(self, spec: UnarySpecT, item: str, provider: StateDictProvider) -> None:
        self.apply_to_target(spec, item, provider)


@dataclass(frozen=True)
class BinaryMappingSpec:
    from_ref: TensorRef
    to_ref: TensorRef

    def collect_models(self) -> set[str]:
        return {must_model(self.from_ref), must_model(self.to_ref)}


BinarySpecT = TypeVar("BinarySpecT", bound=BinaryMappingSpec)


class BinaryMappingTransform(
    IteratingTransform[BinarySpecT, tuple[str, str]], ABC, Generic[BinarySpecT]
):
    error_type: type[TransformError] = TransformError
    spec_type: type[BinarySpecT]
    destination_policy: DestinationPolicy = DestinationPolicy.ANY

    allowed_keys = {"from", "to"}
    required_keys = {"from", "to"}

    def completion_reference_keys(self) -> list[str]:
        return ["from", "to"]

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key=None,
            default_mode="default",
            common_required=set(self.required_keys),
            common_allowed=set(self.allowed_keys),
        )

    def compile(self, payload: dict, default_model: str | None) -> BinarySpecT:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_schema(
            payload,
            op_name=self.name,
            schema=self.payload_schema(),
            error_type=self.error_type,
        )

        from_ref, to_ref = self.parse_refs(payload, default_model)
        self.validate_refs(from_ref, to_ref)

        assert from_ref.model is not None
        assert to_ref.model is not None
        return self.build_spec(from_ref=from_ref, to_ref=to_ref)

    def _infer_output_model(self, spec: object) -> str:
        typed = self.require_spec(spec)
        model = typed.to_ref.model
        if model is None:
            raise self.error_type(f"{self.name} output model missing")
        return model

    def parse_refs(
        self,
        payload: dict,
        default_model: str | None,
    ) -> tuple[TensorRef, TensorRef]:
        raw_from = require_expr(payload, op_name=self.name, key="from")
        raw_to = require_expr(payload, op_name=self.name, key="to")

        from_ref = parse_model_expr(raw_from, default_model=default_model)
        to_ref = parse_model_expr(raw_to, default_model=default_model)
        return from_ref, to_ref

    def build_spec(self, from_ref: TensorRef, to_ref: TensorRef) -> BinarySpecT:
        return self.spec_type(from_ref=from_ref, to_ref=to_ref)

    def resolve_items(
        self,
        spec: BinarySpecT,
        provider: StateDictProvider,
    ) -> list[tuple[str, str]]:
        mappings = resolve_name_mappings(
            from_ref=spec.from_ref,
            to_ref=spec.to_ref,
            provider=provider,
            op_name=self.name,
        )
        return [(item.src_name, item.dst_name) for item in mappings]

    @abstractmethod
    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None: ...

    def validate_resolved_items(
        self,
        spec: BinarySpecT,
        mappings: list[tuple[str, str]],
        provider: StateDictProvider,
    ) -> None:
        dst_model = must_model(spec.to_ref)
        dst_sd = provider.get_state_dict(dst_model)
        dst_names = [dst_name for _, dst_name in mappings]

        if self.destination_policy is DestinationPolicy.MUST_EXIST:
            missing = [name for name in dst_names if name not in dst_sd]
            if missing:
                raise self.error_type(f"{self.name} destination missing: {dst_model}::{missing[0]}")
            return

        if self.destination_policy is DestinationPolicy.MUST_NOT_EXIST:
            existing = [name for name in dst_names if name in dst_sd]
            if existing:
                raise self.error_type(
                    f"{self.name} destination already exists: {dst_model}::{existing[0]}"
                )

    def apply_mapping(
        self, spec: BinarySpecT, src_name: str, dst_name: str, provider: StateDictProvider
    ) -> None:
        del spec, src_name, dst_name, provider
        raise NotImplementedError

    def apply_item(
        self, spec: BinarySpecT, item: tuple[str, str], provider: StateDictProvider
    ) -> None:
        src_name, dst_name = item
        self.apply_mapping(spec, src_name, dst_name, provider)


@dataclass(frozen=True)
class TernaryMappingSpec:
    from_a_ref: TensorRef
    from_b_ref: TensorRef
    to_ref: TensorRef

    def collect_models(self) -> set[str]:
        return {
            must_model(self.from_a_ref),
            must_model(self.from_b_ref),
            must_model(self.to_ref),
        }


TernarySpecT = TypeVar("TernarySpecT", bound=TernaryMappingSpec)


class TernaryMappingTransform(
    IteratingTransform[TernarySpecT, tuple[str, str, str]], ABC, Generic[TernarySpecT]
):
    error_type: type[TransformError] = TransformError
    spec_type: type[TernarySpecT]
    destination_policy: DestinationPolicy = DestinationPolicy.ANY

    allowed_keys = {"from_a", "from_b", "to"}
    required_keys = {"from_a", "from_b", "to"}

    def completion_reference_keys(self) -> list[str]:
        return ["from_a", "from_b", "to"]

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key=None,
            default_mode="default",
            common_required=set(self.required_keys),
            common_allowed=set(self.allowed_keys),
        )

    def compile(self, payload: dict, default_model: str | None) -> TernarySpecT:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_schema(
            payload,
            op_name=self.name,
            schema=self.payload_schema(),
            error_type=self.error_type,
        )

        from_a_ref, from_b_ref, to_ref = self.parse_refs(payload, default_model)
        self.validate_refs(from_a_ref, from_b_ref, to_ref)

        assert from_a_ref.model is not None
        assert from_b_ref.model is not None
        assert to_ref.model is not None
        return self.build_spec(from_a_ref=from_a_ref, from_b_ref=from_b_ref, to_ref=to_ref)

    def _infer_output_model(self, spec: object) -> str:
        typed = self.require_spec(spec)
        model = typed.to_ref.model
        if model is None:
            raise self.error_type(f"{self.name} output model missing")
        return model

    def parse_refs(
        self,
        payload: dict,
        default_model: str | None,
    ) -> tuple[TensorRef, TensorRef, TensorRef]:
        raw_from_a = require_expr(payload, op_name=self.name, key="from_a")
        raw_from_b = require_expr(payload, op_name=self.name, key="from_b")
        raw_to = require_expr(payload, op_name=self.name, key="to")

        from_a_ref = parse_model_expr(raw_from_a, default_model=default_model)
        from_b_ref = parse_model_expr(raw_from_b, default_model=default_model)
        to_ref = parse_model_expr(raw_to, default_model=default_model)
        return from_a_ref, from_b_ref, to_ref

    def build_spec(
        self, from_a_ref: TensorRef, from_b_ref: TensorRef, to_ref: TensorRef
    ) -> TernarySpecT:
        return self.spec_type(from_a_ref=from_a_ref, from_b_ref=from_b_ref, to_ref=to_ref)

    @abstractmethod
    def validate_refs(
        self, from_a_ref: TensorRef, from_b_ref: TensorRef, to_ref: TensorRef
    ) -> None: ...

    def resolve_items(
        self,
        spec: TernarySpecT,
        provider: StateDictProvider,
    ) -> list[tuple[str, str, str]]:
        from_a_ref = spec.from_a_ref
        from_b_ref = spec.from_b_ref
        to_ref = spec.to_ref

        src_a_model = must_model(from_a_ref)
        src_b_model = must_model(from_b_ref)
        dst_model = must_model(to_ref)
        src_a_sd = provider.get_state_dict(src_a_model)
        src_b_sd = provider.get_state_dict(src_b_model)

        a_expr = from_a_ref.expr
        b_expr = from_b_ref.expr
        to_expr = to_ref.expr
        if isinstance(a_expr, str) and isinstance(b_expr, str) and isinstance(to_expr, str):
            src_a_names = match_expr_names(
                expr=a_expr,
                names=src_a_sd.keys(),
                op_name=self.name,
                role="source_a",
            )
            if not src_a_names:
                raise self.error_type(
                    f"{self.name} source_a matched zero tensors: "
                    f"{format_tensor_ref(from_a_ref)}; available tensors: {sorted(src_a_sd.keys())}"
                )

            resolved_regex: list[tuple[str, str, str]] = []
            dst_names_seen_regex: set[str] = set()
            for src_a_name in src_a_names:
                try:
                    src_b_name = re.sub(a_expr, b_expr, src_a_name)
                    dst_name = re.sub(a_expr, to_expr, src_a_name)
                except re.error as exc:
                    raise self.error_type(
                        f"{self.name} invalid regex rewrite from {a_expr!r}: {exc}"
                    ) from exc

                if src_b_name not in src_b_sd:
                    raise self.error_type(
                        f"{self.name} source_b missing: {src_b_model}::{src_b_name}"
                    )
                if dst_name in dst_names_seen_regex:
                    raise self.error_type(
                        f"{self.name} destination collision: {dst_model}::{dst_name}"
                    )
                dst_names_seen_regex.add(dst_name)

                resolved_regex.append((src_a_name, src_b_name, dst_name))
            return resolved_regex

        if isinstance(a_expr, list) and isinstance(b_expr, list) and isinstance(to_expr, list):
            resolved_structured: list[tuple[str, str, str]] = []
            dst_names_seen_structured: set[str] = set()
            matched_any = False
            for src_a_name in sorted(src_a_sd.keys()):
                match = _match_structured_expr(
                    expr=a_expr,
                    name=src_a_name,
                    op_name=self.name,
                    role="source_a",
                )
                if match is None:
                    continue
                matched_any = True

                src_b_name = _rewrite_structured_expr(
                    expr=b_expr,
                    match=match,
                    op_name=self.name,
                    role="source_b",
                )
                dst_name = _rewrite_structured_expr(
                    expr=to_expr,
                    match=match,
                    op_name=self.name,
                    role="destination",
                )

                if src_b_name not in src_b_sd:
                    raise self.error_type(
                        f"{self.name} source_b missing: {src_b_model}::{src_b_name}"
                    )
                if dst_name in dst_names_seen_structured:
                    raise self.error_type(
                        f"{self.name} destination collision: {dst_model}::{dst_name}"
                    )
                dst_names_seen_structured.add(dst_name)

                resolved_structured.append((src_a_name, src_b_name, dst_name))

            if not matched_any:
                raise self.error_type(
                    f"{self.name} source_a matched zero tensors: "
                    f"{format_tensor_ref(from_a_ref)}; available tensors: {sorted(src_a_sd.keys())}"
                )
            return resolved_structured

        raise self.error_type(
            f"{self.name} requires from_a/from_b/to expressions of the same kind: "
            "either all strings (regex mode) or all lists (structured mode)"
        )

    def validate_resolved_items(
        self,
        spec: TernarySpecT,
        mappings: list[tuple[str, str, str]],
        provider: StateDictProvider,
    ) -> None:
        dst_model = must_model(spec.to_ref)
        dst_sd = provider.get_state_dict(dst_model)

        if self.destination_policy is DestinationPolicy.ANY:
            return

        for _, _, dst_name in mappings:
            exists = dst_name in dst_sd

            if self.destination_policy is DestinationPolicy.MUST_EXIST and not exists:
                raise self.error_type(f"{self.name} destination missing: {dst_model}::{dst_name}")
            if self.destination_policy is DestinationPolicy.MUST_NOT_EXIST and exists:
                raise self.error_type(
                    f"{self.name} destination already exists: {dst_model}::{dst_name}"
                )

    @abstractmethod
    def apply_mapping(
        self,
        spec: TernarySpecT,
        src_a_name: str,
        src_b_name: str,
        dst_name: str,
        provider: StateDictProvider,
    ) -> None: ...

    def apply_item(
        self,
        spec: TernarySpecT,
        item: tuple[str, str, str],
        provider: StateDictProvider,
    ) -> None:
        src_a_name, src_b_name, dst_name = item
        self.apply_mapping(spec, src_a_name, src_b_name, dst_name, provider)


__all__ = [
    "BaseTransform",
    "BinaryMappingSpec",
    "BinaryMappingTransform",
    "CompiledTransform",
    "DestinationPolicy",
    "IteratingTransform",
    "TernaryMappingSpec",
    "TernaryMappingTransform",
    "TransformControl",
    "TransformResult",
    "TypedTransform",
    "UnarySpec",
    "UnaryTransform",
    "REGISTRY",
    "apply_transform",
    "get_transform",
    "list_transforms",
    "register_transform",
]
