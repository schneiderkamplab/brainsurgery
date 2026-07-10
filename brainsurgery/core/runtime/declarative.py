import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast

from ..specs import (
    StateDictProvider,
    TensorRef,
    TransformError,
    ensure_mapping_payload,
    parse_slice,
    validate_payload_schema,
)
from .transform import (
    BinaryMappingSpec,
    BinaryMappingTransform,
    DestinationPolicy,
    TernaryMappingSpec,
    TernaryMappingTransform,
    UnarySpec,
    UnaryTransform,
)

UnarySpecT = TypeVar("UnarySpecT", bound=UnarySpec)
BinarySpecT = TypeVar("BinarySpecT", bound=BinaryMappingSpec)
TernarySpecT = TypeVar("TernarySpecT", bound=TernaryMappingSpec)
_YAML_INLINE_EXAMPLE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*\{\s*(.*)\s*\}\s*$")


@dataclass(frozen=True)
class Docs:
    summary: str
    notes: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnaryRefs:
    target_slice: bool = False


@dataclass(frozen=True)
class BinaryRefs:
    from_slice: bool = False
    to_slice: bool = False


@dataclass(frozen=True)
class TernaryRefs:
    from_a_slice: bool = False
    from_b_slice: bool = False
    to_slice: bool = False


def _lines(text: Docs, *rules: str) -> str:
    lines = [text.summary, "", *rules]
    if text.notes:
        lines.extend(["", *text.notes])
    if text.examples:
        rendered_examples: list[str] = []
        for example in text.examples:
            rendered_examples.append(f"  YAML: {example}")
            match = _YAML_INLINE_EXAMPLE_RE.fullmatch(example.strip())
            if match:
                command = match.group(1)
                payload = match.group(2).strip()
                if payload:
                    rendered_examples.append(f"  OLY:  {command}: {payload}")
        lines.extend(["", "Examples:", *rendered_examples])
    return "\n".join(lines)


def _slice_rule(label: str, allowed: bool) -> str:
    return f"{label} may include slicing." if allowed else f"{label} must not be sliced."


def _destination_rule(policy: DestinationPolicy) -> str:
    if policy is DestinationPolicy.MUST_EXIST:
        return "Destination tensors must already exist."
    if policy is DestinationPolicy.MUST_NOT_EXIST:
        return "Destination tensors must not already exist."
    return "Destination tensors may be created or overwritten."


def _validate_slice(
    ref: TensorRef,
    *,
    allowed: bool,
    op_name: str,
    label: str,
    error_type: type[TransformError],
) -> None:
    if ref.slice_spec is None:
        return
    if not allowed:
        raise error_type(f"{op_name} {label} must not be sliced")
    parse_slice(ref.slice_spec)


class DeclarativeUnaryTransform(UnaryTransform[UnarySpecT]):
    spec_type: type[UnarySpecT] = cast(type[UnarySpecT], UnarySpec)
    allowed_keys = {"target"}
    required_keys = {"target"}
    docs: Docs
    refs = UnaryRefs()
    help_text: str
    spec_builder: Callable[[TensorRef, dict], UnarySpecT] | None = None
    progress_desc: str | None = None
    apply_fn: Callable[[UnarySpecT, str, StateDictProvider], None]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "docs", None) is None:
            return
        if cls.progress_desc is None:
            cls.progress_desc = f"Applying {cls.name} transforms"
        cls.slice_policy = "allow" if cls.refs.target_slice else "forbid"
        cls.help_text = _lines(cls.docs, _slice_rule("Target tensors", cls.refs.target_slice))

    def build_spec(self, target_ref: TensorRef, payload: dict) -> UnarySpecT:
        if self.spec_builder is not None:
            return self.spec_builder(target_ref, payload)
        return self.spec_type(target_ref=target_ref)

    def apply_to_target(self, spec: UnarySpecT, name: str, provider: StateDictProvider) -> None:
        self.apply_fn(spec, name, provider)


class DeclarativeBinaryTransform(BinaryMappingTransform[BinarySpecT]):
    spec_type: type[BinarySpecT] = cast(type[BinarySpecT], BinaryMappingSpec)
    allowed_keys = {"from", "to"}
    required_keys = {"from", "to"}
    docs: Docs
    refs = BinaryRefs()
    help_text: str
    spec_builder: Callable[[TensorRef, TensorRef, dict], BinarySpecT] | None = None
    progress_desc: str | None = None
    apply_fn: Callable[[BinarySpecT, str, str, StateDictProvider], None]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "docs", None) is None:
            return
        if cls.progress_desc is None:
            cls.progress_desc = f"Applying {cls.name} transforms"
        cls.help_text = _lines(
            cls.docs,
            _slice_rule("Source references", cls.refs.from_slice),
            _slice_rule("Destination references", cls.refs.to_slice),
            _destination_rule(cls.destination_policy),
        )

    def compile(self, payload: dict, default_model: str | None) -> BinarySpecT:
        if self.spec_builder is None:
            return super().compile(payload, default_model)
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
        return self.spec_builder(from_ref, to_ref, payload)

    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        _validate_slice(
            from_ref,
            allowed=self.refs.from_slice,
            op_name=self.name,
            label="source",
            error_type=self.error_type,
        )
        _validate_slice(
            to_ref,
            allowed=self.refs.to_slice,
            op_name=self.name,
            label="destination",
            error_type=self.error_type,
        )

    def apply_item(
        self, spec: BinarySpecT, item: tuple[str, str], provider: StateDictProvider
    ) -> None:
        src_name, dst_name = item
        self.apply_fn(spec, src_name, dst_name, provider)

    def apply_mapping(
        self, spec: BinarySpecT, src_name: str, dst_name: str, provider: StateDictProvider
    ) -> None:
        del spec, src_name, dst_name, provider
        raise NotImplementedError


class DeclarativeTernaryTransform(TernaryMappingTransform[TernarySpecT]):
    spec_type: type[TernarySpecT] = cast(type[TernarySpecT], TernaryMappingSpec)
    allowed_keys = {"from_a", "from_b", "to"}
    required_keys = {"from_a", "from_b", "to"}
    docs: Docs
    refs = TernaryRefs()
    help_text: str
    progress_desc: str | None = None
    apply_fn: Callable[[TernarySpecT, str, str, str, StateDictProvider], None]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "docs", None) is None:
            return
        if cls.progress_desc is None:
            cls.progress_desc = f"Applying {cls.name} transforms"
        cls.help_text = _lines(
            cls.docs,
            "References may be regex or structured mappings.",
            _slice_rule("'from_a' references", cls.refs.from_a_slice),
            _slice_rule("'from_b' references", cls.refs.from_b_slice),
            _slice_rule("'to' references", cls.refs.to_slice),
            _destination_rule(cls.destination_policy),
        )

    def validate_refs(
        self, from_a_ref: TensorRef, from_b_ref: TensorRef, to_ref: TensorRef
    ) -> None:
        _validate_slice(
            from_a_ref,
            allowed=self.refs.from_a_slice,
            op_name=self.name,
            label="from_a",
            error_type=self.error_type,
        )
        _validate_slice(
            from_b_ref,
            allowed=self.refs.from_b_slice,
            op_name=self.name,
            label="from_b",
            error_type=self.error_type,
        )
        _validate_slice(
            to_ref,
            allowed=self.refs.to_slice,
            op_name=self.name,
            label="destination",
            error_type=self.error_type,
        )

    def apply_item(
        self,
        spec: TernarySpecT,
        item: tuple[str, str, str],
        provider: StateDictProvider,
    ) -> None:
        src_a_name, src_b_name, dst_name = item
        self.apply_fn(spec, src_a_name, src_b_name, dst_name, provider)

    def apply_mapping(
        self,
        spec: TernarySpecT,
        src_a_name: str,
        src_b_name: str,
        dst_name: str,
        provider: StateDictProvider,
    ) -> None:
        del spec, src_a_name, src_b_name, dst_name, provider
        raise NotImplementedError


__all__ = [
    "Docs",
    "UnaryRefs",
    "BinaryRefs",
    "TernaryRefs",
    "DeclarativeUnaryTransform",
    "DeclarativeBinaryTransform",
    "DeclarativeTernaryTransform",
]
