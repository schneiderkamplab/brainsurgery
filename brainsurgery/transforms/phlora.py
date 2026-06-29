from dataclasses import dataclass

from ..algorithms import (
    PhloraSvdCache,
    compute_phlora_factors,
    reconstruct_phlora_rank,
    require_positive_rank,
)
from ..core import (
    IteratingTransform,
    ResolvedMapping,
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformPayloadSchema,
    UnarySpec,
    UnaryTransform,
    ensure_mapping_payload,
    must_model,
    parse_model_expr,
    register_transform,
    require_expr,
    require_numeric,
    resolve_name_mappings,
    validate_payload_schema,
)
from ..engine import emit_verbose_event, emit_verbose_unary_activity


class PhloraTransformError(TransformError):
    pass


@dataclass(frozen=True)
class PhloraSpec:
    source_ref: TensorRef
    factor_b_ref: TensorRef
    factor_a_ref: TensorRef
    rank: int
    delete_source: bool
    require_missing_outputs: bool

    def collect_models(self) -> set[str]:
        return {
            must_model(self.source_ref),
            must_model(self.factor_a_ref),
            must_model(self.factor_b_ref),
        }


@dataclass(frozen=True)
class ResolvedPhloraMapping:
    source_model: str
    source_name: str
    factor_a_model: str
    factor_a_name: str
    factor_b_model: str
    factor_b_name: str


class PhloraTransform(IteratingTransform[PhloraSpec, ResolvedPhloraMapping]):
    name = "phlora"
    error_type = PhloraTransformError
    spec_type = PhloraSpec
    allowed_keys = {
        "target",
        "target_a",
        "target_b",
        "rank",
        "delete_original",
        "require_missing_dest",
    }
    required_keys = {"target", "target_a", "target_b", "rank"}
    progress_desc = "Applying phlora transforms"
    help_text = (
        "Splits each target tensor into PHLoRA low-rank factors B and A.\n"
        "\n"
        "Inputs:\n"
        "  - target: source tensor expression\n"
        "  - target_a: destination expression for A = sqrt(S) * Vh\n"
        "  - target_b: destination expression for B = U * sqrt(S)\n"
        "  - rank: low-rank dimension r\n"
        "\n"
        "target/target_a/target_b support regex or structured expressions like other transforms.\n"
        "Mappings are resolved from target -> target_a and target -> target_b; both must cover "
        "the same source set.\n"
        "\n"
        "Options:\n"
        "  - delete_original (default: true)\n"
        "  - require_missing_dest (default: true)\n"
        "\n"
        "Example:\n"
        "  phlora: { target: '(.*)\\\\.weight', rank: 16, target_a: '\\\\1.a', target_b: '\\\\1.b' }"
    )

    def __init__(self) -> None:
        super().__init__()
        self._svd_cache = PhloraSvdCache()

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key=None,
            default_mode="default",
            common_required=set(self.required_keys),
            common_allowed=set(self.allowed_keys),
        )

    def completion_reference_keys(self) -> list[str]:
        return ["target", "target_a", "target_b"]

    def compile(self, payload: dict, default_model: str | None) -> PhloraSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_schema(
            payload,
            op_name=self.name,
            schema=self.payload_schema(),
            error_type=self.error_type,
        )

        source_ref = parse_model_expr(
            require_expr(payload, op_name=self.name, key="target"),
            default_model=default_model,
        )
        source_model_default = source_ref.model if source_ref.model is not None else default_model
        factor_a_ref = parse_model_expr(
            require_expr(payload, op_name=self.name, key="target_a"),
            default_model=source_model_default,
        )
        factor_b_ref = parse_model_expr(
            require_expr(payload, op_name=self.name, key="target_b"),
            default_model=source_model_default,
        )

        self.validate_refs(source_ref, factor_b_ref, factor_a_ref)

        rank = require_positive_rank(
            require_numeric(payload, op_name=self.name, key="rank"),
            error_type=PhloraTransformError,
            op_name=self.name,
            key="rank",
        )
        delete_source = _require_boolean(
            payload,
            op_name=self.name,
            key="delete_original",
            default=True,
        )
        require_missing_outputs = _require_boolean(
            payload,
            op_name=self.name,
            key="require_missing_dest",
            default=True,
        )
        return PhloraSpec(
            source_ref=source_ref,
            factor_b_ref=factor_b_ref,
            factor_a_ref=factor_a_ref,
            rank=rank,
            delete_source=delete_source,
            require_missing_outputs=require_missing_outputs,
        )

    def validate_refs(
        self, from_a_ref: TensorRef, from_b_ref: TensorRef, to_ref: TensorRef
    ) -> None:
        if from_a_ref.slice_spec is not None:
            raise PhloraTransformError("phlora target must not be sliced")
        if from_b_ref.slice_spec is not None:
            raise PhloraTransformError("phlora target_b must not be sliced")
        if to_ref.slice_spec is not None:
            raise PhloraTransformError("phlora target_a must not be sliced")

    def _infer_output_model(self, spec: object) -> str:
        typed = self.require_spec(spec)
        model = typed.source_ref.model
        if model is None:
            raise PhloraTransformError("phlora output model missing")
        return model

    def resolve_items(
        self,
        spec: PhloraSpec,
        provider: StateDictProvider,
    ) -> list[ResolvedPhloraMapping]:
        factor_a_mappings = resolve_name_mappings(
            from_ref=spec.source_ref,
            to_ref=spec.factor_a_ref,
            provider=provider,
            op_name=self.name,
        )
        factor_b_mappings = resolve_name_mappings(
            from_ref=spec.source_ref,
            to_ref=spec.factor_b_ref,
            provider=provider,
            op_name=self.name,
        )
        pairs = _pair_mappings(factor_a_mappings, factor_b_mappings, op_name=self.name)

        resolved: list[ResolvedPhloraMapping] = []
        for source_item, factor_a_item, factor_b_item in pairs:
            resolved.append(
                ResolvedPhloraMapping(
                    source_model=source_item.src_model,
                    source_name=source_item.src_name,
                    factor_a_model=factor_a_item.dst_model,
                    factor_a_name=factor_a_item.dst_name,
                    factor_b_model=factor_b_item.dst_model,
                    factor_b_name=factor_b_item.dst_name,
                )
            )
        return resolved

    def apply_item(
        self,
        spec: PhloraSpec,
        item: ResolvedPhloraMapping,
        provider: StateDictProvider,
    ) -> None:
        src_sd = provider.get_state_dict(item.source_model)
        factor_a_sd = provider.get_state_dict(item.factor_a_model)
        factor_b_sd = provider.get_state_dict(item.factor_b_model)

        if item.source_name not in src_sd:
            raise PhloraTransformError(
                f"phlora source disappeared during apply: {item.source_model}::{item.source_name}"
            )

        source = src_sd[item.source_name]

        if spec.require_missing_outputs and item.factor_a_name in factor_a_sd:
            raise PhloraTransformError(
                f"phlora destination already exists: {item.factor_a_model}::{item.factor_a_name}"
            )
        if spec.require_missing_outputs and item.factor_b_name in factor_b_sd:
            raise PhloraTransformError(
                f"phlora destination already exists: {item.factor_b_model}::{item.factor_b_name}"
            )
        if item.factor_a_model == item.factor_b_model and item.factor_a_name == item.factor_b_name:
            raise PhloraTransformError(
                f"phlora destination collision for source {item.source_model}::{item.source_name}: "
                f"{item.factor_a_model}::{item.factor_a_name}"
            )

        lora_a, lora_b = compute_phlora_factors(
            source,
            spec.rank,
            cache=self._svd_cache,
            cache_key=f"{item.source_model}::{item.source_name}",
            error_type=PhloraTransformError,
            op_name="phlora",
            tensor_name=f"{item.source_model}::{item.source_name}",
        )

        factor_a_sd[item.factor_a_name] = lora_a.to(dtype=source.dtype, device=source.device)
        factor_b_sd[item.factor_b_name] = lora_b.to(dtype=source.dtype, device=source.device)
        emit_verbose_event(
            self.name,
            f"{item.source_name} -> {item.factor_b_name}, {item.factor_a_name}",
        )

        if spec.delete_source:
            del src_sd[item.source_name]


def _pair_mappings(
    factor_a_mappings: list[ResolvedMapping],
    factor_b_mappings: list[ResolvedMapping],
    *,
    op_name: str,
) -> list[tuple[ResolvedMapping, ResolvedMapping, ResolvedMapping]]:
    factor_a_by_source = {(m.src_model, m.src_name): m for m in factor_a_mappings}
    factor_b_by_source = {(m.src_model, m.src_name): m for m in factor_b_mappings}
    if set(factor_a_by_source.keys()) != set(factor_b_by_source.keys()):
        raise PhloraTransformError(f"{op_name} target_a/target_b mappings do not match source set")
    pairs: list[tuple[ResolvedMapping, ResolvedMapping, ResolvedMapping]] = []
    for key in sorted(factor_a_by_source.keys()):
        source_item = factor_a_by_source[key]
        pairs.append((source_item, factor_a_by_source[key], factor_b_by_source[key]))
    return pairs


def _require_boolean(payload: dict, *, op_name: str, key: str, default: bool) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, bool):
        raise PhloraTransformError(f"{op_name}.{key} must be a boolean when provided")
    return value


register_transform(PhloraTransform())


class PhloraInPlaceTransformError(TransformError):
    pass


@dataclass(frozen=True)
class PhloraInPlaceSpec(UnarySpec):
    rank: int


class PhloraInPlaceTransform(UnaryTransform[PhloraInPlaceSpec]):
    name = "phlora_"
    error_type = PhloraInPlaceTransformError
    spec_type = PhloraInPlaceSpec
    allowed_keys = {"target", "rank"}
    required_keys = {"target", "rank"}
    progress_desc = "Applying phlora_ transforms"
    help_text = (
        "Applies in-place PHLoRA low-rank reconstruction.\n"
        "\n"
        "For each matched 2D tensor W:\n"
        "  u, s, vh = svd(W)\n"
        "  W <- (u[:, :r] * s[:r]) @ vh[:r, :]\n"
        "\n"
        "Examples:\n"
        "  phlora_: { target: '.*weight', rank: 64 }\n"
        "  phlora_: { target: h.0.attn.c_proj.weight, rank: 16 }"
    )

    def __init__(self) -> None:
        super().__init__()
        self._svd_cache = PhloraSvdCache()

    def build_spec(self, target_ref: TensorRef, payload: dict) -> PhloraInPlaceSpec:
        rank = require_positive_rank(
            require_numeric(payload, op_name=self.name, key="rank"),
            error_type=PhloraInPlaceTransformError,
            op_name=self.name,
            key="rank",
        )
        return PhloraInPlaceSpec(target_ref=target_ref, rank=rank)

    def apply_to_target(
        self,
        spec: PhloraInPlaceSpec,
        name: str,
        provider: StateDictProvider,
    ) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)

        source = sd[name]
        new_tensor = reconstruct_phlora_rank(
            source,
            spec.rank,
            cache=self._svd_cache,
            cache_key=f"{model}::{name}",
            error_type=PhloraInPlaceTransformError,
            op_name="phlora_",
            tensor_name=f"{model}::{name}",
        )
        sd[name] = new_tensor.to(dtype=source.dtype, device=source.device)
        emit_verbose_unary_activity(self.name, name)


register_transform(PhloraInPlaceTransform())
