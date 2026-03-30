from dataclasses import dataclass
from typing import Any, Literal, cast

import torch

from ..core import (
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformPayloadSchema,
    TransformResult,
    TypedTransform,
    ensure_mapping_payload,
    match_expr_names,
    must_model,
    parse_model_expr,
    register_transform,
    require_nonempty_string,
    state_dict_for_ref,
    unary_view_for_ref_name,
    validate_payload_schema,
)
from ..engine import emit_verbose_event, get_model_runtime_metadata
from .infer_runtime import _load_infer_runtime_model


class InferTransformError(TransformError):
    pass


InferRuntime = Literal["auto", "synapse", "codegen", "hf"]
ResolvedInferRuntime = Literal["synapse", "codegen", "hf"]


@dataclass(frozen=True)
class InferSpec:
    runtime: InferRuntime
    model_alias: str
    tmp_alias: str | None
    input_ids_ref: TensorRef
    attention_mask_ref: TensorRef | None
    attn_mask_ref: TensorRef | None
    output_ref: TensorRef

    def collect_models(self) -> set[str]:
        models = {
            self.model_alias,
            must_model(self.input_ids_ref),
            must_model(self.output_ref),
        }
        if self.tmp_alias is not None:
            models.add(self.tmp_alias)
        if self.attention_mask_ref is not None:
            models.add(must_model(self.attention_mask_ref))
        if self.attn_mask_ref is not None:
            models.add(must_model(self.attn_mask_ref))
        return models


class InferTransform(TypedTransform[InferSpec]):
    name = "infer"
    error_type = InferTransformError
    spec_type = InferSpec
    help_text = (
        "Runs model inference from tensor inputs already loaded in aliases.\n"
        "\n"
        "Infer runtime/program from metadata attached during load.\n"
        "Optionally override runtime (auto|synapse|codegen|hf).\n"
        "Optionally mirror runtime intermediate tensors into tmp_alias.\n"
        "Output logits are written to 'output' (default: model::logits).\n"
        "\n"
        "Examples:\n"
        "  infer: { model: gpt2, input_ids: gpt2::input_ids }\n"
        "  infer: { model: gpt2, runtime: codegen, input_ids: work::ids, attn_mask: work::mask, output: work::logits }\n"
        "  infer: { model: gpt2, runtime: hf, input_ids: work::ids, attention_mask: work::mask }"
    )

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key="runtime",
            default_mode="auto",
            common_required={"model", "input_ids"},
            common_allowed={
                "runtime",
                "model",
                "tmp_alias",
                "input_ids",
                "attention_mask",
                "attn_mask",
                "output",
            },
            mode_allowed_extra={
                "auto": set(),
                "synapse": set(),
                "codegen": set(),
                "hf": set(),
            },
        )

    def completion_reference_keys(self) -> list[str]:
        return ["input_ids", "attention_mask", "attn_mask", "output"]

    def completion_value_candidates(
        self,
        value_key: str | None,
        prefix_text: str,
        model_aliases: list[str],
    ) -> list[str] | None:
        if value_key == "model":
            return [alias for alias in model_aliases if alias.startswith(prefix_text)]
        if value_key == "runtime":
            return [
                item
                for item in ("auto", "synapse", "codegen", "hf")
                if item.startswith(prefix_text)
            ]
        return None

    def compile(self, payload: Any, default_model: str | None) -> InferSpec:
        del default_model
        payload = ensure_mapping_payload(payload, self.name)
        runtime = cast(
            InferRuntime,
            validate_payload_schema(
                payload,
                op_name=self.name,
                schema=self.payload_schema(),
                error_type=self.error_type,
            ),
        )

        model_alias = require_nonempty_string(payload, op_name=self.name, key="model")
        raw_tmp_alias = payload.get("tmp_alias")
        if raw_tmp_alias is None:
            tmp_alias = None
        elif isinstance(raw_tmp_alias, str) and raw_tmp_alias:
            tmp_alias = raw_tmp_alias
        else:
            raise InferTransformError("infer.tmp_alias must be a non-empty string when provided")

        input_ids_ref = parse_model_expr(payload["input_ids"], default_model=model_alias)
        _require_single_name_ref(input_ids_ref, op_name=self.name, key="input_ids")

        attention_mask_ref = _compile_optional_ref(
            payload=payload,
            key="attention_mask",
            model_alias=model_alias,
        )
        attn_mask_ref = _compile_optional_ref(
            payload=payload,
            key="attn_mask",
            model_alias=model_alias,
        )
        if attention_mask_ref is not None and attn_mask_ref is not None:
            raise InferTransformError("infer accepts at most one of attention_mask and attn_mask")

        raw_output = payload.get("output", f"{model_alias}::logits")
        output_ref = parse_model_expr(raw_output, default_model=model_alias)
        _require_single_name_ref(output_ref, op_name=self.name, key="output", allow_slice=False)

        return InferSpec(
            runtime=runtime,
            model_alias=model_alias,
            tmp_alias=tmp_alias,
            input_ids_ref=input_ids_ref,
            attention_mask_ref=attention_mask_ref,
            attn_mask_ref=attn_mask_ref,
            output_ref=output_ref,
        )

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        runtime, program = _resolve_runtime_and_program(provider=provider, spec=typed)
        weights_sd = provider.get_state_dict(typed.model_alias)
        runtime_state_dict = provider.get_state_dict(typed.tmp_alias) if typed.tmp_alias else None
        model = _load_runtime_model(
            runtime=runtime,
            program=program,
            state_dict={name: tensor for name, tensor in weights_sd.items()},
            runtime_state_dict=runtime_state_dict,
        ).eval()

        input_ids = _resolve_input_tensor(
            provider=provider,
            ref=typed.input_ids_ref,
            role="input_ids",
        )
        if input_ids.ndim != 2:
            raise InferTransformError(
                "infer.input_ids must resolve to a rank-2 tensor [batch, seq]"
            )
        if input_ids.dtype not in {torch.int32, torch.int64}:
            raise InferTransformError("infer.input_ids must be int32 or int64")

        kwargs: dict[str, Any] = {}
        if typed.attention_mask_ref is not None:
            kwargs["attention_mask"] = _resolve_input_tensor(
                provider=provider,
                ref=typed.attention_mask_ref,
                role="attention_mask",
            )
        if typed.attn_mask_ref is not None:
            kwargs["attn_mask"] = _resolve_input_tensor(
                provider=provider,
                ref=typed.attn_mask_ref,
                role="attn_mask",
            )

        with torch.inference_mode():
            out = model(input_ids=input_ids, **kwargs)

        logits = _extract_logits(out)
        out_model = must_model(typed.output_ref)
        out_sd = provider.get_state_dict(out_model)
        assert isinstance(typed.output_ref.expr, str)
        out_name = typed.output_ref.expr
        if out_name in out_sd:
            raise InferTransformError(f"infer destination already exists: {out_model}::{out_name}")
        out_sd[out_name] = logits.detach().clone()
        emit_verbose_event(
            self.name,
            f"{runtime}:{program} -> {out_model}::{out_name}",
        )
        return TransformResult(name=self.name, count=1)

    def _infer_output_model(self, spec: object) -> str:
        return must_model(self.require_spec(spec).output_ref)


def _compile_optional_ref(
    *, payload: dict[str, Any], key: str, model_alias: str
) -> TensorRef | None:
    if key not in payload:
        return None
    ref = parse_model_expr(payload[key], default_model=model_alias)
    _require_single_name_ref(ref, op_name="infer", key=key)
    return ref


def _require_single_name_ref(
    ref: TensorRef,
    *,
    op_name: str,
    key: str,
    allow_slice: bool = True,
) -> None:
    if ref.model is None:
        raise InferTransformError(f"{op_name}.{key} missing model alias")
    if not allow_slice and ref.slice_spec is not None:
        raise InferTransformError(f"{op_name}.{key} must not be sliced")
    if not isinstance(ref.expr, str):
        raise InferTransformError(f"{op_name}.{key} must resolve to a single tensor name")


def _resolve_input_tensor(
    *, provider: StateDictProvider, ref: TensorRef, role: str
) -> torch.Tensor:
    sd = state_dict_for_ref(provider, ref)
    matches = match_expr_names(
        expr=ref.expr,
        names=sd.keys(),
        op_name="infer",
        role=role,
    )
    if len(matches) != 1:
        raise InferTransformError(
            f"infer.{role} must match exactly one tensor, got {len(matches)} for {must_model(ref)}::{ref.expr}"
        )
    _sd, tensor = unary_view_for_ref_name(provider, ref, matches[0])
    return tensor


def _load_runtime_model(
    *,
    runtime: ResolvedInferRuntime,
    program: str,
    state_dict: dict[str, torch.Tensor],
    runtime_state_dict: Any = None,
) -> Any:
    try:
        kwargs: dict[str, Any] = {
            "runtime": runtime,
            "program": program,
            "state_dict": state_dict,
        }
        if runtime_state_dict is not None:
            kwargs["runtime_state_dict"] = runtime_state_dict
        return _load_infer_runtime_model(**kwargs)
    except ValueError as exc:
        raise InferTransformError(str(exc)) from exc


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        if "logits" in output and isinstance(output["logits"], torch.Tensor):
            return output["logits"]
        tensor_values = [value for value in output.values() if isinstance(value, torch.Tensor)]
        if len(tensor_values) == 1:
            return tensor_values[0]
    raise InferTransformError("infer runtime output must be a tensor or include a tensor 'logits'")


def _resolve_runtime_and_program(
    *,
    provider: StateDictProvider,
    spec: InferSpec,
) -> tuple[ResolvedInferRuntime, str]:
    metadata = get_model_runtime_metadata(provider, spec.model_alias)
    if metadata is None:
        raise InferTransformError(
            "infer requires runtime metadata for model alias; load the model via load before infer"
        )

    metadata_program = metadata.get("program")
    if not isinstance(metadata_program, str) or not metadata_program:
        raise InferTransformError(
            "infer requires model runtime metadata with a non-empty 'program' value"
        )

    metadata_runtime = metadata.get("runtime")
    resolved_runtime: ResolvedInferRuntime
    if spec.runtime == "auto":
        if metadata_runtime not in {"synapse", "codegen", "hf"}:
            raise InferTransformError(
                "infer runtime=auto requires model metadata runtime in {synapse, codegen, hf}"
            )
        resolved_runtime = cast(ResolvedInferRuntime, metadata_runtime)
    else:
        resolved_runtime = spec.runtime

    return resolved_runtime, metadata_program


register_transform(InferTransform())
