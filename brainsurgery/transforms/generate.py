import logging
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any, cast

import torch

from ..core import (
    StateDictLike,
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
from ..engine import emit_verbose_event
from .infer import (
    _compile_optional_ref,
    _extract_logits,
    _InferRuntime,
    _load_runtime_model,
    _require_single_name_ref,
    _resolve_runtime_and_program,
)

logger = logging.getLogger("brainsurgery")


class GenerateTransformError(TransformError):
    pass


@dataclass(frozen=True)
class GenerateSpec:
    runtime: _InferRuntime
    model_alias: str
    tmp_alias: str | None
    input_ids_ref: TensorRef
    attention_mask_ref: TensorRef | None
    attn_mask_ref: TensorRef | None
    output_ref: TensorRef
    max_new_tokens: int
    eos_token_id: int | None

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


class GenerateTransform(TypedTransform[GenerateSpec]):
    name = "generate"
    error_type = GenerateTransformError
    spec_type = GenerateSpec
    help_text = (
        "Autoregressively generates tokens by repeatedly running infer-style forward passes.\n"
        "\n"
        "Uses greedy decoding (argmax over logits at the last sequence position).\n"
        "Writes the full generated token id tensor to 'output'.\n"
        "\n"
        "Examples:\n"
        "  generate: { model: gpt2, input_ids: work::input_ids, max_new_tokens: 16, output: work::generated_ids }\n"
        "  generate: { model: gpt2, runtime: codegen, input_ids: work::input_ids, max_new_tokens: 8, eos_token_id: 50256 }"
    )

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key="runtime",
            default_mode="auto",
            common_required={"model", "input_ids", "max_new_tokens"},
            common_allowed={
                "runtime",
                "model",
                "tmp_alias",
                "input_ids",
                "attention_mask",
                "attn_mask",
                "output",
                "max_new_tokens",
                "eos_token_id",
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

    def compile(self, payload: Any, default_model: str | None) -> GenerateSpec:
        del default_model
        payload = ensure_mapping_payload(payload, self.name)
        runtime = cast(
            _InferRuntime,
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
            raise GenerateTransformError(
                "generate.tmp_alias must be a non-empty string when provided"
            )

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
            raise GenerateTransformError(
                "generate accepts at most one of attention_mask and attn_mask"
            )

        raw_output = payload.get("output", f"{model_alias}::generated_ids")
        output_ref = parse_model_expr(raw_output, default_model=model_alias)
        _require_single_name_ref(output_ref, op_name=self.name, key="output", allow_slice=False)

        raw_max_new_tokens = payload["max_new_tokens"]
        max_new_tokens = _coerce_int(
            raw_max_new_tokens,
            field_name="generate.max_new_tokens",
        )
        if max_new_tokens < 0:
            raise GenerateTransformError("generate.max_new_tokens must be non-negative")

        eos_token_id: int | None
        raw_eos_token_id = payload.get("eos_token_id")
        if raw_eos_token_id is None:
            eos_token_id = None
        else:
            eos_token_id = _coerce_int(
                raw_eos_token_id,
                field_name="generate.eos_token_id",
            )

        return GenerateSpec(
            runtime=runtime,
            model_alias=model_alias,
            tmp_alias=tmp_alias,
            input_ids_ref=input_ids_ref,
            attention_mask_ref=attention_mask_ref,
            attn_mask_ref=attn_mask_ref,
            output_ref=output_ref,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        runtime, program = _resolve_runtime_and_program(provider=provider, spec=typed)
        weights_sd = provider.get_state_dict(typed.model_alias)
        runtime_state_dict = provider.get_state_dict(typed.tmp_alias) if typed.tmp_alias else None
        profile_enabled = _is_generate_compute_profile_enabled(
            weights_state_dict=weights_sd,
            runtime_state_dict=runtime_state_dict,
        )
        phase_profiler = _GeneratePhaseProfiler(enabled=profile_enabled)
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
            raise GenerateTransformError(
                "generate.input_ids must resolve to a rank-2 tensor [batch, seq]"
            )
        if input_ids.dtype not in {torch.int32, torch.int64}:
            raise GenerateTransformError("generate.input_ids must be int32 or int64")
        generated = input_ids.detach().clone()

        attention_mask: torch.Tensor | None = None
        if typed.attention_mask_ref is not None:
            attention_mask = _resolve_input_tensor(
                provider=provider,
                ref=typed.attention_mask_ref,
                role="attention_mask",
            )
            if attention_mask.ndim != 2:
                raise GenerateTransformError(
                    "generate.attention_mask must resolve to a rank-2 tensor [batch, seq]"
                )
            if attention_mask.shape != generated.shape:
                raise GenerateTransformError(
                    "generate.attention_mask shape must match input_ids shape"
                )
            attention_mask = attention_mask.detach().clone()

        if typed.attn_mask_ref is not None:
            raise GenerateTransformError("generate.attn_mask is not yet supported")

        with torch.inference_mode():
            with _LeafModuleProfiler(model=model, enabled=profile_enabled) as module_profiler:
                past_key_values: object | None = None
                past_cache_arg_name: str = "past_key_values"
                cache_trace = _GenerateCacheTrace()
                op_profiler = _GenerateOpProfiler(
                    enabled=profile_enabled and _is_generate_op_profile_enabled(),
                    model_alias=typed.model_alias,
                    max_steps=typed.max_new_tokens,
                )
                for step_index in range(typed.max_new_tokens):
                    kwargs: dict[str, Any] = {}
                    if attention_mask is not None:
                        kwargs["attention_mask"] = attention_mask
                    kwargs["use_cache"] = True
                    cache_trace.use_cache_requested_steps += 1
                    if past_key_values is not None:
                        kwargs[past_cache_arg_name] = past_key_values
                        model_input_ids = generated[:, -1:]
                        cache_trace.token_only_input_steps += 1
                    else:
                        model_input_ids = generated
                        cache_trace.full_input_steps += 1
                    cache_trace.past_provided_steps += int(past_key_values is not None)
                    with phase_profiler.measure("model_forward"):
                        out = op_profiler.run_forward(
                            step_index=step_index,
                            model=model,
                            input_ids=model_input_ids,
                            kwargs=kwargs,
                            cache_trace=cache_trace,
                        )
                    with phase_profiler.measure("extract_logits"):
                        logits = _extract_logits(out)
                        past_key_values, cache_arg_name = _extract_past_key_values(out)
                        if cache_arg_name is not None:
                            past_cache_arg_name = cache_arg_name
                        if past_key_values is not None:
                            cache_trace.past_returned_steps += 1
                    if logits.ndim != 3:
                        raise GenerateTransformError(
                            "generate runtime logits must have rank 3 [batch, seq, vocab]"
                        )
                    with phase_profiler.measure("argmax_next_token"):
                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                        next_token = next_token.to(dtype=generated.dtype)
                    with phase_profiler.measure("concat_generated"):
                        generated = torch.cat([generated, next_token], dim=1)
                    if attention_mask is not None:
                        with phase_profiler.measure("concat_attention_mask"):
                            pad = torch.ones(
                                (attention_mask.shape[0], 1),
                                dtype=attention_mask.dtype,
                                device=attention_mask.device,
                            )
                            attention_mask = torch.cat([attention_mask, pad], dim=1)
                    if typed.eos_token_id is not None:
                        with phase_profiler.measure("eos_check"):
                            reached_eos = bool(torch.all(next_token == int(typed.eos_token_id)))
                        if reached_eos:
                            break

        _log_generate_compute_profile(
            enabled=profile_enabled,
            model_alias=typed.model_alias,
            max_new_tokens=typed.max_new_tokens,
            generated_tokens=max(0, generated.shape[1] - input_ids.shape[1]),
            phase_profiler=phase_profiler,
            module_profiler=module_profiler,
            cache_trace=cache_trace,
            op_profiler=op_profiler,
        )

        out_model = must_model(typed.output_ref)
        out_sd = provider.get_state_dict(out_model)
        assert isinstance(typed.output_ref.expr, str)
        out_name = typed.output_ref.expr
        if out_name in out_sd:
            raise GenerateTransformError(
                f"generate destination already exists: {out_model}::{out_name}"
            )
        out_sd[out_name] = generated.detach().clone()
        emit_verbose_event(
            self.name,
            f"{runtime}:{program} -> {out_model}::{out_name} (+{generated.shape[1] - input_ids.shape[1]} tokens)",
        )
        return TransformResult(name=self.name, count=1)

    def _infer_output_model(self, spec: object) -> str:
        return must_model(self.require_spec(spec).output_ref)


def _resolve_input_tensor(
    *, provider: StateDictProvider, ref: TensorRef, role: str
) -> torch.Tensor:
    sd = state_dict_for_ref(provider, ref)
    matches = match_expr_names(
        expr=ref.expr,
        names=sd.keys(),
        op_name="generate",
        role=role,
    )
    if len(matches) != 1:
        raise GenerateTransformError(
            "generate."
            f"{role} must match exactly one tensor, got {len(matches)} for {must_model(ref)}::{ref.expr}"
        )
    _sd, tensor = unary_view_for_ref_name(provider, ref, matches[0])
    return tensor


register_transform(GenerateTransform())


def _coerce_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise GenerateTransformError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text and (text.isdigit() or (text[0] in "+-" and text[1:].isdigit())):
            return int(text, 10)
    raise GenerateTransformError(f"{field_name} must be an integer")


def _extract_past_key_values(output: object) -> tuple[object | None, str | None]:
    if isinstance(output, dict):
        if "new_kv" in output:
            return output.get("new_kv"), "past_kv"
        return output.get("past_key_values"), "past_key_values"
    value = getattr(output, "past_key_values", None)
    if value is not None:
        return value, "past_key_values"
    if isinstance(output, tuple) and len(output) >= 2:
        return output[1], "past_key_values"
    return None, None


@dataclass
class _ProfileRow:
    calls: int = 0
    seconds: float = 0.0


@dataclass
class _GenerateCacheTrace:
    use_cache_requested_steps: int = 0
    use_cache_unsupported_steps: int = 0
    past_provided_steps: int = 0
    past_returned_steps: int = 0
    full_input_steps: int = 0
    token_only_input_steps: int = 0


class _GeneratePhaseProfiler:
    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled
        self._rows: dict[str, _ProfileRow] = {}

    def measure(self, name: str):
        return _ProfileScope(self, name, self._enabled)

    def add(self, name: str, duration_seconds: float) -> None:
        if not self._enabled:
            return
        row = self._rows.get(name)
        if row is None:
            row = _ProfileRow()
            self._rows[name] = row
        row.calls += 1
        row.seconds += duration_seconds

    def rows(self) -> dict[str, _ProfileRow]:
        return self._rows


class _ProfileScope:
    def __init__(self, profiler: _GeneratePhaseProfiler, name: str, enabled: bool) -> None:
        self._profiler = profiler
        self._name = name
        self._enabled = enabled
        self._start: float | None = None

    def __enter__(self) -> "_ProfileScope":
        if self._enabled:
            self._start = perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        if not self._enabled or self._start is None:
            return
        self._profiler.add(self._name, perf_counter() - self._start)


class _LeafModuleProfiler:
    def __init__(self, *, model: Any, enabled: bool) -> None:
        self._enabled = enabled
        self._rows: dict[str, _ProfileRow] = {}
        self._handles: list[Any] = []
        self._start_by_module_id: dict[int, float] = {}
        self._name_by_module_id: dict[int, str] = {}
        self._model = model

    def __enter__(self) -> "_LeafModuleProfiler":
        if not self._enabled or not isinstance(self._model, torch.nn.Module):
            return self
        for name, module in self._model.named_modules():
            if any(True for _ in module.children()):
                continue
            module_name = name if name else "<root>"
            self._name_by_module_id[id(module)] = module_name
            self._handles.append(module.register_forward_pre_hook(self._on_pre_hook))
            self._handles.append(module.register_forward_hook(self._on_post_hook))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._start_by_module_id.clear()
        self._name_by_module_id.clear()

    def _on_pre_hook(self, module: torch.nn.Module, _inputs: tuple[object, ...]) -> None:
        module_id = id(module)
        self._start_by_module_id[module_id] = perf_counter()

    def _on_post_hook(
        self,
        module: torch.nn.Module,
        _inputs: tuple[object, ...],
        _output: object,
    ) -> None:
        module_id = id(module)
        start = self._start_by_module_id.pop(module_id, None)
        if start is None:
            return
        module_name = self._name_by_module_id.get(module_id, "<unknown>")
        row = self._rows.get(module_name)
        if row is None:
            row = _ProfileRow()
            self._rows[module_name] = row
        row.calls += 1
        row.seconds += perf_counter() - start

    def rows(self) -> dict[str, _ProfileRow]:
        return self._rows


@dataclass
class _GenerateOpRow:
    calls: int = 0
    cpu_time_total_us: float = 0.0
    cpu_time_self_us: float = 0.0


class _GenerateOpProfiler:
    def __init__(self, *, enabled: bool, model_alias: str, max_steps: int) -> None:
        self._enabled = enabled
        self._model_alias = model_alias
        self._sample_steps = _sample_step_indices(max_steps)
        self._rows: dict[str, _GenerateOpRow] = {}

    def run_forward(
        self,
        *,
        step_index: int,
        model: Any,
        input_ids: torch.Tensor,
        kwargs: dict[str, Any],
        cache_trace: _GenerateCacheTrace,
    ) -> object:
        if not self._enabled or step_index not in self._sample_steps:
            return _run_model_forward_with_fallback(
                model=model,
                input_ids=input_ids,
                kwargs=kwargs,
                cache_trace=cache_trace,
            )

        activities = [torch.profiler.ProfilerActivity.CPU]
        with torch.profiler.profile(activities=activities) as profiler:
            out = _run_model_forward_with_fallback(
                model=model,
                input_ids=input_ids,
                kwargs=kwargs,
                cache_trace=cache_trace,
            )
        logger.info(
            "[generate-op-profiler:%s] sample step=%d input_seq_len=%d sampled_steps=%s",
            self._model_alias,
            step_index,
            int(input_ids.shape[1]),
            ",".join(str(v) for v in sorted(self._sample_steps)),
        )
        for item in profiler.key_averages():
            key = str(item.key)
            row = self._rows.get(key)
            if row is None:
                row = _GenerateOpRow()
                self._rows[key] = row
            row.calls += int(item.count)
            row.cpu_time_total_us += float(item.cpu_time_total)
            row.cpu_time_self_us += float(item.self_cpu_time_total)
        return out

    def rows(self) -> dict[str, _GenerateOpRow]:
        return self._rows


def _run_model_forward_with_fallback(
    *,
    model: Any,
    input_ids: torch.Tensor,
    kwargs: dict[str, Any],
    cache_trace: _GenerateCacheTrace,
) -> object:
    try:
        return model(input_ids=input_ids, **kwargs)
    except TypeError:
        cache_trace.use_cache_unsupported_steps += 1
        kwargs.pop("use_cache", None)
        kwargs.pop("past_key_values", None)
        kwargs.pop("past_kv", None)
        return model(input_ids=input_ids, **kwargs)


def _sample_step_indices(max_steps: int) -> set[int]:
    if max_steps <= 0:
        return set()
    out = {0}
    if max_steps >= 2:
        out.add(1)
    out.add(max_steps - 1)
    return out


def _is_generate_compute_profile_enabled(
    *,
    weights_state_dict: StateDictLike,
    runtime_state_dict: StateDictLike | None,
) -> bool:
    if bool(getattr(weights_state_dict, "debug_enabled", False)):
        return True
    if runtime_state_dict is not None and bool(getattr(runtime_state_dict, "debug_enabled", False)):
        return True
    return False


def _is_generate_op_profile_enabled() -> bool:
    flag = os.getenv("BRAIN_SURGERY_GENERATE_OP_PROFILE", "")
    return flag.strip().lower() in {"1", "true", "yes", "on"}


def _log_generate_compute_profile(
    *,
    enabled: bool,
    model_alias: str,
    max_new_tokens: int,
    generated_tokens: int,
    phase_profiler: _GeneratePhaseProfiler,
    module_profiler: _LeafModuleProfiler,
    cache_trace: _GenerateCacheTrace,
    op_profiler: _GenerateOpProfiler,
) -> None:
    if not enabled:
        return
    phase_rows = phase_profiler.rows()
    total_phase_seconds = sum(row.seconds for row in phase_rows.values())
    logger.info(
        "[generate-profiler:%s] summary max_new_tokens=%d generated_tokens=%d phase_total_seconds=%.6f phase_kinds=%d",
        model_alias,
        max_new_tokens,
        generated_tokens,
        total_phase_seconds,
        len(phase_rows),
    )
    for rank, (name, row) in enumerate(
        sorted(phase_rows.items(), key=lambda item: item[1].seconds, reverse=True)[:10],
        start=1,
    ):
        logger.info(
            "[generate-profiler:%s] phase rank=%d name=%s calls=%d seconds=%.6f",
            model_alias,
            rank,
            name,
            row.calls,
            row.seconds,
        )
    logger.info(
        "[generate-profiler:%s] cache use_cache_requested_steps=%d use_cache_unsupported_steps=%d past_provided_steps=%d past_returned_steps=%d",
        model_alias,
        cache_trace.use_cache_requested_steps,
        cache_trace.use_cache_unsupported_steps,
        cache_trace.past_provided_steps,
        cache_trace.past_returned_steps,
    )
    logger.info(
        "[generate-profiler:%s] input-path full_input_steps=%d token_only_input_steps=%d",
        model_alias,
        cache_trace.full_input_steps,
        cache_trace.token_only_input_steps,
    )

    module_rows = module_profiler.rows()
    total_module_seconds = sum(row.seconds for row in module_rows.values())
    logger.info(
        "[generate-profiler:%s] modules_total seconds=%.6f unique_leaf_modules=%d",
        model_alias,
        total_module_seconds,
        len(module_rows),
    )
    for rank, (name, row) in enumerate(
        sorted(module_rows.items(), key=lambda item: item[1].seconds, reverse=True)[:20],
        start=1,
    ):
        logger.info(
            "[generate-profiler:%s] module rank=%d name=%s calls=%d seconds=%.6f",
            model_alias,
            rank,
            name,
            row.calls,
            row.seconds,
        )

    op_rows = op_profiler.rows()
    total_op_cpu_us = sum(row.cpu_time_total_us for row in op_rows.values())
    logger.info(
        "[generate-op-profiler:%s] summary enabled=%s unique_ops=%d total_cpu_us=%.3f",
        model_alias,
        bool(op_rows),
        len(op_rows),
        total_op_cpu_us,
    )
    for rank, (name, op_row) in enumerate(
        sorted(op_rows.items(), key=lambda item: item[1].cpu_time_total_us, reverse=True)[:30],
        start=1,
    ):
        logger.info(
            "[generate-op-profiler:%s] op rank=%d name=%s calls=%d cpu_total_us=%.3f cpu_self_us=%.3f",
            model_alias,
            rank,
            name,
            op_row.calls,
            op_row.cpu_time_total_us,
            op_row.cpu_time_self_us,
        )
