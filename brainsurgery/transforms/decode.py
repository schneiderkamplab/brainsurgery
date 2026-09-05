from dataclasses import dataclass
from typing import Any, Literal, cast

from ..core import (
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformPayloadSchema,
    TransformResult,
    TypedTransform,
    complete_filesystem_paths,
    ensure_mapping_payload,
    format_tensor_ref,
    must_model,
    parse_model_expr,
    register_transform,
    unary_view_for_ref_name,
    validate_payload_schema,
)
from ..engine import emit_line, emit_verbose_event
from ._text_codec import _tensor_to_bytes, _tensor_to_token_ids
from ._tokenizer_loader import _load_tokenizer

_DecodeErrorMode = Literal["strict", "ignore", "replace"]


class DecodeTransformError(TransformError):
    pass


@dataclass(frozen=True)
class DecodeSpec:
    from_ref: TensorRef
    tokenizer: str | None
    encoding: str
    errors: _DecodeErrorMode
    max_bytes: int | None
    skip_special_tokens: bool

    def collect_models(self) -> set[str]:
        return {must_model(self.from_ref)}


class DecodeTransform(TypedTransform[DecodeSpec]):
    name = "decode"
    error_type = DecodeTransformError
    spec_type = DecodeSpec
    help_text = (
        "Decodes a tensor containing byte values and prints it as text.\n"
        "\n"
        "When 'tokenizer' is provided, the tensor is interpreted as token ids and decoded by "
        "that tokenizer. The tokenizer value can be a local directory or a HF hub id.\n"
        "\n"
        "The source tensor must contain integer values in [0, 255].\n"
        "\n"
        "Examples:\n"
        "  decode: { from: work::prompt_bytes }\n"
        "  decode: { from: work::prompt_bytes, encoding: utf-8, errors: replace }\n"
        "  decode: { from: work::generated_ids, tokenizer: models/gpt2 }"
    )

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key=None,
            default_mode="default",
            common_required={"from"},
            common_allowed={
                "from",
                "tokenizer",
                "encoding",
                "errors",
                "max_bytes",
                "skip_special_tokens",
            },
        )

    def completion_reference_keys(self) -> list[str]:
        return ["from"]

    def completion_value_candidates(
        self,
        value_key: str | None,
        prefix_text: str,
        model_aliases: list[str],
    ) -> list[str] | None:
        del model_aliases
        if value_key == "encoding":
            return [item for item in ("utf-8", "ascii", "latin-1") if item.startswith(prefix_text)]
        if value_key == "errors":
            return [
                item for item in ("strict", "ignore", "replace") if item.startswith(prefix_text)
            ]
        if value_key == "tokenizer":
            return complete_filesystem_paths(prefix_text)
        if value_key == "skip_special_tokens":
            return [item for item in ("true", "false") if item.startswith(prefix_text)]
        return None

    def compile(self, payload: Any, default_model: str | None) -> DecodeSpec:
        if isinstance(payload, str):
            payload = {"from": payload}
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_schema(
            payload,
            op_name=self.name,
            schema=self.payload_schema(),
            error_type=self.error_type,
        )

        from_ref = parse_model_expr(payload["from"], default_model=default_model)
        if not isinstance(from_ref.expr, str):
            raise DecodeTransformError("decode.from must resolve to a single tensor name")
        raw_tokenizer = payload.get("tokenizer")
        tokenizer: str | None = None
        if raw_tokenizer is not None:
            if not isinstance(raw_tokenizer, str) or not raw_tokenizer.strip():
                raise DecodeTransformError(
                    "decode.tokenizer must be a non-empty string when provided"
                )
            tokenizer = raw_tokenizer.strip()
        raw_encoding = payload.get("encoding", "utf-8")
        if not isinstance(raw_encoding, str) or not raw_encoding:
            raise DecodeTransformError("decode.encoding must be a non-empty string when provided")
        encoding = raw_encoding.strip()

        raw_errors = payload.get("errors", "strict")
        if not isinstance(raw_errors, str) or raw_errors not in {"strict", "ignore", "replace"}:
            raise DecodeTransformError("decode.errors must be one of: strict, ignore, replace")
        errors = cast(_DecodeErrorMode, raw_errors)

        raw_max_bytes = payload.get("max_bytes")
        max_bytes: int | None = None
        if raw_max_bytes is not None:
            if not isinstance(raw_max_bytes, int) or raw_max_bytes <= 0:
                raise DecodeTransformError(
                    "decode.max_bytes must be a positive integer when provided"
                )
            max_bytes = raw_max_bytes
        raw_skip_special_tokens = payload.get("skip_special_tokens", True)
        if not isinstance(raw_skip_special_tokens, bool):
            raise DecodeTransformError("decode.skip_special_tokens must be a boolean when provided")

        return DecodeSpec(
            from_ref=from_ref,
            tokenizer=tokenizer,
            encoding=encoding,
            errors=errors,
            max_bytes=max_bytes,
            skip_special_tokens=raw_skip_special_tokens,
        )

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        tensor_name = typed.from_ref.expr
        assert isinstance(tensor_name, str)
        model = must_model(typed.from_ref)
        try:
            _, tensor_view = unary_view_for_ref_name(provider, typed.from_ref, tensor_name)
        except KeyError as exc:
            raise DecodeTransformError(
                f"decode source missing: {format_tensor_ref(typed.from_ref)}"
            ) from exc
        try:
            if typed.tokenizer is None:
                decoded = _tensor_to_bytes(
                    tensor_view,
                    op_name=self.name,
                    encoding=typed.encoding,
                    errors=typed.errors,
                    max_bytes=typed.max_bytes,
                )
                text = decoded.text
                count = decoded.byte_count
                unit = "bytes"
            else:
                tokenizer = _load_tokenizer(typed.tokenizer)
                token_ids = _tensor_to_token_ids(
                    tensor_view,
                    op_name=self.name,
                    max_tokens=typed.max_bytes,
                )
                text = tokenizer.decode(
                    token_ids,
                    skip_special_tokens=typed.skip_special_tokens,
                )
                count = len(token_ids)
                unit = "tokens"
        except TransformError as exc:
            raise DecodeTransformError(str(exc)) from exc
        except LookupError as exc:
            raise DecodeTransformError(f"decode unknown text encoding: {typed.encoding}") from exc
        except UnicodeDecodeError as exc:
            raise DecodeTransformError(
                f"decode failed for encoding {typed.encoding!r}: {exc.reason}"
            ) from exc
        except Exception as exc:
            if typed.tokenizer is None:
                raise
            raise DecodeTransformError(
                f"decode failed with tokenizer {typed.tokenizer!r}: {exc}"
            ) from exc

        emit_line(text)
        emit_verbose_event(self.name, f"{model}::{tensor_name} ({count} {unit})")
        return TransformResult(name=self.name, count=1)

    def _infer_output_model(self, spec: object) -> str:
        del spec
        raise DecodeTransformError("decode does not infer an output model")

    def contributes_output_model(self, spec: object) -> bool:
        del spec
        return False


register_transform(DecodeTransform())


__all__ = [
    "DecodeTransformError",
    "DecodeSpec",
    "DecodeTransform",
]
