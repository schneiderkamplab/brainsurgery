from dataclasses import dataclass
from typing import Any

import torch

from ..core import (
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformPayloadSchema,
    TransformResult,
    TypedTransform,
    complete_filesystem_paths,
    ensure_mapping_payload,
    must_model,
    parse_model_expr,
    parse_torch_dtype,
    register_transform,
    state_dict_for_ref,
    validate_payload_schema,
)
from ..engine import emit_verbose_event
from ._text_codec import _text_to_tensor, _token_ids_to_tensor
from ._tokenizer_loader import _load_tokenizer

_ALLOWED_DTYPES = {
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class EncodeTransformError(TransformError):
    pass


@dataclass(frozen=True)
class EncodeSpec:
    text: str
    to_ref: TensorRef
    tokenizer: str | None
    encoding: str
    dtype: torch.dtype
    add_special_tokens: bool

    def collect_models(self) -> set[str]:
        return {must_model(self.to_ref)}


class EncodeTransform(TypedTransform[EncodeSpec]):
    name = "encode"
    error_type = EncodeTransformError
    spec_type = EncodeSpec
    help_text = (
        "Encodes text into a 1D byte tensor and writes it to a new tensor.\n"
        "\n"
        "When 'tokenizer' is provided, text is tokenized into token ids instead of raw bytes.\n"
        "The tokenizer value can be a local model/tokenizer directory or a HF hub id.\n"
        "\n"
        "The destination tensor must not already exist. The default encoding is utf-8.\n"
        "\n"
        "Examples:\n"
        '  encode: { text: "hello", to: work::prompt_bytes }\n'
        '  encode: { text: "hej", to: work::ids, encoding: utf-8, dtype: int64 }\n'
        '  encode: { text: "hello", to: work::input_ids, tokenizer: models/gpt2 }'
    )

    def payload_schema(self) -> TransformPayloadSchema:
        return TransformPayloadSchema(
            mode_key=None,
            default_mode="default",
            common_required={"text", "to"},
            common_allowed={"text", "to", "tokenizer", "encoding", "dtype", "add_special_tokens"},
        )

    def completion_reference_keys(self) -> list[str]:
        return ["to"]

    def completion_value_candidates(
        self,
        value_key: str | None,
        prefix_text: str,
        model_aliases: list[str],
    ) -> list[str] | None:
        del model_aliases
        if value_key == "encoding":
            return [item for item in ("utf-8", "ascii", "latin-1") if item.startswith(prefix_text)]
        if value_key == "dtype":
            return [
                item
                for item in ("uint8", "int16", "int32", "int64")
                if item.startswith(prefix_text)
            ]
        if value_key == "tokenizer":
            return complete_filesystem_paths(prefix_text)
        if value_key == "add_special_tokens":
            return [item for item in ("true", "false") if item.startswith(prefix_text)]
        return None

    def compile(self, payload: Any, default_model: str | None) -> EncodeSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_schema(
            payload,
            op_name=self.name,
            schema=self.payload_schema(),
            error_type=self.error_type,
        )
        raw_text = payload.get("text")
        if not isinstance(raw_text, str):
            raise EncodeTransformError("encode.text must be a string")
        raw_to = payload["to"]
        to_ref = parse_model_expr(raw_to, default_model=default_model)
        if to_ref.slice_spec is not None:
            raise EncodeTransformError("encode.to must not be sliced")
        if not isinstance(to_ref.expr, str):
            raise EncodeTransformError("encode.to must resolve to a single tensor name")
        raw_tokenizer = payload.get("tokenizer")
        tokenizer: str | None = None
        if raw_tokenizer is not None:
            if not isinstance(raw_tokenizer, str) or not raw_tokenizer.strip():
                raise EncodeTransformError(
                    "encode.tokenizer must be a non-empty string when provided"
                )
            tokenizer = raw_tokenizer.strip()
        raw_encoding = payload.get("encoding", "utf-8")
        if not isinstance(raw_encoding, str) or not raw_encoding:
            raise EncodeTransformError("encode.encoding must be a non-empty string when provided")
        encoding = raw_encoding.strip()
        raw_dtype = payload.get("dtype", "int64" if tokenizer is not None else "uint8")
        if not isinstance(raw_dtype, str) or not raw_dtype.strip():
            raise EncodeTransformError("encode.dtype must be a non-empty string when provided")
        dtype = parse_torch_dtype(
            raw_dtype,
            error_type=EncodeTransformError,
            op_name=self.name,
            field_name="dtype",
        )
        if dtype not in _ALLOWED_DTYPES:
            raise EncodeTransformError("encode.dtype must be one of: uint8, int16, int32, int64")
        raw_add_special_tokens = payload.get("add_special_tokens", True)
        if not isinstance(raw_add_special_tokens, bool):
            raise EncodeTransformError("encode.add_special_tokens must be a boolean when provided")
        return EncodeSpec(
            text=raw_text,
            to_ref=to_ref,
            tokenizer=tokenizer,
            encoding=encoding,
            dtype=dtype,
            add_special_tokens=raw_add_special_tokens,
        )

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        tensor_name = typed.to_ref.expr
        assert isinstance(tensor_name, str)
        model = must_model(typed.to_ref)
        state_dict = state_dict_for_ref(provider, typed.to_ref)
        if tensor_name in state_dict:
            raise EncodeTransformError(f"encode destination already exists: {model}::{tensor_name}")
        try:
            if typed.tokenizer is None:
                state_dict[tensor_name] = _text_to_tensor(
                    typed.text,
                    encoding=typed.encoding,
                    dtype=typed.dtype,
                )
            else:
                tokenizer = _load_tokenizer(typed.tokenizer)
                token_ids = tokenizer.encode(
                    typed.text, add_special_tokens=typed.add_special_tokens
                )
                if not isinstance(token_ids, list) or not all(
                    isinstance(item, int) for item in token_ids
                ):
                    raise EncodeTransformError("encode tokenizer returned non-integer token ids")
                state_dict[tensor_name] = _token_ids_to_tensor(token_ids, dtype=typed.dtype)
        except LookupError as exc:
            raise EncodeTransformError(f"encode unknown text encoding: {typed.encoding}") from exc
        except UnicodeEncodeError as exc:
            raise EncodeTransformError(
                f"encode failed for encoding {typed.encoding!r}: {exc.reason}"
            ) from exc
        except Exception as exc:
            if typed.tokenizer is None:
                raise
            raise EncodeTransformError(
                f"encode failed to tokenize with {typed.tokenizer!r}: {exc}"
            ) from exc
        emit_verbose_event(self.name, f"{model}::{tensor_name} ({len(typed.text)} chars)")
        return TransformResult(name=self.name, count=1)

    def _infer_output_model(self, spec: object) -> str:
        typed = self.require_spec(spec)
        return must_model(typed.to_ref)


register_transform(EncodeTransform())


__all__ = [
    "EncodeTransformError",
    "EncodeSpec",
    "EncodeTransform",
]
