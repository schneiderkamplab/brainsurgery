from dataclasses import dataclass

import torch

from ..core import TransformError


@dataclass(frozen=True)
class _DecodedText:
    text: str
    byte_count: int


def _tensor_to_token_ids(
    tensor: object,
    *,
    op_name: str,
    max_tokens: int | None = None,
) -> list[int]:
    if not isinstance(tensor, torch.Tensor):
        raise TransformError(f"{op_name}.from must resolve to a torch.Tensor")
    if tensor.is_complex():
        raise TransformError(f"{op_name}.from tensor must have an integer dtype")
    if tensor.dtype.is_floating_point:
        raise TransformError(f"{op_name}.from tensor must have an integer dtype")
    flattened = tensor.detach().reshape(-1).to(device="cpu", dtype=torch.int64)
    values = flattened.tolist()
    if max_tokens is not None:
        values = values[:max_tokens]
    for idx, value in enumerate(values):
        if value < 0:
            raise TransformError(
                f"{op_name}.from tensor token ids must be >= 0; got {value} at index {idx}"
            )
    return values


def _tensor_to_bytes(
    tensor: object,
    *,
    op_name: str,
    encoding: str,
    errors: str,
    max_bytes: int | None = None,
) -> _DecodedText:
    if not isinstance(tensor, torch.Tensor):
        raise TransformError(f"{op_name}.from must resolve to a torch.Tensor")
    if tensor.is_complex():
        raise TransformError(f"{op_name}.from tensor must have an integer dtype")
    if tensor.dtype.is_floating_point:
        raise TransformError(f"{op_name}.from tensor must have an integer dtype")
    flattened = tensor.detach().reshape(-1).to(device="cpu", dtype=torch.int64)
    values = flattened.tolist()
    if max_bytes is not None:
        values = values[:max_bytes]
    for idx, value in enumerate(values):
        if value < 0 or value > 255:
            raise TransformError(
                f"{op_name}.from tensor values must be in [0, 255]; got {value} at index {idx}"
            )
    byte_values = bytes(values)
    return _DecodedText(
        text=byte_values.decode(encoding, errors=errors),
        byte_count=len(byte_values),
    )


def _text_to_tensor(
    text: str,
    *,
    encoding: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(list(text.encode(encoding)), dtype=dtype)


def _token_ids_to_tensor(
    token_ids: list[int],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(token_ids, dtype=dtype)


__all__: list[str] = []
