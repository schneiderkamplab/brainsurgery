from importlib import import_module

import pytest
import torch

from brainsurgery.engine.state_dicts import _InMemoryStateDict

_module = import_module("brainsurgery.transforms.encode")
EncodeTransform = _module.EncodeTransform
EncodeTransformError = _module.EncodeTransformError


class _Provider:
    def __init__(self) -> None:
        self._state_dict = _InMemoryStateDict()

    def get_state_dict(self, model: str):
        assert model == "m"
        return self._state_dict


def test_encode_creates_new_tensor() -> None:
    provider = _Provider()
    spec = EncodeTransform().compile(
        {"text": "hello", "to": "x"},
        default_model="m",
    )
    result = EncodeTransform().apply(spec, provider)
    assert result.count == 1
    assert provider._state_dict["x"].dtype == torch.uint8
    assert provider._state_dict["x"].tolist() == [104, 101, 108, 108, 111]


def test_encode_rejects_existing_destination() -> None:
    provider = _Provider()
    provider._state_dict["x"] = torch.zeros((1,), dtype=torch.uint8)
    spec = EncodeTransform().compile(
        {"text": "a", "to": "x"},
        default_model="m",
    )
    with pytest.raises(EncodeTransformError, match="destination already exists"):
        EncodeTransform().apply(spec, provider)


def test_encode_validation_errors() -> None:
    with pytest.raises(EncodeTransformError, match="encode.text must be a string"):
        EncodeTransform().compile(
            {"text": 123, "to": "x"},
            default_model="m",
        )
    with pytest.raises(EncodeTransformError, match="encode.dtype must be one of"):
        EncodeTransform().compile(
            {"text": "a", "to": "x", "dtype": "float32"},
            default_model="m",
        )


def test_encode_with_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert text == "hello"
            assert add_special_tokens is True
            return [10, 20, 30]

    monkeypatch.setattr(_module, "_load_tokenizer", lambda source: _Tokenizer())

    provider = _Provider()
    spec = EncodeTransform().compile(
        {"text": "hello", "to": "x", "tokenizer": "models/gpt2"},
        default_model="m",
    )
    EncodeTransform().apply(spec, provider)
    assert provider._state_dict["x"].dtype == torch.int64
    assert provider._state_dict["x"].tolist() == [10, 20, 30]
