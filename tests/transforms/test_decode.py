from importlib import import_module

import pytest
import torch

from brainsurgery.engine.state_dicts import _InMemoryStateDict

_module = import_module("brainsurgery.transforms.decode")
DecodeTransform = _module.DecodeTransform
DecodeTransformError = _module.DecodeTransformError


class _Provider:
    def __init__(self) -> None:
        self._state_dict = _InMemoryStateDict()

    def get_state_dict(self, model: str):
        assert model == "m"
        return self._state_dict


def test_decode_emits_text(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _Provider()
    provider._state_dict["x"] = torch.tensor([104, 101, 106], dtype=torch.uint8)
    lines: list[str] = []
    monkeypatch.setattr(_module, "emit_line", lines.append)
    spec = DecodeTransform().compile({"from": "x"}, default_model="m")
    result = DecodeTransform().apply(spec, provider)
    assert result.count == 1
    assert lines == ["hej"]


def test_decode_rejects_out_of_range_values() -> None:
    provider = _Provider()
    provider._state_dict["x"] = torch.tensor([256], dtype=torch.int64)
    spec = DecodeTransform().compile({"from": "x"}, default_model="m")
    with pytest.raises(DecodeTransformError, match="values must be in \\[0, 255\\]"):
        DecodeTransform().apply(spec, provider)


def test_decode_compile_validation_and_max_bytes() -> None:
    with pytest.raises(DecodeTransformError, match="decode.errors must be one of"):
        DecodeTransform().compile({"from": "x", "errors": "bad"}, default_model="m")
    with pytest.raises(DecodeTransformError, match="decode.max_bytes must be a positive integer"):
        DecodeTransform().compile({"from": "x", "max_bytes": 0}, default_model="m")

    provider = _Provider()
    provider._state_dict["x"] = torch.tensor([104, 101, 106], dtype=torch.uint8)
    lines: list[str] = []
    spec = DecodeTransform().compile({"from": "x", "max_bytes": 2}, default_model="m")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(_module, "emit_line", lines.append)
        DecodeTransform().apply(spec, provider)
    assert lines == ["he"]


def test_decode_with_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Tokenizer:
        def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
            assert token_ids == [1, 2, 3]
            assert skip_special_tokens is True
            return "hello"

    monkeypatch.setattr(_module, "_load_tokenizer", lambda source: _Tokenizer())

    provider = _Provider()
    provider._state_dict["x"] = torch.tensor([1, 2, 3], dtype=torch.int64)
    lines: list[str] = []
    spec = DecodeTransform().compile(
        {"from": "x", "tokenizer": "models/gpt2"},
        default_model="m",
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_module, "emit_line", lines.append)
        DecodeTransform().apply(spec, provider)
    assert lines == ["hello"]
