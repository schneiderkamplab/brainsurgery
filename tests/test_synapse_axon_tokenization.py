from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import brainsurgery.synapse.axon.tokenization as tokenization
from brainsurgery.synapse.axon import (
    candidate_tokenizer_dirs,
    preferred_padding_side,
    spec_padding_side,
    tokenize_prompts,
)


def test_tokenization_helpers_are_exported_from_axon() -> None:
    assert "tokenize_prompts" in tokenization.__all__
    assert "load_tokenizer" in tokenization.__all__
    assert "candidate_tokenizer_dirs" in tokenization.__all__


def test_candidate_tokenizer_dirs_order_starts_with_model_then_old(tmp_path: Path) -> None:
    model_dir = tmp_path / "my_model_variant_v1"
    resolved = candidate_tokenizer_dirs(model_dir)
    assert resolved[0] == model_dir.resolve()
    assert resolved[1] == model_dir.with_name("my_model_variant_v1.old").resolve()


def test_spec_padding_side_validation() -> None:
    assert spec_padding_side({"model": {"meta": {"padding_side": "right"}}}) == "right"
    assert preferred_padding_side({"model": {"meta": {}}}) == "left"
    with pytest.raises(ValueError, match="Invalid model.meta.padding_side"):
        spec_padding_side({"model": {"meta": {"padding_side": "center"}}})


def test_tokenize_prompts_sets_padding_side_and_pad_token(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class _FakeBatch(dict[str, torch.Tensor]):
        def to(self, device: torch.device) -> "_FakeBatch":
            del device
            return self

    class _FakeTokenizer:
        def __init__(self) -> None:
            self.padding_side = "right"
            self.pad_token_id: int | None = None
            self.eos_token_id = 7
            self.eos_token = "</s>"
            self.pad_token: str | None = None

        def __call__(
            self,
            prompts: list[str],
            *,
            return_tensors: str,
            padding: bool,
            truncation: bool,
            max_length: int,
        ) -> _FakeBatch:
            assert return_tensors == "pt"
            assert padding is True
            assert truncation is True
            assert max_length == 16
            assert prompts == ["a", "b"]
            return _FakeBatch(
                {
                    "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.long),
                }
            )

    fake_tokenizer = _FakeTokenizer()

    def _fake_from_pretrained(source: str, **kwargs: Any) -> _FakeTokenizer:
        calls.append((source, kwargs))
        return fake_tokenizer

    monkeypatch.setattr(tokenization.AutoTokenizer, "from_pretrained", _fake_from_pretrained)

    tok, input_ids, attention_mask = tokenize_prompts(
        prompts=["a", "b"],
        tokenizer_source="local-tokenizer",
        device=torch.device("cpu"),
        max_len=16,
        lowered_spec={"model": {"meta": {"padding_side": "left"}}},
    )

    assert tok is fake_tokenizer
    assert input_ids.shape == (2, 2)
    assert attention_mask is not None and attention_mask.shape == (2, 2)
    assert fake_tokenizer.padding_side == "left"
    assert fake_tokenizer.pad_token == fake_tokenizer.eos_token
    assert calls
