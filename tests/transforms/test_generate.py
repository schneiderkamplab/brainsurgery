from __future__ import annotations

import logging

import pytest
import torch

import brainsurgery.transforms.generate as generate_module
from brainsurgery.transforms.generate import GenerateTransform, GenerateTransformError


class _Provider:
    def __init__(self) -> None:
        self._models: dict[str, dict[str, torch.Tensor]] = {
            "model": {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.int64),
                "w": torch.tensor([1.0], dtype=torch.float32),
            },
            "work": {},
        }
        self.model_runtime_metadata: dict[str, dict[str, object]] = {
            "model": {"runtime": "synapse", "program": "examples/gpt2.axon"}
        }

    def get_state_dict(self, model: str) -> dict[str, torch.Tensor]:
        return self._models[model]


def test_generate_compile_defaults_output_name() -> None:
    spec = GenerateTransform().compile(
        {
            "model": "model",
            "input_ids": "model::input_ids",
            "max_new_tokens": 4,
        },
        default_model=None,
    )
    assert spec.runtime == "auto"
    assert spec.output_ref.model == "model"
    assert spec.output_ref.expr == "generated_ids"


def test_generate_compile_rejects_dual_attention_masks() -> None:
    with pytest.raises(GenerateTransformError, match="at most one of attention_mask and attn_mask"):
        GenerateTransform().compile(
            {
                "model": "model",
                "input_ids": "model::input_ids",
                "attention_mask": "model::attention_mask",
                "attn_mask": "model::attention_mask",
                "max_new_tokens": 1,
            },
            default_model=None,
        )


def test_generate_apply_writes_generated_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Runtime:
        def __init__(self) -> None:
            self.calls = 0

        def eval(self) -> "_Runtime":
            return self

        def __call__(self, *, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
            del kwargs
            self.calls += 1
            batch, seq = input_ids.shape
            vocab = 16
            logits = torch.full((batch, seq, vocab), -1000.0, dtype=torch.float32)
            next_ids = (input_ids[:, -1] + 1) % vocab
            logits[torch.arange(batch), seq - 1, next_ids] = 0.0
            return logits

    runtime = _Runtime()
    monkeypatch.setattr(
        generate_module,
        "_load_runtime_model",
        lambda **_kwargs: runtime,
    )

    transform = GenerateTransform()
    spec = transform.compile(
        {
            "runtime": "hf",
            "model": "model",
            "input_ids": "model::input_ids",
            "max_new_tokens": 3,
            "output": "work::generated_ids",
        },
        default_model=None,
    )
    provider = _Provider()
    result = transform.apply(spec, provider)

    assert result.name == "generate"
    assert result.count == 1
    generated = provider.get_state_dict("work")["generated_ids"]
    assert generated.tolist() == [[1, 2, 3, 4, 5, 6]]
    assert runtime.calls == 3


def test_generate_apply_honors_eos_and_extends_attention_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention_mask_lengths: list[int] = []

    class _Runtime:
        def eval(self) -> "_Runtime":
            return self

        def __call__(self, *, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
            mask = kwargs.get("attention_mask")
            assert isinstance(mask, torch.Tensor)
            attention_mask_lengths.append(int(mask.shape[1]))
            batch, seq = input_ids.shape
            vocab = 16
            logits = torch.full((batch, seq, vocab), -1000.0, dtype=torch.float32)
            next_ids = (input_ids[:, -1] + 1) % vocab
            logits[torch.arange(batch), seq - 1, next_ids] = 0.0
            return logits

    monkeypatch.setattr(
        generate_module,
        "_load_runtime_model",
        lambda **_kwargs: _Runtime(),
    )

    transform = GenerateTransform()
    spec = transform.compile(
        {
            "runtime": "hf",
            "model": "model",
            "input_ids": "model::input_ids",
            "attention_mask": "model::attention_mask",
            "max_new_tokens": 10,
            "eos_token_id": 5,
            "output": "work::generated_ids",
        },
        default_model=None,
    )
    provider = _Provider()
    transform.apply(spec, provider)

    generated = provider.get_state_dict("work")["generated_ids"]
    assert generated.tolist() == [[1, 2, 3, 4, 5]]
    assert attention_mask_lengths == [3, 4]


def test_generate_logs_compute_profile_when_gpu_cache_debug_enabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _DebugDict(dict[str, torch.Tensor]):
        @property
        def debug_enabled(self) -> bool:
            return True

    class _ProviderWithDebug(_Provider):
        def __init__(self) -> None:
            super().__init__()
            self._models = {
                "model": _DebugDict(self._models["model"]),
                "work": _DebugDict(self._models["work"]),
            }

    class _TinyRuntime(torch.nn.Module):
        def eval(self) -> "_TinyRuntime":
            return self

        def forward(self, *, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
            del kwargs
            batch, seq = input_ids.shape
            vocab = 16
            logits = torch.full((batch, seq, vocab), -1000.0, dtype=torch.float32)
            next_ids = (input_ids[:, -1] + 1) % vocab
            logits[torch.arange(batch), seq - 1, next_ids] = 0.0
            return logits

    monkeypatch.setattr(
        generate_module,
        "_load_runtime_model",
        lambda **_kwargs: _TinyRuntime(),
    )

    transform = GenerateTransform()
    spec = transform.compile(
        {
            "runtime": "hf",
            "model": "model",
            "input_ids": "model::input_ids",
            "max_new_tokens": 2,
            "output": "work::generated_ids",
        },
        default_model=None,
    )
    provider = _ProviderWithDebug()
    with caplog.at_level(logging.INFO, logger="brainsurgery"):
        transform.apply(spec, provider)

    messages = [record.getMessage() for record in caplog.records]
    assert any("[generate-profiler:model] summary" in message for message in messages)
    assert any("[generate-profiler:model] phase " in message for message in messages)
    assert any("[generate-profiler:model] cache " in message for message in messages)
    assert any("[generate-profiler:model] input-path " in message for message in messages)


def test_generate_uses_past_key_values_when_runtime_provides_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_seq_lens: list[int] = []
    use_cache_flags: list[bool] = []

    class _Runtime:
        def eval(self) -> "_Runtime":
            return self

        def __call__(self, *, input_ids: torch.Tensor, **kwargs: object) -> dict[str, object]:
            input_seq_lens.append(int(input_ids.shape[1]))
            use_cache_flags.append(bool(kwargs.get("use_cache", False)))
            batch, seq = input_ids.shape
            vocab = 16
            logits = torch.full((batch, seq, vocab), -1000.0, dtype=torch.float32)
            next_ids = (input_ids[:, -1] + 1) % vocab
            logits[torch.arange(batch), seq - 1, next_ids] = 0.0
            return {"logits": logits, "past_key_values": ("cached", len(input_seq_lens))}

    monkeypatch.setattr(
        generate_module,
        "_load_runtime_model",
        lambda **_kwargs: _Runtime(),
    )

    transform = GenerateTransform()
    spec = transform.compile(
        {
            "runtime": "hf",
            "model": "model",
            "input_ids": "model::input_ids",
            "max_new_tokens": 3,
            "output": "work::generated_ids",
        },
        default_model=None,
    )
    provider = _Provider()
    transform.apply(spec, provider)

    assert input_seq_lens == [3, 1, 1]
    assert use_cache_flags == [True, True, True]


def test_generate_uses_synapse_style_past_kv_when_runtime_returns_new_kv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_seq_lens: list[int] = []
    past_kv_seen: list[bool] = []

    class _Runtime:
        def eval(self) -> "_Runtime":
            return self

        def __call__(self, *, input_ids: torch.Tensor, **kwargs: object) -> dict[str, object]:
            input_seq_lens.append(int(input_ids.shape[1]))
            past_kv_seen.append(kwargs.get("past_kv") is not None)
            batch, seq = input_ids.shape
            vocab = 16
            logits = torch.full((batch, seq, vocab), -1000.0, dtype=torch.float32)
            next_ids = (input_ids[:, -1] + 1) % vocab
            logits[torch.arange(batch), seq - 1, next_ids] = 0.0
            return {"logits": logits, "new_kv": [("k", len(input_seq_lens))]}

    monkeypatch.setattr(
        generate_module,
        "_load_runtime_model",
        lambda **_kwargs: _Runtime(),
    )

    transform = GenerateTransform()
    spec = transform.compile(
        {
            "runtime": "synapse",
            "model": "model",
            "input_ids": "model::input_ids",
            "max_new_tokens": 3,
            "output": "work::generated_ids",
        },
        default_model=None,
    )
    provider = _Provider()
    transform.apply(spec, provider)

    assert input_seq_lens == [3, 1, 1]
    assert past_kv_seen == [False, True, True]
