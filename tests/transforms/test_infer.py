from __future__ import annotations

import pytest
import torch

import brainsurgery.transforms.infer as infer_module
import brainsurgery.transforms.infer_runtime as infer_runtime_module
from brainsurgery.transforms.infer import InferTransform, InferTransformError


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


def test_infer_compile_defaults_to_auto_and_output_logits() -> None:
    spec = InferTransform().compile(
        {
            "model": "model",
            "input_ids": "model::input_ids",
        },
        default_model=None,
    )
    assert spec.runtime == "auto"
    assert spec.output_ref.model == "model"
    assert spec.output_ref.expr == "logits"


def test_infer_compile_rejects_unknown_key_program() -> None:
    with pytest.raises(InferTransformError, match="unknown keys"):
        InferTransform().compile(
            {"model": "model", "program": "examples/gpt2.axon", "input_ids": "model::input_ids"},
            default_model=None,
        )


def test_infer_compile_rejects_dual_attention_masks() -> None:
    with pytest.raises(InferTransformError, match="at most one of attention_mask and attn_mask"):
        InferTransform().compile(
            {
                "model": "model",
                "input_ids": "model::input_ids",
                "attention_mask": "model::attention_mask",
                "attn_mask": "model::attention_mask",
            },
            default_model=None,
        )


def test_infer_apply_writes_output_and_forwards_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Runtime:
        def eval(self) -> "_Runtime":
            return self

        def __call__(self, *, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
            captured["input_ids"] = input_ids
            captured["kwargs"] = kwargs
            return torch.randn((input_ids.shape[0], input_ids.shape[1], 8), dtype=torch.float32)

    monkeypatch.setattr(
        infer_module,
        "_load_runtime_model",
        lambda **_kwargs: _Runtime(),
    )

    transform = InferTransform()
    spec = transform.compile(
        {
            "runtime": "hf",
            "model": "model",
            "input_ids": "model::input_ids",
            "attention_mask": "model::attention_mask",
            "output": "work::logits",
        },
        default_model=None,
    )
    provider = _Provider()
    result = transform.apply(spec, provider)

    assert result.name == "infer"
    assert result.count == 1
    assert "logits" in provider.get_state_dict("work")
    assert isinstance(captured["input_ids"], torch.Tensor)
    assert "attention_mask" in dict(captured["kwargs"])  # type: ignore[arg-type]
    assert provider.model_runtime_metadata["model"]["runtime"] == "synapse"


def test_infer_apply_requires_runtime_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        infer_module,
        "_load_runtime_model",
        lambda **_kwargs: object(),
    )
    transform = InferTransform()
    spec = transform.compile(
        {"model": "model", "input_ids": "model::input_ids"},
        default_model=None,
    )
    provider = _Provider()
    provider.model_runtime_metadata = {}
    with pytest.raises(InferTransformError, match="requires runtime metadata"):
        transform.apply(spec, provider)


def test_load_runtime_model_dispatches() -> None:
    captured: dict[str, object] = {}

    def _engine_loader(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return object()

    original_loader = infer_module._load_infer_runtime_model
    infer_module._load_infer_runtime_model = _engine_loader  # type: ignore[assignment]
    try:
        infer_module._load_runtime_model(runtime="codegen", program="a.axon", state_dict={})
    finally:
        infer_module._load_infer_runtime_model = original_loader  # type: ignore[assignment]
    assert captured == {"runtime": "codegen", "program": "a.axon", "state_dict": {}}


def test_adapt_hf_state_dict_keys_for_gpt2_normalized_names() -> None:
    state_dict = {
        "wte.weight": torch.ones((2, 2)),
        "h.0.attn.c_attn.weight": torch.ones((2, 2)),
        "h.0.attn.bias": torch.ones((1, 1, 2, 2)),
    }
    adapted = infer_runtime_module._adapt_hf_state_dict_keys(state_dict)
    assert "transformer.wte.weight" in adapted
    assert "transformer.h.0.attn.c_attn.weight" in adapted
    assert "transformer.h.0.attn.bias" not in adapted
