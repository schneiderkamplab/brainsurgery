from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from brainsurgery.synapse import SynapseProgramModel
from tests.synapse_test_utils import build_codegen_model


def _spec(*, float_mask_additive: bool, float_mask_floor_keep: bool) -> dict[str, Any]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"q": {}, "k": {}, "v": {}, "mask": {}},
            "graph": [
                {
                    "attn": {
                        "_op": "attention",
                        "_args": ["q", "k", "v"],
                        "_bind": "out",
                        "mask": "mask",
                        "causal": False,
                        "eager": True,
                        "float_mask_additive": float_mask_additive,
                        "float_mask_floor_keep": float_mask_floor_keep,
                        "scale": 1.0,
                    }
                }
            ],
            "outputs": {"out": "out"},
        },
    }


@torch.inference_mode()
def test_attention_eager_float_mask_floor_keep_matches_reference() -> None:
    spec = _spec(float_mask_additive=False, float_mask_floor_keep=True)
    runtime = SynapseProgramModel.from_spec(spec).eval()
    codegen = build_codegen_model(spec, "AttentionFloorKeepGenerated", {}).eval()

    q = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
    k = torch.zeros((1, 1, 3, 2), dtype=torch.float32)
    v = torch.tensor([[[[1.0, 0.0], [3.0, 1.0], [50.0, 50.0]]]], dtype=torch.float32)

    floor = torch.finfo(torch.float32).min
    mask = torch.tensor([[[[0.25, -0.35, floor]]]], dtype=torch.float32)

    runtime_out = runtime(q=q, k=k, v=v, mask=mask)["out"]
    codegen_out = codegen(q=q, k=k, v=v, mask=mask)["out"]

    expected_probs = F.softmax(mask[..., :2], dim=-1, dtype=torch.float32)
    expected = expected_probs[..., :1] * v[..., :1, :] + expected_probs[..., 1:2] * v[..., 1:2, :]

    assert torch.allclose(runtime_out, expected, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(codegen_out, expected, atol=1.0e-6, rtol=1.0e-6)
    assert float(runtime_out.abs().sum()) > 0.0
    assert float(codegen_out.abs().sum()) > 0.0


@torch.inference_mode()
def test_attention_eager_float_mask_additive_matches_reference() -> None:
    spec = _spec(float_mask_additive=True, float_mask_floor_keep=False)
    runtime = SynapseProgramModel.from_spec(spec).eval()
    codegen = build_codegen_model(spec, "AttentionAdditiveGenerated", {}).eval()

    q = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
    k = torch.zeros((1, 1, 3, 2), dtype=torch.float32)
    v = torch.tensor([[[[1.0, 0.0], [3.0, 1.0], [50.0, 50.0]]]], dtype=torch.float32)

    floor = torch.finfo(torch.float32).min
    mask = torch.tensor([[[[0.25, -0.35, floor]]]], dtype=torch.float32)

    runtime_out = runtime(q=q, k=k, v=v, mask=mask)["out"]
    codegen_out = codegen(q=q, k=k, v=v, mask=mask)["out"]

    expected_probs = F.softmax(mask[..., :2], dim=-1, dtype=torch.float32)
    expected = expected_probs[..., :1] * v[..., :1, :] + expected_probs[..., 1:2] * v[..., 1:2, :]

    assert torch.allclose(runtime_out, expected, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(codegen_out, expected, atol=1.0e-6, rtol=1.0e-6)
    assert float(runtime_out.abs().sum()) > 0.0
    assert float(codegen_out.abs().sum()) > 0.0
