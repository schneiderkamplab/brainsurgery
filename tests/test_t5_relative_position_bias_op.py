from __future__ import annotations

import math
from typing import Any

import torch

from brainsurgery.synapse import SynapseProgramModel
from tests.synapse_test_utils import build_codegen_model


def _relative_position_bucket(
    relative_position: torch.Tensor, *, bidirectional: bool, num_buckets: int, max_distance: int
) -> torch.Tensor:
    relative_buckets = torch.zeros_like(relative_position, dtype=torch.long)
    if bidirectional:
        half_buckets = num_buckets // 2
        relative_buckets = relative_buckets + (relative_position > 0).to(torch.long) * half_buckets
        relative_position = torch.abs(relative_position)
        bucket_count = half_buckets
    else:
        relative_position = -torch.minimum(relative_position, torch.zeros_like(relative_position))
        bucket_count = num_buckets

    max_exact = max(1, bucket_count // 2)
    rel_clamped = torch.clamp(relative_position.to(torch.float32), min=float(max_exact))
    rel_large = max_exact + (
        torch.log(rel_clamped / float(max_exact))
        / math.log(float(max_distance) / float(max_exact))
        * float(bucket_count - max_exact)
    ).to(torch.long)
    rel_large = torch.minimum(rel_large, torch.full_like(rel_large, bucket_count - 1))
    return relative_buckets + torch.where(
        relative_position < max_exact, relative_position, rel_large
    )


def _reference_bias(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    weight: torch.Tensor,
    bidirectional: bool,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    q_len = int(q.shape[-2])
    k_len = int(k.shape[-2])
    context_position = torch.arange(q_len, device=q.device, dtype=torch.long)[:, None]
    memory_position = torch.arange(k_len, device=q.device, dtype=torch.long)[None, :]
    relative_position = memory_position - context_position
    buckets = _relative_position_bucket(
        relative_position,
        bidirectional=bidirectional,
        num_buckets=num_buckets,
        max_distance=max_distance,
    )
    return weight.to(device=q.device, dtype=q.dtype)[buckets].permute(2, 0, 1).unsqueeze(0)


def _spec(*, bidirectional: bool) -> dict[str, Any]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"q": {}, "k": {}},
            "graph": [
                {
                    "rel_bias": {
                        "_op": "t5_relative_position_bias",
                        "_args": ["q", "k"],
                        "_bind": "bias",
                        "num_buckets": 32,
                        "max_distance": 128,
                        "bidirectional": bidirectional,
                    }
                }
            ],
            "outputs": {"bias": "bias"},
        },
    }


@torch.inference_mode()
def test_t5_relative_position_bias_matches_reference_bidirectional() -> None:
    torch.manual_seed(0)
    spec = _spec(bidirectional=True)
    weight = torch.randn(32, 4, dtype=torch.float32)
    state_dict = {"rel_bias.weight": weight}

    runtime = SynapseProgramModel.from_spec(spec, state_dict=state_dict).eval()
    codegen = build_codegen_model(spec, "RelBiasBidirectionalGenerated", state_dict).eval()

    q = torch.randn(2, 4, 5, 8, dtype=torch.float32)
    k = torch.randn(2, 4, 7, 8, dtype=torch.float32)

    runtime_bias = runtime(q=q, k=k)["bias"]
    codegen_bias = codegen(q=q, k=k)["bias"]
    expected = _reference_bias(q=q, k=k, weight=weight, bidirectional=True)

    assert runtime_bias.shape == (1, 4, 5, 7)
    assert torch.allclose(runtime_bias, expected, atol=0.0, rtol=0.0)
    assert torch.allclose(codegen_bias, expected, atol=0.0, rtol=0.0)


@torch.inference_mode()
def test_t5_relative_position_bias_matches_reference_unidirectional() -> None:
    torch.manual_seed(0)
    spec = _spec(bidirectional=False)
    weight = torch.randn(32, 4, dtype=torch.float32)
    state_dict = {"rel_bias.weight": weight}

    runtime = SynapseProgramModel.from_spec(spec, state_dict=state_dict).eval()

    q = torch.randn(1, 4, 6, 8, dtype=torch.float32)
    k = torch.randn(1, 4, 6, 8, dtype=torch.float32)

    runtime_bias = runtime(q=q, k=k)["bias"]
    expected = _reference_bias(q=q, k=k, weight=weight, bidirectional=False)

    assert runtime_bias.shape == (1, 4, 6, 6)
    assert torch.allclose(runtime_bias, expected, atol=0.0, rtol=0.0)
