from __future__ import annotations

from typing import Any

import torch

from brainsurgery.synapse import SynapseProgramModel
from tests.synapse_test_utils import build_codegen_model


def _spec(*, window: int | None, with_padding: bool) -> dict[str, Any]:
    node: dict[str, Any] = {
        "_op": "bidirectional_mask",
        "_args": ["q", "k"],
        "_bind": "mask",
    }
    if window is not None:
        node["window"] = window
    if with_padding:
        node["padding_mask"] = "padding_mask"
    return {
        "synapse": 1,
        "model": {
            "inputs": {"q": {}, "k": {}, "padding_mask": {"optional": True}},
            "graph": [{"bidir": node}],
            "outputs": {"mask": "mask"},
        },
    }


def _reference_mask(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    window: int | None,
    padding_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    q_len = int(q.shape[-2])
    k_len = int(k.shape[-2])
    if window is None and padding_mask is None:
        return None

    q_idx = torch.arange(q_len, device=q.device).unsqueeze(1)
    k_idx = torch.arange(k_len, device=q.device).unsqueeze(0)
    if window is None:
        keep = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device)
    else:
        if q_len == k_len:
            keep = torch.abs(q_idx - k_idx) <= int(window)
        else:
            keep = torch.abs((q_idx + (k_len - q_len)) - k_idx) <= int(window)

    if padding_mask is not None:
        keep = keep.unsqueeze(0).unsqueeze(0) & padding_mask[:, -k_len:].to(torch.bool).unsqueeze(
            1
        ).unsqueeze(1)
    else:
        keep = keep.unsqueeze(0).unsqueeze(0)

    floor = torch.finfo(q.dtype).min
    return torch.where(
        keep,
        torch.zeros((), dtype=q.dtype, device=q.device),
        torch.full((), floor, dtype=q.dtype, device=q.device),
    )


@torch.inference_mode()
def test_bidirectional_mask_matches_reference_with_window_and_padding() -> None:
    torch.manual_seed(0)
    spec = _spec(window=1, with_padding=True)
    runtime = SynapseProgramModel.from_spec(spec).eval()
    codegen = build_codegen_model(spec, "BidirectionalMaskGenerated", {}).eval()

    q = torch.randn(2, 4, 5, 8, dtype=torch.float32)
    k = torch.randn(2, 4, 7, 8, dtype=torch.float32)
    padding_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1]],
        dtype=torch.long,
    )

    runtime_mask = runtime(q=q, k=k, padding_mask=padding_mask)["mask"]
    codegen_mask = codegen(q=q, k=k, padding_mask=padding_mask)["mask"]
    expected = _reference_mask(q=q, k=k, window=1, padding_mask=padding_mask)

    assert expected is not None
    assert runtime_mask.shape == (2, 1, 5, 7)
    assert torch.allclose(runtime_mask, expected, atol=0.0, rtol=0.0)
    assert torch.allclose(codegen_mask, expected, atol=0.0, rtol=0.0)


@torch.inference_mode()
def test_bidirectional_mask_returns_none_when_unconstrained() -> None:
    spec = _spec(window=None, with_padding=False)
    runtime = SynapseProgramModel.from_spec(spec).eval()
    codegen = build_codegen_model(spec, "BidirectionalMaskNoneGenerated", {}).eval()

    q = torch.randn(1, 2, 4, 8, dtype=torch.float32)
    k = torch.randn(1, 2, 4, 8, dtype=torch.float32)

    assert runtime(q=q, k=k)["mask"] is None
    assert codegen(q=q, k=k)["mask"] is None
