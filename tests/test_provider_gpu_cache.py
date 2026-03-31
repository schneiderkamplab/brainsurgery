from __future__ import annotations

import logging

import pytest
import torch

from brainsurgery.engine.arena import ProviderError
from brainsurgery.engine.providers import InMemoryStateDictProvider, wrap_provider_with_gpu_cache
from brainsurgery.engine.state_dicts import GpuCacheConfig, GpuCachedStateDict


def test_gpu_cached_provider_write_back_on_mark_write() -> None:
    base_provider = InMemoryStateDictProvider({}, max_io_workers=1)
    base_sd = base_provider.get_or_create_alias_state_dict("model")
    base_sd["weight"] = torch.ones(4, dtype=torch.float32)

    cached_provider = wrap_provider_with_gpu_cache(
        base_provider,
        cache_config=GpuCacheConfig(device="cpu", max_cache_bytes=1024),
    )
    cached_sd = cached_provider.get_state_dict("model")

    view = cached_sd["weight"]
    view.add_(2.0)
    cached_sd.mark_write("weight")

    expected = torch.full((4,), 3.0, dtype=torch.float32)
    assert torch.equal(base_sd["weight"], expected)
    assert torch.equal(cached_sd["weight"], expected)


def test_gpu_cached_provider_eviction_under_budget() -> None:
    base_provider = InMemoryStateDictProvider({}, max_io_workers=1)
    base_sd = base_provider.get_or_create_alias_state_dict("model")
    base_sd["a"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    base_sd["b"] = torch.tensor([3.0, 4.0], dtype=torch.float32)

    cached_provider = wrap_provider_with_gpu_cache(
        base_provider,
        cache_config=GpuCacheConfig(device="cpu", max_cache_bytes=8),
    )
    cached_sd = cached_provider.get_state_dict("model")

    first_a = cached_sd["a"]
    _ = cached_sd["b"]
    second_a = cached_sd["a"]

    # max_cache_bytes allows only one 2xfloat32 tensor; re-reading "a" reloads it.
    assert first_a.data_ptr() != second_a.data_ptr()
    assert cached_sd.cache_bytes() <= 8


def test_gpu_cached_provider_uses_fractional_auto_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    base_provider = InMemoryStateDictProvider({}, max_io_workers=1)
    base_sd = base_provider.get_or_create_alias_state_dict("model")
    base_sd["x"] = torch.ones(1, dtype=torch.float32)

    monkeypatch.setattr(
        "brainsurgery.engine.state_dicts._detect_total_memory_bytes_for_device",
        lambda _device: 1000,
    )
    cached_provider = wrap_provider_with_gpu_cache(
        base_provider,
        cache_config=GpuCacheConfig(device="cpu", memory_fraction=0.8),
    )
    cached_sd = cached_provider.get_state_dict("model")
    assert isinstance(cached_sd, GpuCachedStateDict)
    assert cached_sd._max_cache_bytes == 800


def test_gpu_cached_provider_rejects_unavailable_cuda_backend() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment; unavailability test not applicable")

    base_provider = InMemoryStateDictProvider({}, max_io_workers=1)
    base_sd = base_provider.get_or_create_alias_state_dict("model")
    base_sd["x"] = torch.ones(1, dtype=torch.float32)

    cached_provider = wrap_provider_with_gpu_cache(
        base_provider,
        cache_config=GpuCacheConfig(device="cuda", max_cache_bytes=1024),
    )
    with pytest.raises(ProviderError, match="not available"):
        cached_provider.get_state_dict("model")


def test_gpu_cached_provider_debug_logs_timing_by_model_part(
    caplog: pytest.LogCaptureFixture,
) -> None:
    base_provider = InMemoryStateDictProvider({}, max_io_workers=1)
    base_sd = base_provider.get_or_create_alias_state_dict("model")
    base_sd["model.layers.0.mlp.experts.1.gate_proj.weight"] = torch.ones(32, dtype=torch.float32)
    base_sd["model.layers.0.self_attn.q_proj.weight"] = torch.ones(32, dtype=torch.float32)

    cached_provider = wrap_provider_with_gpu_cache(
        base_provider,
        cache_config=GpuCacheConfig(device="cpu", max_cache_bytes=1 << 20, debug=True),
    )
    cached_sd = cached_provider.get_state_dict("model")

    with caplog.at_level(logging.INFO, logger="brainsurgery"):
        _ = cached_sd["model.layers.0.mlp.experts.1.gate_proj.weight"]
        _ = cached_sd["model.layers.0.self_attn.q_proj.weight"]
        cached_sd.flush()

    messages = [record.getMessage() for record in caplog.records]
    assert any("timing-total timing_event=clone_local_device" in message for message in messages)
    assert any(
        "timing-top-parts timing_event=clone_local_device" in message
        and "part=model.layers.0.mlp.experts" in message
        for message in messages
    )
