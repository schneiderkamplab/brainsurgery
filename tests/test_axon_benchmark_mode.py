import pytest

from brainsurgery.synapse.axon_test import (
    _resolve_benchmark_mode,
    _should_generate_for_benchmark,
)


def test_benchmark_mode_auto_preserves_current_task_defaults() -> None:
    assert _should_generate_for_benchmark(
        model_task="causal_lm",
        benchmark_mode=_resolve_benchmark_mode("auto"),
    )
    assert not _should_generate_for_benchmark(
        model_task="seq2seq_lm",
        benchmark_mode=_resolve_benchmark_mode("auto"),
    )
    assert not _should_generate_for_benchmark(
        model_task="masked_lm",
        benchmark_mode=_resolve_benchmark_mode("auto"),
    )


def test_benchmark_mode_generate_supports_seq2seq_but_not_encoder_only() -> None:
    assert _should_generate_for_benchmark(
        model_task="seq2seq_lm",
        benchmark_mode=_resolve_benchmark_mode("generate"),
    )
    with pytest.raises(ValueError, match="encoder-only"):
        _should_generate_for_benchmark(
            model_task="masked_lm",
            benchmark_mode=_resolve_benchmark_mode("generate"),
        )


def test_benchmark_mode_forward_overrides_causal_generation() -> None
    assert not _should_generate_for_benchmark(
        model_task="causal_lm",
        benchmark_mode=_resolve_benchmark_mode("forward"),
    )
