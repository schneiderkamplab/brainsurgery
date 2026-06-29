from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from brainsurgery.serving.engine import Engine
from brainsurgery.serving.model.base import ModelConfig, ServingModel


# ---------------------------------------------------------------------------
# Mock model – deterministic, always predicts token 42
# ---------------------------------------------------------------------------

class _MockServingModel(ServingModel):
    def __init__(
        self,
        vocab_size: int = 100,
        num_layers: int = 2,
        num_heads: int = 4,
        head_dim: int = 32,
        hidden_dim: int = 128,
    ):
        self.config = ModelConfig(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_dim=hidden_dim,
        )
        self._backend = "codegen2-torch"
        self._paged_attention = False

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_kv: Any = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.config.vocab_size)
        logits[:, :, 42] = 1.0
        new_kv = tuple(
            (
                torch.zeros(batch, self.config.num_heads, seq_len, self.config.head_dim),
                torch.zeros(batch, self.config.num_heads, seq_len, self.config.head_dim),
            )
            for _ in range(self.config.num_layers)
        )
        return logits, new_kv


# ---------------------------------------------------------------------------
# Synchronous path  (step / run)
# ---------------------------------------------------------------------------

class TestMultiSequenceSync:
    def test_two_sequences(self):
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=4)
        id1 = engine.add_request([1, 2, 3], max_tokens=8, temperature=0.0)
        id2 = engine.add_request([4, 5], max_tokens=5, temperature=0.0)
        outputs = engine.run(max_steps=20)

        by_seq: dict[int, list[int]] = {}
        for o in outputs:
            by_seq.setdefault(o["seq_id"], []).append(o["token_id"])

        assert set(by_seq) == {id1, id2}
        assert len(by_seq[id1]) == 5
        assert len(by_seq[id2]) == 3
        for tid in by_seq[id1]:
            assert tid == 42
        for tid in by_seq[id2]:
            assert tid == 42

    def test_at_capacity_batch(self):
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=3)
        ids = [engine.add_request([1], max_tokens=3, temperature=0.0) for _ in range(3)]
        outputs = engine.run(max_steps=20)

        by_seq: dict[int, list[int]] = {}
        for o in outputs:
            by_seq.setdefault(o["seq_id"], []).append(o["token_id"])

        assert len(by_seq) == 3
        for seq_id in ids:
            assert len(by_seq[seq_id]) == 2

    def test_mixed_prompt_lengths(self):
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=4)
        id1 = engine.add_request([1] * 10, max_tokens=12, temperature=0.0)
        id2 = engine.add_request([2] * 50, max_tokens=52, temperature=0.0)
        id3 = engine.add_request([3] * 3, max_tokens=5, temperature=0.0)
        outputs = engine.run(max_steps=20)

        by_seq: dict[int, list[int]] = {}
        for o in outputs:
            by_seq.setdefault(o["seq_id"], []).append(o["token_id"])

        assert len(by_seq) == 3
        for seq_id in (id1, id2, id3):
            assert len(by_seq[seq_id]) == 2, f"seq {seq_id} expected 2 tokens"

    def test_sequences_exceed_batch_size(self):
        """More sequences than max_batch_size — scheduler drains them in waves."""
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=2)
        ids = [engine.add_request([1], max_tokens=2, temperature=0.0) for _ in range(5)]
        outputs = engine.run(max_steps=20)

        by_seq: dict[int, list[int]] = {}
        for o in outputs:
            by_seq.setdefault(o["seq_id"], []).append(o["token_id"])

        assert len(by_seq) == 5
        for seq_id in ids:
            assert len(by_seq[seq_id]) == 1

    def test_different_temperatures(self):
        """Non-zero temperature sequences should still all complete."""
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=4)
        id1 = engine.add_request([1], max_tokens=4, temperature=0.0)
        id2 = engine.add_request([2], max_tokens=4, temperature=0.5)
        id3 = engine.add_request([3], max_tokens=4, temperature=1.0)
        outputs = engine.run(max_steps=20)

        by_seq: dict[int, list[int]] = {}
        for o in outputs:
            by_seq.setdefault(o["seq_id"], []).append(o["token_id"])

        assert set(by_seq) == {id1, id2, id3}
        for seq_id in (id1, id2, id3):
            assert len(by_seq[seq_id]) == 3

    def test_empty_prompt(self):
        """A single-token 'prompt' (minimum) should still work."""
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=4)
        seq_id = engine.add_request([0], max_tokens=3, temperature=0.0)
        outputs = engine.run(max_steps=10)
        tokens = [o["token_id"] for o in outputs if o["seq_id"] == seq_id]
        assert len(tokens) == 2
        assert all(t == 42 for t in tokens)

    def test_batched_prefill_cache_position_correct(self):
        """After a batched prefill with different prompt lengths, the cached
        position must reflect the actual token count, not the padded length."""
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=4)
        short = engine.add_request([1], max_tokens=3, temperature=0.0)
        long = engine.add_request([2] * 20, max_tokens=22, temperature=0.0)

        outputs = engine.run(max_steps=20)

        by_seq: dict[int, list[int]] = {}
        for o in outputs:
            by_seq.setdefault(o["seq_id"], []).append(o["token_id"])

        assert len(by_seq[short]) == 2
        assert len(by_seq[long]) == 2

        short_idx = [i for i, o in enumerate(outputs) if o["seq_id"] == short]
        long_idx = [i for i, o in enumerate(outputs) if o["seq_id"] == long]
        assert short_idx[0] != long_idx[0], "should interleave or at least not clobber"

    def test_long_sequence_completes_within_max_steps(self):
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=2)
        seq_id = engine.add_request([1] * 10, max_tokens=20, temperature=0.0)
        outputs = engine.run(max_steps=30)
        tokens = [o["token_id"] for o in outputs if o["seq_id"] == seq_id]
        assert len(tokens) == 10

    def test_sequence_exactly_max_seq_len(self):
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=2, max_seq_len=16)
        prompt_len = 8
        seq_id = engine.add_request([1] * prompt_len, max_tokens=16, temperature=0.0)
        outputs = engine.run(max_steps=20)
        tokens = [o["token_id"] for o in outputs if o["seq_id"] == seq_id]
        assert len(tokens) == 8


# ---------------------------------------------------------------------------
# Background loop path  (async queues)
# ---------------------------------------------------------------------------

class TestMultiSequenceBackgroundLoop:
    @pytest.mark.asyncio
    async def test_two_sequences(self):
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=4)
        engine.start_background_loop()
        try:
            id1 = engine.add_request([1, 2, 3], max_tokens=6, temperature=0.0)
            id2 = engine.add_request([4, 5, 6, 7], max_tokens=6, temperature=0.0)

            tokens1, tokens2 = await asyncio.gather(
                _collect(engine, id1), _collect(engine, id2),
            )

            assert len(tokens1) == 3
            assert len(tokens2) == 2
            assert all(t == 42 for t in tokens1)
            assert all(t == 42 for t in tokens2)
        finally:
            engine.stop_background_loop()

    @pytest.mark.asyncio
    async def test_concurrent_collection(self):
        """Use asyncio.gather to collect tokens from two sequences concurrently."""
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=4)
        engine.start_background_loop()
        try:
            id1 = engine.add_request([1], max_tokens=6, temperature=0.0)
            id2 = engine.add_request([2], max_tokens=6, temperature=0.0)

            tokens1, tokens2 = await asyncio.gather(
                _collect(engine, id1), _collect(engine, id2),
            )
            assert len(tokens1) == 5
            assert len(tokens2) == 5
            assert tokens1 == [42] * 5
            assert tokens2 == [42] * 5
        finally:
            engine.stop_background_loop()

    @pytest.mark.asyncio
    async def test_batch_at_capacity(self):
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=3)
        engine.start_background_loop()
        try:
            ids = [engine.add_request([1], max_tokens=3, temperature=0.0) for _ in range(3)]
            results = await asyncio.gather(*[_collect(engine, sid) for sid in ids])
            for tokens in results:
                assert len(tokens) == 2
                assert tokens == [42] * 2
        finally:
            engine.stop_background_loop()

    @pytest.mark.asyncio
    async def test_more_sequences_than_batch_size(self):
        """Background loop should drain pending sequences across multiple rounds."""
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=2)
        engine.start_background_loop()
        try:
            ids = [engine.add_request([1], max_tokens=2, temperature=0.0) for _ in range(5)]
            results = await asyncio.gather(*[_collect(engine, sid) for sid in ids])
            for tokens in results:
                assert len(tokens) == 1, f"expected 1 token, got {len(tokens)}"
                assert tokens == [42]
        finally:
            engine.stop_background_loop()

    @pytest.mark.asyncio
    async def test_nonzero_temperature(self):
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=4)
        engine.start_background_loop()
        try:
            id1 = engine.add_request([1], max_tokens=4, temperature=0.8)
            id2 = engine.add_request([2], max_tokens=4, temperature=0.0)
            tokens1, tokens2 = await asyncio.gather(
                _collect(engine, id1), _collect(engine, id2),
            )
            assert len(tokens1) == 3
            assert len(tokens2) == 3
        finally:
            engine.stop_background_loop()

    @pytest.mark.asyncio
    async def test_queue_cleanup_after_finish(self):
        """After a sequence finishes, its queue should be removed from
        _token_queues."""
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=4)
        engine.start_background_loop()
        try:
            sid = engine.add_request([1], max_tokens=1, temperature=0.0)
            tokens = await _collect(engine, sid)
            assert len(tokens) == 1

            # Give the background loop a moment to clean up
            await asyncio.sleep(0.1)

            # The queue should be gone
            with engine._loop_lock:
                assert sid not in engine._token_queues
                assert sid not in engine._request_params
        finally:
            engine.stop_background_loop()


# ---------------------------------------------------------------------------
# Chunked prefill
# ---------------------------------------------------------------------------

class TestChunkedPrefill:
    def test_single_sequence_long_prompt(self):
        """A long prompt processed with chunked prefill produces the same
        number of generated tokens as without chunking."""
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=4, prefill_chunk_size=4)
        prompt = list(range(20))
        seq_id = engine.add_request(prompt, max_tokens=25, temperature=0.0)
        outputs = engine.run(max_steps=30)
        tokens = [o["token_id"] for o in outputs if o["seq_id"] == seq_id]
        assert len(tokens) == 5

    def test_decodes_interleave_with_prefill_chunks(self):
        """When a long-prompt prefill is chunked, decode sequences produce
        tokens before the prefill finishes."""
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=4, prefill_chunk_size=4)
        long = engine.add_request([1] * 20, max_tokens=22, temperature=0.0)
        short = engine.add_request([2], max_tokens=3, temperature=0.0)
        outputs = engine.run(max_steps=30)

        by_seq: dict[int, list[int]] = {}
        for o in outputs:
            by_seq.setdefault(o["seq_id"], []).append(o["token_id"])

        assert len(by_seq[long]) == 2
        assert len(by_seq[short]) == 2

        # Short seq should produce its first token before long finishes
        long_last_step = max(
            i for i, o in enumerate(outputs) if o["seq_id"] == long
        )
        short_first_step = min(
            i for i, o in enumerate(outputs) if o["seq_id"] == short
        )
        assert short_first_step < long_last_step, (
            "short decode should interleave before long prefill finishes"
        )

    def test_chunked_batch_with_waiting_sequences(self):
        """Sequences waiting in the queue get processed while a long prefill
        is chunking."""
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=2, prefill_chunk_size=4)
        long = engine.add_request([1] * 16, max_tokens=18, temperature=0.0)
        # Small delays between add_request calls so background loop can
        # interleave
        ids = [engine.add_request([i], max_tokens=2, temperature=0.0) for i in range(3)]
        outputs = engine.run(max_steps=30)

        by_seq: dict[int, list[int]] = {}
        for o in outputs:
            by_seq.setdefault(o["seq_id"], []).append(o["token_id"])

        assert len(by_seq) == 4
        for sid in ids:
            assert len(by_seq[sid]) == 1
        assert len(by_seq[long]) == 2

    def test_multiple_long_prompts(self):
        """Two long prompts chunked concurrently should both complete."""
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=4, prefill_chunk_size=8)
        id_a = engine.add_request([1] * 20, max_tokens=24, temperature=0.0)
        id_b = engine.add_request([2] * 30, max_tokens=34, temperature=0.0)
        outputs = engine.run(max_steps=40)

        by_seq: dict[int, list[int]] = {}
        for o in outputs:
            by_seq.setdefault(o["seq_id"], []).append(o["token_id"])

        assert len(by_seq[id_a]) == 4
        assert len(by_seq[id_b]) == 4

    @pytest.mark.asyncio
    async def test_background_loop_with_chunking(self):
        """Background loop with chunked prefill works correctly."""
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=4, prefill_chunk_size=4)
        engine.start_background_loop()
        try:
            long = engine.add_request([1] * 12, max_tokens=15, temperature=0.0)
            short = engine.add_request([2], max_tokens=3, temperature=0.0)

            tokens_long, tokens_short = await asyncio.gather(
                _collect(engine, long), _collect(engine, short),
            )
            assert len(tokens_long) == 3
            assert len(tokens_short) == 2
            assert tokens_long == [42] * 3
            assert tokens_short == [42] * 2
        finally:
            engine.stop_background_loop()

    def test_chunk_size_exceeds_prompt(self):
        """When chunk_size > prompt length, behaves like normal prefill."""
        model = _MockServingModel()
        engine = Engine(model, max_batch_size=4, prefill_chunk_size=128)
        seq_id = engine.add_request([1, 2, 3], max_tokens=5, temperature=0.0)
        outputs = engine.run(max_steps=10)
        tokens = [o["token_id"] for o in outputs if o["seq_id"] == seq_id]
        assert len(tokens) == 2


# ---------------------------------------------------------------------------
# Prefix caching tests
# ---------------------------------------------------------------------------

class TestPrefixCaching:
    """Prefix caching across multiple requests."""

    def _prefix_engine(self, **kw):
        model = _MockServingModel()
        model._backend = 'test-backend'
        return Engine(model, max_batch_size=4, block_size=4, cache_blocks=64, **kw)

    def test_identical_prompts(self):
        """Two identical prompts reuse all cached blocks."""
        engine = self._prefix_engine()
        prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 tokens, 2 full blocks + 2 partial

        a = engine.add_request(prompt, max_tokens=14, temperature=0.0)
        all_outputs = engine.run(max_steps=2)

        assert len(engine.cache._hash_to_block) == 2  # 2 full blocks registered
        table_a = engine.cache.get_block_table(a)
        assert len(table_a) == 3  # 10 tokens → 3 blocks

        b = engine.add_request(prompt, max_tokens=14, temperature=0.0)
        table_b = engine.cache.get_block_table(b)

        # First 2 blocks should be reused
        assert table_b[:2] == table_a[:2], f"{table_b[:2]} != {table_a[:2]}"
        assert engine.cache.get_position(b) == 8, f"position {engine.cache.get_position(b)} != 8"

        # Both complete correctly
        all_outputs.extend(engine.run(max_steps=20))
        by_seq: dict[int, list[int]] = {}
        for o in all_outputs:
            by_seq.setdefault(o["seq_id"], []).append(o["token_id"])
        assert len(by_seq[a]) == 4  # total=14, prompt=10, gen=4
        assert len(by_seq[b]) == 4
        assert all(t == 42 for t in by_seq[a])
        assert all(t == 42 for t in by_seq[b])

    def test_shared_prefix(self):
        """Two requests with a shared prefix reuse prefix blocks."""
        engine = self._prefix_engine()
        prompt_a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        prompt_b = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]

        a = engine.add_request(prompt_a, max_tokens=14, temperature=0.0)
        all_outputs = engine.run(max_steps=2)
        table_a = engine.cache.get_block_table(a)

        b = engine.add_request(prompt_b, max_tokens=14, temperature=0.0)
        table_b = engine.cache.get_block_table(b)

        assert table_b[:2] == table_a[:2]
        assert engine.cache.get_position(b) == 8

        all_outputs.extend(engine.run(max_steps=20))
        by_seq: dict[int, list[int]] = {}
        for o in all_outputs:
            by_seq.setdefault(o["seq_id"], []).append(o["token_id"])
        assert len(by_seq[a]) == 4
        assert len(by_seq[b]) == 4
        assert all(t == 42 for t in by_seq[a])
        assert all(t == 42 for t in by_seq[b])

    def test_prefix_caching_with_chunking(self):
        """Prefix caching works with chunked prefill."""
        engine = self._prefix_engine(prefill_chunk_size=6)
        prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        a = engine.add_request(prompt, max_tokens=14, temperature=0.0)
        all_outputs = engine.run(max_steps=4)
        table_a = engine.cache.get_block_table(a)

        b = engine.add_request(prompt, max_tokens=14, temperature=0.0)
        table_b = engine.cache.get_block_table(b)
        assert table_b[:2] == table_a[:2]
        assert engine.cache.get_position(b) == 8

        all_outputs.extend(engine.run(max_steps=20))
        by_seq: dict[int, list[int]] = {}
        for o in all_outputs:
            by_seq.setdefault(o["seq_id"], []).append(o["token_id"])
        assert len(by_seq[a]) == 4
        assert len(by_seq[b]) == 4
        assert all(t == 42 for t in by_seq[a])
        assert all(t == 42 for t in by_seq[b])

    @pytest.mark.asyncio
    async def test_prefix_caching_background_loop(self):
        """Prefix caching works with the background loop."""
        engine = self._prefix_engine()
        prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        a = engine.add_request(prompt, max_tokens=14, temperature=0.0)
        engine.run(max_steps=1)
        table_a = engine.cache.get_block_table(a)

        engine.start_background_loop()
        try:
            b = engine.add_request(prompt, max_tokens=14, temperature=0.0)
            table_b = engine.cache.get_block_table(b)
            assert table_b[:2] == table_a[:2]
            assert engine.cache.get_position(b) == 8

            tokens_a, tokens_b = await asyncio.gather(
                _collect(engine, a), _collect(engine, b),
            )
            assert len(tokens_a) == 3  # 1 token already consumed by run()
            assert len(tokens_b) == 4
            assert all(t == 42 for t in tokens_a)
            assert all(t == 42 for t in tokens_b)
        finally:
            engine.stop_background_loop()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def _collect(engine: Engine, seq_id: int) -> list[int]:
    tokens: list[int] = []
    while True:
        t = await engine.await_token(seq_id)
        if t is None:
            break
        tokens.append(t)
    return tokens
