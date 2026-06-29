from __future__ import annotations

import logging
import threading
from typing import Any

from .base import BatchPlan, Phase, Scheduler, SeqInput, Sequence

logger = logging.getLogger("brainsurgery.serving.scheduler")


class ContinuousBatchScheduler(Scheduler):
    def __init__(
        self,
        *,
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
        prefill_chunk_size: int = 0,
    ):
        self._prefill_chunk_size = prefill_chunk_size
        self._max_batch_size = max_batch_size
        self._max_seq_len = max_seq_len
        self._sequences: dict[int, Sequence] = {}
        self._waiting: list[Sequence] = []
        self._running: dict[int, Sequence] = {}
        self._finished: list[Sequence] = []
        self._next_seq_id: int = 0
        self._lock = threading.Lock()

    def add(
        self,
        prompt: list[int],
        *,
        max_tokens: int = 32,
        **kwargs: Any,
    ) -> int:
        with self._lock:
            seq = Sequence(
                seq_id=self._next_seq_id,
                prompt=prompt,
                max_tokens=max_tokens,
                eos_token_id=kwargs.get("eos_token_id"),
            )
            self._next_seq_id += 1
            self._sequences[seq.seq_id] = seq
            self._waiting.append(seq)
        logger.debug("Added seq %d: %d prompt tokens", seq.seq_id, len(prompt))
        return seq.seq_id

    def schedule(self) -> BatchPlan:
        with self._lock:
            avail = self._max_batch_size - len(self._running)
            while self._waiting and avail > 0:
                seq = self._waiting.pop(0)
                self._running[seq.seq_id] = seq
                avail -= 1
                logger.debug("Scheduled seq %d (prefill)", seq.seq_id)

            planned: list[SeqInput] = []
            for seq in self._running.values():
                if seq.finished:
                    continue
                if seq.phase == Phase.PREFILL:
                    chunk_size = len(seq.prompt) - seq.prefill_pos
                    if self._prefill_chunk_size > 0:
                        chunk_size = min(chunk_size, self._prefill_chunk_size)
                    chunk = seq.prompt[seq.prefill_pos:seq.prefill_pos + chunk_size]
                    is_last = seq.prefill_pos + chunk_size >= len(seq.prompt)
                    seq._pending_prefill_tokens = chunk_size
                    planned.append(SeqInput(
                        seq_id=seq.seq_id,
                        input_ids=chunk,
                        phase=Phase.PREFILL,
                        num_tokens=chunk_size,
                        is_last_prefill_chunk=is_last,
                    ))
                else:
                    planned.append(SeqInput(
                        seq_id=seq.seq_id,
                        input_ids=[seq.sampled_token] if seq.sampled_token is not None else [seq.prompt[-1]],
                        phase=Phase.DECODE,
                        num_tokens=1,
                    ))
        return BatchPlan(sequences=planned)

    def on_step_complete(self, seq_id: int, token_id: int) -> None:
        with self._lock:
            seq = self._running.get(seq_id)
            if seq is None:
                return
            seq.generated.append(token_id)
            seq.sampled_token = token_id
            if seq.phase == Phase.PREFILL:
                seq.prefill_pos += seq._pending_prefill_tokens
                seq._pending_prefill_tokens = 0
                seq.phase = Phase.DECODE
                logger.debug("Seq %d transitioned to decode", seq_id)
            full_len = len(seq.prompt) + len(seq.generated)
            if token_id == seq.eos_token_id or full_len >= seq.max_tokens:
                seq.finished = True
                self._finish(seq_id)

    def advance_prefill(self, seq_id: int, num_tokens: int) -> None:
        with self._lock:
            seq = self._running.get(seq_id)
            if seq is not None and seq.phase == Phase.PREFILL:
                seq.prefill_pos += num_tokens

    def set_prefill_pos(self, seq_id: int, pos: int) -> None:
        with self._lock:
            seq = self._sequences.get(seq_id)
            if seq is not None:
                seq.prefill_pos = pos

    def _finish(self, seq_id: int) -> None:
        seq = self._running.pop(seq_id, None)
        if seq is not None:
            self._finished.append(seq)
            logger.debug("Seq %d finished: %d generated tokens", seq_id, len(seq.generated))

    def finished_count(self) -> int:
        return len(self._finished)

    def pending_count(self) -> int:
        return len(self._waiting)

    def running_count(self) -> int:
        return len(self._running)

    def is_running(self, seq_id: int) -> bool:
        with self._lock:
            return seq_id in self._running
