from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class Phase(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclass
class Sequence:
    seq_id: int
    prompt: list[int]
    generated: list[int] = field(default_factory=list)
    phase: Phase = Phase.PREFILL
    max_tokens: int = 32
    eos_token_id: int | None = None
    finished: bool = False
    sampled_token: int | None = None


@dataclass
class SeqInput:
    seq_id: int
    input_ids: Any
    phase: Phase
    num_tokens: int


@dataclass
class BatchPlan:
    sequences: list[SeqInput]


class Scheduler(ABC):
    @abstractmethod
    def add(self, prompt: list[int], *, max_tokens: int = 32, **kwargs: Any) -> int:
        ...

    @abstractmethod
    def schedule(self) -> BatchPlan:
        ...

    @abstractmethod
    def on_step_complete(self, seq_id: int, token_id: int) -> None:
        ...

    @abstractmethod
    def finished_count(self) -> int:
        ...

    @abstractmethod
    def pending_count(self) -> int:
        ...

    @abstractmethod
    def running_count(self) -> int:
        ...

    @abstractmethod
    def is_running(self, seq_id: int) -> bool:
        ...
