from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ModelConfig:
    max_seq_len: int = 2048
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    vocab_size: int = 50257
    dtype: str = "float32"
    hidden_dim: int = 768
    extra: dict[str, Any] = field(default_factory=dict)


CacheState = Any


class ServingModel(ABC):
    config: ModelConfig

    @abstractmethod
    def forward(
        self,
        input_ids: Any,
        *,
        past_kv: CacheState | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, CacheState]:
        ...

    def sample(
        self,
        logits: Any,
        temperature: float = 0.0,
        top_p: float = 1.0,
        prefill: bool = True,
    ) -> int:
        last = logits[0, -1, :] if prefill else logits[0, 0, :]
        if temperature > 0.0:
            return self._sample_categorical(last, temperature, top_p)
        return int(last.argmax().item())

    def _sample_categorical(self, logits_1d: torch.Tensor, temperature: float, top_p: float) -> int:
        if top_p < 1.0:
            return int(_top_p_sample(logits_1d, temperature, top_p).item())
        return int(torch.multinomial(torch.softmax(logits_1d / temperature, dim=-1), 1).item())


def _top_p_sample(logits_1d: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    probs = torch.softmax(logits_1d / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    idx = int(torch.multinomial(sorted_probs / sorted_probs.sum(), 1).item())
    return sorted_indices[idx]
