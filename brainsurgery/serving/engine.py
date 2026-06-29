from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Any

import torch

from .cache.base import KVCache
from .cache.paged import TorchPagedKVCache
from .cache.mlx_paged import MLXPagedKVCache
from .cache.tinygrad_paged import TinygradPagedKVCache
from .model.base import CacheState, ServingModel
from .scheduler.base import BatchPlan, Phase, Scheduler
from .scheduler.continuous import ContinuousBatchScheduler

logger = logging.getLogger("brainsurgery.serving.engine")


class Engine:
    def __init__(
        self,
        model: ServingModel,
        *,
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
        block_size: int = 16,
        cache_blocks: int = 1024,
        device: str = "cpu",
        dtype: str = "float32",
    ):
        self._model = model
        self._device = torch.device(device)
        self._dtype = getattr(torch, dtype, torch.float32)
        self._device_str = device
        self._dtype_str = dtype
        self._backend = getattr(model, '_backend', 'codegen2-torch')
        cfg = model.config
        self._scheduler = ContinuousBatchScheduler(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
        self._cache = self._create_cache(cfg, block_size, cache_blocks, dtype)
        self._tokenizer: Any = None

        # Concurrent serving state
        self._token_queues: dict[int, queue.Queue] = {}
        self._request_params: dict[int, dict[str, float]] = {}
        self._loop_lock = threading.Lock()
        self._loop_event = threading.Event()
        self._stop_event = threading.Event()
        self._loop_thread: threading.Thread | None = None

    def _create_cache(self, cfg: Any, block_size: int, cache_blocks: int, dtype: str) -> KVCache:
        if self._backend == 'codegen2-mlx':
            return MLXPagedKVCache(
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                head_dim=cfg.head_dim,
                block_size=block_size,
                max_blocks=cache_blocks,
                dtype=dtype,
            )
        if self._backend == 'codegen2-tinygrad':
            return TinygradPagedKVCache(
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                head_dim=cfg.head_dim,
                block_size=block_size,
                max_blocks=cache_blocks,
                dtype=dtype,
            )
        return TorchPagedKVCache(
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            block_size=block_size,
            max_blocks=cache_blocks,
            dtype=self._dtype,
            device=self._device,
        )

    def _make_input_tensor(self, tokens: list[int]) -> Any:
        if self._backend == 'codegen2-mlx':
            import mlx.core as mx
            return mx.array([tokens], dtype=mx.int32)
        if self._backend == 'codegen2-tinygrad':
            from tinygrad import Tensor, dtypes as tg_dtypes
            return Tensor([tokens]).cast(tg_dtypes.int64)
        return torch.tensor([tokens], dtype=torch.long, device=self._device)

    def set_tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    # --- Background inference loop ---

    def start_background_loop(self) -> None:
        self._stop_event.clear()
        self._loop_thread = threading.Thread(target=self._run_background_loop, daemon=True)
        self._loop_thread.start()

    def stop_background_loop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        self._loop_event.set()
        if self._loop_thread:
            self._loop_thread.join(timeout=timeout)
            self._loop_thread = None

    def _run_background_loop(self) -> None:
        if self._backend == "codegen2-mlx":
            import mlx.core as mx
            mx.set_default_device(mx.gpu)
            mx.default_stream(mx.gpu)
        while not self._stop_event.is_set():
            with self._loop_lock:
                plan = self._scheduler.schedule()
            if not plan.sequences:
                self._loop_event.wait(timeout=0.05)
                self._loop_event.clear()
                continue

            outputs = self._execute_plan(plan)

            with self._loop_lock:
                for o in outputs:
                    seq_id = o['seq_id']
                    token_id = o['token_id']
                    self._scheduler.on_step_complete(seq_id, token_id)
                    q = self._token_queues.get(seq_id)
                    if q is not None:
                        q.put(token_id)
                for seq_id in list(self._token_queues.keys()):
                    if not self._scheduler.is_running(seq_id):
                        q = self._token_queues.pop(seq_id, None)
                        if q is not None:
                            q.put(None)
                        self._request_params.pop(seq_id, None)

    def _ensure_tokenizer(self, prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, str):
            if self._tokenizer is None:
                raise RuntimeError("Tokenizer required for string prompts.")
            return self._tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        return prompt

    def add_request(
        self,
        prompt: str | list[int],
        *,
        max_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> int:
        prompt_ids = self._ensure_tokenizer(prompt)
        with self._loop_lock:
            seq_id = self._scheduler.add(prompt_ids, max_tokens=max_tokens, **kwargs)
            self._cache.init_entry(seq_id)
            self._token_queues[seq_id] = queue.Queue()
            self._request_params[seq_id] = {"temperature": temperature, "top_p": top_p}
        self._loop_event.set()
        return seq_id

    async def await_token(self, seq_id: int) -> int | None:
        """Wait for the next token from the background loop. Returns None when finished."""
        loop = asyncio.get_event_loop()
        while True:
            q = self._token_queues.get(seq_id)
            if q is None:
                return None
            token_id = await loop.run_in_executor(None, q.get)
            return token_id

    # --- Synchronous step (CLI mode, no background loop) ---

    def step(self, temperature: float = 0.0, top_p: float = 1.0) -> list[dict[str, Any]]:
        plan = self._scheduler.schedule()
        if not plan.sequences:
            return []

        outputs = self._execute_plan(plan, temperature=temperature, top_p=top_p)

        for o in outputs:
            self._scheduler.on_step_complete(o['seq_id'], o['token_id'])

        return outputs

    def _execute_plan(
        self,
        plan: BatchPlan,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        is_paged = getattr(self._model, '_paged_attention', False)
        for seq_input in plan.sequences:
            seq_id = seq_input.seq_id
            tokens = seq_input.input_ids
            input_tensor = self._make_input_tensor(tokens)

            if temperature is None or top_p is None:
                with self._loop_lock:
                    params = self._request_params.get(seq_id, {"temperature": 0.0, "top_p": 1.0})
                temp = params["temperature"]
                tp = params["top_p"]
            else:
                temp = temperature
                tp = top_p

            if is_paged:
                block_table = self._cache.get_block_table(seq_id)
                position = self._cache.get_position(seq_id)
                num_tokens = len(tokens)
                num_blocks_needed = (position + num_tokens + self._cache.block_size - 1) // self._cache.block_size
                while len(block_table) < num_blocks_needed:
                    blk = self._cache._alloc_block()
                    if blk is None:
                        raise RuntimeError("Cache full")
                    block_table.append(blk)
                forward_kwargs = dict(
                    k_blocks=self._cache.k_blocks,
                    v_blocks=self._cache.v_blocks,
                    block_table=block_table,
                    position=position,
                    block_size=self._cache.block_size,
                )
            else:
                past_kv = self._cache.gather(seq_id)
                forward_kwargs = dict(past_kv=past_kv, use_cache=True)

            logits, new_kv = self._model.forward(input_tensor, **forward_kwargs)

            if new_kv is not None and seq_input.phase == Phase.PREFILL:
                self._store_prefill_cache(seq_id, new_kv)
            if seq_input.phase == Phase.DECODE and new_kv is not None:
                self._store_decode_cache(seq_id, new_kv)

            prefill = seq_input.phase == Phase.PREFILL
            next_token = self._model.sample(
                logits,
                temperature=temp,
                top_p=tp,
                prefill=prefill,
            )
            outputs.append({"seq_id": seq_id, "token_id": next_token})

        return outputs

    def _store_prefill_cache(self, seq_id: int, new_kv: CacheState) -> None:
        for layer_idx, (k, v) in enumerate(new_kv):
            self._cache.append_layer_tokens(
                seq_id, layer_idx,
                k[0], v[0],
            )
        self._cache.advance_tokens(seq_id, new_kv[0][0].shape[2])

    def _store_decode_cache(self, seq_id: int, new_kv: CacheState) -> None:
        for layer_idx, (k, v) in enumerate(new_kv):
            self._cache.append_layer_tokens(
                seq_id, layer_idx,
                k[0, :, -1:, :], v[0, :, -1:, :],
            )
        self._cache.advance_tokens(seq_id, 1)

    def run(self, max_steps: int = 512) -> list[dict[str, Any]]:
        all_outputs: list[dict[str, Any]] = []
        for _ in range(max_steps):
            outputs = self.step()
            all_outputs.extend(outputs)
            if self._scheduler.pending_count() == 0 and self._scheduler.running_count() == 0:
                break
        return all_outputs

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    @property
    def cache(self) -> KVCache:
        return self._cache
