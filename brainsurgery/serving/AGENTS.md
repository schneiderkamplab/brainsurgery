# Serving Module (`brainsurgery/serving/`)

Online server for compiled Axon models with dynamic batching and PagedAttention.

## Package Structure

```
serving/
├── __init__.py            # exports Engine, ServingModel, ModelConfig
├── engine.py              # main loop: scheduler → model → cache
├── cli.py                 # `brainsurgery serve serve|http <axon> <weights>` CLI
├── model/
│   ├── __init__.py
│   ├── base.py            # ServingModel ABC + ModelConfig dataclass
│   └── axon.py            # compiles .axon, loads weights, wraps _forward
├── cache/
│   ├── __init__.py
│   ├── base.py            # KVCache ABC
│   └── paged.py           # PagedKVCache (block-based allocation)
├── scheduler/
│   ├── __init__.py
│   ├── base.py            # Scheduler ABC + Phase, Sequence, BatchPlan
│   └── continuous.py      # ContinuousBatchScheduler
└── server/
    ├── __init__.py
    └── app.py             # FastAPI server with OpenAI-compatible API
```

## Architecture

```
Engine
 ├─ ContinuousBatchScheduler  — manages request queue, prefill→decode transitions
 ├─ PagedKVCache              — block-based KV cache (gather/scatter between model and blocks)
 ├─ AxonServingModel          — wraps compiled model, handles backend conversion
 └─ Background Loop Thread    — runs step() continuously, pushes tokens per-seqid
         ↓                           queues
    HTTP Server (async)
      ├─ POST /v1/completions        reads token queues
      └─ POST /v1/chat/completions   streams tokens via SSE
```

### Concurrent Serving

The HTTP server uses a background inference loop instead of per-request locks:

1. `start_background_loop()` launches a daemon thread running `_run_background_loop()`
2. The loop calls `scheduler.schedule()` then `_execute_plan()` (model forward + sample)
3. Output tokens are pushed into per-request `queue.Queue` instances
4. Finished sequences get a `None` sentinel; the queue is then removed
5. HTTP handlers call `await_token(seq_id)` which does `loop.run_in_executor(None, q.get)`
6. The scheduler is protected by `threading.Lock` — `add()` from the async thread and
   `schedule()`/`on_step_complete()` from the background thread are serialized
7. The CLI (`brainsurgery serve serve`) does NOT start the background loop — it uses
   `step()` and `run()` synchronously as before

Thread safety:
- `Scheduler` has its own `threading.Lock` for `add()`/`schedule()`/`on_step_complete()`
- `Engine._loop_lock` protects `_token_queues`, `_request_params`, and scheduler access
- Cache state for running sequences is only accessed from the background thread
- Token queues (`queue.Queue`) are thread-safe by design

## Key Abstractions

- **`ServingModel`**: `forward(input_ids, *, past_kv, use_cache) → (logits, new_kv)`.
  - `sample(logits, temperature, top_p, prefill) → int` — backend-native token sampling.
- **`KVCache`**: cache for KV tensors.
  - Non-paged: `gather(seq_id)` → list of `(k_tensor, v_tensor)` tuples.
  - Paged: block-based allocation with `k_blocks`/`v_blocks` properties for model gather.
  - Three implementations: `TorchPagedKVCache`, `MLXPagedKVCache`, `TinygradPagedKVCache`.
  - For paged paths, the model returns sliced `new_kv` (new tokens only); the engine stores
    them via `cache.append_layer_tokens()` which writes directly to the pool with
    original-axis indexing (avoids strided-view mutation issues).
- **`ContinuousBatchScheduler`**: `waiting → prefill → decode → finished`.
- **`Engine`**: main loop calls `scheduler.schedule()`, runs forward, stores cache, samples token.
  - Constructor accepts `device: str` and `dtype: str` (backend-agnostic).

## Backend Support

All six configurations verified with identical output for GPT-2:

| Backend | Non-paged | Paged |
|---------|-----------|-------|
| `codegen2-torch` | ✅ Verified | ✅ Verified |
| `codegen2-mlx`   | ✅ Verified | ✅ Verified |
| `codegen2-tinygrad` | ✅ Verified (cpu) | ✅ Verified (cpu) |

## Usage

```bash
# Non-HTTP inference (CLI with paged attention default)
brainsurgery serve serve <model.axon> <weights_dir> \
  --backend codegen2-torch --device cpu \
  --prompt "Hello" --max-tokens 16

# HTTP API server (OpenAI-compatible, paged attention by default)
brainsurgery serve http <model.axon> <weights_dir> \
  --backend codegen2-mlx --device cpu \
  --port 8000

# Disable paged attention
brainsurgery serve http <model.axon> <weights_dir> \
  --backend codegen2-torch --no-paged-attention
```

MLX/Metal works: `--backend codegen2-mlx --device cpu` (the "cpu" device arg is
a no-op — MLX always uses Metal on Apple Silicon). Paged attention is now the
default for CLI usage (`--paged-attention/--no-paged-attention`).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/models` | List available models |
| POST | `/v1/completions` | Text completions (streaming via SSE) |
| POST | `/v1/chat/completions` | Chat completions (streaming via SSE) |
| GET | `/health` | Engine status (running/pending counts) |

## Sampling

Backend-native `sample()` dispatches:

- **torch**: `torch.softmax` + `torch.multinomial` (CPU/GPU)
- **MLX**: `mx.softmax` + `mx.random.categorical` (Metal)
- **tinygrad**: `Tensor.softmax` + `Tensor.multinomial`

Supports temperature and top-p (nucleus) sampling on all three.

## Phases

- **Phase 1 (done)**: Core engine, paged cache, continuous batching, CLI, all 3 backends.
- **Phase 2 (done)**: Paged attention compiler intrinsics, block pre-allocation, e2e verified.
- **Phase 3 (done)**: HTTP server with OpenAI-compatible API + SSE streaming.
- **Phase 4 (done)**: MLX paged fix, tinygrad e2e, `TinygradPagedKVCache`.
- **Phase 5 (done)**: Concurrent request handling via background inference loop
  + per-request token queues. Removed global `asyncio.Lock` from HTTP server.
  Scheduler made thread-safe with `threading.Lock`.

## Known Issues

- **Tinygrad on Metal**: Tinygrad's GPU compilation may fail on Metal for
  complex graphs; CPU mode works reliably.
- **CPU inference speed**: GPT-2 12-layer model forward can be slow on CPU
  (30-60s per step). This is a pre-existing performance limitation, not a
  regression from the concurrent architecture.
