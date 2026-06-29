# brainsurgery serve

Online inference engine for compiled Axon models with dynamic batching and
PagedAttention. Supports PyTorch (CPU/CUDA/MPS), MLX (Apple Silicon), and
Tinygrad backends.

## Usage

```bash
# Start an OpenAI-compatible HTTP server
brainsurgery serve http <model.axon> <weights_dir>

# Runs on port 8000 by default; override with --port
brainsurgery serve http <model.axon> <weights_dir> --port 9000
```

### Backends

| `--backend`         | Device              | Notes                        |
|---------------------|---------------------|------------------------------|
| `codegen2-torch`    | `--device cpu`      | Default                      |
| `codegen2-torch`    | `--device cuda`     | NVIDIA GPU                   |
| `codegen2-torch`    | `--device mps`      | Apple Silicon via PyTorch    |
| `codegen2-mlx`      | `--device cpu`      | Apple Silicon via Metal MLX  |
| `codegen2-tinygrad` | `--device cpu`      | Portable CPU backend         |

The `--device cpu` argument is accepted by all backends but has no effect on
MLX (which always uses Metal on Apple Silicon).

### Paged attention

Enabled by default. Disable with `--no-paged-attention`:

```bash
brainsurgery serve http <model.axon> <weights_dir> --no-paged-attention
```

### Models

Weights are loaded from a Hugging Face-style directory containing
`model.safetensors` and `config.json`. Axon model definitions live in
`brainsurgery/synapse/models/<arch>/<variant>.axon` (prepackaged, stable
grammar) or `examples/*.axon` (bleeding-edge, may require newer grammar):

```bash
# GPT-2 (PyTorch CPU)
brainsurgery serve http brainsurgery/synapse/models/gpt2/gpt2.axon models/gpt2 \
  --backend codegen2-torch --device cpu

# GPT-2 (MLX / Apple Silicon Metal)
brainsurgery serve http brainsurgery/synapse/models/gpt2/gpt2.axon models/gpt2 \
  --backend codegen2-mlx

# GPT-2 with graph optimization and paged attention
brainsurgery serve http brainsurgery/synapse/models/gpt2/gpt2.axon models/gpt2 \
  --backend codegen2-torch --optimize-graph --paged-attention
```

### One-shot CLI mode (no HTTP server)

```bash
brainsurgery serve serve <model.axon> <weights_dir> \
  --prompt "Hello" --max-tokens 16
```

## API

The HTTP server exposes an OpenAI-compatible API:

```
POST /v1/completions          Text completions (streaming via SSE)
POST /v1/chat/completions     Chat completions (streaming via SSE)
GET  /v1/models               List available model
GET  /health                  Engine status
```

### Examples

Start the server (default port 8000):

```bash
brainsurgery serve http brainsurgery/synapse/models/gpt2/gpt2.axon models/gpt2 \
  --backend codegen2-torch --device cpu
```

#### Text completions (sync)

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The meaning of life is",
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

#### Text completions (streaming SSE)

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 64,
    "stream": true
  }'
```

#### Chat completions (sync)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 32,
    "temperature": 0.0
  }'
```

#### Chat completions (streaming SSE)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Write a short poem about AI."}
    ],
    "max_tokens": 64,
    "stream": true
  }'
```

#### Health check

```bash
curl http://localhost:8000/health
```

## Architecture

```
HTTP Request  ──→  add_request()  ──→  Scheduler (waiting queue)
                                             │
                                    Background Loop Thread
                                             │
                              schedule() → execute_plan() → on_step_complete()
                                             │
                                      per-seqid token queues
                                             │
HTTP Handler  ←──  await_token()  ←─────────┘
```

The server runs a single background thread that continuously processes batches
from the scheduler. Each HTTP request registers its prompt, then reads
generated tokens from its own thread-safe queue without holding any global
lock. Multiple requests are automatically batched together by the scheduler.
