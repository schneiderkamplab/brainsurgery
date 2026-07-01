from __future__ import annotations

import logging
from pathlib import Path

import safetensors
import typer

from .model.axon import AxonServingModel

app = typer.Typer(help="Serve compiled Axon models.")
logger = logging.getLogger("brainsurgery.serving")

_SAFETENSORS_DTYPE_MAP = {
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}


def _resolve_dtype(weights: Path, dtype: str) -> str:
    if dtype != "auto":
        return dtype
    safetensors_paths = sorted(weights.glob("*.safetensors"))
    if not safetensors_paths:
        logger.warning("No .safetensors files found in %s, falling back to float32", weights)
        return "float32"
    for path in safetensors_paths:
        try:
            with safetensors.safe_open(str(path), framework="pt") as f:
                for key in f.keys():
                    raw = f.get_slice(key).get_dtype()
                    mapped = _SAFETENSORS_DTYPE_MAP.get(raw)
                    if mapped is not None:
                        logger.info("Detected dtype=%s from %s", mapped, path.name)
                        return mapped
        except Exception:
            continue
    logger.warning("Could not detect dtype from safetensors files in %s, falling back to float32", weights)
    return "float32"


@app.command("serve")
def serve_command(
    axon_file: Path = typer.Argument(
        ...,
        help="Path to the .axon model definition file.",
    ),
    weights: Path = typer.Argument(
        ...,
        help="Path to the weights directory (safetensors files).",
    ),
    backend: str = typer.Option(
        "codegen2-torch",
        "--backend",
        help="Target backend: codegen2-torch, codegen2-mlx, or codegen2-tinygrad.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Device to run the model on (cpu, mps, cuda).",
    ),
    dtype: str = typer.Option(
        "auto",
        "--dtype",
        help="Floating-point dtype for model weights (default: auto-detected from safetensors files in weights path).",
    ),
    max_batch_size: int = typer.Option(
        8,
        "--max-batch-size",
        help="Maximum number of sequences in a single batch.",
    ),
    max_seq_len: int = typer.Option(
        2048,
        "--max-seq-len",
        help="Maximum sequence length.",
    ),
    block_size: int = typer.Option(
        16,
        "--block-size",
        help="KV cache block size (tokens per block).",
    ),
    cache_blocks: int = typer.Option(
        1024,
        "--cache-blocks",
        help="Total number of KV cache blocks.",
    ),
    optimize_graph: bool = typer.Option(
        False,
        "--optimize-graph",
        help="Enable graph-level IR optimization.",
    ),
    backend_intrinsics: str | None = typer.Option(
        None,
        "--backend-intrinsics",
        help="Enable backend-specific intrinsics (e.g. __mlx_sdpa,__mlx_rope).",
    ),
    paged_attention: bool = typer.Option(
        True,
        "--paged-attention/--no-paged-attention",
        help="Enable paged attention for memory-efficient KV cache.",
    ),
    compile_mode: str | None = typer.Option(
        None,
        "--compile",
        help="Apply torch.compile/mx.compile/TinyJit to serving model.",
    ),
    prompt: str = typer.Option(
        "Hello, world!",
        "--prompt",
        help="Prompt text for the model.",
    ),
    max_tokens: int = typer.Option(
        32,
        "--max-tokens",
        help="Maximum tokens to generate.",
    ),
) -> None:
    dtype = _resolve_dtype(weights, dtype)
    logger.info(
        "Loading model: axon=%s weights=%s backend=%s device=%s dtype=%s",
        axon_file, weights, backend, device, dtype,
    )
    model = AxonServingModel.load(
        axon_file,
        weights,
        backend=backend,
        device=device,
        dtype=dtype,
        optimize_graph=optimize_graph,
        graph_backend_intrinsics=backend_intrinsics,
        paged_attention=paged_attention,
        compile_mode=compile_mode,
    )
    logger.info("Model loaded successfully.")

    from .engine import Engine

    engine = Engine(
        model,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        block_size=block_size,
        cache_blocks=cache_blocks,
        device=device,
        dtype=dtype,
        compile_mode=compile_mode,
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(weights))
    engine.set_tokenizer(tokenizer)
    logger.info("Engine ready. Processing prompt: %s", prompt)
    seq_id = engine.add_request(prompt, max_tokens=max_tokens)
    outputs = engine.run(max_steps=max_tokens + 1)
    tokens = [o["token_id"] for o in outputs if o["seq_id"] == seq_id]
    logger.info("Generated %d tokens for seq %d: %s", len(tokens), seq_id, tokens)


@app.command("http", help="Start an OpenAI-compatible HTTP API server.")
def http_command(
    axon_file: Path = typer.Argument(
        ...,
        help="Path to the .axon model definition file.",
    ),
    weights: Path = typer.Argument(
        ...,
        help="Path to the weights directory (safetensors files).",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind the server to."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind the server to."),
    backend: str = typer.Option(
        "codegen2-torch",
        "--backend",
        help="Target backend: codegen2-torch, codegen2-mlx, or codegen2-tinygrad.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Device to run the model on (cpu, mps, cuda).",
    ),
    dtype: str = typer.Option(
        "auto",
        "--dtype",
        help="Floating-point dtype for model weights (default: auto-detected from safetensors files in weights path).",
    ),
    max_batch_size: int = typer.Option(
        8,
        "--max-batch-size",
        help="Maximum number of sequences in a single batch.",
    ),
    max_seq_len: int = typer.Option(
        2048,
        "--max-seq-len",
        help="Maximum sequence length.",
    ),
    block_size: int = typer.Option(
        16,
        "--block-size",
        help="KV cache block size (tokens per block).",
    ),
    cache_blocks: int = typer.Option(
        1024,
        "--cache-blocks",
        help="Total number of KV cache blocks.",
    ),
    optimize_graph: bool = typer.Option(
        False,
        "--optimize-graph",
        help="Enable graph-level IR optimization.",
    ),
    backend_intrinsics: str | None = typer.Option(
        None,
        "--backend-intrinsics",
        help="Enable backend-specific intrinsics (e.g. __mlx_sdpa,__mlx_rope).",
    ),
    paged_attention: bool = typer.Option(
        True,
        "--paged-attention/--no-paged-attention",
        help="Enable paged attention for memory-efficient KV cache.",
    ),
    compile_mode: str | None = typer.Option(
        None,
        "--compile",
        help="Apply torch.compile/mx.compile/TinyJit to serving model.",
    ),
) -> None:
    import uvicorn

    dtype = _resolve_dtype(weights, dtype)
    logger.info(
        "Loading model: axon=%s weights=%s backend=%s device=%s dtype=%s",
        axon_file, weights, backend, device, dtype,
    )
    model = AxonServingModel.load(
        axon_file,
        weights,
        backend=backend,
        device=device,
        dtype=dtype,
        optimize_graph=optimize_graph,
        graph_backend_intrinsics=backend_intrinsics,
        paged_attention=paged_attention,
        compile_mode=compile_mode,
    )
    logger.info("Model loaded successfully.")

    from transformers import AutoTokenizer

    from .engine import Engine

    engine = Engine(
        model,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        block_size=block_size,
        cache_blocks=cache_blocks,
        device=device,
        dtype=dtype,
        compile_mode=compile_mode,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(weights))
    engine.set_tokenizer(tokenizer)
    logger.info("Engine ready. Starting HTTP server on %s:%s", host, port)

    from .server import create_app

    app = create_app(engine)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
