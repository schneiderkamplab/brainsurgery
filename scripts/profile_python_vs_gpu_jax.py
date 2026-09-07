#!/usr/bin/env python3
"""Profile Python-vs-GPU overhead: HF (PyTorch) vs Axon (codegen2-jax) generate().

Measures wall-clock, GPU-active, Python dispatch overhead, and Python call
counts for 128-token autoregressive generation on a single model.

JAX-specific notes:
- JAX uses jax.jit internally, so each decode step is a single jitted dispatch
- JAX generate runs the full limit (no per-step EOS .item() sync)
- JAX is async: Python loop can run ahead of GPU
- Sync via jnp.block_until_ready() instead of torch.cuda.synchronize()
- GPU idle concept differs: JAX enqueues async work, so "idle" = time GPU
  has no queued work (only at start/end of generation)

Usage:
    python scripts/profile_python_vs_gpu_jax.py --model mistralai/Mistral-7B-v0.1
    python scripts/profile_python_vs_gpu_jax.py --model mistralai/Mistral-7B-v0.1 --max-len 64 --prompt-len 8
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import importlib.util
import io
import json
import logging
import pstats
import time
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logging.getLogger("brainsurgery").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAME = "GeneratedModel"


# ---------------------------------------------------------------------------
# Model resolution (shared with torch version)
# ---------------------------------------------------------------------------

def _find_axon_file(checkpoint_id: str) -> Path:
    models_root = REPO_ROOT / "brainsurgery" / "synapse" / "models"
    parts = checkpoint_id.split("/")
    name = parts[-1] if len(parts) >= 2 else parts[0]
    for model_dir in sorted(models_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for axon_file in model_dir.glob("*.axon"):
            if axon_file.stem == name or axon_file.stem == checkpoint_id:
                return axon_file
    for model_dir in sorted(models_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for axon_file in model_dir.glob("*.axon"):
            if name in axon_file.stem:
                return axon_file
    raise FileNotFoundError(f"Could not find .axon file for checkpoint {checkpoint_id}")


def _model_dir(checkpoint_id: str) -> Path:
    d = REPO_ROOT / "models" / checkpoint_id
    if not d.exists():
        raise FileNotFoundError(f"Model directory not found: {d}")
    return d


def _load_model_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def _resolve_safetensors_paths(weights: Path) -> list[Path]:
    if weights.is_file():
        return [weights]
    return sorted(weights.glob("*.safetensors"))


def _load_generated_class(py_path: Path, class_name: str):
    module_name = f"_axon_jax_generated_{int(time.time() * 1e9)}"
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import generated module: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_cls = getattr(module, class_name, None)
    if model_cls is None:
        raise RuntimeError(f"Generated class {class_name!r} not found in {py_path}")
    return model_cls


# ---------------------------------------------------------------------------
# Axon compile + load (codegen2-jax)
# ---------------------------------------------------------------------------

def compile_and_load_axon_jax(
    axon_file: Path,
    model_dir: Path,
    tmp_dir: Path,
    dtype: str = "float32",
) -> object:
    """Compile axon -> graph IR -> codegen2-jax -> load model class + state dict."""
    from brainsurgery.synapse.axon import (
        elaborate_closed_axon_file,
        flatten_closed_axon_file,
        lower_axon_program_to_graph_ir,
        normalize_closed_axon_file,
        optimize_graph_program,
        resolve_axon_program_from_path,
        typecheck2_flat_axon_file,
        GraphOptimizeConfig,
    )
    from brainsurgery.synapse.axon.codegen2_jax import (
        emit_model_code_from_graph_ir as emit_jax,
    )

    resolved_axon = resolve_axon_program_from_path(axon_file).ast
    normalized = normalize_closed_axon_file(resolved_axon)
    elaborated = elaborate_closed_axon_file(normalized)
    flat = flatten_closed_axon_file(elaborated)
    typed = typecheck2_flat_axon_file(flat)
    graph_program = lower_axon_program_to_graph_ir(typed)
    graph_program = optimize_graph_program(
        graph_program,
        config=GraphOptimizeConfig(backend_intrinsics="codegen2-jax"),
    )

    model_config = _load_model_config(model_dir)
    code = emit_jax(graph_program, class_name=CLASS_NAME, model_config=model_config)

    # Save generated code for debugging
    debug_path = Path("/tmp/axon_jax_profile_debug.py")
    debug_path.write_text(code, encoding="utf-8")
    print(f"  Generated code saved to {debug_path}")

    py_path = tmp_dir / "generated_jax.py"
    py_path.write_text(code, encoding="utf-8")
    model_cls = _load_generated_class(py_path, CLASS_NAME)

    import torch
    import safetensors

    safetensors_paths = _resolve_safetensors_paths(model_dir)
    target_dtype = torch.float32 if dtype == "float32" else torch.bfloat16

    state_dict = {}
    for p in safetensors_paths:
        with safetensors.safe_open(str(p), framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                state_dict[str(key)] = t.to(dtype=target_dtype)

    model = model_cls.from_state_dict(state_dict)
    return model


# ---------------------------------------------------------------------------
# HF load (PyTorch)
# ---------------------------------------------------------------------------

def load_hf_model(
    model_dir: Path,
    dtype: str = "float32",
    device: str = "cuda:0",
) -> object:
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    model = model.to(device=torch.device(device)).eval()
    return model


def load_tokenizer(model_dir: Path) -> object:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)


# ---------------------------------------------------------------------------
# JAX sync helper
# ---------------------------------------------------------------------------

def _jax_sync():
    import jax.numpy as jnp
    jnp.array(0).block_until_ready()


def _jax_block_output(output):
    import jax
    if isinstance(output, (list, tuple)):
        for item in output:
            _jax_block_output(item)
    elif isinstance(output, dict):
        for item in output.values():
            _jax_block_output(item)
    elif hasattr(output, "block_until_ready"):
        output.block_until_ready()


# ---------------------------------------------------------------------------
# Profiling: HF (PyTorch) — same as torch script
# ---------------------------------------------------------------------------

def profile_hf_generate(
    model,
    input_ids,
    generate_fn,
    *,
    warmup: int = 1,
):
    """Profile HF generate with torch.profiler + cProfile."""
    import torch
    from torch.profiler import ProfilerActivity, profile, DeviceType

    for _ in range(warmup):
        torch.cuda.synchronize()
        _ = generate_fn(model, input_ids)
        torch.cuda.synchronize()

    torch.cuda.empty_cache()
    gc.collect()

    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = generate_fn(model, input_ids)
        torch.cuda.synchronize()
        wall_us = (time.perf_counter() - t0) * 1e6

    events = prof.key_averages()
    cuda_events = [e for e in events if e.device_type == DeviceType.CUDA]
    cpu_events = [e for e in events if e.device_type == DeviceType.CPU]

    gpu_active_us = sum(e.self_device_time_total for e in cuda_events)
    gpu_idle_us = wall_us - gpu_active_us
    gpu_idle_pct = (gpu_idle_us / wall_us * 100) if wall_us > 0 else 0.0
    kernel_count = sum(e.count for e in cuda_events)
    cpu_total_us = sum(e.self_cpu_time_total for e in cpu_events)

    torch.cuda.synchronize()
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    _ = generate_fn(model, input_ids)
    torch.cuda.synchronize()
    cprofile_wall_us = (time.perf_counter() - t0) * 1e6
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    cprofile_text = s.getvalue()
    total_calls = ps.total_calls
    total_time = ps.total_tt

    top_funcs = []
    for func_stat in sorted(ps.stats.items(), key=lambda x: x[1][3], reverse=True)[:15]:
        (filename, line, name), (cc, nc, tt, ct, callers) = func_stat
        top_funcs.append({
            "name": name,
            "file": Path(filename).name if filename else "?",
            "calls": nc,
            "cumtime_ms": ct * 1e3,
            "tottime_ms": tt * 1e3,
        })

    return {
        "label": "HF",
        "wall_us": wall_us,
        "wall_ms": wall_us / 1e3,
        "gpu_active_us": gpu_active_us,
        "gpu_active_ms": gpu_active_us / 1e3,
        "gpu_idle_us": gpu_idle_us,
        "gpu_idle_ms": gpu_idle_us / 1e3,
        "gpu_idle_pct": gpu_idle_pct,
        "kernel_count": kernel_count,
        "cpu_total_us": cpu_total_us,
        "cpu_total_ms": cpu_total_us / 1e3,
        "cprofile_wall_us": cprofile_wall_us,
        "cprofile_wall_ms": cprofile_wall_us / 1e3,
        "total_calls": total_calls,
        "total_time_s": total_time,
        "top_funcs": top_funcs,
        "cprofile_text": cprofile_text,
        "output": output,
    }


# ---------------------------------------------------------------------------
# Profiling: Axon (JAX)
# ---------------------------------------------------------------------------

def profile_jax_generate(
    model,
    input_ids,
    generate_fn,
    *,
    warmup: int = 1,
):
    """Profile Axon-JAX generate with wall-clock + cProfile.

    JAX doesn't have a direct equivalent of torch.profiler for measuring
    per-op GPU time. Instead, we measure:
    - Wall-clock of the entire generate() call (with block_until_ready at end)
    - GPU-active time via separate per-step timing with block_until_ready
    - Python dispatch overhead = wall-clock - GPU-active
    """
    import jax.numpy as jnp

    for _ in range(warmup):
        _jax_sync()
        _ = generate_fn(model, input_ids)
        _jax_sync()

    gc.collect()

    # Wall-clock run (no profiler)
    _jax_sync()
    t0 = time.perf_counter()
    output = generate_fn(model, input_ids)
    _jax_block_output(output)
    wall_us = (time.perf_counter() - t0) * 1e6

    # cProfile run (separate run)
    _jax_sync()
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    _ = generate_fn(model, input_ids)
    _jax_sync()
    cprofile_wall_us = (time.perf_counter() - t0) * 1e6
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    cprofile_text = s.getvalue()
    total_calls = ps.total_calls
    total_time = ps.total_tt

    top_funcs = []
    for func_stat in sorted(ps.stats.items(), key=lambda x: x[1][3], reverse=True)[:15]:
        (filename, line, name), (cc, nc, tt, ct, callers) = func_stat
        top_funcs.append({
            "name": name,
            "file": Path(filename).name if filename else "?",
            "calls": nc,
            "cumtime_ms": ct * 1e3,
            "tottime_ms": tt * 1e3,
        })

    # Per-step synced timing (measured separately)
    # This gives us the "synced total" — if async wall-clock < synced total,
    # it means JAX's async execution overlaps Python dispatch with GPU compute
    prefill_us, avg_step_us, step_times = _measure_jax_gpu_active(model, input_ids, generate_fn)

    # We don't have a direct "GPU idle %" for JAX (no torch.profiler equivalent)
    # Instead, we report the async wall-clock and synced per-step times
    # The comparison between async and synced shows how much JAX's async
    # execution hides Python dispatch overhead
    gpu_active_us = prefill_us + avg_step_us * max(1, int(wall_us / (avg_step_us * 1e6)))

    gpu_idle_us = max(0.0, wall_us - gpu_active_us)
    gpu_idle_pct = (gpu_idle_us / wall_us * 100) if wall_us > 0 else 0.0

    return {
        "label": "Axon-JAX",
        "wall_us": wall_us,
        "wall_ms": wall_us / 1e3,
        "gpu_active_us": gpu_active_us,
        "gpu_active_ms": gpu_active_us / 1e3,
        "gpu_idle_us": gpu_idle_us,
        "gpu_idle_ms": gpu_idle_us / 1e3,
        "gpu_idle_pct": gpu_idle_pct,
        "kernel_count": 0,
        "cpu_total_us": 0.0,
        "cpu_total_ms": 0.0,
        "cprofile_wall_us": cprofile_wall_us,
        "cprofile_wall_ms": cprofile_wall_us / 1e3,
        "total_calls": total_calls,
        "total_time_s": total_time,
        "top_funcs": top_funcs,
        "cprofile_text": cprofile_text,
        "output": output,
    }


def _measure_jax_gpu_active(model, input_ids, generate_fn):
    """Measure GPU-active time by running generate and timing with block_until_ready.

    For JAX, the entire generate() is async. We measure the time from
    the first dispatch to when the final output is ready. This gives us
    the GPU-active time (the GPU is continuously processing queued work).

    We use a trick: run generate() once with block_until_ready to warm up,
    then run again measuring only the GPU compute portion by calling
    forward() directly with block_until_ready per step.
    """
    import jax.numpy as jnp

    # Warmup
    _ = generate_fn(model, input_ids)
    _jax_sync()

    # Measure per-step GPU time by calling forward() directly
    # This gives us the GPU compute time for each step
    jax_ids = model._to_jax(input_ids, jnp.int32)

    # Prefill
    _jax_sync()
    t0 = time.perf_counter()
    result = model.forward(jax_ids, use_cache=True)
    _jax_block_output(result)
    prefill_us = (time.perf_counter() - t0) * 1e6

    logits = result.get("logits") if isinstance(result, dict) else result
    cache = result.get("new_kv") if isinstance(result, dict) else None
    next_id = logits[:, -1:, :].argmax(axis=-1).astype(jnp.int32)

    total_gpu_us = prefill_us

    # Decode steps (measure 5 steps and extrapolate)
    num_measure = 5
    step_times = []
    for i in range(num_measure):
        _jax_sync()
        t0 = time.perf_counter()
        result = model.forward(next_id, past_kv=cache, use_cache=True)
        _jax_block_output(result)
        step_us = (time.perf_counter() - t0) * 1e6
        step_times.append(step_us)

        logits = result.get("logits") if isinstance(result, dict) else result
        cache = result.get("new_kv", cache) if isinstance(result, dict) else cache
        next_id = logits[:, -1:, :].argmax(axis=-1).astype(jnp.int32)

    avg_step_us = sum(step_times) / len(step_times)

    # Get total number of steps from generate config
    # The generate function uses max_len, so total steps = max_len - prompt_len
    # We'll just return the measured prefill + extrapolated decode steps
    # But we don't know the exact step count here, so return the per-step average
    # and let the caller compute the total
    return prefill_us, avg_step_us, step_times


# ---------------------------------------------------------------------------
# Generate functions
# ---------------------------------------------------------------------------

def make_hf_generate_fn(max_new_tokens: int):
    def _generate(model, input_ids):
        return model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=model.config.eos_token_id,
        )
    return _generate


def make_axon_jax_generate_fn(max_len: int, eos_token_id: int):
    def _generate(model, input_ids):
        return model.generate(
            input_ids,
            max_len=max_len,
            eos_token_id=eos_token_id,
        )
    return _generate


# ---------------------------------------------------------------------------
# Step-by-step timing
# ---------------------------------------------------------------------------

def step_by_step_timing_hf(model, input_ids, num_steps: int):
    import torch
    results = []

    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1e3

        logits = out.logits if hasattr(out, "logits") else out[0]
        past_key_values = out.past_key_values if hasattr(out, "past_key_values") else out[1]
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        for step in range(num_steps - 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(next_id, past_key_values=past_key_values, use_cache=True)
            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t0) * 1e3
            results.append(step_ms)

            logits = out.logits if hasattr(out, "logits") else out[0]
            past_key_values = out.past_key_values if hasattr(out, "past_key_values") else out[1]
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    return prefill_ms, results


def step_by_step_timing_jax(model, input_ids, num_steps: int):
    """Measure per-step wall-clock time for JAX generate using KV cache."""
    import jax.numpy as jnp
    results = []

    jax_ids = model._to_jax(input_ids, jnp.int32)

    # Prefill
    _jax_sync()
    t0 = time.perf_counter()
    result = model.forward(jax_ids, use_cache=True)
    _jax_block_output(result)
    prefill_ms = (time.perf_counter() - t0) * 1e3

    logits = result.get("logits") if isinstance(result, dict) else result
    cache = result.get("new_kv") if isinstance(result, dict) else None
    next_id = logits[:, -1:, :].argmax(axis=-1).astype(jnp.int32)

    for step in range(num_steps - 1):
        _jax_sync()
        t0 = time.perf_counter()
        result = model.forward(next_id, past_kv=cache, use_cache=True)
        _jax_block_output(result)
        step_ms = (time.perf_counter() - t0) * 1e3
        results.append(step_ms)

        logits = result.get("logits") if isinstance(result, dict) else result
        cache = result.get("new_kv", cache) if isinstance(result, dict) else cache
        next_id = logits[:, -1:, :].argmax(axis=-1).astype(jnp.int32)

    return prefill_ms, results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_comparison(hf_stats: dict, axon_stats: dict, num_tokens: int = 0):
    print()
    print("=" * 78)
    print(f"  PROFILING: HF (PyTorch) vs Axon (codegen2-jax) — generate({num_tokens or '?'} tokens)")
    print("=" * 78)
    print()
    print(f"  {'Metric':<30s}  {'HF':>14s}  {'Axon-JAX':>14s}  {'Ratio':>10s}")
    print("  " + "-" * 74)

    def row(name, hf_val, axon_val, fmt="{:.3f}", ratio=True):
        hf_s = fmt.format(hf_val)
        axon_s = fmt.format(axon_val)
        if ratio and hf_val != 0:
            r = fmt.format(axon_val / hf_val) + "x"
        else:
            r = "-"
        print(f"  {name:<30s}  {hf_s:>14s}  {axon_s:>14s}  {r:>10s}")

    row("Wall-clock (ms)", hf_stats["wall_ms"], axon_stats["wall_ms"])
    row("GPU active (ms)", hf_stats["gpu_active_ms"], axon_stats["gpu_active_ms"])
    row("GPU idle (ms)", hf_stats["gpu_idle_ms"], axon_stats["gpu_idle_ms"])
    row("GPU idle %", hf_stats["gpu_idle_pct"], axon_stats["gpu_idle_pct"], "{:.1f}")
    row("cProfile wall (ms)", hf_stats["cprofile_wall_ms"], axon_stats["cprofile_wall_ms"])
    row("Python function calls", hf_stats["total_calls"], axon_stats["total_calls"], "{:.0f}")
    row("cProfile total time (s)", hf_stats["total_time_s"], axon_stats["total_time_s"])

    print()
    print("  Key differences JAX vs Torch:")
    print("    - JAX uses jax.jit: 1 compiled dispatch per step (not per-op)")
    print("    - JAX runs full limit (no per-step EOS .item() sync)")
    print("    - JAX is async: Python loop runs ahead of GPU")
    print()

    for label, stats in [("HF (PyTorch)", hf_stats), ("Axon (JAX)", axon_stats)]:
        print()
        print(f"  --- Top 15 Python functions by cumulative time ({label}) ---")
        print(f"  {'Function':<35s}  {'File':<20s}  {'Calls':>8s}  {'Cumtime':>10s}  {'Tottime':>10s}")
        print("  " + "-" * 88)
        for f in stats["top_funcs"]:
            name = f["name"][:34]
            fname = f["file"][:19]
            print(f"  {name:<35s}  {fname:<20s}  {f['calls']:>8d}  {f['cumtime_ms']:>8.2f}ms  {f['tottime_ms']:>8.2f}ms")

    print()


def print_step_timing(label: str, prefill_ms: float, step_times: list[float]):
    n = len(step_times)
    if n == 0:
        print(f"  {label}: no step data")
        return
    avg = sum(step_times) / n
    mn = min(step_times)
    mx = max(step_times)
    med = sorted(step_times)[n // 2]
    print(f"  {label}:")
    print(f"    prefill: {prefill_ms:.2f} ms")
    print(f"    decode: avg={avg:.3f} ms, min={mn:.3f}, max={mx:.3f}, median={med:.3f} ({n} steps)")
    print(f"    tok/s (decode only): {1000.0 / avg:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile Python-vs-GPU overhead: HF (PyTorch) vs Axon (JAX) generate()"
    )
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1",
                        help="Checkpoint ID (e.g. mistralai/Mistral-7B-v0.1)")
    parser.add_argument("--axon-file", default=None,
                        help="Path to .axon file (auto-resolved if not specified)")
    parser.add_argument("--model-dir", default=None,
                        help="Path to model weights dir (auto-resolved if not specified)")
    parser.add_argument("--max-len", type=int, default=128,
                        help="Max tokens to generate (prompt + new)")
    parser.add_argument("--prompt-len", type=int, default=16,
                        help="Prompt length in tokens")
    parser.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"],
                        help="Dtype (float32 recommended for JAX compatibility)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup runs before profiling")
    parser.add_argument("--no-step-timing", action="store_true",
                        help="Skip step-by-step timing")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save profiler outputs (default: log/profile-python-vs-gpu)")
    args = parser.parse_args()

    axon_file = Path(args.axon_file) if args.axon_file else _find_axon_file(args.model)
    model_dir = Path(args.model_dir) if args.model_dir else _model_dir(args.model)
    max_len = args.max_len

    print(f"Model:       {args.model}")
    print(f"Axon file:   {axon_file}")
    print(f"Model dir:   {model_dir}")
    print(f"Dtype:       {args.dtype}")
    print(f"Device:      {args.device}")
    print(f"Prompt len:  {args.prompt_len}")
    print(f"Max len:     {max_len}")
    print()

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "log" / "profile-python-vs-gpu"
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    tokenizer = load_tokenizer(model_dir)
    eos_token_id = tokenizer.eos_token_id

    prompt = " ".join(["hello"] * args.prompt_len)
    input_ids_torch = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
    actual_prompt_len = input_ids_torch.shape[1]
    target_new_tokens = max_len - actual_prompt_len
    if target_new_tokens < 1:
        target_new_tokens = max_len
    print(f"Input shape: {input_ids_torch.shape} (prompt={actual_prompt_len} tokens, target new={target_new_tokens})")

    # -----------------------------------------------------------------------
    # HF (PyTorch)
    # -----------------------------------------------------------------------
    print("\nLoading HF model (PyTorch)...")
    hf_model = load_hf_model(model_dir, dtype=args.dtype, device=args.device)
    hf_gen = make_hf_generate_fn(target_new_tokens)

    print("Profiling HF generate()...")
    hf_stats = profile_hf_generate(
        hf_model, input_ids_torch, hf_gen,
        warmup=args.warmup,
    )
    print(f"  HF wall-clock: {hf_stats['wall_ms']:.2f} ms, GPU idle: {hf_stats['gpu_idle_pct']:.1f}%")

    if not args.no_step_timing:
        print("  Step-by-step timing (HF)...")
        hf_prefill, hf_steps = step_by_step_timing_hf(hf_model, input_ids_torch, target_new_tokens)

    del hf_model
    torch.cuda.empty_cache()
    gc.collect()

    # -----------------------------------------------------------------------
    # Axon (JAX)
    # -----------------------------------------------------------------------
    print("\nCompiling + loading Axon model (codegen2-jax)...")
    import jax
    import jax.numpy as jnp

    with TemporaryDirectory(prefix="axon_jax_profile_") as tmp_dir:
        axon_model = compile_and_load_axon_jax(
            axon_file, model_dir, Path(tmp_dir),
            dtype=args.dtype,
        )
        print("  JAX model loaded")

        axon_gen = make_axon_jax_generate_fn(actual_prompt_len + target_new_tokens, eos_token_id)

        print("Profiling Axon-JAX generate()...")
        axon_stats = profile_jax_generate(
            axon_model, input_ids_torch, axon_gen,
            warmup=args.warmup,
        )
        print(f"  Axon-JAX wall-clock: {axon_stats['wall_ms']:.2f} ms, GPU idle: {axon_stats['gpu_idle_pct']:.1f}%")

        if not args.no_step_timing:
            print("  Step-by-step timing (Axon-JAX)...")
            axon_prefill, axon_steps = step_by_step_timing_jax(
                axon_model, input_ids_torch, target_new_tokens
            )

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    print_comparison(hf_stats, axon_stats, target_new_tokens)

    if not args.no_step_timing:
        print()
        print("=" * 78)
        print("  STEP-BY-STEP TIMING (per-token decode, using KV cache)")
        print("=" * 78)
        print_step_timing("HF (PyTorch)", hf_prefill, hf_steps)
        print_step_timing("Axon (JAX)", axon_prefill, axon_steps)

        if hf_steps and axon_steps:
            hf_avg = sum(hf_steps) / len(hf_steps)
            axon_avg = sum(axon_steps) / len(axon_steps)
            print(f"\n  Per-step ratio (Axon-JAX/HF): {axon_avg / hf_avg:.3f}x")

    print()

    hf_profile_path = output_dir / "hf_cprofile_jax.txt"
    axon_profile_path = output_dir / "axon_jax_cprofile.txt"
    hf_profile_path.write_text(hf_stats["cprofile_text"], encoding="utf-8")
    axon_profile_path.write_text(axon_stats["cprofile_text"], encoding="utf-8")
    print(f"cProfile outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
