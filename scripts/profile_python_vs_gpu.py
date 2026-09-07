#!/usr/bin/env python3
"""Profile Python-vs-GPU overhead: HF vs Axon (codegen2-torch) generate().

Measures wall-clock, GPU-active, GPU-idle, kernel count, and Python call
counts for 128-token autoregressive generation on a single model.

Usage:
    python scripts/profile_python_vs_gpu.py --model mistralai/Mistral-7B-v0.1
    python scripts/profile_python_vs_gpu.py --model mistralai/Mistral-7B-v0.1 --compile
    python scripts/profile_python_vs_gpu.py --model mistralai/Mistral-7B-v0.1 --max-len 64 --prompt-len 8
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
import sys
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
# Model resolution
# ---------------------------------------------------------------------------

def _find_axon_file(checkpoint_id: str) -> Path:
    """Resolve .axon file for a checkpoint by scanning model dirs."""
    models_root = REPO_ROOT / "brainsurgery" / "synapse" / "models"
    # Try direct match: e.g. mistralai/Mistral-7B-v0.1 -> mistral/Mistral-7B-v0.1.axon
    parts = checkpoint_id.split("/")
    if len(parts) >= 2:
        family = parts[0]
        name = parts[1]
    else:
        family = ""
        name = parts[0]
    # Common family aliases
    family_aliases = {
        "meta-llama": "llama3",
        "mistralai": "mistral",
        "google": "gemma",
        "gemma2": "gemma2",
    }
    axon_family = family_aliases.get(family, family)
    # Search all model dirs for matching .axon file
    for model_dir in sorted(models_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for axon_file in model_dir.glob("*.axon"):
            stem = axon_file.stem
            if stem == name or stem == checkpoint_id:
                return axon_file
    # Fallback: search by stem containing the name
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


# ---------------------------------------------------------------------------
# Axon compile + load (codegen2-torch only)
# ---------------------------------------------------------------------------

def _load_generated_class(py_path: Path, class_name: str):
    module_name = f"_axon_generated_{int(time.time() * 1e9)}"
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import generated module: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_cls = getattr(module, class_name, None)
    if model_cls is None:
        raise RuntimeError(f"Generated class {class_name!r} not found in {py_path}")
    return model_cls


def compile_and_load_axon(
    axon_file: Path,
    model_dir: Path,
    tmp_dir: Path,
    dtype: str = "bfloat16",
    device: str = "cuda:0",
) -> object:
    """Compile axon -> graph IR -> codegen2-torch -> load model class + state dict."""
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
    from brainsurgery.synapse.axon.codegen2_torch import (
        emit_model_code_from_graph_ir as emit_torch,
    )

    resolved_axon = resolve_axon_program_from_path(axon_file).ast
    normalized = normalize_closed_axon_file(resolved_axon)
    elaborated = elaborate_closed_axon_file(normalized)
    flat = flatten_closed_axon_file(elaborated)
    typed = typecheck2_flat_axon_file(flat)
    graph_program = lower_axon_program_to_graph_ir(typed)
    graph_program = optimize_graph_program(
        graph_program,
        config=GraphOptimizeConfig(backend_intrinsics="codegen2-torch"),
    )

    model_config = _load_model_config(model_dir)
    code = emit_torch(graph_program, class_name=CLASS_NAME, model_config=model_config)

    py_path = tmp_dir / "generated_torch.py"
    py_path.write_text(code, encoding="utf-8")
    model_cls = _load_generated_class(py_path, CLASS_NAME)

    import torch
    target_device = torch.device(device)
    target_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    safetensors_paths = _resolve_safetensors_paths(model_dir)
    state_dict = {}
    for p in safetensors_paths:
        with __import__("safetensors").safe_open(str(p), framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                state_dict[str(key)] = t.to(device=target_device, dtype=target_dtype)

    model = model_cls.from_state_dict(state_dict).to(target_device).eval()
    return model


# ---------------------------------------------------------------------------
# HF load
# ---------------------------------------------------------------------------

def load_hf_model(
    model_dir: Path,
    dtype: str = "bfloat16",
    device: str = "cuda:0",
) -> object:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
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
# Profiling
# ---------------------------------------------------------------------------

def profile_generate(
    label: str,
    model: object,
    input_ids,
    generate_fn,
    *,
    warmup: int = 1,
):
    """Profile a generate() call with torch.profiler + cProfile.

    Returns dict with wall-clock, GPU-active, GPU-idle, kernel count,
    cProfile top functions.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile, DeviceType

    # Warmup (not profiled)
    for _ in range(warmup):
        torch.cuda.synchronize()
        _ = generate_fn(model, input_ids)
        torch.cuda.synchronize()

    # Clear cache between warmup and profiled run
    torch.cuda.empty_cache()
    gc.collect()

    # Profiled run: torch.profiler
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

    # Extract profiler stats
    events = prof.key_averages()
    cuda_events = [e for e in events if e.device_type == DeviceType.CUDA]
    cpu_events = [e for e in events if e.device_type == DeviceType.CPU]

    gpu_active_us = sum(e.self_device_time_total for e in cuda_events)
    gpu_idle_us = wall_us - gpu_active_us
    gpu_idle_pct = (gpu_idle_us / wall_us * 100) if wall_us > 0 else 0.0
    kernel_count = sum(e.count for e in cuda_events)
    cpu_total_us = sum(e.self_cpu_time_total for e in cpu_events)

    # cProfile run (separate run to avoid profiler overhead interference)
    torch.cuda.synchronize()
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    _ = generate_fn(model, input_ids)
    torch.cuda.synchronize()
    cprofile_wall_us = (time.perf_counter() - t0) * 1e6
    pr.disable()

    # Parse cProfile stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    cprofile_text = s.getvalue()

    # Extract total function calls
    total_calls = ps.total_calls
    total_time = ps.total_tt

    # Top functions by cumulative time (parse from stats)
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
        "label": label,
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


def make_axon_generate_fn(max_len: int, eos_token_id: int):
    def _generate(model, input_ids):
        return model.generate(
            input_ids,
            max_len=max_len,
            eos_token_id=eos_token_id,
        )
    return _generate


# ---------------------------------------------------------------------------
# Step-by-step timing (per-token breakdown)
# ---------------------------------------------------------------------------

def step_by_step_timing_hf(model, input_ids, num_steps: int):
    """Measure per-step wall-clock time for HF generate using past_key_values."""
    import torch
    results = []

    with torch.no_grad():
        # Prefill
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


def step_by_step_timing_axon(model, input_ids, num_steps: int, eos_token_id: int):
    """Measure per-step wall-clock time for Axon generate using KV cache."""
    import torch
    results = []

    with torch.no_grad():
        # Prefill
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = model._forward(input_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1e3

        logits = result.get("logits") if isinstance(result, dict) else result
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        cache = result.get("past_key_values", None) if isinstance(result, dict) else None

        for step in range(num_steps - 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = model._forward(next_id, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t0) * 1e3
            results.append(step_ms)

            logits = result.get("logits") if isinstance(result, dict) else result
            cache = result.get("past_key_values", cache) if isinstance(result, dict) else cache
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    return prefill_ms, results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_comparison(hf_stats: dict, axon_stats: dict, num_tokens: int = 0):
    print()
    print("=" * 78)
    print(f"  PROFILING RESULTS: HF vs Axon (codegen2-torch) — generate({num_tokens or '?'} tokens)")
    print("=" * 78)
    print()
    print(f"  {'Metric':<30s}  {'HF':>14s}  {'Axon':>14s}  {'Ratio':>10s}")
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
    row("CUDA kernel count", hf_stats["kernel_count"], axon_stats["kernel_count"], "{:.0f}")
    row("CPU self time (ms)", hf_stats["cpu_total_ms"], axon_stats["cpu_total_ms"])
    row("cProfile wall (ms)", hf_stats["cprofile_wall_ms"], axon_stats["cprofile_wall_ms"])
    row("Python function calls", hf_stats["total_calls"], axon_stats["total_calls"], "{:.0f}")
    row("cProfile total time (s)", hf_stats["total_time_s"], axon_stats["total_time_s"])

    print()
    print("  Key insight: GPU idle % = fraction of wall-clock where GPU waited for Python.")
    print("  Lower GPU idle % = better Python dispatch efficiency.")
    print()

    # Top functions comparison
    for label, stats in [("HF", hf_stats), ("Axon", axon_stats)]:
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
        description="Profile Python-vs-GPU overhead: HF vs Axon generate()"
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
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for both HF and Axon")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup runs before profiling")
    parser.add_argument("--no-step-timing", action="store_true",
                        help="Skip step-by-step timing")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save profiler traces (default: log/profile-python-vs-gpu)")
    args = parser.parse_args()

    # Resolve paths
    axon_file = Path(args.axon_file) if args.axon_file else _find_axon_file(args.model)
    model_dir = Path(args.model_dir) if args.model_dir else _model_dir(args.model)
    max_len = args.max_len

    print(f"Model:       {args.model}")
    print(f"Axon file:   {axon_file}")
    print(f"Model dir:   {model_dir}")
    print(f"Dtype:       {args.dtype}")
    print(f"Device:      {args.device}")
    print(f"Compile:     {args.compile}")
    print(f"Prompt len:  {args.prompt_len}")
    print(f"Max len:     {max_len}")
    print()

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "log" / "profile-python-vs-gpu"
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    # Load tokenizer
    tokenizer = load_tokenizer(model_dir)
    eos_token_id = tokenizer.eos_token_id

    # Create input
    prompt = " ".join(["hello"] * args.prompt_len)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
    actual_prompt_len = input_ids.shape[1]
    target_new_tokens = max_len - actual_prompt_len
    if target_new_tokens < 1:
        target_new_tokens = max_len
    print(f"Input shape: {input_ids.shape} (prompt={actual_prompt_len} tokens, target new={target_new_tokens})")

    # -----------------------------------------------------------------------
    # HF
    # -----------------------------------------------------------------------
    print("\nLoading HF model...")
    hf_model = load_hf_model(model_dir, dtype=args.dtype, device=args.device)
    if args.compile:
        print("Compiling HF model with torch.compile...")
        hf_model = torch.compile(hf_model)

    hf_gen = make_hf_generate_fn(target_new_tokens)

    print("Profiling HF generate()...")
    hf_stats = profile_generate(
        "HF", hf_model, input_ids, hf_gen,
        warmup=args.warmup,
    )
    print(f"  HF wall-clock: {hf_stats['wall_ms']:.2f} ms, GPU idle: {hf_stats['gpu_idle_pct']:.1f}%")

    if not args.no_step_timing:
        print("  Step-by-step timing (HF)...")
        hf_prefill, hf_steps = step_by_step_timing_hf(hf_model, input_ids, target_new_tokens)

    # Free HF model
    del hf_model
    torch.cuda.empty_cache()
    gc.collect()

    # -----------------------------------------------------------------------
    # Axon
    # -----------------------------------------------------------------------
    print("\nCompiling + loading Axon model (codegen2-torch)...")
    with TemporaryDirectory(prefix="axon_profile_") as tmp_dir:
        axon_model = compile_and_load_axon(
            axon_file, model_dir, Path(tmp_dir),
            dtype=args.dtype, device=args.device,
        )
        if args.compile:
            print("Compiling Axon model with torch.compile...")
            axon_model = torch.compile(axon_model)

        axon_gen = make_axon_generate_fn(actual_prompt_len + target_new_tokens, eos_token_id)

        print("Profiling Axon generate()...")
        axon_stats = profile_generate(
            "Axon", axon_model, input_ids, axon_gen,
            warmup=args.warmup,
        )
        print(f"  Axon wall-clock: {axon_stats['wall_ms']:.2f} ms, GPU idle: {axon_stats['gpu_idle_pct']:.1f}%")

        if not args.no_step_timing:
            print("  Step-by-step timing (Axon)...")
            axon_prefill, axon_steps = step_by_step_timing_axon(
                axon_model, input_ids, target_new_tokens, eos_token_id
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
        print_step_timing("HF", hf_prefill, hf_steps)
        print_step_timing("Axon", axon_prefill, axon_steps)

        if hf_steps and axon_steps:
            hf_avg = sum(hf_steps) / len(hf_steps)
            axon_avg = sum(axon_steps) / len(axon_steps)
            print(f"\n  Per-step ratio (Axon/HF): {axon_avg / hf_avg:.3f}x")

    print()

    # Save cProfile text outputs
    hf_profile_path = output_dir / "hf_cprofile.txt"
    axon_profile_path = output_dir / "axon_cprofile.txt"
    hf_profile_path.write_text(hf_stats["cprofile_text"], encoding="utf-8")
    axon_profile_path.write_text(axon_stats["cprofile_text"], encoding="utf-8")
    print(f"cProfile outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
