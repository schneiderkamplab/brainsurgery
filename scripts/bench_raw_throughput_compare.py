#!/usr/bin/env python3
"""Raw model throughput: codegen2-torch vs codegen2-mlx vs codegen2-triton.

Uses the direct Axon codegen pipeline (no serving layer).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logging.getLogger("brainsurgery").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parent.parent

if "--gemma3" in sys.argv:
    AXON_FILE = REPO_ROOT / "brainsurgery" / "synapse" / "models" / "gemma3" / "gemma-3-270m.axon"
    WEIGHTS_DIR = REPO_ROOT / "models" / "gemma-3-270m"
    MODEL_NAME = "Gemma-3-270M"
else:
    AXON_FILE = REPO_ROOT / "brainsurgery" / "synapse" / "models" / "gpt2" / "gpt2.axon"
    WEIGHTS_DIR = REPO_ROOT / "models" / "gpt2"
    if not (WEIGHTS_DIR / "model.safetensors").exists():
        WEIGHTS_DIR = REPO_ROOT / "models" / "openai-community" / "gpt2"
    MODEL_NAME = "GPT-2"

PROMPT_LENS = [16, 64, 256, 512, 1024]
GEN_STEPS = 64
TRIALS = 3
CLASS_NAME = "GeneratedModel"
MAX_KV_WARMUP = 1100


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


def _resolve_safetensors_paths(weights: Path) -> list[Path]:
    if weights.is_file():
        return [weights]
    return sorted(weights.glob("*.safetensors"))


def _load_model_config(model_dir: Path) -> dict:
    import json
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def compile_and_load(backend: str, tmp_dir: Path):
    """Compile axon → graph IR → codegen → load model class + state dict."""
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
        graph_main_output_names,
    )
    from brainsurgery.synapse.axon.codegen2_mlx import (
        emit_model_code_from_graph_ir as emit_mlx,
    )
    try:
        from brainsurgery.synapse.axon.codegen2_triton import (
            emit_model_code_from_graph_ir as emit_triton,
        )
    except ImportError:
        emit_triton = None

    resolved_axon = resolve_axon_program_from_path(AXON_FILE).ast
    normalized = normalize_closed_axon_file(resolved_axon)
    elaborated = elaborate_closed_axon_file(normalized)
    flat = flatten_closed_axon_file(elaborated)
    typed = typecheck2_flat_axon_file(flat)
    graph_program = lower_axon_program_to_graph_ir(typed)

    # Optimize graph with backend intrinsics
    backend_intrinsics = None
    if backend == "codegen2-triton":
        backend_intrinsics = "codegen2-triton"
    elif backend == "codegen2-torch":
        backend_intrinsics = "codegen2-torch"
    if backend_intrinsics:
        graph_program = optimize_graph_program(
            graph_program,
            config=GraphOptimizeConfig(backend_intrinsics=backend_intrinsics),
        )

    model_config = _load_model_config(WEIGHTS_DIR)

    if backend == "codegen2-torch":
        code = emit_torch(graph_program, class_name=CLASS_NAME, model_config=model_config)
    elif backend in ("codegen2-mlx", "codegen2-mlx-compiled"):
        code = emit_mlx(graph_program, class_name=CLASS_NAME, model_config=model_config)
    elif backend == "codegen2-triton":
        if emit_triton is None:
            raise ImportError("codegen2-triton not available on this branch")
        code = emit_triton(graph_program, class_name=CLASS_NAME, model_config=model_config)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    py_path = tmp_dir / f"generated_{backend.replace('-', '_')}.py"
    py_path.write_text(code, encoding="utf-8")
    model_cls = _load_generated_class(py_path, CLASS_NAME)

    # Load weights
    import safetensors
    safetensors_paths = _resolve_safetensors_paths(WEIGHTS_DIR)
    if backend in ("codegen2-mlx", "codegen2-mlx-compiled"):
        import mlx.core as mx
        import safetensors
        state_dict = {}
        for p in safetensors_paths:
            with safetensors.safe_open(str(p), framework="pt") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    state_dict[str(key)] = mx.array(t.float().numpy())
        model = model_cls.from_state_dict(state_dict).eval()
        return model, "mlx"
    else:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        target_device = torch.device(device)
        state_dict = {}
        for p in safetensors_paths:
            with safetensors.safe_open(str(p), framework="pt") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    state_dict[str(key)] = t.to(device=target_device, dtype=torch.float32)
        if backend == "codegen2-triton":
            model = model_cls.from_state_dict(state_dict, param_devices=[device]).eval()
        else:
            model = model_cls.from_state_dict(state_dict).to(target_device).eval()
        return model, "torch"


def to_int_array(ids, kind: str):
    if kind == "mlx":
        import mlx.core as mx
        return mx.array([ids], dtype=mx.int32)
    else:
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        return torch.tensor([ids], dtype=torch.int32, device=dev)


def argmax_last(logits, kind: str) -> int:
    if kind == "mlx":
        import mlx.core as mx
        return int(mx.argmax(logits[:, -1, :], axis=-1).item())
    else:
        import torch
        return int(torch.argmax(logits[:, -1, :], dim=-1).item())


def sync(kind: str):
    if kind == "mlx":
        import mlx.core as mx
        mx.eval(mx.array(0))
    else:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.synchronize()


def _unpack(result, kind: str):
    """forward returns {'logits':..., 'new_kv':...} on feat-mlx, tuple on feat-serving."""
    if isinstance(result, dict):
        return result.get('logits'), result.get('new_kv')
    return result[0], result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else (result[0], None)


def run_backend(label: str, backend: str, tmp_dir: Path):
    print(f"  Loading [{label}]...", end=" ", flush=True)
    try:
        model, kind = compile_and_load(backend, tmp_dir)
        if backend == "codegen2-mlx-compiled":
            import mlx.core as mx
            print(f"compiling...", end=" ", flush=True)
            model.compile(max_kv_length=MAX_KV_WARMUP)
            mx.eval(mx.array(0))
            print("OK")
        else:
            print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {}

    print(f"\n{MODEL_NAME} raw {label} throughput — {GEN_STEPS} decode tokens, {TRIALS} trials\n")
    print(f"  {'Prompt':<10s}  {'Prefill ms':<12s}  {'Decode ms/step':<16s}  {'Tok/s':<8s}")
    print("  " + "-" * 52)

    per_prompt: dict[int, float] = {}
    for plen in PROMPT_LENS:
        prompt_ids = list(range(plen))
        try:
            # Warmup
            warmup_steps = GEN_STEPS if hasattr(model, '_compiled_fn') and model._compiled_fn is not None else min(4, GEN_STEPS)
            for _ in range(2):
                inp = to_int_array(prompt_ids, kind)
                logits, kv = _unpack(model.forward(inp, use_cache=True), kind)
                for _ in range(warmup_steps):
                    next_id = argmax_last(logits, kind)
                    inp = to_int_array([next_id], kind)
                    logits, kv = _unpack(model.forward(inp, past_kv=kv, use_cache=True), kind)
                sync(kind)

            prefill_times = []
            decode_times = []
            for _ in range(TRIALS):
                inp = to_int_array(prompt_ids, kind)
                t0 = time.perf_counter()
                logits, kv = _unpack(model.forward(inp, use_cache=True), kind)
                sync(kind)
                prefill_ms = (time.perf_counter() - t0) * 1000
                prefill_times.append(prefill_ms)

                next_id = argmax_last(logits, kind)
                step_times = []
                for _ in range(GEN_STEPS - 1):
                    inp = to_int_array([next_id], kind)
                    t0 = time.perf_counter()
                    logits, kv = _unpack(model.forward(inp, past_kv=kv, use_cache=True), kind)
                    sync(kind)
                    step_ms = (time.perf_counter() - t0) * 1000
                    step_times.append(step_ms)
                    next_id = argmax_last(logits, kind)

                decode_times.append(sum(step_times) / len(step_times))

            avg_prefill = sum(prefill_times) / len(prefill_times)
            avg_decode = sum(decode_times) / len(decode_times)
            tps = 1000.0 / avg_decode
            per_prompt[plen] = tps
            print(f"  p={plen:<8d}  {avg_prefill:>10.1f}    {avg_decode:>12.2f}      {tps:>6.1f}")
        except Exception as e:
            print(f"  p={plen:<8d}  ERROR: {e}")
            import traceback
            traceback.print_exc()

    return per_prompt


def main():
    configs = [
        ("torch (MPS)", "codegen2-torch"),
        ("triton (MPS)", "codegen2-triton"),
        ("mlx (GPU)", "codegen2-mlx"),
        ("mlx+compile", "codegen2-mlx-compiled"),
    ]

    results: dict[str, dict[int, float]] = {}

    with TemporaryDirectory(prefix="axon_bench_") as tmp_dir:
        for label, backend in configs:
            per_prompt = run_backend(label, backend, Path(tmp_dir))
            results[label] = per_prompt
            print()

    # Summary
    print(f"\n{MODEL_NAME} raw decode tok/s comparison (no serving)\n")
    print(f"  {'Backend':<16s}" + "".join(f"  p={p:<5d}" for p in PROMPT_LENS))
    print("  " + "-" * 16 + "".join("-------" for _ in PROMPT_LENS))
    for label, _ in configs:
        row = results.get(label, {})
        if not row:
            print(f"  {label:<16s}" + "  n/a     " * len(PROMPT_LENS))
            continue
        cells = "".join(f"  {row.get(p, float('nan')):>7.1f}" for p in PROMPT_LENS)
        print(f"  {label:<16s}{cells}")

    print("\n  (raw forward only, no paged attention, no serving overhead)")


if __name__ == "__main__":
    main()
