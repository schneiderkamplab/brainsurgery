#!/usr/bin/env python3
"""Worker: benchmark native vLLM generation via ModelRegistry registration.

Generates Axon model code, registers the model class with vLLM's ModelRegistry,
creates a vllm.LLM with patched config (architectures=["AxonGeneratedForCausalLM"]),
and runs native generation with PagedAttention + KV cache reuse.

Measures:
- HF generate time (transformers, baseline)
- Axon native vLLM generate time
- tok/s for both
"""
import sys, os, csv, time, json, shutil, tempfile, traceback, importlib.util
from pathlib import Path
from typing import Any

REPO = "/work/dfm/jacobwashere/brainsurgery"
sys.path.insert(0, REPO)
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["CPATH"] = "/usr/local/cuda-13.2/targets/x86_64-linux/include:" + os.environ.get("CPATH", "")
os.environ["LIBRARY_PATH"] = "/usr/local/cuda-13.2/targets/x86_64-linux/lib/stubs:" + os.environ.get("LIBRARY_PATH", "")

import torch

ARCH_NAME = "AxonGeneratedForCausalLM"
CLASS_NAME = "AxonGeneratedModel"
MAX_NEW_TOKENS = 128
PROMPTS = ["The future of AI is"]
WARMUP = 1
REPEAT = 3

CSV_FIELDS = [
    "backend", "axon", "checkpoint", "model_dir", "fallback", "benchmark_path",
    "hf_time", "axon_time", "speed_ratio_axon_over_hf",
    "forward_warmup", "forward_repeat", "generate_warmup", "generate_repeat",
    "hf_sample_count", "axon_sample_count", "hf_warmup_sample_count", "axon_warmup_sample_count",
    "axon_profile_top_region", "axon_profile_top_seconds", "axon_profile_top_calls",
    "masked_top1_eq", "masked_max_abs_diff", "masked_max_rel_diff",
]


def _generate_model_code(axon_file: Path, model_dir: Path) -> tuple[str, dict[str, Any] | None, Any | None]:
    """Run the Axon codegen pipeline to generate vLLM model code."""
    from brainsurgery.synapse.axon import (
        resolve_axon_program_from_path,
        normalize_closed_axon_file,
        elaborate_closed_axon_file,
        flatten_closed_axon_file,
        typecheck2_flat_axon_file,
        optimize_safe_flat_typed_axon_file,
        lower_axon_program_to_graph_ir,
        optimize_graph_program,
        GraphOptimizeConfig,
    )
    from brainsurgery.synapse.axon.codegen2_vllm import emit_model_code_from_graph_ir
    from brainsurgery.synapse.axon_test import (
        _resolve_safetensors_paths,
        _load_model_config,
        _normalize_config_keys,
        _augment_model_config_from_checkpoint,
        _default_graph_backend_intrinsics,
        _load_auto_config_with_compat_fallback,
    )

    safetensors_files = _resolve_safetensors_paths(model_dir)
    resolved_hf_model_dir = model_dir.resolve()
    model_config = _augment_model_config_from_checkpoint(
        model_dir=resolved_hf_model_dir,
        safetensors_files=safetensors_files,
        model_config=_load_model_config(resolved_hf_model_dir),
    )
    model_config = _normalize_config_keys(model_config)

    try:
        hf_config = _load_auto_config_with_compat_fallback(
            resolved_hf_model_dir, trust_remote_code=False,
        )
    except Exception:
        try:
            hf_config = _load_auto_config_with_compat_fallback(
                resolved_hf_model_dir, trust_remote_code=True,
            )
        except Exception:
            # Some models (e.g. BlackMamba) have no model_type in config.json.
            # Temporarily add a generic model_type so AutoConfig can load it.
            cfg_path = resolved_hf_model_dir / "config.json"
            cfg_text = cfg_path.read_text()
            import json as _json
            cfg = _json.loads(cfg_text)
            if cfg.get("model_type") is None:
                cfg["model_type"] = "llama"
                cfg_path.write_text(_json.dumps(cfg))
                try:
                    hf_config = _load_auto_config_with_compat_fallback(
                        resolved_hf_model_dir, trust_remote_code=False,
                    )
                finally:
                    cfg_path.write_text(cfg_text)

    resolved_axon = resolve_axon_program_from_path(axon_file).ast
    normalized_axon = normalize_closed_axon_file(resolved_axon)
    elaborated_axon = elaborate_closed_axon_file(normalized_axon)
    flat_axon = flatten_closed_axon_file(elaborated_axon)
    typed_axon = typecheck2_flat_axon_file(flat_axon)
    typed_axon = optimize_safe_flat_typed_axon_file(typed_axon)
    graph_program = lower_axon_program_to_graph_ir(typed_axon)

    effective_intrinsics = _default_graph_backend_intrinsics(
        axon_backend="codegen2-vllm",
        graph_backend_intrinsics=None,
    )
    graph_program = optimize_graph_program(
        graph_program,
        config=GraphOptimizeConfig(backend_intrinsics=effective_intrinsics),
    )

    code = emit_model_code_from_graph_ir(
        graph_program,
        class_name=CLASS_NAME,
        model_config=model_config,
    )
    return code, model_config, hf_config


def _load_generated_class(code: str, class_name: str, py_path: Path) -> type:
    py_path.write_text(code, encoding="utf-8")
    module_name = f"_axon_native_{int(time.time() * 1e9)}"
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import generated module: {py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    model_cls = getattr(module, class_name, None)
    if model_cls is None:
        raise RuntimeError(f"Generated class {class_name!r} not found in {py_path}")
    return model_cls


def _create_patched_model_dir(model_dir: Path, hf_config: Any, model_config: dict | None) -> Path:
    """Create a temp dir with patched config.json and symlinks to original files."""
    config_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    config_dict["architectures"] = [ARCH_NAME]
    # Strip auto_map so vLLM doesn't look for custom code files that don't exist
    config_dict.pop("auto_map", None)
    # If model_type is missing or not recognized by transformers, set a generic
    # type so AutoConfig.from_pretrained doesn't fail (vLLM uses architectures
    # to find our registered model class, not model_type)
    _mt = config_dict.get("model_type")
    if _mt is None:
        config_dict["model_type"] = "llama"
    else:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        if _mt not in CONFIG_MAPPING:
            config_dict["model_type"] = "llama"
    # Strip rope_parameters/rope_scaling — the Axon generated model handles RoPE itself.
    # Leaving them in can trigger transformers rope standardization code that fails on
    # PreTrainedConfig (which vLLM creates for unknown architectures).
    config_dict.pop("rope_parameters", None)
    config_dict.pop("rope_scaling", None)

    if "hidden_size" not in config_dict:
        config_dict["hidden_size"] = (model_config or {}).get(
            "d_model", (model_config or {}).get("n_embed", (model_config or {}).get("n_embd", (model_config or {}).get("dim", 0)))
        )
    if "num_hidden_layers" not in config_dict:
        config_dict["num_hidden_layers"] = (model_config or {}).get(
            "num_layers", (model_config or {}).get("n_layer", (model_config or {}).get("encoder_layers", 0))
        )
    if "num_attention_heads" not in config_dict:
        config_dict["num_attention_heads"] = (model_config or {}).get(
            "num_heads", (model_config or {}).get("n_head", (model_config or {}).get("encoder_attention_heads", 1))
        )
    # Only set head_dim if not already present; some config classes (e.g. FalconConfig)
    # have head_dim as a read-only property that can't be set via the constructor.
    # vLLM can compute it from hidden_size / num_attention_heads.
    # if "head_dim" not in config_dict:
    #     hs = config_dict.get("hidden_size", 0)
    #     nah = config_dict.get("num_attention_heads", 0)
    #     if hs and nah:
    #         config_dict["head_dim"] = hs // nah

    tmp_dir = Path(tempfile.mkdtemp(prefix="axon_native_"))
    with open(tmp_dir / "config.json", "w") as f:
        json.dump(config_dict, f)

    for item in model_dir.resolve().iterdir():
        if item.name == "config.json":
            continue
        target = tmp_dir / item.name
        if not target.exists():
            try:
                os.symlink(item, target)
            except OSError:
                shutil.copy2(item, target)

    return tmp_dir


def _prepopulate_trc_cache(model_dir: Path) -> None:
    """Copy ALL .py files from model_dir to the transformers dynamic module cache.

    Transformers only copies direct relative imports of the main modeling file for
    local trust_remote_code models, missing transitive imports (e.g. configuration
    files that import tokenizer files). This pre-copies everything to avoid
    FileNotFoundError in get_relative_import_files.
    """
    import glob as _glob
    py_files = _glob.glob(str(model_dir / "*.py"))
    if not py_files:
        return
    try:
        from transformers.dynamic_module_utils import (
            _sanitize_module_name,
            _compute_local_source_files_hash,
            check_imports,
            TRANSFORMERS_DYNAMIC_MODULE_NAME,
            HF_MODULES_CACHE,
        )
        module_file = "modeling.py"
        for f in py_files:
            if "modeling" in os.path.basename(f):
                module_file = os.path.basename(f)
                break
        resolved = str(model_dir / module_file)
        modules_needed = check_imports(resolved)
        local_model_name = _sanitize_module_name(
            os.path.basename(os.path.normpath(str(model_dir)))
        )
        local_source_files_hash = _compute_local_source_files_hash(
            str(model_dir), module_file, resolved, modules_needed
        )
        submodule = os.path.sep.join([local_model_name, local_source_files_hash])
        full_submodule = TRANSFORMERS_DYNAMIC_MODULE_NAME + os.path.sep + submodule
        submodule_path = Path(HF_MODULES_CACHE) / full_submodule
        submodule_path.mkdir(parents=True, exist_ok=True)
        for py_file in py_files:
            dst = submodule_path / os.path.basename(py_file)
            shutil.copy2(py_file, dst)
        importlib.invalidate_caches()
    except Exception:
        pass


def _run_hf_generate(model_dir: Path, patched_dir: Path | None = None) -> tuple[float, int]:
    """Run HF generate and return (best_wall_time, total_generated_tokens)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Pre-populate transformers dynamic module cache for trust_remote_code models
    _prepopulate_trc_cache(model_dir)

    # Monkey-patch torch.autocast to handle 'meta' device (some trust_remote_code
    # models like Phi-3-small call torch.autocast during __init__ on meta device)
    _orig_autocast = torch.autocast
    class _MetaSafeAutocast(_orig_autocast):
        def __init__(self, device_type, enabled=True, **kwargs):
            if device_type == "meta":
                device_type = "cpu"
            super().__init__(device_type=device_type, enabled=enabled, **kwargs)
    torch.autocast = _MetaSafeAutocast

    # Try loading tokenizer from original dir first; if that fails (e.g. rope_parameters
    # triggering PreTrainedConfig issues), try from patched dir (which has rope params stripped).
    # Also try the checkpoint name (HF Hub download) for models without local tokenizer files.
    tokenizer = None
    ckpt_name = sys.argv[3] if len(sys.argv) > 3 else None
    for tdir in [model_dir, patched_dir, ckpt_name]:
        if tdir is None:
            continue
        for trc in [False, True]:
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(tdir), trust_remote_code=trc)
                break
            except Exception:
                continue
        if tokenizer is not None:
            break
    if tokenizer is None:
        raise RuntimeError("Failed to load tokenizer")
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Try without trust_remote_code first (built-in models like Falcon, Phi-3)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=False,
        )
    except Exception:
        # For trust_remote_code models, load config separately and null out
        # rope_scaling (PreTrainedConfig standardizes it, breaking custom
        # RotaryEmbedding implementations like Phi-3-small's LongRopeConfig).
        from transformers import AutoConfig
        try:
            hf_config = AutoConfig.from_pretrained(
                str(model_dir), trust_remote_code=True
            )
            if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling is not None:
                hf_config.rope_scaling = None
        except Exception:
            hf_config = None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
                config=hf_config,
            )
        except Exception:
            # Fallback for multimodal config classes (e.g. Mistral3Config)
            # that AutoModelForCausalLM doesn't recognize. Try loading with
            # the model's architecture class directly.
            import json
            cfg = json.load(open(Path(model_dir) / "config.json"))
            archs = cfg.get("architectures", [])
            model = None
            for arch in archs:
                try:
                    cls = getattr(__import__("transformers", fromlist=[arch]), arch)
                    model = cls.from_pretrained(
                        str(model_dir),
                        torch_dtype=torch.bfloat16,
                        device_map="cuda",
                        trust_remote_code=True,
                        config=hf_config,
                    )
                    break
                except (AttributeError, Exception):
                    continue
            if model is None:
                raise
    model.eval()
    torch.autocast = _orig_autocast

    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    input_len = input_ids.shape[-1]

    # Warmup
    with torch.no_grad():
        _ = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()

    # Measure
    best_time = float("inf")
    total_gen_tokens = 0
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        best_time = min(best_time, elapsed)
        total_gen_tokens = (output.shape[-1] - input_len) * len(PROMPTS)

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    return best_time, total_gen_tokens


def _run_native_vllm_generate(
    model_cls: type, patched_dir: Path, tokenizer_path: str | None = None,
) -> tuple[float, int, list[int]]:
    """Run native vLLM generate and return (best_wall_time, total_tokens, token_ids)."""
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.registry import ModelRegistry

    # Set flag so forward() returns hidden states (vLLM calls compute_logits separately)
    model_cls._vllm_native_mode = True

    ModelRegistry.register_model(ARCH_NAME, model_cls)

    # Adapt gpu_memory_utilization to what's actually free
    _free, _total = torch.cuda.mem_get_info()
    _gpu_mem = min(0.85, (_free / _total) * 0.95)  # use 95% of what's free
    _gpu_mem = max(0.30, _gpu_mem)  # at least 30%

    llm_kwargs = dict(
        model=str(patched_dir),
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=_gpu_mem,
        max_model_len=512,
        max_num_batched_tokens=4096,
        trust_remote_code=True,
    )
    # For SSM/Mamba hybrid models, set mamba_block_size (vLLM's built-in
    # ModelArchitectureConfig would do this for recognized architectures)
    if getattr(model_cls, "is_hybrid", False) or getattr(model_cls, "is_attention_free", False):
        llm_kwargs["mamba_block_size"] = 512
    if tokenizer_path is not None:
        llm_kwargs["tokenizer"] = tokenizer_path

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=MAX_NEW_TOKENS,
    )

    # Warmup
    _ = llm.generate(PROMPTS[:1], sampling_params)

    # Measure
    best_time = float("inf")
    total_gen_tokens = 0
    gen_token_ids: list[int] = []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = llm.generate(PROMPTS, sampling_params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        best_time = min(best_time, elapsed)
        total_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        gen_token_ids = list(outputs[0].outputs[0].token_ids)

    del llm
    torch.cuda.empty_cache()

    return best_time, total_gen_tokens, gen_token_ids


def main():
    axon_file = Path(sys.argv[1])
    model_dir = Path(sys.argv[2])
    ckpt = sys.argv[3]
    csv_path = Path(sys.argv[4])

    try:
        # Step 1: Generate model code
        code, model_config, hf_config = _generate_model_code(axon_file, model_dir)

        # Step 2: Import model class
        with tempfile.TemporaryDirectory(prefix="axon_native_code_") as code_dir:
            py_path = Path(code_dir) / "generated_model.py"
            model_cls = _load_generated_class(code, CLASS_NAME, py_path)

            # Step 3: Create patched model dir
            patched_dir = _create_patched_model_dir(model_dir, hf_config, model_config)

            try:
                # Step 4: Run HF generate (may fail for models requiring flash_attn etc.)
                try:
                    hf_time, hf_gen_tokens = _run_hf_generate(model_dir, patched_dir)
                except Exception as hf_err:
                    print(f"HF baseline failed ({type(hf_err).__name__}), running Axon-only")
                    hf_time, hf_gen_tokens = 0.0, 0

                # Step 5: Run native vLLM generate
                # Check if model_dir has tokenizer files; if not, pass ckpt name for HF Hub download
                _has_tok = any(
                    f.name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                                "spiece.model", "vocab.json", "merges.txt")
                    for f in model_dir.iterdir() if f.is_file()
                ) if model_dir.is_dir() else False
                _tok_path = ckpt if not _has_tok else None
                axon_time, axon_gen_tokens, _ = _run_native_vllm_generate(model_cls, patched_dir, tokenizer_path=_tok_path)
            finally:
                shutil.rmtree(patched_dir, ignore_errors=True)

        speed_ratio = axon_time / hf_time if hf_time and hf_time > 0 else 0
        hf_tok_s = hf_gen_tokens / hf_time if hf_time > 0 else 0
        axon_tok_s = axon_gen_tokens / axon_time if axon_time > 0 else 0
        status = "PASS" if axon_gen_tokens > 0 else "FAIL"

        row = {
            "backend": "codegen2-vllm-native",
            "axon": str(axon_file),
            "checkpoint": ckpt,
            "model_dir": str(model_dir),
            "fallback": "none",
            "benchmark_path": "generate-native",
            "hf_time": f"{hf_time:g}",
            "axon_time": f"{axon_time:g}",
            "speed_ratio_axon_over_hf": f"{speed_ratio:g}",
            "forward_warmup": 0, "forward_repeat": 0,
            "generate_warmup": WARMUP, "generate_repeat": REPEAT,
            "hf_sample_count": REPEAT, "axon_sample_count": REPEAT,
            "hf_warmup_sample_count": WARMUP, "axon_warmup_sample_count": WARMUP,
            "axon_profile_top_region": "", "axon_profile_top_seconds": "", "axon_profile_top_calls": "",
            "masked_top1_eq": status,
            "masked_max_abs_diff": f"{axon_tok_s:g}",
            "masked_max_rel_diff": f"{hf_tok_s:g}",
        }
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerow(row)
        print(
            f"RESULT: {status} hf_time={hf_time:.4f} axon_time={axon_time:.4f} "
            f"speed={speed_ratio:.3f} hf_tok/s={hf_tok_s:.1f} axon_tok/s={axon_tok_s:.1f}"
        )

    except Exception as e:
        row = {
            "backend": "codegen2-vllm-native",
            "axon": str(axon_file),
            "checkpoint": ckpt,
            "model_dir": str(model_dir),
            "fallback": "error",
            "benchmark_path": "generate-native",
            "hf_time": "", "axon_time": "", "speed_ratio_axon_over_hf": "",
            "forward_warmup": 0, "forward_repeat": 0,
            "generate_warmup": WARMUP, "generate_repeat": REPEAT,
            "hf_sample_count": 0, "axon_sample_count": 0,
            "hf_warmup_sample_count": 0, "axon_warmup_sample_count": 0,
            "axon_profile_top_region": "", "axon_profile_top_seconds": "", "axon_profile_top_calls": "",
            "masked_top1_eq": "ERROR",
            "masked_max_abs_diff": "ERROR",
            "masked_max_rel_diff": "ERROR",
        }
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerow(row)
        print(f"ERROR: {type(e).__name__}: {str(e)[:500]}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
