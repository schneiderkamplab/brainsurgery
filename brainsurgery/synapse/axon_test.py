from __future__ import annotations

import gc
import importlib.util
import json
import time
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import safetensors
import torch
from mltiming import timing
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils.quantization_config import Mxfp4Config

from .axon import (
    candidate_tokenizer_dirs,
    looks_like_tokenizer_dir,
    lower_axon_program_to_synapse_spec,
    parse_axon_program_from_path,
    tokenize_prompts,
)
from .black_mamba_reference import BlackMambaReferenceModel, is_black_mamba_config_dir
from .codegen import emit_model_code_from_synapse_spec
from .mxfp4 import materialize_mxfp4_aliases


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device, but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS device, but MPS is unavailable")
    return device


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def _extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        logits = output.get("logits")
        if torch.is_tensor(logits):
            return logits
        if len(output) == 1:
            only_value = next(iter(output.values()))
            if torch.is_tensor(only_value):
                return only_value
        raise ValueError("Expected tensor logits in dict output")
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise ValueError(
        f"Unsupported model output type for logits extraction: {type(output).__name__}"
    )


def _load_state_dict(
    paths: list[Path],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for path in paths:
        st = safetensors.safe_open(str(path), framework="pt")
        for key in st.keys():
            if key in out:
                raise ValueError(f"Duplicate tensor key while reading safetensors shards: {key}")
            tensor = st.get_tensor(key)
            if tensor.is_floating_point():
                tensor = tensor.to(device=device, dtype=dtype)
            else:
                tensor = tensor.to(device=device)
            out[key] = tensor
    materialize_mxfp4_aliases(out, dtype=dtype, drop_packed=True)
    return out


def _resolve_safetensors_paths(weights: Path) -> list[Path]:
    if weights.is_file():
        if weights.suffix != ".safetensors":
            raise ValueError(f"Expected a .safetensors file, got: {weights}")
        return [weights]

    if not weights.is_dir():
        raise FileNotFoundError(f"Weights path not found: {weights}")

    index_path = weights / "model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid safetensors index (missing weight_map): {index_path}")
        shard_names = sorted({str(name) for name in weight_map.values()})
        paths = [weights / name for name in shard_names]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing safetensors shard(s) from index: {missing}")
        return paths

    preferred = weights / "model.safetensors"
    if preferred.exists():
        return [preferred]

    candidates = sorted(weights.glob("*.safetensors"))
    if len(candidates) == 1:
        return [candidates[0]]
    if not candidates:
        raise FileNotFoundError(f"No .safetensors files found in directory: {weights}")
    if all(path.name.startswith("model-") and "-of-" in path.name for path in candidates):
        return candidates
    raise ValueError(
        f"Multiple .safetensors files found in {weights}; pass an explicit .safetensors file path."
    )


def _normalize_rope_numeric_fields(config: Any) -> Any:
    def _normalize_dict(mapping: Any) -> None:
        if not isinstance(mapping, dict):
            return
        for key in ("factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim"):
            value = mapping.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                mapping[key] = float(value)

    rope_scaling = getattr(config, "rope_scaling", None)
    _normalize_dict(rope_scaling)
    rope_parameters = getattr(config, "rope_parameters", None)
    _normalize_dict(rope_parameters)
    return config


def _read_quant_method(config: Any) -> str | None:
    quant_cfg = getattr(config, "quantization_config", None)
    if quant_cfg is None:
        return None
    if isinstance(quant_cfg, dict):
        value = quant_cfg.get("quant_method")
        return None if value is None else str(value)
    value = getattr(quant_cfg, "quant_method", None)
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    return str(value)


def _build_non_mxfp4_quantization_config(config: Any) -> Mxfp4Config | None:
    if _read_quant_method(config) != "mxfp4":
        return None
    quant_cfg = getattr(config, "quantization_config", None)
    modules_to_not_convert: list[str] | None = None
    if isinstance(quant_cfg, dict):
        raw = quant_cfg.get("modules_to_not_convert")
        if isinstance(raw, list):
            modules_to_not_convert = [str(item) for item in raw]
    return Mxfp4Config(
        modules_to_not_convert=modules_to_not_convert,
        dequantize=True,
    )


def _load_generated_class(py_path: Path, class_name: str) -> type[Any]:
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


def _time_generate(label: str, fn: Any) -> tuple[Any, float]:
    t0 = time.perf_counter()
    with timing(message=label):
        out = fn()
    dt = time.perf_counter() - t0
    return out, dt


def _maybe_compile_model(
    model: Any,
    *,
    enabled: bool,
    backend: str | None,
    mode: str | None,
    fullgraph: bool,
    dynamic: bool,
) -> Any:
    if not enabled:
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        raise ValueError("torch.compile is not available in this PyTorch build")
    kwargs: dict[str, Any] = {
        "fullgraph": bool(fullgraph),
        "dynamic": bool(dynamic),
    }
    if backend:
        kwargs["backend"] = backend
    if mode:
        kwargs["mode"] = mode
    return compile_fn(model, **kwargs)


def _normalize_texts(text: str | Sequence[str]) -> list[str]:
    if isinstance(text, str):
        return [text]
    out: list[str] = []
    for item in text:
        if not isinstance(item, str):
            raise ValueError("All prompts passed via --text must be strings")
        out.append(item)
    if not out:
        return ["The future of AI is"]
    return out


def _extract_hidden_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, tuple) and value and torch.is_tensor(value[0]):
        return value[0]
    raise ValueError(f"Unable to extract hidden tensor from type: {type(value).__name__}")


def _to_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(dtype=torch.float32, device="cpu")


def run_axon_test(
    *,
    axon_file: Path,
    weights: Path,
    device: str = "cpu",
    text: str | Sequence[str] = "The future of AI is",
    max_len: int = 32,
    hf_model_dir: Path | None = None,
    tokenizer: str | None = None,
    class_name: str = "AxonGeneratedModel",
    main_module: str | None = None,
    dtype: str = "float32",
    trace_layers: bool = False,
    hf_align_bf16_profile: bool = False,
    hf_align_mask_contract: bool = False,
    hf_align_position_ids: bool = False,
    hf_align_add_fp32_accum: bool = False,
    hf_align_linear_fp32_accum: bool = False,
    hf_align_norm_fp32: bool = False,
    compile_hf: bool = False,
    compile_axon: bool = False,
    compile_backend: str | None = None,
    compile_mode: str | None = None,
    compile_fullgraph: bool = False,
    compile_dynamic: bool = False,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype)
    align_mask_contract = bool(hf_align_bf16_profile or hf_align_mask_contract)
    align_position_ids = bool(hf_align_bf16_profile or hf_align_position_ids)
    align_add_fp32 = bool(hf_align_bf16_profile or hf_align_add_fp32_accum)
    align_linear_fp32 = bool(hf_align_bf16_profile or hf_align_linear_fp32_accum)
    align_norm_fp32 = bool(hf_align_bf16_profile or hf_align_norm_fp32)

    axon_file = axon_file.resolve()
    weights_path = weights.resolve()
    if not axon_file.exists():
        raise FileNotFoundError(f"Axon file not found: {axon_file}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights path not found: {weights_path}")

    safetensors_files = _resolve_safetensors_paths(weights_path)
    default_hf_dir = weights_path if weights_path.is_dir() else safetensors_files[0].parent
    resolved_hf_model_dir = (hf_model_dir or default_hf_dir).resolve()
    tokenizer_source = tokenizer or str(resolved_hf_model_dir)
    if tokenizer is None:
        for candidate in candidate_tokenizer_dirs(resolved_hf_model_dir):
            if looks_like_tokenizer_dir(candidate):
                tokenizer_source = str(candidate)
                break
    tokenizer_fallback = resolved_hf_model_dir.name if tokenizer is None else None
    prompts = _normalize_texts(text)

    with TemporaryDirectory(prefix="axon_benchmark_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        synapse_yaml_path = tmp_path / "lowered_synapse.yaml"
        generated_py_path = tmp_path / "generated_model.py"

        modules = parse_axon_program_from_path(axon_file)
        synapse_spec = lower_axon_program_to_synapse_spec(modules, main_module=main_module)

        synapse_yaml_path.write_text(
            OmegaConf.to_yaml(synapse_spec, resolve=True), encoding="utf-8"
        )
        loaded = OmegaConf.load(synapse_yaml_path)
        loaded_dict = OmegaConf.to_container(loaded, resolve=True)
        if not isinstance(loaded_dict, dict):
            raise ValueError("Lowered synapse YAML did not produce a mapping")
        lowered_spec: dict[str, Any] = {str(key): value for key, value in loaded_dict.items()}

        code = emit_model_code_from_synapse_spec(lowered_spec, class_name=class_name)
        generated_py_path.write_text(code, encoding="utf-8")

        model_cls = _load_generated_class(generated_py_path, class_name)

        state_dict: dict[str, torch.Tensor] | None = None
        try:
            hf_config = AutoConfig.from_pretrained(
                str(resolved_hf_model_dir), local_files_only=True
            )
            hf_config = _normalize_rope_numeric_fields(hf_config)
            non_mxfp4_quant_config = _build_non_mxfp4_quantization_config(hf_config)
            hf_model: Any = AutoModelForCausalLM.from_pretrained(
                str(resolved_hf_model_dir),
                local_files_only=True,
                dtype=resolved_dtype,
                config=hf_config,
                quantization_config=non_mxfp4_quant_config,
            )
            hf = hf_model.to(device=resolved_device, dtype=resolved_dtype).eval()
        except Exception:
            if not is_black_mamba_config_dir(resolved_hf_model_dir):
                raise
            state_dict = _load_state_dict(
                safetensors_files,
                device=resolved_device,
                dtype=resolved_dtype,
            )
            hf = (
                BlackMambaReferenceModel.from_state_dict(
                    model_dir=resolved_hf_model_dir, state_dict=state_dict
                )
                .to(resolved_device)
                .eval()
            )
        hf = _maybe_compile_model(
            hf,
            enabled=compile_hf,
            backend=compile_backend,
            mode=compile_mode,
            fullgraph=compile_fullgraph,
            dynamic=compile_dynamic,
        )
        if hasattr(hf, "generation_config"):
            hf.generation_config.do_sample = False
            hf.generation_config.top_p = None
            hf.generation_config.top_k = None

        tokenizer_obj, input_ids, attention_mask = tokenize_prompts(
            prompts=prompts,
            tokenizer_source=tokenizer_source,
            tokenizer_fallback=tokenizer_fallback,
            device=resolved_device,
            lowered_spec=lowered_spec,
        )
        model_inputs = lowered_spec.get("model", {}).get("inputs", {})
        model_input_names = (
            set(model_inputs.keys()) if isinstance(model_inputs, dict) else {"input_ids"}
        )
        syn_mask_key = (
            "attn_mask"
            if "attn_mask" in model_input_names
            else ("attention_mask" if "attention_mask" in model_input_names else None)
        )
        hf_inputs: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            hf_inputs["attention_mask"] = attention_mask
        use_mask_for_syn = bool(attention_mask is not None and syn_mask_key is not None)

        def _run_hf_generate(model: Any = hf) -> torch.Tensor:
            return model.generate(
                **hf_inputs,
                max_new_tokens=max(1, max_len - int(input_ids.shape[1])),
                eos_token_id=tokenizer_obj.eos_token_id,
                pad_token_id=tokenizer_obj.eos_token_id,
            )

        hf_gen, hf_time = _time_generate("HF", _run_hf_generate)
        hf_forward_inputs = dict(hf_inputs)
        if attention_mask is not None:
            # Align forward-logit comparison with decoder generation semantics under padding.
            pos_ids = attention_mask.to(torch.long).cumsum(dim=-1) - 1
            pos_ids = pos_ids.masked_fill(attention_mask == 0, 1)
            hf_forward_inputs["position_ids"] = pos_ids
        hf_layer_inputs: dict[int, torch.Tensor] = {}
        hf_layer_outputs: dict[int, torch.Tensor] = {}
        hf_hook_handles: list[Any] = []
        if trace_layers:
            hf_layers = getattr(getattr(hf, "model", None), "layers", None)
            if hf_layers is not None:
                for idx, layer in enumerate(hf_layers):

                    def _hf_pre_hook(
                        module: Any, args: tuple[Any, ...], *, _idx: int = idx
                    ) -> None:
                        del module
                        if not args:
                            return
                        hidden = _extract_hidden_tensor(args[0])
                        hf_layer_inputs[_idx] = _to_cpu_float(hidden)

                    def _hf_post_hook(
                        module: Any, args: tuple[Any, ...], out: Any, *, _idx: int = idx
                    ) -> None:
                        del module, args
                        hidden = _extract_hidden_tensor(out)
                        hf_layer_outputs[_idx] = _to_cpu_float(hidden)

                    hf_hook_handles.append(layer.register_forward_pre_hook(_hf_pre_hook))
                    hf_hook_handles.append(layer.register_forward_hook(_hf_post_hook))
        with torch.no_grad():
            hf_logits = hf(**hf_forward_inputs, use_cache=False).logits
        for handle in hf_hook_handles:
            handle.remove()

        del hf
        _cleanup(resolved_device)

        if state_dict is None:
            state_dict = _load_state_dict(
                safetensors_files,
                device=resolved_device,
                dtype=resolved_dtype,
            )
        syn = model_cls.from_state_dict(state_dict).to(resolved_device).eval()
        state_dict.clear()
        del state_dict
        _cleanup(resolved_device)
        setattr(syn, "_hf_align_mask_contract", align_mask_contract)
        setattr(syn, "_hf_align_position_ids", align_position_ids)
        setattr(syn, "_hf_align_add_fp32_accum", align_add_fp32)
        setattr(syn, "_hf_align_linear_fp32_accum", align_linear_fp32)
        setattr(syn, "_hf_align_norm_fp32", align_norm_fp32)
        syn = _maybe_compile_model(
            syn,
            enabled=compile_axon,
            backend=compile_backend,
            mode=compile_mode,
            fullgraph=compile_fullgraph,
            dynamic=compile_dynamic,
        )

        def _run_syn_generate(model: Any = syn) -> torch.Tensor:
            generate_kwargs: dict[str, Any] = {
                "eos_token_id": tokenizer_obj.eos_token_id,
                "max_len": max_len,
            }
            if use_mask_for_syn and attention_mask is not None:
                if syn_mask_key == "attn_mask":
                    generate_kwargs["attn_mask"] = attention_mask
                elif syn_mask_key == "attention_mask":
                    generate_kwargs["attention_mask"] = attention_mask
            return model.generate(input_ids, **generate_kwargs)

        syn_gen, syn_time = _time_generate("AxonDerived", _run_syn_generate)
        syn_inputs: dict[str, Any] = {"input_ids": input_ids}
        if use_mask_for_syn and attention_mask is not None and syn_mask_key is not None:
            syn_inputs[syn_mask_key] = attention_mask
        syn_layer_inputs: dict[int, torch.Tensor] = {}
        syn_layer_outputs: dict[int, torch.Tensor] = {}
        original_block_call = getattr(syn, "_block_gpt_oss_block", None)
        if trace_layers and callable(original_block_call):

            def _syn_block_wrapper(
                *, x: Any, i: Any, pos_ids: Any, attn_mask: Any, past_kv: Any, scope: str
            ) -> Any:
                layer_idx = int(i)
                if torch.is_tensor(x):
                    syn_layer_inputs[layer_idx] = _to_cpu_float(x)
                out = original_block_call(
                    x=x,
                    i=i,
                    pos_ids=pos_ids,
                    attn_mask=attn_mask,
                    past_kv=past_kv,
                    scope=scope,
                )
                if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
                    syn_layer_outputs[layer_idx] = _to_cpu_float(out[0])
                return out

            setattr(syn, "_block_gpt_oss_block", _syn_block_wrapper)
        with torch.no_grad():
            syn_logits = _extract_logits(syn(**syn_inputs))
        if trace_layers and callable(original_block_call):
            setattr(syn, "_block_gpt_oss_block", original_block_call)

        gen_hf = int(hf_gen.shape[1] - input_ids.shape[1])
        gen_syn = int(syn_gen.shape[1] - input_ids.shape[1])

        if syn_logits.device != hf_logits.device:
            syn_logits = syn_logits.to(hf_logits.device)
        diff = (syn_logits.float() - hf_logits.float()).abs()
        rel_denom = torch.maximum(
            torch.maximum(syn_logits.float().abs(), hf_logits.float().abs()),
            torch.tensor(1.0e-12, device=diff.device, dtype=diff.dtype),
        )
        rel_diff = diff / rel_denom
        mean_diff = float(diff.mean())
        max_diff = float(diff.max())
        last_max_diff = float(diff[:, -1, :].max())
        mean_rel_diff = float(rel_diff.mean())
        max_rel_diff = float(rel_diff.max())
        top1_eq = bool((syn_logits[:, -1, :].argmax(-1) == hf_logits[:, -1, :].argmax(-1)).all())

        masked_mean_diff: float | None = None
        masked_max_diff: float | None = None
        masked_last_max_diff: float | None = None
        masked_mean_rel_diff: float | None = None
        masked_max_rel_diff: float | None = None
        masked_top1_eq: bool | None = None
        if attention_mask is not None:
            mask_bool = attention_mask.to(torch.bool)
            valid = mask_bool.unsqueeze(-1).expand_as(diff)
            valid_count = int(valid.sum().item())
            if valid_count > 0:
                valid_diff = diff[valid]
                valid_rel_diff = rel_diff[valid]
                masked_mean_diff = float(valid_diff.mean())
                masked_max_diff = float(valid_diff.max())
                masked_mean_rel_diff = float(valid_rel_diff.mean())
                masked_max_rel_diff = float(valid_rel_diff.max())
            else:
                masked_mean_diff = 0.0
                masked_max_diff = 0.0
                masked_mean_rel_diff = 0.0
                masked_max_rel_diff = 0.0

            attn_bool = attention_mask.to(torch.bool)
            rev_last = torch.argmax(attn_bool.flip(dims=[1]).to(torch.long), dim=1)
            lengths = (attn_bool.shape[1] - 1) - rev_last
            any_valid = attn_bool.any(dim=1)
            lengths = torch.where(lengths >= 0, lengths, torch.zeros_like(lengths))
            lengths = torch.where(any_valid, lengths, torch.zeros_like(lengths))
            b_idx = torch.arange(attention_mask.shape[0], device=attention_mask.device)
            syn_last = syn_logits[b_idx, lengths]
            hf_last = hf_logits[b_idx, lengths]
            masked_last_max_diff = float((syn_last.float() - hf_last.float()).abs().max())
            masked_top1_eq = bool((syn_last.argmax(-1) == hf_last.argmax(-1)).all())

        layer_diffs: list[dict[str, float | int]] = []
        if trace_layers and hf_layer_outputs and syn_layer_outputs:
            common_layers = sorted(set(hf_layer_outputs) & set(syn_layer_outputs))
            for layer_idx in common_layers:
                hf_out = hf_layer_outputs[layer_idx]
                syn_out = syn_layer_outputs[layer_idx]
                if hf_out.shape != syn_out.shape:
                    continue
                out_diff = (syn_out - hf_out).abs()
                out_mean = float(out_diff.mean())
                out_max = float(out_diff.max())
                out_last_max = float(out_diff[:, -1, :].max()) if out_diff.ndim >= 3 else out_max
                in_mean = float("nan")
                in_max = float("nan")
                if layer_idx in hf_layer_inputs and layer_idx in syn_layer_inputs:
                    hf_in = hf_layer_inputs[layer_idx]
                    syn_in = syn_layer_inputs[layer_idx]
                    if hf_in.shape == syn_in.shape:
                        in_diff = (syn_in - hf_in).abs()
                        in_mean = float(in_diff.mean())
                        in_max = float(in_diff.max())
                layer_diffs.append(
                    {
                        "layer": int(layer_idx),
                        "in_mean": in_mean,
                        "in_max": in_max,
                        "out_mean": out_mean,
                        "out_max": out_max,
                        "out_last_max": out_last_max,
                    }
                )

        if len(safetensors_files) == 1:
            safetensors_desc = str(safetensors_files[0])
        else:
            safetensors_desc = (
                f"{len(safetensors_files)} shards (first: {safetensors_files[0].name})"
            )

        print(f"Axon file:      {axon_file}")
        print(f"Safetensors:    {safetensors_desc}")
        print(f"Weights input:  {weights_path}")
        print(f"HF model dir:   {resolved_hf_model_dir}")
        print(f"Tokenizer:      {tokenizer_source}")
        print(f"Padding side:   {tokenizer_obj.padding_side}")
        print(f"Device:         {resolved_device}")
        print(f"Prompts:        {len(prompts)}")
        print(f"HF-align bf16 profile: {bool(hf_align_bf16_profile)}")
        print(f"HF-align mask:         {align_mask_contract}")
        print(f"HF-align posid:        {align_position_ids}")
        print(f"HF-align add fp32:     {align_add_fp32}")
        print(f"HF-align linear fp32:  {align_linear_fp32}")
        print(f"HF-align norm fp32:    {align_norm_fp32}")
        print(f"Compile HF:            {bool(compile_hf)}")
        print(f"Compile Axon:          {bool(compile_axon)}")
        print(f"Compile backend:       {compile_backend}")
        print(f"Compile mode:          {compile_mode}")
        print(f"Compile fullgraph:     {bool(compile_fullgraph)}")
        print(f"Compile dynamic:       {bool(compile_dynamic)}")
        print()
        print(
            f"HF:             {hf_time:.4f}s total, {gen_hf / max(hf_time, 1e-9):.2f} tok/s, generated={gen_hf}"
        )
        print(
            f"Axon-derived:   {syn_time:.4f}s total, {gen_syn / max(syn_time, 1e-9):.2f} tok/s, generated={gen_syn}"
        )
        print(f"Speed ratio (Axon/HF): {syn_time / max(hf_time, 1e-9):.3f}x")
        print()
        for idx, prompt in enumerate(prompts):
            print(f"Prompt[{idx}]: {prompt!r}")
            print("HF completion:")
            print(tokenizer_obj.decode(hf_gen[idx].tolist(), skip_special_tokens=True)[:80])
            print("Axon-derived completion:")
            print(tokenizer_obj.decode(syn_gen[idx].tolist(), skip_special_tokens=True)[:80])
            print()
        print(
            "Logits diff (raw) | mean/max/last_max/top1_eq:",
            mean_diff,
            max_diff,
            last_max_diff,
            top1_eq,
        )
        print(
            "Logits rel diff (raw) | mean/max:",
            mean_rel_diff,
            max_rel_diff,
        )
        if attention_mask is not None:
            print(
                "Logits diff (masked) | mean/max/last_max/top1_eq:",
                masked_mean_diff,
                masked_max_diff,
                masked_last_max_diff,
                masked_top1_eq,
            )
            print(
                "Logits rel diff (masked) | mean/max:",
                masked_mean_rel_diff,
                masked_max_rel_diff,
            )
        if trace_layers and layer_diffs:
            print()
            print("Layer diffs (HF vs Axon) | layer in_mean in_max out_mean out_max out_last_max")
            for row in layer_diffs:
                print(
                    int(row["layer"]),
                    row["in_mean"],
                    row["in_max"],
                    row["out_mean"],
                    row["out_max"],
                    row["out_last_max"],
                )
            first_large = next((row for row in layer_diffs if float(row["out_mean"]) > 0.05), None)
            if first_large is not None:
                print(
                    "First large layer diff (out_mean > 0.05):",
                    int(first_large["layer"]),
                    first_large["out_mean"],
                    first_large["out_max"],
                )

        result = {
            "hf_time": hf_time,
            "axon_time": syn_time,
            "speed_ratio_axon_over_hf": syn_time / max(hf_time, 1.0e-9),
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "last_max_diff": last_max_diff,
            "mean_rel_diff": mean_rel_diff,
            "max_rel_diff": max_rel_diff,
            "top1_eq": top1_eq,
            "masked_mean_diff": masked_mean_diff,
            "masked_max_diff": masked_max_diff,
            "masked_last_max_diff": masked_last_max_diff,
            "masked_mean_rel_diff": masked_mean_rel_diff,
            "masked_max_rel_diff": masked_max_rel_diff,
            "masked_top1_eq": masked_top1_eq,
            "layer_diffs": layer_diffs if trace_layers else None,
            "compile_hf": bool(compile_hf),
            "compile_axon": bool(compile_axon),
            "compile_backend": compile_backend,
            "compile_mode": compile_mode,
            "compile_fullgraph": bool(compile_fullgraph),
            "compile_dynamic": bool(compile_dynamic),
            "prompts": prompts,
            "generated_hf": hf_gen,
            "generated_axon": syn_gen,
        }

        del syn
        _cleanup(resolved_device)
        return result


__all__ = ["run_axon_test"]
