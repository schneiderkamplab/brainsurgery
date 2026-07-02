from __future__ import annotations

import gc
import hashlib
import html
import inspect
import importlib.machinery
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
from copy import deepcopy
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any, cast

import safetensors
import torch
from accelerate import dispatch_model, init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from mltiming import timing
from omegaconf import OmegaConf
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.utils import import_utils as transformers_import_utils
from transformers.utils.quantization_config import FineGrainedFP8Config, Mxfp4Config

from .axon import (
    candidate_tokenizer_dirs,
    elaborate_closed_axon_file,
    flatten_closed_axon_file,
    looks_like_tokenizer_dir,
    lower_axon_program_to_graph_ir,
    normalize_closed_axon_file,
    optimize_graph_program,
    optimize_safe_flat_typed_axon_file,
    parse_axon_program_from_path,
    resolve_axon_program_from_path,
    resolve_main_module,
    tokenize_prompts,
    typecheck2_flat_axon_file,
)
from .axon.ast import AxonFile, TypeOptional
from .axon.codegen2_tinygrad import emit_model_code_from_graph_ir as emit_tinygrad_model_code_from_graph_ir
from .axon.codegen2_triton import emit_model_code_from_graph_ir as emit_triton_model_code_from_graph_ir
from .axon.codegen2_torch import (
    emit_model_code_from_graph_ir as emit_torch_model_code_from_graph_ir,
    graph_main_output_names as _graph_main_output_names,
    make_runtime2_model_class as make_runtime2_torch_model_class,
)
from .axon.codegen2_vllm import emit_model_code_from_graph_ir as emit_vllm_model_code_from_graph_ir
from .axon_runner_common import cleanup_cuda_after_oom as _cleanup_cuda_after_oom
from .axon_runner_common import is_cuda_oom as _is_cuda_oom
from .black_mamba_reference import BlackMambaReferenceModel, is_black_mamba_config_dir
from .matrix_models import ModelDownloadSpec, ensure_model_downloaded
from .mxfp4 import materialize_mxfp4_aliases


def _default_graph_backend_intrinsics(
    *,
    axon_backend: str,
    graph_backend_intrinsics: str | None,
) -> str | None:
    if graph_backend_intrinsics is not None:
        return graph_backend_intrinsics
    if axon_backend == "codegen2-triton":
        return "codegen2-triton"
    if axon_backend == "codegen2-vllm":
        return "codegen2-vllm"
    return None


def _format_metric_value(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(cast(Any, value)):.6g}"
    except Exception:
        return str(value)


def _format_checkpoint_summary_table(
    rows: Sequence[dict[str, object]],
    *,
    table_format: str,
) -> str:
    if table_format not in {"plain", "markdown", "html"}:
        raise ValueError("table_format must be 'plain', 'markdown', or 'html'")
    headers = [
        "axon",
        "checkpoint",
        "model dir",
        "fallback",
        "masked top-1 eq",
        "masked max abs diff",
        "masked max rel diff",
    ]
    body = [
        [
            str(row["axon"]),
            str(row["checkpoint"]),
            str(row["model_dir"]),
            str(row.get("fallback", "none")),
            str(row["masked_top1_eq"]),
            str(row["masked_max_abs_diff"]),
            str(row["masked_max_rel_diff"]),
        ]
        for row in rows
    ]
    if table_format == "html":

        def _numeric(cell: str) -> float | None:
            try:
                return float(cell)
            except Exception:
                return None

        def _cell_style(header: str, cell: str) -> str:
            if header == "masked top-1 eq" and cell != "True":
                return "background-color: #f8d7da;"
            if header == "masked max abs diff":
                numeric = _numeric(cell)
                if numeric is not None:
                    if numeric > 1e-2:
                        return "background-color: #ffe5b4;"
                    if numeric > 1e-3:
                        return "background-color: #fff3cd;"
            if "max abs diff" in header and header != "masked max abs diff":
                numeric = _numeric(cell)
                if numeric is not None and numeric > 1.0:
                    return "background-color: #fff3cd;"
            return ""

        def _row_style(line: Sequence[str]) -> str:
            masked_top1 = line[4]
            if masked_top1 != "True":
                return "background-color: #dc3545; color: #ffffff;"
            return ""

        out = [
            "<table>",
            "  <thead>",
            "    <tr>",
            *[f"      <th>{html.escape(header)}</th>" for header in headers],
            "    </tr>",
            "  </thead>",
            "  <tbody>",
        ]
        for line in body:
            row_style = _row_style(line)
            row_style_attr = f' style="{row_style}"' if row_style else ""
            out.append(f"    <tr{row_style_attr}>")
            for header, cell in zip(headers, line, strict=False):
                style = "" if row_style else _cell_style(header, cell)
                style_attr = f' style="{style}"' if style else ""
                out.append(f"      <td{style_attr}>{html.escape(cell)}</td>")
            out.append("    </tr>")
        out.extend(["  </tbody>", "</table>"])
        return "\n".join(out)
    if table_format == "markdown":
        out = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
        ]
        out.extend("| " + " | ".join(line) + " |" for line in body)
        return "\n".join(out)

    widths = [len(header) for header in headers]
    for line in body:
        for idx, cell in enumerate(line):
            widths[idx] = max(widths[idx], len(cell))

    def _fmt(line: Sequence[str]) -> str:
        return " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(line))

    divider = "-+-".join("-" * width for width in widths)
    return "\n".join([_fmt(headers), divider, *(_fmt(line) for line in body)])


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


def _resolve_optional_torch_dtype_name(name: object) -> torch.dtype | None:
    if not isinstance(name, str):
        return None
    normalized = name.strip().lower()
    if normalized in {"float32", "torch.float32", "fp32"}:
        return torch.float32
    if normalized in {"bfloat16", "torch.bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "torch.float16", "fp16"}:
        return torch.float16
    return None


def _resolve_model_task(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized == "auto":
        return normalized
    if normalized == "causal_lm":
        return normalized
    if normalized == "masked_lm":
        return normalized
    if normalized == "seq2seq_lm":
        return normalized
    raise ValueError(
        f"Unsupported model_task: {name!r} (expected 'auto', 'causal_lm', 'masked_lm', or 'seq2seq_lm')"
    )


def _resolve_benchmark_mode(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized in {"auto", "forward", "generate"}:
        return normalized
    raise ValueError(
        f"Unsupported benchmark_mode: {name!r} (expected 'auto', 'forward', or 'generate')"
    )


def _should_generate_for_benchmark(*, model_task: str, benchmark_mode: str) -> bool:
    if benchmark_mode == "forward":
        return False
    if benchmark_mode == "auto":
        return model_task == "causal_lm"
    if model_task in {"causal_lm", "seq2seq_lm"}:
        return True
    raise ValueError(
        f"benchmark_mode='generate' is not supported for model_task={model_task!r}; "
        "encoder-only and masked-LM models only support forward benchmarking"
    )


def _task_pragma_from_axon(*, axon_file: Path) -> str | None:
    parsed = parse_axon_program_from_path(axon_file)
    module = _select_main_axon_module(parsed)
    raw = (getattr(module, "pragmas", None) or {}).get("task")
    if raw is None:
        raw = (getattr(parsed, "pragmas", None) or {}).get("task")
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"causal_lm", "masked_lm", "seq2seq_lm"}:
        return normalized
    raise ValueError(
        f"Unsupported TASK pragma in {axon_file}: {raw!r}"
        " (expected 'causal_lm', 'masked_lm', or 'seq2seq_lm')"
    )


def _tokenizer_pragma_from_axon(*, axon_file: Path) -> str | None:
    parsed = parse_axon_program_from_path(axon_file)
    module = _select_main_axon_module(parsed)
    raw = (getattr(module, "pragmas", None) or {}).get("tokenizer")
    if raw is None:
        raw = (getattr(parsed, "pragmas", None) or {}).get("tokenizer")
    if raw is None:
        return None
    if isinstance(raw, str) and raw:
        return raw
    raise ValueError(
        f"Unsupported TOKENIZER pragma in {axon_file}: {raw!r}"
        " (use _tokenizer_pragma_for_checkpoint for structured TOKENIZER pragmas)"
    )


def _tokenizer_pragma_for_checkpoint(
    *,
    axon_file: Path,
    checkpoint_id: str,
) -> str | None:
    modules = parse_axon_program_from_path(axon_file)
    module = _select_main_axon_module(modules)
    raw = (getattr(module, "pragmas", None) or {}).get("tokenizer")
    if raw is None:
        raw = (getattr(modules, "pragmas", None) or {}).get("tokenizer")
    if raw is None:
        return None

    global_tokenizer: str | None = None
    by_checkpoint: dict[str, str] = {}

    def _consume(entry: object) -> None:
        nonlocal global_tokenizer
        if isinstance(entry, str):
            if not entry:
                raise ValueError(f"Unsupported TOKENIZER pragma in {axon_file}: {raw!r}")
            if global_tokenizer is not None and global_tokenizer != entry:
                raise ValueError(
                    f"Unsupported TOKENIZER pragma in {axon_file}: conflicting global tokenizers"
                )
            global_tokenizer = entry
            return
        if (
            isinstance(entry, list | tuple)
            and len(entry) == 2
            and all(isinstance(item, str) and item for item in entry)
        ):
            checkpoint, tokenizer = cast(tuple[str, str], tuple(entry))
            prev = by_checkpoint.get(checkpoint)
            if prev is not None and prev != tokenizer:
                raise ValueError(
                    f"Unsupported TOKENIZER pragma in {axon_file}: conflicting tokenizer for {checkpoint}"
                )
            by_checkpoint[checkpoint] = tokenizer
            return
        raise ValueError(f"Unsupported TOKENIZER pragma in {axon_file}: {raw!r}")

    if isinstance(raw, str):
        _consume(raw)
    elif (
        isinstance(raw, list | tuple)
        and len(raw) == 2
        and all(isinstance(item, str) and item for item in raw)
    ):
        _consume(raw)
    elif (
        isinstance(raw, dict)
        and set(raw) == {"__pragma_occurrences__"}
        and isinstance(raw["__pragma_occurrences__"], list | tuple)
    ):
        for entry in raw["__pragma_occurrences__"]:
            _consume(entry)
    elif isinstance(raw, list | tuple):
        for entry in raw:
            _consume(entry)
    else:
        raise ValueError(f"Unsupported TOKENIZER pragma in {axon_file}: {raw!r}")

    return by_checkpoint.get(checkpoint_id, global_tokenizer)


def _infer_model_task(*, axon_file: Path, weights: Path) -> str:
    pragma_task = _task_pragma_from_axon(axon_file=axon_file)
    if pragma_task is not None:
        return pragma_task
    axon_stem = axon_file.stem.lower()
    model_dir_name = (weights if weights.is_dir() else weights.parent).name.lower()
    masked_lm_stems = {
        "albert",
        "bert",
        "deberta_v2",
        "distilbert",
        "electra",
        "longformer",
        "modernbert",
        "roberta",
    }
    masked_lm_model_dirs = {
        "albert",
        "bert",
        "camembert",
        "deberta_v2",
        "distilbert",
        "electra",
        "longformer",
        "modernbert",
        "roberta",
        "xlm_roberta",
    }
    seq2seq_stem_markers = (
        "t5",
        "mt5",
        "bart",
        "mbart",
        "marian",
        "t5gemma",
    )
    seq2seq_model_dirs = {
        "t5_small",
        "t5_base",
        "t5_large",
        "t5_3b",
        "t5_11b",
        "mt5_small",
        "mt5_base",
        "mt5_large",
        "mt5_xl",
        "mt5_xxl",
        "bart_base",
        "mbart_large_50_m2m",
        "marian_en_de",
        "t5gemma_s_s_ul2",
        "t5gemma2_270m",
    }
    if axon_stem in masked_lm_stems or model_dir_name in masked_lm_model_dirs:
        return "masked_lm"
    if (
        axon_stem in {"t5", "t5_small", "mt5", "bart", "mbart", "marian", "t5gemma", "t5gemma2"}
        or any(marker in axon_stem for marker in seq2seq_stem_markers)
        or model_dir_name in seq2seq_model_dirs
        or model_dir_name.startswith("t5gemma")
    ):
        return "seq2seq_lm"
    return "causal_lm"


def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        current = torch.cuda.current_device()
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        torch.cuda.set_device(current)
    if device.type == "mps":
        torch.mps.empty_cache()


def _extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    logits_attr = getattr(output, "logits", None)
    if torch.is_tensor(logits_attr):
        return logits_attr
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


def _run_phi3small_chunked_logits(
    model: Any,
    *,
    hf_forward_kwargs: dict[str, Any],
    chunk_size: int = 4096,
) -> torch.Tensor:
    model_outputs = model.model(**hf_forward_kwargs, return_dict=True)
    hidden = getattr(model_outputs, "last_hidden_state", None)
    if not torch.is_tensor(hidden):
        hidden = model_outputs[0]
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("Phi-3-small reference is missing output embeddings")
    weight = cast(torch.Tensor, output_embeddings.weight)
    bias = getattr(output_embeddings, "bias", None)
    vocab = int(weight.shape[0])
    chunked_logits: list[torch.Tensor] = []
    for start in range(0, vocab, int(chunk_size)):
        stop = min(vocab, start + int(chunk_size))
        chunk_bias = None
        if torch.is_tensor(bias):
            chunk_bias = bias[start:stop]
        chunk = torch.nn.functional.linear(hidden, weight[start:stop], chunk_bias)
        chunked_logits.append(chunk)
    logits = torch.cat(chunked_logits, dim=-1).float()
    width_multiplier = getattr(model, "mup_width_multiplier", None)
    if width_multiplier:
        logits = logits / float(width_multiplier)
    dummy_mask = getattr(model, "dummy_tokens_mask", None)
    if torch.is_tensor(dummy_mask) and int(dummy_mask.numel()) == int(logits.shape[-1]):
        logits = logits.masked_fill(dummy_mask, torch.finfo(logits.dtype).min)
    return logits


class _DebertaV2ModernMaskedLMReference(torch.nn.Module):
    def __init__(
        self,
        *,
        backbone: Any,
        dense_weight: torch.Tensor,
        dense_bias: torch.Tensor,
        layer_norm_weight: torch.Tensor,
        layer_norm_bias: torch.Tensor,
        decoder_bias: torch.Tensor,
        eps: float,
    ) -> None:
        super().__init__()
        self.deberta = backbone
        hidden = int(dense_weight.shape[0])
        self.lm_dense = torch.nn.Linear(hidden, hidden, bias=True)
        self.lm_dense.weight.data.copy_(dense_weight)
        self.lm_dense.bias.data.copy_(dense_bias)
        self.lm_layer_norm = torch.nn.LayerNorm(hidden, eps=float(eps))
        self.lm_layer_norm.weight.data.copy_(layer_norm_weight)
        self.lm_layer_norm.bias.data.copy_(layer_norm_bias)
        self.lm_bias = torch.nn.Parameter(decoder_bias.clone())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if not torch.is_tensor(hidden):
            hidden = outputs[0]
        hidden = self.lm_dense(hidden)
        hidden = torch.nn.functional.gelu(hidden)
        hidden = self.lm_layer_norm(hidden)
        logits = torch.nn.functional.linear(
            hidden,
            self.deberta.embeddings.word_embeddings.weight,
            self.lm_bias,
        )
        return {"logits": logits}


def _ensure_transformers_import_compat() -> None:
    if hasattr(transformers_import_utils, "is_torch_fx_available"):
        return

    def _is_torch_fx_available() -> bool:
        try:
            import torch.fx  # noqa: F401
        except Exception:
            return False
        return True

    setattr(transformers_import_utils, "is_torch_fx_available", _is_torch_fx_available)


def _patch_cache_api_compat() -> None:
    try:
        from transformers import cache_utils as _cache_utils
    except Exception:
        return

    def _patch_class(cls: Any) -> None:
        if cls is None:
            return
        if not hasattr(cls, "seen_tokens"):

            def _seen_tokens(self: Any) -> int:
                get_seq_length = getattr(self, "get_seq_length", None)
                if callable(get_seq_length):
                    try:
                        return int(get_seq_length())
                    except Exception:
                        return 0
                return 0

            setattr(cls, "seen_tokens", property(_seen_tokens))
        if not hasattr(cls, "get_max_length"):
            setattr(cls, "get_max_length", lambda self: None)
        if not hasattr(cls, "get_usable_length"):

            def _get_usable_length(
                self: Any, new_seq_length: int, layer_idx: int | None = None
            ) -> int:
                del new_seq_length
                get_seq_length = getattr(self, "get_seq_length", None)
                if not callable(get_seq_length):
                    return 0
                try:
                    if layer_idx is not None:
                        return int(get_seq_length(layer_idx))
                except TypeError:
                    pass
                except Exception:
                    return 0
                try:
                    return int(get_seq_length())
                except Exception:
                    return 0

            setattr(cls, "get_usable_length", _get_usable_length)

    _patch_class(getattr(_cache_utils, "Cache", None))
    _patch_class(getattr(_cache_utils, "DynamicCache", None))


def _is_deepseek_family_model_type(model_type: str) -> bool:
    return model_type in {"deepseek", "deepseek_v2", "deepseek_v3", "deepseekv3"}


def _ensure_einops_import_compat() -> None:
    if "einops" in sys.modules:
        return
    try:
        __import__("einops")
        return
    except Exception:
        pass

    shim = ModuleType("einops")
    shim.__spec__ = importlib.machinery.ModuleSpec("einops", loader=None)

    def rearrange(x: torch.Tensor, pattern: str, **axes_lengths: Any) -> torch.Tensor:
        del axes_lengths
        normalized = " ".join(str(pattern).split())
        if normalized != "bs sq group nh hn -> bs sq (group nh) hn":
            raise ImportError(f"Unsupported einops.rearrange pattern shim: {pattern!r}")
        if not torch.is_tensor(x) or x.ndim != 5:
            raise ImportError("einops.rearrange shim expects rank-5 tensor input")
        bs, sq, group, nh, hn = x.shape
        return x.reshape(bs, sq, group * nh, hn)

    setattr(shim, "rearrange", rearrange)
    sys.modules["einops"] = shim


def _checkpoint_contains_any_key(
    safetensors_files: Sequence[Path],
    *,
    prefixes: Sequence[str],
) -> bool:
    normalized = tuple(str(prefix) for prefix in prefixes if str(prefix))
    if not normalized:
        return False
    for path in safetensors_files:
        st = safetensors.safe_open(str(path), framework="pt")
        for key in st.keys():
            if any(str(key).startswith(prefix) for prefix in normalized):
                return True
    return False


def _load_checkpoint_tensor(
    safetensors_files: Sequence[Path],
    key: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    for path in safetensors_files:
        st = safetensors.safe_open(str(path), framework="pt")
        if key not in st.keys():
            continue
        tensor = st.get_tensor(key)
        if tensor.is_floating_point():
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)
    raise KeyError(key)


def _is_deberta_v2_modern_mlm_checkpoint(
    *,
    model_dir: Path,
    model_config: dict[str, Any] | None,
    safetensors_files: Sequence[Path],
) -> bool:
    if not isinstance(model_config, dict):
        return False
    if str(model_config.get("model_type", "")).strip().lower() != "deberta-v2":
        return False
    has_modern = _checkpoint_contains_any_key(
        safetensors_files,
        prefixes=("lm_predictions.lm_head.",),
    )
    has_legacy = _checkpoint_contains_any_key(
        safetensors_files,
        prefixes=("cls.predictions.",),
    )
    return has_modern and not has_legacy


def _load_hf_masked_lm_reference(
    *,
    model_dir: Path,
    safetensors_files: Sequence[Path],
    resolved_dtype: torch.dtype,
    resolved_device: torch.device,
    hf_config: Any,
    trust_remote_code: bool,
    model_config: dict[str, Any] | None,
) -> Any:
    _ensure_transformers_import_compat()
    if _is_deberta_v2_modern_mlm_checkpoint(
        model_dir=model_dir,
        model_config=model_config,
        safetensors_files=safetensors_files,
    ):
        backbone = AutoModel.from_pretrained(
            str(model_dir),
            local_files_only=True,
            dtype=resolved_dtype,
            config=hf_config,
            trust_remote_code=trust_remote_code,
        )
        model = _DebertaV2ModernMaskedLMReference(
            backbone=backbone,
            dense_weight=_load_checkpoint_tensor(
                safetensors_files,
                "lm_predictions.lm_head.dense.weight",
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            dense_bias=_load_checkpoint_tensor(
                safetensors_files,
                "lm_predictions.lm_head.dense.bias",
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            layer_norm_weight=_load_checkpoint_tensor(
                safetensors_files,
                "lm_predictions.lm_head.LayerNorm.weight",
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            layer_norm_bias=_load_checkpoint_tensor(
                safetensors_files,
                "lm_predictions.lm_head.LayerNorm.bias",
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            decoder_bias=_load_checkpoint_tensor(
                safetensors_files,
                "lm_predictions.lm_head.bias",
                device=resolved_device,
                dtype=resolved_dtype,
            ),
            eps=float(getattr(hf_config, "layer_norm_eps", 1.0e-7)),
        )
        return model.to(device=resolved_device, dtype=resolved_dtype).eval()

    model = AutoModelForMaskedLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=resolved_dtype,
        config=hf_config,
        trust_remote_code=trust_remote_code,
    )
    return model.to(device=resolved_device, dtype=resolved_dtype).eval()


def _load_state_dict(
    paths: list[Path],
    *,
    device: torch.device,
    dtype: torch.dtype,
    model_config: dict[str, Any] | None = None,
    param_devices: Sequence[str] | None = None,
    storage_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    if (
        isinstance(model_config, dict)
        and str(model_config.get("model_type", "")).strip().lower() == "deepseek_v4"
        and isinstance(model_config.get("quantization_config"), dict)
        and str(model_config["quantization_config"].get("quant_method", "")).strip().lower()
        == "fp8"
    ):
        return _load_dequantized_deepseek_v4_fp8_state_dict(
            paths,
            device=device,
            dtype=dtype,
            storage_dtype=storage_dtype or dtype,
            param_devices=param_devices,
        )

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
    if param_devices:
        _drop_pipeline_duplicate_state_aliases(out)

    final_logits_bias = out.get("final_logits_bias")
    if (
        torch.is_tensor(final_logits_bias)
        and final_logits_bias.ndim == 2
        and int(final_logits_bias.shape[0]) == 1
    ):
        out.setdefault("final_logits_bias_flat", final_logits_bias.squeeze(0))

    if isinstance(model_config, dict):
        model_type = str(model_config.get("model_type", "")).strip().lower()
        if model_type == "marian" and bool(model_config.get("static_position_embeddings", False)):
            max_positions = int(model_config.get("max_position_embeddings", 0))
            d_model = int(model_config.get("d_model", 0))
            if max_positions > 0 and d_model > 0:
                sentinel = d_model // 2 if (d_model % 2 == 0) else (d_model // 2) + 1
                pos = torch.arange(max_positions, device=device, dtype=torch.float32).unsqueeze(1)
                dims = torch.arange(d_model, device=device, dtype=torch.float32)
                scale = torch.pow(10000.0, (2.0 * torch.floor(dims / 2.0)) / float(d_model))
                position_enc = pos / scale.unsqueeze(0)
                sinusoidal = torch.empty(
                    (max_positions, d_model), device=device, dtype=torch.float32
                )
                sinusoidal[:, 0:sentinel] = torch.sin(position_enc[:, 0::2])
                sinusoidal[:, sentinel:] = torch.cos(position_enc[:, 1::2])
                sinusoidal = sinusoidal.to(dtype=dtype)
                out.setdefault("model.encoder.embed_positions.weight", sinusoidal)
                out.setdefault("model.decoder.embed_positions.weight", sinusoidal)
    return out


def _drop_pipeline_duplicate_state_aliases(state_dict: dict[str, torch.Tensor]) -> None:
    """Drop wrapper-namespace duplicates before pipeline placement.

    Some multimodal checkpoints expose the same text weights under both their
    canonical nested text-module path and a flattened convenience alias. Keeping
    both is harmless for single-device loading but can double large tensors
    during pipeline placement. This is intentionally suffix-based and model
    agnostic: only a shorter key is dropped when a longer key with the same
    suffix and tensor metadata is present.
    """
    by_suffix: dict[str, list[str]] = {}
    for key in state_dict:
        pieces = key.split(".")
        for idx in range(1, len(pieces)):
            by_suffix.setdefault(".".join(pieces[idx:]), []).append(key)

    drop: set[str] = set()
    for key, value in state_dict.items():
        if key in drop:
            continue
        candidates = by_suffix.get(key, ())
        if not candidates:
            continue
        for candidate in candidates:
            if candidate == key or candidate in drop:
                continue
            other = state_dict.get(candidate)
            if not torch.is_tensor(value) or not torch.is_tensor(other):
                continue
            if tuple(value.shape) != tuple(other.shape) or value.dtype != other.dtype:
                continue
            if candidate.count(".") <= key.count("."):
                continue
            drop.add(key)
            break
    for key in drop:
        state_dict.pop(key, None)


def _clone_hf_state_dict(
    model: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, tensor in model.state_dict().items():
        if not torch.is_tensor(tensor):
            continue
        value = tensor.detach()
        if value.is_floating_point():
            value = value.to(device=device, dtype=dtype)
        else:
            value = value.to(device=device)
        out[str(key)] = value
    materialize_mxfp4_aliases(out, dtype=dtype, drop_packed=True)
    return out


def _resolve_safetensors_paths(weights: Path) -> list[Path]:
    if weights.is_file():
        if weights.suffix != ".safetensors":
            raise ValueError(f"Expected a .safetensors file, got: {weights}")
        return [weights]

    if not weights.is_dir():
        raise FileNotFoundError(f"Weights path not found: {weights}")

    def _normalize_pytorch_bins_to_safetensors(model_dir: Path) -> None:
        bin_paths = sorted(model_dir.glob("*.bin"))
        if not bin_paths:
            return
        safetensor_paths = sorted(model_dir.glob("*.safetensors"))
        if safetensor_paths:
            return

        def _extract_tensor_mapping(payload: object) -> dict[str, torch.Tensor] | None:
            if not isinstance(payload, dict):
                return None
            if all(isinstance(k, str) and torch.is_tensor(v) for k, v in payload.items()):
                return {str(k): v for k, v in payload.items()}
            state_dict = payload.get("state_dict")
            if isinstance(state_dict, dict) and all(
                isinstance(k, str) and torch.is_tensor(v) for k, v in state_dict.items()
            ):
                return {str(k): v for k, v in state_dict.items()}
            if len(payload) == 1:
                only_value = next(iter(payload.values()))
                if isinstance(only_value, dict) and all(
                    isinstance(k, str) and torch.is_tensor(v) for k, v in only_value.items()
                ):
                    return {str(k): v for k, v in only_value.items()}
            return None

        for bin_path in bin_paths:
            payload = torch.load(str(bin_path), map_location="cpu", weights_only=False)
            tensor_payload = _extract_tensor_mapping(payload)
            if tensor_payload is None:
                raise RuntimeError(f"Unsupported PyTorch checkpoint payload in {bin_path}")
            tensor_map: dict[str, torch.Tensor] = {}
            for key, value in tensor_payload.items():
                tensor_map[key] = value.detach().cpu().clone().contiguous()
            if not tensor_map:
                raise RuntimeError(f"No tensors found in PyTorch checkpoint {bin_path}")
            out_name = (
                "model.safetensors"
                if bin_path.name == "pytorch_model.bin"
                else f"{bin_path.stem}.safetensors"
            )
            safetensors.torch.save_file(tensor_map, str(model_dir / out_name))

    _normalize_pytorch_bins_to_safetensors(weights)

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
    if all(
        (path.name.startswith("model-") or path.name.startswith("pytorch_model-"))
        and "-of-" in path.name
        for path in candidates
    ):
        return candidates
    raise ValueError(
        f"Multiple .safetensors files found in {weights}; pass an explicit .safetensors file path."
    )


def _load_model_config(model_dir: Path) -> dict[str, Any] | None:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {config_path}, got {type(payload).__name__}")
    return {str(key): value for key, value in payload.items()}


def _infer_tensor_shape_from_checkpoint(
    safetensors_files: Sequence[Path],
    *,
    suffixes: Sequence[str],
) -> tuple[int, ...] | None:
    normalized_suffixes = tuple(str(suffix) for suffix in suffixes if str(suffix))
    if not normalized_suffixes:
        return None
    for path in safetensors_files:
        st = safetensors.safe_open(str(path), framework="pt")
        for key in st.keys():
            if any(str(key).endswith(suffix) for suffix in normalized_suffixes):
                return tuple(int(dim) for dim in st.get_tensor(key).shape)
    return None


def _augment_model_config_from_checkpoint(
    *,
    model_dir: Path,
    safetensors_files: Sequence[Path],
    model_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if isinstance(model_config, dict) and str(model_config.get("model_type", "")).strip().lower() == "deepseek_v4":
        enriched = dict(model_config)
        n_layers = int(enriched.get("num_hidden_layers", 0) or 0)
        if n_layers > 0 and not isinstance(enriched.get("layer_types"), list):
            compress_ratios = enriched.get("compress_ratios")
            ratio_to_layer = {
                0: "sliding_attention",
                4: "compressed_sparse_attention",
                128: "heavily_compressed_attention",
            }
            if isinstance(compress_ratios, list):
                enriched["layer_types"] = [
                    ratio_to_layer[int(item)] for item in compress_ratios[:n_layers]
                ]
            else:
                interleave = [
                    "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
                    for i in range(max(n_layers - 2, 0))
                ]
                enriched["layer_types"] = [
                    *(["heavily_compressed_attention"] * min(n_layers, 2)),
                    *interleave,
                ][:n_layers]
        if n_layers > 0 and not isinstance(enriched.get("mlp_layer_types"), list):
            n_hash_raw = enriched.get("num_hash_layers", 3)
            n_hash = int(n_hash_raw) if isinstance(n_hash_raw, int) and not isinstance(n_hash_raw, bool) else 3
            enriched["mlp_layer_types"] = [
                *(["hash_moe"] * min(n_layers, n_hash)),
                *(["moe"] * max(0, n_layers - n_hash)),
            ]
        return enriched

    if not is_black_mamba_config_dir(model_dir):
        return model_config
    enriched = dict(model_config or {})
    time_step_rank = enriched.get("time_step_rank")
    if isinstance(time_step_rank, int) and not isinstance(time_step_rank, bool):
        return enriched
    dt_proj_shape = _infer_tensor_shape_from_checkpoint(
        safetensors_files,
        suffixes=("dt_proj.weight",),
    )
    if dt_proj_shape is not None and len(dt_proj_shape) >= 2:
        enriched["time_step_rank"] = int(dt_proj_shape[1])
    return enriched


def _build_phi3small_dummy_vocab_mask(
    *,
    model_config: dict[str, Any] | None,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if not isinstance(model_config, dict):
        return None
    model_type = str(model_config.get("model_type", "")).strip().lower()
    if model_type != "phi3small":
        return None

    # Matches tokenization_phi3_small.py constants.
    # Real vocab content ends at 100275; many higher ids are dummy padding ids.
    mask = torch.zeros((int(vocab_size),), dtype=torch.bool, device=device)
    fixed_dummy_ids = {
        100256,  # dummy_id_2
        100258,  # fim_prefix
        100259,  # fim_middle
        100260,  # fim_suffix
        100264,  # dummy_id_0
        100265,  # dummy_id_1
        100276,  # endofprompt
    }
    fixed_dummy_ids.update(range(100267, 100276))  # dummy_id_3..dummy_id_11
    for idx in fixed_dummy_ids:
        if 0 <= idx < int(vocab_size):
            mask[idx] = True
    if int(vocab_size) > 100277:
        mask[100277 : int(vocab_size)] = True
    return mask


def _rebuild_hf_dummy_tokens_mask_from_config(model: Any) -> bool:
    """Rebuild custom-model dummy token mask buffers from config, if present.

    Some trust_remote_code loading paths can leave non-persistent buffers in an
    invalid state. For Phi-3-small this affects `dummy_tokens_mask` and can
    produce catastrophic `finfo.min` logits. Rebuilding from config keeps HF
    and Axon comparisons stable.
    """
    if not hasattr(model, "dummy_tokens_mask") or not hasattr(model, "config"):
        return False

    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "vocab_size"):
        return False
    raw_indices = getattr(config, "dummy_token_indices", None)
    if raw_indices is None:
        return False

    vocab_size = int(getattr(config, "vocab_size"))
    expected = torch.zeros((vocab_size,), dtype=torch.bool)

    if torch.is_tensor(raw_indices):
        if raw_indices.ndim == 0:
            candidate_indices = [int(raw_indices.item())]
        else:
            candidate_indices = [int(x) for x in raw_indices.reshape(-1).tolist()]
    else:
        candidate_indices = [int(x) for x in list(raw_indices)]

    for idx in candidate_indices:
        if 0 <= idx < vocab_size:
            expected[idx] = True

    current = getattr(model, "dummy_tokens_mask")
    if not torch.is_tensor(current):
        setattr(model, "dummy_tokens_mask", expected)
        return True

    expected = expected.to(device=current.device)
    needs_update = (
        current.dtype != torch.bool
        or tuple(current.shape) != tuple(expected.shape)
        or not torch.equal(current, expected)
    )
    if needs_update:
        setattr(model, "dummy_tokens_mask", expected)
        return True
    return False


def _phi3small_longrope_buffer_is_invalid(
    tensor: Any,
    *,
    expected_shape: tuple[int, ...],
    allow_zero: bool,
    expected: torch.Tensor | None = None,
) -> bool:
    if not torch.is_tensor(tensor):
        return True
    if tuple(int(dim) for dim in tensor.shape) != expected_shape:
        return True
    if tensor.numel() == 0:
        return True
    data = tensor.detach().float()
    if bool(torch.isnan(data).any()) or bool(torch.isinf(data).any()):
        return True
    if not allow_zero and float(data.abs().max()) == 0.0:
        return True
    if expected is not None:
        expected_data = expected.detach().float()
        if data.shape != expected_data.shape:
            return True
        if not torch.allclose(data.cpu(), expected_data.cpu(), atol=0.0, rtol=0.0):
            return True
    return False


def _rebuild_hf_phi3small_longrope_buffers(model: Any) -> int:
    """Rebuild Phi-3-small LongRoPE buffers from config when HF load corrupts them."""

    rebuilt = 0
    for module in model.modules():
        if not bool(getattr(module, "is_longrope", False)):
            continue
        longrope_config = getattr(module, "longrope_config", None)
        if longrope_config is None:
            continue
        max_seq_len = getattr(module, "max_seq_len", None)
        dim_model = getattr(module, "dim_model", None)
        short_factor = getattr(longrope_config, "short_factor", None)
        long_factor = getattr(longrope_config, "long_factor", None)
        if (
            not isinstance(max_seq_len, int)
            or max_seq_len <= 0
            or not isinstance(dim_model, int)
            or dim_model <= 0
            or not isinstance(short_factor, list)
            or not isinstance(long_factor, list)
        ):
            continue
        target_device = torch.device("cpu")
        for attr_name in ("range_vector", "short_factors", "long_factors"):
            value = getattr(module, attr_name, None)
            if torch.is_tensor(value):
                target_device = value.device
                break
        if target_device.type == "meta":
            target_device = torch.device("cpu")
        expected_half_dim = dim_model // 2
        expected_range = torch.arange(max_seq_len, device=target_device, dtype=torch.float32)
        expected_short = torch.tensor(short_factor, device=target_device, dtype=torch.float32)
        expected_long = torch.tensor(long_factor, device=target_device, dtype=torch.float32)
        if expected_short.shape != (expected_half_dim,) or expected_long.shape != (
            expected_half_dim,
        ):
            continue
        invalid = (
            _phi3small_longrope_buffer_is_invalid(
                getattr(module, "range_vector", None),
                expected_shape=(max_seq_len,),
                allow_zero=True,
                expected=expected_range,
            )
            or _phi3small_longrope_buffer_is_invalid(
                getattr(module, "short_factors", None),
                expected_shape=(expected_half_dim,),
                allow_zero=False,
                expected=expected_short,
            )
            or _phi3small_longrope_buffer_is_invalid(
                getattr(module, "long_factors", None),
                expected_shape=(expected_half_dim,),
                allow_zero=False,
                expected=expected_long,
            )
        )
        if not invalid:
            continue
        setattr(module, "range_vector", expected_range)
        setattr(module, "short_factors", expected_short)
        setattr(module, "long_factors", expected_long)
        rebuilt += 1
    return rebuilt


def _checkpoint_has_explicit_output_head_weight(model_dir: Path) -> bool:
    output_head_suffixes = ("lm_head.weight", "embed_out.weight", "head.weight")

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        weight_map = payload.get("weight_map")
        if isinstance(weight_map, dict):
            return any(
                any(str(key).endswith(suffix) for suffix in output_head_suffixes)
                for key in weight_map
            )
        return False
    for shard_path in _resolve_safetensors_paths(model_dir):
        st = safetensors.safe_open(str(shard_path), framework="pt")
        for key in st.keys():
            if any(str(key).endswith(suffix) for suffix in output_head_suffixes):
                return True
    return False


def _prime_tiktoken_cache_from_model_dir(model_dir: Path) -> None:
    cl100k_path = model_dir / "cl100k_base.tiktoken"
    if not cl100k_path.exists():
        return
    cache_dir = model_dir.parent / ".tiktoken_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = hashlib.sha1(
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken".encode()
    ).hexdigest()
    cache_target = cache_dir / cache_name
    if not cache_target.exists():
        cache_target.write_bytes(cl100k_path.read_bytes())
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache_dir))


def _has_local_custom_code_artifacts(model_dir: Path) -> bool:
    for pattern in ("configuration_*.py", "modeling_*.py", "tokenization_*.py"):
        if any(model_dir.glob(pattern)):
            return True
    return False


def _iter_auto_map_module_names(auto_map: Any) -> list[str]:
    if not isinstance(auto_map, dict):
        return []
    modules: list[str] = []
    for value in auto_map.values():
        items = list(value) if isinstance(value, list | tuple) else [value]
        for item in items:
            if not isinstance(item, str):
                continue
            raw_ref = item.split("--", 1)[-1].strip()
            if not raw_ref:
                continue
            module_name = raw_ref.split(".", 1)[0].strip()
            if module_name:
                modules.append(module_name)
    return modules


def _should_trust_remote_code(
    model_dir: Path,
    *,
    model_config: dict[str, Any] | None,
) -> bool:
    model_name = model_dir.name.strip().lower()
    normalized_model_name = re.sub(r"[^a-z0-9]+", "_", model_name).strip("_")
    normalized_model_name = normalized_model_name.replace("phi_3", "phi3")
    if model_name == "deepseek_v2_lite":
        return False
    if normalized_model_name.startswith("deepseek_v2_lite"):
        return False
    if normalized_model_name.startswith("deepseek_coder_v2_lite"):
        return False
    if normalized_model_name.startswith("phi3_mini_") or normalized_model_name.startswith(
        "phi3_medium_"
    ):
        return False
    if normalized_model_name.startswith("phi3_small_"):
        return True
    has_local_artifacts = _has_local_custom_code_artifacts(model_dir)
    if not isinstance(model_config, dict):
        return has_local_artifacts
    auto_map = model_config.get("auto_map")
    module_names = _iter_auto_map_module_names(auto_map)
    if module_names:
        return all((model_dir / f"{module_name}.py").exists() for module_name in module_names)
    return has_local_artifacts


_GENERATION_MIXIN_CLASS_CACHE: dict[type[Any], type[Any]] = {}


def _generation_config_from_model_config(config: Any) -> GenerationConfig:
    try:
        return GenerationConfig.from_model_config(config)
    except AttributeError as exc:
        if "to_dict" not in str(exc):
            raise
    generation_config = GenerationConfig()
    for name in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "decoder_start_token_id",
        "max_length",
        "min_length",
    ):
        if hasattr(config, name):
            setattr(generation_config, name, getattr(config, name))
    return generation_config


def _ensure_hf_generation_mixin(model: Any) -> bool:
    added = False
    if not hasattr(model, "generate"):
        if not hasattr(model, "prepare_inputs_for_generation"):
            return False
        cls = model.__class__
        mixed_cls = _GENERATION_MIXIN_CLASS_CACHE.get(cls)
        if mixed_cls is None:
            mixed_cls = type(
                f"{cls.__name__}WithGenerationMixin",
                (cls, GenerationMixin),
                {"__module__": cls.__module__},
            )
            _GENERATION_MIXIN_CLASS_CACHE[cls] = mixed_cls
        model.__class__ = mixed_cls
        added = True
    if not hasattr(model, "generation_config") or model.generation_config is None:
        config = getattr(model, "config", None)
        if config is not None:
            model.generation_config = _generation_config_from_model_config(config)
    return added


def _normalize_rope_numeric_fields(config: Any) -> Any:
    def _normalize_dict(mapping: Any) -> None:
        if not isinstance(mapping, dict):
            return
        rope_type = mapping.get("rope_type")
        if isinstance(rope_type, str) and "type" not in mapping:
            mapping["type"] = rope_type
        if (
            "type" not in mapping
            and "rope_type" not in mapping
            and isinstance(mapping.get("factor"), int | float)
            and not isinstance(mapping.get("factor"), bool)
        ):
            # Older DeepSeek configs can carry only {"factor": ...}; HF remote code
            # then indexes rope_scaling["type"] directly.
            mapping["type"] = "linear"
        for key in ("factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim"):
            value = mapping.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                mapping[key] = float(value)

    rope_scaling = getattr(config, "rope_scaling", None)
    _normalize_dict(rope_scaling)
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
        has_longrope_factors = "short_factor" in rope_scaling and "long_factor" in rope_scaling
        raw_theta = rope_scaling.get("rope_theta")
        has_nondefault_theta = (
            isinstance(raw_theta, int | float)
            and not isinstance(raw_theta, bool)
            and not math.isclose(float(raw_theta), 10000.0)
        )
        if rope_type == "default" and not has_longrope_factors and not has_nondefault_theta:
            setattr(config, "rope_scaling", None)
            if str(getattr(config, "model_type", "")).strip().lower() == "phi3small":
                if hasattr(config, "rope_parameters"):
                    setattr(config, "rope_parameters", None)
                return config
            if hasattr(config, "rope_parameters"):
                setattr(config, "rope_parameters", None)
            rope_scaling = None
    rope_scaling_theta = None
    if isinstance(rope_scaling, dict):
        raw_theta = rope_scaling.get("rope_theta")
        if isinstance(raw_theta, int | float) and not isinstance(raw_theta, bool):
            rope_scaling_theta = float(raw_theta)
    if (
        isinstance(rope_scaling, dict)
        and rope_scaling.get("type", rope_scaling.get("rope_type")) == "default"
        and set(rope_scaling.keys()) <= {"rope_theta", "rope_type", "type"}
        and (rope_scaling_theta is None or math.isclose(rope_scaling_theta, 10000.0))
    ):
        setattr(config, "rope_scaling", None)
        rope_scaling = None
    rope_parameters = getattr(config, "rope_parameters", None)
    _normalize_dict(rope_parameters)
    if not isinstance(rope_parameters, dict):
        if (
            str(getattr(config, "model_type", "")).strip().lower() == "phi3small"
            and not isinstance(rope_scaling, dict)
        ):
            if hasattr(config, "rope_parameters"):
                setattr(config, "rope_parameters", None)
            return config
        if isinstance(rope_scaling, dict):
            rope_parameters = dict(rope_scaling)
            rope_type = rope_parameters.get("type", rope_parameters.get("rope_type"))
            if isinstance(rope_type, str):
                rope_parameters["rope_type"] = rope_type
        else:
            rope_parameters = {"rope_type": "default"}
        setattr(config, "rope_parameters", rope_parameters)
    rope_theta = getattr(config, "rope_theta", None)
    if isinstance(rope_parameters, dict) and "rope_theta" not in rope_parameters:
        if isinstance(rope_theta, int | float) and not isinstance(rope_theta, bool):
            rope_parameters["rope_theta"] = float(rope_theta)
        else:
            rope_parameters["rope_theta"] = 10000.0
    original_ctx = getattr(config, "original_max_position_embeddings", None)
    if isinstance(original_ctx, int) and not isinstance(original_ctx, bool):
        for mapping in (rope_scaling, rope_parameters):
            if not isinstance(mapping, dict):
                continue
            rope_type = mapping.get("rope_type", mapping.get("type"))
            if rope_type not in {"longrope", "su"}:
                continue
            mapping.setdefault("original_max_position_embeddings", original_ctx)
    return config


def _patch_mistral4_config_compat(config: Any) -> Any:
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return config
    if str(getattr(config, "model_type", "")).strip().lower() != "mistral3":
        return config
    if str(getattr(text_config, "model_type", "")).strip().lower() != "mistral4":
        return config
    if not hasattr(text_config, "num_experts"):
        routed = getattr(text_config, "n_routed_experts", None)
        if isinstance(routed, int) and not isinstance(routed, bool):
            setattr(text_config, "num_experts", int(routed))
    if not hasattr(text_config, "num_local_experts"):
        routed = getattr(text_config, "n_routed_experts", None)
        if isinstance(routed, int) and not isinstance(routed, bool):
            setattr(text_config, "num_local_experts", int(routed))
    # transformers.integrations.moe supports eager, batched_mm, and grouped_mm expert dispatch.
    # For this FP8 checkpoint we want the dequantized float32 reference path, and grouped_mm still
    # routes through fp8/static-specific kernels. Force eager only for mistral4 text configs.
    setattr(text_config, "_experts_implementation", "eager")
    return config


def _get_nested_attr(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _iter_hf_layer_stack_candidates(model: Any) -> list[Any]:
    stacks: list[Any] = []
    for path in (
        "model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "encoder.layers",
        "encoder.block",
        "decoder.layers",
        "decoder.block",
    ):
        layers = _get_nested_attr(model, path)
        if layers is not None:
            stacks.append(layers)
    return stacks


def _patch_hf_shared_modules_for_device_map(model: Any) -> list[Any]:
    handles: list[Any] = []
    seen_modules: set[int] = set()
    for module in model.modules():
        if not isinstance(module, torch.nn.Embedding):
            continue
        weight = getattr(module, "weight", None)
        hook = getattr(module, "_hf_hook", None)
        if torch.is_tensor(weight) and hook is not None and hasattr(hook, "execution_device"):
            with suppress(Exception):
                hook.execution_device = weight.device

        def _move_embedding_input(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            weight = getattr(module, "weight", None)
            if not torch.is_tensor(weight):
                return args, kwargs
            args_out = args
            kwargs_out = kwargs
            if args and torch.is_tensor(args[0]) and args[0].device != weight.device:
                args_out = (args[0].to(device=weight.device), *args[1:])
            input_value = kwargs.get("input")
            if torch.is_tensor(input_value) and input_value.device != weight.device:
                kwargs_out = dict(kwargs)
                kwargs_out["input"] = input_value.to(device=weight.device)
            return args_out, kwargs_out

        handles.append(module.register_forward_pre_hook(_move_embedding_input, with_kwargs=True))

    for path in ("model.encoder", "model.decoder", "encoder", "decoder"):
        module = _get_nested_attr(model, path)
        if module is None or id(module) in seen_modules:
            continue
        seen_modules.add(id(module))
        target_device: torch.device | None = None
        embed_tokens = getattr(module, "embed_tokens", None)
        target_device = _module_parameter_device(embed_tokens)
        if target_device is None:
            target_device = _module_parameter_device(module)
        if target_device is None:
            continue

        def _move_stack_inputs(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            *,
            _target_device: torch.device = target_device,
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            del module
            args_out = args
            if args and torch.is_tensor(args[0]) and args[0].device != _target_device:
                args_out = (args[0].to(device=_target_device), *args[1:])
            kwargs_out = kwargs
            for key in (
                "input_ids",
                "attention_mask",
                "decoder_attention_mask",
                "position_ids",
                "cache_position",
            ):
                value = kwargs_out.get(key)
                if torch.is_tensor(value) and value.device != _target_device:
                    if kwargs_out is kwargs:
                        kwargs_out = dict(kwargs)
                    kwargs_out[key] = value.to(device=_target_device)
            encoder_hidden_states = kwargs_out.get("encoder_hidden_states")
            encoder_attention_mask = kwargs_out.get("encoder_attention_mask")
            if (
                torch.is_tensor(encoder_hidden_states)
                and torch.is_tensor(encoder_attention_mask)
                and encoder_attention_mask.device != encoder_hidden_states.device
            ):
                if kwargs_out is kwargs:
                    kwargs_out = dict(kwargs)
                kwargs_out["encoder_attention_mask"] = encoder_attention_mask.to(
                    device=encoder_hidden_states.device
                )
            return args_out, kwargs_out

        handles.append(module.register_forward_pre_hook(_move_stack_inputs, with_kwargs=True))

    layer_iter: list[Any] = []
    for layers in _iter_hf_layer_stack_candidates(model):
        try:
            layer_iter.extend(list(layers))
        except Exception:
            continue
    if not layer_iter:
        return handles

    for layer in layer_iter:

        def _move_shared_kwargs(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            target_device = None
            target_dtype: torch.dtype | None = None
            for value in module.parameters():
                target_device = value.device
                target_dtype = value.dtype if value.is_floating_point() else None
                break
            if target_device is None:
                hidden_states = args[0] if args else kwargs.get("hidden_states")
                if not torch.is_tensor(hidden_states):
                    return args, kwargs
                target_device = hidden_states.device
                target_dtype = hidden_states.dtype if hidden_states.is_floating_point() else None
            args_out = args
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(args) >= 6:
                position_embeddings = args[5]
            if (
                isinstance(position_embeddings, tuple)
                and len(position_embeddings) == 2
                and all(torch.is_tensor(item) for item in position_embeddings)
            ):
                moved_position_embeddings = tuple(
                    item.to(
                        device=target_device,
                        dtype=(
                            target_dtype
                            if target_dtype is not None and item.is_floating_point()
                            else item.dtype
                        ),
                    )
                    if (item.device != target_device)
                    or (
                        target_dtype is not None
                        and item.is_floating_point()
                        and item.dtype != target_dtype
                    )
                    else item
                    for item in position_embeddings
                )
                if len(args) >= 6:
                    args_list = list(args)
                    args_list[5] = moved_position_embeddings
                    args_out = tuple(args_list)
                kwargs_out = dict(kwargs)
                kwargs_out["position_embeddings"] = moved_position_embeddings
            else:
                kwargs_out = kwargs
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None and len(args) >= 2:
                attention_mask = args[1]
            if torch.is_tensor(attention_mask) and attention_mask.device != target_device:
                if len(args) >= 2:
                    args_list = list(args_out)
                    args_list[1] = attention_mask.to(target_device)
                    args_out = tuple(args_list)
                if kwargs_out is kwargs:
                    kwargs_out = dict(kwargs)
                kwargs_out["attention_mask"] = attention_mask.to(target_device)
            return args_out, kwargs_out

        handles.append(layer.register_forward_pre_hook(_move_shared_kwargs, with_kwargs=True))
    return handles


@contextmanager
def _patch_transformers_mask_device_map_inputs(enabled: bool) -> Any:
    if not enabled:
        yield
        return
    try:
        import transformers.masking_utils as masking_utils
    except Exception:
        yield
        return
    original = getattr(masking_utils, "create_bidirectional_mask", None)
    original_preprocess = getattr(masking_utils, "_preprocess_mask_arguments", None)
    if not callable(original):
        yield
        return

    def _create_bidirectional_mask_device_safe(*args: Any, **kwargs: Any) -> Any:
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None and len(args) >= 2:
            inputs_embeds = args[1]
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        if encoder_hidden_states is None and len(args) >= 4:
            encoder_hidden_states = args[3]
        attention_mask = kwargs.get("attention_mask")
        attention_mask_arg_index: int | None = None
        if attention_mask is None and len(args) >= 3:
            attention_mask = args[2]
            attention_mask_arg_index = 2
        reference = encoder_hidden_states if torch.is_tensor(encoder_hidden_states) else inputs_embeds
        if torch.is_tensor(attention_mask) and torch.is_tensor(reference):
            # Transformers' SDPA mask helpers close over 2D padding masks and index
            # them with helper tensors allocated from a separate reference device.
            # CPU padding masks are valid for indexing from any device and avoid
            # device-map splits leaking into the closure.
            target_device = (
                torch.device("cpu")
                if attention_mask.ndim == 2
                else reference.device
            )
            moved_attention_mask = (
                attention_mask.to(device=target_device)
                if attention_mask.device != target_device
                else attention_mask
            )
        else:
            moved_attention_mask = attention_mask
        if moved_attention_mask is not attention_mask:
            if attention_mask_arg_index is not None:
                args_list = list(args)
                args_list[attention_mask_arg_index] = moved_attention_mask
                args = tuple(args_list)
            else:
                kwargs = dict(kwargs)
                kwargs["attention_mask"] = moved_attention_mask
        return original(*args, **kwargs)

    patched: list[tuple[Any, str, Any]] = []
    if callable(original_preprocess):

        def _preprocess_mask_arguments_device_safe(*args: Any, **kwargs: Any) -> Any:
            result = original_preprocess(*args, **kwargs)
            if not (
                isinstance(result, tuple)
                and len(result) >= 2
                and torch.is_tensor(result[1])
                and result[1].ndim == 2
            ):
                return result
            encoder_hidden_states = kwargs.get("encoder_hidden_states")
            if encoder_hidden_states is None and len(args) >= 7:
                encoder_hidden_states = args[6]
            if (
                torch.is_tensor(encoder_hidden_states)
                and result[1].device != encoder_hidden_states.device
            ):
                result_list = list(result)
                result_list[1] = result[1].to(device=encoder_hidden_states.device)
                return tuple(result_list)
            return result

        try:
            setattr(
                masking_utils,
                "_preprocess_mask_arguments",
                _preprocess_mask_arguments_device_safe,
            )
            patched.append((masking_utils, "_preprocess_mask_arguments", original_preprocess))
        except Exception:
            pass
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        module_dict = getattr(module, "__dict__", None)
        if not isinstance(module_dict, dict):
            continue
        value = module_dict.get("create_bidirectional_mask")
        if value is original:
            try:
                setattr(module, "create_bidirectional_mask", _create_bidirectional_mask_device_safe)
            except Exception:
                continue
            patched.append((module, "create_bidirectional_mask", original))
    try:
        yield
    finally:
        for module, name, value in patched:
            with suppress(Exception):
                setattr(module, name, value)


@contextmanager
def _patch_torch_equal_for_cross_device_hf_load(enabled: bool) -> Any:
    if not enabled:
        yield
        return
    original_equal = torch.equal

    def _equal_device_safe(left: Any, right: Any) -> bool:
        if (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.device != right.device
        ):
            return bool(original_equal(left.detach().cpu(), right.detach().cpu()))
        return bool(original_equal(left, right))

    torch.equal = _equal_device_safe  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.equal = original_equal  # type: ignore[assignment]


def _force_model_floating_dtype(model: Any, *, target_dtype: torch.dtype) -> tuple[int, int]:
    coerced_params = 0
    coerced_buffers = 0
    with torch.no_grad():
        for param in model.parameters():
            data = param.data
            if data.is_floating_point() and data.dtype != target_dtype:
                param.data = data.to(dtype=target_dtype)
                coerced_params += 1
        for module in model.modules():
            buffers = getattr(module, "_buffers", None)
            if not isinstance(buffers, dict):
                continue
            for name, value in list(buffers.items()):
                if not torch.is_tensor(value):
                    continue
                if not value.is_floating_point() or value.dtype == target_dtype:
                    continue
                buffers[name] = value.to(dtype=target_dtype)
                coerced_buffers += 1
    return coerced_params, coerced_buffers


def _is_mistral4_text_config(config: Any) -> bool:
    text_config = getattr(config, "text_config", None)
    return (
        text_config is not None
        and str(getattr(config, "model_type", "")).strip().lower() == "mistral3"
        and str(getattr(text_config, "model_type", "")).strip().lower() == "mistral4"
    )


def _patch_hf_mistral4_experts_from_checkpoint(
    model: Any,
    *,
    config: Any,
    safetensors_files: Sequence[Path],
    resolved_dtype: torch.dtype,
    resolved_device: torch.device,
) -> int:
    if not _is_mistral4_text_config(config):
        return 0

    text_model = getattr(getattr(model, "model", None), "language_model", None)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        return 0

    patched = 0
    with torch.no_grad():
        for layer_idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            experts = getattr(mlp, "experts", None)
            if experts is None:
                continue
            for local_name in ("gate_up_proj", "down_proj"):
                weight_key = f"language_model.model.layers.{layer_idx}.mlp.experts.{local_name}"
                scale_key = f"{weight_key}_scale_inv"
                try:
                    weight = _load_checkpoint_tensor(
                        safetensors_files,
                        weight_key,
                        device=resolved_device,
                        dtype=resolved_dtype,
                    )
                    scale = _load_checkpoint_tensor(
                        safetensors_files,
                        scale_key,
                        device=resolved_device,
                        dtype=resolved_dtype,
                    )
                except KeyError:
                    continue
                dequantized = weight * scale
                target = getattr(experts, local_name, None)
                if not torch.is_tensor(target):
                    continue
                target.data.copy_(dequantized.to(device=target.device, dtype=target.dtype))
                patched += 1
    return patched


def _patch_rope_payload_for_compat(
    payload: dict[str, Any],
    *,
    error_text: str,
) -> bool:
    changed = False
    original_ctx = payload.get("original_max_position_embeddings")
    for field_name in ("rope_scaling", "rope_parameters"):
        field = payload.get(field_name)
        if not isinstance(field, dict):
            continue
        rope_type = field.get("rope_type", field.get("type"))
        if isinstance(rope_type, str) and "type" not in field:
            field["type"] = rope_type
            changed = True
        if (
            isinstance(original_ctx, int)
            and not isinstance(original_ctx, bool)
            and rope_type in {"longrope", "su"}
            and "original_max_position_embeddings" not in field
            and "original_max_position_embeddings" in error_text
        ):
            field["original_max_position_embeddings"] = int(original_ctx)
            changed = True
        if "must be a dictionary with three fields" in error_text:
            allowed = {"type", "short_factor", "long_factor"}
            extra_keys = [key for key in list(field.keys()) if key not in allowed]
            if extra_keys:
                for key in extra_keys:
                    field.pop(key, None)
                changed = True
    return changed


def _load_auto_config_with_compat_fallback(model_dir: Path, *, trust_remote_code: bool) -> Any:
    _ensure_transformers_import_compat()
    compat_exc: Exception | None = None
    try:
        return AutoConfig.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
    except (KeyError, ValueError) as exc:
        error_text = str(exc)
        if (
            "original_max_position_embeddings" not in error_text
            and "must be a dictionary with three fields" not in error_text
        ):
            raise
        compat_exc = exc

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise compat_exc or RuntimeError("Unable to load model config")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise compat_exc or ValueError(
            f"Expected mapping in {config_path}, got {type(payload).__name__}"
        )
    changed = _patch_rope_payload_for_compat(payload, error_text=str(compat_exc))
    if not changed:
        raise compat_exc or RuntimeError("No compatible rope settings to patch")

    with TemporaryDirectory(prefix="axon_hf_config_patch_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        tmp_config = tmp_dir_path / "config.json"
        tmp_config.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        if trust_remote_code:
            for py_file in model_dir.glob("*.py"):
                shutil.copy2(py_file, tmp_dir_path / py_file.name)
        return AutoConfig.from_pretrained(
            str(tmp_config.parent),
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )


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


def _build_reference_quantization_config(config: Any) -> Any | None:
    quant_method = _read_quant_method(config)
    if quant_method not in {"mxfp4", "fp8"}:
        return None
    quant_cfg = getattr(config, "quantization_config", None)
    modules_to_not_convert: list[str] | None = None
    if isinstance(quant_cfg, dict):
        raw = quant_cfg.get("modules_to_not_convert")
        if isinstance(raw, list):
            modules_to_not_convert = [str(item) for item in raw]
    if quant_method == "fp8":
        return FineGrainedFP8Config(
            activation_scheme="dynamic",
            modules_to_not_convert=modules_to_not_convert,
            dequantize=True,
        )
    return Mxfp4Config(
        modules_to_not_convert=modules_to_not_convert,
        dequantize=True,
    )


def _patch_deepseek_v4_reference_runtime_config(config: Any) -> Any:
    """Use deterministic eager HF reference kernels for DeepSeek-V4 comparisons."""
    if str(getattr(config, "model_type", "")).strip().lower() != "deepseek_v4":
        return config
    setattr(config, "_attn_implementation", "eager")
    setattr(config, "_experts_implementation", "eager")
    return config


_HF_MXFP4_EXPERT_ALIAS_RE = re.compile(r"\.mlp\.experts\.\d+\.")


_DEEPSEEK_V4_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\.(?P<proj>w[123])\.weight$"
)
_DEEPSEEK_V4_FP4_E2M1_LUT = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _hf_dequantized_mxfp4_state_key(key: str) -> str | None:
    if _HF_MXFP4_EXPERT_ALIAS_RE.search(key):
        return None
    suffix_map = {
        ".mlp.experts.gate_up_proj.weight": ".mlp.experts.gate_up_proj",
        ".mlp.experts.down_proj.weight": ".mlp.experts.down_proj",
        ".mlp.experts.gate_up_proj.bias": ".mlp.experts.gate_up_proj_bias",
        ".mlp.experts.down_proj.bias": ".mlp.experts.down_proj_bias",
    }
    for suffix, replacement in suffix_map.items():
        if key.endswith(suffix):
            return f"{key[: -len(suffix)]}{replacement}"
    return key


def _load_dequantized_mxfp4_hf_state_dict(
    paths: list[Path],
    *,
    dtype: torch.dtype,
    model_config: dict[str, Any] | None,
) -> dict[str, torch.Tensor]:
    raw = _load_state_dict(
        paths,
        device=torch.device("cpu"),
        dtype=dtype,
        model_config=model_config,
    )
    remapped: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        mapped = _hf_dequantized_mxfp4_state_key(key)
        if mapped is None:
            continue
        remapped.setdefault(mapped, value)
    raw.clear()
    return remapped


def _device_for_hf_tensor_name(
    name: str,
    *,
    device_map: dict[str, str] | None,
    default_device: torch.device,
) -> torch.device:
    if not device_map:
        return default_device
    best_prefix: str | None = None
    best_device: str | None = None
    for prefix, device in device_map.items():
        if name == prefix or name.startswith(prefix + "."):
            if best_prefix is None or len(prefix) > len(best_prefix):
                best_prefix = prefix
                best_device = device
    if best_device is None:
        return default_device
    return torch.device(best_device)


def _load_hf_causal_lm_from_dequantized_mxfp4_state(
    *,
    safetensors_files: list[Path],
    dtype: torch.dtype,
    hf_config: Any,
    trust_remote_code: bool,
    device_map: dict[str, str] | None,
    target_device: torch.device,
    model_config: dict[str, Any] | None,
) -> Any:
    plain_config = _patch_deepseek_v4_reference_runtime_config(deepcopy(hf_config))
    if hasattr(plain_config, "quantization_config"):
        try:
            delattr(plain_config, "quantization_config")
        except Exception:
            setattr(plain_config, "quantization_config", None)

    state = _load_dequantized_mxfp4_hf_state_dict(
        safetensors_files,
        dtype=dtype,
        model_config=model_config,
    )
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            plain_config,
            trust_remote_code=trust_remote_code,
        )

    expected = set(dict(model.named_parameters()).keys())
    expected.update(dict(model.named_buffers()).keys())
    loaded: set[str] = set()
    for name, value in state.items():
        if name not in expected:
            continue
        device = _device_for_hf_tensor_name(
            name,
            device_map=device_map,
            default_device=target_device,
        )
        set_module_tensor_to_device(model, name, device, value=value)
        loaded.add(name)
    state.clear()

    missing = sorted(
        name
        for name, parameter in model.named_parameters()
        if name not in loaded and getattr(parameter, "device", None).type == "meta"
    )
    if missing:
        has_explicit_lm_head = any(name == "lm_head.weight" for name in loaded)
        if not has_explicit_lm_head and missing == ["lm_head.weight"]:
            input_weight = model.get_input_embeddings().weight
            model.get_output_embeddings().weight = input_weight
        else:
            preview = ", ".join(missing[:8])
            suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
            raise RuntimeError(f"dequantized MXFP4 HF load left meta parameters: {preview}{suffix}")

    if device_map is not None:
        model = dispatch_model(model, device_map=device_map)

    return model.eval()


def _deepseek_v4_hf_base_key(raw_key: str) -> str:
    key = raw_key
    replacements = [
        (r"^embed\.weight$", "embed_tokens.weight"),
        (r"^head\.weight$", "lm_head.weight"),
        (r"^norm\.weight$", "norm.weight"),
        (r"^hc_head_fn$", "hc_head.hc_fn"),
        (r"^hc_head_base$", "hc_head.hc_base"),
        (r"^hc_head_scale$", "hc_head.hc_scale"),
        (r"^layers\.(\d+)\.attn_norm\.", r"layers.\1.input_layernorm."),
        (r"^layers\.(\d+)\.ffn_norm\.", r"layers.\1.post_attention_layernorm."),
        (r"^layers\.(\d+)\.hc_attn_fn$", r"layers.\1.attn_hc.fn"),
        (r"^layers\.(\d+)\.hc_attn_base$", r"layers.\1.attn_hc.base"),
        (r"^layers\.(\d+)\.hc_attn_scale$", r"layers.\1.attn_hc.scale"),
        (r"^layers\.(\d+)\.hc_ffn_fn$", r"layers.\1.ffn_hc.fn"),
        (r"^layers\.(\d+)\.hc_ffn_base$", r"layers.\1.ffn_hc.base"),
        (r"^layers\.(\d+)\.hc_ffn_scale$", r"layers.\1.ffn_hc.scale"),
        (r"^layers\.(\d+)\.attn\.", r"layers.\1.self_attn."),
        (r"^layers\.(\d+)\.ffn\.", r"layers.\1.mlp."),
        (r"^layers\.(\d+)\.self_attn\.attn_sink$", r"layers.\1.self_attn.sinks"),
        (
            r"^layers\.(\d+)\.self_attn\.indexer\.compressor\.norm\.",
            r"layers.\1.self_attn.compressor.indexer.kv_norm.",
        ),
        (
            r"^layers\.(\d+)\.self_attn\.indexer\.compressor\.ape$",
            r"layers.\1.self_attn.compressor.indexer.position_bias",
        ),
        (
            r"^layers\.(\d+)\.self_attn\.indexer\.compressor\.",
            r"layers.\1.self_attn.compressor.indexer.",
        ),
        (r"^layers\.(\d+)\.self_attn\.indexer\.", r"layers.\1.self_attn.compressor.indexer."),
        (
            r"^layers\.(\d+)\.self_attn\.compressor\.norm\.",
            r"layers.\1.self_attn.compressor.kv_norm.",
        ),
        (
            r"^layers\.(\d+)\.self_attn\.compressor\.ape$",
            r"layers.\1.self_attn.compressor.position_bias",
        ),
        (
            r"^layers\.(\d+)\.self_attn\.(.*?)\.wq_a\.",
            r"layers.\1.self_attn.\2.q_a_proj.",
        ),
        (
            r"^layers\.(\d+)\.self_attn\.(.*?)\.wq_b\.",
            r"layers.\1.self_attn.\2.q_b_proj.",
        ),
        (
            r"^layers\.(\d+)\.self_attn\.(.*?)\.wkv\.",
            r"layers.\1.self_attn.\2.kv_proj.",
        ),
        (
            r"^layers\.(\d+)\.self_attn\.(.*?)\.wgate\.",
            r"layers.\1.self_attn.\2.gate_proj.",
        ),
        (
            r"^layers\.(\d+)\.self_attn\.(.*?)\.wo_a\.",
            r"layers.\1.self_attn.\2.o_a_proj.",
        ),
        (
            r"^layers\.(\d+)\.self_attn\.(.*?)\.wo_b\.",
            r"layers.\1.self_attn.\2.o_b_proj.",
        ),
        (r"^layers\.(\d+)\.self_attn\.wq_a\.", r"layers.\1.self_attn.q_a_proj."),
        (r"^layers\.(\d+)\.self_attn\.wq_b\.", r"layers.\1.self_attn.q_b_proj."),
        (r"^layers\.(\d+)\.self_attn\.wkv\.", r"layers.\1.self_attn.kv_proj."),
        (r"^layers\.(\d+)\.self_attn\.wo_a\.", r"layers.\1.self_attn.o_a_proj."),
        (r"^layers\.(\d+)\.self_attn\.wo_b\.", r"layers.\1.self_attn.o_b_proj."),
        (r"^layers\.(\d+)\.self_attn\.q_norm\.", r"layers.\1.self_attn.q_a_norm."),
        (r"^layers\.(\d+)\.mlp\.gate\.bias$", r"layers.\1.mlp.gate.e_score_correction_bias"),
        (
            r"^layers\.(\d+)\.mlp\.shared_experts\.w1\.",
            r"layers.\1.mlp.shared_experts.gate_proj.",
        ),
        (
            r"^layers\.(\d+)\.mlp\.shared_experts\.w2\.",
            r"layers.\1.mlp.shared_experts.down_proj.",
        ),
        (
            r"^layers\.(\d+)\.mlp\.shared_experts\.w3\.",
            r"layers.\1.mlp.shared_experts.up_proj.",
        ),
    ]
    for pattern, replacement in replacements:
        key = re.sub(pattern, replacement, key)
    key = key.replace("..", ".")
    if (
        key.startswith("layers.")
        or key.startswith("embed_tokens.")
        or key.startswith("norm.")
        or key.startswith("hc_head.")
    ):
        return f"model.{key}"
    return key


def _deepseek_v4_unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(_DEEPSEEK_V4_FP4_E2M1_LUT, dtype=torch.float32, device=packed.device)
    u8 = packed.contiguous().view(torch.uint8)
    low = (u8 & 0xF).long()
    high = ((u8 >> 4) & 0xF).long()
    unpacked = torch.stack([lut[low], lut[high]], dim=-1)
    return unpacked.reshape(*packed.shape[:-1], 2 * packed.shape[-1])


def _deepseek_v4_dequantize_fp8_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if weight.dtype == torch.int8 or (fp4_dtype is not None and weight.dtype == fp4_dtype):
        weight_fp32 = _deepseek_v4_unpack_fp4(weight)
    else:
        weight_fp32 = weight.to(torch.float32)
    rows, cols = weight_fp32.shape[-2:]
    scale_rows, scale_cols = scale.shape[-2:]
    if rows % scale_rows != 0 or cols % scale_cols != 0:
        raise ValueError(
            f"DeepSeek-V4 FP8 weight shape ({rows}, {cols}) is not divisible by scale grid "
            f"({scale_rows}, {scale_cols})"
        )
    block_m = rows // scale_rows
    block_n = cols // scale_cols
    original_shape = weight_fp32.shape
    q = weight_fp32.reshape(-1, scale_rows, block_m, scale_cols, block_n)
    s = scale.to(torch.float32).reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)
    return (q * s).reshape(original_shape).to(dtype=dtype)


def _safetensors_key_index(paths: list[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in paths:
        st = safetensors.safe_open(str(path), framework="pt")
        for key in st.keys():
            if key in out:
                raise ValueError(f"Duplicate tensor key while indexing safetensors shards: {key}")
            out[key] = path
    return out


def _read_safetensor_indexed(index: dict[str, Path], key: str) -> torch.Tensor:
    st = safetensors.safe_open(str(index[key]), framework="pt")
    return st.get_tensor(key)


def _set_hf_tensor_from_value(
    model: Any,
    name: str,
    value: torch.Tensor,
    *,
    dtype: torch.dtype,
    device_map: dict[str, str] | None,
    target_device: torch.device,
) -> None:
    device = _device_for_hf_tensor_name(name, device_map=device_map, default_device=target_device)
    if value.is_floating_point():
        value = value.to(device=device, dtype=dtype)
    else:
        value = value.to(device=device)
    set_module_tensor_to_device(model, name, device, value=value)


def _first_numeric_path_segment(key: str) -> int | None:
    for part in str(key).split("."):
        if part.isdigit():
            return int(part)
    return None


def _axon_param_placement_plan(keys: Sequence[str], devices: Sequence[torch.device]) -> dict[str, torch.device]:
    if not devices:
        return {}
    layer_ids = sorted(
        idx for key in keys if (idx := _first_numeric_path_segment(str(key))) is not None
    )
    if not layer_ids:
        return {str(key): devices[0] for key in keys}
    layer_to_device = {
        layer_id: devices[min(len(devices) - 1, pos * len(devices) // len(layer_ids))]
        for pos, layer_id in enumerate(layer_ids)
    }
    out: dict[str, torch.device] = {}
    for key in keys:
        idx = _first_numeric_path_segment(str(key))
        out[str(key)] = layer_to_device[idx] if idx is not None else devices[0]
    return out


def _load_dequantized_deepseek_v4_fp8_state_dict(
    paths: list[Path],
    *,
    device: torch.device,
    dtype: torch.dtype,
    storage_dtype: torch.dtype,
    param_devices: Sequence[str] | None,
) -> dict[str, torch.Tensor]:
    index = _safetensors_key_index(paths)
    expert_groups: dict[tuple[int, int], dict[str, str]] = {}
    output_keys: list[str] = []

    for raw_key in sorted(index):
        expert_match = _DEEPSEEK_V4_EXPERT_RE.match(raw_key)
        if expert_match is not None:
            expert_groups.setdefault(
                (int(expert_match.group("layer")), int(expert_match.group("expert"))), {}
            )[expert_match.group("proj")] = raw_key
            continue
        if raw_key.endswith(".scale") and raw_key[: -len(".scale")] + ".weight" in index:
            continue
        output_keys.append(_deepseek_v4_hf_base_key(raw_key))

    by_layer: dict[int, dict[int, dict[str, str]]] = {}
    for (layer, expert), proj_keys in expert_groups.items():
        by_layer.setdefault(layer, {})[expert] = proj_keys
    for layer in by_layer:
        output_keys.append(f"model.layers.{layer}.mlp.experts.gate_up_proj")
        output_keys.append(f"model.layers.{layer}.mlp.experts.down_proj")

    placement_devices = [torch.device(item) for item in (param_devices or [])]
    plan = _axon_param_placement_plan(output_keys, placement_devices)

    def target_device(key: str) -> torch.device:
        return plan.get(key, device)

    out: dict[str, torch.Tensor] = {}
    for raw_key in sorted(index):
        if _DEEPSEEK_V4_EXPERT_RE.match(raw_key) is not None:
            continue
        if raw_key.endswith(".scale") and raw_key[: -len(".scale")] + ".weight" in index:
            continue
        mapped = _deepseek_v4_hf_base_key(raw_key)
        value = _read_safetensor_indexed(index, raw_key)
        scale_key = raw_key[: -len(".weight")] + ".scale" if raw_key.endswith(".weight") else ""
        if scale_key in index:
            value = _deepseek_v4_dequantize_fp8_weight(
                value,
                _read_safetensor_indexed(index, scale_key),
                dtype=storage_dtype,
            )
        if value.is_floating_point():
            value = value.to(device=target_device(mapped), dtype=storage_dtype)
        else:
            value = value.to(device=target_device(mapped))
        out[mapped] = value

    for layer, experts in sorted(by_layer.items()):
        if not experts:
            continue
        first_expert = experts[min(experts)]
        if not {"w1", "w2", "w3"}.issubset(first_expert):
            raise KeyError(f"DeepSeek-V4 missing FP8 expert tensors for layer={layer}")
        first_gate = _deepseek_v4_dequantize_fp8_weight(
            _read_safetensor_indexed(index, first_expert["w1"]),
            _read_safetensor_indexed(index, first_expert["w1"][: -len(".weight")] + ".scale"),
            dtype=storage_dtype,
        )
        first_up = _deepseek_v4_dequantize_fp8_weight(
            _read_safetensor_indexed(index, first_expert["w3"]),
            _read_safetensor_indexed(index, first_expert["w3"][: -len(".weight")] + ".scale"),
            dtype=storage_dtype,
        )
        first_down = _deepseek_v4_dequantize_fp8_weight(
            _read_safetensor_indexed(index, first_expert["w2"]),
            _read_safetensor_indexed(index, first_expert["w2"][: -len(".weight")] + ".scale"),
            dtype=storage_dtype,
        )
        num_experts = max(experts) + 1
        gate_name = f"model.layers.{layer}.mlp.experts.gate_up_proj"
        down_name = f"model.layers.{layer}.mlp.experts.down_proj"
        gate_device = target_device(gate_name)
        down_device = target_device(down_name)
        gate_up = torch.empty(
            (num_experts, first_gate.shape[0] + first_up.shape[0], first_gate.shape[1]),
            device=gate_device,
            dtype=storage_dtype,
        )
        down = torch.empty((num_experts, *first_down.shape), device=down_device, dtype=storage_dtype)

        def fill_expert(expert: int, proj_keys: dict[str, str]) -> None:
            if not {"w1", "w2", "w3"}.issubset(proj_keys):
                raise KeyError(
                    f"DeepSeek-V4 missing FP8 expert tensors for layer={layer} expert={expert}"
                )
            w1_key = proj_keys["w1"]
            w2_key = proj_keys["w2"]
            w3_key = proj_keys["w3"]
            gate = _deepseek_v4_dequantize_fp8_weight(
                _read_safetensor_indexed(index, w1_key),
                _read_safetensor_indexed(index, w1_key[: -len(".weight")] + ".scale"),
                dtype=storage_dtype,
            )
            up = _deepseek_v4_dequantize_fp8_weight(
                _read_safetensor_indexed(index, w3_key),
                _read_safetensor_indexed(index, w3_key[: -len(".weight")] + ".scale"),
                dtype=storage_dtype,
            )
            down_weight = _deepseek_v4_dequantize_fp8_weight(
                _read_safetensor_indexed(index, w2_key),
                _read_safetensor_indexed(index, w2_key[: -len(".weight")] + ".scale"),
                dtype=storage_dtype,
            )
            gate_up[expert].copy_(torch.cat([gate, up], dim=0).to(device=gate_device, dtype=storage_dtype))
            down[expert].copy_(down_weight.to(device=down_device, dtype=storage_dtype))

        for expert, proj_keys in sorted(experts.items()):
            fill_expert(expert, proj_keys)
        out[gate_name] = gate_up
        out[down_name] = down

    return out


def _load_hf_causal_lm_from_dequantized_deepseek_v4_fp8_state(
    *,
    safetensors_files: list[Path],
    dtype: torch.dtype,
    hf_config: Any,
    trust_remote_code: bool,
    device_map: dict[str, str] | None,
    target_device: torch.device,
) -> Any:
    plain_config = deepcopy(hf_config)
    if hasattr(plain_config, "quantization_config"):
        try:
            delattr(plain_config, "quantization_config")
        except Exception:
            setattr(plain_config, "quantization_config", None)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            plain_config,
            trust_remote_code=trust_remote_code,
        )

    expected = set(model.state_dict().keys())
    loaded: set[str] = set()
    index = _safetensors_key_index(safetensors_files)
    expert_groups: dict[tuple[int, int], dict[str, str]] = {}

    for raw_key in sorted(index):
        expert_match = _DEEPSEEK_V4_EXPERT_RE.match(raw_key)
        if expert_match is not None:
            expert_groups.setdefault(
                (int(expert_match.group("layer")), int(expert_match.group("expert"))), {}
            )[expert_match.group("proj")] = raw_key
            continue
        if raw_key.endswith(".scale") and raw_key[: -len(".scale")] + ".weight" in index:
            continue

        value = _read_safetensor_indexed(index, raw_key)
        scale_key = raw_key[: -len(".weight")] + ".scale" if raw_key.endswith(".weight") else ""
        if scale_key in index:
            value = _deepseek_v4_dequantize_fp8_weight(value, _read_safetensor_indexed(index, scale_key), dtype=dtype)

        mapped = _deepseek_v4_hf_base_key(raw_key)
        if mapped not in expected:
            continue
        _set_hf_tensor_from_value(
            model,
            mapped,
            value,
            dtype=dtype,
            device_map=device_map,
            target_device=target_device,
        )
        loaded.add(mapped)

    by_layer: dict[int, dict[int, dict[str, str]]] = {}
    for (layer, expert), proj_keys in expert_groups.items():
        by_layer.setdefault(layer, {})[expert] = proj_keys

    state_shapes = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    for layer, experts in sorted(by_layer.items()):
        gate_name = f"model.layers.{layer}.mlp.experts.gate_up_proj"
        down_name = f"model.layers.{layer}.mlp.experts.down_proj"
        if gate_name not in expected or down_name not in expected:
            continue
        gate_device = _device_for_hf_tensor_name(gate_name, device_map=device_map, default_device=target_device)
        down_device = _device_for_hf_tensor_name(down_name, device_map=device_map, default_device=target_device)
        gate_up = torch.empty(state_shapes[gate_name], device=gate_device, dtype=dtype)
        down = torch.empty(state_shapes[down_name], device=down_device, dtype=dtype)
        for expert in range(gate_up.shape[0]):
            proj_keys = experts.get(expert)
            if proj_keys is None or not {"w1", "w2", "w3"}.issubset(proj_keys):
                raise KeyError(f"DeepSeek-V4 missing FP8 expert tensors for layer={layer} expert={expert}")
            w1_key = proj_keys["w1"]
            w2_key = proj_keys["w2"]
            w3_key = proj_keys["w3"]
            gate = _deepseek_v4_dequantize_fp8_weight(
                _read_safetensor_indexed(index, w1_key),
                _read_safetensor_indexed(index, w1_key[: -len(".weight")] + ".scale"),
                dtype=dtype,
            )
            up = _deepseek_v4_dequantize_fp8_weight(
                _read_safetensor_indexed(index, w3_key),
                _read_safetensor_indexed(index, w3_key[: -len(".weight")] + ".scale"),
                dtype=dtype,
            )
            gate_up[expert].copy_(torch.cat([gate, up], dim=0).to(device=gate_device, dtype=dtype))
            down[expert].copy_(
                _deepseek_v4_dequantize_fp8_weight(
                    _read_safetensor_indexed(index, w2_key),
                    _read_safetensor_indexed(index, w2_key[: -len(".weight")] + ".scale"),
                    dtype=dtype,
                ).to(device=down_device, dtype=dtype)
            )
        set_module_tensor_to_device(model, gate_name, gate_device, value=gate_up)
        set_module_tensor_to_device(model, down_name, down_device, value=down)
        loaded.add(gate_name)
        loaded.add(down_name)

    missing = sorted(
        name
        for name, parameter in model.named_parameters()
        if name not in loaded and getattr(parameter, "device", None).type == "meta"
    )
    if missing == ["lm_head.weight"] and "model.embed_tokens.weight" in loaded:
        model.get_output_embeddings().weight = model.get_input_embeddings().weight
        missing = []
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise RuntimeError(f"dequantized DeepSeek-V4 FP8 HF load left meta parameters: {preview}{suffix}")

    if device_map is not None:
        model = dispatch_model(model, device_map=device_map)
    return model.eval()


def _refresh_hf_rotary_caches_if_needed(model: Any, *, dtype: torch.dtype) -> int:
    refreshed = 0
    for module in model.modules():
        set_cache = getattr(module, "_set_cos_sin_cache", None)
        if not callable(set_cache):
            continue
        module_name = module.__class__.__name__.lower()
        is_deepseek_rotary = "deepseek" in module_name
        max_seq_len = getattr(module, "max_seq_len", None)
        if not isinstance(max_seq_len, int) or max_seq_len <= 0:
            max_seq_len = getattr(module, "max_position_embeddings", None)
        if not isinstance(max_seq_len, int) or max_seq_len <= 0:
            continue
        cos_cached = getattr(module, "cos_cached", None)
        sin_cached = getattr(module, "sin_cached", None)
        needs_refresh = False
        if not torch.is_tensor(cos_cached) or not torch.is_tensor(sin_cached):
            needs_refresh = True
        elif cos_cached.device.type == "meta" or sin_cached.device.type == "meta":
            needs_refresh = True
        elif cos_cached.numel() == 0 or sin_cached.numel() == 0:
            needs_refresh = True
        else:
            cos_abs_max = float(cos_cached.detach().abs().max())
            sin_abs_max = float(sin_cached.detach().abs().max())
            # Valid rotary caches should stay within [-1, 1].
            if (
                (cos_abs_max == 0.0 and sin_abs_max == 0.0)
                or cos_abs_max > 1.01
                or sin_abs_max > 1.01
            ):
                needs_refresh = True
        if is_deepseek_rotary:
            needs_refresh = True
        if not needs_refresh:
            continue
        inv_freq = getattr(module, "inv_freq", None)
        # Some DeepSeek remote-code checkpoints can load with corrupted inv_freq buffers
        # (all-zero / random / NaN). Rebuild from (base, dim) before regenerating caches.
        if torch.is_tensor(inv_freq) and is_deepseek_rotary:
            dim = getattr(module, "dim", None)
            base = getattr(module, "base", None)
            if isinstance(dim, int) and dim > 0 and base is not None:
                expected_inv = 1.0 / (
                    float(base)
                    ** (
                        torch.arange(0, dim, 2, device=inv_freq.device, dtype=torch.float32)
                        / float(dim)
                    )
                )
                should_replace_inv = (not torch.isfinite(inv_freq).all()) or (
                    inv_freq.shape != expected_inv.shape
                )
                if not should_replace_inv:
                    delta = (inv_freq.to(torch.float32) - expected_inv).abs()
                    should_replace_inv = float(delta.max()) > 1e-6
                if should_replace_inv:
                    try:
                        module.register_buffer("inv_freq", expected_inv, persistent=False)
                        inv_freq = expected_inv
                    except Exception:
                        pass
        target_device = inv_freq.device if torch.is_tensor(inv_freq) else torch.device("cpu")
        try:
            set_cache(seq_len=int(max_seq_len), device=target_device, dtype=dtype)
            refreshed += 1
        except Exception:
            continue
    return refreshed


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


def _collect_hf_param_names_for_device_map(
    *,
    model_task: str,
    hf_config: Any | None,
    trust_remote_code: bool,
) -> set[str]:
    if hf_config is None:
        return set()
    _ensure_transformers_import_compat()
    try:
        from accelerate import init_empty_weights
    except Exception:
        return set()

    names: set[str] = set()
    constructors: list[Any] = []
    if model_task == "seq2seq_lm":
        constructors.append(AutoModelForSeq2SeqLM)
    elif model_task == "causal_lm":
        constructors.append(AutoModelForCausalLM)
        if str(getattr(hf_config, "model_type", "")).strip().lower() in {
            "gemma4",
            "mistral3",
            "mistral4",
            "llama4",
        }:
            constructors.append(AutoModelForImageTextToText)
    else:
        return names

    for ctor in constructors:
        try:
            with init_empty_weights():
                model = ctor.from_config(hf_config, trust_remote_code=trust_remote_code)
            names = {name for name, _ in model.named_parameters()}
            names.update({name for name, _ in model.named_buffers()})
            del model
            if names:
                return names
        except Exception:
            continue
    return names


def _collect_ordered_hf_param_names_for_device_map(
    *,
    model_task: str,
    hf_config: Any | None,
    trust_remote_code: bool,
) -> list[str]:
    if hf_config is None:
        return []
    _ensure_transformers_import_compat()
    try:
        from accelerate import init_empty_weights
    except Exception:
        return []

    constructors: list[Any] = []
    if model_task == "seq2seq_lm":
        constructors.append(AutoModelForSeq2SeqLM)
    elif model_task == "masked_lm":
        constructors.append(AutoModelForMaskedLM)
    elif model_task == "causal_lm":
        constructors.append(AutoModelForCausalLM)
        if str(getattr(hf_config, "model_type", "")).strip().lower() in {
            "gemma4",
            "mistral3",
            "mistral4",
            "llama4",
        }:
            constructors.append(AutoModelForImageTextToText)
    else:
        return []

    fallback_ordered: list[str] = []
    for ctor in constructors:
        try:
            with init_empty_weights():
                model = ctor.from_config(hf_config, trust_remote_code=trust_remote_code)
            names = [name for name, _ in model.named_parameters()]
            names.extend(name for name, _ in model.named_buffers())
            del model
            if names:
                seen: set[str] = set()
                ordered: list[str] = []
                for name in names:
                    if name in seen:
                        continue
                    seen.add(name)
                    ordered.append(name)
                if any(_numeric_segment(name) is not None for name in ordered):
                    return ordered
                if not fallback_ordered:
                    fallback_ordered = ordered
        except Exception:
            continue
    return fallback_ordered


def _numeric_segment(name: str) -> tuple[int, int] | None:
    for index, segment in enumerate(name.split(".")):
        if segment.isdigit():
            return index, int(segment)
    return None


def _module_prefix_for_param_name(name: str) -> str:
    pieces = [piece for piece in name.split(".") if piece]
    numeric = _numeric_segment(name)
    if numeric is not None:
        index, _ = numeric
        return ".".join(pieces[: index + 1])
    if len(pieces) <= 1:
        return ""
    return ".".join(pieces[:-1])


def _build_generic_hf_device_map_from_param_names(
    ordered_names: Sequence[str],
    *,
    devices: Sequence[str],
) -> tuple[dict[str, str] | None, tuple[tuple[int, int, str], ...]]:
    unique_devices = tuple(str(device) for device in devices if str(device).strip())
    if len(unique_devices) <= 1:
        return None, ()

    names = tuple(str(name) for name in ordered_names if str(name).strip())
    if not names:
        return {"": unique_devices[0]}, ()

    numeric_by_name: dict[str, tuple[int, int]] = {}
    layer_indices: list[int] = []
    for name in names:
        numeric = _numeric_segment(name)
        if numeric is None:
            continue
        numeric_by_name[name] = numeric
        layer_indices.append(numeric[1])

    distinct_layers = tuple(sorted(set(layer_indices)))
    layer_position = {layer: index for index, layer in enumerate(distinct_layers)}
    if distinct_layers:
        first_layer_pos = min(index for index, name in enumerate(names) if name in numeric_by_name)
        last_layer_pos = max(index for index, name in enumerate(names) if name in numeric_by_name)
    else:
        first_layer_pos = len(names)
        last_layer_pos = -1

    def _device_for_layer(layer_index: int) -> str:
        if not distinct_layers:
            return unique_devices[0]
        ordinal = layer_position[layer_index]
        device_index = min(
            len(unique_devices) - 1,
            (ordinal * len(unique_devices)) // len(distinct_layers),
        )
        return unique_devices[device_index]

    device_map: dict[str, str] = {}
    for ordinal, name in enumerate(names):
        prefix = _module_prefix_for_param_name(name)
        numeric = numeric_by_name.get(name)
        if numeric is not None:
            device = _device_for_layer(numeric[1])
        elif ordinal < first_layer_pos:
            device = unique_devices[0]
        elif ordinal > last_layer_pos:
            device = unique_devices[-1]
        else:
            # Non-layer parameters interleaved with layers are safest on the current stage-0
            # fallback unless a more specific prefix is already assigned.
            device = unique_devices[0]
        device_map.setdefault(prefix, device)

    spans: list[tuple[int, int, str]] = []
    if distinct_layers:
        for device_index, device in enumerate(unique_devices):
            start_pos = (device_index * len(distinct_layers)) // len(unique_devices)
            stop_pos = ((device_index + 1) * len(distinct_layers)) // len(unique_devices)
            if start_pos >= stop_pos:
                continue
            spans.append(
                (
                    distinct_layers[start_pos],
                    distinct_layers[stop_pos - 1] + 1,
                    device,
                )
            )
    return device_map, tuple(spans)


def _colocate_tied_hf_output_embeddings(
    device_map: dict[str, str],
    *,
    has_explicit_output_head_weight: bool,
) -> None:
    if has_explicit_output_head_weight:
        return
    embedding_device = next(
        (
            device
            for prefix, device in device_map.items()
            if prefix.endswith("embed_tokens")
            or prefix.endswith("word_embeddings")
            or prefix.endswith("wte")
        ),
        None,
    )
    if embedding_device is None:
        return
    for prefix in (
        "lm_head",
        "model.lm_head",
        "language_model.lm_head",
        "language_model.model.lm_head",
        "transformer.lm_head",
    ):
        if prefix in device_map:
            device_map[prefix] = embedding_device


def _time_generate(label: str, fn: Any) -> tuple[Any, float]:
    t0 = time.perf_counter()
    with timing(message=label):
        out = fn()
    dt = time.perf_counter() - t0
    return out, dt


def _time_generate_repeated(
    label: str,
    fn: Any,
    *,
    warmup: int,
    repeat: int,
) -> tuple[Any, float, list[float]]:
    warmup = max(0, int(warmup))
    repeat = max(1, int(repeat))
    out: Any = None

    def _sync_cuda() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    with timing(message=label):
        for _ in range(warmup):
            out = fn()
            _sync_cuda()
        samples: list[float] = []
        for _ in range(repeat):
            _sync_cuda()
            t0 = time.perf_counter()
            out = fn()
            _sync_cuda()
            samples.append(time.perf_counter() - t0)
    return out, sum(samples) / max(1, len(samples)), samples


def _time_forward_repeated(
    label: str,
    fn: Any,
    *,
    warmup: int,
    repeat: int,
) -> tuple[Any, float, list[float]]:
    warmup = max(0, int(warmup))
    repeat = max(1, int(repeat))
    out: Any = None
    def _sync_cuda() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    with timing(message=label):
        for _ in range(warmup):
            out = fn()
            _sync_cuda()
        samples: list[float] = []
        for _ in range(repeat):
            _sync_cuda()
            t0 = time.perf_counter()
            out = fn()
            _sync_cuda()
            samples.append(time.perf_counter() - t0)
    return out, sum(samples) / max(1, len(samples)), samples


def _call_generate_compatible(model: Any, **kwargs: Any) -> Any:
    """Call HF/reference generate with only supported keyword arguments."""

    generate = model.generate
    try:
        signature = inspect.signature(generate)
    except (TypeError, ValueError):
        return generate(**kwargs)
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return generate(**kwargs)
    accepted = {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return generate(**{key: value for key, value in kwargs.items() if key in accepted})


def _module_parameter_device(module: Any) -> torch.device | None:
    if module is None:
        return None
    try:
        for value in module.parameters(recurse=False):
            return value.device
    except Exception:
        return None
    return None


def _hf_input_embedding_device(model: Any) -> torch.device | None:
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        device = _module_parameter_device(get_embeddings())
        if device is not None:
            return device
    for path in (
        "model.encoder.embed_tokens",
        "encoder.embed_tokens",
        "model.embed_tokens",
        "embed_tokens",
    ):
        device = _module_parameter_device(_get_nested_attr(model, path))
        if device is not None:
            return device
    return None


def _hf_decoder_embedding_device(model: Any) -> torch.device | None:
    get_decoder = getattr(model, "get_decoder", None)
    decoder = None
    if callable(get_decoder):
        try:
            decoder = get_decoder()
        except Exception:
            decoder = None
    if decoder is None:
        decoder = getattr(model, "decoder", None)
    if decoder is not None:
        get_embeddings = getattr(decoder, "get_input_embeddings", None)
        if callable(get_embeddings):
            device = _module_parameter_device(get_embeddings())
            if device is not None:
                return device
        embed_tokens = getattr(decoder, "embed_tokens", None)
        device = _module_parameter_device(embed_tokens)
        if device is not None:
            return device
    for path in (
        "model.decoder.embed_tokens",
        "decoder.embed_tokens",
        "model.embed_tokens",
        "embed_tokens",
    ):
        device = _module_parameter_device(_get_nested_attr(model, path))
        if device is not None:
            return device
    return _hf_input_embedding_device(model)


def _move_hf_token_inputs_to_embedding_devices(
    model: Any, kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    out = dict(kwargs)
    input_device = _hf_input_embedding_device(model)
    if input_device is not None and torch.is_tensor(out.get("input_ids")):
        out["input_ids"] = out["input_ids"].to(device=input_device)
    decoder_device = _hf_decoder_embedding_device(model)
    if decoder_device is not None and torch.is_tensor(out.get("decoder_input_ids")):
        out["decoder_input_ids"] = out["decoder_input_ids"].to(device=decoder_device)
    return out


def _normalize_hf_experts_implementation(token: str | None) -> str | None:
    if token is None:
        return None
    normalized = str(token).strip()
    if not normalized:
        raise ValueError("--hf-experts-implementation must not be empty")
    return normalized


def _apply_hf_experts_implementation(config: Any | None, token: str | None) -> None:
    if config is None or token is None:
        return
    setattr(config, "_experts_implementation", token)


@contextmanager
def _preserve_requested_hf_experts_during_generate(model: Any, token: str | None) -> Any:
    if token is None:
        yield
        return
    set_impl = getattr(model, "set_experts_implementation", None)
    if callable(set_impl):
        set_impl(token)
    original_decode_opt = getattr(model, "_optimize_model_for_decode", None)
    if not callable(original_decode_opt):
        yield
        return

    def _no_decode_expert_rewrite(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs

        @contextmanager
        def _ctx() -> Any:
            yield

        return _ctx()

    setattr(model, "_optimize_model_for_decode", _no_decode_expert_rewrite)
    try:
        yield
    finally:
        setattr(model, "_optimize_model_for_decode", original_decode_opt)
        if callable(set_impl):
            set_impl(token)


def _print_axon_profile_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        print("Axon profile: no recorded regions")
        return
    print("Axon profile top regions:")
    print("| rank | region | calls | total_s | avg_ms |")
    print("|---:|---|---:|---:|---:|")
    for rank, row in enumerate(rows, start=1):
        seconds = float(row.get("seconds", 0.0))
        count = int(row.get("count", 0))
        avg_ms = float(row.get("avg_seconds", 0.0)) * 1000.0
        print(
            f"| {rank} | {row.get('name', '')} | {count} | "
            f"{seconds:.6f} | {avg_ms:.3f} |"
        )


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
        return ["The future of AI is", "Hello World"]
    return out


def _extract_hidden_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, tuple) and value and torch.is_tensor(value[0]):
        return value[0]
    raise ValueError(f"Unable to extract hidden tensor from type: {type(value).__name__}")


def _to_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(dtype=torch.float32, device="cpu")


def _pick_first_existing_name(candidates: Sequence[str], names: set[str]) -> str | None:
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _tokenizer_fallback_repo_id(model_dir: Path) -> str | None:
    aliases = {
        "black_mamba_2_8b": "EleutherAI/gpt-neox-20b",
        "black_mamba": "EleutherAI/gpt-neox-20b",
        "marian_en_de": "Helsinki-NLP/opus-mt-en-de",
        "opus-mt-en-de": "Helsinki-NLP/opus-mt-en-de",
        "Devstral-Small-2507": "mistralai/Devstral-Small-2507",
        "Magistral-Small-2509": "mistralai/Magistral-Small-2509",
    }
    return aliases.get(model_dir.name)


def _checkpoint_requires_local_tokenizer(checkpoint_id: str) -> bool:
    return checkpoint_id not in {
        "mistralai/Devstral-Small-2507",
        "mistralai/Magistral-Small-2509",
        "Zyphra/BlackMamba-2.8B",
    }


def _build_seq2seq_decoder_inputs(
    *,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor | None,
    hf_config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if encoder_input_ids.ndim != 2:
        raise ValueError("seq2seq_lm expects encoder input_ids to be rank-2 [batch, seq]")
    if encoder_input_ids.shape[1] < 1:
        raise ValueError("seq2seq_lm expects non-empty encoder input_ids sequence")

    start_token_id = getattr(hf_config, "decoder_start_token_id", None)
    if not isinstance(start_token_id, int):
        # Some encoder-decoder configs (e.g. T5Gemma variants) omit decoder_start_token_id
        # and rely on pad/bos fallback behavior in higher-level generation utilities.
        bos_token_id = getattr(hf_config, "bos_token_id", None)
        pad_token_id = getattr(hf_config, "pad_token_id", None)
        if isinstance(bos_token_id, int):
            start_token_id = int(bos_token_id)
        elif isinstance(pad_token_id, int):
            start_token_id = int(pad_token_id)
        else:
            raise ValueError(
                "seq2seq_lm requires config.decoder_start_token_id (or bos_token_id/pad_token_id fallback)"
            )

    decoder_input_ids = encoder_input_ids.clone()
    decoder_input_ids[:, 1:] = encoder_input_ids[:, :-1]
    decoder_input_ids[:, 0] = int(start_token_id)

    if encoder_attention_mask is not None:
        if encoder_attention_mask.shape != encoder_input_ids.shape:
            raise ValueError("seq2seq_lm encoder attention_mask shape must match input_ids")
        decoder_attention_mask = encoder_attention_mask.to(dtype=torch.long).clone()
        decoder_attention_mask[:, 0] = 1
    else:
        decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.long)

    return decoder_input_ids, decoder_attention_mask


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _select_main_axon_module(modules: Sequence[Any] | Any) -> Any:
    raw_modules = getattr(modules, "modules", modules)
    if not isinstance(raw_modules, Sequence):
        raise TypeError("expected AxonFile or sequence of modules")
    if not raw_modules:
        raise ValueError("Axon program contains no modules")
    if not hasattr(modules, "pragmas"):
        return raw_modules[-1]
    main_module = resolve_main_module(cast(AxonFile, modules))
    for module in raw_modules:
        if getattr(module, "name", None) == main_module:
            return module
    raise ValueError(f"Main Axon module not found: {main_module}")


def _declared_checkpoints_from_axon(
    *,
    axon_file: Path,
) -> tuple[str, ...]:
    parsed = parse_axon_program_from_path(axon_file)
    module = _select_main_axon_module(parsed)
    raw = (getattr(module, "pragmas", None) or {}).get("checkpoints")
    if raw is None:
        raw = (getattr(parsed, "pragmas", None) or {}).get("checkpoints")
    if raw is None:
        raise ValueError(
            f"No CHECKPOINTS pragma declared in {axon_file}"
        )
    checkpoints: tuple[str, ...]
    if isinstance(raw, str):
        checkpoints = (raw,)
    elif isinstance(raw, tuple | list):
        checkpoints = tuple(str(item) for item in raw)
    else:
        raise ValueError("CHECKPOINTS pragma must be a string or tuple/list of strings")
    if not checkpoints or any(not item for item in checkpoints):
        raise ValueError("CHECKPOINTS pragma must contain at least one non-empty checkpoint id")
    return checkpoints


def _ensure_checkpoint_model_dir(*, repo_root: Path, checkpoint_id: str) -> Path:
    def _status(message: str) -> None:
        print(f"[model-download] {message}")

    return ensure_model_downloaded(
        repo_root=repo_root,
        spec=ModelDownloadSpec(
            local_dir=checkpoint_id,
            repo_id=checkpoint_id,
            require_tokenizer=_checkpoint_requires_local_tokenizer(checkpoint_id),
        ),
        status_cb=_status,
    )


def _run_axon_test_single(
    *,
    axon_file: Path,
    weights: Path,
    device: str = "cpu",
    text: str | Sequence[str] = ("The future of AI is", "Hello World"),
    max_len: int = 32,
    hf_model_dir: Path | None = None,
    tokenizer: str | None = None,
    class_name: str = "AxonGeneratedModel",
    dtype: str = "float32",
    model_task: str = "auto",
    benchmark_mode: str = "auto",
    trace_layers: bool = False,
    hf_align_bf16_profile: bool = False,
    hf_align_mask_contract: bool = False,
    hf_align_position_ids: bool = False,
    hf_align_add_fp32_accum: bool = False,
    hf_align_linear_fp32_accum: bool = False,
    hf_align_norm_fp32: bool = False,
    hf_attn_implementation: str | None = None,
    hf_experts_implementation: str | None = None,
    compile_hf: bool = False,
    compile_axon: bool = False,
    compile_backend: str | None = None,
    compile_mode: str | None = None,
    compile_fullgraph: bool = False,
    compile_dynamic: bool = False,
    trust_remote_code: bool = False,
    axon_backend: str = "codegen2-torch",
    axon_typechecker: str = "typecheck2",
    optimize_ast: bool = False,
    optimize_graph: bool = False,
    graph_backend_intrinsics: str | None = None,
    skip_hf: bool = False,
    hf_strict_dtype: bool = False,
    oom_cpu_fallback: bool = True,
    profile_axon: bool = False,
    profile_axon_top_n: int = 40,
    forward_warmup: int = 0,
    forward_repeat: int = 1,
    generate_warmup: int = 0,
    generate_repeat: int = 1,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype)
    resolved_model_task = _resolve_model_task(model_task)
    resolved_benchmark_mode = _resolve_benchmark_mode(benchmark_mode)
    forward_warmup = max(0, int(forward_warmup))
    forward_repeat = max(1, int(forward_repeat))
    generate_warmup = max(0, int(generate_warmup))
    generate_repeat = max(1, int(generate_repeat))
    resolved_hf_experts_implementation = _normalize_hf_experts_implementation(
        hf_experts_implementation
    )
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
    if resolved_model_task == "auto":
        resolved_model_task = _infer_model_task(axon_file=axon_file, weights=weights_path)
    run_generate_benchmark = _should_generate_for_benchmark(
        model_task=resolved_model_task,
        benchmark_mode=resolved_benchmark_mode,
    )
    backend_token = str(axon_backend).strip().lower()
    if backend_token == "single":
        backend_token = "codegen2-torch"
    valid_backends = {
        "codegen2-torch",
        "codegen2-tinygrad",
        "codegen2-triton",
        "codegen2-vllm",
        "runtime2-torch",
        "pipeline2-torch",
    }
    if backend_token not in valid_backends:
        raise ValueError(
            "axon_backend must be 'codegen2-torch', 'codegen2-tinygrad', "
            "'codegen2-triton', 'codegen2-vllm', 'runtime2-torch', or 'pipeline2-torch'"
        )
    axon_backend = backend_token
    typechecker_token = str(axon_typechecker).strip().lower()
    if typechecker_token != "typecheck2":
        raise ValueError("axon_typechecker must be 'typecheck2'")
    axon_typechecker = typechecker_token
    if axon_backend == "pipeline2-torch":
        if resolved_model_task not in {"causal_lm", "masked_lm", "seq2seq_lm"}:
            raise ValueError(
                "axon_backend='pipeline2-torch' currently supports only "
                "causal_lm, masked_lm, and seq2seq_lm models"
            )
        if not str(device).startswith("cuda"):
            raise ValueError("axon_backend='pipeline2-torch' currently requires a CUDA device target")
        if compile_axon:
            raise ValueError("compile_axon is not supported with axon_backend='pipeline2-torch'")
        if trace_layers:
            raise ValueError("trace_layers is not supported with axon_backend='pipeline2-torch'")
    if axon_backend == "runtime2-torch" and trace_layers:
        raise ValueError(f"trace_layers is not supported with axon_backend={axon_backend!r}")
    if profile_axon and compile_axon:
        raise ValueError("--profile-axon is not supported together with --compile-axon")

    safetensors_files = _resolve_safetensors_paths(weights_path)
    default_hf_dir = weights_path if weights_path.is_dir() else safetensors_files[0].parent
    resolved_hf_model_dir = (hf_model_dir or default_hf_dir).resolve()
    model_config = _augment_model_config_from_checkpoint(
        model_dir=resolved_hf_model_dir,
        safetensors_files=safetensors_files,
        model_config=_load_model_config(resolved_hf_model_dir),
    )
    resolved_model_type = (
        str(model_config.get("model_type", "")).strip().lower()
        if isinstance(model_config, dict)
        else ""
    )
    if resolved_hf_experts_implementation is None and resolved_model_type == "gpt_oss":
        resolved_hf_experts_implementation = "grouped_mm"
    adapter_storage_dtype = (
        resolved_dtype
        if hf_strict_dtype
        else (
            _resolve_optional_torch_dtype_name(model_config.get("torch_dtype"))
            if isinstance(model_config, dict)
            else None
        )
        or resolved_dtype
    )
    effective_trust_remote_code = bool(
        trust_remote_code
        or _should_trust_remote_code(
            resolved_hf_model_dir,
            model_config=model_config,
        )
    )
    if effective_trust_remote_code:
        _prime_tiktoken_cache_from_model_dir(resolved_hf_model_dir)
        _ensure_einops_import_compat()
    declared_tokenizer = _tokenizer_pragma_for_checkpoint(
        axon_file=axon_file,
        checkpoint_id=str(resolved_hf_model_dir.relative_to(_repo_root() / "models"))
        if (_repo_root() / "models") in resolved_hf_model_dir.parents
        else resolved_hf_model_dir.name,
    )
    tokenizer_source = tokenizer or declared_tokenizer or str(resolved_hf_model_dir)
    if tokenizer is None and declared_tokenizer is None:
        for candidate in candidate_tokenizer_dirs(resolved_hf_model_dir):
            if looks_like_tokenizer_dir(candidate):
                tokenizer_source = str(candidate)
                break
        else:
            fallback_repo = _tokenizer_fallback_repo_id(resolved_hf_model_dir)
            if fallback_repo is not None:
                tokenizer_source = fallback_repo
    tokenizer_fallback = (
        _tokenizer_fallback_repo_id(resolved_hf_model_dir) or resolved_hf_model_dir.name
        if tokenizer is None and declared_tokenizer is None
        else None
    )
    prompts = _normalize_texts(text)

    with TemporaryDirectory(prefix="axon_benchmark_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        synapse_yaml_path = tmp_path / "lowered_synapse.yaml"
        generated_py_path = tmp_path / "generated_model.py"
        if effective_trust_remote_code and not os.environ.get("HF_MODULES_CACHE"):
            modules_cache = resolved_hf_model_dir.parent / ".hf_modules_cache"
            modules_cache.mkdir(parents=True, exist_ok=True)
            os.environ["HF_MODULES_CACHE"] = str(modules_cache)
            from transformers import dynamic_module_utils
            from transformers.utils import hub as transformers_hub

            dynamic_module_utils.HF_MODULES_CACHE = str(modules_cache)
            if hasattr(transformers_hub, "HF_MODULES_CACHE"):
                transformers_hub.HF_MODULES_CACHE = str(modules_cache)

        lowered_spec: dict[str, Any]
        resolved_axon = resolve_axon_program_from_path(axon_file).ast
        normalized_axon = normalize_closed_axon_file(resolved_axon)
        elaborated_axon = elaborate_closed_axon_file(normalized_axon)
        flat_axon = flatten_closed_axon_file(elaborated_axon)
        typed_axon = typecheck2_flat_axon_file(flat_axon)
        if optimize_ast:
            typed_axon = optimize_safe_flat_typed_axon_file(typed_axon)
        graph_program = lower_axon_program_to_graph_ir(typed_axon)
        if optimize_graph:
            from .axon import GraphOptimizeConfig

            effective_graph_backend_intrinsics = _default_graph_backend_intrinsics(
                axon_backend=axon_backend,
                graph_backend_intrinsics=graph_backend_intrinsics,
            )
            graph_program = optimize_graph_program(
                graph_program,
                config=GraphOptimizeConfig(backend_intrinsics=effective_graph_backend_intrinsics),
            )
        main_graph_module = next(
            module for module in graph_program.modules if module.name == graph_program.main_module
        )
        output_names = _graph_main_output_names(graph_program, main_graph_module)
        lowered_spec = {
            "synapse": 1,
            "model": {
                "inputs": {
                    value.name: {
                        "optional": value.optional or isinstance(value.type_expr, TypeOptional)
                    }
                    for value in main_graph_module.inputs
                },
                "outputs": {name: name for name in output_names},
                "graph": [],
                "blocks": {},
                "config": model_config or {},
                "meta": dict(graph_program.pragmas),
            },
        }
        if axon_backend == "runtime2-torch":
            model_cls = make_runtime2_torch_model_class(graph_program, model_config=model_config)
        else:
            if axon_backend == "codegen2-tinygrad":
                code = emit_tinygrad_model_code_from_graph_ir(
                    graph_program,
                    class_name=class_name,
                    model_config=model_config,
                    profile=profile_axon,
                )
            elif axon_backend == "codegen2-triton":
                code = emit_triton_model_code_from_graph_ir(
                    graph_program,
                    class_name=class_name,
                    model_config=model_config,
                    profile=profile_axon,
                )
            elif axon_backend == "codegen2-vllm":
                code = emit_vllm_model_code_from_graph_ir(
                    graph_program,
                    class_name=class_name,
                    model_config=model_config,
                    profile=profile_axon,
                )
            else:
                code = emit_torch_model_code_from_graph_ir(
                    graph_program,
                    class_name=class_name,
                    model_config=model_config,
                    profile=profile_axon,
                    align_devices=axon_backend == "pipeline2-torch",
                )
            generated_py_path.write_text(code, encoding="utf-8")
            model_cls = _load_generated_class(generated_py_path, class_name)

        hf_config: Any | None = None
        reference_quant_config: Any | None = None
        from_pretrained_quant_config: Any | None = None
        reference_quant_method: str | None = None
        if resolved_model_task in {"masked_lm", "seq2seq_lm"} or resolved_model_type in {
            "phi3",
            "phi3small",
            "deepseek",
            "deepseek_v2",
            "deepseek_v3",
            "deepseekv3",
            "deepseek_v4",
            "gemma3",
            "gemma4",
            "gpt_oss",
            "mistral4",
            "mistral3",
            "llama4",
        }:
            hf_config = _load_auto_config_with_compat_fallback(
                resolved_hf_model_dir,
                trust_remote_code=effective_trust_remote_code,
            )
            hf_config = _normalize_rope_numeric_fields(hf_config)
            hf_config = _patch_mistral4_config_compat(hf_config)
            hf_config = _patch_deepseek_v4_reference_runtime_config(hf_config)
            if str(getattr(hf_config, "model_type", "")).strip().lower() == "deepseek":
                rope_scaling = getattr(hf_config, "rope_scaling", None)
                if isinstance(rope_scaling, dict):
                    rope_type = str(rope_scaling.get("type", rope_scaling.get("rope_type", "")))
                    if rope_type in {"", "default"}:
                        setattr(hf_config, "rope_scaling", None)
            reference_quant_method = _read_quant_method(hf_config)
            reference_quant_config = _build_reference_quantization_config(hf_config)
            from_pretrained_quant_config = reference_quant_config
            if hf_strict_dtype and reference_quant_config is not None:
                print(
                    "HF strict dtype enabled; not passing quantization_config to from_pretrained"
                )
                from_pretrained_quant_config = None
            if hf_strict_dtype and getattr(hf_config, "quantization_config", None) is not None:
                print("HF strict dtype enabled; removing config.quantization_config")
                try:
                    delattr(hf_config, "quantization_config")
                except Exception:
                    pass
            model_type = str(getattr(hf_config, "model_type", "")).strip().lower()
            if model_type in {"deepseek_v3", "deepseek"}:
                if not hasattr(hf_config, "num_experts") and hasattr(
                    hf_config, "num_local_experts"
                ):
                    setattr(hf_config, "num_experts", int(getattr(hf_config, "num_local_experts")))
        if (
            hf_config is None
            and hf_attn_implementation is None
            and resolved_hf_experts_implementation is None
            and axon_backend == "pipeline2-torch"
            and resolved_model_task in {"causal_lm", "seq2seq_lm"}
            and not skip_hf
        ):
            hf_config = _load_auto_config_with_compat_fallback(
                resolved_hf_model_dir,
                trust_remote_code=effective_trust_remote_code,
            )
            hf_config = _patch_deepseek_v4_reference_runtime_config(hf_config)
        if hf_attn_implementation is not None:
            token = str(hf_attn_implementation).strip()
            if not token:
                raise ValueError("--hf-attn-implementation must not be empty")
            if hf_config is None:
                hf_config = _load_auto_config_with_compat_fallback(
                    resolved_hf_model_dir,
                    trust_remote_code=effective_trust_remote_code,
                )
            setattr(hf_config, "_attn_implementation", token)
        if resolved_hf_experts_implementation is not None:
            if hf_config is None:
                hf_config = _load_auto_config_with_compat_fallback(
                    resolved_hf_model_dir,
                    trust_remote_code=effective_trust_remote_code,
                )
            _apply_hf_experts_implementation(hf_config, resolved_hf_experts_implementation)
        exec_device_str = str(resolved_device)
        tokenizer_obj, input_ids_cpu, attention_mask_cpu = tokenize_prompts(
            prompts=prompts,
            tokenizer_source=tokenizer_source,
            tokenizer_fallback=tokenizer_fallback,
            device=torch.device("cpu"),
            max_len=max_len,
            lowered_spec=lowered_spec,
            trust_remote_code=effective_trust_remote_code,
        )
        model_inputs = lowered_spec.get("model", {}).get("inputs", {})
        model_input_names = (
            set(model_inputs.keys()) if isinstance(model_inputs, dict) else {"input_ids"}
        )
        syn_mask_key = _pick_first_existing_name(
            ("attn_mask", "attention_mask", "encoder_attention_mask"),
            model_input_names,
        )
        syn_input_ids_key = _pick_first_existing_name(
            ("input_ids", "encoder_input_ids"),
            model_input_names,
        )
        if syn_input_ids_key is None:
            syn_input_ids_key = "input_ids"
        syn_decoder_input_ids_key = _pick_first_existing_name(
            ("decoder_input_ids",),
            model_input_names,
        )
        syn_decoder_mask_key = _pick_first_existing_name(
            ("decoder_attention_mask", "decoder_attn_mask"),
            model_input_names,
        )

        def _build_io_for_device(target_device: torch.device) -> dict[str, Any]:
            input_ids = input_ids_cpu.to(target_device)
            attention_mask = (
                None if attention_mask_cpu is None else attention_mask_cpu.to(target_device)
            )
            hf_inputs: dict[str, Any] = {"input_ids": input_ids}
            if attention_mask is not None:
                hf_inputs["attention_mask"] = attention_mask
            hf_generate_inputs = dict(hf_inputs)
            decoder_input_ids: torch.Tensor | None = None
            decoder_attention_mask: torch.Tensor | None = None
            if resolved_model_task == "seq2seq_lm":
                if hf_config is None:
                    raise ValueError("seq2seq_lm failed to resolve HF config")
                decoder_input_ids, decoder_attention_mask = _build_seq2seq_decoder_inputs(
                    encoder_input_ids=input_ids,
                    encoder_attention_mask=attention_mask,
                    hf_config=hf_config,
                )
                hf_inputs["decoder_input_ids"] = decoder_input_ids
                hf_inputs["decoder_attention_mask"] = decoder_attention_mask

            use_mask_for_syn = bool(attention_mask is not None and syn_mask_key is not None)
            syn_inputs: dict[str, Any] = {syn_input_ids_key: input_ids}
            if use_mask_for_syn and attention_mask is not None and syn_mask_key is not None:
                syn_inputs[syn_mask_key] = attention_mask
            if resolved_model_task == "seq2seq_lm":
                if decoder_input_ids is None:
                    raise ValueError("seq2seq_lm missing decoder_input_ids for Synapse forward")
                if syn_decoder_input_ids_key is None:
                    raise ValueError(
                        "seq2seq_lm requires Axon model input 'decoder_input_ids' for Synapse forward"
                    )
                syn_inputs[syn_decoder_input_ids_key] = decoder_input_ids
                if decoder_attention_mask is not None and syn_decoder_mask_key is not None:
                    syn_inputs[syn_decoder_mask_key] = decoder_attention_mask

            hf_forward_inputs = dict(hf_inputs)
            if (
                resolved_model_task == "causal_lm"
                and attention_mask is not None
                and resolved_model_type != "deepseek"
                and resolved_model_type != "gpt_neox"
                and resolved_model_type != "exaone4"
            ):
                pos_ids = attention_mask.to(torch.long).cumsum(dim=-1) - 1
                pos_ids = pos_ids.masked_fill(attention_mask == 0, 1)
                hf_forward_inputs["position_ids"] = pos_ids
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "hf_inputs": hf_inputs,
                "hf_generate_inputs": hf_generate_inputs,
                "hf_forward_inputs": hf_forward_inputs,
                "syn_inputs": syn_inputs,
            }

        def _run_hf_side(target_device_str: str) -> dict[str, Any]:
            target_device = _resolve_device(target_device_str)
            _patch_cache_api_compat()
            local_state_ref_cpu: dict[str, torch.Tensor] | None = None
            hf_device_map: dict[str, str] | None = None

            def _mistral4_device_map_for_itt(
                device_map: dict[str, str] | None,
            ) -> dict[str, str] | None:
                if device_map is None:
                    return None
                # Pipeline stage inference uses language_model.* paths from Axon specs.
                # HF mistral4 text stack is registered under model.* / lm_head.
                rewritten: dict[str, str] = {}
                for key, value in device_map.items():
                    if key.startswith("language_model.model."):
                        rewritten[f"model.{key[len('language_model.model.') :]}"] = value
                    elif key == "language_model.lm_head":
                        rewritten["lm_head"] = value
                    else:
                        rewritten[key] = value
                return rewritten

            def _normalize_hf_device_map_for_loading(
                device_map: dict[str, str],
                *,
                hf_param_names: set[str] | None,
                state_dict: dict[str, torch.Tensor],
                first_device: str,
                last_device: str,
            ) -> dict[str, str]:
                def _prefix_covers_name(prefix: str, name: str) -> bool:
                    return name == prefix or name.startswith(prefix + ".")

                def _candidate_hf_prefixes(prefix: str) -> tuple[str, ...]:
                    out = [
                        prefix,
                        f"model.{prefix}",
                        f"transformer.{prefix}",
                        prefix.replace("language_model.model.", "language_model.", 1),
                        f"model.{prefix.replace('language_model.model.', 'language_model.', 1)}",
                        prefix.replace("language_model.", "model.language_model.", 1),
                        prefix.replace("vision_tower.", "model.vision_tower.", 1),
                        prefix.replace("multi_modal_projector.", "model.multi_modal_projector.", 1),
                    ]
                    dedup: list[str] = []
                    seen: set[str] = set()
                    for item in out:
                        item = item.strip(".")
                        if not item or item in seen:
                            continue
                        seen.add(item)
                        dedup.append(item)
                    return tuple(dedup)

                def _match_prefix_to_hf_names(prefix: str, names: set[str]) -> str:
                    if not names:
                        return prefix
                    candidates = _candidate_hf_prefixes(prefix)
                    best = prefix
                    best_score = -1
                    for candidate in candidates:
                        score = sum(1 for name in names if _prefix_covers_name(candidate, name))
                        if score > best_score:
                            best = candidate
                            best_score = score
                    return best

                normalized: dict[str, str] = dict(device_map)
                if not hf_param_names:
                    rewritten: dict[str, str] = {}
                    for key, value in normalized.items():
                        mapped = key
                        # Runtime stage usage uses stripped GPT/BLOOM prefixes for some specs.
                        # Normalize these for HF loading when HF key introspection is unavailable.
                        if (
                            key.startswith("h.")
                            or key == "ln_f"
                            or key.startswith("word_embeddings")
                        ):
                            mapped = f"transformer.{key}"
                        rewritten.setdefault(mapped, value)
                    normalized = rewritten
                else:
                    rewritten = {}
                    for key, value in normalized.items():
                        mapped = _match_prefix_to_hf_names(key, hf_param_names)
                        rewritten.setdefault(mapped, value)
                    normalized = rewritten
                # Some HF models keep RoPE buffers in a module without parameters.
                normalized.setdefault("rotary_emb", first_device)
                normalized.setdefault("model.rotary_emb", first_device)
                normalized.setdefault("transformer.rotary_emb", first_device)
                embed_device = next(
                    (
                        device
                        for prefix, device in normalized.items()
                        if prefix.endswith("embed_tokens")
                        or prefix.endswith("word_embeddings")
                        or prefix.endswith("wte")
                    ),
                    None,
                )
                if embed_device is not None:
                    lm_head_device = (
                        embed_device if "lm_head.weight" not in state_dict else last_device
                    )
                    normalized.setdefault("lm_head", lm_head_device)
                    normalized.setdefault("language_model.lm_head", lm_head_device)
                    normalized.setdefault("model.lm_head", lm_head_device)
                if hf_param_names:
                    uncovered: list[str] = []
                    for name in hf_param_names:
                        if any(_prefix_covers_name(prefix, name) for prefix in normalized):
                            continue
                        uncovered.append(name)
                    for name in sorted(uncovered):
                        root = name.rsplit(".", 1)[0] if "." in name else name
                        normalized.setdefault(root, last_device)
                return normalized

            hf_input_device = target_device
            if axon_backend == "pipeline2-torch" and resolved_model_task in {
                "causal_lm",
                "seq2seq_lm",
            }:
                visible_devices = [
                    f"cuda:{idx}" for idx in range(max(1, torch.cuda.device_count()))
                ]
                ordered_hf_param_names = _collect_ordered_hf_param_names_for_device_map(
                    model_task=resolved_model_task,
                    hf_config=hf_config,
                    trust_remote_code=effective_trust_remote_code,
                )
                hf_device_map, hf_stage_spans = _build_generic_hf_device_map_from_param_names(
                    ordered_hf_param_names,
                    devices=visible_devices,
                )
                if hf_device_map is not None:
                    _colocate_tied_hf_output_embeddings(
                        hf_device_map,
                        has_explicit_output_head_weight=_checkpoint_has_explicit_output_head_weight(
                            resolved_hf_model_dir
                        ),
                    )
                    hf_input_device = torch.device(visible_devices[0])
                    if hf_stage_spans:
                        print(
                            "HF device_map stages:",
                            ", ".join(
                                f"[{start},{stop})->{device}"
                                for start, stop, device in hf_stage_spans
                            ),
                        )
                    else:
                        print(
                            "HF device_map stages:",
                            ", ".join(visible_devices),
                            "(no numeric layer axis discovered)",
                        )
            hf_io_device = hf_input_device
            io = _build_io_for_device(hf_io_device)
            hf: Any
            hf_model: Any | None = None
            if resolved_model_task == "masked_lm":
                if reference_quant_config is not None:
                    raise RuntimeError(
                        "masked_lm reference path does not support mxfp4 quantization"
                    )
                hf = _load_hf_masked_lm_reference(
                    model_dir=resolved_hf_model_dir,
                    safetensors_files=safetensors_files,
                    resolved_dtype=resolved_dtype,
                    resolved_device=target_device,
                    hf_config=hf_config,
                    trust_remote_code=effective_trust_remote_code,
                    model_config=model_config,
                )
                hf_model = hf
            elif resolved_model_task == "seq2seq_lm":
                _ensure_transformers_import_compat()
                seq2seq_kwargs: dict[str, Any] = {
                    "local_files_only": True,
                    "torch_dtype": resolved_dtype,
                    "config": hf_config,
                    "quantization_config": from_pretrained_quant_config,
                    "trust_remote_code": effective_trust_remote_code,
                    "device_map": hf_device_map,
                }
                if resolved_hf_experts_implementation is not None:
                    seq2seq_kwargs["experts_implementation"] = (
                        resolved_hf_experts_implementation
                    )
                with _patch_torch_equal_for_cross_device_hf_load(
                    enabled=hf_device_map is not None
                ):
                    hf_model = AutoModelForSeq2SeqLM.from_pretrained(
                        str(resolved_hf_model_dir),
                        **seq2seq_kwargs,
                    )
                hf = (
                    hf_model.eval()
                    if hf_device_map is not None
                    else hf_model.to(device=target_device, dtype=resolved_dtype).eval()
                )
            else:
                try:
                    _ensure_transformers_import_compat()
                    if (
                        resolved_model_type in {"deepseek", "deepseek_v3", "deepseekv3"}
                        and resolved_hf_model_dir.name == "DeepSeek-V3-Test"
                    ):
                        local_state_ref_cpu = _load_state_dict(
                            safetensors_files,
                            device=torch.device("cpu"),
                            dtype=resolved_dtype,
                            model_config=model_config,
                            storage_dtype=adapter_storage_dtype,
                        )
                        hf_model = AutoModelForCausalLM.from_config(
                            hf_config,
                            trust_remote_code=effective_trust_remote_code,
                        )
                        hf_model.load_state_dict(local_state_ref_cpu, strict=True)
                    elif (
                        reference_quant_method == "mxfp4"
                        and reference_quant_config is not None
                        and bool(getattr(reference_quant_config, "dequantize", False))
                    ):
                        print("HF: loading dequantized MXFP4 checkpoint via explicit state-dict path")
                        hf_model = _load_hf_causal_lm_from_dequantized_mxfp4_state(
                            safetensors_files=safetensors_files,
                            dtype=resolved_dtype,
                            hf_config=hf_config,
                            trust_remote_code=effective_trust_remote_code,
                            device_map=hf_device_map,
                            target_device=target_device,
                            model_config=model_config,
                        )
                    elif (
                        str(getattr(hf_config, "model_type", "")).strip().lower()
                        == "deepseek_v4"
                        and reference_quant_method == "fp8"
                        and reference_quant_config is not None
                        and bool(getattr(reference_quant_config, "dequantize", False))
                    ):
                        print(
                            "HF: loading dequantized DeepSeek-V4 FP8 checkpoint via explicit state-dict path"
                        )
                        hf_model = _load_hf_causal_lm_from_dequantized_deepseek_v4_fp8_state(
                            safetensors_files=safetensors_files,
                            dtype=resolved_dtype,
                            hf_config=hf_config,
                            trust_remote_code=effective_trust_remote_code,
                            device_map=hf_device_map,
                            target_device=target_device,
                        )
                    else:
                        causal_kwargs: dict[str, Any] = {
                            "local_files_only": True,
                            "torch_dtype": resolved_dtype,
                            "config": hf_config,
                            "trust_remote_code": effective_trust_remote_code,
                        }
                        if from_pretrained_quant_config is not None:
                            causal_kwargs["quantization_config"] = from_pretrained_quant_config
                        if hf_device_map is not None:
                            causal_kwargs["device_map"] = hf_device_map
                        if resolved_hf_experts_implementation is not None:
                            causal_kwargs["experts_implementation"] = (
                                resolved_hf_experts_implementation
                            )
                        with _patch_torch_equal_for_cross_device_hf_load(
                            enabled=hf_device_map is not None
                        ):
                            hf_model = AutoModelForCausalLM.from_pretrained(
                                str(resolved_hf_model_dir),
                                **causal_kwargs,
                            )
                    hf = (
                        cast(Any, hf_model).eval()
                        if hf_device_map is not None
                        else cast(Any, hf_model)
                        .to(device=target_device, dtype=resolved_dtype)
                        .eval()
                    )
                except Exception:
                    model_type = (
                        str(getattr(hf_config, "model_type", "")).strip().lower()
                        if hf_config is not None
                        else ""
                    )
                    if model_type in {"mistral4", "gemma4", "mistral3", "llama4"}:
                        _ensure_transformers_import_compat()
                        mistral4_auto_map_target = Path(
                            "models/mistralai/Mistral-Small-4-119B-2603"
                        )
                        use_mistral4_auto_map = (
                            model_type == "mistral4"
                            and resolved_hf_model_dir.as_posix().endswith(
                                mistral4_auto_map_target.as_posix()
                            )
                        )
                        multimodal_device_map: dict[str, str] | str | None
                        if use_mistral4_auto_map:
                            multimodal_device_map = "auto"
                        else:
                            multimodal_device_map = (
                                _mistral4_device_map_for_itt(hf_device_map)
                                if model_type == "mistral4"
                                else hf_device_map
                            )
                        mistral4_key_mapping: dict[str, str] | None = None
                        load_dtype = resolved_dtype
                        if model_type == "mistral4":
                            mistral4_key_mapping = {
                                "language_model.model.": "model.",
                                "language_model.lm_head": "lm_head",
                            }
                            # Mistral4 checkpoints with fp8 tensors materialize cleanly via bf16 load,
                            # then can be promoted to fp32 when strict float32 is requested.
                            if resolved_dtype == torch.float32:
                                load_dtype = torch.bfloat16
                        with _patch_torch_equal_for_cross_device_hf_load(
                            enabled=multimodal_device_map is not None
                        ):
                            hf_model = AutoModelForImageTextToText.from_pretrained(
                                str(resolved_hf_model_dir),
                                local_files_only=True,
                                torch_dtype=load_dtype,
                                config=hf_config,
                                trust_remote_code=effective_trust_remote_code,
                                device_map=multimodal_device_map,
                                key_mapping=mistral4_key_mapping,
                            )
                        hf_base = (
                            cast(Any, hf_model).eval()
                            if multimodal_device_map is not None
                            else cast(Any, hf_model)
                            .to(device=target_device, dtype=load_dtype)
                            .eval()
                        )
                        hf = (
                            cast(Any, hf_base).to(dtype=torch.float32).eval()
                            if model_type == "mistral4" and resolved_dtype == torch.float32
                            else hf_base
                        )
                    elif is_black_mamba_config_dir(resolved_hf_model_dir):
                        generated_state = _load_state_dict(
                            safetensors_files,
                            device=target_device,
                            dtype=resolved_dtype,
                            model_config=model_config,
                            storage_dtype=adapter_storage_dtype,
                        )
                        hf = (
                            BlackMambaReferenceModel.from_state_dict(
                                model_dir=resolved_hf_model_dir,
                                state_dict=dict(generated_state),
                            )
                            .to(target_device)
                            .eval()
                        )
                        generated_state.clear()
                    else:
                        raise
            if (
                resolved_model_task == "causal_lm"
                and not _checkpoint_has_explicit_output_head_weight(resolved_hf_model_dir)
                and hasattr(hf, "get_output_embeddings")
                and hasattr(hf, "get_input_embeddings")
            ):
                out_emb = hf.get_output_embeddings()
                in_emb = hf.get_input_embeddings()
                if (
                    out_emb is not None
                    and in_emb is not None
                    and hasattr(out_emb, "weight")
                    and hasattr(in_emb, "weight")
                ):
                    out_emb.weight = in_emb.weight
                    print(
                        "HF: tied output embeddings to input embeddings (checkpoint has no explicit output head weight)"
                    )
            if hf_strict_dtype:
                coerced_params, coerced_buffers = _force_model_floating_dtype(
                    hf, target_dtype=resolved_dtype
                )
                if coerced_params or coerced_buffers:
                    print(
                        "HF strict dtype cast:",
                        f"params={coerced_params}",
                        f"buffers={coerced_buffers}",
                        f"dtype={resolved_dtype}",
                    )
            if _rebuild_hf_dummy_tokens_mask_from_config(hf):
                print("HF: rebuilt dummy_tokens_mask from config")
            if resolved_model_task == "causal_lm" and _ensure_hf_generation_mixin(hf):
                print("HF: added GenerationMixin to custom causal LM class")
            if resolved_hf_experts_implementation is not None:
                set_experts = getattr(hf, "set_experts_implementation", None)
                if callable(set_experts):
                    set_experts(resolved_hf_experts_implementation)
                print(
                    "HF experts implementation:",
                    resolved_hf_experts_implementation,
                )
            patched_mistral4_experts = _patch_hf_mistral4_experts_from_checkpoint(
                hf,
                config=hf_config,
                safetensors_files=safetensors_files,
                resolved_dtype=resolved_dtype,
                resolved_device=target_device,
            )
            if patched_mistral4_experts > 0:
                print(
                    f"HF: patched Mistral4 experts from checkpoint ({patched_mistral4_experts} tensors)"
                )
            rebuilt_phi3small_longrope = _rebuild_hf_phi3small_longrope_buffers(hf)
            if rebuilt_phi3small_longrope > 0:
                print(
                    f"HF: rebuilt Phi-3-small LongRoPE buffers ({rebuilt_phi3small_longrope} modules)"
                )
            refreshed_rotary = _refresh_hf_rotary_caches_if_needed(hf, dtype=resolved_dtype)
            if refreshed_rotary > 0:
                print(f"HF: refreshed rotary caches ({refreshed_rotary} modules)")
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

            hf_layer_inputs: dict[int, torch.Tensor] = {}
            hf_layer_outputs: dict[int, torch.Tensor] = {}
            hf_hook_handles: list[Any] = []
            if hf_device_map is not None:
                hf_hook_handles.extend(_patch_hf_shared_modules_for_device_map(hf))
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

            def _run_hf_forward(model: Any) -> torch.Tensor:
                hf_forward_kwargs = _move_hf_token_inputs_to_embedding_devices(
                    model, io["hf_forward_inputs"]
                )
                if resolved_model_task in {"causal_lm", "seq2seq_lm"}:
                    hf_forward_kwargs["use_cache"] = False
                if _is_deepseek_family_model_type(resolved_model_type):
                    batch_size = int(io["input_ids"].shape[0])
                    if batch_size > 1:
                        logits_parts: list[torch.Tensor] = []
                        for batch_idx in range(batch_size):
                            sample_kwargs: dict[str, Any] = {}
                            for key, value in hf_forward_kwargs.items():
                                if (
                                    torch.is_tensor(value)
                                    and value.ndim > 0
                                    and int(value.shape[0]) == batch_size
                                ):
                                    sample_kwargs[key] = value[batch_idx : batch_idx + 1]
                                else:
                                    sample_kwargs[key] = value
                            logits_parts.append(_extract_logits(model(**sample_kwargs)))
                        if logits_parts:
                            return torch.cat(logits_parts, dim=0)
                if resolved_model_type == "phi3small":
                    return _run_phi3small_chunked_logits(
                        model,
                        hf_forward_kwargs=hf_forward_kwargs,
                    )
                return _extract_logits(model(**hf_forward_kwargs))

            hf_gen: torch.Tensor | None = None
            hf_time = 0.0
            hf_forward_samples: list[float] = []
            hf_generate_samples: list[float] = []
            with _patch_transformers_mask_device_map_inputs(enabled=hf_device_map is not None):
                if not run_generate_benchmark:
                    with torch.no_grad():
                        hf_logits, hf_time, hf_forward_samples = _time_forward_repeated(
                            "HF",
                            lambda model=hf: _run_hf_forward(model),
                            warmup=forward_warmup,
                            repeat=forward_repeat,
                        )
                else:
                    hf_forward_time = 0.0
                    with torch.no_grad():
                        hf_t0 = time.perf_counter()
                        hf_logits = _run_hf_forward(hf)
                        hf_forward_time = time.perf_counter() - hf_t0
                        if _is_deepseek_family_model_type(resolved_model_type) and not bool(
                            torch.isfinite(hf_logits).all()
                        ):
                            print(
                                "HF logits contained non-finite values for DeepSeek; "
                                "retrying prompt-by-prompt HF forward"
                            )
                            batch_size = int(io["input_ids"].shape[0])
                            if batch_size > 1:
                                hf_logits_parts: list[torch.Tensor] = []
                                for batch_idx in range(batch_size):
                                    sample_kwargs: dict[str, Any] = {}
                                    for key, value in io["hf_forward_inputs"].items():
                                        if (
                                            torch.is_tensor(value)
                                            and value.ndim > 0
                                            and int(value.shape[0]) == batch_size
                                        ):
                                            sample_kwargs[key] = value[batch_idx : batch_idx + 1]
                                        else:
                                            sample_kwargs[key] = value
                                    sample_kwargs["use_cache"] = False
                                    hf_logits_parts.append(_extract_logits(hf(**sample_kwargs)))
                                if hf_logits_parts:
                                    hf_logits = torch.cat(hf_logits_parts, dim=0)
                            if not bool(torch.isfinite(hf_logits).all()):
                                print(
                                    "DeepSeek HF logits still non-finite after per-prompt retry; "
                                    "retrying one additional full forward pass"
                                )
                                hf_t0 = time.perf_counter()
                                hf_logits = _run_hf_forward(hf)
                                hf_forward_time = time.perf_counter() - hf_t0

                    hf_max_new_tokens = (
                        max(1, max_len)
                        if resolved_model_task == "seq2seq_lm"
                        else max(1, max_len - int(io["input_ids"].shape[1]))
                    )

                    def _run_hf_generate(model: Any) -> torch.Tensor:
                        with _preserve_requested_hf_experts_during_generate(
                            model, resolved_hf_experts_implementation
                        ):
                            if _is_deepseek_family_model_type(resolved_model_type):
                                pad_id = tokenizer_obj.eos_token_id
                                generated: list[torch.Tensor] = []
                                batch_size = int(io["input_ids"].shape[0])
                                for batch_idx in range(batch_size):
                                    sample_inputs: dict[str, Any] = {}
                                    for key, value in io["hf_generate_inputs"].items():
                                        if (
                                            torch.is_tensor(value)
                                            and value.ndim > 0
                                            and int(value.shape[0]) == batch_size
                                        ):
                                            sample_value = value[batch_idx : batch_idx + 1]
                                            if key in {"input_ids", "attention_mask"}:
                                                mask = io["attention_mask"][batch_idx : batch_idx + 1]
                                                keep = mask.to(dtype=torch.bool)[0]
                                                sample_value = sample_value[:, keep]
                                            sample_inputs[key] = sample_value
                                        else:
                                            sample_inputs[key] = value
                                    generated.append(
                                        _call_generate_compatible(
                                            model,
                                            **sample_inputs,
                                            max_new_tokens=hf_max_new_tokens,
                                            eos_token_id=tokenizer_obj.eos_token_id,
                                            pad_token_id=pad_id,
                                            num_beams=1,
                                            do_sample=False,
                                            use_cache=False,
                                        )[0]
                                    )
                                max_out = max(int(item.shape[0]) for item in generated)
                                padded = [
                                    torch.nn.functional.pad(
                                        item,
                                        (0, max_out - int(item.shape[0])),
                                        value=int(pad_id if pad_id is not None else 0),
                                    )
                                    for item in generated
                                ]
                                return torch.stack(padded, dim=0)
                            return _call_generate_compatible(
                                model,
                                **_move_hf_token_inputs_to_embedding_devices(
                                    model, io["hf_generate_inputs"]
                                ),
                                max_new_tokens=hf_max_new_tokens,
                                eos_token_id=tokenizer_obj.eos_token_id,
                                pad_token_id=tokenizer_obj.eos_token_id,
                                num_beams=1,
                                do_sample=False,
                                use_cache=False,
                            )

                    hf_gen, hf_time, hf_generate_samples = _time_generate_repeated(
                        "HF",
                        lambda model=hf: _run_hf_generate(model),
                        warmup=generate_warmup,
                        repeat=generate_repeat,
                    )
            hf_logits_cpu = hf_logits.detach().cpu()
            hf_gen_cpu = None if hf_gen is None else hf_gen.detach().cpu()
            dummy_mask = _build_phi3small_dummy_vocab_mask(
                model_config=model_config,
                vocab_size=int(hf_logits_cpu.shape[-1]),
                device=torch.device("cpu"),
            )
            if resolved_model_type == "phi3small":
                local_state_ref_cpu = _clone_hf_state_dict(
                    hf,
                    device=torch.device("cpu"),
                    dtype=resolved_dtype,
                )
            for handle in hf_hook_handles:
                handle.remove()
            del hf_hook_handles
            del hf
            if hf_model is not None:
                del hf_model
            _cleanup(target_device)
            return {
                "logits": hf_logits_cpu,
                "gen": hf_gen_cpu,
                "time": hf_time,
                "forward_samples": hf_forward_samples,
                "generate_samples": hf_generate_samples,
                "dummy_mask": dummy_mask,
                "decoder_attention_mask": io.get("decoder_attention_mask"),
                "layer_inputs": hf_layer_inputs,
                "layer_outputs": hf_layer_outputs,
                "state_ref_cpu": local_state_ref_cpu,
                "device": str(target_device),
            }

        hf_result: dict[str, Any] = {}
        hf_logits: torch.Tensor | None = None
        hf_gen: torch.Tensor | None = None
        hf_time: float | None = None
        hf_dummy_tokens_mask: torch.Tensor | None = None
        hf_layer_inputs: dict[int, torch.Tensor] = {}
        hf_layer_outputs: dict[int, torch.Tensor] = {}
        hf_forward_samples: list[float] = []
        hf_generate_samples: list[float] = []
        hf_exec_device_str = "skipped"
        state_ref_cpu: dict[str, torch.Tensor] | None = None
        decoder_attention_mask_for_metrics: torch.Tensor | None = None
        if not skip_hf:
            try:
                hf_result = _run_hf_side(exec_device_str)
            except Exception as exc:
                if not _is_cuda_oom(exc, device=exec_device_str):
                    raise
                if not oom_cpu_fallback:
                    raise
                _cleanup_cuda_after_oom(exec_device_str)
                print(f"CUDA OOM on {exec_device_str}; retrying HF on cpu")
                hf_result = _run_hf_side("cpu")

            hf_logits = cast(torch.Tensor, hf_result["logits"])
            hf_gen = cast(torch.Tensor | None, hf_result["gen"])
            hf_time = float(hf_result["time"])
            hf_forward_samples = list(cast(Sequence[float], hf_result.get("forward_samples", [])))
            hf_generate_samples = list(cast(Sequence[float], hf_result.get("generate_samples", [])))
            hf_dummy_tokens_mask = cast(torch.Tensor | None, hf_result["dummy_mask"])
            hf_layer_inputs = cast(dict[int, torch.Tensor], hf_result["layer_inputs"])
            hf_layer_outputs = cast(dict[int, torch.Tensor], hf_result["layer_outputs"])
            hf_exec_device_str = cast(str, hf_result["device"])
            state_ref_cpu = cast(dict[str, torch.Tensor] | None, hf_result["state_ref_cpu"])
            decoder_attention_mask_for_metrics = cast(
                torch.Tensor | None, hf_result.get("decoder_attention_mask")
            )
            hf_result.clear()
            _cleanup(_resolve_device(exec_device_str))

        def _run_syn_side(target_device_str: str) -> dict[str, Any]:
            target_device = _resolve_device(target_device_str)
            io = _build_io_for_device(target_device)
            local_state_dict = state_ref_cpu
            param_devices = (
                [f"cuda:{idx}" for idx in range(max(1, torch.cuda.device_count()))]
                if axon_backend == "pipeline2-torch"
                else [str(target_device)]
                if axon_backend in {"codegen2-torch", "codegen2-triton"}
                else None
            )
            state_load_device = (
                torch.device("cpu")
                if axon_backend in {"pipeline2-torch", "codegen2-torch", "codegen2-triton"}
                else target_device
            )
            if axon_backend == "codegen2-tinygrad":
                if local_state_dict is None:
                    syn = model_cls.from_safetensors(
                        safetensors_files,
                        model_config=model_config,
                        dtype=str(resolved_dtype).removeprefix("torch."),
                    ).to(target_device).eval()
                else:
                    syn = model_cls.from_state_dict(local_state_dict).to(target_device).eval()
            elif local_state_dict is None:
                local_state_dict = _load_state_dict(
                    safetensors_files,
                    device=state_load_device,
                    dtype=resolved_dtype,
                    model_config=model_config,
                    storage_dtype=adapter_storage_dtype,
                    param_devices=param_devices,
                )
            elif axon_backend != "pipeline2-torch":
                local_state_dict = {
                    key: (
                        value.to(device=target_device, dtype=resolved_dtype)
                        if value.is_floating_point()
                        else value.to(device=target_device)
                    )
                    for key, value in local_state_dict.items()
                }
            if axon_backend in {"pipeline2-torch", "codegen2-torch", "codegen2-triton"}:
                syn = model_cls.from_state_dict(
                    local_state_dict,
                    param_devices=param_devices,
                ).eval()
            elif axon_backend != "codegen2-tinygrad":
                syn = model_cls.from_state_dict(local_state_dict).to(target_device).eval()
            if profile_axon:
                enable_profile = getattr(syn, "enable_profile", None)
                if not callable(enable_profile):
                    raise ValueError(f"--profile-axon is not supported with axon_backend={axon_backend!r}")
                enable_profile(True, cuda=target_device.type == "cuda", reset=True)
            if local_state_dict is not None and local_state_dict is not state_ref_cpu:
                local_state_dict.clear()
            if local_state_dict is not None:
                del local_state_dict
            _cleanup(target_device)
            align_targets = [syn]
            stage_models = getattr(syn, "stages", None)
            if isinstance(stage_models, Sequence):
                align_targets.extend(stage_models)
            for align_model in align_targets:
                setattr(align_model, "_hf_align_mask_contract", align_mask_contract)
                setattr(align_model, "_hf_align_position_ids", align_position_ids)
                setattr(align_model, "_hf_align_add_fp32_accum", align_add_fp32)
                setattr(align_model, "_hf_align_linear_fp32_accum", align_linear_fp32)
                setattr(align_model, "_hf_align_norm_fp32", align_norm_fp32)
            syn = _maybe_compile_model(
                syn,
                enabled=compile_axon,
                backend=compile_backend,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=compile_dynamic,
            )
            syn_layer_inputs: dict[int, torch.Tensor] = {}
            syn_layer_outputs: dict[int, torch.Tensor] = {}
            original_block_name: str | None = None
            original_block_call: Any | None = None
            if trace_layers:
                preferred_block_names = (
                    "_block_gpt_oss_block",
                    "_block_phi3_block",
                    "_block_phi3small_block",
                    "_block_phi3minimedium_block",
                )
                for candidate_name in preferred_block_names:
                    block_candidate = getattr(syn, candidate_name, None)
                    if callable(block_candidate):
                        original_block_name = candidate_name
                        original_block_call = block_candidate
                        break
                if original_block_call is None:
                    for attr in dir(syn):
                        if not (attr.startswith("_block_") and attr.endswith("_block")):
                            continue
                        block_candidate = getattr(syn, attr, None)
                        if callable(block_candidate):
                            original_block_name = attr
                            original_block_call = block_candidate
                            break
            if (
                trace_layers
                and callable(original_block_call)
                and isinstance(original_block_name, str)
            ):

                def _syn_block_wrapper(*args: Any, **kwargs: Any) -> Any:
                    layer_raw = kwargs.get("i")
                    if layer_raw is None and len(args) >= 2:
                        layer_raw = args[1]
                    x_raw = kwargs.get("x")
                    if x_raw is None and len(args) >= 1:
                        x_raw = args[0]
                    if layer_raw is not None:
                        layer_idx = int(layer_raw)
                        if torch.is_tensor(x_raw):
                            syn_layer_inputs[layer_idx] = _to_cpu_float(x_raw)
                    out = original_block_call(*args, **kwargs)
                    if layer_raw is not None:
                        layer_idx = int(layer_raw)
                        if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
                            syn_layer_outputs[layer_idx] = _to_cpu_float(out[0])
                    return out

                setattr(syn, original_block_name, _syn_block_wrapper)

            def _run_syn_forward(model: Any = syn) -> torch.Tensor:
                if _is_deepseek_family_model_type(resolved_model_type):
                    syn_inputs = io["syn_inputs"]
                    sample_batch_size: int | None = None
                    for value in syn_inputs.values():
                        if torch.is_tensor(value) and value.ndim > 0:
                            sample_batch_size = int(value.shape[0])
                            break
                    if sample_batch_size is not None and sample_batch_size > 1:
                        logits_parts: list[torch.Tensor] = []
                        for batch_idx in range(sample_batch_size):
                            sample_inputs: dict[str, Any] = {}
                            for key, value in syn_inputs.items():
                                if (
                                    torch.is_tensor(value)
                                    and value.ndim > 0
                                    and int(value.shape[0]) == sample_batch_size
                                ):
                                    sample_inputs[key] = value[batch_idx : batch_idx + 1]
                                else:
                                    sample_inputs[key] = value
                            logits_parts.append(_extract_logits(model(**sample_inputs)))
                        if logits_parts:
                            return torch.cat(logits_parts, dim=0)
                return _extract_logits(model(**io["syn_inputs"]))

            syn_gen: torch.Tensor | None = None
            syn_time = 0.0
            syn_forward_samples: list[float] = []
            syn_generate_samples: list[float] = []
            if not run_generate_benchmark:
                with torch.no_grad():
                    syn_logits, syn_time, syn_forward_samples = _time_forward_repeated(
                        "AxonDerived",
                        _run_syn_forward,
                        warmup=forward_warmup,
                        repeat=forward_repeat,
                    )
            else:
                with torch.no_grad():
                    syn_logits = _run_syn_forward()

                def _run_syn_generate(model: Any = syn) -> torch.Tensor:
                    generate_kwargs: dict[str, Any] = {
                        "eos_token_id": tokenizer_obj.eos_token_id,
                        "max_len": max_len,
                    }
                    attention_mask = io["attention_mask"]
                    if attention_mask is not None:
                        if syn_mask_key == "attn_mask":
                            generate_kwargs["attn_mask"] = attention_mask
                        elif syn_mask_key == "attention_mask":
                            generate_kwargs["attention_mask"] = attention_mask
                    return model.generate(io["input_ids"], **generate_kwargs)

                syn_gen, syn_time, syn_generate_samples = _time_generate_repeated(
                    "AxonDerived",
                    _run_syn_generate,
                    warmup=generate_warmup,
                    repeat=generate_repeat,
                )
            if (
                trace_layers
                and callable(original_block_call)
                and isinstance(original_block_name, str)
            ):
                setattr(syn, original_block_name, original_block_call)
            profile_rows: list[dict[str, Any]] = []
            if profile_axon:
                profile_summary = getattr(syn, "profile_summary", None)
                if callable(profile_summary):
                    profile_rows = list(profile_summary(profile_axon_top_n))
                    _print_axon_profile_summary(profile_rows)
            syn_logits_cpu = syn_logits.detach().cpu()
            syn_gen_cpu = None if syn_gen is None else syn_gen.detach().cpu()
            del syn
            _cleanup(target_device)
            return {
                "logits": syn_logits_cpu,
                "gen": syn_gen_cpu,
                "time": syn_time,
                "forward_samples": syn_forward_samples,
                "generate_samples": syn_generate_samples,
                "layer_inputs": syn_layer_inputs,
                "layer_outputs": syn_layer_outputs,
                "device": str(target_device),
                "profile": profile_rows,
            }

        syn_result: dict[str, Any]
        try:
            syn_result = _run_syn_side(exec_device_str)
        except Exception as exc:
            if not _is_cuda_oom(exc, device=exec_device_str):
                raise
            if axon_backend == "pipeline2-torch":
                raise
            if not oom_cpu_fallback:
                raise
            _cleanup_cuda_after_oom(exec_device_str)
            print(f"CUDA OOM on {exec_device_str}; retrying AxonDerived on cpu")
            syn_result = _run_syn_side("cpu")

        syn_logits = cast(torch.Tensor, syn_result["logits"])
        syn_gen = cast(torch.Tensor | None, syn_result["gen"])
        syn_time = float(syn_result["time"])
        syn_forward_samples = list(cast(Sequence[float], syn_result.get("forward_samples", [])))
        syn_generate_samples = list(cast(Sequence[float], syn_result.get("generate_samples", [])))
        syn_layer_inputs = cast(dict[int, torch.Tensor], syn_result["layer_inputs"])
        syn_layer_outputs = cast(dict[int, torch.Tensor], syn_result["layer_outputs"])
        syn_exec_device_str = cast(str, syn_result["device"])
        syn_profile = cast(list[dict[str, Any]], syn_result.get("profile", []))
        requested_device_str = str(resolved_device)
        requested_cuda = requested_device_str.startswith("cuda")
        hf_fallback = bool((not skip_hf) and requested_cuda and hf_exec_device_str == "cpu")
        syn_fallback = bool(requested_cuda and syn_exec_device_str == "cpu")
        if skip_hf:
            fallback = "skip-hf"
        elif hf_fallback and syn_fallback:
            fallback = "HF+Axon->cpu"
        elif hf_fallback:
            fallback = "HF->cpu"
        elif syn_fallback:
            fallback = "Axon->cpu"
        else:
            fallback = "none"

        input_ids = input_ids_cpu
        attention_mask = attention_mask_cpu

        def _generated_token_count(generated: torch.Tensor | None) -> int:
            if generated is None:
                return 0
            if resolved_model_task == "seq2seq_lm":
                return int(generated.shape[1])
            return max(0, int(generated.shape[1] - input_ids.shape[1]))

        gen_hf = _generated_token_count(hf_gen)
        gen_syn = _generated_token_count(syn_gen)

        hf_nan_count = 0 if hf_logits is None else int(torch.isnan(hf_logits).sum().item())
        syn_nan_count = int(torch.isnan(syn_logits).sum().item())
        if hf_nan_count > 0 or syn_nan_count > 0:
            print(f"NaN logits detected | hf={hf_nan_count} syn={syn_nan_count}")

        excluded_dummy_vocab = 0
        mean_diff: float | None = None
        max_diff: float | None = None
        last_max_diff: float | None = None
        mean_rel_diff: float | None = None
        max_rel_diff: float | None = None
        top1_eq: bool | None = None
        masked_mean_diff: float | None = None
        masked_max_diff: float | None = None
        masked_last_max_diff: float | None = None
        masked_mean_rel_diff: float | None = None
        masked_max_rel_diff: float | None = None
        masked_top1_eq: bool | None = None
        if not skip_hf:
            assert hf_logits is not None
            if syn_logits.device != hf_logits.device:
                syn_logits = syn_logits.to(hf_logits.device)
            compare_hf_logits = hf_logits
            compare_syn_logits = syn_logits
            if hf_dummy_tokens_mask is not None and int(hf_dummy_tokens_mask.numel()) == int(
                hf_logits.shape[-1]
            ):
                keep_vocab = ~hf_dummy_tokens_mask
                excluded_dummy_vocab = int(hf_dummy_tokens_mask.sum().item())
                if bool(keep_vocab.any()):
                    compare_hf_logits = compare_hf_logits[..., keep_vocab]
                    compare_syn_logits = compare_syn_logits[..., keep_vocab]

            finite_mask = torch.isfinite(compare_hf_logits) & torch.isfinite(compare_syn_logits)
            diff = (compare_syn_logits.float() - compare_hf_logits.float()).abs()
            finite_diff = diff[finite_mask]
            rel_denom = torch.maximum(
                torch.maximum(compare_syn_logits.float().abs(), compare_hf_logits.float().abs()),
                torch.tensor(1.0e-12, device=diff.device, dtype=diff.dtype),
            )
            rel_diff = diff / rel_denom
            finite_rel_diff = rel_diff[finite_mask]
            if int(finite_diff.numel()) > 0:
                mean_diff = float(finite_diff.mean())
                max_diff = float(finite_diff.max())
                mean_rel_diff = float(finite_rel_diff.mean())
                max_rel_diff = float(finite_rel_diff.max())
            else:
                mean_diff = float("nan")
                max_diff = float("nan")
                mean_rel_diff = float("nan")
                max_rel_diff = float("nan")

            last_diff = diff[:, -1, :]
            last_finite = finite_mask[:, -1, :]
            finite_last_diff = last_diff[last_finite]
            last_max_diff = (
                float(finite_last_diff.max()) if int(finite_last_diff.numel()) > 0 else float("nan")
            )

            syn_last_all = compare_syn_logits[:, -1, :]
            hf_last_all = compare_hf_logits[:, -1, :]
            valid_last_vocab = torch.isfinite(syn_last_all) & torch.isfinite(hf_last_all)
            if bool(valid_last_vocab.any()):
                syn_last_for_top1 = torch.where(
                    valid_last_vocab,
                    syn_last_all,
                    torch.full_like(syn_last_all, -torch.inf),
                )
                hf_last_for_top1 = torch.where(
                    valid_last_vocab,
                    hf_last_all,
                    torch.full_like(hf_last_all, -torch.inf),
                )
                has_valid_last = valid_last_vocab.any(dim=-1)
                top1_matches = syn_last_for_top1.argmax(-1) == hf_last_for_top1.argmax(-1)
                top1_eq = (
                    bool(top1_matches[has_valid_last].all()) if bool(has_valid_last.any()) else None
                )

            metric_attention_mask = (
                decoder_attention_mask_for_metrics
                if resolved_model_task == "seq2seq_lm"
                else attention_mask
            )
            if metric_attention_mask is not None:
                if metric_attention_mask.device != diff.device:
                    metric_attention_mask = metric_attention_mask.to(diff.device)
                mask_bool = metric_attention_mask.to(torch.bool)
                valid = mask_bool.unsqueeze(-1).expand_as(diff) & finite_mask
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

                attn_bool = metric_attention_mask.to(torch.bool)
                rev_last = torch.argmax(attn_bool.flip(dims=[1]).to(torch.long), dim=1)
                lengths = (attn_bool.shape[1] - 1) - rev_last
                any_valid = attn_bool.any(dim=1)
                lengths = torch.where(lengths >= 0, lengths, torch.zeros_like(lengths))
                lengths = torch.where(any_valid, lengths, torch.zeros_like(lengths))
                b_idx = torch.arange(
                    metric_attention_mask.shape[0], device=metric_attention_mask.device
                )
                syn_last = compare_syn_logits[b_idx, lengths]
                hf_last = compare_hf_logits[b_idx, lengths]
                last_valid_vocab = torch.isfinite(syn_last) & torch.isfinite(hf_last)
                last_diff_vals = (syn_last.float() - hf_last.float()).abs()
                finite_last_vals = last_diff_vals[last_valid_vocab]
                masked_last_max_diff = (
                    float(finite_last_vals.max())
                    if int(finite_last_vals.numel()) > 0
                    else float("nan")
                )
                if bool(last_valid_vocab.any()):
                    syn_last_for_top1 = torch.where(
                        last_valid_vocab,
                        syn_last,
                        torch.full_like(syn_last, -torch.inf),
                    )
                    hf_last_for_top1 = torch.where(
                        last_valid_vocab,
                        hf_last,
                        torch.full_like(hf_last, -torch.inf),
                    )
                    valid_rows = last_valid_vocab.any(dim=-1)
                    top1_matches = syn_last_for_top1.argmax(-1) == hf_last_for_top1.argmax(-1)
                    masked_top1_eq = (
                        bool(top1_matches[valid_rows].all()) if bool(valid_rows.any()) else None
                    )

        layer_diffs: list[dict[str, float | int]] = []
        if trace_layers and (not skip_hf) and hf_layer_outputs and syn_layer_outputs:
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
        print(f"Requested device:{requested_device_str}")
        print(f"HF device:      {hf_exec_device_str}")
        print(f"Axon device:    {syn_exec_device_str}")
        print(f"Fallback:       {fallback}")
        print(f"Prompts:        {len(prompts)}")
        print(f"Model task:     {resolved_model_task}")
        print(f"Benchmark mode: {resolved_benchmark_mode}")
        print(f"Benchmark path: {'generate' if run_generate_benchmark else 'forward'}")
        print(f"HF-align bf16 profile: {bool(hf_align_bf16_profile)}")
        print(f"HF-align mask:         {align_mask_contract}")
        print(f"HF-align posid:        {align_position_ids}")
        print(f"HF-align add fp32:     {align_add_fp32}")
        print(f"HF-align linear fp32:  {align_linear_fp32}")
        print(f"HF-align norm fp32:    {align_norm_fp32}")
        print(f"Excluded dummy vocab:  {excluded_dummy_vocab}")
        print(f"Compile HF:            {bool(compile_hf)}")
        print(f"Compile Axon:          {bool(compile_axon)}")
        print(f"Compile backend:       {compile_backend}")
        print(f"Compile mode:          {compile_mode}")
        print(f"Compile fullgraph:     {bool(compile_fullgraph)}")
        print(f"Compile dynamic:       {bool(compile_dynamic)}")
        if not run_generate_benchmark:
            print(f"Forward warmup/repeat: {forward_warmup}/{forward_repeat}")
        else:
            print(f"Generate warmup/repeat:{generate_warmup}/{generate_repeat}")
        print()
        hf_time_safe = 0.0 if hf_time is None else hf_time
        if not skip_hf:
            assert hf_time is not None
        if skip_hf:
            print("HF:             skipped")
            if run_generate_benchmark:
                print(
                    f"Axon-derived:   {syn_time:.4f}s total, {gen_syn / max(syn_time, 1e-9):.2f} tok/s, generated={gen_syn}"
                )
                if generate_repeat > 1 or generate_warmup:
                    print(
                        "Axon generate samples | "
                        f"mean/min/last: {syn_time:.4f}s/"
                        f"{min(syn_generate_samples or [syn_time]):.4f}s/"
                        f"{(syn_generate_samples or [syn_time])[-1]:.4f}s "
                        f"samples={[round(item, 6) for item in (syn_generate_samples or [syn_time])]}"
                    )
            else:
                print(f"Axon forward:   {syn_time:.4f}s total")
            print("Speed ratio (Axon/HF): N/A")
        elif run_generate_benchmark:
            print(
                f"HF:             {hf_time_safe:.4f}s total, {gen_hf / max(hf_time_safe, 1e-9):.2f} tok/s, generated={gen_hf}"
            )
            print(
                f"Axon-derived:   {syn_time:.4f}s total, {gen_syn / max(syn_time, 1e-9):.2f} tok/s, generated={gen_syn}"
            )
            if generate_repeat > 1 or generate_warmup:
                if hf_generate_samples:
                    print(
                        "HF generate samples | "
                        f"mean/min/last: {hf_time_safe:.4f}s/"
                        f"{min(hf_generate_samples):.4f}s/{hf_generate_samples[-1]:.4f}s "
                        f"samples={[round(item, 6) for item in hf_generate_samples]}"
                    )
                if syn_generate_samples:
                    print(
                        "Axon generate samples | "
                        f"mean/min/last: {syn_time:.4f}s/"
                        f"{min(syn_generate_samples):.4f}s/{syn_generate_samples[-1]:.4f}s "
                        f"samples={[round(item, 6) for item in syn_generate_samples]}"
                    )
        else:
            print(f"HF forward:     {hf_time_safe:.4f}s total")
            print(f"Axon forward:   {syn_time:.4f}s total")
            print(f"Speed ratio (Axon/HF): {syn_time / max(hf_time_safe, 1e-9):.3f}x")
            if forward_repeat > 1 or forward_warmup:
                if hf_forward_samples:
                    print(
                        "HF forward samples | "
                        f"mean/min/last: {hf_time_safe:.4f}s/"
                        f"{min(hf_forward_samples):.4f}s/{hf_forward_samples[-1]:.4f}s "
                        f"samples={[round(item, 6) for item in hf_forward_samples]}"
                    )
                if syn_forward_samples:
                    print(
                        "Axon forward samples | "
                        f"mean/min/last: {syn_time:.4f}s/"
                        f"{min(syn_forward_samples):.4f}s/{syn_forward_samples[-1]:.4f}s "
                        f"samples={[round(item, 6) for item in syn_forward_samples]}"
                    )
        print()
        if (
            (not skip_hf)
            and run_generate_benchmark
            and hf_gen is not None
            and syn_gen is not None
        ):
            for idx, prompt in enumerate(prompts):
                print(f"Prompt[{idx}]: {prompt!r}")
                print("HF completion:")
                print(tokenizer_obj.decode(hf_gen[idx].tolist(), skip_special_tokens=True)[:80])
                print("Axon-derived completion:")
                print(tokenizer_obj.decode(syn_gen[idx].tolist(), skip_special_tokens=True)[:80])
                print()
        if skip_hf:
            print("Logits diff:    skipped (HF reference disabled)")
        else:
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
                print("Masked abs diff (max):", masked_max_diff)
                print("Masked top1_eq:", masked_top1_eq)
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
            "hf_forward_samples": hf_forward_samples,
            "axon_forward_samples": syn_forward_samples,
            "forward_warmup": forward_warmup,
            "forward_repeat": forward_repeat,
            "hf_generate_samples": hf_generate_samples,
            "axon_generate_samples": syn_generate_samples,
            "generate_warmup": generate_warmup,
            "generate_repeat": generate_repeat,
            "speed_ratio_axon_over_hf": (
                None if hf_time is None else syn_time / max(hf_time, 1.0e-9)
            ),
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "last_max_diff": last_max_diff,
            "mean_rel_diff": mean_rel_diff,
            "max_rel_diff": max_rel_diff,
            "top1_eq": top1_eq,
            "masked_mean_diff": masked_mean_diff,
            "masked_abs_diff": masked_max_diff,
            "masked_max_diff": masked_max_diff,
            "masked_last_max_diff": masked_last_max_diff,
            "masked_mean_rel_diff": masked_mean_rel_diff,
            "masked_max_rel_diff": masked_max_rel_diff,
            "masked_top1_eq": masked_top1_eq,
            "hf_nan_count": hf_nan_count,
            "syn_nan_count": syn_nan_count,
            "layer_diffs": layer_diffs if trace_layers else None,
            "compile_hf": bool(compile_hf),
            "compile_axon": bool(compile_axon),
            "compile_backend": compile_backend,
            "compile_mode": compile_mode,
            "compile_fullgraph": bool(compile_fullgraph),
            "compile_dynamic": bool(compile_dynamic),
            "model_task": resolved_model_task,
            "benchmark_mode": resolved_benchmark_mode,
            "benchmark_path": "generate" if run_generate_benchmark else "forward",
            "prompts": prompts,
            "generated_hf": hf_gen,
            "generated_axon": syn_gen,
            "fallback": fallback,
            "hf_device": hf_exec_device_str,
            "axon_device": syn_exec_device_str,
            "skip_hf": bool(skip_hf),
            "axon_profile": syn_profile,
        }

        return result


def run_axon_test(
    *,
    axon_file: Path,
    weights: Path,
    device: str = "cpu",
    text: str | Sequence[str] = ("The future of AI is", "Hello World"),
    max_len: int = 32,
    hf_model_dir: Path | None = None,
    tokenizer: str | None = None,
    class_name: str = "AxonGeneratedModel",
    dtype: str = "float32",
    model_task: str = "auto",
    benchmark_mode: str = "auto",
    trace_layers: bool = False,
    hf_align_bf16_profile: bool = False,
    hf_align_mask_contract: bool = False,
    hf_align_position_ids: bool = False,
    hf_align_add_fp32_accum: bool = False,
    hf_align_linear_fp32_accum: bool = False,
    hf_align_norm_fp32: bool = False,
    hf_attn_implementation: str | None = None,
    hf_experts_implementation: str | None = None,
    compile_hf: bool = False,
    compile_axon: bool = False,
    compile_backend: str | None = None,
    compile_mode: str | None = None,
    compile_fullgraph: bool = False,
    compile_dynamic: bool = False,
    trust_remote_code: bool = False,
    axon_backend: str = "codegen2-torch",
    axon_typechecker: str = "typecheck2",
    optimize_ast: bool = False,
    optimize_graph: bool = False,
    graph_backend_intrinsics: str | None = None,
    skip_hf: bool = False,
    hf_strict_dtype: bool = False,
    profile_axon: bool = False,
    profile_axon_top_n: int = 40,
    forward_warmup: int = 0,
    forward_repeat: int = 1,
    generate_warmup: int = 0,
    generate_repeat: int = 1,
) -> dict[str, Any]:
    return _run_axon_test_single(
        axon_file=axon_file,
        weights=weights,
        device=device,
        text=text,
        max_len=max_len,
        hf_model_dir=hf_model_dir,
        tokenizer=tokenizer,
        class_name=class_name,
        dtype=dtype,
        model_task=model_task,
        benchmark_mode=benchmark_mode,
        trace_layers=trace_layers,
        hf_align_bf16_profile=hf_align_bf16_profile,
        hf_align_mask_contract=hf_align_mask_contract,
        hf_align_position_ids=hf_align_position_ids,
        hf_align_add_fp32_accum=hf_align_add_fp32_accum,
        hf_align_linear_fp32_accum=hf_align_linear_fp32_accum,
        hf_align_norm_fp32=hf_align_norm_fp32,
        hf_attn_implementation=hf_attn_implementation,
        hf_experts_implementation=hf_experts_implementation,
        compile_hf=compile_hf,
        compile_axon=compile_axon,
        compile_backend=compile_backend,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        compile_dynamic=compile_dynamic,
        trust_remote_code=trust_remote_code,
        axon_backend=axon_backend,
        axon_typechecker=axon_typechecker,
        optimize_ast=optimize_ast,
        optimize_graph=optimize_graph,
        graph_backend_intrinsics=graph_backend_intrinsics,
        skip_hf=skip_hf,
        hf_strict_dtype=hf_strict_dtype,
        profile_axon=profile_axon,
        profile_axon_top_n=profile_axon_top_n,
        forward_warmup=forward_warmup,
        forward_repeat=forward_repeat,
        generate_warmup=generate_warmup,
        generate_repeat=generate_repeat,
    )


def run_axon_benchmark(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .axon_benchmark import run_axon_benchmark as _run_axon_benchmark

    return _run_axon_benchmark(*args, **kwargs)


__all__ = ["run_axon_test", "run_axon_benchmark"]
