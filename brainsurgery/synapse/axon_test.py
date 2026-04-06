from __future__ import annotations

import gc
import hashlib
import html
import importlib.machinery
import importlib.util
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any, cast

import safetensors
import torch
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
from transformers.utils import import_utils as transformers_import_utils
from transformers.utils.quantization_config import Mxfp4Config

from .axon import (
    candidate_tokenizer_dirs,
    looks_like_tokenizer_dir,
    lower_axon_program_to_synapse_spec,
    parse_axon_program_from_path,
    tokenize_prompts,
)
from .axon_runner_common import cleanup_cuda_after_oom as _cleanup_cuda_after_oom
from .axon_runner_common import is_cuda_oom as _is_cuda_oom
from .black_mamba_reference import BlackMambaReferenceModel, is_black_mamba_config_dir
from .codegen import emit_model_code_from_synapse_spec
from .matrix_models import ModelDownloadSpec, ensure_model_downloaded
from .mxfp4 import materialize_mxfp4_aliases


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
        "masked top-1 eq",
        "masked max abs diff",
        "masked max rel diff",
    ]
    body = [
        [
            str(row["axon"]),
            str(row["checkpoint"]),
            str(row["model_dir"]),
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
            masked_top1 = line[3]
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


def _task_pragma_from_axon(*, axon_file: Path) -> str | None:
    modules = parse_axon_program_from_path(axon_file)
    module = _select_main_axon_module(modules, main_module=None)
    raw = (getattr(module, "pragmas", None) or {}).get("task")
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"causal_lm", "masked_lm", "seq2seq_lm"}:
        return normalized
    raise ValueError(
        f"Unsupported TASK pragma in {axon_file}: {raw!r}"
        " (expected 'causal_lm', 'masked_lm', or 'seq2seq_lm')"
    )


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
        torch.cuda.empty_cache()
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
    if all(path.name.startswith("model-") and "-of-" in path.name for path in candidates):
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


def _checkpoint_has_lm_head_weight(model_dir: Path) -> bool:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        weight_map = payload.get("weight_map")
        if isinstance(weight_map, dict):
            return any(str(key).endswith("lm_head.weight") for key in weight_map)
        return False
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
    if model_name == "deepseek_v2_lite":
        return False
    if model_name.startswith("phi3_mini_") or model_name.startswith("phi3_medium_"):
        return False
    if model_name.startswith("phi3_small_"):
        return True
    has_local_artifacts = _has_local_custom_code_artifacts(model_dir)
    if not isinstance(model_config, dict):
        return has_local_artifacts
    auto_map = model_config.get("auto_map")
    module_names = _iter_auto_map_module_names(auto_map)
    if module_names:
        return all((model_dir / f"{module_name}.py").exists() for module_name in module_names)
    return has_local_artifacts


def _normalize_rope_numeric_fields(config: Any) -> Any:
    def _normalize_dict(mapping: Any) -> None:
        if not isinstance(mapping, dict):
            return
        rope_type = mapping.get("rope_type")
        if isinstance(rope_type, str) and "type" not in mapping:
            mapping["type"] = rope_type
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
        tmp_config = Path(tmp_dir) / "config.json"
        tmp_config.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
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


def _refresh_hf_rotary_caches_if_needed(model: Any, *, dtype: torch.dtype) -> int:
    refreshed = 0
    for module in model.modules():
        set_cache = getattr(module, "_set_cos_sin_cache", None)
        if not callable(set_cache):
            continue
        max_seq_len = getattr(module, "max_seq_len", None)
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
        if not needs_refresh:
            continue
        inv_freq = getattr(module, "inv_freq", None)
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
    }
    return aliases.get(model_dir.name)


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


def _select_main_axon_module(modules: Sequence[Any], *, main_module: str | None) -> Any:
    if not modules:
        raise ValueError("Axon program contains no modules")
    if main_module is None:
        return modules[-1]
    for module in modules:
        if getattr(module, "name", None) == main_module:
            return module
    raise ValueError(f"Main Axon module not found: {main_module}")


def _declared_checkpoints_from_axon(
    *,
    axon_file: Path,
    main_module: str | None,
) -> tuple[str, ...]:
    modules = parse_axon_program_from_path(axon_file)
    module = _select_main_axon_module(modules, main_module=main_module)
    raw = (getattr(module, "pragmas", None) or {}).get("checkpoints")
    if raw is None:
        raise ValueError(
            f"No CHECKPOINTS pragma declared in {axon_file}"
            + ("" if main_module is None else f" for main module {main_module!r}")
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
        spec=ModelDownloadSpec(local_dir=checkpoint_id, repo_id=checkpoint_id),
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
    main_module: str | None = None,
    dtype: str = "float32",
    model_task: str = "auto",
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
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype)
    resolved_model_task = _resolve_model_task(model_task)
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
    tokenizer_source = tokenizer or str(resolved_hf_model_dir)
    if tokenizer is None:
        for candidate in candidate_tokenizer_dirs(resolved_hf_model_dir):
            if looks_like_tokenizer_dir(candidate):
                tokenizer_source = str(candidate)
                break
    tokenizer_fallback = (
        _tokenizer_fallback_repo_id(resolved_hf_model_dir) or resolved_hf_model_dir.name
        if tokenizer is None
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

        modules = parse_axon_program_from_path(axon_file)
        synapse_spec = lower_axon_program_to_synapse_spec(modules, main_module=main_module)
        if model_config is not None:
            model_section = synapse_spec.get("model")
            if not isinstance(model_section, dict):
                raise ValueError("Lowered synapse spec has invalid model section")
            model_section["config"] = model_config

        synapse_yaml_path.write_text(
            OmegaConf.to_yaml(synapse_spec, resolve=True), encoding="utf-8"
        )
        loaded = OmegaConf.load(synapse_yaml_path)
        loaded_dict = OmegaConf.to_container(loaded, resolve=True)
        if not isinstance(loaded_dict, dict):
            raise ValueError("Lowered synapse YAML did not produce a mapping")
        lowered_spec: dict[str, Any] = {str(key): value for key, value in loaded_dict.items()}
        if model_config is not None:
            lowered_model = lowered_spec.get("model")
            if isinstance(lowered_model, dict):
                lowered_model["config"] = model_config

        code = emit_model_code_from_synapse_spec(lowered_spec, class_name=class_name)
        generated_py_path.write_text(code, encoding="utf-8")

        model_cls = _load_generated_class(generated_py_path, class_name)

        hf_config: Any | None = None
        non_mxfp4_quant_config: Any | None = None
        if resolved_model_task in {"masked_lm", "seq2seq_lm"} or resolved_model_type in {
            "phi3",
            "phi3small",
        }:
            hf_config = _load_auto_config_with_compat_fallback(
                resolved_hf_model_dir,
                trust_remote_code=effective_trust_remote_code,
            )
            hf_config = _normalize_rope_numeric_fields(hf_config)
            non_mxfp4_quant_config = _build_non_mxfp4_quantization_config(hf_config)
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
            if resolved_model_task == "causal_lm" and attention_mask is not None:
                pos_ids = attention_mask.to(torch.long).cumsum(dim=-1) - 1
                pos_ids = pos_ids.masked_fill(attention_mask == 0, 1)
                hf_forward_inputs["position_ids"] = pos_ids
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "hf_inputs": hf_inputs,
                "hf_forward_inputs": hf_forward_inputs,
                "syn_inputs": syn_inputs,
            }

        def _run_hf_side(target_device_str: str) -> dict[str, Any]:
            target_device = _resolve_device(target_device_str)
            local_model_type = resolved_model_type
            local_state_ref_cpu: dict[str, torch.Tensor] | None = None
            io = _build_io_for_device(target_device)
            hf: Any
            hf_model: Any | None = None
            if resolved_model_task == "masked_lm":
                if non_mxfp4_quant_config is not None:
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
                hf_model = AutoModelForSeq2SeqLM.from_pretrained(
                    str(resolved_hf_model_dir),
                    local_files_only=True,
                    dtype=resolved_dtype,
                    config=hf_config,
                    quantization_config=non_mxfp4_quant_config,
                    trust_remote_code=effective_trust_remote_code,
                )
                hf = hf_model.to(device=target_device, dtype=resolved_dtype).eval()
            else:
                try:
                    _ensure_transformers_import_compat()
                    causal_kwargs: dict[str, Any] = {
                        "local_files_only": True,
                        "dtype": resolved_dtype,
                        "config": hf_config,
                        "trust_remote_code": effective_trust_remote_code,
                    }
                    if non_mxfp4_quant_config is not None:
                        causal_kwargs["quantization_config"] = non_mxfp4_quant_config
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        str(resolved_hf_model_dir),
                        **causal_kwargs,
                    )
                    hf = cast(Any, hf_model).to(device=target_device, dtype=resolved_dtype).eval()
                except Exception:
                    if (
                        hf_config is not None
                        and str(getattr(hf_config, "model_type", "")).strip().lower() == "gemma4"
                    ):
                        _ensure_transformers_import_compat()
                        hf_model = AutoModelForImageTextToText.from_pretrained(
                            str(resolved_hf_model_dir),
                            local_files_only=True,
                            dtype=resolved_dtype,
                            config=hf_config,
                            trust_remote_code=effective_trust_remote_code,
                        )
                        hf = (
                            cast(Any, hf_model)
                            .to(device=target_device, dtype=resolved_dtype)
                            .eval()
                        )
                    elif is_black_mamba_config_dir(resolved_hf_model_dir):
                        generated_state = _load_state_dict(
                            safetensors_files,
                            device=target_device,
                            dtype=resolved_dtype,
                            model_config=model_config,
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
                and not _checkpoint_has_lm_head_weight(resolved_hf_model_dir)
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
                        "HF: tied output embeddings to input embeddings (checkpoint has no lm_head.weight)"
                    )
            if _rebuild_hf_dummy_tokens_mask_from_config(hf):
                print("HF: rebuilt dummy_tokens_mask from config")
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
                hf_forward_kwargs = dict(io["hf_forward_inputs"])
                if resolved_model_task in {"causal_lm", "seq2seq_lm"}:
                    hf_forward_kwargs["use_cache"] = False
                if local_model_type == "phi3small":
                    return _run_phi3small_chunked_logits(
                        model,
                        hf_forward_kwargs=hf_forward_kwargs,
                    )
                return _extract_logits(model(**hf_forward_kwargs))

            hf_gen: torch.Tensor | None = None
            hf_time = 0.0
            if resolved_model_task in {"masked_lm", "seq2seq_lm"}:
                hf_t0 = time.perf_counter()
                with timing(message="HF"), torch.no_grad():
                    hf_logits = _run_hf_forward(hf)
                hf_time = time.perf_counter() - hf_t0
            else:
                with torch.no_grad():
                    hf_logits = _run_hf_forward(hf)

                def _run_hf_generate(model: Any) -> torch.Tensor:
                    return model.generate(
                        **io["hf_inputs"],
                        max_new_tokens=max(1, max_len - int(io["input_ids"].shape[1])),
                        eos_token_id=tokenizer_obj.eos_token_id,
                        pad_token_id=tokenizer_obj.eos_token_id,
                    )

                try:
                    hf_gen, hf_time = _time_generate("HF", lambda model=hf: _run_hf_generate(model))
                except Exception as exc:
                    print(
                        "HF generate failed; falling back to prompt-only decode:",
                        f"{type(exc).__name__}: {exc}",
                    )
                    hf_gen = io["input_ids"]
                    hf_time = 0.0
            hf_logits_cpu = hf_logits.detach().cpu()
            hf_gen_cpu = None if hf_gen is None else hf_gen.detach().cpu()
            dummy_mask = _build_phi3small_dummy_vocab_mask(
                model_config=model_config,
                vocab_size=int(hf_logits_cpu.shape[-1]),
                device=torch.device("cpu"),
            )
            if local_model_type == "phi3small":
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
                "dummy_mask": dummy_mask,
                "decoder_attention_mask": io.get("decoder_attention_mask"),
                "layer_inputs": hf_layer_inputs,
                "layer_outputs": hf_layer_outputs,
                "state_ref_cpu": local_state_ref_cpu,
                "device": str(target_device),
            }

        hf_result: dict[str, Any]
        try:
            hf_result = _run_hf_side(exec_device_str)
        except Exception as exc:
            if not _is_cuda_oom(exc, device=exec_device_str):
                raise
            _cleanup_cuda_after_oom(exec_device_str)
            print(f"CUDA OOM on {exec_device_str}; retrying HF on cpu")
            hf_result = _run_hf_side("cpu")

        hf_logits = cast(torch.Tensor, hf_result["logits"])
        hf_gen = cast(torch.Tensor | None, hf_result["gen"])
        hf_time = float(hf_result["time"])
        hf_dummy_tokens_mask = cast(torch.Tensor | None, hf_result["dummy_mask"])
        hf_layer_inputs = cast(dict[int, torch.Tensor], hf_result["layer_inputs"])
        hf_layer_outputs = cast(dict[int, torch.Tensor], hf_result["layer_outputs"])
        hf_exec_device_str = cast(str, hf_result["device"])
        state_ref_cpu = cast(dict[str, torch.Tensor] | None, hf_result["state_ref_cpu"])

        def _run_syn_side(target_device_str: str) -> dict[str, Any]:
            target_device = _resolve_device(target_device_str)
            io = _build_io_for_device(target_device)
            local_state_dict = state_ref_cpu
            if local_state_dict is None:
                local_state_dict = _load_state_dict(
                    safetensors_files,
                    device=target_device,
                    dtype=resolved_dtype,
                    model_config=model_config,
                )
            elif resolved_model_type == "phi3small":
                local_state_dict = {
                    key: value.to(device=target_device, dtype=resolved_dtype)
                    for key, value in local_state_dict.items()
                }
            syn = model_cls.from_state_dict(local_state_dict).to(target_device).eval()
            if local_state_dict is not state_ref_cpu:
                local_state_dict.clear()
            del local_state_dict
            _cleanup(target_device)
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
                return _extract_logits(model(**io["syn_inputs"]))

            syn_gen: torch.Tensor | None = None
            syn_time = 0.0
            if resolved_model_task in {"masked_lm", "seq2seq_lm"}:
                syn_t0 = time.perf_counter()
                with timing(message="AxonDerived"), torch.no_grad():
                    syn_logits = _run_syn_forward()
                syn_time = time.perf_counter() - syn_t0
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

                try:
                    syn_gen, syn_time = _time_generate("AxonDerived", _run_syn_generate)
                except Exception as exc:
                    print(
                        "Axon generate failed; falling back to prompt-only decode:",
                        f"{type(exc).__name__}: {exc}",
                    )
                    syn_gen = io["input_ids"]
                    syn_time = 0.0
            if (
                trace_layers
                and callable(original_block_call)
                and isinstance(original_block_name, str)
            ):
                setattr(syn, original_block_name, original_block_call)
            syn_logits_cpu = syn_logits.detach().cpu()
            syn_gen_cpu = None if syn_gen is None else syn_gen.detach().cpu()
            del syn
            _cleanup(target_device)
            return {
                "logits": syn_logits_cpu,
                "gen": syn_gen_cpu,
                "time": syn_time,
                "layer_inputs": syn_layer_inputs,
                "layer_outputs": syn_layer_outputs,
                "device": str(target_device),
            }

        syn_result: dict[str, Any]
        try:
            syn_result = _run_syn_side(exec_device_str)
        except Exception as exc:
            if not _is_cuda_oom(exc, device=exec_device_str):
                raise
            _cleanup_cuda_after_oom(exec_device_str)
            print(f"CUDA OOM on {exec_device_str}; retrying AxonDerived on cpu")
            syn_result = _run_syn_side("cpu")

        syn_logits = cast(torch.Tensor, syn_result["logits"])
        syn_gen = cast(torch.Tensor | None, syn_result["gen"])
        syn_time = float(syn_result["time"])
        syn_layer_inputs = cast(dict[int, torch.Tensor], syn_result["layer_inputs"])
        syn_layer_outputs = cast(dict[int, torch.Tensor], syn_result["layer_outputs"])
        syn_exec_device_str = cast(str, syn_result["device"])

        input_ids = input_ids_cpu
        attention_mask = attention_mask_cpu
        gen_hf = 0 if hf_gen is None else int(hf_gen.shape[1] - input_ids.shape[1])
        gen_syn = 0 if syn_gen is None else int(syn_gen.shape[1] - input_ids.shape[1])

        hf_nan_count = int(torch.isnan(hf_logits).sum().item())
        syn_nan_count = int(torch.isnan(syn_logits).sum().item())
        if hf_nan_count > 0 or syn_nan_count > 0:
            print(f"NaN logits detected | hf={hf_nan_count} syn={syn_nan_count}")

        if syn_logits.device != hf_logits.device:
            syn_logits = syn_logits.to(hf_logits.device)
        compare_hf_logits = hf_logits
        compare_syn_logits = syn_logits
        excluded_dummy_vocab = 0
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
        else:
            top1_eq = None

        masked_mean_diff: float | None = None
        masked_max_diff: float | None = None
        masked_last_max_diff: float | None = None
        masked_mean_rel_diff: float | None = None
        masked_max_rel_diff: float | None = None
        masked_top1_eq: bool | None = None
        decoder_attention_mask = cast(torch.Tensor | None, hf_result.get("decoder_attention_mask"))
        metric_attention_mask = (
            decoder_attention_mask if resolved_model_task == "seq2seq_lm" else attention_mask
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
                float(finite_last_vals.max()) if int(finite_last_vals.numel()) > 0 else float("nan")
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
            else:
                masked_top1_eq = None

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
        print(f"Requested device:{resolved_device}")
        print(f"HF device:      {hf_exec_device_str}")
        print(f"Axon device:    {syn_exec_device_str}")
        print(f"Prompts:        {len(prompts)}")
        print(f"Model task:     {resolved_model_task}")
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
        print()
        if resolved_model_task == "causal_lm":
            print(
                f"HF:             {hf_time:.4f}s total, {gen_hf / max(hf_time, 1e-9):.2f} tok/s, generated={gen_hf}"
            )
            print(
                f"Axon-derived:   {syn_time:.4f}s total, {gen_syn / max(syn_time, 1e-9):.2f} tok/s, generated={gen_syn}"
            )
        else:
            print(f"HF forward:     {hf_time:.4f}s total")
            print(f"Axon forward:   {syn_time:.4f}s total")
        print(f"Speed ratio (Axon/HF): {syn_time / max(hf_time, 1e-9):.3f}x")
        print()
        if resolved_model_task == "causal_lm" and hf_gen is not None and syn_gen is not None:
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
            "speed_ratio_axon_over_hf": syn_time / max(hf_time, 1.0e-9),
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
            "prompts": prompts,
            "generated_hf": hf_gen,
            "generated_axon": syn_gen,
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
    main_module: str | None = None,
    dtype: str = "float32",
    model_task: str = "auto",
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
    trust_remote_code: bool = False,
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
        main_module=main_module,
        dtype=dtype,
        model_task=model_task,
        trace_layers=trace_layers,
        hf_align_bf16_profile=hf_align_bf16_profile,
        hf_align_mask_contract=hf_align_mask_contract,
        hf_align_position_ids=hf_align_position_ids,
        hf_align_add_fp32_accum=hf_align_add_fp32_accum,
        hf_align_linear_fp32_accum=hf_align_linear_fp32_accum,
        hf_align_norm_fp32=hf_align_norm_fp32,
        compile_hf=compile_hf,
        compile_axon=compile_axon,
        compile_backend=compile_backend,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        compile_dynamic=compile_dynamic,
        trust_remote_code=trust_remote_code,
    )


def run_axon_benchmark(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .axon_benchmark import run_axon_benchmark as _run_axon_benchmark

    return _run_axon_benchmark(*args, **kwargs)


__all__ = ["run_axon_test", "run_axon_benchmark"]
