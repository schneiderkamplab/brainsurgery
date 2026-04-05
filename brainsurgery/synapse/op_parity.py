from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from .axon import (
    candidate_tokenizer_dirs,
    looks_like_tokenizer_dir,
    lower_axon_program_to_synapse_spec,
    parse_axon_program_from_path,
    tokenize_prompts,
)
from .axon_test import (
    _augment_model_config_from_checkpoint,
    _ensure_transformers_import_compat,
    _extract_logits,
    _load_auto_config_with_compat_fallback,
    _load_generated_class,
    _load_hf_masked_lm_reference,
    _load_model_config,
    _load_state_dict,
    _normalize_rope_numeric_fields,
    _resolve_device,
    _resolve_safetensors_paths,
    _should_trust_remote_code,
)
from .codegen import emit_model_code_from_synapse_spec
from .runtime import SynapseProgramModel

_CANONICAL_DTYPES = ("float32", "bfloat16", "float16")
_MASKED_LM_MODEL_TYPES = {
    "albert",
    "bert",
    "camembert",
    "deberta-v2",
    "deberta",
    "distilbert",
    "electra",
    "longformer",
    "modernbert",
    "roberta",
    "xlm-roberta",
}


def _resolve_dtype_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _normalize_dtypes(dtypes: Sequence[str] | None) -> list[str]:
    if dtypes is None:
        return list(_CANONICAL_DTYPES)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in dtypes:
        key = str(item).strip().lower()
        if key not in _CANONICAL_DTYPES:
            raise ValueError(f"Unsupported dtype in sweep: {item!r}")
        if key not in seen:
            seen.add(key)
            normalized.append(key)
    if not normalized:
        raise ValueError("At least one dtype is required")
    return normalized


def _normalize_texts(text: str | Sequence[str]) -> list[str]:
    if isinstance(text, str):
        return [text]
    out = [str(item) for item in text]
    return out if out else ["The future of AI is", "Hello world"]


def _resolve_hf_loader_kind(hf_config: Any) -> str:
    if bool(getattr(hf_config, "is_encoder_decoder", False)):
        return "seq2seq_lm"
    model_type = str(getattr(hf_config, "model_type", "")).strip().lower()
    if model_type in _MASKED_LM_MODEL_TYPES:
        return "masked_lm"
    return "causal_lm"


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, dict):
        for key in ("logits", "hidden_states", "last_hidden_state"):
            if key in value:
                tensor = _first_tensor(value[key])
                if tensor is not None:
                    return tensor
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    return None


def _hf_kind(module: nn.Module) -> str | None:
    if isinstance(module, nn.Linear):
        return "linear"
    if isinstance(module, nn.LayerNorm):
        return "layernorm"
    if isinstance(module, nn.Embedding):
        return "embedding"
    name = module.__class__.__name__
    if name == "Conv1D":
        return "linear"
    if "RMSNorm" in name:
        return "rmsnorm"
    if "Attention" in name or name.endswith("SdpaAttention"):
        return "attention"
    return None


def _synapse_kind(op_name: str) -> str | None:
    if op_name in {"linear", "layernorm", "rmsnorm", "add", "embedding"}:
        return op_name
    return None


def _is_attention_output_projection(weight_path: str) -> bool:
    normalized = weight_path.replace("/", ".")
    suffixes = (
        ".self_attn.o_proj.weight",
        ".attn.c_proj.weight",
        ".attention.wo.weight",
        ".self_attn.out_proj.weight",
    )
    return any(normalized.endswith(suffix) for suffix in suffixes)


def _canonical_path(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.replace("/", ".").strip(".")
    return normalized or None


def _hf_weight_path(module_name: str, module: nn.Module) -> str | None:
    normalized = _canonical_path(module_name)
    if normalized is None:
        return None
    if hasattr(module, "weight"):
        return f"{normalized}.weight"
    return None


def _meta_pair_key(meta: dict[str, Any] | None) -> str | None:
    if not isinstance(meta, dict):
        return None
    for key in ("weight_path", "module_name", "node_path"):
        normalized = _canonical_path(meta.get(key))
        if normalized is not None:
            return normalized
    return None


class _TracingSynapseProgramModel(SynapseProgramModel):
    def __init__(
        self,
        spec: dict[str, Any],
        state_dict: dict[str, torch.Tensor],
        *,
        trace_kinds: set[str],
    ) -> None:
        super().__init__(spec=spec, state_dict=state_dict)
        self._trace_kinds = trace_kinds
        self.trace: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.trace_meta: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def _append_trace(
        self,
        kind: str,
        tensor: torch.Tensor,
        *,
        meta: dict[str, Any],
    ) -> None:
        self.trace[kind].append(tensor.detach().float().cpu())
        self.trace_meta[kind].append(meta)

    def _execute_op(
        self,
        op: str,
        node_spec: dict[str, Any],
        env: dict[str, Any],
        *,
        node_path: str,
        scope: str,
        symbols: dict[str, int | float | bool],
    ) -> None:
        super()._execute_op(op, node_spec, env, node_path=node_path, scope=scope, symbols=symbols)
        kind = _synapse_kind(op)
        if kind is None or kind not in self._trace_kinds:
            kind = None
        bind = node_spec.get("_bind")
        bind_names = [bind] if isinstance(bind, str) else (bind if isinstance(bind, list) else [])
        weight_path: str | None = None
        if op == "linear":
            try:
                maybe_path = self._infer_param_path(
                    node_spec, node_path=node_path, param_name="weight"
                )
                if isinstance(maybe_path, str):
                    weight_path = maybe_path
            except Exception:
                weight_path = None
        if kind is not None:
            for name in bind_names:
                if not isinstance(name, str):
                    continue
                value = env.get(name)
                tensor = _first_tensor(value)
                if tensor is None:
                    continue
                meta = {"node_path": node_path, "op": op, "bind": name}
                if weight_path is not None:
                    meta["weight_path"] = weight_path
                self._append_trace(kind, tensor, meta=meta)
        if op == "linear" and "attention" in self._trace_kinds:
            if isinstance(weight_path, str) and _is_attention_output_projection(weight_path):
                for name in bind_names:
                    if not isinstance(name, str):
                        continue
                    value = env.get(name)
                    tensor = _first_tensor(value)
                    if tensor is None:
                        continue
                    self._append_trace(
                        "attention",
                        tensor,
                        meta={
                            "node_path": node_path,
                            "op": op,
                            "bind": name,
                            "weight_path": weight_path,
                            "attention_via": "linear_out_proj",
                        },
                    )


class _NodeTracingSynapseProgramModel(SynapseProgramModel):
    def __init__(self, spec: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> None:
        super().__init__(spec=spec, state_dict=state_dict)
        self.trace_ops: list[dict[str, Any]] = []

    def _execute_op(
        self,
        op: str,
        node_spec: dict[str, Any],
        env: dict[str, Any],
        *,
        node_path: str,
        scope: str,
        symbols: dict[str, int | float | bool],
    ) -> None:
        super()._execute_op(op, node_spec, env, node_path=node_path, scope=scope, symbols=symbols)
        bind = node_spec.get("_bind")
        bind_names = [bind] if isinstance(bind, str) else (bind if isinstance(bind, list) else [])
        for name in bind_names:
            if not isinstance(name, str):
                continue
            value = env.get(name)
            tensor = _first_tensor(value)
            if tensor is None:
                continue
            self.trace_ops.append(
                {
                    "node_path": node_path,
                    "op": op,
                    "bind": name,
                    "dtype": str(tensor.dtype),
                    "tensor": tensor.detach().float().cpu(),
                }
            )


class _AddTraceMode(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.outputs: list[torch.Tensor] = []
        self.meta: list[dict[str, Any]] = []

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        call_kwargs = kwargs or {}
        out = func(*args, **call_kwargs)
        try:
            packet = getattr(func, "overloadpacket", None)
            if packet is torch.ops.aten.add:
                if len(args) >= 2 and torch.is_tensor(args[0]) and torch.is_tensor(args[1]):
                    if args[0].is_floating_point() and args[1].is_floating_point():
                        tensor = _first_tensor(out)
                        if tensor is not None and tensor.is_floating_point():
                            self.outputs.append(tensor.detach().float().cpu())
                            self.meta.append(
                                {
                                    "func": str(func),
                                    "lhs_shape": list(args[0].shape),
                                    "rhs_shape": list(args[1].shape),
                                }
                            )
        except Exception:
            pass
        return out


def _capture_hf_trace(
    model: nn.Module, trace_kinds: set[str]
) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[dict[str, Any]]], list[Any]]:
    trace: dict[str, list[torch.Tensor]] = defaultdict(list)
    trace_meta: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hooks: list[Any] = []
    for module_name, module in model.named_modules():
        kind = _hf_kind(module)
        if kind is None or kind not in trace_kinds:
            continue
        kind_name = kind

        def _hook(
            _module: nn.Module,
            _inp: Any,
            out: Any,
            *,
            _kind: str = kind_name,
            _module_name: str = module_name,
        ) -> None:
            tensor = _first_tensor(out)
            if tensor is None:
                return
            trace[_kind].append(tensor.detach().float().cpu())
            trace_meta[_kind].append(
                {
                    "module_name": _module_name,
                    "module_type": _module.__class__.__name__,
                    "weight_path": _hf_weight_path(_module_name, _module),
                }
            )

        hooks.append(module.register_forward_hook(_hook))
    return trace, trace_meta, hooks


def _shape_key(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(x) for x in tensor.shape)


def _ordered_float16_like_bits(x: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    q = x.to(dtype=dtype)
    bits = q.view(torch.int16).to(torch.int32)
    sign = (bits >> 15) & 1
    return torch.where(sign.bool(), (~bits) & 0xFFFF, bits | 0x8000)


def _ulp_distance(a: torch.Tensor, b: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    oa = _ordered_float16_like_bits(a, dtype=dtype)
    ob = _ordered_float16_like_bits(b, dtype=dtype)
    return (oa - ob).abs()


def _pair_stats(
    hf_tensors: list[torch.Tensor],
    syn_tensors: list[torch.Tensor],
    *,
    hf_meta: list[dict[str, Any]] | None = None,
    syn_meta: list[dict[str, Any]] | None = None,
    dtype_name: str | None = None,
) -> dict[str, Any]:
    matched = 0
    skipped_shape = 0
    numel = 0
    sum_abs = 0.0
    max_abs = 0.0
    sum_rel = 0.0
    max_rel = 0.0
    ulp_values: list[torch.Tensor] = []
    offenders: list[dict[str, Any]] = []
    hf_meta = hf_meta or []
    syn_meta = syn_meta or []
    hf_remaining: list[tuple[int, torch.Tensor]] = list(enumerate(hf_tensors))
    syn_remaining: list[tuple[int, torch.Tensor]] = list(enumerate(syn_tensors))
    hf_by_key: dict[str, list[tuple[int, torch.Tensor]]] = defaultdict(list)
    syn_by_key: dict[str, list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for idx, tensor in hf_remaining:
        key = _meta_pair_key(hf_meta[idx] if idx < len(hf_meta) else None)
        if key is not None:
            hf_by_key[key].append((idx, tensor))
    for idx, tensor in syn_remaining:
        key = _meta_pair_key(syn_meta[idx] if idx < len(syn_meta) else None)
        if key is not None:
            syn_by_key[key].append((idx, tensor))

    pairings: list[tuple[int, torch.Tensor, int, torch.Tensor]] = []
    paired_hf: set[int] = set()
    paired_syn: set[int] = set()

    all_keys = set(hf_by_key.keys()) | set(syn_by_key.keys())
    for key in sorted(all_keys):
        hf_list = hf_by_key.get(key, [])
        syn_list = syn_by_key.get(key, [])
        local_pairs = min(len(hf_list), len(syn_list))
        for pair_idx in range(local_pairs):
            hf_idx, hf_t = hf_list[pair_idx]
            syn_idx, syn_t = syn_list[pair_idx]
            pairings.append((hf_idx, hf_t, syn_idx, syn_t))
            paired_hf.add(hf_idx)
            paired_syn.add(syn_idx)

    hf_by_shape: dict[tuple[int, ...], list[tuple[int, torch.Tensor]]] = defaultdict(list)
    syn_by_shape: dict[tuple[int, ...], list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for idx, tensor in hf_remaining:
        if idx not in paired_hf:
            hf_by_shape[_shape_key(tensor)].append((idx, tensor))
    for idx, tensor in syn_remaining:
        if idx not in paired_syn:
            syn_by_shape[_shape_key(tensor)].append((idx, tensor))
    all_shapes = set(hf_by_shape.keys()) | set(syn_by_shape.keys())
    for shape in sorted(all_shapes):
        hf_list = hf_by_shape.get(shape, [])
        syn_list = syn_by_shape.get(shape, [])
        local_pairs = min(len(hf_list), len(syn_list))
        skipped_shape += abs(len(hf_list) - len(syn_list))
        for pair_idx in range(local_pairs):
            hf_idx, hf_t = hf_list[pair_idx]
            syn_idx, syn_t = syn_list[pair_idx]
            pairings.append((hf_idx, hf_t, syn_idx, syn_t))

    pairings.sort(key=lambda item: (item[0], item[2]))
    for pair_idx, (hf_idx, hf_t, syn_idx, syn_t) in enumerate(pairings):
        diff = (syn_t - hf_t).abs()
        local_mean = float(diff.mean()) if diff.numel() else 0.0
        local_max = float(diff.max()) if diff.numel() else 0.0
        offenders.append(
            {
                "index": pair_idx,
                "hf_index": hf_idx,
                "syn_index": syn_idx,
                "shape": list(_shape_key(hf_t)),
                "mean_abs": local_mean,
                "max_abs": local_max,
                "hf_meta": hf_meta[hf_idx] if hf_idx < len(hf_meta) else None,
                "syn_meta": syn_meta[syn_idx] if syn_idx < len(syn_meta) else None,
            }
        )
        matched += 1
        n = int(diff.numel())
        numel += n
        sum_abs += float(diff.sum())
        denom = torch.maximum(torch.maximum(hf_t.abs(), syn_t.abs()), torch.tensor(1.0e-12))
        rel = diff / denom
        sum_rel += float(rel.sum())
        local_rel_max = float(rel.max()) if rel.numel() else 0.0
        if local_rel_max > max_rel:
            max_rel = local_rel_max
        if local_max > max_abs:
            max_abs = local_max
        if dtype_name == "bfloat16":
            ulp_values.append(_ulp_distance(hf_t, syn_t, dtype=torch.bfloat16).reshape(-1))
        elif dtype_name == "float16":
            ulp_values.append(_ulp_distance(hf_t, syn_t, dtype=torch.float16).reshape(-1))
    offenders.sort(key=lambda item: item["mean_abs"], reverse=True)
    shape_hf_counts = {str(list(shape)): len(items) for shape, items in hf_by_shape.items()}
    shape_syn_counts = {str(list(shape)): len(items) for shape, items in syn_by_shape.items()}
    mean_ulp: float | None = None
    max_ulp: float | None = None
    p95_ulp: float | None = None
    p99_ulp: float | None = None
    if ulp_values:
        all_ulp = torch.cat(ulp_values).to(torch.float32)
        mean_ulp = float(all_ulp.mean())
        max_ulp = float(all_ulp.max())
        p95_ulp = float(torch.quantile(all_ulp, 0.95))
        p99_ulp = float(torch.quantile(all_ulp, 0.99))
    return {
        "hf_count": len(hf_tensors),
        "syn_count": len(syn_tensors),
        "paired": len(pairings),
        "matched": matched,
        "skipped_shape": skipped_shape,
        "mean_abs": (sum_abs / numel) if numel > 0 else None,
        "max_abs": max_abs if matched > 0 else None,
        "mean_rel": (sum_rel / numel) if numel > 0 else None,
        "max_rel": max_rel if matched > 0 else None,
        "mean_ulp": mean_ulp,
        "max_ulp": max_ulp,
        "p95_ulp": p95_ulp,
        "p99_ulp": p99_ulp,
        "top_offenders": offenders[:10],
        "shape_hf_counts": shape_hf_counts,
        "shape_syn_counts": shape_syn_counts,
    }


def _tensor_diff_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float]:
    lhs_f = lhs.float()
    rhs_f = rhs.float()
    diff = (lhs_f - rhs_f).abs()
    if diff.numel() == 0:
        return {
            "mean_abs": 0.0,
            "max_abs": 0.0,
            "mean_rel": 0.0,
            "max_rel": 0.0,
        }
    denom = torch.maximum(
        torch.maximum(lhs_f.abs(), rhs_f.abs()),
        torch.tensor(1.0e-12, device=diff.device, dtype=diff.dtype),
    )
    rel = diff / denom
    return {
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "mean_rel": float(rel.mean()),
        "max_rel": float(rel.max()),
    }


def _build_inputs(
    *,
    lowered_spec: dict[str, Any],
    prompts: list[str],
    tokenizer_source: str,
    tokenizer_fallback: str | None,
    device: torch.device,
) -> tuple[Any, torch.Tensor, torch.Tensor | None]:
    return tokenize_prompts(
        prompts=prompts,
        tokenizer_source=tokenizer_source,
        tokenizer_fallback=tokenizer_fallback,
        device=device,
        lowered_spec=lowered_spec,
    )


def _forward_kwargs(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    for_hf: bool,
    model_input_names: set[str],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"input_ids": input_ids}
    if attention_mask is None:
        return kwargs
    if for_hf:
        kwargs["attention_mask"] = attention_mask
        pos_ids = attention_mask.to(torch.long).cumsum(dim=-1) - 1
        pos_ids = pos_ids.masked_fill(attention_mask == 0, 1)
        kwargs["position_ids"] = pos_ids
        return kwargs
    if "attn_mask" in model_input_names:
        kwargs["attn_mask"] = attention_mask
    if "attention_mask" in model_input_names:
        kwargs["attention_mask"] = attention_mask
    return kwargs


def _run_single_dtype(
    *,
    dtype_name: str,
    dtype: torch.dtype,
    resolved_hf_model_dir: Path,
    lowered_spec: dict[str, Any],
    state_dict_paths: list[Path],
    tokenizer_source: str,
    tokenizer_fallback: str | None,
    prompts: list[str],
    device: torch.device,
    hf_config: Any,
    hf_loader_kind: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    trace_kinds = {"linear", "layernorm", "rmsnorm", "attention", "embedding", "add"}
    model_input_names = set(lowered_spec.get("model", {}).get("inputs", {}).keys())
    tokenizer_obj, input_ids, attention_mask = _build_inputs(
        lowered_spec=lowered_spec,
        prompts=prompts,
        tokenizer_source=tokenizer_source,
        tokenizer_fallback=tokenizer_fallback,
        device=device,
    )
    del tokenizer_obj
    hf_kwargs = _forward_kwargs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        for_hf=True,
        model_input_names=model_input_names,
    )
    syn_kwargs = _forward_kwargs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        for_hf=False,
        model_input_names=model_input_names,
    )

    hf_trace: dict[str, list[torch.Tensor]] = defaultdict(list)
    hf_trace_meta: dict[str, list[dict[str, Any]]] = defaultdict(list)
    syn_trace: dict[str, list[torch.Tensor]] = defaultdict(list)
    syn_trace_meta: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hf_error: str | None = None
    syn_error: str | None = None
    logits_summary: dict[str, Any] | None = None
    try:
        _ensure_transformers_import_compat()
        if hf_loader_kind == "masked_lm":
            hf_model = _load_hf_masked_lm_reference(
                model_dir=resolved_hf_model_dir,
                safetensors_files=state_dict_paths,
                resolved_dtype=dtype,
                resolved_device=device,
                hf_config=hf_config,
                trust_remote_code=trust_remote_code,
                model_config=lowered_spec.get("model", {}).get("config")
            )
        elif hf_loader_kind == "seq2seq_lm":
            hf_model = AutoModelForSeq2SeqLM.from_pretrained(
                str(resolved_hf_model_dir),
                local_files_only=True,
                dtype=dtype,
                config=hf_config,
                trust_remote_code=trust_remote_code,
            )
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                str(resolved_hf_model_dir),
                local_files_only=True,
                dtype=dtype,
                config=hf_config,
                trust_remote_code=trust_remote_code,
            )
        hf = hf_model.to(device)
        hf.eval()
        hooks: list[Any] = []
        add_mode = _AddTraceMode()
        try:
            hf_trace, hf_trace_meta, hooks = _capture_hf_trace(hf, trace_kinds=trace_kinds)
            with add_mode, torch.no_grad():
                hf_logits = _extract_logits(hf(**hf_kwargs, use_cache=False)).detach().float().cpu()
            hf_trace["add"] = list(add_mode.outputs)
            hf_trace_meta["add"] = list(add_mode.meta)
        finally:
            for hook in hooks:
                hook.remove()
            del hf
    except Exception as exc:  # noqa: BLE001
        hf_error = f"{type(exc).__name__}: {exc}"
        hf_logits = None

    try:
        state_dict = _load_state_dict(state_dict_paths, device=device, dtype=dtype)
        syn = _TracingSynapseProgramModel(lowered_spec, state_dict, trace_kinds=trace_kinds).to(
            device
        )
        syn.eval()
        with torch.no_grad():
            syn_logits = _extract_logits(syn(**syn_kwargs)).detach().float().cpu()
        syn_trace = syn.trace
        syn_trace_meta = syn.trace_meta
        del syn
        del state_dict
    except Exception as exc:  # noqa: BLE001
        syn_error = f"{type(exc).__name__}: {exc}"
        syn_logits = None

    if hf_logits is not None and syn_logits is not None:
        if hf_logits.shape == syn_logits.shape:
            diff = (syn_logits - hf_logits).abs()
            logits_summary = {
                "mean_abs": float(diff.mean()),
                "max_abs": float(diff.max()),
                "last_max_abs": float(diff[:, -1, :].max()),
                "top1_eq": bool(
                    (syn_logits[:, -1, :].argmax(-1) == hf_logits[:, -1, :].argmax(-1)).all()
                ),
            }
        else:
            logits_summary = {
                "shape_mismatch": {
                    "hf": list(hf_logits.shape),
                    "synapse": list(syn_logits.shape),
                }
            }

    by_op: dict[str, Any] = {}
    for kind in sorted(trace_kinds):
        by_op[kind] = _pair_stats(
            hf_trace.get(kind, []),
            syn_trace.get(kind, []),
            hf_meta=hf_trace_meta.get(kind, []),
            syn_meta=syn_trace_meta.get(kind, []),
            dtype_name=dtype_name,
        )

    return {
        "dtype": dtype_name,
        "hf_error": hf_error,
        "synapse_error": syn_error,
        "logits": logits_summary,
        "by_op": by_op,
    }


def run_axon_layer_op_parity(
    *,
    axon_file: Path,
    weights: Path,
    layer_index: int,
    hf_model_dir: Path | None = None,
    tokenizer: str | None = None,
    text: str | Sequence[str] = ("Hello world",),
    device: str = "cpu",
    dtype: str = "float32",
    class_name: str = "AxonLayerParityModel",
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype_name(dtype)
    prompts = _normalize_texts(text)
    axon_path = axon_file.resolve()
    weights_path = weights.resolve()
    if not axon_path.exists():
        raise FileNotFoundError(f"Axon file not found: {axon_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights path not found: {weights_path}")

    state_dict_paths = _resolve_safetensors_paths(weights_path)
    default_hf_dir = weights_path if weights_path.is_dir() else state_dict_paths[0].parent
    resolved_hf_model_dir = (hf_model_dir or default_hf_dir).resolve()
    model_config = _augment_model_config_from_checkpoint(
        model_dir=resolved_hf_model_dir,
        safetensors_files=state_dict_paths,
        model_config=_load_model_config(resolved_hf_model_dir),
    )
    effective_trust_remote_code = _should_trust_remote_code(
        resolved_hf_model_dir,
        model_config=model_config,
    )
    hf_config = _load_auto_config_with_compat_fallback(
        resolved_hf_model_dir,
        trust_remote_code=effective_trust_remote_code,
    )
    hf_config = _normalize_rope_numeric_fields(hf_config)
    tokenizer_source = tokenizer or str(resolved_hf_model_dir)
    if tokenizer is None:
        for candidate in candidate_tokenizer_dirs(resolved_hf_model_dir):
            if looks_like_tokenizer_dir(candidate):
                tokenizer_source = str(candidate)
                break
    tokenizer_fallback = resolved_hf_model_dir.name if tokenizer is None else None

    layer_scope = f"model.layers.{int(layer_index)}."
    trace_kinds = {"linear", "layernorm", "rmsnorm", "attention", "embedding", "add"}

    with TemporaryDirectory(prefix="axon_layer_op_parity_") as tmp_dir:
        modules = parse_axon_program_from_path(axon_path)
        lowered_spec = lower_axon_program_to_synapse_spec(modules)
        loaded = OmegaConf.create(lowered_spec)
        loaded_dict = OmegaConf.to_container(loaded, resolve=True)
        if not isinstance(loaded_dict, dict):
            raise ValueError("Lowered synapse spec did not produce a mapping")
        final_spec: dict[str, Any] = {str(k): v for k, v in loaded_dict.items()}
        model_section = final_spec.get("model")
        if isinstance(model_section, dict) and model_config is not None:
            model_section["config"] = model_config
        model_input_names = set(final_spec.get("model", {}).get("inputs", {}).keys())

        tokenizer_obj, input_ids, attention_mask = _build_inputs(
            lowered_spec=final_spec,
            prompts=prompts,
            tokenizer_source=tokenizer_source,
            tokenizer_fallback=tokenizer_fallback,
            device=resolved_device,
        )
        del tokenizer_obj
        hf_kwargs = _forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            for_hf=True,
            model_input_names=model_input_names,
        )
        syn_kwargs = _forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            for_hf=False,
            model_input_names=model_input_names,
        )

        _ensure_transformers_import_compat()
        hf_model: Any = AutoModelForCausalLM.from_pretrained(
            str(resolved_hf_model_dir), local_files_only=True, dtype=resolved_dtype
        )
        hf = hf_model.to(resolved_device).eval()
        hooks: list[Any] = []
        hf_trace: dict[str, list[torch.Tensor]] = defaultdict(list)
        hf_trace_meta: dict[str, list[dict[str, Any]]] = defaultdict(list)
        try:
            raw_trace, raw_meta, hooks = _capture_hf_trace(hf, trace_kinds=trace_kinds)
            with torch.no_grad():
                _ = _extract_logits(hf(**hf_kwargs, use_cache=False))
            for hf_trace_kind, items in raw_trace.items():
                meta_items = raw_meta.get(hf_trace_kind, [])
                for idx, tensor in enumerate(items):
                    meta = meta_items[idx] if idx < len(meta_items) else {}
                    module_name = str(meta.get("module_name", ""))
                    if layer_scope in module_name:
                        hf_trace[hf_trace_kind].append(tensor)
                        hf_trace_meta[hf_trace_kind].append(meta)
        finally:
            for hook in hooks:
                hook.remove()
            del hf
            del hf_model

        state_dict = _load_state_dict(state_dict_paths, device=resolved_device, dtype=resolved_dtype)
        generated_py_path = Path(tmp_dir) / "generated_model.py"
        generated_py_path.write_text(
            emit_model_code_from_synapse_spec(final_spec, class_name=class_name),
            encoding="utf-8",
        )
        model_cls = _load_generated_class(generated_py_path, class_name)
        generated = model_cls.from_state_dict(state_dict).to(resolved_device).eval()
        setattr(generated, "_trace_enabled", True)
        if hasattr(generated, "_reset_trace"):
            generated._reset_trace()
        with torch.no_grad():
            _ = _extract_logits(generated(**syn_kwargs))
        generated_trace_raw = list(getattr(generated, "trace_ops", []))
        del generated
        del state_dict

    syn_trace: dict[str, list[torch.Tensor]] = defaultdict(list)
    syn_trace_meta: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ordered_synapse_ops: list[dict[str, Any]] = []
    for item in generated_trace_raw:
        if not isinstance(item, dict):
            continue
        node_path = str(item.get("node_path", ""))
        if layer_scope not in node_path:
            continue
        tensor_raw = item.get("tensor")
        if not torch.is_tensor(tensor_raw):
            continue
        tensor = tensor_raw
        op = str(item.get("op", ""))
        trace_kind: str | None = None
        if op == "linear":
            weight_path = str(item.get("weight_path", ""))
            if _is_attention_output_projection(weight_path):
                trace_kind = "attention"
        if trace_kind is None:
            trace_kind = _synapse_kind(op)
        if trace_kind is None:
            ordered_synapse_ops.append(
                {
                    "node_path": node_path,
                    "op": op,
                    "bind": str(item.get("bind", "")),
                    "shape": list(tensor.shape),
                }
            )
            continue
        meta = {
            "node_path": node_path,
            "op": op,
            "bind": str(item.get("bind", "")),
        }
        weight_path_raw = item.get("weight_path")
        if isinstance(weight_path_raw, str) and weight_path_raw:
            meta["weight_path"] = weight_path_raw
        syn_trace[trace_kind].append(tensor)
        syn_trace_meta[trace_kind].append(meta)
        ordered_synapse_ops.append(
            {
                "node_path": node_path,
                "op": op,
                "bind": str(item.get("bind", "")),
                "shape": list(tensor.shape),
            }
        )

    by_op: dict[str, Any] = {}
    for kind in sorted(trace_kinds):
        by_op[kind] = _pair_stats(
            hf_trace.get(kind, []),
            syn_trace.get(kind, []),
            hf_meta=hf_trace_meta.get(kind, []),
            syn_meta=syn_trace_meta.get(kind, []),
            dtype_name=dtype,
        )

    return {
        "axon_file": str(axon_path),
        "weights": str(weights_path),
        "hf_model_dir": str(resolved_hf_model_dir),
        "device": str(resolved_device),
        "dtype": dtype,
        "layer_index": int(layer_index),
        "prompts": prompts,
        "hf_trace_counts": {kind: len(values) for kind, values in hf_trace.items()},
        "syn_trace_counts": {kind: len(values) for kind, values in syn_trace.items()},
        "by_op": by_op,
        "ordered_synapse_ops": ordered_synapse_ops,
    }


def run_axon_op_parity(
    *,
    axon_file: Path,
    weights: Path,
    hf_model_dir: Path | None = None,
    tokenizer: str | None = None,
    text: str | Sequence[str] = ("The future of AI is", "Hello world"),
    device: str = "cpu",
    dtypes: Sequence[str] | None = None,
    output_json: Path | None = None,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    resolved_dtypes = _normalize_dtypes(dtypes)
    prompts = _normalize_texts(text)
    axon_path = axon_file.resolve()
    weights_path = weights.resolve()
    if not axon_path.exists():
        raise FileNotFoundError(f"Axon file not found: {axon_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights path not found: {weights_path}")
    state_dict_paths = _resolve_safetensors_paths(weights_path)
    default_hf_dir = weights_path if weights_path.is_dir() else state_dict_paths[0].parent
    resolved_hf_model_dir = (hf_model_dir or default_hf_dir).resolve()
    model_config = _augment_model_config_from_checkpoint(
        model_dir=resolved_hf_model_dir,
        safetensors_files=state_dict_paths,
        model_config=_load_model_config(resolved_hf_model_dir),
    )
    effective_trust_remote_code = _should_trust_remote_code(
        resolved_hf_model_dir,
        model_config=model_config,
    )
    hf_config = _load_auto_config_with_compat_fallback(
        resolved_hf_model_dir,
        trust_remote_code=effective_trust_remote_code,
    )
    hf_config = _normalize_rope_numeric_fields(hf_config)
    hf_loader_kind = _resolve_hf_loader_kind(hf_config)
    tokenizer_source = tokenizer or str(resolved_hf_model_dir)
    if tokenizer is None:
        for candidate in candidate_tokenizer_dirs(resolved_hf_model_dir):
            if looks_like_tokenizer_dir(candidate):
                tokenizer_source = str(candidate)
                break
    tokenizer_fallback = resolved_hf_model_dir.name if tokenizer is None else None

    with TemporaryDirectory(prefix="axon_op_parity_") as _tmp_dir:
        modules = parse_axon_program_from_path(axon_path)
        lowered_spec = lower_axon_program_to_synapse_spec(modules)
        loaded = OmegaConf.create(lowered_spec)
        loaded_dict = OmegaConf.to_container(loaded, resolve=True)
        if not isinstance(loaded_dict, dict):
            raise ValueError("Lowered synapse spec did not produce a mapping")
        final_spec: dict[str, Any] = {str(k): v for k, v in loaded_dict.items()}
        model_section = final_spec.get("model")
        if isinstance(model_section, dict) and model_config is not None:
            model_section["config"] = model_config

        results: list[dict[str, Any]] = []
        for dtype_name in resolved_dtypes:
            dtype = _resolve_dtype_name(dtype_name)
            print(f"[op-parity] running dtype={dtype_name}")
            result = _run_single_dtype(
                dtype_name=dtype_name,
                dtype=dtype,
                resolved_hf_model_dir=resolved_hf_model_dir,
                lowered_spec=final_spec,
                state_dict_paths=state_dict_paths,
                tokenizer_source=tokenizer_source,
                tokenizer_fallback=tokenizer_fallback,
                prompts=prompts,
                device=resolved_device,
                hf_config=hf_config,
                hf_loader_kind=hf_loader_kind,
                trust_remote_code=effective_trust_remote_code,
            )
            results.append(result)
            logits = result.get("logits")
            if isinstance(logits, dict) and "mean_abs" in logits:
                print(
                    f"[op-parity] dtype={dtype_name} logits mean/max/last/top1="
                    f"{logits['mean_abs']:.6f}/{logits['max_abs']:.6f}/{logits['last_max_abs']:.6f}/{logits['top1_eq']}"
                )
            if result.get("hf_error") or result.get("synapse_error"):
                print(
                    f"[op-parity] dtype={dtype_name} hf_error={result.get('hf_error')} synapse_error={result.get('synapse_error')}"
                )

    payload = {
        "axon_file": str(axon_path),
        "weights": str(weights_path),
        "hf_model_dir": str(resolved_hf_model_dir),
        "tokenizer_source": tokenizer_source,
        "device": str(resolved_device),
        "prompts": prompts,
        "results": results,
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[op-parity] wrote report: {output_json}")
    return payload


def run_codegen_runtime_parity(
    *,
    axon_file: Path,
    weights: Path,
    hf_model_dir: Path | None = None,
    tokenizer: str | None = None,
    text: str | Sequence[str] = ("The future of AI is",),
    device: str = "cpu",
    dtype: str = "float32",
    class_name: str = "AxonGeneratedParityModel",
    max_reported: int = 20,
    abs_tol: float = 1.0e-5,
    rel_tol: float = 1.0e-3,
    output_json: Path | None = None,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype_name(dtype)
    prompts = _normalize_texts(text)
    axon_path = axon_file.resolve()
    weights_path = weights.resolve()
    if not axon_path.exists():
        raise FileNotFoundError(f"Axon file not found: {axon_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights path not found: {weights_path}")

    state_dict_paths = _resolve_safetensors_paths(weights_path)
    default_hf_dir = weights_path if weights_path.is_dir() else state_dict_paths[0].parent
    resolved_hf_model_dir = (hf_model_dir or default_hf_dir).resolve()
    tokenizer_source = tokenizer or str(resolved_hf_model_dir)
    if tokenizer is None:
        for candidate in candidate_tokenizer_dirs(resolved_hf_model_dir):
            if looks_like_tokenizer_dir(candidate):
                tokenizer_source = str(candidate)
                break
    tokenizer_fallback = resolved_hf_model_dir.name if tokenizer is None else None

    with TemporaryDirectory(prefix="axon_codegen_runtime_parity_") as tmp_dir:
        modules = parse_axon_program_from_path(axon_path)
        lowered_spec = lower_axon_program_to_synapse_spec(modules)
        loaded = OmegaConf.create(lowered_spec)
        loaded_dict = OmegaConf.to_container(loaded, resolve=True)
        if not isinstance(loaded_dict, dict):
            raise ValueError("Lowered synapse spec did not produce a mapping")
        final_spec: dict[str, Any] = {str(k): v for k, v in loaded_dict.items()}
        model_input_names = set(final_spec.get("model", {}).get("inputs", {}).keys())

        tokenizer_obj, input_ids, attention_mask = _build_inputs(
            lowered_spec=final_spec,
            prompts=prompts,
            tokenizer_source=tokenizer_source,
            tokenizer_fallback=tokenizer_fallback,
            device=resolved_device,
        )
        del tokenizer_obj
        syn_kwargs = _forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            for_hf=False,
            model_input_names=model_input_names,
        )

        state_dict = _load_state_dict(
            state_dict_paths, device=resolved_device, dtype=resolved_dtype
        )
        runtime = _NodeTracingSynapseProgramModel(final_spec, state_dict).to(resolved_device).eval()

        generated_py_path = Path(tmp_dir) / "generated_model.py"
        generated_py_path.write_text(
            emit_model_code_from_synapse_spec(final_spec, class_name=class_name),
            encoding="utf-8",
        )
        model_cls = _load_generated_class(generated_py_path, class_name)
        generated = model_cls.from_state_dict(state_dict).to(resolved_device).eval()
        setattr(generated, "_trace_enabled", True)
        if hasattr(generated, "_reset_trace"):
            generated._reset_trace()

        with torch.no_grad():
            generated_logits = _extract_logits(generated(**syn_kwargs)).detach().float().cpu()
            runtime_logits = _extract_logits(runtime(**syn_kwargs)).detach().float().cpu()

        generated_trace_raw = list(getattr(generated, "trace_ops", []))
        runtime_trace_raw = list(runtime.trace_ops)
        generated_trace = [
            {
                "index": idx,
                "node_path": str(item["node_path"]),
                "op": str(item["op"]),
                "bind": str(item["bind"]),
                "dtype": str(item.get("dtype", "")),
                "tensor": item["tensor"],
            }
            for idx, item in enumerate(generated_trace_raw)
            if isinstance(item, dict) and torch.is_tensor(item.get("tensor"))
        ]
        runtime_trace = [
            {
                "index": idx,
                "node_path": str(item["node_path"]),
                "op": str(item["op"]),
                "bind": str(item["bind"]),
                "dtype": str(item.get("dtype", "")),
                "tensor": item["tensor"],
            }
            for idx, item in enumerate(runtime_trace_raw)
            if isinstance(item, dict) and torch.is_tensor(item.get("tensor"))
        ]

        first_divergence: dict[str, Any] | None = None
        compared: list[dict[str, Any]] = []
        pair_count = min(len(generated_trace), len(runtime_trace))
        for idx in range(pair_count):
            gen_item = generated_trace[idx]
            run_item = runtime_trace[idx]
            if (
                gen_item["node_path"] != run_item["node_path"]
                or gen_item["op"] != run_item["op"]
                or gen_item["bind"] != run_item["bind"]
            ):
                mismatch = {
                    "index": idx,
                    "reason": "trace-key-mismatch",
                    "generated": {
                        "node_path": gen_item["node_path"],
                        "op": gen_item["op"],
                        "bind": gen_item["bind"],
                    },
                    "runtime": {
                        "node_path": run_item["node_path"],
                        "op": run_item["op"],
                        "bind": run_item["bind"],
                    },
                }
                if first_divergence is None:
                    first_divergence = mismatch
                compared.append(mismatch)
                continue
            if "float" not in gen_item["dtype"] or "float" not in run_item["dtype"]:
                continue
            if gen_item["tensor"].shape != run_item["tensor"].shape:
                mismatch = {
                    "index": idx,
                    "reason": "shape-mismatch",
                    "node_path": gen_item["node_path"],
                    "op": gen_item["op"],
                    "bind": gen_item["bind"],
                    "generated_shape": list(gen_item["tensor"].shape),
                    "runtime_shape": list(run_item["tensor"].shape),
                }
                if first_divergence is None:
                    first_divergence = mismatch
                compared.append(mismatch)
                continue
            stats = _tensor_diff_stats(gen_item["tensor"], run_item["tensor"])
            row = {
                "index": idx,
                "node_path": gen_item["node_path"],
                "op": gen_item["op"],
                "bind": gen_item["bind"],
                **stats,
            }
            compared.append(row)
            if first_divergence is None and (
                stats["max_abs"] > abs_tol or stats["max_rel"] > rel_tol
            ):
                first_divergence = row

        logits_summary: dict[str, Any]
        if generated_logits.shape == runtime_logits.shape:
            logits_summary = {}
            logits_summary.update(_tensor_diff_stats(generated_logits, runtime_logits))
            logits_summary["shape"] = list(generated_logits.shape)
            logits_summary["top1_eq"] = bool(
                (generated_logits[:, -1, :].argmax(-1) == runtime_logits[:, -1, :].argmax(-1)).all()
            )
        else:
            logits_summary = {
                "shape_mismatch": {
                    "generated": list(generated_logits.shape),
                    "runtime": list(runtime_logits.shape),
                }
            }

        numeric_rows = [
            item
            for item in compared
            if isinstance(item.get("max_abs"), float) and isinstance(item.get("max_rel"), float)
        ]
        numeric_rows.sort(key=lambda item: float(item["max_abs"]), reverse=True)

    payload = {
        "axon_file": str(axon_path),
        "weights": str(weights_path),
        "hf_model_dir": str(resolved_hf_model_dir),
        "tokenizer_source": tokenizer_source,
        "device": str(resolved_device),
        "dtype": dtype,
        "prompts": prompts,
        "trace_counts": {
            "generated": len(generated_trace),
            "runtime": len(runtime_trace),
            "paired": pair_count,
        },
        "tolerances": {"abs_tol": float(abs_tol), "rel_tol": float(rel_tol)},
        "logits": logits_summary,
        "first_divergence": first_divergence,
        "top_divergences": numeric_rows[: max(1, int(max_reported))],
    }
    if len(generated_trace) != len(runtime_trace) and first_divergence is None:
        payload["first_divergence"] = {
            "reason": "trace-length-mismatch",
            "generated": len(generated_trace),
            "runtime": len(runtime_trace),
        }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[op-parity] wrote codegen/runtime report: {output_json}")
    return payload


__all__ = ["run_axon_op_parity", "run_codegen_runtime_parity"]
