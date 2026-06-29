from __future__ import annotations

import importlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import safetensors
import torch

from brainsurgery.synapse.axon import (
    elaborate_closed_axon_file,
    flatten_closed_axon_file,
    lower_axon_program_to_graph_ir,
    normalize_closed_axon_file,
    resolve_axon_program_from_path,
    typecheck2_flat_axon_file,
)
from brainsurgery.synapse.axon.graph_ir.optimize import (
    GraphOptimizeConfig,
    optimize_graph_program,
)

from .base import CacheState, ModelConfig, ServingModel

logger = logging.getLogger("brainsurgery.serving")

_BACKEND_EMITTERS: dict[str, str] = {
    "codegen2-torch": "brainsurgery.synapse.axon.codegen2_torch",
    "codegen2-mlx": "brainsurgery.synapse.axon.codegen2_mlx",
    "codegen2-tinygrad": "brainsurgery.synapse.axon.codegen2_tinygrad",
}


def _load_model_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def _resolve_safetensors_paths(weights: Path) -> list[Path]:
    if weights.is_file():
        return [weights]
    paths = sorted(weights.rglob("*.safetensors"))
    if not paths:
        paths = sorted(weights.glob("*.safetensors"))
    return paths


def _load_state_dict_torch(paths: list[Path], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for path in paths:
        st = safetensors.safe_open(str(path), framework="pt")
        for key in st.keys():
            tensor = st.get_tensor(key)
            if tensor.is_floating_point():
                tensor = tensor.to(device=device, dtype=dtype)
            else:
                tensor = tensor.to(device=device)
            out[key] = tensor
    return out


def _load_generated_class(code: str, class_name: str) -> type:
    namespace: dict[str, Any] = {}
    exec(code, namespace)
    return namespace[class_name]


def _detect_model_config(model_config_dict: dict[str, Any]) -> ModelConfig:
    hf = model_config_dict
    num_layers = (
        hf.get("num_hidden_layers")
        or hf.get("n_layer")
        or hf.get("num_layers", 12)
    )
    num_heads = hf.get("num_attention_heads") or hf.get("n_head", 12)
    head_dim = hf.get("head_dim") or (hf.get("hidden_size", 768) // num_heads)
    vocab_size = hf.get("vocab_size", 50257)
    hidden_size = hf.get("hidden_size") or hf.get("n_embd", 768)
    max_seq_len = hf.get("max_position_embeddings") or hf.get("seq_length", 2048)
    return ModelConfig(
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        hidden_dim=hidden_size,
        dtype=hf.get("torch_dtype", "float32"),
        extra=hf,
    )


class AxonServingModel(ServingModel):
    def __init__(
        self,
        model: Any,
        config: ModelConfig,
        backend: str,
        device: torch.device,
        dtype: torch.dtype,
        paged_attention: bool = False,
    ):
        self._model = model
        self.config = config
        self._backend = backend
        self._device = device
        self._dtype = dtype
        self._paged_attention = paged_attention

    @staticmethod
    def _to_torch(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: AxonServingModel._to_torch(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(AxonServingModel._to_torch(v) for v in value)
        if isinstance(value, list):
            return [AxonServingModel._to_torch(v) for v in value]
        if _is_mlx_array(value):
            import numpy as np
            return torch.from_numpy(np.asarray(value))
        return value

    @classmethod
    def load(
        cls,
        axon_file: Path,
        weights: Path,
        *,
        backend: str = "codegen2-torch",
        device: str = "cpu",
        dtype: str = "float32",
        optimize_graph: bool = False,
        graph_backend_intrinsics: str | None = None,
        paged_attention: bool = False,
        class_name: str = "AxonServingModel",
    ) -> AxonServingModel:
        resolved_dtype: torch.dtype = getattr(torch, dtype, torch.float32)
        target_device = torch.device(device)

        logger.info(
            "Compiling %s with backend=%s device=%s dtype=%s paged=%s",
            axon_file, backend, device, dtype, paged_attention,
        )

        graph_program = _compile_to_graph_ir(axon_file, optimize_graph, graph_backend_intrinsics)
        model_config_dict = _load_model_config(weights)
        code = _codegen(graph_program, backend, class_name, model_config_dict, paged_attention=paged_attention)
        ModelClass = _load_generated_class(code, class_name)

        safetensors_paths = _resolve_safetensors_paths(weights)
        if backend == "codegen2-mlx":
            import mlx.core as mx
            state_dict: dict[str, Any] = {}
            for path in safetensors_paths:
                with safetensors.safe_open(str(path), framework="np") as f:
                    for key in f.keys():
                        state_dict[str(key)] = mx.array(f.get_tensor(key))
            model = ModelClass(state_dict).eval()
        else:
            state_dict = _load_state_dict_torch(safetensors_paths, target_device, resolved_dtype)
            if backend == "codegen2-tinygrad":
                model = ModelClass(state_dict).eval()
            else:
                model = ModelClass.from_state_dict(state_dict).eval()

        detected = _detect_model_config(model_config_dict)
        logger.info(
            "Model loaded: %d layers, %d heads, %d vocab, %d max_seq_len, backend=%s",
            detected.num_layers, detected.num_heads, detected.vocab_size,
            detected.max_seq_len, backend,
        )
        return cls(model, detected, backend, target_device, resolved_dtype, paged_attention=paged_attention)

    def forward(
        self,
        input_ids: Any,
        *,
        past_kv: CacheState | None = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, CacheState]:
        if self._paged_attention:
            pass
        else:
            kwargs["use_cache"] = use_cache
        if self._backend == "codegen2-mlx":
            if self._paged_attention:
                output = self._forward_mlx_paged(input_ids, **kwargs)
            else:
                output = self._forward_mlx(input_ids, past_kv, **kwargs)
        elif self._backend == "codegen2-tinygrad":
            if not self._paged_attention and past_kv is not None:
                kwargs["past_kv"] = past_kv
            output = self._model._forward_maybe_jit(input_ids, **kwargs)
        else:
            if not self._paged_attention and past_kv is not None:
                kwargs["past_kv"] = past_kv
            output = self._model(input_ids, **kwargs)
        if self._backend in ("codegen2-mlx", "codegen2-tinygrad"):
            return _unpack_output(output, paged=self._paged_attention)
        output = self._to_torch(output)
        return _unpack_output(output, paged=self._paged_attention)

    def sample(
        self,
        logits: Any,
        temperature: float = 0.0,
        top_p: float = 1.0,
        prefill: bool = True,
    ) -> int:
        if self._backend == "codegen2-mlx":
            return self._sample_mlx(logits, temperature, top_p, prefill)
        if self._backend == "codegen2-tinygrad":
            return self._sample_tinygrad(logits, temperature, top_p, prefill)
        return super().sample(logits, temperature, top_p, prefill)

    def _sample_mlx(
        self,
        logits: Any,
        temperature: float,
        top_p: float,
        prefill: bool,
    ) -> int:
        import numpy as np

        last = logits[0, -1, :] if prefill else logits[0, 0, :]
        if isinstance(last, (list, tuple)):
            last_np = np.asarray(last, dtype=np.float32)
        else:
            last_np = np.array(last, dtype=np.float32)
        if temperature > 0.0:
            scaled = last_np / temperature
            probs = np.exp(scaled - scaled.max())
            probs = probs / probs.sum()
            if top_p < 1.0:
                sorted_idx = np.argsort(-probs)
                sorted_probs = probs[sorted_idx]
                cumsum = np.cumsum(sorted_probs)
                cutoff = cumsum > top_p
                cutoff = np.roll(cutoff, 1)
                cutoff[0] = False
                sorted_probs[cutoff] = 0.0
                probs[sorted_idx] = sorted_probs
                probs = probs / probs.sum()
            idx = int(np.random.choice(len(probs), p=probs))
            return idx
        return int(np.argmax(last_np).item())

    def _sample_tinygrad(
        self,
        logits: Any,
        temperature: float,
        top_p: float,
        prefill: bool,
    ) -> int:
        from tinygrad import Tensor

        last = logits[0, -1, :] if prefill else logits[0, 0, :]
        if not isinstance(last, Tensor):
            last = Tensor(last.detach().cpu().numpy())
        tg = last
        if temperature > 0.0:
            scaled = tg / temperature
            if top_p < 1.0:
                probs = scaled.softmax()
                sv, si = probs.sort(descending=True)
                cumsum = sv.cumsum(axis=-1)
                mask = cumsum > top_p
                sv = sv.where(~mask, Tensor.zeros_like(sv))
                probs_masked = sv / sv.sum()
                idx = int(probs_masked.multinomial(1).item())
                return int(si[idx].item())
            return int(scaled.multinomial(1).item())
        return int(tg.argmax().item())

    def _forward_mlx(
        self,
        input_ids: torch.Tensor,
        past_kv: CacheState | None,
        **kwargs: Any,
    ) -> Any:
        import mlx.core as mx
        import numpy as np

        def _to_mx(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, mx.array):
                return value
            if isinstance(value, torch.Tensor):
                return mx.array(value.cpu().numpy())
            if isinstance(value, list):
                return [_to_mx(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_to_mx(v) for v in value)
            return mx.array(np.asarray(value))

        original_to_mlx = self._model._to_mlx

        def _patched_to_mlx(value: Any, dtype: Any = None) -> Any:
            if isinstance(value, list):
                return [_patched_to_mlx(v, dtype) for v in value]
            if isinstance(value, tuple):
                return tuple(_patched_to_mlx(v, dtype) for v in value)
            return original_to_mlx(value, dtype)

        self._model._to_mlx = _patched_to_mlx
        try:
            mx_input = _to_mx(input_ids)
            mx_kwargs = {k: _to_mx(v) for k, v in kwargs.items()}
            mx_kwargs["past_kv"] = _to_mx(past_kv)
            return self._model._forward(mx_input, **mx_kwargs)
        finally:
            self._model._to_mlx = original_to_mlx

    def _forward_mlx_paged(
        self,
        input_ids: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        import mlx.core as mx
        import numpy as np

        def _to_mx(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, mx.array):
                return value
            if isinstance(value, torch.Tensor):
                return mx.array(value.cpu().numpy())
            if isinstance(value, list):
                return mx.array(np.asarray(value))
            if isinstance(value, tuple):
                return tuple(_to_mx(v) for v in value)
            return mx.array(np.asarray(value))

        mx_input = _to_mx(input_ids)
        mx_kwargs = {}
        for k, v in kwargs.items():
            if k in ("position", "block_size"):
                mx_kwargs[k] = int(v) if isinstance(v, (int, torch.Tensor)) else v
            else:
                mx_kwargs[k] = _to_mx(v)
        return self._model._forward(mx_input, **mx_kwargs)


def _compile_to_graph_ir(
    axon_file: Path,
    optimize_graph: bool = False,
    graph_backend_intrinsics: str | None = None,
) -> Any:
    resolved = resolve_axon_program_from_path(axon_file).ast
    normalized = normalize_closed_axon_file(resolved)
    elaborated = elaborate_closed_axon_file(normalized)
    flattened = flatten_closed_axon_file(elaborated)
    typed = typecheck2_flat_axon_file(flattened)
    graph = lower_axon_program_to_graph_ir(typed)
    if optimize_graph:
        graph = optimize_graph_program(
            graph,
            config=GraphOptimizeConfig(backend_intrinsics=graph_backend_intrinsics),
        )
    return graph


def _codegen(
    graph_program: Any,
    backend: str,
    class_name: str,
    model_config_dict: dict[str, Any],
    *,
    paged_attention: bool = False,
) -> str:
    emit_module_path = _BACKEND_EMITTERS.get(backend)
    if emit_module_path is None:
        raise ValueError(f"Unknown backend: {backend}. Choose from: {', '.join(_BACKEND_EMITTERS)}")
    emit_module = importlib.import_module(emit_module_path)
    emit_fn = getattr(emit_module, "emit_model_code_from_graph_ir")
    return emit_fn(
        graph_program,
        class_name=class_name,
        model_config=model_config_dict,
        paged_attention=paged_attention,
    )


def _is_mlx_array(value: Any) -> bool:
    try:
        import mlx.core as mx
        return isinstance(value, mx.array)
    except ImportError:
        return False


def _unpack_output(output: Any, *, paged: bool = False) -> tuple[torch.Tensor, CacheState]:
    if isinstance(output, dict):
        logits = output.get("logits", output.get("result"))
        if paged:
            new_kv = output.get("new_kv")
        else:
            new_kv = output.get("new_kv") or output.get("past_kv")
        if logits is None:
            keys = list(output.keys())
            logits = output[keys[0]]
            new_kv = output[keys[1]] if len(keys) > 1 else None
        return logits, new_kv
    return output, None
