from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..ast import TypeBool, TypeDim, TypeFloat, TypeInt, TypeList, TypeOptional
from ..codegen2_common import normalize_primitive_op
from ..codegen2_torch.core import _DirectTorchEmitter, _is_static_mask_type, graph_main_output_names
from ..graph_ir import GraphProgram, validate_graph_program


SHARED_COMMON_PRIMITIVES: frozenset[str] = frozenset({
    "params_param",
    "params_has_root",
    "config_int",
    "config_dim",
    "config_float",
    "config_bool",
    "config_str",
    "config_value",
    "config_list",
    "config_has",
    "config_has_value",
    "shape",
    "tensor_size",
    "list_init",
    "list_append",
    "list_index",
    "require",
    "list_length",
})

SUPPORTED_TINYGRAD_PRIMITIVES: frozenset[str] = frozenset({
    "embedding",
    "linear",
    "expert_linear",
    "layernorm",
    "rmsnorm",
    "l2norm",
    "reshape",
    "arange",
    "slice",
    "assign_slice",
    "chunk",
    "split",
    "concat",
    "repeat",
    "expand",
    "permute",
    "transpose",
    "unsqueeze",
    "matmul",
    "softmax",
    "sum",
    "where",
    "where_indices",
    "gather",
    "scatter",
    "index_add",
    "clamp",
    "le",
    "eq",
    "and",
    "add",
    "mul",
    "div",
    "pow",
    "floor",
    "sqrt",
    "sin",
    "cos",
    "exp",
    "log",
    "cast",
    "cast_like",
    "cumsum",
    "dtype_value",
    "empty_like",
    "fill",
    "empty",
    "zeros",
    "full",
    "zeros_like",
    "tensor_like",
    "topk",
    "activations_tanh",
    "activations_silu",
    "activations_sigmoid",
    "activations_relu",
    "activations_relu2",
    "activations_swiglu",
    "activations_gelu",
    "activations_gelu_new",
    "activations_gelu_pytorch_tanh",
    "activations_gegelu",
    "activations_xielu",
    "_tinygrad_sdpa",
})

# Kept as a diagnostic table for genuinely missing tinygrad coverage. It should
# stay empty for models we claim to support.
NON_OBVIOUS_TINYGRAD_OPS: dict[str, str] = {}


@dataclass(frozen=True)
class TinygradUnsupportedOp:
    op: str
    count: int
    reason: str


def _normalize_primitive_op(name: str) -> str:
    return normalize_primitive_op(name)


def non_obvious_tinygrad_ops(graph: GraphProgram) -> tuple[TinygradUnsupportedOp, ...]:
    counts: Counter[str] = Counter()
    module_names = {module.name for module in graph.modules}
    for module in graph.modules:
        for node in module.nodes:
            op = node.op.name
            if op.startswith("core.") or op.startswith("core.binary.") or op in module_names:
                continue
            primitive = _normalize_primitive_op(op)
            if primitive in SHARED_COMMON_PRIMITIVES or primitive in SUPPORTED_TINYGRAD_PRIMITIVES:
                continue
            counts[primitive] += 1
    return tuple(
        TinygradUnsupportedOp(
            op=op,
            count=count,
            reason=NON_OBVIOUS_TINYGRAD_OPS.get(op, "no tinygrad lowering classified yet"),
        )
        for op, count in sorted(counts.items())
    )


def tinygrad_op_table_markdown(graph: GraphProgram) -> str:
    rows = non_obvious_tinygrad_ops(graph)
    if not rows:
        return "| Op | Count | Reason |\n|---|---:|---|\n"
    lines = ["| Op | Count | Reason |", "|---|---:|---|"]
    for row in rows:
        lines.append(f"| `{row.op}` | {row.count} | {row.reason} |")
    return "\n".join(lines)


class _DirectTinygradEmitter(_DirectTorchEmitter):
    def emit(self) -> str:
        return super().emit().replace(f"class {self.class_name}(nn.Module):", f"class {self.class_name}:")

    def _emit_common(self, lines: list[str]) -> None:
        add = self._add
        add(lines, 4, "def __init__(self, state_dict: dict[str, torch.Tensor], config: dict | None = None, param_devices=None):")
        add(lines, 8, "super().__init__()")
        add(lines, 8, "self._torch_device = torch.device('cpu')")
        add(lines, 8, "self._tiny_device = self._infer_tiny_device(state_dict)")
        add(lines, 8, "self._torch_backing_tensors = []")
        add(lines, 8, "self.state_dict_tensors = {}")
        add(lines, 8, "self._embedding_aranges = {}")
        add(lines, 8, "self.config = dict(({} if _MODEL_CONFIG is None else _MODEL_CONFIG) if config is None else config)")
        add(lines, 8, "self._profile_enabled = False")
        add(lines, 8, "self._profile_cuda = True")
        add(lines, 8, "self._profile_records = {}")
        add(lines, 8, "self._symbols = {}")
        add(lines, 8, "self._jit_enabled = True")
        add(lines, 8, "self._forward_jits = {}")
        add(lines, 8, "self._forward_jit_seen = set()")
        add(lines, 8, "self._tinygrad_mutable_slices = False")
        add(lines, 8, "self._tinygrad_owned_static_cache_roots = set()")
        add(lines, 8, "self._tinygrad_owned_static_cache_views = {}")
        add(lines, 8, "self._tinygrad_pending_static_cache_assign = None")
        add(lines, 8, "self._tinygrad_decode_cache_kv = None")
        add(lines, 8, "self._tinygrad_owned_decode = bool(int(os.environ.get('AXON_TINYGRAD_OWNED_DECODE', '0')))")
        add(lines, 8, "self.load_state_dict(state_dict)")
        add(lines, 4, "")
        add(lines, 4, "def enable_profile(self, enabled=True, *, cuda=True, reset=True):")
        add(lines, 8, "self._profile_enabled = bool(enabled)")
        add(lines, 8, "self._profile_cuda = bool(cuda)")
        add(lines, 8, "if reset:")
        add(lines, 12, "self._profile_records = {}")
        add(lines, 8, "return self")
        add(lines, 4, "")
        add(lines, 4, "def _profile_realize(self, value):")
        add(lines, 8, "if isinstance(value, Tensor):")
        add(lines, 12, "return value.realize()")
        add(lines, 8, "if isinstance(value, tuple):")
        add(lines, 12, "return tuple(self._profile_realize(item) for item in value)")
        add(lines, 8, "if isinstance(value, list):")
        add(lines, 12, "return [self._profile_realize(item) for item in value]")
        add(lines, 8, "if isinstance(value, dict):")
        add(lines, 12, "return {key: self._profile_realize(item) for key, item in value.items()}")
        add(lines, 8, "return value")
        add(lines, 4, "")
        add(lines, 4, "def _profile_call(self, name, fn, *args, **kwargs):")
        add(lines, 8, "if not self._profile_enabled:")
        add(lines, 12, "return fn(*args, **kwargs)")
        add(lines, 8, "use_cuda = bool(self._profile_cuda and torch.cuda.is_available())")
        add(lines, 8, "if use_cuda:")
        add(lines, 12, "torch.cuda.synchronize()")
        add(lines, 8, "start = time.perf_counter()")
        add(lines, 8, "try:")
        add(lines, 12, "result = fn(*args, **kwargs)")
        add(lines, 12, "result = self._profile_realize(result)")
        add(lines, 12, "return result")
        add(lines, 8, "finally:")
        add(lines, 12, "if use_cuda:")
        add(lines, 16, "torch.cuda.synchronize()")
        add(lines, 12, "elapsed = time.perf_counter() - start")
        add(lines, 12, "count, total = self._profile_records.get(name, (0, 0.0))")
        add(lines, 12, "self._profile_records[name] = (count + 1, total + elapsed)")
        add(lines, 4, "")
        add(lines, 4, "def profile_summary(self, top_n=40):")
        add(lines, 8, "rows = []")
        add(lines, 8, "for name, (count, total) in self._profile_records.items():")
        add(lines, 12, "rows.append({'name': name, 'count': count, 'seconds': total, 'avg_seconds': total / max(1, count)})")
        add(lines, 8, "rows.sort(key=lambda row: row['seconds'], reverse=True)")
        add(lines, 8, "return rows[: int(top_n)]")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def from_state_dict(cls, state_dict, *, graph=None, model_config=None, param_devices=None):")
        add(lines, 8, "return cls(state_dict, config=_MODEL_CONFIG if model_config is None else model_config, param_devices=param_devices)")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def from_safetensors(cls, safetensors_files, *, graph=None, model_config=None, dtype='float32', param_devices=None):")
        add(lines, 8, "state_dict = {}")
        add(lines, 8, "target_dtype = cls._dtype_from_name(dtype)")
        add(lines, 8, "target_device = 'CUDA' if torch.cuda.is_available() else 'CPU'")
        add(lines, 8, "for path in safetensors_files:")
        add(lines, 12, "loaded = tiny_state.safe_load(str(path))")
        add(lines, 12, "for key, value in loaded.items():")
        add(lines, 16, "if key in state_dict:")
        add(lines, 20, "raise ValueError(f'Duplicate tensor key while reading safetensors shards: {key}')")
        add(lines, 16, "if isinstance(value, Tensor):")
        add(lines, 20, "value = value.to(target_device)")
        add(lines, 16, "if target_dtype is not None and isinstance(value, Tensor) and dtypes.is_float(value.dtype):")
        add(lines, 20, "value = value.cast(target_dtype)")
        add(lines, 16, "state_dict[key] = value")
        add(lines, 8, "return cls(state_dict, config=_MODEL_CONFIG if model_config is None else model_config, param_devices=param_devices)")
        add(lines, 4, "")
        add(lines, 4, "def load_state_dict(self, state_dict, strict=True):")
        add(lines, 8, "del strict")
        add(lines, 8, "state_dict = self._materialize_state_aliases(dict(state_dict))")
        add(lines, 8, "self._tiny_device = self._infer_tiny_device(state_dict)")
        add(lines, 8, "self._torch_backing_tensors = []")
        add(lines, 8, "self.state_dict_tensors = {str(k): self._to_tiny(v) for k, v in state_dict.items()}")
        add(lines, 8, "self._embedding_aranges = {}")
        add(lines, 8, "self._forward_jits = {}")
        add(lines, 8, "self._forward_jit_seen = set()")
        add(lines, 8, "self._tinygrad_owned_static_cache_views = {}")
        add(lines, 8, "self._tinygrad_pending_static_cache_assign = None")
        add(lines, 8, "self._tinygrad_decode_cache_kv = None")
        add(lines, 8, "self.setup()")
        add(lines, 8, "self._symbols = self._eval_symbols()")
        add(lines, 8, "return self")
        add(lines, 4, "")
        add(lines, 4, "def setup(self):")
        add(lines, 8, "pass")
        add(lines, 8, "return None")
        add(lines, 4, "")
        add(lines, 4, "def to(self, device=None, *args, **kwargs):")
        add(lines, 8, "if device is not None:")
        add(lines, 12, "self._torch_device = torch.device(device)")
        add(lines, 12, "self._tiny_device = self._torch_to_tiny_device(self._torch_device)")
        add(lines, 12, "self.state_dict_tensors = {k: self._move_tiny(v, self._tiny_device) for k, v in self.state_dict_tensors.items()}")
        add(lines, 12, "self._embedding_aranges = {}")
        add(lines, 12, "self._forward_jits = {}")
        add(lines, 12, "self._forward_jit_seen = set()")
        add(lines, 12, "self.after_to()")
        add(lines, 8, "return self")
        add(lines, 4, "")
        add(lines, 4, "def enable_jit(self, enabled=True, reset=True):")
        add(lines, 8, "self._jit_enabled = bool(enabled)")
        add(lines, 8, "if reset:")
        add(lines, 12, "self._forward_jits = {}")
        add(lines, 12, "self._forward_jit_seen = set()")
        add(lines, 8, "return self")
        add(lines, 4, "")
        add(lines, 4, "def _jit_signature(self, value):")
        add(lines, 8, "if isinstance(value, Tensor):")
        add(lines, 12, "return ('tensor', tuple(value.shape), str(value.dtype), str(value.device))")
        add(lines, 8, "if isinstance(value, UOp):")
        add(lines, 12, "try:")
        add(lines, 16, "var, _ = value.unbind()")
        add(lines, 16, "return ('uop', var.expr, var.arg[1], var.arg[2], str(var.dtype))")
        add(lines, 12, "except Exception:")
        add(lines, 16, "return ('uop-expr', str(value))")
        add(lines, 8, "if isinstance(value, tuple):")
        add(lines, 12, "return ('tuple', tuple(self._jit_signature(item) for item in value))")
        add(lines, 8, "if isinstance(value, list):")
        add(lines, 12, "return ('list', tuple(self._jit_signature(item) for item in value))")
        add(lines, 8, "if isinstance(value, dict):")
        add(lines, 12, "return ('dict', tuple((key, self._jit_signature(item)) for key, item in sorted(value.items())))")
        add(lines, 8, "return ('value', value)")
        add(lines, 4, "")
        add(lines, 4, "def _collect_jit_uops(self, value, out=None):")
        add(lines, 8, "if out is None:")
        add(lines, 12, "out = []")
        add(lines, 8, "if isinstance(value, UOp):")
        add(lines, 12, "out.append(value)")
        add(lines, 8, "elif isinstance(value, (tuple, list)):")
        add(lines, 12, "for item in value:")
        add(lines, 16, "self._collect_jit_uops(item, out)")
        add(lines, 8, "elif isinstance(value, dict):")
        add(lines, 12, "for item in value.values():")
        add(lines, 16, "self._collect_jit_uops(item, out)")
        add(lines, 8, "return out")
        add(lines, 4, "")
        add(lines, 4, "def _jit_return_safe(self, value):")
        add(lines, 8, "if isinstance(value, UOp):")
        add(lines, 12, "return Tensor.empty((), dtype=dtypes.int64, device=self._tiny_device)")
        add(lines, 8, "if isinstance(value, tuple):")
        add(lines, 12, "return tuple(self._jit_return_safe(item) for item in value)")
        add(lines, 8, "if isinstance(value, list):")
        add(lines, 12, "return [self._jit_return_safe(item) for item in value]")
        add(lines, 8, "if isinstance(value, dict):")
        add(lines, 12, "return {key: self._jit_return_safe(item) for key, item in value.items()}")
        add(lines, 8, "return value")
        add(lines, 4, "")
        add(lines, 4, "def _forward_jit_entry(self, input_ids, **inputs):")
        add(lines, 8, "inputs = {k: v for k, v in inputs.items() if not k.startswith('__jit_uop_')}")
        add(lines, 8, "return self._jit_return_safe(self._forward(input_ids, **inputs))")
        add(lines, 4, "")
        add(lines, 4, "def _forward_maybe_jit(self, input_ids, **inputs):")
        add(lines, 8, "if input_ids is not None:")
        add(lines, 12, "input_ids = self._to_tiny(input_ids)")
        add(lines, 8, "inputs = {k: self._to_tiny(v) for k, v in inputs.items()}")
        add(lines, 8, "if not self._jit_enabled or self._profile_enabled:")
        add(lines, 12, "return self._forward(input_ids, **inputs)")
        add(lines, 8, "jit_uops = {f'__jit_uop_{i}': uop for i, uop in enumerate(self._collect_jit_uops((input_ids, inputs)))}")
        add(lines, 8, "signature = self._jit_signature((input_ids, inputs))")
        add(lines, 8, "if signature not in self._forward_jit_seen:")
        add(lines, 12, "self._forward_jit_seen.add(signature)")
        add(lines, 12, "return self._forward(input_ids, **inputs)")
        add(lines, 8, "jit = self._forward_jits.get(signature)")
        add(lines, 8, "if jit is None:")
        add(lines, 12, "jit = TinyJit(self._forward_jit_entry, prune=True)")
        add(lines, 12, "self._forward_jits[signature] = jit")
        add(lines, 8, "return jit(input_ids, **inputs, **jit_uops)")
        add(lines, 4, "")
        add(lines, 4, "def _forward_for_generate(self, input_ids, **inputs):")
        add(lines, 8, "return self._forward_maybe_jit(input_ids, **inputs)")
        add(lines, 4, "")
        add(lines, 4, "def after_to(self):")
        add(lines, 8, "return None")
        add(lines, 4, "")
        add(lines, 4, "def eval(self):")
        add(lines, 8, "return self")
        add(lines, 4, "")
        add(lines, 4, "def __call__(self, *args, **kwargs):")
        add(lines, 8, "return self.forward(*args, **kwargs)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _value(value):")
        add(lines, 8, "while isinstance(value, (tuple, list)) and len(value) == 1:")
        add(lines, 12, "value = value[0]")
        add(lines, 8, "return value")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _is_tiny(value):")
        add(lines, 8, "return isinstance(value, Tensor)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _materialize_state_aliases(state_dict):")
        add(lines, 8, "for key, value in list(state_dict.items()):")
        add(lines, 12, "flat_key = f'{key}_flat'")
        add(lines, 12, "if flat_key in state_dict:")
        add(lines, 16, "continue")
        add(lines, 12, "shape = tuple(getattr(value, 'shape', ()))")
        add(lines, 12, "if len(shape) == 2 and int(shape[0]) == 1:")
        add(lines, 16, "state_dict[flat_key] = value.reshape((int(shape[1]),))")
        add(lines, 8, "return state_dict")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _torch_to_tiny_dtype(dtype):")
        add(lines, 8, "if dtype is torch.float32: return dtypes.float32")
        add(lines, 8, "if dtype is torch.float16: return dtypes.float16")
        add(lines, 8, "if dtype is torch.bfloat16: return dtypes.bfloat16")
        add(lines, 8, "if dtype is torch.float64: return dtypes.float64")
        add(lines, 8, "if dtype is torch.int64: return dtypes.int64")
        add(lines, 8, "if dtype is torch.int32: return dtypes.int32")
        add(lines, 8, "if dtype is torch.int16: return dtypes.int16")
        add(lines, 8, "if dtype is torch.int8: return dtypes.int8")
        add(lines, 8, "if dtype is torch.uint8: return dtypes.uint8")
        add(lines, 8, "if dtype is torch.bool: return dtypes.bool")
        add(lines, 8, "raise TypeError(f'unsupported torch dtype for tinygrad backend: {dtype}')")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _torch_to_tiny_device(device):")
        add(lines, 8, "device = torch.device(device)")
        add(lines, 8, "if device.type == 'cuda':")
        add(lines, 12, "index = 0 if device.index is None else int(device.index)")
        add(lines, 12, "return 'CUDA' if index == 0 else f'CUDA:{index}'")
        add(lines, 8, "if device.type == 'cpu': return 'CPU'")
        add(lines, 8, "return str(device).upper()")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _infer_tiny_device(cls, state_dict):")
        add(lines, 8, "for value in dict(state_dict).values():")
        add(lines, 12, "if isinstance(value, Tensor):")
        add(lines, 16, "if isinstance(value.device, str) and value.device.upper().startswith('DISK:'):")
        add(lines, 20, "continue")
        add(lines, 16, "return value.device")
        add(lines, 12, "if torch.is_tensor(value):")
        add(lines, 16, "return cls._torch_to_tiny_device(value.device)")
        add(lines, 8, "return 'CUDA' if torch.cuda.is_available() else 'CPU'")
        add(lines, 4, "")
        add(lines, 4, "def _to_tiny(self, value):")
        add(lines, 8, "if isinstance(value, Tensor):")
        add(lines, 12, "return value if value.device == self._tiny_device else value.to(self._tiny_device)")
        add(lines, 8, "if torch.is_tensor(value):")
        add(lines, 12, "tensor = value.detach()")
        add(lines, 12, "if not tensor.is_contiguous():")
        add(lines, 16, "tensor = tensor.contiguous()")
        add(lines, 12, "tiny_device = self._torch_to_tiny_device(tensor.device)")
        add(lines, 12, "self._torch_backing_tensors.append(tensor)")
        add(lines, 12, "return Tensor.from_blob(tensor.data_ptr(), tuple(tensor.shape), dtype=self._torch_to_tiny_dtype(tensor.dtype), device=tiny_device).to(self._tiny_device)")
        add(lines, 8, "if isinstance(value, tuple):")
        add(lines, 12, "return tuple(self._to_tiny(item) for item in value)")
        add(lines, 8, "if isinstance(value, list):")
        add(lines, 12, "return [self._to_tiny(item) for item in value]")
        add(lines, 8, "return value")
        add(lines, 4, "")
        add(lines, 4, "def _move_tiny(self, value, device):")
        add(lines, 8, "if isinstance(value, Tensor):")
        add(lines, 12, "return value.to(device)")
        add(lines, 8, "if isinstance(value, tuple):")
        add(lines, 12, "return tuple(self._move_tiny(item, device) for item in value)")
        add(lines, 8, "if isinstance(value, list):")
        add(lines, 12, "return [self._move_tiny(item, device) for item in value]")
        add(lines, 8, "if isinstance(value, dict):")
        add(lines, 12, "return {k: self._move_tiny(v, device) for k, v in value.items()}")
        add(lines, 8, "return value")
        add(lines, 4, "")
        add(lines, 4, "def _to_torch(self, value):")
        add(lines, 8, "if isinstance(value, Tensor):")
        add(lines, 12, "self._flush_tinygrad_pending_cache_assign()")
        add(lines, 12, "return torch.from_numpy(value.numpy()).to(self._torch_device)")
        add(lines, 8, "if isinstance(value, tuple):")
        add(lines, 12, "return tuple(self._to_torch(item) for item in value)")
        add(lines, 8, "if isinstance(value, list):")
        add(lines, 12, "return [self._to_torch(item) for item in value]")
        add(lines, 8, "if isinstance(value, dict):")
        add(lines, 12, "return {k: self._to_torch(v) for k, v in value.items()}")
        add(lines, 8, "return value")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _move_to(cls, value, device):")
        add(lines, 8, "return value")
        add(lines, 4, "")
        add(lines, 4, "def _remember_owned_cache_view(self, view, root, kv, layer=None):")
        add(lines, 8, "if isinstance(view, Tensor) and isinstance(root, Tensor):")
        add(lines, 12, "self._tinygrad_owned_static_cache_views[id(view)] = {'root': root, 'kv': int(kv), 'layer': layer}")
        add(lines, 8, "return view")
        add(lines, 4, "")
        add(lines, 4, "def _owned_cache_tuple_from_kv(self, cache_kv, length):")
        add(lines, 8, "keys = cache_kv[0]")
        add(lines, 8, "values = cache_kv[1]")
        add(lines, 8, "self._remember_owned_cache_view(keys, cache_kv, 0)")
        add(lines, 8, "self._remember_owned_cache_view(values, cache_kv, 1)")
        add(lines, 8, "return (keys, values, length)")
        add(lines, 4, "")
        add(lines, 4, "def _owned_cache_view_info(self, value):")
        add(lines, 8, "return self._tinygrad_owned_static_cache_views.get(id(value))")
        add(lines, 4, "")
        add(lines, 4, "def _flush_tinygrad_pending_cache_assign(self):")
        add(lines, 8, "pending = self._tinygrad_pending_static_cache_assign")
        add(lines, 8, "if pending is None:")
        add(lines, 12, "return")
        add(lines, 8, "self._tinygrad_pending_static_cache_assign = None")
        add(lines, 8, "target = pending['x'].shrink(tuple((pending['start'], pending['end']) if i == pending['dim'] else None for i in range(len(pending['x'].shape))))")
        add(lines, 8, "target.assign(pending['src']).realize()")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _align_pair(left, right, *, prefer='right'):")
        add(lines, 8, "return left, right")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _binary_op(op, left, right, *, prefer='right'):")
        add(lines, 8, "if op == '+': return left + right")
        add(lines, 8, "if op == '-': return left - right")
        add(lines, 8, "if op == '*': return left * right")
        add(lines, 8, "if op == '/': return left / right")
        add(lines, 8, "if op == '%': return left % right")
        add(lines, 8, "if op == '<=': return left <= right")
        add(lines, 8, "if op == '<': return left < right")
        add(lines, 8, "if op == '>=': return left >= right")
        add(lines, 8, "if op == '>': return left > right")
        add(lines, 8, "if op == '==':")
        add(lines, 12, "if left is None or right is None: return left is right")
        add(lines, 12, "return left == right")
        add(lines, 8, "if op == '!=':")
        add(lines, 12, "if left is None or right is None: return left is not right")
        add(lines, 12, "return left != right")
        add(lines, 8, "raise NotImplementedError(f'unsupported binary op {op!r}')")
        add(lines, 4, "")
        add(lines, 4, "_compose_path = staticmethod(_common_compose_path)")
        add(lines, 4, "_render_path = staticmethod(_common_render_path)")
        add(lines, 4, "_require_value = staticmethod(_common_require_value)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _path_template_part(value):")
        add(lines, 8, "if isinstance(value, str) and value.startswith('@@'):")
        add(lines, 12, "return value[2:].strip('.')")
        add(lines, 8, "if isinstance(value, str) and value.startswith('@'):")
        add(lines, 12, "return value[1:].strip('.')")
        add(lines, 8, "return value")
        add(lines, 4, "")
        add(lines, 4, "def _param(self, path):")
        add(lines, 8, "key = str(path).lstrip('@')")
        add(lines, 8, "self._materialize_expert_bank_for_path(key)")
        add(lines, 8, "return _common_required_state_value(self.state_dict_tensors, path)")
        add(lines, 4, "")
        add(lines, 4, "def _optional_param(self, path):")
        add(lines, 8, "key = str(path).lstrip('@')")
        add(lines, 8, "self._materialize_expert_bank_for_path(key)")
        add(lines, 8, "return _common_optional_state_value(self.state_dict_tensors, path)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _collapse_one_numeric_segment(key):")
        add(lines, 8, "parts = str(key).split('.')")
        add(lines, 8, "for index, part in enumerate(parts):")
        add(lines, 12, "if part.isdigit():")
        add(lines, 16, "return '.'.join(parts[:index] + parts[index + 1:]), int(part), index")
        add(lines, 8, "return None")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _collapsed_numeric_segments(key):")
        add(lines, 8, "parts = str(key).split('.')")
        add(lines, 8, "return [('.'.join(parts[:index] + parts[index + 1:]), int(part), index) for index, part in enumerate(parts) if part.isdigit()]")
        add(lines, 4, "")
        add(lines, 4, "def _keys_for_collapsed_bank(self, bank_key):")
        add(lines, 8, "items = {}")
        add(lines, 8, "numeric_index = None")
        add(lines, 8, "for key in self.state_dict_tensors:")
        add(lines, 12, "for collapsed_key, expert, index in self._collapsed_numeric_segments(str(key)):")
        add(lines, 16, "if collapsed_key != bank_key:")
        add(lines, 20, "continue")
        add(lines, 16, "if numeric_index is None:")
        add(lines, 20, "numeric_index = index")
        add(lines, 16, "elif numeric_index != index:")
        add(lines, 20, "continue")
        add(lines, 16, "items[expert] = str(key)")
        add(lines, 16, "break")
        add(lines, 8, "if not items:")
        add(lines, 12, "return []")
        add(lines, 8, "ordered = [items[i] for i in range(len(items)) if i in items]")
        add(lines, 8, "return ordered if len(ordered) == len(items) else []")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _fused_gate_up_source_bank_keys(bank_key):")
        add(lines, 8, "parts = str(bank_key).split('.')")
        add(lines, 8, "for index, part in enumerate(parts):")
        add(lines, 12, "if 'gate_up' not in part:")
        add(lines, 16, "continue")
        add(lines, 12, "gate_parts = list(parts)")
        add(lines, 12, "up_parts = list(parts)")
        add(lines, 12, "gate_parts[index] = part.replace('gate_up', 'gate', 1)")
        add(lines, 12, "up_parts[index] = part.replace('gate_up', 'up', 1)")
        add(lines, 12, "return '.'.join(gate_parts), '.'.join(up_parts)")
        add(lines, 8, "return None")
        add(lines, 4, "")
        add(lines, 4, "def _materialize_expert_bank_for_path(self, bank_key):")
        add(lines, 8, "existing = self.state_dict_tensors.get(bank_key)")
        add(lines, 8, "if isinstance(existing, Tensor):")
        add(lines, 12, "return existing")
        add(lines, 8, "ordered_keys = self._keys_for_collapsed_bank(bank_key)")
        add(lines, 8, "if ordered_keys:")
        add(lines, 12, "ordered = [self.state_dict_tensors[key] for key in ordered_keys]")
        add(lines, 12, "first_shape = ordered[0].shape")
        add(lines, 12, "if all(isinstance(t, Tensor) and t.shape == first_shape for t in ordered):")
        add(lines, 16, "bank = ordered[0].stack(*ordered[1:], dim=0) if len(ordered) > 1 else ordered[0].reshape((1,) + tuple(ordered[0].shape))")
        add(lines, 16, "for key in ordered_keys:")
        add(lines, 20, "self.state_dict_tensors.pop(key, None)")
        add(lines, 16, "self.state_dict_tensors[bank_key] = bank")
        add(lines, 16, "return bank")
        add(lines, 8, "fused_sources = self._fused_gate_up_source_bank_keys(bank_key)")
        add(lines, 8, "if fused_sources is None:")
        add(lines, 12, "return None")
        add(lines, 8, "gate_key, up_key = fused_sources")
        add(lines, 8, "gate = self._materialize_expert_bank_for_path(gate_key)")
        add(lines, 8, "up = self._materialize_expert_bank_for_path(up_key)")
        add(lines, 8, "if not isinstance(gate, Tensor) or not isinstance(up, Tensor):")
        add(lines, 12, "return None")
        add(lines, 8, "if gate.shape[:-2] != up.shape[:-2] or gate.shape[-1:] != up.shape[-1:]:")
        add(lines, 12, "return None")
        add(lines, 8, "concat_dim = -2 if len(gate.shape) >= 2 else -1")
        add(lines, 8, "bank = gate.cat(up, dim=concat_dim)")
        add(lines, 8, "self.state_dict_tensors[bank_key] = bank")
        add(lines, 8, "return bank")
        add(lines, 4, "")
        add(lines, 4, "def _config(self, path, default=None):")
        add(lines, 8, "return _common_config_value(self.config, path, default)")
        add(lines, 4, "")
        add(lines, 4, "def _has_config(self, path):")
        add(lines, 8, "return _common_has_config_value(self.config, path)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _dtype_from_name(value):")
        add(lines, 8, "if value is None: return None")
        add(lines, 8, "token = str(value).strip().lower()")
        add(lines, 8, "if token in ('', 'none', 'null', 'default'): return None")
        add(lines, 8, "if token in ('float32', 'fp32', 'single'): return dtypes.float32")
        add(lines, 8, "if token in ('float16', 'fp16', 'half'): return dtypes.float16")
        add(lines, 8, "if token in ('bfloat16', 'bf16'): return dtypes.bfloat16")
        add(lines, 8, "if token in ('int64', 'long'): return dtypes.int64")
        add(lines, 8, "if token in ('int32', 'int'): return dtypes.int32")
        add(lines, 8, "if token in ('bool', 'boolean'): return dtypes.bool")
        add(lines, 8, "raise ValueError(f'unsupported dtype name {value!r}')")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _dtype_value(dtype, kind):")
        add(lines, 8, "token = str(dtype)")
        add(lines, 8, "if 'float16' in token or 'half' in token:")
        add(lines, 12, "values = {'min': -65504.0, 'max': 65504.0, 'eps': 0.0009765625, 'tiny': 6.103515625e-05}")
        add(lines, 8, "elif 'float64' in token or 'double' in token:")
        add(lines, 12, "values = {'min': -1.7976931348623157e308, 'max': 1.7976931348623157e308, 'eps': 2.220446049250313e-16, 'tiny': 2.2250738585072014e-308}")
        add(lines, 8, "else:")
        add(lines, 12, "values = {'min': -3.4028234663852886e38, 'max': 3.4028234663852886e38, 'eps': 1.1920928955078125e-07, 'tiny': 1.1754943508222875e-38}")
        add(lines, 8, "values = {**values, 'inf': float('inf'), '-inf': float('-inf')}")
        add(lines, 8, "return float(values[str(kind)])")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _binary_add(cls, left, right):")
        add(lines, 8, "if left is None: return right")
        add(lines, 8, "if right is None: return left")
        add(lines, 8, "return left + right")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _binary_sub(cls, left, right):")
        add(lines, 8, "if right is None: return left")
        add(lines, 8, "return left - right")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _eq(left, right):")
        add(lines, 8, "if left is None or right is None:")
        add(lines, 12, "return left is right")
        add(lines, 8, "return left == right")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _where(cls, cond, yes, no):")
        add(lines, 8, "if not isinstance(cond, Tensor):")
        add(lines, 12, "return yes if cond else no")
        add(lines, 8, "return cond.where(yes, no)")
        add(lines, 4, "")
        add(lines, 4, "def _where_indices(self, x):")
        add(lines, 8, "indices = x.numpy().nonzero()")
        add(lines, 8, "return tuple(Tensor(index.astype('int64'), dtype=dtypes.int64, device=self._tiny_device) for index in indices)")
        add(lines, 4, "")
        add(lines, 4, "def _embedding(self, base, ids):")
        add(lines, 8, "weight_path = self._compose_path(base, 'weight')")
        add(lines, 8, "weight = self._param(weight_path)")
        add(lines, 8, "if not isinstance(ids, Tensor):")
        add(lines, 12, "ids = Tensor(ids, dtype=dtypes.int64, device=weight.device)")
        add(lines, 8, "elif not dtypes.is_int(ids.dtype):")
        add(lines, 12, "ids = ids.cast(dtypes.int64)")
        add(lines, 8, "ids = ids.to(weight.device)")
        add(lines, 8, "vocab_size, embed_size = int(weight.shape[0]), int(weight.shape[1])")
        add(lines, 8, "cache_key = (str(weight_path), str(weight.device), vocab_size)")
        add(lines, 8, "arange = self._embedding_aranges.get(cache_key)")
        add(lines, 8, "if arange is None:")
        add(lines, 12, "arange = Tensor.arange(vocab_size, requires_grad=False, device=weight.device).unsqueeze(-1)")
        add(lines, 12, "self._embedding_aranges[cache_key] = arange")
        add(lines, 8, "big_shape = tuple(ids.shape) + (vocab_size, embed_size)")
        add(lines, 8, "idx = ids.reshape(tuple(ids.shape) + (1, 1)).expand(big_shape)")
        add(lines, 8, "vals = weight.expand(big_shape)")
        add(lines, 8, "return (arange.expand(big_shape) == idx).where(vals, 0).sum(-2, dtype=vals.dtype)")
        add(lines, 4, "")
        add(lines, 4, "def _slice(self, x, dim, start, end):")
        add(lines, 8, "dim = int(dim)")
        add(lines, 8, "if dim < 0:")
        add(lines, 12, "dim += len(x.shape)")
        add(lines, 8, "raw_start, raw_end = start, end")
        add(lines, 8, "if not isinstance(start, UOp):")
        add(lines, 12, "start = int(start)")
        add(lines, 8, "if not isinstance(end, UOp):")
        add(lines, 12, "end = int(end)")
        add(lines, 8, "view = x.shrink(tuple((start, end) if i == dim else None for i in range(len(x.shape))))")
        add(lines, 8, "info = self._owned_cache_view_info(x)")
        add(lines, 8, "if info is not None and dim == 0 and not isinstance(raw_start, UOp) and not isinstance(raw_end, UOp) and int(end) == int(start) + 1:")
        add(lines, 12, "self._remember_owned_cache_view(view, info['root'], info['kv'], int(start))")
        add(lines, 8, "return view")
        add(lines, 4, "")
        add(lines, 4, "def _reshape(self, x, shape):")
        add(lines, 8, "view = x.reshape(tuple(int(item) for item in shape))")
        add(lines, 8, "info = self._owned_cache_view_info(x)")
        add(lines, 8, "if info is not None:")
        add(lines, 12, "self._remember_owned_cache_view(view, info['root'], info['kv'], info.get('layer'))")
        add(lines, 8, "return view")
        add(lines, 4, "")
        add(lines, 4, "def _assign_slice(self, x, src, dim, start, end):")
        add(lines, 8, "dim = int(dim)")
        add(lines, 8, "if dim < 0:")
        add(lines, 12, "dim += len(x.shape)")
        add(lines, 8, "if isinstance(src, Tensor) and src.device != x.device:")
        add(lines, 12, "src = src.to(x.device)")
        add(lines, 8, "if self._tinygrad_mutable_slices:")
        add(lines, 12, "info = self._owned_cache_view_info(x)")
        add(lines, 12, "if info is not None and info.get('layer') is not None and dim == len(x.shape) - 2:")
        add(lines, 16, "if not isinstance(start, UOp): start = int(start)")
        add(lines, 16, "if not isinstance(end, UOp): end = int(end)")
        add(lines, 16, "pending = self._tinygrad_pending_static_cache_assign")
        add(lines, 16, "key = (id(info['root']), int(info['layer']), dim, start, end)")
        add(lines, 16, "if int(info['kv']) == 0:")
        add(lines, 20, "if pending is not None:")
        add(lines, 24, "self._flush_tinygrad_pending_cache_assign()")
        add(lines, 20, "self._tinygrad_pending_static_cache_assign = {'key': key, 'x': x, 'src': src, 'dim': dim, 'start': start, 'end': end}")
        add(lines, 20, "return x")
        add(lines, 16, "if int(info['kv']) == 1 and pending is not None and pending.get('key') == key:")
        add(lines, 20, "root = info['root']")
        add(lines, 20, "layer = int(info['layer'])")
        add(lines, 20, "self._tinygrad_pending_static_cache_assign = None")
        add(lines, 20, "target_slice = []")
        add(lines, 20, "for axis in range(len(root.shape)):")
        add(lines, 24, "if axis == 0: target_slice.append((0, 2))")
        add(lines, 24, "elif axis == 1: target_slice.append((layer, layer + 1))")
        add(lines, 24, "elif axis == dim + 2: target_slice.append((start, end))")
        add(lines, 24, "else: target_slice.append(None)")
        add(lines, 20, "packed_src = Tensor.stack(pending['src'], src, dim=0).unsqueeze(1)")
        add(lines, 20, "root.shrink(tuple(target_slice)).assign(packed_src).realize()")
        add(lines, 20, "return x")
        add(lines, 16, "if pending is not None:")
        add(lines, 20, "self._flush_tinygrad_pending_cache_assign()")
        add(lines, 12, "if self._tinygrad_pending_static_cache_assign is not None:")
        add(lines, 16, "self._flush_tinygrad_pending_cache_assign()")
        add(lines, 12, "if id(x) in self._tinygrad_owned_static_cache_roots and dim == 0 and not isinstance(start, UOp) and not isinstance(end, UOp) and int(end) == int(start) + 1:")
        add(lines, 16, "# StaticCache.static_put writes the layer view back into the")
        add(lines, 16, "# root cache. In owned decode mode the view update already")
        add(lines, 16, "# mutated the root buffer, matching tinygrad's native cache.")
        add(lines, 16, "return x")
        add(lines, 12, "target = x.shrink(tuple((start, end) if i == dim else None for i in range(len(x.shape))))")
        add(lines, 12, "target.assign(src).realize()")
        add(lines, 12, "return x")
        add(lines, 8, "if isinstance(start, UOp) or isinstance(end, UOp):")
        add(lines, 12, "if int(src.shape[dim]) != 1:")
        add(lines, 16, "raise NotImplementedError('symbolic _assign_slice currently requires unit source extent')")
        add(lines, 12, "idx_shape = [1] * len(x.shape)")
        add(lines, 12, "idx_shape[dim] = int(x.shape[dim])")
        add(lines, 12, "idx = Tensor.arange(0, int(x.shape[dim]), dtype=dtypes.int64, device=x.device).reshape(tuple(idx_shape))")
        add(lines, 12, "mask = (idx >= start) & (idx < end)")
        add(lines, 12, "return mask.where(src, x)")
        add(lines, 8, "start = int(start)")
        add(lines, 8, "end = int(end)")
        add(lines, 8, "parts = []")
        add(lines, 8, "if start > 0:")
        add(lines, 12, "parts.append(x.shrink(tuple((0, start) if i == dim else None for i in range(len(x.shape)))))")
        add(lines, 8, "parts.append(src)")
        add(lines, 8, "if end < int(x.shape[dim]):")
        add(lines, 12, "parts.append(x.shrink(tuple((end, int(x.shape[dim])) if i == dim else None for i in range(len(x.shape)))))")
        add(lines, 8, "first, *rest = parts")
        add(lines, 8, "return first if not rest else first.cat(*rest, dim=dim)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _concat(*args, dim=None):")
        add(lines, 8, "if dim is None:")
        add(lines, 12, "*items, dim = args")
        add(lines, 8, "else:")
        add(lines, 12, "items = args")
        add(lines, 8, "if len(items) == 1 and isinstance(items[0], (list, tuple)):")
        add(lines, 12, "items = tuple(items[0])")
        add(lines, 8, "first, *rest = items")
        add(lines, 8, "return first.cat(*rest, dim=int(dim))")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _unsqueeze(x, dim):")
        add(lines, 8, "rank = len(x.shape)")
        add(lines, 8, "dim = int(dim)")
        add(lines, 8, "if dim < 0:")
        add(lines, 12, "dim += rank + 1")
        add(lines, 8, "return x.reshape(x.shape[:dim] + (1,) + x.shape[dim:])")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _topk(x, k, dim=-1, largest=True, sorted_=True):")
        add(lines, 8, "# tinygrad does not support sorted_=False. Sorted top-k has the same")
        add(lines, 8, "# selected value/index pairs, which is sufficient for Axon's top-k users.")
        add(lines, 8, "return x.topk(int(k), dim=int(dim), largest=bool(largest), sorted_=True)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _index_add(x, index, src, dim=0):")
        add(lines, 8, "dim = int(dim)")
        add(lines, 8, "if dim < 0:")
        add(lines, 12, "dim += len(x.shape)")
        add(lines, 8, "if isinstance(index, Tensor) and len(index.shape) != len(src.shape):")
        add(lines, 12, "shape = [1] * len(src.shape)")
        add(lines, 12, "shape[dim] = index.shape[0]")
        add(lines, 12, "index = index.reshape(tuple(shape)).expand(src.shape)")
        add(lines, 8, "return x.scatter_reduce(dim, index, src, reduce='sum', include_self=True)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _linear(base, x, bias=False, transpose=False, expert=None, weight_leaf='weight', bias_leaf='bias'):")
        add(lines, 8, "raise RuntimeError('internal tinygrad _linear helper should not be called directly')")
        add(lines, 4, "")
        add(lines, 4, "def _expert_linear(self, base, x, expert_idx, bias=False, transpose=False, weight_leaf='weight', bias_leaf='bias'):")
        add(lines, 8, "weight = self._param(self._compose_path(base, weight_leaf))")
        add(lines, 8, "idx = expert_idx.cast(dtypes.int64) if isinstance(expert_idx, Tensor) else expert_idx")
        add(lines, 8, "selected_weight = weight[idx]")
        add(lines, 8, "bias_value = self._optional_param(self._compose_path(base, bias_leaf)) if bias else None")
        add(lines, 8, "selected_bias = bias_value[idx] if bias_value is not None else None")
        add(lines, 8, "weight_run = selected_weight.cast(x.dtype) if dtypes.is_float(x.dtype) and dtypes.is_float(selected_weight.dtype) and x.dtype != selected_weight.dtype else selected_weight")
        add(lines, 8, "bias_run = selected_bias.cast(x.dtype) if selected_bias is not None and dtypes.is_float(x.dtype) and dtypes.is_float(selected_bias.dtype) and x.dtype != selected_bias.dtype else selected_bias")
        add(lines, 8, "if transpose:")
        add(lines, 12, "y = (x.unsqueeze(-2) @ weight_run).squeeze(-2)")
        add(lines, 8, "else:")
        add(lines, 12, "y = (x.unsqueeze(-2) @ weight_run.transpose(-1, -2)).squeeze(-2)")
        add(lines, 8, "return y + bias_run if bias_run is not None else y")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _gegelu(x, limit=None):")
        add(lines, 8, "if x.shape[-1] % 2 != 0: raise ValueError('gegelu requires even last dimension')")
        add(lines, 8, "x_gelu = x[..., ::2]")
        add(lines, 8, "x_linear = x[..., 1::2]")
        add(lines, 8, "if limit is not None:")
        add(lines, 12, "limit = float(limit)")
        add(lines, 12, "x_gelu = x_gelu.minimum(limit)")
        add(lines, 12, "x_linear = x_linear.maximum(-limit).minimum(limit)")
        add(lines, 8, "return x_gelu * (1.702 * x_gelu).sigmoid() * (x_linear + 1.0)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _xielu(x, alpha_p_raw, alpha_n_raw, beta_raw, eps_raw):")
        add(lines, 8, "beta = alpha_p_raw.full_like(float(beta_raw)) if not isinstance(beta_raw, Tensor) else beta_raw")
        add(lines, 8, "alpha_p = alpha_p_raw.softplus() if isinstance(alpha_p_raw, Tensor) else Tensor(float(alpha_p_raw), device=x.device).softplus()")
        add(lines, 8, "alpha_n_base = alpha_n_raw.softplus() if isinstance(alpha_n_raw, Tensor) else Tensor(float(alpha_n_raw), device=x.device).softplus()")
        add(lines, 8, "alpha_n = beta + alpha_n_base")
        add(lines, 8, "eps = float(eps_raw.item()) if isinstance(eps_raw, Tensor) else float(eps_raw)")
        add(lines, 8, "return (x > 0).where(alpha_p * x * x + beta * x, ((x.minimum(eps).exp() - 1.0) - x) * alpha_n + beta * x)")

    def _emit_forward(self, lines: list[str]) -> None:
        main = self.modules_by_name[self.program.main_module]
        add = self._add
        add(lines, 4, "def _forward(self, input_ids=None, **inputs):")
        add(lines, 8, "if input_ids is not None:")
        add(lines, 12, "input_ids = self._to_tiny(input_ids)")
        add(lines, 8, "inputs = {k: self._to_tiny(v) for k, v in inputs.items()}")
        args: list[str] = []
        static_attention_inputs = {
            value.name
            for value in main.inputs
            if value.name in {"attn_mask", "attention_mask", "decoder_attention_mask"}
            and _is_static_mask_type(value.type_expr)
        }
        first_input = main.inputs[0].name if main.inputs else None
        for value in main.inputs:
            if value.name == "input_ids":
                add(lines, 8, "if input_ids is None:")
                add(lines, 12, "input_ids = inputs.get('input_ids')")
                add(lines, 8, "if input_ids is None:")
                add(lines, 12, "raise ValueError('Missing required input: input_ids')")
                args.append("input_ids")
            elif value.name == first_input:
                add(lines, 8, f"{value.name} = inputs.get({value.name!r}, input_ids)")
                if not (value.optional or isinstance(value.type_expr, TypeOptional)):
                    add(lines, 8, f"if {value.name} is None:")
                    add(lines, 12, f"raise ValueError('Missing required input: {value.name}')")
                args.append(value.name)
            else:
                if value.optional or isinstance(value.type_expr, TypeOptional):
                    add(lines, 8, f"{value.name} = inputs.get({value.name!r}, None)")
                else:
                    add(lines, 8, f"if {value.name!r} not in inputs:")
                    add(lines, 12, f"raise ValueError('Missing required input: {value.name}')")
                    add(lines, 8, f"{value.name} = inputs[{value.name!r}]")
                if value.name in static_attention_inputs:
                    add(lines, 8, f"if isinstance({value.name}, Tensor):")
                    add(lines, 12, f"__static_len_{value.name} = int({value.name}.shape[1])")
                    add(lines, 12, f"__static_capacity_{value.name} = max(int(self.config.get('n_positions', self.config.get('max_position_embeddings', __static_len_{value.name}))), __static_len_{value.name})")
                    add(lines, 12, f"__static_store_{value.name} = Tensor.zeros({value.name}.shape[0], __static_capacity_{value.name}, dtype={value.name}.dtype, device={value.name}.device)")
                    add(lines, 12, f"__static_store_{value.name} = self._assign_slice(__static_store_{value.name}, {value.name}, 1, 0, __static_len_{value.name})")
                    add(lines, 12, f"{value.name} = (__static_store_{value.name}, __static_len_{value.name})")
                args.append(value.name)
        add(lines, 8, f"result = self.{self.method_names[main.name]}({', '.join(args)})")
        names = graph_main_output_names(self.program, main)
        if len(names) == 1:
            add(lines, 8, "return result[0]")
        else:
            add(lines, 8, f"return {{{', '.join(f'{name!r}: result[{idx}]' for idx, name in enumerate(names))}}}")
        add(lines, 4, "")
        add(lines, 4, "def forward(self, input_ids=None, **inputs):")
        add(lines, 8, "return self._to_torch(self._forward_maybe_jit(input_ids, **inputs))")

    def _emit_generate(self, lines: list[str]) -> None:
        add = self._add
        main = self.modules_by_name[self.program.main_module]
        input_names = {value.name for value in main.inputs}
        output_names = set(graph_main_output_names(self.program, main))
        attention_name = "attn_mask" if "attn_mask" in input_names else (
            "attention_mask" if "attention_mask" in input_names else None
        )
        attention_value = next((value for value in main.inputs if value.name == attention_name), None)
        uses_static_attention_mask = (
            attention_value is not None and _is_static_mask_type(attention_value.type_expr)
        )
        decoder_attention_name = "decoder_attention_mask" if "decoder_attention_mask" in input_names else None
        cache_name = "past_kv" if "past_kv" in input_names else (
            "past_cache" if "past_cache" in input_names else None
        )
        cache_input = next((value for value in main.inputs if value.name == cache_name), None)
        cache_type = cache_input.type_expr if cache_input is not None else None
        cache_inner_type = cache_type.inner if isinstance(cache_type, TypeOptional) else cache_type
        has_static_cache_shape = cache_name is not None and cache_inner_type is not None and not isinstance(cache_inner_type, TypeList)
        cache_output_name = "new_kv" if "new_kv" in output_names else (
            "past_kv" if "past_kv" in output_names else ("cache" if "cache" in output_names else None)
        )
        use_cache_name = "use_cache" if "use_cache" in input_names else None
        has_decoder_inputs = "decoder_input_ids" in input_names
        # tinygrad's TinyJit needs stable input shapes for reuse. Only static
        # cache inputs have stable tensor shapes; list caches grow every token.
        is_cached_decoder = (
            not has_decoder_inputs
            and cache_name is not None
            and cache_output_name is not None
            and has_static_cache_shape
        )
        is_decoder_only = not has_decoder_inputs
        add(lines, 4, "def generate(self, input_ids, max_new_tokens=20, **kwargs):")
        add(lines, 8, "input_ids = self._to_tiny(input_ids)")
        add(lines, 8, "kwargs = {k: self._to_tiny(v) for k, v in kwargs.items()}")
        add(lines, 8, "def _logits(result):")
        add(lines, 12, "return result.get('logits') if isinstance(result, dict) else result")
        add(lines, 8, "def _next_id(logits):")
        add(lines, 12, "return logits[:, -1, :].argmax(axis=-1).reshape((logits.shape[0], 1)).cast(dtypes.int64)")
        add(lines, 8, "def _ones_like_ids(ids):")
        add(lines, 12, "return Tensor.ones(*ids.shape, dtype=dtypes.int64, device=self._tiny_device)")
        add(lines, 8, "def _static_attention_mask(mask, prompt_ids, capacity):")
        add(lines, 12, "valid = _ones_like_ids(prompt_ids) if mask is None else mask")
        add(lines, 12, "# Generated tokens are always valid; the length field still")
        add(lines, 12, "# controls which columns are visible at each decode step.")
        add(lines, 12, "store = Tensor.ones(valid.shape[0], int(capacity), dtype=valid.dtype, device=valid.device)")
        add(lines, 12, "length = int(valid.shape[1])")
        add(lines, 12, "store = self._assign_slice(store, valid, 1, 0, length)")
        add(lines, 12, "return (store, length)")
        add(lines, 8, "def _append_static_attention_mask(mask, next_id):")
        add(lines, 12, "store, length = mask")
        add(lines, 12, "src = _ones_like_ids(next_id).cast(store.dtype)")
        add(lines, 12, "store = self._assign_slice(store, src, 1, length, length + 1)")
        add(lines, 12, "return (store, length + 1)")
        add(lines, 8, "def _static_tuple_length(value, default=0):")
        add(lines, 12, "if value is None: return int(default)")
        add(lines, 12, "try: return int(value[-1])")
        add(lines, 12, "except Exception: return int(default)")
        add(lines, 8, "def _with_static_tuple_length(value, length):")
        add(lines, 12, "if value is None: return None")
        add(lines, 12, "return tuple(value[:-1]) + (int(length),)")
        add(lines, 8, "def _realize_static_tuple(value):")
        add(lines, 12, "if value is None: return None")
        add(lines, 12, "return tuple(item.realize() if isinstance(item, Tensor) else item for item in value)")
        add(lines, 8, "def _pack_owned_static_cache(value):")
        add(lines, 12, "if value is None: return None, None")
        add(lines, 12, "value = _realize_static_tuple(value)")
        add(lines, 12, "packed = Tensor.stack(value[0], value[1], dim=0).contiguous().realize()")
        add(lines, 12, "self._tinygrad_decode_cache_kv = packed")
        add(lines, 12, "return self._owned_cache_tuple_from_kv(packed, value[-1]), packed")
        add(lines, 8, "def _bind_static_tuple_length(value, var, length):")
        add(lines, 12, "if value is None: return None")
        add(lines, 12, "return tuple(value[:-1]) + (var.bind(int(length)),)")
        add(lines, 8, "def _maybe_bind_static_tuple_length(value, var, length):")
        add(lines, 12, "# Symbolic length UOps are only safe through the decode-step JIT")
        add(lines, 12, "# wrapper, where they are explicit top-level TinyJit inputs.")
        add(lines, 12, "return value")
        add(lines, 8, "def _generation_limit(prompt_ids):")
        add(lines, 12, "requested = kwargs.pop('max_new_tokens', None)")
        add(lines, 12, "if requested is not None: return int(requested)")
        add(lines, 12, "max_len = kwargs.pop('max_len', None)")
        add(lines, 12, "return int(max_new_tokens) if max_len is None else max(1, int(max_len) - int(prompt_ids.shape[1]))")
        add(lines, 8, "def _eos_state(batch_size):")
        add(lines, 12, "eos_token_id = kwargs.pop('eos_token_id', self.config.get('eos_token_id', None))")
        add(lines, 12, "pad_token_id = kwargs.pop('pad_token_id', eos_token_id)")
        add(lines, 12, "if eos_token_id is None: return None, None, None")
        add(lines, 12, "eos_values = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)")
        add(lines, 12, "eos = Tensor(eos_values, dtype=dtypes.int64, device=self._tiny_device).reshape((-1,))")
        add(lines, 12, "pad = int(eos_values[0] if pad_token_id is None else pad_token_id)")
        add(lines, 12, "finished = Tensor.zeros(batch_size, 1, dtype=dtypes.bool, device=self._tiny_device)")
        add(lines, 12, "return eos, pad, finished")
        add(lines, 8, "def _apply_eos(next_id, eos, pad, finished):")
        add(lines, 12, "if eos is None: return next_id, finished")
        add(lines, 12, "raw_next = next_id")
        add(lines, 12, "next_id = finished.where(Tensor.full(next_id.shape, pad, dtype=dtypes.int64, device=self._tiny_device), next_id)")
        add(lines, 12, "hit = (raw_next == eos.reshape((1, -1))).max(axis=1, keepdim=True)")
        add(lines, 12, "finished = finished | hit")
        add(lines, 12, "return next_id.realize(), finished.realize()")
        add(lines, 8, "def _all_done(finished):")
        add(lines, 12, "return finished is not None and bool(finished.min().item())")
        if is_cached_decoder:
            add(lines, 8, "out = input_ids")
            add(lines, 8, "last_token = None")
            add(lines, 8, "limit = _generation_limit(out)")
            add(lines, 8, "eos, pad, finished = _eos_state(out.shape[0])")
            add(lines, 8, f"cache = kwargs.pop({cache_name!r}, None)")
            add(lines, 8, "cache_length = _static_tuple_length(cache, 0)")
            if has_static_cache_shape:
                add(lines, 8, "cache_owned_realized = False")
                add(lines, 8, "owned_cache_kv = None")
            if attention_name is not None:
                other = "attention_mask" if attention_name == "attn_mask" else "attn_mask"
                add(lines, 8, f"attention_mask = kwargs.pop({attention_name!r}, kwargs.pop({other!r}, None))")
                add(lines, 8, "if attention_mask is None: attention_mask = _ones_like_ids(out)")
                if uses_static_attention_mask:
                    add(lines, 8, "static_capacity = max(int(self.config.get('n_positions', self.config.get('max_position_embeddings', out.shape[1] + limit))), int(out.shape[1]) + int(limit))")
                    add(lines, 8, "attention_mask = _static_attention_mask(attention_mask, out, static_capacity)")
                    add(lines, 8, "attention_length = _static_tuple_length(attention_mask, int(out.shape[1]))")
                    add(lines, 8, "attention_length_var = Variable('attention_length', 0, int(static_capacity))")
            if has_static_cache_shape:
                add(lines, 8, "static_cache_capacity = int(static_capacity) if 'static_capacity' in locals() else int(self.config.get('n_positions', self.config.get('max_position_embeddings', out.shape[1] + limit)))")
                add(lines, 8, "cache_length_var = Variable('cache_length', 0, int(static_cache_capacity))")
            if use_cache_name is not None:
                add(lines, 8, f"kwargs.pop({use_cache_name!r}, None)")
            if has_static_cache_shape and uses_static_attention_mask and attention_name is not None:
                add(lines, 8, "if not kwargs:")
                add(lines, 12, "decode_jit_key = ('generate_static_decode', bool(self._tinygrad_owned_decode))")
                add(lines, 12, "decode_step_jit = self._forward_jits.get(decode_jit_key)")
                add(lines, 12, "if decode_step_jit is None:")
                add(lines, 16, "if self._tinygrad_owned_decode:")
                add(lines, 20, "def _decode_step_jit_entry(step_ids, attention_store, cache_len_bound, attention_len_bound):")
                add(lines, 24, "step_inputs = {}")
                add(lines, 24, "cache_kv = self._tinygrad_decode_cache_kv")
                add(lines, 24, f"step_inputs[{cache_name!r}] = self._owned_cache_tuple_from_kv(cache_kv, cache_len_bound)")
                if use_cache_name is not None:
                    add(lines, 24, f"step_inputs[{use_cache_name!r}] = True")
                add(lines, 24, f"step_inputs[{attention_name!r}] = (attention_store, attention_len_bound)")
                add(lines, 24, "step_result = self._forward(step_ids, **step_inputs)")
                add(lines, 24, "return _next_id(step_result['logits'])")
                add(lines, 16, "else:")
                add(lines, 20, "def _decode_step_jit_entry(step_ids, cache_keys, cache_values, attention_store, cache_len_bound, attention_len_bound):")
                add(lines, 24, "step_inputs = {}")
                add(lines, 24, f"step_inputs[{cache_name!r}] = (cache_keys, cache_values, cache_len_bound)")
                if use_cache_name is not None:
                    add(lines, 24, f"step_inputs[{use_cache_name!r}] = True")
                add(lines, 24, f"step_inputs[{attention_name!r}] = (attention_store, attention_len_bound)")
                add(lines, 24, "step_result = self._forward(step_ids, **step_inputs)")
                add(lines, 24, f"step_cache = step_result.get({cache_output_name!r}) if isinstance(step_result, dict) else None")
                add(lines, 24, "return _next_id(step_result['logits']), step_cache[0], step_cache[1]")
                add(lines, 16, "decode_step_jit = TinyJit(_decode_step_jit_entry, prune=True)")
                add(lines, 16, "self._forward_jits[decode_jit_key] = decode_step_jit")
                add(lines, 8, "else:")
                add(lines, 12, "if self._tinygrad_owned_decode:")
                add(lines, 16, "def _decode_step_jit_entry(step_ids, attention_store, cache_len_bound, attention_len_bound):")
                add(lines, 20, "step_kwargs = dict(kwargs)")
                add(lines, 20, "cache_kv = self._tinygrad_decode_cache_kv")
                add(lines, 20, f"step_kwargs[{cache_name!r}] = self._owned_cache_tuple_from_kv(cache_kv, cache_len_bound)")
                if use_cache_name is not None:
                    add(lines, 20, f"step_kwargs[{use_cache_name!r}] = True")
                add(lines, 20, f"step_kwargs[{attention_name!r}] = (attention_store, attention_len_bound)")
                add(lines, 20, "step_result = self._forward(step_ids, **step_kwargs)")
                add(lines, 20, "return _next_id(step_result['logits'])")
                add(lines, 12, "else:")
                add(lines, 16, "def _decode_step_jit_entry(step_ids, cache_keys, cache_values, attention_store, cache_len_bound, attention_len_bound):")
                add(lines, 20, "step_kwargs = dict(kwargs)")
                add(lines, 20, f"step_kwargs[{cache_name!r}] = (cache_keys, cache_values, cache_len_bound)")
                if use_cache_name is not None:
                    add(lines, 20, f"step_kwargs[{use_cache_name!r}] = True")
                add(lines, 20, f"step_kwargs[{attention_name!r}] = (attention_store, attention_len_bound)")
                add(lines, 20, "step_result = self._forward(step_ids, **step_kwargs)")
                add(lines, 20, f"step_cache = step_result.get({cache_output_name!r}) if isinstance(step_result, dict) else None")
                add(lines, 20, "return _next_id(step_result['logits']), step_cache[0], step_cache[1]")
                add(lines, 12, "decode_step_jit = TinyJit(_decode_step_jit_entry, prune=True)")
            else:
                add(lines, 8, "decode_step_jit = None")
            loop_indent = 8
            body_indent = 12
            add(lines, loop_indent, "for _ in range(limit):")
            add(lines, body_indent, "step_input = (last_token if last_token is not None else out[:, -1:].contiguous().realize()) if cache is not None else out")
            add(lines, body_indent, "step_len = int(step_input.shape[1])")
            add(lines, body_indent, "forward_kwargs = dict(kwargs)")
            if has_static_cache_shape:
                add(lines, body_indent, f"forward_kwargs[{cache_name!r}] = _maybe_bind_static_tuple_length(cache, cache_length_var, cache_length) if cache is not None and step_len == 1 else cache")
            else:
                add(lines, body_indent, f"forward_kwargs[{cache_name!r}] = cache")
            if use_cache_name is not None:
                add(lines, body_indent, f"forward_kwargs[{use_cache_name!r}] = True")
            if attention_name is not None:
                if uses_static_attention_mask:
                    add(lines, body_indent, f"forward_kwargs[{attention_name!r}] = _maybe_bind_static_tuple_length(attention_mask, attention_length_var, attention_length) if step_len == 1 else attention_mask")
                else:
                    add(lines, body_indent, f"forward_kwargs[{attention_name!r}] = attention_mask")
            if has_static_cache_shape and uses_static_attention_mask and attention_name is not None:
                add(lines, body_indent, "used_decode_jit = self._jit_enabled and not self._profile_enabled and cache is not None and step_len == 1")
                add(lines, body_indent, "if used_decode_jit:")
                add(lines, body_indent + 4, "if not self._tinygrad_owned_decode:")
                add(lines, body_indent + 8, "next_id, cache_keys, cache_values = decode_step_jit(step_input, cache[0], cache[1], attention_mask[0], cache_length_var.bind(cache_length), attention_length_var.bind(attention_length))")
                add(lines, body_indent + 8, f"result = {{{cache_output_name!r}: (cache_keys, cache_values, cache_length + step_len)}}")
                add(lines, body_indent + 4, "else:")
                add(lines, body_indent + 8, "if not cache_owned_realized:")
                add(lines, body_indent + 12, "cache, owned_cache_kv = _pack_owned_static_cache(cache)")
                add(lines, body_indent + 12, "cache_owned_realized = True")
                add(lines, body_indent + 12, "self._tinygrad_owned_static_cache_roots = {id(cache[0]), id(cache[1])}")
                add(lines, body_indent + 8, "previous_mutable_slices = self._tinygrad_mutable_slices")
                add(lines, body_indent + 8, "previous_cache_roots = self._tinygrad_owned_static_cache_roots")
                add(lines, body_indent + 8, "self._tinygrad_owned_static_cache_roots = {id(cache[0]), id(cache[1])}")
                add(lines, body_indent + 8, "self._tinygrad_mutable_slices = True")
                add(lines, body_indent + 8, "try:")
                add(lines, body_indent + 12, "next_id = decode_step_jit(step_input, attention_mask[0], cache_length_var.bind(cache_length), attention_length_var.bind(attention_length))")
                add(lines, body_indent + 8, "finally:")
                add(lines, body_indent + 12, "self._tinygrad_mutable_slices = previous_mutable_slices")
                add(lines, body_indent + 12, "self._tinygrad_owned_static_cache_roots = previous_cache_roots")
                add(lines, body_indent + 8, f"result = {{{cache_output_name!r}: cache}}")
                add(lines, body_indent, "else:")
                add(lines, body_indent + 4, "result = self._forward_for_generate(step_input, **forward_kwargs) if cache is not None and int(step_input.shape[1]) == 1 else self._forward(step_input, **forward_kwargs)")
            else:
                add(lines, body_indent, "used_decode_jit = False")
                add(lines, body_indent, "result = self._forward_for_generate(step_input, **forward_kwargs) if cache is not None and int(step_input.shape[1]) == 1 else self._forward(step_input, **forward_kwargs)")
            add(lines, body_indent, "if isinstance(result, dict): cache = result.get(" + repr(cache_output_name) + ", cache)")
            if has_static_cache_shape:
                add(lines, body_indent, "if cache is not None:")
                add(lines, body_indent + 4, "cache_length += step_len")
                add(lines, body_indent + 4, "cache = _with_static_tuple_length(cache, cache_length)")
            add(lines, body_indent, "if not used_decode_jit:")
            add(lines, body_indent + 4, "next_id = _next_id(_logits(result))")
            add(lines, body_indent, "next_id, finished = _apply_eos(next_id, eos, pad, finished)")
            add(lines, body_indent, "last_token = next_id")
            add(lines, body_indent, "out = out.cat(next_id, dim=1)")
            if attention_name is not None:
                if uses_static_attention_mask:
                    add(lines, body_indent, "attention_length += 1")
                else:
                    add(lines, body_indent, "attention_mask = attention_mask.cat(_ones_like_ids(next_id), dim=1)")
            add(lines, body_indent, "if _all_done(finished): break")
            add(lines, 8, "return self._to_torch(out)")
            return
        if is_decoder_only:
            add(lines, 8, "out = input_ids")
            add(lines, 8, "limit = _generation_limit(out)")
            add(lines, 8, "eos, pad, finished = _eos_state(out.shape[0])")
            if attention_name is not None:
                other = "attention_mask" if attention_name == "attn_mask" else "attn_mask"
                add(lines, 8, f"attention_mask = kwargs.pop({attention_name!r}, kwargs.pop({other!r}, None))")
                add(lines, 8, "if attention_mask is None: attention_mask = _ones_like_ids(out)")
            add(lines, 8, "for _ in range(limit):")
            add(lines, 12, "forward_kwargs = dict(kwargs)")
            if attention_name is not None:
                add(lines, 12, f"forward_kwargs[{attention_name!r}] = attention_mask")
            add(lines, 12, "result = self._forward_for_generate(out, **forward_kwargs)")
            add(lines, 12, "next_id = _next_id(_logits(result))")
            add(lines, 12, "next_id, finished = _apply_eos(next_id, eos, pad, finished)")
            add(lines, 12, "out = out.cat(next_id, dim=1)")
            if attention_name is not None:
                add(lines, 12, "attention_mask = attention_mask.cat(_ones_like_ids(next_id), dim=1)")
            add(lines, 12, "if _all_done(finished): break")
            add(lines, 8, "return self._to_torch(out)")
            return
        add(lines, 8, "decoder_input_ids = kwargs.pop('decoder_input_ids', None)")
        add(lines, 8, "if decoder_input_ids is None:")
        add(lines, 12, "start_id = kwargs.pop('decoder_start_token_id', self.config.get('decoder_start_token_id', self.config.get('pad_token_id', 0)))")
        add(lines, 12, "decoder_input_ids = Tensor.full((input_ids.shape[0], 1), int(start_id), dtype=dtypes.int64, device=self._tiny_device)")
        add(lines, 8, "limit = _generation_limit(input_ids)")
        add(lines, 8, "eos, pad, finished = _eos_state(decoder_input_ids.shape[0])")
        if attention_name is not None:
            other = "attention_mask" if attention_name == "attn_mask" else "attn_mask"
            add(lines, 8, f"attention_mask = kwargs.pop({attention_name!r}, kwargs.pop({other!r}, None))")
            add(lines, 8, "if attention_mask is None: attention_mask = _ones_like_ids(input_ids)")
        if decoder_attention_name is not None:
            add(lines, 8, f"decoder_attention_mask = kwargs.pop({decoder_attention_name!r}, None)")
            add(lines, 8, "if decoder_attention_mask is None: decoder_attention_mask = _ones_like_ids(decoder_input_ids)")
        add(lines, 8, "for _ in range(limit):")
        add(lines, 12, "forward_kwargs = dict(kwargs)")
        add(lines, 12, "forward_kwargs['decoder_input_ids'] = decoder_input_ids")
        if attention_name is not None:
            add(lines, 12, f"forward_kwargs[{attention_name!r}] = attention_mask")
        if decoder_attention_name is not None:
            add(lines, 12, f"forward_kwargs[{decoder_attention_name!r}] = decoder_attention_mask")
        add(lines, 12, "result = self._forward(input_ids, **forward_kwargs)")
        add(lines, 12, "next_id = _next_id(_logits(result))")
        add(lines, 12, "next_id, finished = _apply_eos(next_id, eos, pad, finished)")
        add(lines, 12, "decoder_input_ids = decoder_input_ids.cat(next_id, dim=1)")
        if decoder_attention_name is not None:
            add(lines, 12, "decoder_attention_mask = decoder_attention_mask.cat(_ones_like_ids(next_id), dim=1)")
        add(lines, 12, "if _all_done(finished): break")
        add(lines, 8, "return self._to_torch(decoder_input_ids)")

    def _primitive_expr(self, primitive: str, node: Any, *, local: set[str], symbols_dict: str) -> str:
        args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
        attrs = {k: self._operand_expr(v, local=local, symbols_dict=symbols_dict) for k, v in node.attrs.items()}
        if primitive == "embedding":
            return f"self._embedding({args[0]}, {args[1]})"
        if primitive == "linear":
            bias = args[3] if len(args) > 3 else "False"
            transpose = args[4] if len(args) > 4 else "False"
            expert = args[5] if len(args) > 5 else "None"
            weight_leaf = args[6] if len(args) > 6 else "'weight'"
            bias_leaf = args[7] if len(args) > 7 else "'bias'"
            return (
                f"(lambda _w, _b: {args[1]}.linear(_w, _b) "
                f"if bool({transpose}) else {args[1]}.linear(_w.transpose(-1, -2), _b))"
                f"((lambda _w: (_w[int({expert})] if {expert} is not None else _w))(self._param(self._compose_path({args[0]}, {weight_leaf}))), "
                f"((lambda _b: (_b[int({expert})] if (_b is not None and {expert} is not None and len(_b.shape) >= 2) else _b))(self._optional_param(self._compose_path({args[0]}, {bias_leaf}))) if bool({bias}) else None))"
            )
        if primitive == "expert_linear":
            bias = args[4] if len(args) > 4 else "False"
            transpose = args[5] if len(args) > 5 else "False"
            weight_leaf = args[6] if len(args) > 6 else "'weight'"
            bias_leaf = args[7] if len(args) > 7 else "'bias'"
            return f"self._expert_linear({args[0]}, {args[1]}, {args[2]}, bias=bool({bias}), transpose=bool({transpose}), weight_leaf={weight_leaf}, bias_leaf={bias_leaf})"
        if primitive == "_tinygrad_sdpa":
            if len(args) < 6:
                raise ValueError("__tinygrad_sdpa expects q, k, v, additive_mask, scale, enable_gqa")
            if not self._literal_null_arg(node.inputs[4]):
                raise ValueError("__tinygrad_sdpa lowering currently requires default scale")
            return (
                f"{args[0]}.scaled_dot_product_attention("
                f"{args[1]}, {args[2]}, attn_mask={args[3]}, dropout_p=0.0, "
                f"is_causal=False, enable_gqa=bool({args[5]}))"
            )
        if primitive == "layernorm":
            eps = args[2] if len(args) > 2 else "1e-5"
            weight_leaf = args[4] if len(args) > 4 else "'weight'"
            bias = args[5] if len(args) > 5 else "True"
            bias_leaf = args[6] if len(args) > 6 else "'bias'"
            return f"(lambda _w, _b, _y: (_y * _w + (_b if _b is not None else 0))) (self._param(self._compose_path({args[0]}, {weight_leaf})), (self._optional_param(self._compose_path({args[0]}, {bias_leaf})) if {bias} else None), {args[1]}.layernorm(axis=-1, eps=float({eps})))"
        if primitive == "rmsnorm":
            x = args[0]
            eps = args[1] if len(args) > 1 else "1e-6"
            return f"({x} * (({x} * {x}).mean(axis=-1, keepdim=True) + float({eps})).rsqrt())"
        if primitive == "tensor_like":
            dtype = args[2] if len(args) > 2 else "None"
            return f"({args[0]}.cast(self._dtype_from_name({dtype}) or {args[1]}.dtype) if isinstance({args[0]}, Tensor) else Tensor({args[0]}, dtype=(self._dtype_from_name({dtype}) or {args[1]}.dtype), device=self._tiny_device))"
        if primitive == "softmax":
            dim = args[1] if len(args) > 1 else "-1"
            dtype = args[2] if len(args) > 2 else "None"
            return f"{args[0]}.softmax(axis=int({dim}), dtype=self._dtype_from_name({dtype}))"
        if primitive == "topk":
            return f"self._topk({args[0]}, {args[1]}, dim={args[2]}, largest={args[3]}, sorted_={args[4]})"
        if primitive == "concat":
            if "dim" in attrs:
                return f"self._concat({', '.join(args)}, dim={attrs['dim']})"
            return f"self._concat({', '.join(args[:-1])}, dim={args[-1]})"
        if primitive.startswith("config_") or primitive in {"params_param", "params_has_root"}:
            return super()._primitive_expr(primitive, node, local=local, symbols_dict=symbols_dict)
        simple = {
            "reshape": lambda: f"self._reshape({args[0]}, {args[1]})",
            "arange": lambda: f"Tensor.arange(int({args[1]}), int(({args[0]}.shape[-2] if {args[2]} is None and len({args[0]}.shape) >= 2 else ({args[0]}.shape[-1] if {args[2]} is None else {args[2]}))), dtype=dtypes.int64, device=self._tiny_device)",
            "slice": lambda: f"self._slice({args[0]}, {args[1]}, {args[2]}, {args[3]})",
            "assign_slice": lambda: f"self._assign_slice({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]})",
            "chunk": lambda: f"{args[0]}.chunk(int({args[2] if len(args) > 2 else attrs.get('parts', '1')}), dim=int({args[1] if len(args) > 1 else attrs.get('dim', '-1')}))",
            "split": lambda: f"{args[0]}.split([int(x) for x in {args[2] if len(args) > 2 else attrs.get('sizes', '[]')}], dim=int({args[1] if len(args) > 1 else attrs.get('dim', '-1')}))",
            "sum": lambda: f"{args[0]}.sum(axis=int({args[1] if len(args) > 1 else '-1'}), keepdim=bool({args[2] if len(args) > 2 else 'False'}))",
            "expand": lambda: f"{args[0]}.expand(tuple(int(x) for x in {args[1]}))",
            "permute": lambda: f"{args[0]}.permute(tuple(int(x) for x in {args[1]}))",
            "transpose": lambda: f"{args[0]}.transpose(int({args[1]}), int({args[2]}))",
            "unsqueeze": lambda: f"self._unsqueeze({args[0]}, {args[1]})",
            "repeat": lambda: f"{args[0]}.repeat_interleave(int({args[1]}), dim=(int({args[2]}) if int({args[2]}) >= 0 else int({args[2]}) + len({args[0]}.shape)))",
            "matmul": lambda: f"{args[0]}.matmul({args[1]})",
            "where": lambda: f"self._where({args[0]}, {args[1]}, {args[2]})",
            "where_indices": lambda: f"self._where_indices({args[0]})",
            "require": lambda: f"self._require_value({args[0]})",
            "list_length": lambda: f"len({args[0]})",
            "gather": lambda: f"{args[0]}.gather(int({args[2] if len(args) > 2 else '-1'}), {args[1]})",
            "scatter": lambda: f"{args[0]}.scatter(int({args[3] if len(args) > 3 else '-1'}), {args[1]}, {args[2]})",
            "index_add": lambda: f"self._index_add({args[0]}, {args[1]}, {args[2]}, {args[3] if len(args) > 3 else '0'})",
            "le": lambda: f"({args[0]} <= {args[1]})",
            "eq": lambda: f"self._eq({args[0]}, {args[1]})",
            "and": lambda: f"({args[0]} & {args[1]})",
            "add": lambda: f"self._binary_add({args[0]}, {args[1]})",
            "mul": lambda: f"({args[0]} * {args[1]})",
            "div": lambda: f"({args[0]} / {args[1]})",
            "pow": lambda: f"({args[0]} ** {args[1]})",
            "floor": lambda: f"{args[0]}.floor() if isinstance({args[0]}, Tensor) else int({args[0]} // 1)",
            "sqrt": lambda: f"{args[0]}.sqrt() if isinstance({args[0]}, Tensor) else ({args[0]} ** 0.5)",
            "sin": lambda: f"{args[0]}.sin() if isinstance({args[0]}, Tensor) else __import__('math').sin(float({args[0]}))",
            "cos": lambda: f"{args[0]}.cos() if isinstance({args[0]}, Tensor) else __import__('math').cos(float({args[0]}))",
            "exp": lambda: f"{args[0]}.exp() if isinstance({args[0]}, Tensor) else __import__('math').exp(float({args[0]}))",
            "log": lambda: f"{args[0]}.log() if isinstance({args[0]}, Tensor) else __import__('math').log(float({args[0]}))",
            "cast": lambda: f"{args[0]}.cast(self._dtype_from_name({args[1]}))",
            "cast_like": lambda: f"{args[0]}.cast({args[1]}.dtype)",
            "dtype_value": lambda: f"(lambda _x: self._dtype_value(_x.dtype, {args[1]}))(self._value({args[0]}))",
            "cumsum": lambda: f"{args[0]}.cumsum(axis=int({args[1] if len(args) > 1 else '-1'}))",
            "empty_like": lambda: f"(lambda _x: Tensor.empty(*_x.shape, dtype=_x.dtype, device=_x.device))(self._value({args[0]}))",
            "fill": lambda: f"(lambda _x: Tensor.full(_x.shape, {args[1]}, dtype=(_x.dtype if self._dtype_from_name({args[2] if len(args) > 2 else 'None'}) is None else self._dtype_from_name({args[2] if len(args) > 2 else 'None'})), device=_x.device))(self._value({args[0]}))",
            "empty": lambda: f"Tensor.empty(*tuple(int(x) for x in {args[1]}), dtype=((self._dtype_from_name({args[2] if len(args) > 2 else 'None'}) if {args[2] if len(args) > 2 else 'None'} is not None else None) or {args[0]}.dtype), device=self._tiny_device)",
            "zeros": lambda: f"Tensor.zeros(*tuple(int(x) for x in {args[1]}), dtype=((self._dtype_from_name({args[2] if len(args) > 2 else 'None'}) if {args[2] if len(args) > 2 else 'None'} is not None else None) or {args[0]}.dtype), device=self._tiny_device)",
            "full": lambda: f"Tensor.full(tuple(int(x) for x in {args[1]}), {args[2]}, dtype=((self._dtype_from_name({args[3] if len(args) > 3 else 'None'}) if {args[3] if len(args) > 3 else 'None'} is not None else None) or {args[0]}.dtype), device=self._tiny_device)",
            "zeros_like": lambda: f"(lambda _x: Tensor.zeros(*_x.shape, dtype=_x.dtype, device=_x.device))(self._value({args[0]}))",
            "activations_tanh": lambda: f"{args[0]}.tanh()",
            "activations_silu": lambda: f"{args[0]}.silu()",
            "activations_sigmoid": lambda: f"{args[0]}.sigmoid()",
            "activations_swiglu": lambda: f"({args[0]}.silu() * {args[0]})",
            "l2norm": lambda: f"({args[0]} * (({args[0]} * {args[0]}).mean(axis=-1, keepdim=True) + float({args[1] if len(args) > 1 else '1e-6'})).rsqrt())",
            "activations_relu": lambda: f"{args[0]}.relu()",
            "activations_relu2": lambda: f"({args[0]}.relu() * {args[0]}.relu())",
            "activations_gelu": lambda: f"(0.5 * {args[0]} * (1.0 + ({args[0]} / 1.4142135623730951).erf()))",
            "activations_gelu_new": lambda: f"(0.5 * {args[0]} * (1.0 + (0.7978845608028654 * ({args[0]} + 0.044715 * {args[0]} * {args[0]} * {args[0]})).tanh()))",
            "activations_gelu_pytorch_tanh": lambda: f"(0.5 * {args[0]} * (1.0 + (0.7978845608028654 * ({args[0]} + 0.044715 * {args[0]} * {args[0]} * {args[0]})).tanh()))",
            "activations_gegelu": lambda: f"self._gegelu({args[0]}, {args[1] if len(args) > 1 else 'None'})",
            "activations_xielu": lambda: f"self._xielu({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]})",
            "list_init": lambda: "[]",
            "list_append": lambda: f"([*({args[0]} or []), {args[1]}])",
            "list_index": lambda: f"{args[0]}[int({args[1]})]",
            "list_length": lambda: f"len({args[0]} or [])",
            "shape": lambda: f"list(self._value({args[0]}).shape)",
            "tensor_size": lambda: f"self._value({args[0]}).shape[int({args[1]})]",
        }
        if primitive == "clamp":
            min_value = args[1] if len(args) > 1 else attrs.get("min", "None")
            max_value = args[2] if len(args) > 2 else attrs.get("max", "None")
            return f"({args[0]}.maximum({min_value}) if {max_value} is None else ({args[0]}.minimum({max_value}) if {min_value} is None else {args[0]}.maximum({min_value}).minimum({max_value})))"
        if primitive in simple:
            return simple[primitive]()
        raise NotImplementedError(f"direct codegen2-tinygrad unsupported graph op {primitive!r}")

    def _emit_layernorm_node(
        self,
        lines: list[str],
        node: Any,
        *,
        target: str,
        indent: int,
        local: set[str],
        symbols_dict: str,
    ) -> bool:
        if len(node.inputs) < 2:
            return False
        args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
        eps = (
            self._scalar_operand_expr(
                node.inputs[2],
                local=local,
                symbols_dict=symbols_dict,
                expected=(TypeFloat, TypeInt, TypeDim),
                cast="float",
            )
            if len(node.inputs) > 2
            else "1e-5"
        )
        bias_expr = (
            self._scalar_operand_expr(
                node.inputs[5],
                local=local,
                symbols_dict=symbols_dict,
                expected=(TypeBool,),
                cast="bool",
            )
            if len(node.inputs) > 5
            else "True"
        )
        bias_literal = self._literal_bool_arg(node.inputs[5]) if len(node.inputs) > 5 else True
        weight = self._param_expr_for_path(
            node.inputs[0],
            node.inputs[4] if len(node.inputs) > 4 else "weight",
            local=local,
            symbols_dict=symbols_dict,
        )
        bias_value = self._param_expr_for_path(
            node.inputs[0],
            node.inputs[6] if len(node.inputs) > 6 else "bias",
            optional=True,
            local=local,
            symbols_dict=symbols_dict,
        )
        weight_name = f"{target}__weight"
        bias_name = f"{target}__bias"
        self._add(lines, indent, f"{weight_name} = {weight}")
        self._emit_optional_param_bind(
            lines,
            target=bias_name,
            value_expr=bias_value,
            flag_expr=bias_expr,
            flag_literal=bias_literal,
            indent=indent,
        )
        y_name = f"{target}__y"
        self._add(lines, indent, f"{y_name} = {args[1]}.layernorm(axis=-1, eps={eps})")
        op_expr = f"({y_name} * {weight_name} + ({bias_name} if {bias_name} is not None else 0))"
        if self.profile:
            self._add(lines, indent, f"{target} = self._profile_call({f'node:{target}:_layernorm'!r}, lambda: {op_expr})")
        else:
            self._add(lines, indent, f"{target} = {op_expr}")
        return True

    def _emit_linear_node(
        self,
        lines: list[str],
        node: Any,
        *,
        target: str,
        indent: int,
        local: set[str],
        symbols_dict: str,
    ) -> bool:
        if len(node.inputs) < 2:
            return False
        if len(node.inputs) > 5 and not self._literal_null_arg(node.inputs[5]):
            return False
        args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
        bias_expr = (
            self._scalar_operand_expr(
                node.inputs[3],
                local=local,
                symbols_dict=symbols_dict,
                expected=(TypeBool,),
                cast="bool",
            )
            if len(node.inputs) > 3
            else "False"
        )
        transpose_expr = (
            self._scalar_operand_expr(
                node.inputs[4],
                local=local,
                symbols_dict=symbols_dict,
                expected=(TypeBool,),
                cast="bool",
            )
            if len(node.inputs) > 4
            else "False"
        )
        bias_literal = self._literal_bool_arg(node.inputs[3]) if len(node.inputs) > 3 else False
        transpose_literal = self._literal_bool_arg(node.inputs[4]) if len(node.inputs) > 4 else False
        weight = self._param_expr_for_path(
            node.inputs[0],
            node.inputs[6] if len(node.inputs) > 6 else "weight",
            local=local,
            symbols_dict=symbols_dict,
        )
        bias_value = self._param_expr_for_path(
            node.inputs[0],
            node.inputs[7] if len(node.inputs) > 7 else "bias",
            optional=True,
            local=local,
            symbols_dict=symbols_dict,
        )
        weight_name = f"{target}__weight"
        bias_name = f"{target}__bias"
        x_name = f"{target}__x"
        self._add(lines, indent, f"{weight_name} = {weight}")
        if bias_literal is False:
            bias_arg = "None"
        else:
            self._emit_optional_param_bind(
                lines,
                target=bias_name,
                value_expr=bias_value,
                flag_expr=bias_expr,
                flag_literal=bias_literal,
                indent=indent,
            )
            bias_arg = bias_name
        self._add(lines, indent, f"{x_name} = {args[1]}")
        if transpose_literal is True:
            op_expr = f"{x_name}.linear({weight_name}, {bias_arg})"
        elif transpose_literal is False:
            op_expr = f"{x_name}.linear({weight_name}.transpose(-1, -2), {bias_arg})"
        else:
            direct = f"{x_name}.linear({weight_name}, {bias_arg})"
            standard = f"{x_name}.linear({weight_name}.transpose(-1, -2), {bias_arg})"
            op_expr = f"({direct} if {transpose_expr} else {standard})"
        if self.profile:
            self._add(lines, indent, f"{target} = self._profile_call({f'node:{target}:_linear'!r}, lambda: {op_expr})")
        else:
            self._add(lines, indent, f"{target} = {op_expr}")
        return True


def emit_model_code_from_graph_ir(
    graph: GraphProgram,
    *,
    class_name: str = "AxonTinygradModel",
    model_config: dict[str, Any] | None = None,
    profile: bool = False,
) -> str:
    validate_graph_program(graph)
    unsupported = non_obvious_tinygrad_ops(graph)
    if unsupported:
        table = tinygrad_op_table_markdown(graph)
        raise NotImplementedError(
            "codegen2-tinygrad cannot emit this Graph IR yet.\n"
            "Unsupported Graph IR ops:\n"
            f"{table}"
        )
    emitter = _DirectTinygradEmitter(program=graph, class_name=class_name, profile=profile)
    body = emitter.emit()
    # Tensor execution inside generated definitions uses tinygrad and converts
    # back to torch at the public forward/generate boundary.
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "import os",
            "import time",
            "import torch",
            "from tinygrad import Tensor, TinyJit, dtypes, UOp, Variable",
            "from tinygrad.nn import state as tiny_state",
            "from brainsurgery.synapse.axon.codegen2_common import (",
            "    compose_path as _common_compose_path,",
            "    config_value as _common_config_value,",
            "    has_config_value as _common_has_config_value,",
            "    optional_state_value as _common_optional_state_value,",
            "    render_path as _common_render_path,",
            "    required_state_value as _common_required_state_value,",
            "    require_value as _common_require_value,",
            ")",
            "",
            f"_MODEL_CONFIG = {model_config!r}",
            "",
            body,
        ]
    )


OBVIOUS_TINYGRAD_PRIMITIVES = SUPPORTED_TINYGRAD_PRIMITIVES

__all__ = [
    "NON_OBVIOUS_TINYGRAD_OPS",
    "OBVIOUS_TINYGRAD_PRIMITIVES",
    "SHARED_COMMON_PRIMITIVES",
    "SUPPORTED_TINYGRAD_PRIMITIVES",
    "TinygradUnsupportedOp",
    "emit_model_code_from_graph_ir",
    "non_obvious_tinygrad_ops",
    "tinygrad_op_table_markdown",
]
