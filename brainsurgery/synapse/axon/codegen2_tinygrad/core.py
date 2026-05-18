from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..ast import TypeOptional
from ..codegen2_common import normalize_primitive_op
from ..codegen2_torch.core import _DirectTorchEmitter
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
        add(lines, 8, "self.config = dict(({} if _MODEL_CONFIG is None else _MODEL_CONFIG) if config is None else config)")
        add(lines, 8, "self._profile_enabled = False")
        add(lines, 8, "self._profile_cuda = True")
        add(lines, 8, "self._profile_records = {}")
        add(lines, 8, "self._symbols = {}")
        add(lines, 8, "self.load_state_dict(state_dict)")
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
        add(lines, 8, "self.setup()")
        add(lines, 8, "self._symbols = self._eval_symbols()")
        add(lines, 8, "return self")
        add(lines, 4, "")
        add(lines, 4, "def setup(self):")
        add(lines, 8, "self._materialize_expert_banks()")
        add(lines, 8, "return None")
        add(lines, 4, "")
        add(lines, 4, "def to(self, device=None, *args, **kwargs):")
        add(lines, 8, "if device is not None:")
        add(lines, 12, "self._torch_device = torch.device(device)")
        add(lines, 12, "self._tiny_device = self._torch_to_tiny_device(self._torch_device)")
        add(lines, 12, "self.state_dict_tensors = {k: self._move_tiny(v, self._tiny_device) for k, v in self.state_dict_tensors.items()}")
        add(lines, 12, "self.after_to()")
        add(lines, 8, "return self")
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
        add(lines, 8, "if device.type == 'cuda': return 'CUDA'")
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
        add(lines, 12, "return value.to(self._tiny_device)")
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
        add(lines, 4, "_cache_past_length = staticmethod(_common_cache_past_length)")
        add(lines, 4, "")
        add(lines, 4, "def _param(self, path):")
        add(lines, 8, "return _common_required_state_value(self.state_dict_tensors, path)")
        add(lines, 4, "")
        add(lines, 4, "def _optional_param(self, path):")
        add(lines, 8, "return _common_optional_state_value(self.state_dict_tensors, path)")
        add(lines, 4, "")
        add(lines, 4, "def _materialize_expert_banks(self):")
        add(lines, 8, "groups = {}")
        add(lines, 8, "for key, value in list(self.state_dict_tensors.items()):")
        add(lines, 12, "parts = key.split('.')")
        add(lines, 12, "for idx, part in enumerate(parts):")
        add(lines, 16, "if part == 'experts' and idx + 2 < len(parts) and parts[idx + 1].isdigit():")
        add(lines, 20, "expert = int(parts[idx + 1])")
        add(lines, 20, "bank_key = '.'.join(parts[:idx + 1] + parts[idx + 2:])")
        add(lines, 20, "groups.setdefault(bank_key, {})[expert] = value")
        add(lines, 20, "break")
        add(lines, 8, "for bank_key, items in groups.items():")
        add(lines, 12, "if bank_key in self.state_dict_tensors or not items:")
        add(lines, 16, "continue")
        add(lines, 12, "ordered = [items[i] for i in range(len(items)) if i in items]")
        add(lines, 12, "if len(ordered) != len(items):")
        add(lines, 16, "continue")
        add(lines, 12, "first_shape = ordered[0].shape")
        add(lines, 12, "if any(t.shape != first_shape for t in ordered):")
        add(lines, 16, "continue")
        add(lines, 12, "self.state_dict_tensors[bank_key] = ordered[0].stack(*ordered[1:], dim=0) if len(ordered) > 1 else ordered[0].reshape((1,) + tuple(ordered[0].shape))")
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
                args.append(value.name)
        add(lines, 8, f"result = self.{self.method_names[main.name]}({', '.join(args)})")
        names = main.output_names or ("logits",)
        if len(names) == 1:
            add(lines, 8, "return result[0]")
        else:
            add(lines, 8, f"return {{{', '.join(f'{name!r}: result[{idx}]' for idx, name in enumerate(names))}}}")
        add(lines, 4, "")
        add(lines, 4, "def forward(self, input_ids=None, **inputs):")
        add(lines, 8, "return self._to_torch(self._forward(input_ids, **inputs))")

    def _emit_generate(self, lines: list[str]) -> None:
        add = self._add
        main = self.modules_by_name[self.program.main_module]
        input_names = {value.name for value in main.inputs}
        output_names = set(main.output_names or ("logits",))
        attention_name = "attn_mask" if "attn_mask" in input_names else (
            "attention_mask" if "attention_mask" in input_names else None
        )
        decoder_attention_name = "decoder_attention_mask" if "decoder_attention_mask" in input_names else None
        cache_name = "past_kv" if "past_kv" in input_names else (
            "past_cache" if "past_cache" in input_names else None
        )
        cache_output_name = "new_kv" if "new_kv" in output_names else (
            "past_kv" if "past_kv" in output_names else ("cache" if "cache" in output_names else None)
        )
        use_cache_name = "use_cache" if "use_cache" in input_names else None
        has_decoder_inputs = "decoder_input_ids" in input_names
        is_cached_decoder = not has_decoder_inputs and cache_name is not None and cache_output_name is not None
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
        add(lines, 12, "return next_id, finished")
        add(lines, 8, "def _all_done(finished):")
        add(lines, 12, "return finished is not None and bool(finished.min().item())")
        if is_cached_decoder:
            add(lines, 8, "out = input_ids")
            add(lines, 8, "limit = _generation_limit(out)")
            add(lines, 8, "eos, pad, finished = _eos_state(out.shape[0])")
            add(lines, 8, f"cache = kwargs.pop({cache_name!r}, None)")
            if attention_name is not None:
                other = "attention_mask" if attention_name == "attn_mask" else "attn_mask"
                add(lines, 8, f"attention_mask = kwargs.pop({attention_name!r}, kwargs.pop({other!r}, None))")
                add(lines, 8, "if attention_mask is None: attention_mask = _ones_like_ids(out)")
            if use_cache_name is not None:
                add(lines, 8, f"kwargs.pop({use_cache_name!r}, None)")
            add(lines, 8, "for _ in range(limit):")
            add(lines, 12, "step_input = out[:, -1:] if cache is not None else out")
            add(lines, 12, "forward_kwargs = dict(kwargs)")
            add(lines, 12, f"forward_kwargs[{cache_name!r}] = cache")
            if use_cache_name is not None:
                add(lines, 12, f"forward_kwargs[{use_cache_name!r}] = True")
            if attention_name is not None:
                add(lines, 12, f"forward_kwargs[{attention_name!r}] = attention_mask")
            add(lines, 12, "result = self._forward(step_input, **forward_kwargs)")
            add(lines, 12, "if isinstance(result, dict): cache = result.get(" + repr(cache_output_name) + ", cache)")
            add(lines, 12, "next_id = _next_id(_logits(result))")
            add(lines, 12, "next_id, finished = _apply_eos(next_id, eos, pad, finished)")
            add(lines, 12, "out = out.cat(next_id, dim=1)")
            if attention_name is not None:
                add(lines, 12, "attention_mask = attention_mask.cat(_ones_like_ids(next_id), dim=1)")
            add(lines, 12, "if _all_done(finished): break")
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
            add(lines, 12, "result = self._forward(out, **forward_kwargs)")
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
            return f"self._param(self._compose_path({args[0]}, 'weight'))[{args[1]}]"
        if primitive == "linear":
            bias = args[3] if len(args) > 3 else "False"
            transpose = args[4] if len(args) > 4 else "False"
            expert = args[5] if len(args) > 5 else "None"
            weight_leaf = args[6] if len(args) > 6 else "'weight'"
            bias_leaf = args[7] if len(args) > 7 else "'bias'"
            return (
                f"(lambda _w, _b: ({args[1]}.matmul(_w) + (_b if _b is not None else 0)) "
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
            "reshape": lambda: f"{args[0]}.reshape(tuple(int(x) for x in {args[1]}))",
            "arange": lambda: f"Tensor.arange(int({args[1]}), int(({args[0]}.shape[-2] if {args[2]} is None and len({args[0]}.shape) >= 2 else ({args[0]}.shape[-1] if {args[2]} is None else {args[2]}))), dtype=dtypes.int64, device=self._tiny_device)",
            "slice": lambda: f"{args[0]}.shrink(tuple((int({args[2]}), int({args[3]})) if i == (int({args[1]}) % len({args[0]}.shape)) else None for i in range(len({args[0]}.shape))))",
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


def emit_model_code_from_graph_ir(
    graph: GraphProgram,
    *,
    class_name: str = "AxonTinygradModel",
    model_config: dict[str, Any] | None = None,
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
    emitter = _DirectTinygradEmitter(program=graph, class_name=class_name)
    body = emitter.emit()
    # Tensor execution inside generated definitions uses tinygrad and converts
    # back to torch at the public forward/generate boundary.
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "import torch",
            "from tinygrad import Tensor, dtypes",
            "from tinygrad.nn import state as tiny_state",
            "from brainsurgery.synapse.axon.codegen2_common import (",
            "    cache_past_length as _common_cache_past_length,",
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
