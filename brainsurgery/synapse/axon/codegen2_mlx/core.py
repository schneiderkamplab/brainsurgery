from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..ast import TypeBool, TypeDim, TypeFloat, TypeInt, TypeOptional
from ..codegen2_common import normalize_primitive_op
from ..codegen2_torch.core import _DirectTorchEmitter, graph_main_output_names
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

SUPPORTED_MLX_PRIMITIVES: frozenset[str] = frozenset({
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
    "cumsum",
    "_mlx_sdpa",
    "_mlx_rope",
})

NON_OBVIOUS_MLX_OPS: dict[str, str] = {}


@dataclass(frozen=True)
class MlxUnsupportedOp:
    op: str
    count: int
    reason: str


def _normalize_primitive_op(name: str) -> str:
    return normalize_primitive_op(name)


def non_obvious_mlx_ops(graph: GraphProgram) -> tuple[MlxUnsupportedOp, ...]:
    counts: Counter[str] = Counter()
    module_names = {module.name for module in graph.modules}
    for module in graph.modules:
        for node in module.nodes:
            op = node.op.name
            if op.startswith("core.") or op.startswith("core.binary.") or op in module_names:
                continue
            primitive = _normalize_primitive_op(op)
            if primitive in SHARED_COMMON_PRIMITIVES or primitive in SUPPORTED_MLX_PRIMITIVES:
                continue
            counts[primitive] += 1
    return tuple(
        MlxUnsupportedOp(
            op=op,
            count=count,
            reason=NON_OBVIOUS_MLX_OPS.get(op, "no MLX lowering classified yet"),
        )
        for op, count in sorted(counts.items())
    )


def mlx_op_table_markdown(graph: GraphProgram) -> str:
    rows = non_obvious_mlx_ops(graph)
    if not rows:
        return "| Op | Count | Reason |\n|---|---:|---|\n"
    lines = ["| Op | Count | Reason |", "|---|---:|---|"]
    for row in rows:
        lines.append(f"| `{row.op}` | {row.count} | {row.reason} |")
    return "\n".join(lines)


def _py_ident(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe or "_"


def _path_to_safe_attr(path: str) -> str:
    safe = path.replace(".", "_")
    safe = re.sub(r"[^A-Za-z0-9_]", "_", safe)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe


def _collect_static_param_paths(program: GraphProgram) -> dict[str, str]:
    paths: dict[str, str] = {}
    for module in program.modules:
        for node in module.nodes:
            op = _normalize_primitive_op(node.op.name)
            if op not in {"linear", "embedding", "layernorm", "rmsnorm"}:
                continue
            if not node.inputs:
                continue
            path_operand = node.inputs[0]
            if not hasattr(path_operand, "parts"):
                continue
            if any("{" in part or "}" in part for part in path_operand.parts):
                continue
            joined = ".".join(part for part in path_operand.parts if part)
            if not joined:
                continue
            if joined not in paths:
                paths[joined] = op
    return paths


def _is_static_path(operand: Any) -> bool:
    if not hasattr(operand, "parts"):
        return False
    return not any("{" in part or "}" in part for part in operand.parts)


def _path_joined(operand: Any) -> str | None:
    if not hasattr(operand, "parts") or not operand.absolute:
        return None
    if any("{" in part or "}" in part for part in operand.parts):
        return None
    joined = ".".join(part for part in operand.parts if part)
    return joined or None


class _DirectMlxEmitter(_DirectTorchEmitter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._static_param_ops: dict[str, str] = _collect_static_param_paths(self.program)
        self._emitted_module_attrs: list[str] = []

    def emit(self) -> str:
        return super().emit().replace(
            f"class {self.class_name}(nn.Module):",
            f"class {self.class_name}(nn.Module):",
        )

    def _emit_common(self, lines: list[str]) -> None:
        add = self._add
        add(lines, 4, "def __init__(self, state_dict: dict[str, mx.array], config: dict | None = None):")
        add(lines, 8, "super().__init__()")
        add(lines, 8, "object.__setattr__(self, '_flat_tensors', {})")
        add(lines, 8, "object.__setattr__(self, '_path_cache', {})")
        add(lines, 8, "object.__setattr__(self, '_compiled_fn', None)")
        add(lines, 8, "object.__setattr__(self, 'config', dict(({} if _MODEL_CONFIG is None else _MODEL_CONFIG) if config is None else config))")
        add(lines, 8, "object.__setattr__(self, '_jit_enabled', False)")
        add(lines, 8, "object.__setattr__(self, '_quantized', False)")
        add(lines, 8, "object.__setattr__(self, '_profile_enabled', False)")
        add(lines, 8, "object.__setattr__(self, '_profile_records', {})")
        add(lines, 8, "object.__setattr__(self, '_symbols', {})")
        add(lines, 8, "self.load_state_dict(state_dict)")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def from_state_dict(cls, state_dict, *, graph=None, model_config=None):")
        add(lines, 8, "return cls(state_dict, config=_MODEL_CONFIG if model_config is None else model_config)")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def from_safetensors(cls, safetensors_files, *, model_config=None):")
        add(lines, 8, "state_dict = {}")
        add(lines, 8, "for path in safetensors_files:")
        add(lines, 12, "with safe_open(str(path), framework='np') as f:")
        add(lines, 16, "for key in f.keys():")
        add(lines, 20, "state_dict[str(key)] = mx.array(f.get_tensor(key))")
        add(lines, 8, "return cls(state_dict, config=_MODEL_CONFIG if model_config is None else model_config)")
        add(lines, 4, "")
        self._emit_load_state_dict(lines)
        add(lines, 4, "")
        add(lines, 4, "def enable_jit(self, enabled=True, reset=True):")
        add(lines, 8, "self._jit_enabled = bool(enabled)")
        add(lines, 8, "return self")
        add(lines, 4, "")
        add(lines, 4, "def enable_profile(self, enabled=True, *, reset=True):")
        add(lines, 8, "self._profile_enabled = bool(enabled)")
        add(lines, 8, "if reset:")
        add(lines, 12, "self._profile_records = {}")
        add(lines, 8, "return self")
        add(lines, 4, "")
        add(lines, 4, "def profile_summary(self, top_n=40):")
        add(lines, 8, "rows = []")
        add(lines, 8, "for name, (count, total) in self._profile_records.items():")
        add(lines, 12, "rows.append({'name': name, 'count': count, 'seconds': total, 'avg_seconds': total / max(1, count)})")
        add(lines, 8, "rows.sort(key=lambda row: row['seconds'], reverse=True)")
        add(lines, 8, "return rows[: int(top_n)]")
        add(lines, 4, "")
        add(lines, 4, "def to(self, *_args, **_kwargs):")
        add(lines, 8, "return self")
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
        add(lines, 4, "def _is_mlx(value):")
        add(lines, 8, "return isinstance(value, mx.array)")
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
        add(lines, 4, "_compose_path = staticmethod(_common_compose_path)")
        add(lines, 4, "_render_path = staticmethod(_common_render_path)")
        add(lines, 4, "_require_value = staticmethod(_common_require_value)")
        add(lines, 4, "")
        add(lines, 4, "def _path_template_part(self, value):")
        add(lines, 8, "cached = self._path_cache.get(value)")
        add(lines, 8, "if cached is not None:")
        add(lines, 12, "return cached")
        add(lines, 8, "if isinstance(value, str) and value.startswith('@@'):")
        add(lines, 12, "result = value[2:].strip('.')")
        add(lines, 8, "elif isinstance(value, str) and value.startswith('@'):")
        add(lines, 12, "result = value[1:].strip('.')")
        add(lines, 8, "else:")
        add(lines, 12, "result = value")
        add(lines, 8, "self._path_cache[value] = result")
        add(lines, 8, "return result")
        add(lines, 4, "")
        add(lines, 4, "def _param(self, path):")
        add(lines, 8, "key = str(path).lstrip('@')")
        add(lines, 8, "if key in self._flat_tensors:")
        add(lines, 12, "return self._flat_tensors[key]")
        add(lines, 8, "self._materialize_expert_bank_for_path(key)")
        add(lines, 8, "return _common_required_state_value(self._flat_tensors, path)")
        add(lines, 4, "")
        add(lines, 4, "def _optional_param(self, path):")
        add(lines, 8, "key = str(path).lstrip('@')")
        add(lines, 8, "if key in self._flat_tensors:")
        add(lines, 12, "return self._flat_tensors[key]")
        add(lines, 8, "self._materialize_expert_bank_for_path(key)")
        add(lines, 8, "return _common_optional_state_value(self._flat_tensors, path)")
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
        add(lines, 8, "for key in self._flat_tensors:")
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
        add(lines, 8, "existing = self._flat_tensors.get(bank_key)")
        add(lines, 8, "if isinstance(existing, mx.array):")
        add(lines, 12, "return existing")
        add(lines, 8, "ordered_keys = self._keys_for_collapsed_bank(bank_key)")
        add(lines, 8, "if ordered_keys:")
        add(lines, 12, "ordered = [self._flat_tensors[key] for key in ordered_keys]")
        add(lines, 12, "first_shape = ordered[0].shape")
        add(lines, 12, "if all(isinstance(t, mx.array) and t.shape == first_shape for t in ordered):")
        add(lines, 16, "bank = mx.stack(ordered, axis=0) if len(ordered) > 1 else ordered[0].reshape((1,) + tuple(ordered[0].shape))")
        add(lines, 16, "for key in ordered_keys:")
        add(lines, 20, "self._flat_tensors.pop(key, None)")
        add(lines, 16, "self._flat_tensors[bank_key] = bank")
        add(lines, 16, "return bank")
        add(lines, 8, "fused_sources = self._fused_gate_up_source_bank_keys(bank_key)")
        add(lines, 8, "if fused_sources is None:")
        add(lines, 12, "return None")
        add(lines, 8, "gate_key, up_key = fused_sources")
        add(lines, 8, "gate = self._materialize_expert_bank_for_path(gate_key)")
        add(lines, 8, "up = self._materialize_expert_bank_for_path(up_key)")
        add(lines, 8, "if not isinstance(gate, mx.array) or not isinstance(up, mx.array):")
        add(lines, 12, "return None")
        add(lines, 8, "if gate.shape[:-2] != up.shape[:-2] or gate.shape[-1:] != up.shape[-1:]:")
        add(lines, 12, "return None")
        add(lines, 8, "concat_dim = -2 if len(gate.shape) >= 2 else -1")
        add(lines, 8, "bank = mx.concatenate([gate, up], axis=concat_dim)")
        add(lines, 8, "self._flat_tensors[bank_key] = bank")
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
        add(lines, 8, "if token in ('float32', 'fp32', 'single'): return mx.float32")
        add(lines, 8, "if token in ('float16', 'fp16', 'half'): return mx.float16")
        add(lines, 8, "if token in ('bfloat16', 'bf16'): return mx.bfloat16")
        add(lines, 8, "if token in ('int64', 'long'): return mx.int64")
        add(lines, 8, "if token in ('int32', 'int'): return mx.int32")
        add(lines, 8, "if token in ('bool', 'boolean'): return mx.bool_")
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
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _from_numpy(value):")
        add(lines, 8, "if isinstance(value, mx.array):")
        add(lines, 12, "return value")
        add(lines, 8, "import numpy as np")
        add(lines, 8, "return mx.array(np.asarray(value))")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _to_mlx(value, dtype=None):")
        add(lines, 8, "if value is None:")
        add(lines, 12, "return None")
        add(lines, 8, "if isinstance(value, mx.array):")
        add(lines, 12, "if dtype is not None and value.dtype != dtype:")
        add(lines, 16, "return value.astype(dtype)")
        add(lines, 12, "return value")
        add(lines, 8, "import numpy as np")
        add(lines, 8, "if hasattr(value, 'cpu'):")
        add(lines, 12, "value = value.cpu()")
        add(lines, 8, "return mx.array(np.asarray(value), dtype=dtype)")
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
        add(lines, 8, "if not isinstance(cond, mx.array):")
        add(lines, 12, "return yes if cond else no")
        add(lines, 8, "return mx.where(cond, yes, no)")
        add(lines, 4, "")
        add(lines, 4, "def _where_indices(self, x):")
        add(lines, 8, "indices = np.array(x).nonzero()")
        add(lines, 8, "import numpy as np")
        add(lines, 8, "return tuple(mx.array(index.astype('int64')) for index in indices)")
        add(lines, 4, "")
        add(lines, 4, "def _embedding(self, base, ids):")
        add(lines, 8, "weight = self._param(self._compose_path(base, 'weight'))")
        add(lines, 8, "if not isinstance(ids, mx.array):")
        add(lines, 12, "ids = self._to_mlx(ids, mx.int64)")
        add(lines, 8, "elif ids.dtype not in (mx.int64, mx.int32):")
        add(lines, 12, "ids = ids.astype(mx.int64)")
        add(lines, 8, "return weight[ids]")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _concat(*args, dim=None):")
        add(lines, 8, "if dim is None:")
        add(lines, 12, "*items, dim = args")
        add(lines, 8, "else:")
        add(lines, 12, "items = args")
        add(lines, 8, "if len(items) == 1 and isinstance(items[0], (list, tuple)):")
        add(lines, 12, "items = tuple(items[0])")
        add(lines, 8, "return mx.concatenate(items, axis=int(dim))")
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
        add(lines, 8, "return mx.topk(x, int(k), axis=int(dim))")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _index_add(x, index, src, dim=0):")
        add(lines, 8, "dim = int(dim)")
        add(lines, 8, "if dim < 0:")
        add(lines, 12, "dim += len(x.shape)")
        add(lines, 8, "if isinstance(index, mx.array) and len(index.shape) != len(src.shape):")
        add(lines, 12, "shape = [1] * len(src.shape)")
        add(lines, 12, "shape[dim] = index.shape[0]")
        add(lines, 12, "index = index.reshape(tuple(shape)).broadcast_to(src.shape)")
        add(lines, 8, "if dim == 0:")
        add(lines, 12, "return mx.scatter_add(x, index, src, axes=[0])")
        add(lines, 8, "return mx.scatter_add(x, index, src, axes=[dim])")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _linear(base, x, bias=False, transpose=False, expert=None, weight_leaf='weight', bias_leaf='bias'):")
        add(lines, 8, "raise RuntimeError('internal MLX _linear helper should not be called directly')")
        add(lines, 4, "")
        add(lines, 4, "def _expert_linear(self, base, x, expert_idx, bias=False, transpose=False, weight_leaf='weight', bias_leaf='bias'):")
        add(lines, 8, "weight = self._param(self._compose_path(base, weight_leaf))")
        add(lines, 8, "idx = expert_idx.astype(mx.int64) if isinstance(expert_idx, mx.array) else expert_idx")
        add(lines, 8, "selected_weight = weight[idx]")
        add(lines, 8, "bias_value = self._optional_param(self._compose_path(base, bias_leaf)) if bias else None")
        add(lines, 8, "selected_bias = bias_value[idx] if bias_value is not None else None")
        add(lines, 8, "weight_run = selected_weight.astype(x.dtype) if x.dtype != selected_weight.dtype and x.dtype in (mx.float32, mx.float16, mx.bfloat16) else selected_weight")
        add(lines, 8, "bias_run = selected_bias.astype(x.dtype) if selected_bias is not None and x.dtype != selected_bias.dtype else selected_bias")
        add(lines, 8, "if transpose:")
        add(lines, 12, "y = (x[..., None, :] @ weight_run).squeeze(-2)")
        add(lines, 8, "else:")
        add(lines, 12, "y = (x[..., None, :] @ weight_run.swapaxes(-1, -2)).squeeze(-2)")
        add(lines, 8, "return y + bias_run if bias_run is not None else y")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _gegelu(x, limit=None):")
        add(lines, 8, "if x.shape[-1] % 2 != 0: raise ValueError('gegelu requires even last dimension')")
        add(lines, 8, "x_gelu = x[..., ::2]")
        add(lines, 8, "x_linear = x[..., 1::2]")
        add(lines, 8, "if limit is not None:")
        add(lines, 12, "limit_val = float(limit)")
        add(lines, 12, "x_gelu = mx.clip(x_gelu, -limit_val, limit_val)")
        add(lines, 12, "x_linear = mx.clip(x_linear, -limit_val, limit_val)")
        add(lines, 8, "return x_gelu * mx.sigmoid(1.702 * x_gelu) * (x_linear + 1.0)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _xielu(x, alpha_p_raw, alpha_n_raw, beta_raw, eps_raw):")
        add(lines, 8, "beta = mx.full(x.shape, float(beta_raw)) if not isinstance(beta_raw, mx.array) else beta_raw")
        add(lines, 8, "alpha_p = mx.softplus(alpha_p_raw) if isinstance(alpha_p_raw, mx.array) else mx.softplus(mx.full(x.shape, float(alpha_p_raw)))")
        add(lines, 8, "alpha_n_base = mx.softplus(alpha_n_raw) if isinstance(alpha_n_raw, mx.array) else mx.softplus(mx.full(x.shape, float(alpha_n_raw)))")
        add(lines, 8, "alpha_n = beta + alpha_n_base")
        add(lines, 8, "eps = float(eps_raw.item()) if isinstance(eps_raw, mx.array) else float(eps_raw)")
        add(lines, 8, "return mx.where(x > 0, alpha_p * x * x + beta * x, (mx.exp(mx.minimum(x, eps)) - 1.0 - x) * alpha_n + beta * x)")

    def _emit_load_state_dict(self, lines: list[str]) -> None:
        add = self._add
        add(lines, 4, "def load_state_dict(self, state_dict, *, quantize=False):")
        add(lines, 8, "state_dict = self._materialize_state_aliases(state_dict)")
        add(lines, 8, "tensors = {}")
        add(lines, 8, "for k, v in state_dict.items():")
        add(lines, 12, "if isinstance(v, mx.array):")
        add(lines, 16, "tensors[str(k)] = v")
        add(lines, 12, "else:")
        add(lines, 16, "tensors[str(k)] = self._from_numpy(v)")
        add(lines, 8, "")
        for path, op_type in sorted(self._static_param_ops.items()):
            safe = _path_to_safe_attr(path)
            weight_key = f"{path}.weight"
            bias_key = f"{path}.bias"
            if op_type == "linear":
                add(lines, 8, f"if '{weight_key}' in tensors:")
                add(lines, 12, f"w = tensors['{weight_key}']")
                add(lines, 12, f"self._{safe} = nn.Linear(w.shape[1], w.shape[0])")
                add(lines, 12, f"self._{safe}.weight = w")
                add(lines, 12, f"if '{bias_key}' in tensors:")
                add(lines, 16, f"self._{safe}.bias = tensors['{bias_key}']")
                add(lines, 8, "")
            elif op_type == "embedding":
                add(lines, 8, f"if '{weight_key}' in tensors:")
                add(lines, 12, f"w = tensors['{weight_key}']")
                add(lines, 12, f"self._{safe} = nn.Embedding(w.shape[0], w.shape[1])")
                add(lines, 12, f"self._{safe}.weight = w")
                add(lines, 8, "")
            elif op_type in ("layernorm", "rmsnorm"):
                add(lines, 8, f"if '{weight_key}' in tensors:")
                add(lines, 12, f"w = tensors['{weight_key}']")
                add(lines, 12, f"self._{safe} = nn.LayerNorm(w.shape[0])")
                add(lines, 12, f"self._{safe}.weight = w")
                add(lines, 12, f"if '{bias_key}' in tensors:")
                add(lines, 16, f"self._{safe}.bias = tensors['{bias_key}']")
                add(lines, 8, "")
        self._emitted_module_attrs = [_path_to_safe_attr(p) for p in self._static_param_ops]
        add(lines, 8, "self._flat_tensors = tensors")
        add(lines, 8, "")
        add(lines, 8, "object.__setattr__(self, '_symbols', self._eval_symbols())")
        add(lines, 8, "")
        add(lines, 8, "if quantize:")
        add(lines, 12, "nn.quantize(self)")
        add(lines, 12, "self._quantized = True")

    def _param_expr_for_mlx_attr(
        self,
        path_operand: Any,
        *,
        local: set[str],
        symbols_dict: str,
    ) -> str | None:
        joined = _path_joined(path_operand)
        if joined is not None and joined in self._static_param_ops:
            safe = _path_to_safe_attr(joined)
            return f"_{safe}"
        return None

    def _emit_forward(self, lines: list[str]) -> None:
        main = self.modules_by_name[self.program.main_module]
        add = self._add
        add(lines, 4, "def _forward(self, input_ids=None, **inputs):")
        args: list[str] = []
        first_input = main.inputs[0].name if main.inputs else None
        for value in main.inputs:
            if value.name == "input_ids":
                add(lines, 8, "if input_ids is None:")
                add(lines, 12, "input_ids = inputs.get('input_ids')")
                add(lines, 8, "if input_ids is None:")
                add(lines, 12, "raise ValueError('Missing required input: input_ids')")
                add(lines, 8, "input_ids = self._to_mlx(input_ids, mx.int64)")
                args.append("input_ids")
            elif value.name == "use_cache":
                add(lines, 8, "use_cache = inputs.get('use_cache', None)")
                args.append("use_cache")
            elif value.name == "past_kv":
                add(lines, 8, "past_kv = inputs.get('past_kv', None)")
                args.append("past_kv")
            elif value.name == first_input:
                add(lines, 8, f"{value.name} = inputs.get({value.name!r}, input_ids)")
                if not (value.optional or isinstance(value.type_expr, TypeOptional)):
                    add(lines, 8, f"if {value.name} is None:")
                    add(lines, 12, f"raise ValueError('Missing required input: {value.name}')")
                add(lines, 8, f"{value.name} = self._to_mlx({value.name})")
                args.append(value.name)
            else:
                if value.optional or isinstance(value.type_expr, TypeOptional):
                    add(lines, 8, f"{value.name} = inputs.get({value.name!r}, None)")
                else:
                    add(lines, 8, f"if {value.name!r} not in inputs:")
                    add(lines, 12, f"raise ValueError('Missing required input: {value.name}')")
                    add(lines, 8, f"{value.name} = inputs[{value.name!r}]")
                add(lines, 8, f"{value.name} = self._to_mlx({value.name})")
                args.append(value.name)
        add(lines, 8, f"result = self.{self.method_names[main.name]}({', '.join(args)})")
        names = graph_main_output_names(self.program, main)
        if len(names) == 1:
            add(lines, 8, "return result[0]")
        else:
            add(lines, 8, f"return {{{', '.join(f'{name!r}: result[{idx}]' for idx, name in enumerate(names))}}}")
        add(lines, 4, "")
        add(lines, 4, "def forward(self, input_ids=None, **inputs):")
        add(lines, 8, "input_ids = self._to_mlx(input_ids, mx.int64) if input_ids is not None else None")
        add(lines, 8, "for _k, _v in list(inputs.items()):")
        add(lines, 12, "if _k in ('use_cache', 'past_kv'):")
        add(lines, 16, "continue")
        add(lines, 12, "inputs[_k] = self._to_mlx(_v)")
        add(lines, 8, "if self._compiled_fn is not None:")
        add(lines, 12, "result = self._compiled_fn(input_ids, **inputs)")
        add(lines, 12, "if isinstance(result, (list, tuple)):")
        add(lines, 16, f"return {{{', '.join(f'{name!r}: result[{idx}]' for idx, name in enumerate(names))}}}")
        add(lines, 12, "return result")
        add(lines, 8, "return self._forward(input_ids, **inputs)")
        add(lines, 4, "")
        add(lines, 4, "def compile(self, max_kv_length=2048):")
        add(lines, 8, "\"\"\"Compile _forward with mx.compile and warmup KV shapes 0..max_kv_length.\"\"\"")
        add(lines, 8, "if self._compiled_fn is not None:")
        add(lines, 12, "return self._compiled_fn")
        add(lines, 8, "self._compiled_fn = mx.compile(self._forward)")
        add(lines, 8, "prompt_ids = mx.zeros((1, 1), dtype=mx.int64)")
        add(lines, 8, "kv = None")
        add(lines, 8, "for length in range(1, max_kv_length + 1):")
        add(lines, 12, "inp = mx.array([[0]], dtype=mx.int64)")
        add(lines, 12, "result = self._compiled_fn(inp, past_kv=kv, use_cache=True)")
        add(lines, 12, "if isinstance(result, (list, tuple)):")
        add(lines, 16, "kv = result[1] if len(result) > 1 else None")
        add(lines, 12, "else:")
        add(lines, 16, "kv = result.get('new_kv') if isinstance(result, dict) else None")
        add(lines, 12, "if kv is None:")
        add(lines, 16, "break")
        add(lines, 8, "mx.eval(mx.array(0))")
        add(lines, 8, "return self._compiled_fn")

    def _emit_generate(self, lines: list[str]) -> None:
        add = self._add
        main = self.modules_by_name[self.program.main_module]
        input_names = {value.name for value in main.inputs}
        output_names = set(graph_main_output_names(self.program, main))
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
        is_cached_decoder = False
        is_decoder_only = not has_decoder_inputs
        add(lines, 4, "def generate(self, input_ids, max_new_tokens=20, **kwargs):")
        add(lines, 8, "input_ids = self._to_mlx(input_ids, mx.int64)")
        add(lines, 8, "def _logits(result):")
        add(lines, 12, "return result.get('logits') if isinstance(result, dict) else result")
        add(lines, 8, "def _next_id(logits):")
        add(lines, 12, "return logits[:, -1:, :].argmax(axis=-1).astype(mx.int64)")
        add(lines, 8, "def _ones_like_ids(ids):")
        add(lines, 12, "return mx.ones(ids.shape, dtype=mx.int64)")
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
        add(lines, 12, "eos = mx.array(eos_values, dtype=mx.int64).reshape((-1,))")
        add(lines, 12, "pad = int(eos_values[0] if pad_token_id is None else pad_token_id)")
        add(lines, 12, "finished = mx.zeros((batch_size, 1), dtype=mx.bool_)")
        add(lines, 12, "return eos, pad, finished")
        add(lines, 8, "def _apply_eos(next_id, eos, pad, finished):")
        add(lines, 12, "if eos is None: return next_id, finished")
        add(lines, 12, "raw_next = next_id")
        add(lines, 12, "next_id = mx.where(finished, mx.full(next_id.shape, pad, dtype=mx.int64), next_id)")
        add(lines, 12, "hit = (raw_next == eos.reshape((1, -1))).max(axis=1, keepdims=True)")
        add(lines, 12, "finished = finished | hit")
        add(lines, 12, "return next_id, finished")
        add(lines, 8, "def _all_done(finished):")
        add(lines, 12, "return finished is not None and bool(finished.min().item())")
        if is_cached_decoder:
            add(lines, 8, "out = input_ids")
            add(lines, 8, "limit = _generation_limit(out)")
            add(lines, 8, "eos, pad, finished = _eos_state(out.shape[0])")
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
            add(lines, 12, "result = self.forward(step_input, **forward_kwargs)")
            add(lines, 12, "if isinstance(result, dict): cache = result.get(" + repr(cache_output_name) + ", cache)")
            add(lines, 12, "next_id = _next_id(_logits(result))")
            add(lines, 12, "next_id, finished = _apply_eos(next_id, eos, pad, finished)")
            add(lines, 12, "out = mx.concatenate([out, next_id], axis=1)")
            if attention_name is not None:
                add(lines, 12, "attention_mask = mx.concatenate([attention_mask, _ones_like_ids(next_id)], axis=1)")
            add(lines, 12, "if _all_done(finished): break")
            add(lines, 8, "return out")
            return
        if is_decoder_only:
            add(lines, 8, "out = input_ids")
            add(lines, 8, "limit = _generation_limit(out)")
            add(lines, 8, "eos, pad, finished = _eos_state(out.shape[0])")
            if attention_name is not None:
                other = "attention_mask" if attention_name == "attn_mask" else "attn_mask"
                add(lines, 8, f"attention_mask = kwargs.pop({attention_name!r}, kwargs.pop({other!r}, None))")
                add(lines, 8, "if attention_mask is None: attention_mask = _ones_like_ids(out)")
                add(lines, 8, "else: attention_mask = self._to_mlx(attention_mask)")
            add(lines, 8, "for _ in range(limit):")
            add(lines, 12, "forward_kwargs = dict(kwargs)")
            if attention_name is not None:
                add(lines, 12, f"forward_kwargs[{attention_name!r}] = attention_mask")
            add(lines, 12, "result = self.forward(out, **forward_kwargs)")
            add(lines, 12, "next_id = _next_id(_logits(result))")
            add(lines, 12, "next_id, finished = _apply_eos(next_id, eos, pad, finished)")
            add(lines, 12, "out = mx.concatenate([out, next_id], axis=1)")
            if attention_name is not None:
                add(lines, 12, "attention_mask = mx.concatenate([attention_mask, _ones_like_ids(next_id)], axis=1)")
            add(lines, 12, "if _all_done(finished): break")
            add(lines, 8, "return out")
            return
        add(lines, 8, "decoder_input_ids = kwargs.pop('decoder_input_ids', None)")
        add(lines, 8, "if decoder_input_ids is None:")
        add(lines, 12, "start_id = kwargs.pop('decoder_start_token_id', self.config.get('decoder_start_token_id', self.config.get('pad_token_id', 0)))")
        add(lines, 12, "decoder_input_ids = mx.full((input_ids.shape[0], 1), int(start_id), dtype=mx.int64)")
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
        add(lines, 12, "result = self.forward(input_ids, **forward_kwargs)")
        add(lines, 12, "next_id = _next_id(_logits(result))")
        add(lines, 12, "next_id, finished = _apply_eos(next_id, eos, pad, finished)")
        add(lines, 12, "decoder_input_ids = mx.concatenate([decoder_input_ids, next_id], axis=1)")
        if decoder_attention_name is not None:
            add(lines, 12, "decoder_attention_mask = mx.concatenate([decoder_attention_mask, _ones_like_ids(next_id)], axis=1)")
        add(lines, 12, "if _all_done(finished): break")
        add(lines, 8, "return decoder_input_ids")

    def _primitive_expr(self, primitive: str, node: Any, *, local: set[str], symbols_dict: str) -> str:
        args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
        attrs = {k: self._operand_expr(v, local=local, symbols_dict=symbols_dict) for k, v in node.attrs.items()}
        if primitive == "embedding":
            path_operand = node.inputs[0]
            attr_expr = self._param_expr_for_mlx_attr(path_operand, local=local, symbols_dict=symbols_dict)
            if attr_expr is not None:
                return f"self.{attr_expr}({args[1]})"
            return f"self._embedding({args[0]}, {args[1]})"
        if primitive == "linear":
            bias = args[3] if len(args) > 3 else "False"
            transpose = args[4] if len(args) > 4 else "False"
            expert = args[5] if len(args) > 5 else "None"
            weight_leaf = args[6] if len(args) > 6 else "'weight'"
            bias_leaf = args[7] if len(args) > 7 else "'bias'"
            return (
                f"(lambda _w, _b: "
                f"(mx.addmm(_b, {args[1]}, _w.swapaxes(-1, -2)) if _b is not None else ({args[1]} @ _w.swapaxes(-1, -2)))"
                f"if not bool({transpose}) else "
                f"(mx.addmm(_b, {args[1]}, _w) if _b is not None else ({args[1]} @ _w)))"
                f"((lambda _w: (_w[int({expert})] if {expert} is not None else _w))(self._param(self._compose_path({args[0]}, {weight_leaf}))), "
                f"((lambda _b: (_b[int({expert})] if (_b is not None and {expert} is not None and len(_b.shape) >= 2) else _b))(self._optional_param(self._compose_path({args[0]}, {bias_leaf}))) if bool({bias}) else None))"
            )
        if primitive == "expert_linear":
            bias = args[4] if len(args) > 4 else "False"
            transpose = args[5] if len(args) > 5 else "False"
            weight_leaf = args[6] if len(args) > 6 else "'weight'"
            bias_leaf = args[7] if len(args) > 7 else "'bias'"
            return f"self._expert_linear({args[0]}, {args[1]}, {args[2]}, bias=bool({bias}), transpose=bool({transpose}), weight_leaf={weight_leaf}, bias_leaf={bias_leaf})"
        if primitive == "_mlx_sdpa":
            if len(args) < 6:
                raise ValueError("__mlx_sdpa expects q, k, v, additive_mask, scale, enable_gqa")
            scale = f"float({args[4]})" if args[4] != "None" else "None"
            if scale == "None":
                return (
                    f"mx.fast.scaled_dot_product_attention("
                    f"{args[0]}, {args[1]}, {args[2]}, "
                    f"mask={args[3]}, scale=1.0)"
                )
            return (
                f"mx.fast.scaled_dot_product_attention("
                f"{args[0]}, {args[1]}, {args[2]}, "
                f"mask={args[3]}, scale={scale})"
            )
        if primitive == "layernorm":
            path_operand = node.inputs[0]
            attr_expr = self._param_expr_for_mlx_attr(path_operand, local=local, symbols_dict=symbols_dict)
            if attr_expr is not None:
                return f"self.{attr_expr}({args[1]})"
            eps = args[2] if len(args) > 2 else "1e-5"
            weight_leaf = args[4] if len(args) > 4 else "'weight'"
            bias = args[5] if len(args) > 5 else "True"
            bias_leaf = args[6] if len(args) > 6 else "'bias'"
            return (
                f"mx.fast.layer_norm({args[1]}, "
                f"self._param(self._compose_path({args[0]}, {weight_leaf})), "
                f"(self._optional_param(self._compose_path({args[0]}, {bias_leaf})) if {bias} else None), "
                f"eps=float({eps}))"
            )
        if primitive == "rmsnorm":
            x = args[0]
            eps = args[1] if len(args) > 1 else "1e-6"
            return f"mx.fast.rms_norm({x}, None, float({eps}))"
        if primitive == "tensor_like":
            dtype = args[2] if len(args) > 2 else "None"
            return f"({args[0]}.astype(self._dtype_from_name({dtype}) or {args[1]}.dtype) if isinstance({args[0]}, mx.array) else mx.array({args[0]}, dtype=(self._dtype_from_name({dtype}) or {args[1]}.dtype)))"
        if primitive == "softmax":
            dim = args[1] if len(args) > 1 else "-1"
            dtype = args[2] if len(args) > 2 else "None"
            if dtype == "None":
                return f"mx.softmax({args[0]}, axis=int({dim}))"
            return f"mx.softmax({args[0]}.astype(self._dtype_from_name({dtype})) if {dtype} != None else {args[0]}, axis=int({dim}))"
        if primitive == "topk":
            return f"self._topk({args[0]}, {args[1]}, dim={args[2]}, largest={args[3]}, sorted_={args[4]})"
        if primitive == "concat":
            if "dim" in attrs:
                return f"self._concat({', '.join(args)}, dim={attrs['dim']})"
            return f"self._concat({', '.join(args[:-1])}, dim={args[-1]})"
        if primitive == "slice":
            ndim = f"len(({args[0]}).shape)"
            dim = f"(({args[1]})+{ndim})%{ndim}"
            return f"({args[0]})[(slice(None),)*({dim}) + (slice({args[2]}, {args[3]}),) + (slice(None),)*({ndim}-({dim})-1)]"
        if primitive == "_mlx_rope":
            return f"mx.fast.rope({args[0]}, dims={args[0]}.shape[-1], traditional=bool({args[1]}))"
        if primitive == "params_has_root":
            return f"any(k == {args[0]} or k.startswith(str({args[0]}) + '.') for k in self._flat_tensors)"
        if primitive.startswith("config_") or primitive in {"params_param"}:
            return super()._primitive_expr(primitive, node, local=local, symbols_dict=symbols_dict)
        simple = {
            "reshape": lambda: f"{args[0]}.reshape(tuple(int(x) for x in {args[1]}))",
            "arange": lambda: f"mx.arange(int({args[1]}), dtype=mx.int64)" if args[2] == 'None' else f"mx.arange(int({args[1]}), int({args[2]}), dtype=mx.int64)",
            "chunk": lambda: f"mx.split({args[0]}, indices_or_sections=int({args[2] if len(args) > 2 else attrs.get('parts', '1')}), axis=int({args[1] if len(args) > 1 else attrs.get('dim', '-1')}))",
            "split": lambda: f"mx.split({args[0]}, indices_or_sections=[int(x) for x in {args[2] if len(args) > 2 else attrs.get('sizes', '[]')}], axis=int({args[1] if len(args) > 1 else attrs.get('dim', '-1')}))",
            "sum": lambda: f"{args[0]}.sum(axis=int({args[1] if len(args) > 1 else '-1'}), keepdims=bool({args[2] if len(args) > 2 else 'False'}))",
            "expand": lambda: f"mx.broadcast_to({args[0]}, tuple(int(x) for x in {args[1]}))",
            "permute": lambda: f"mx.transpose({args[0]}, axes=tuple(int(x) for x in {args[1]}))",
            "transpose": lambda: f"mx.swapaxes({args[0]}, int({args[1]}), int({args[2]}))",
            "unsqueeze": lambda: f"self._unsqueeze({args[0]}, {args[1]})",
            "repeat": lambda: f"{args[0]}.repeat(int({args[1]}), axis=(int({args[2]}) if int({args[2]}) >= 0 else int({args[2]}) + len({args[0]}.shape)))",
            "matmul": lambda: f"({args[0]} @ {args[1]})",
            "where": lambda: f"self._where({args[0]}, {args[1]}, {args[2]})",
            "where_indices": lambda: f"self._where_indices({args[0]})",
            "require": lambda: f"self._require_value({args[0]})",
            "gather": lambda: f"mx.take({args[0]}, {args[1]}.astype(mx.int64), axis=int({args[2] if len(args) > 2 else '-1'}))",
            "scatter": lambda: f"mx.scatter({args[0]}, {args[1]}, {args[2]}, axis=int({args[3] if len(args) > 3 else '-1'}))",
            "index_add": lambda: f"self._index_add({args[0]}, {args[1]}, {args[2]}, {args[3] if len(args) > 3 else '0'})",
            "le": lambda: f"({args[0]} <= {args[1]})",
            "eq": lambda: f"self._eq({args[0]}, {args[1]})",
            "and": lambda: f"({args[0]} & {args[1]})",
            "add": lambda: f"self._binary_add({args[0]}, {args[1]})",
            "mul": lambda: f"({args[0]} * {args[1]})",
            "div": lambda: f"({args[0]} / {args[1]})",
            "pow": lambda: f"mx.power({args[0]}, {args[1]})",
            "floor": lambda: f"mx.floor({args[0]}) if isinstance({args[0]}, mx.array) else int({args[0]} // 1)",
            "sqrt": lambda: f"mx.sqrt({args[0]}) if isinstance({args[0]}, mx.array) else ({args[0]} ** 0.5)",
            "sin": lambda: f"mx.sin({args[0]}) if isinstance({args[0]}, mx.array) else __import__('math').sin(float({args[0]}))",
            "cos": lambda: f"mx.cos({args[0]}) if isinstance({args[0]}, mx.array) else __import__('math').cos(float({args[0]}))",
            "exp": lambda: f"mx.exp({args[0]}) if isinstance({args[0]}, mx.array) else __import__('math').exp(float({args[0]}))",
            "log": lambda: f"mx.log({args[0]}) if isinstance({args[0]}, mx.array) else __import__('math').log(float({args[0]}))",
            "cast": lambda: f"{args[0]}.astype(self._dtype_from_name({args[1]}) or {args[0]}.dtype)",
            "cast_like": lambda: f"{args[0]}.astype({args[1]}.dtype)",
            "dtype_value": lambda: f"(lambda _x: self._dtype_value(_x.dtype, {args[1]}))(self._value({args[0]}))",
            "cumsum": lambda: f"{args[0]}.cumsum(axis=int({args[1] if len(args) > 1 else '-1'}))",
            "empty_like": lambda: f"(lambda _x: mx.zeros(_x.shape, dtype=_x.dtype))(self._value({args[0]}))",
            "fill": lambda: f"(lambda _x: mx.full(_x.shape, {args[1]}, dtype=(_x.dtype if self._dtype_from_name({args[2] if len(args) > 2 else 'None'}) is None else self._dtype_from_name({args[2] if len(args) > 2 else 'None'}))))(self._value({args[0]}))",
            "empty": lambda: f"mx.zeros(tuple(int(x) for x in {args[1]}), dtype=((self._dtype_from_name({args[2] if len(args) > 2 else 'None'}) if {args[2] if len(args) > 2 else 'None'} is not None else None) or {args[0]}.dtype))",
            "zeros": lambda: f"mx.zeros(tuple(int(x) for x in {args[1]}), dtype=((self._dtype_from_name({args[2] if len(args) > 2 else 'None'}) if {args[2] if len(args) > 2 else 'None'} is not None else None) or {args[0]}.dtype))",
            "full": lambda: f"mx.full(tuple(int(x) for x in {args[1]}), {args[2]}, dtype=((self._dtype_from_name({args[3] if len(args) > 3 else 'None'}) if {args[3] if len(args) > 3 else 'None'} is not None else None) or {args[0]}.dtype))",
            "zeros_like": lambda: f"(lambda _x: mx.zeros(_x.shape, dtype=_x.dtype))(self._value({args[0]}))",
            "activations_tanh": lambda: f"mx.tanh({args[0]})",
            "activations_silu": lambda: f"mx.sigmoid({args[0]}) * {args[0]}",
            "activations_sigmoid": lambda: f"mx.sigmoid({args[0]})",
            "activations_swiglu": lambda: f"(mx.sigmoid({args[0]}) * {args[0]} * {args[0]})",
            "l2norm": lambda: f"({args[0]} * mx.rsqrt(({args[0]} * {args[0]}).mean(axis=-1, keepdims=True) + float({args[1] if len(args) > 1 else '1e-6'})))",
            "activations_relu": lambda: f"mx.maximum({args[0]}, 0)",
            "activations_relu2": lambda: f"(mx.maximum({args[0]}, 0) * mx.maximum({args[0]}, 0))",
            "activations_gelu": lambda: f"nn.gelu({args[0]})",
            "activations_gelu_new": lambda: f"(0.5 * {args[0]} * (1.0 + mx.tanh(0.7978845608028654 * ({args[0]} + 0.044715 * {args[0]} * {args[0]} * {args[0]}))))",
            "activations_gelu_pytorch_tanh": lambda: f"(0.5 * {args[0]} * (1.0 + mx.tanh(0.7978845608028654 * ({args[0]} + 0.044715 * {args[0]} * {args[0]} * {args[0]}))))",
            "activations_gegelu": lambda: f"self._gegelu({args[0]}, {args[1] if len(args) > 1 else 'None'})",
            "activations_xielu": lambda: f"self._xielu({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]})",
            "list_init": lambda: "[]",
            "list_append": lambda: f"([*({args[0]} if {args[0]} is not None else []), {args[1]}])",
            "list_index": lambda: f"{args[0]}[int({args[1]})]",
            "list_length": lambda: f"(0 if {args[0]} is None else len({args[0]}))",
            "shape": lambda: f"list(self._value({args[0]}).shape)",
            "tensor_size": lambda: f"self._value({args[0]}).shape[int({args[1]})]",
        }
        if primitive == "clamp":
            min_value = args[1] if len(args) > 1 else attrs.get("min", "None")
            max_value = args[2] if len(args) > 2 else attrs.get("max", "None")
            if min_value == "None" and max_value == "None":
                return args[0]
            if min_value == "None":
                return f"mx.minimum({args[0]}, {max_value})"
            if max_value == "None":
                return f"mx.maximum({args[0]}, {min_value})"
            return f"mx.clip({args[0]}, {min_value}, {max_value})"
        if primitive in simple:
            return simple[primitive]()
        raise NotImplementedError(f"direct codegen2-mlx unsupported graph op {primitive!r}")

    @staticmethod
    def _dtype_value(dtype: Any, kind: str) -> float:
        token = str(dtype)
        if "float16" in token or "half" in token:
            values = {"min": -65504.0, "max": 65504.0, "eps": 0.0009765625, "tiny": 6.103515625e-05}
        elif "float64" in token or "double" in token:
            values = {"min": -1.7976931348623157e308, "max": 1.7976931348623157e308, "eps": 2.220446049250313e-16, "tiny": 2.2250738585072014e-308}
        else:
            values = {"min": -3.4028234663852886e38, "max": 3.4028234663852886e38, "eps": 1.1920928955078125e-07, "tiny": 1.1754943508222875e-38}
        values = {**values, "inf": float('inf'), "-inf": float('-inf')}
        return float(values[str(kind)])

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
        op_expr = f"mx.fast.layer_norm({args[1]}, {weight_name}, {bias_name}, eps={eps})"
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
        return False


def emit_model_code_from_graph_ir(
    graph: GraphProgram,
    *,
    class_name: str = "AxonMlxModel",
    model_config: dict[str, Any] | None = None,
    profile: bool = False,
) -> str:
    validate_graph_program(graph)
    unsupported = non_obvious_mlx_ops(graph)
    if unsupported:
        table = mlx_op_table_markdown(graph)
        raise NotImplementedError(
            "codegen2-mlx cannot emit this Graph IR yet.\n"
            "Unsupported Graph IR ops:\n"
            f"{table}"
        )
    emitter = _DirectMlxEmitter(program=graph, class_name=class_name, profile=profile)
    body = emitter.emit()
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "import mlx.core as mx",
            "import mlx.nn as nn",
            "from safetensors import safe_open",
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


def torch_state_dict_to_mlx(state_dict: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx
    import torch

    return {k: mx.array(v.numpy()) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}


OBVIOUS_MLX_PRIMITIVES = SUPPORTED_MLX_PRIMITIVES

__all__ = [
    "NON_OBVIOUS_MLX_OPS",
    "OBVIOUS_MLX_PRIMITIVES",
    "SHARED_COMMON_PRIMITIVES",
    "SUPPORTED_MLX_PRIMITIVES",
    "MlxUnsupportedOp",
    "emit_model_code_from_graph_ir",
    "non_obvious_mlx_ops",
    "mlx_op_table_markdown",
    "torch_state_dict_to_mlx",
]
