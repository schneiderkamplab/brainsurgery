from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..ast import DimExprBinary, TypeBool, TypeDim, TypeFloat, TypeInt, TypeOptional
from ..ast import TypeTensor
from ..codegen2_common import normalize_primitive_op
from ..codegen2_torch.core import (
    _DirectTorchEmitter,
    _dim_ident,
    _graph_uses_expert_linear,
    _is_static_mask_type,
    _static_mask_capacity_dim,
    _static_mask_capacity_expr,
    _state_key_filter_prefixes,
    graph_main_output_names,
)
from ..graph_ir import GraphExpr, GraphLiteral, GraphProgram, validate_graph_program

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

SUPPORTED_JAX_PRIMITIVES: frozenset[str] = frozenset({
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
    "conv1d",
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
    "_jax_expert_packed_swiglu_ffn",
    "_jax_expert_swiglu_ffn",
    "_jax_sdpa",
    "_jax_selected_expert_clamped_packed_swiglu_ffn",
    "_jax_selected_expert_packed_gegelu_ffn",
    "_jax_selected_expert_packed_swiglu_ffn",
    "_jax_selected_expert_relu2_ffn",
    "_jax_selected_expert_swiglu_ffn",
    "_jax_swiglu_ffn",
    "_jax_weighted_topk_sum",
    "_jax_rope",
})

NON_OBVIOUS_JAX_OPS: dict[str, str] = {}


@dataclass(frozen=True)
class JaxUnsupportedOp:
    op: str
    count: int
    reason: str


def _normalize_primitive_op(name: str) -> str:
    return normalize_primitive_op(name)


def non_obvious_jax_ops(graph: GraphProgram) -> tuple[JaxUnsupportedOp, ...]:
    counts: Counter[str] = Counter()
    module_names = {module.name for module in graph.modules}
    for module in graph.modules:
        for node in module.nodes:
            op = node.op.name
            if op.startswith("core.") or op.startswith("core.binary.") or op in module_names:
                continue
            primitive = _normalize_primitive_op(op)
            if primitive in SHARED_COMMON_PRIMITIVES or primitive in SUPPORTED_JAX_PRIMITIVES:
                continue
            counts[primitive] += 1
    return tuple(
        JaxUnsupportedOp(
            op=op,
            count=count,
            reason=NON_OBVIOUS_JAX_OPS.get(op, "no JAX lowering classified yet"),
        )
        for op, count in sorted(counts.items())
    )


def jax_op_table_markdown(graph: GraphProgram) -> str:
    rows = non_obvious_jax_ops(graph)
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


def _graph_path_pattern_key(path: Any) -> str:
    return ".".join(part for part in path.parts if part)


def _packed_parameter_spec_payload(packed: Any) -> dict[str, Any]:
    return {
        "output": _graph_path_pattern_key(packed.output),
        "inputs": tuple(_graph_path_pattern_key(item) for item in packed.inputs),
        "dim": int(packed.dim),
        "mode": "cat",
        "remove_inputs": bool(packed.remove_inputs),
    }


def _jax_path_pattern_regex(pattern: str) -> Any:
    pieces: list[str] = []
    cursor = 0
    used: set[str] = set()
    for match in re.finditer(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", pattern):
        pieces.append(re.escape(pattern[cursor : match.start()]))
        name = match.group(1)
        if name in used:
            pieces.append(f"(?P={name})")
        else:
            pieces.append(f"(?P<{name}>[^.]+)")
            used.add(name)
        cursor = match.end()
    pieces.append(re.escape(pattern[cursor:]))
    return re.compile("^" + "".join(pieces) + "$")


def _jax_format_path_pattern(pattern: str, values: dict[str, str]) -> str:
    out = pattern
    for key, value in values.items():
        out = out.replace("{" + key + "}", str(value))
    return out


def _jax_materialize_joined_parameter(
    state: dict[str, Any],
    output_key: str,
    input_keys: list[str],
    *,
    dim: int,
    mode: str,
    remove_inputs: bool,
) -> Any | None:
    import jax
    import jax.numpy as jnp

    existing = state.get(output_key)
    if isinstance(existing, jax.Array):
        return existing
    tensors = [state.get(key) for key in input_keys]
    if not tensors or not all(isinstance(item, jax.Array) for item in tensors):
        return None
    if mode == "cat":
        joined = jnp.concatenate(tensors, axis=int(dim))
    elif mode == "stack":
        joined = jnp.stack(tensors, axis=int(dim))
    else:
        raise ValueError(f"unknown parameter join mode {mode!r}")
    state[output_key] = joined
    if remove_inputs:
        for key in input_keys:
            state.pop(key, None)
    return joined


def _jax_materialize_packed_parameters(
    state: dict[str, Any],
    specs: tuple[dict[str, Any], ...],
    *,
    target_key: str | None = None,
) -> None:
    import jax

    for spec in specs:
        output_pattern = str(spec["output"])
        regex = _jax_path_pattern_regex(output_pattern)
        candidates: list[tuple[str, dict[str, str]]] = []
        if target_key is not None:
            match = regex.match(str(target_key))
            if match is not None:
                candidates.append((str(target_key), match.groupdict()))
        else:
            literal = "{" not in output_pattern
            if literal:
                candidates.append((output_pattern, {}))
            else:
                keys = list(state)
                for key in keys:
                    match = regex.match(str(key))
                    if match is not None:
                        candidates.append((str(key), match.groupdict()))
                input_patterns = tuple(str(item) for item in spec["inputs"])
                if input_patterns:
                    input_regex = _jax_path_pattern_regex(input_patterns[0])
                    for key in keys:
                        match = input_regex.match(str(key))
                        if match is None:
                            continue
                        values = match.groupdict()
                        output_key = _jax_format_path_pattern(output_pattern, values)
                        candidates.append((output_key, values))
        seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        for output_key, values in candidates:
            candidate_key = (output_key, tuple(sorted(values.items())))
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            if isinstance(state.get(output_key), jax.Array):
                continue
            input_keys = [_jax_format_path_pattern(str(item), values) for item in spec["inputs"]]
            _jax_materialize_joined_parameter(
                state,
                output_key,
                input_keys,
                dim=int(spec["dim"]),
                mode=str(spec.get("mode", "cat")),
                remove_inputs=bool(spec.get("remove_inputs", True)),
            )


class _DirectJaxEmitter(_DirectTorchEmitter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._static_param_ops: dict[str, str] = _collect_static_param_paths(self.program)
        self._emitted_module_attrs: list[str] = []
        self.state_key_filter_prefixes = _state_key_filter_prefixes(self.program)
        self.needs_expert_banks = _graph_uses_expert_linear(self.program)

    def emit(self) -> str:
        return super().emit().replace(f"class {self.class_name}(nn.Module):", f"class {self.class_name}:")

    def _emit_common(self, lines: list[str]) -> None:
        add = self._add
        add(lines, 4, "def __init__(self, state_dict: dict[str, jax.Array], config: dict | None = None, param_devices=None):")
        add(lines, 8, "object.__setattr__(self, '_flat_tensors', {})")
        add(lines, 8, "object.__setattr__(self, '_path_cache', {})")
        add(lines, 8, "object.__setattr__(self, '_compiled_fn', None)")
        add(lines, 8, "object.__setattr__(self, '_compiled_cached_fn', None)")
        add(lines, 8, "object.__setattr__(self, '_compiled_seq2seq_by_decoder_length', {})")
        add(lines, 8, "object.__setattr__(self, 'config', dict(({} if _MODEL_CONFIG is None else _MODEL_CONFIG) if config is None else config))")
        add(lines, 8, "jit_default = str(os.environ.get('AXON_JAX_ENABLE_JIT', '1')).strip().lower()")
        add(lines, 8, "object.__setattr__(self, '_jit_enabled', jit_default not in {'0', 'false', 'no', 'off'})")
        add(lines, 8, "object.__setattr__(self, '_quantized', False)")
        add(lines, 8, "object.__setattr__(self, '_profile_enabled', False)")
        add(lines, 8, "object.__setattr__(self, '_profile_records', {})")
        add(lines, 8, "object.__setattr__(self, '_symbols', {})")
        add(lines, 8, "self.load_state_dict(state_dict, param_devices=param_devices)")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def from_state_dict(cls, state_dict, *, graph=None, model_config=None, param_devices=None):")
        add(lines, 8, "return cls(state_dict, config=_MODEL_CONFIG if model_config is None else model_config, param_devices=param_devices)")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def from_safetensors(cls, safetensors_files, *, model_config=None, dtype=None):")
        add(lines, 8, "state_dict = {}")
        add(lines, 8, "dtype_name = None if dtype is None else str(dtype).removeprefix('torch.').lower()")
        add(lines, 8, "dtype_map = {")
        add(lines, 12, "'float32': jnp.float32, 'fp32': jnp.float32, 'single': jnp.float32,")
        add(lines, 12, "'float16': jnp.float16, 'fp16': jnp.float16, 'half': jnp.float16,")
        add(lines, 12, "'bfloat16': jnp.bfloat16, 'bf16': jnp.bfloat16,")
        add(lines, 8, "}")
        add(lines, 8, "target_dtype = None if dtype_name in (None, '', 'none', 'null', 'default') else dtype_map[dtype_name]")
        add(lines, 8, "for path in safetensors_files:")
        add(lines, 12, "with safe_open(str(path), framework='pt') as f:")
        add(lines, 16, "for key in f.keys():")
        add(lines, 20, "t = f.get_tensor(key)")
        add(lines, 20, "if t.is_floating_point() and target_dtype is not None:")
        add(lines, 24, "state_dict[str(key)] = jnp.asarray(t.float().numpy(), dtype=target_dtype)")
        add(lines, 20, "elif t.dtype == __import__('torch').bfloat16:")
        add(lines, 24, "state_dict[str(key)] = jnp.asarray(t.float().numpy())")
        add(lines, 20, "else:")
        add(lines, 24, "state_dict[str(key)] = jnp.asarray(t.numpy())")
        add(lines, 8, "return cls(state_dict, config=_MODEL_CONFIG if model_config is None else model_config)")
        add(lines, 4, "")
        self._emit_load_state_dict(lines)
        add(lines, 4, "")
        add(lines, 4, "def enable_jit(self, enabled=True, reset=True):")
        add(lines, 8, "self._jit_enabled = bool(enabled)")
        add(lines, 8, "if reset:")
        add(lines, 12, "self._compiled_fn = None")
        add(lines, 12, "self._compiled_cached_fn = None")
        add(lines, 12, "self._compiled_seq2seq_by_decoder_length = {}")
        add(lines, 8, "return self")
        add(lines, 4, "")
        add(lines, 4, "def enable_profile(self, enabled=True, *, cuda=True, reset=True):")
        add(lines, 8, "del cuda")
        add(lines, 8, "self._profile_enabled = bool(enabled)")
        add(lines, 8, "if reset:")
        add(lines, 12, "self._profile_records = {}")
        add(lines, 8, "return self")
        add(lines, 4, "")
        add(lines, 4, "def _profile_call(self, name, fn, *args, **kwargs):")
        add(lines, 8, "if not self._profile_enabled:")
        add(lines, 12, "return fn(*args, **kwargs)")
        add(lines, 8, "import time")
        add(lines, 8, "start = time.perf_counter()")
        add(lines, 8, "result = fn(*args, **kwargs)")
        add(lines, 8, "if hasattr(result, 'block_until_ready'):")
        add(lines, 12, "result.block_until_ready()")
        add(lines, 8, "elapsed = time.perf_counter() - start")
        add(lines, 8, "count, total = self._profile_records.get(name, (0, 0.0))")
        add(lines, 8, "self._profile_records[name] = (count + 1, total + elapsed)")
        add(lines, 8, "return result")
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
        add(lines, 4, "def _forward_with_state(self, state_tensors, input_ids=None, **inputs):")
        add(lines, 8, "old_tensors = self._flat_tensors")
        add(lines, 8, "self._flat_tensors = state_tensors")
        add(lines, 8, "try:")
        add(lines, 12, "return self._forward(input_ids, **inputs)")
        add(lines, 8, "finally:")
        add(lines, 12, "self._flat_tensors = old_tensors")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _value(value):")
        add(lines, 8, "while isinstance(value, (tuple, list)) and len(value) == 1:")
        add(lines, 12, "value = value[0]")
        add(lines, 8, "return value")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _is_jax(value):")
        add(lines, 8, "return isinstance(value, jax.Array)")
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
        add(lines, 4, f"_HAS_EXPERT_LINEAR = {bool(self.needs_expert_banks)!r}")
        add(lines, 4, f"_STATE_KEY_FILTER_PREFIXES = {self.state_key_filter_prefixes!r}")
        add(lines, 4, f"_PACKED_PARAMETER_SPECS = {tuple(_packed_parameter_spec_payload(item) for item in self.program.packed_parameters)!r}")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _packed_parameter_candidates(spec, keys):")
        add(lines, 8, "output_pattern = str(spec['output'])")
        add(lines, 8, "regex = _jax_path_pattern_regex(output_pattern)")
        add(lines, 8, "candidates = []")
        add(lines, 8, "literal = '{' not in output_pattern")
        add(lines, 8, "if literal:")
        add(lines, 12, "candidates.append((output_pattern, {}))")
        add(lines, 8, "else:")
        add(lines, 12, "for key in keys:")
        add(lines, 16, "match = regex.match(str(key))")
        add(lines, 16, "if match is not None:")
        add(lines, 20, "candidates.append((str(key), match.groupdict()))")
        add(lines, 12, "input_patterns = tuple(str(item) for item in spec['inputs'])")
        add(lines, 12, "if input_patterns:")
        add(lines, 16, "input_regex = _jax_path_pattern_regex(input_patterns[0])")
        add(lines, 16, "for key in keys:")
        add(lines, 20, "match = input_regex.match(str(key))")
        add(lines, 20, "if match is None:")
        add(lines, 24, "continue")
        add(lines, 20, "values = match.groupdict()")
        add(lines, 20, "candidates.append((_jax_format_path_pattern(output_pattern, values), values))")
        add(lines, 8, "seen = set()")
        add(lines, 8, "out = []")
        add(lines, 8, "for output_key, values in candidates:")
        add(lines, 12, "candidate_key = (output_key, tuple(sorted(values.items())))")
        add(lines, 12, "if candidate_key in seen:")
        add(lines, 16, "continue")
        add(lines, 12, "seen.add(candidate_key)")
        add(lines, 12, "out.append((output_key, values))")
        add(lines, 8, "return out")
        add(lines, 4, "")
        add(lines, 4, "def _materialize_packed_parameters_from_state_dict(self, state_dict, tensors):")
        add(lines, 8, "available_keys = {str(key) for key in state_dict}")
        add(lines, 8, "skipped_sources = set()")
        add(lines, 8, "for spec in self._PACKED_PARAMETER_SPECS:")
        add(lines, 12, "if not bool(spec.get('remove_inputs', True)):")
        add(lines, 16, "continue")
        add(lines, 12, "for output_key, values in self._packed_parameter_candidates(spec, available_keys | set(tensors)):")
        add(lines, 16, "if output_key in tensors:")
        add(lines, 20, "continue")
        add(lines, 16, "input_keys = [_jax_format_path_pattern(str(item), values) for item in spec['inputs']]")
        add(lines, 16, "if not input_keys or not all((key in tensors) or (key in state_dict) for key in input_keys):")
        add(lines, 20, "continue")
        add(lines, 16, "items = []")
        add(lines, 16, "for key in input_keys:")
        add(lines, 20, "items.append(tensors[key] if key in tensors else self._state_array_from_numpy(state_dict[key]))")
        add(lines, 16, "mode = str(spec.get('mode', 'cat'))")
        add(lines, 16, "dim = int(spec['dim'])")
        add(lines, 16, "if mode == 'cat':")
        add(lines, 20, "joined = jnp.concatenate(items, axis=dim)")
        add(lines, 16, "elif mode == 'stack':")
        add(lines, 20, "joined = jnp.stack(items, axis=dim)")
        add(lines, 16, "else:")
        add(lines, 20, "raise ValueError(f'unknown parameter join mode {mode!r}')")
        add(lines, 16, "tensors[output_key] = joined")
        add(lines, 16, "skipped_sources.update(input_keys)")
        add(lines, 16, "del items")
        add(lines, 8, "return skipped_sources")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _keep_state_key(cls, key):")
        add(lines, 8, "prefixes = cls._STATE_KEY_FILTER_PREFIXES")
        add(lines, 8, "if not prefixes:")
        add(lines, 12, "return True")
        add(lines, 8, "key = str(key)")
        add(lines, 8, "for prefix in prefixes:")
        add(lines, 12, "if key == prefix or key.startswith(prefix + '.'):")
        add(lines, 16, "return True")
        add(lines, 8, "return False")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _filter_state_dict(cls, state_dict):")
        add(lines, 8, "if not cls._STATE_KEY_FILTER_PREFIXES:")
        add(lines, 12, "return state_dict")
        add(lines, 8, "return {key: value for key, value in state_dict.items() if cls._keep_state_key(key)}")
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
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _drop_collapsed_numeric_state_aliases(cls, state_dict):")
        add(lines, 8, "for key in list(state_dict):")
        add(lines, 12, "value = state_dict.get(key)")
        add(lines, 12, "value_shape = tuple(getattr(value, 'shape', ()))")
        add(lines, 12, "for collapsed_key, index, _segment_index in cls._collapsed_numeric_segments(str(key)):")
        add(lines, 16, "bank = state_dict.get(collapsed_key)")
        add(lines, 16, "bank_shape = tuple(getattr(bank, 'shape', ()))")
        add(lines, 16, "if not bank_shape or int(index) >= int(bank_shape[0]):")
        add(lines, 20, "continue")
        add(lines, 16, "if tuple(bank_shape[1:]) == value_shape:")
        add(lines, 20, "state_dict.pop(key, None)")
        add(lines, 20, "break")
        add(lines, 8, "return state_dict")
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
        add(lines, 8, "if isinstance(existing, jax.Array):")
        add(lines, 12, "return existing")
        add(lines, 8, "ordered_keys = self._keys_for_collapsed_bank(bank_key)")
        add(lines, 8, "if ordered_keys:")
        add(lines, 12, "ordered = [self._flat_tensors[key] for key in ordered_keys]")
        add(lines, 12, "first_shape = ordered[0].shape")
        add(lines, 12, "if all(isinstance(t, jax.Array) and t.shape == first_shape for t in ordered):")
        add(lines, 16, "bank = jnp.stack(ordered, axis=0) if len(ordered) > 1 else ordered[0].reshape((1,) + tuple(ordered[0].shape))")
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
        add(lines, 8, "if not isinstance(gate, jax.Array) or not isinstance(up, jax.Array):")
        add(lines, 12, "return None")
        add(lines, 8, "if gate.shape[:-2] != up.shape[:-2] or gate.shape[-1:] != up.shape[-1:]:")
        add(lines, 12, "return None")
        add(lines, 8, "concat_dim = -2 if len(gate.shape) >= 2 else -1")
        add(lines, 8, "bank = jnp.concatenate([gate, up], axis=concat_dim)")
        add(lines, 8, "self._flat_tensors[bank_key] = bank")
        add(lines, 8, "return bank")
        add(lines, 4, "")
        add(lines, 4, "def _materialize_expert_banks(self):")
        add(lines, 8, "return None")
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
        add(lines, 8, "if token in ('float32', 'fp32', 'single'): return jnp.float32")
        add(lines, 8, "if token in ('float16', 'fp16', 'half'): return jnp.float16")
        add(lines, 8, "if token in ('bfloat16', 'bf16'): return jnp.bfloat16")
        add(lines, 8, "if token in ('int64', 'long'): return jnp.int32")
        add(lines, 8, "if token in ('int32', 'int'): return jnp.int32")
        add(lines, 8, "if token in ('bool', 'boolean'): return jnp.bool_")
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
        add(lines, 8, "if isinstance(value, jax.Array):")
        add(lines, 12, "return value")
        add(lines, 8, "import numpy as np")
        add(lines, 8, "return jnp.asarray(np.asarray(value))")
        add(lines, 4, "")
        add(lines, 4, "def _state_array_from_numpy(self, value):")
        add(lines, 8, "if isinstance(value, jax.Array):")
        add(lines, 12, "return value")
        add(lines, 8, "import numpy as np")
        add(lines, 8, "return jnp.asarray(np.asarray(value))")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _resolve_param_devices(param_devices):")
        add(lines, 8, "if not param_devices:")
        add(lines, 12, "return ()")
        add(lines, 8, "available = list(jax.devices())")
        add(lines, 8, "out = []")
        add(lines, 8, "for item in param_devices:")
        add(lines, 12, "if hasattr(item, 'platform'):")
        add(lines, 16, "out.append(item)")
        add(lines, 16, "continue")
        add(lines, 12, "text = str(item).strip().lower()")
        add(lines, 12, "if text in {'', 'none'}:")
        add(lines, 16, "continue")
        add(lines, 12, "if text.startswith('cuda:') or text.startswith('gpu:'):")
        add(lines, 16, "index = int(text.split(':', 1)[1])")
        add(lines, 16, "gpu_devices = [device for device in available if device.platform in {'gpu', 'cuda'}]")
        add(lines, 16, "out.append(gpu_devices[index])")
        add(lines, 12, "elif text == 'cuda' or text == 'gpu':")
        add(lines, 16, "gpu_devices = [device for device in available if device.platform in {'gpu', 'cuda'}]")
        add(lines, 16, "out.append(gpu_devices[0])")
        add(lines, 12, "elif text.startswith('cpu:') or text == 'cpu':")
        add(lines, 16, "cpu_devices = [device for device in available if device.platform == 'cpu']")
        add(lines, 16, "out.append(cpu_devices[int(text.split(':', 1)[1])] if ':' in text else cpu_devices[0])")
        add(lines, 12, "else:")
        add(lines, 16, "out.append(available[int(text)])")
        add(lines, 8, "return tuple(out)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _numeric_segments(key):")
        add(lines, 8, "out = []")
        add(lines, 8, "for part in str(key).split('.'):")
        add(lines, 12, "if part.isdigit():")
        add(lines, 16, "out.append(int(part))")
        add(lines, 8, "return out")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _param_stage_index(cls, key, stage_count, layer_span):")
        add(lines, 8, "if stage_count <= 1 or layer_span is None:")
        add(lines, 12, "return 0")
        add(lines, 8, "segments = cls._numeric_segments(key)")
        add(lines, 8, "if not segments:")
        add(lines, 12, "return 0")
        add(lines, 8, "layer_min, layer_max = layer_span")
        add(lines, 8, "layer_count = int(layer_max) - int(layer_min) + 1")
        add(lines, 8, "if layer_count <= 0:")
        add(lines, 12, "return 0")
        add(lines, 8, "for value in segments:")
        add(lines, 12, "if int(layer_min) <= int(value) <= int(layer_max):")
        add(lines, 16, "relative = int(value) - int(layer_min)")
        add(lines, 16, "return min(stage_count - 1, max(0, (relative * stage_count) // layer_count))")
        add(lines, 8, "return 0")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _infer_layer_span(cls, keys):")
        add(lines, 8, "values = []")
        add(lines, 8, "for key in keys:")
        add(lines, 12, "segments = cls._numeric_segments(key)")
        add(lines, 12, "if segments:")
        add(lines, 16, "values.append(segments[0])")
        add(lines, 8, "if not values:")
        add(lines, 12, "return None")
        add(lines, 8, "return (min(values), max(values))")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _place_param_array(cls, key, value, devices, layer_span):")
        add(lines, 8, "if not devices:")
        add(lines, 12, "return value")
        add(lines, 8, "stage = cls._param_stage_index(key, len(devices), layer_span)")
        add(lines, 8, "return jax.device_put(value, devices[stage])")
        add(lines, 4, "")
        add(lines, 4, "def _to_jax(self, value, dtype=None):")
        add(lines, 8, "if value is None:")
        add(lines, 12, "return None")
        add(lines, 8, "if isinstance(value, tuple):")
        add(lines, 12, "return tuple(self._to_jax(item, dtype=dtype) for item in value)")
        add(lines, 8, "if isinstance(value, list):")
        add(lines, 12, "return [self._to_jax(item, dtype=dtype) for item in value]")
        add(lines, 8, "if isinstance(value, jax.Array):")
        add(lines, 12, "if dtype is not None and value.dtype != dtype:")
        add(lines, 16, "return value.astype(dtype)")
        add(lines, 12, "return value")
        add(lines, 8, "import numpy as np")
        add(lines, 8, "if hasattr(value, 'cpu'):")
        add(lines, 12, "if hasattr(value, 'detach'):")
        add(lines, 16, "value = value.detach()")
        add(lines, 12, "value = value.cpu()")
        add(lines, 12, "if hasattr(value, 'numpy'):")
        add(lines, 16, "value = value.numpy()")
        add(lines, 8, "return jnp.asarray(np.asarray(value), dtype=dtype)")
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
        add(lines, 8, "if not isinstance(cond, jax.Array):")
        add(lines, 12, "return yes if cond else no")
        add(lines, 8, "return jnp.where(cond, yes, no)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _matmul(left, right):")
        add(lines, 8, "if isinstance(left, jax.Array) and isinstance(right, jax.Array) and len(left.shape) == 4 and len(right.shape) == 4:")
        add(lines, 12, "lh = int(left.shape[1])")
        add(lines, 12, "rh = int(right.shape[1])")
        add(lines, 12, "if lh != rh:")
        add(lines, 16, "if lh > rh and rh > 0 and lh % rh == 0:")
        add(lines, 20, "right = jnp.repeat(right, lh // rh, axis=1)")
        add(lines, 16, "elif rh > lh and lh > 0 and rh % lh == 0:")
        add(lines, 20, "left = jnp.repeat(left, rh // lh, axis=1)")
        add(lines, 8, "return left @ right")
        add(lines, 4, "")
        add(lines, 4, "def _where_indices(self, x):")
        add(lines, 8, "if isinstance(x, jax.Array):")
        add(lines, 12, "return tuple(index.astype(jnp.int32) for index in jnp.nonzero(x))")
        add(lines, 8, "import numpy as np")
        add(lines, 8, "indices = np.asarray(x).nonzero()")
        add(lines, 8, "return tuple(jnp.asarray(index.astype('int32')) for index in indices)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _gather(x, index, dim=-1):")
        add(lines, 8, "axis = int(dim)")
        add(lines, 8, "if axis < 0:")
        add(lines, 12, "axis += len(x.shape)")
        add(lines, 8, "idx = index.astype(jnp.int32)")
        add(lines, 8, "while len(idx.shape) < len(x.shape):")
        add(lines, 12, "idx = jnp.expand_dims(idx, axis=axis)")
        add(lines, 8, "if len(idx.shape) != len(x.shape):")
        add(lines, 12, "raise ValueError(f'gather index rank {len(idx.shape)} does not match input rank {len(x.shape)}')")
        add(lines, 8, "return jnp.take_along_axis(x, idx, axis=axis)")
        add(lines, 4, "")
        add(lines, 4, "def _embedding(self, base, ids):")
        add(lines, 8, "weight = self._param(self._compose_path(base, 'weight'))")
        add(lines, 8, "if not isinstance(ids, jax.Array):")
        add(lines, 12, "ids = self._to_jax(ids, jnp.int32)")
        add(lines, 8, "elif ids.dtype != jnp.int32:")
        add(lines, 12, "ids = ids.astype(jnp.int32)")
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
        add(lines, 8, "return jnp.concatenate(items, axis=int(dim))")
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
        add(lines, 4, "def _expand(x, shape):")
        add(lines, 8, "target = [int(dim) for dim in shape]")
        add(lines, 8, "offset = len(target) - len(x.shape)")
        add(lines, 8, "if offset < 0:")
        add(lines, 12, "raise ValueError(f'cannot expand rank {len(x.shape)} tensor to lower-rank shape {tuple(target)}')")
        add(lines, 8, "for index, dim in enumerate(target):")
        add(lines, 12, "if dim != -1:")
        add(lines, 16, "continue")
        add(lines, 12, "source_index = index - offset")
        add(lines, 12, "if source_index < 0:")
        add(lines, 16, "raise ValueError(f'cannot infer new leading expand dimension in shape {tuple(target)}')")
        add(lines, 12, "target[index] = int(x.shape[source_index])")
        add(lines, 8, "return jnp.broadcast_to(x, tuple(target))")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _arange(ref, start=0, end=None):")
        add(lines, 8, "if end is None:")
        add(lines, 12, "end = ref.shape[-2] if len(ref.shape) >= 3 else ref.shape[-1]")
        add(lines, 8, "if isinstance(start, jax.Array) or isinstance(end, jax.Array):")
        add(lines, 12, "length = int(ref.shape[-2] if len(ref.shape) >= 3 else ref.shape[-1])")
        add(lines, 12, "return jnp.arange(length, dtype=jnp.int32) + start")
        add(lines, 8, "return jnp.arange(int(start), int(end), dtype=jnp.int32)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _slice(x, dim, start, end, size=None):")
        add(lines, 8, "rank = len(x.shape)")
        add(lines, 8, "dim = int(dim)")
        add(lines, 8, "if dim < 0: dim += rank")
        add(lines, 8, "if not isinstance(start, jax.Array): start = int(start)")
        add(lines, 8, "if not isinstance(end, jax.Array): end = int(end)")
        add(lines, 8, "if size is None:")
        add(lines, 12, "return x[(slice(None),) * dim + (slice(start, end),) + (slice(None),) * (rank - dim - 1)]")
        add(lines, 8, "starts = [0] * rank")
        add(lines, 8, "starts[dim] = start")
        add(lines, 8, "sizes = list(x.shape)")
        add(lines, 8, "sizes[dim] = int(size)")
        add(lines, 8, "return jax.lax.dynamic_slice(x, tuple(starts), tuple(sizes))")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _assign_slice(x, src, dim, start, end):")
        add(lines, 8, "rank = len(x.shape)")
        add(lines, 8, "dim = int(dim)")
        add(lines, 8, "if dim < 0:")
        add(lines, 12, "dim += rank")
        add(lines, 8, "starts = [0] * rank")
        add(lines, 8, "starts[dim] = start")
        add(lines, 8, "return jax.lax.dynamic_update_slice(x, src.astype(x.dtype) if hasattr(src, 'dtype') and src.dtype != x.dtype else src, tuple(starts))")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _topk(x, k, dim=-1, largest=True, sorted_=True):")
        add(lines, 8, "del sorted_")
        add(lines, 8, "dim = int(dim)")
        add(lines, 8, "if dim < 0: dim += len(x.shape)")
        add(lines, 8, "moved = jnp.moveaxis(x, dim, -1)")
        add(lines, 8, "values, indices = jax.lax.top_k(moved if largest else -moved, int(k))")
        add(lines, 8, "if not largest: values = -values")
        add(lines, 8, "return jnp.moveaxis(values, -1, dim), jnp.moveaxis(indices.astype(jnp.int32), -1, dim)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _index_add(x, index, src, dim=0):")
        add(lines, 8, "dim = int(dim)")
        add(lines, 8, "if dim < 0:")
        add(lines, 12, "dim += len(x.shape)")
        add(lines, 8, "if dim == 0:")
        add(lines, 12, "return x.at[index].add(src)")
        add(lines, 8, "return jnp.swapaxes(jnp.swapaxes(x, 0, dim).at[index].add(jnp.swapaxes(src, 0, dim)), 0, dim)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _scatter(x, index, src, dim=-1):")
        add(lines, 8, "dim = int(dim)")
        add(lines, 8, "if dim < 0:")
        add(lines, 12, "dim += len(x.shape)")
        add(lines, 8, "index = index.astype(jnp.int32)")
        add(lines, 8, "if isinstance(src, jax.Array):")
        add(lines, 12, "values = src.astype(x.dtype) if jnp.issubdtype(x.dtype, jnp.floating) and jnp.issubdtype(src.dtype, jnp.floating) and src.dtype != x.dtype else src")
        add(lines, 12, "if values.shape != index.shape:")
        add(lines, 16, "values = jnp.broadcast_to(values, index.shape)")
        add(lines, 8, "else:")
        add(lines, 12, "values = jnp.full(index.shape, src, dtype=x.dtype)")
        add(lines, 8, "return jnp.put_along_axis(x, index, values, axis=dim, inplace=False)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _linear(base, x, bias=False, transpose=False, expert=None, weight_leaf='weight', bias_leaf='bias'):")
        add(lines, 8, "raise RuntimeError('internal JAX _linear helper should not be called directly')")
        add(lines, 4, "")
        add(lines, 4, "def _expert_linear(self, base, x, expert_idx, bias=False, transpose=False, weight_leaf='weight', bias_leaf='bias'):")
        add(lines, 8, "weight = self._param(self._compose_path(base, weight_leaf))")
        add(lines, 8, "idx = expert_idx.astype(jnp.int32) if isinstance(expert_idx, jax.Array) else expert_idx")
        add(lines, 8, "selected_weight = weight[idx]")
        add(lines, 8, "bias_value = self._optional_param(self._compose_path(base, bias_leaf)) if bias else None")
        add(lines, 8, "selected_bias = bias_value[idx] if bias_value is not None else None")
        add(lines, 8, "weight_run = selected_weight.astype(x.dtype) if x.dtype != selected_weight.dtype and x.dtype in (jnp.float32, jnp.float16, jnp.bfloat16) else selected_weight")
        add(lines, 8, "bias_run = selected_bias.astype(x.dtype) if selected_bias is not None and x.dtype != selected_bias.dtype else selected_bias")
        add(lines, 8, "if transpose:")
        add(lines, 12, "y = (x[..., None, :] @ weight_run).squeeze(-2)")
        add(lines, 8, "else:")
        add(lines, 12, "y = (x[..., None, :] @ weight_run.swapaxes(-1, -2)).squeeze(-2)")
        add(lines, 8, "return y + bias_run if bias_run is not None else y")
        add(lines, 4, "")
        add(lines, 4, "def _expert_linear_weight(self, x, expert_idx, weight_path, bias_value=None, transpose=False):")
        add(lines, 8, "weight = self._param(weight_path)")
        add(lines, 8, "idx = expert_idx.astype(jnp.int32) if isinstance(expert_idx, jax.Array) else expert_idx")
        add(lines, 8, "selected_weight = weight[idx]")
        add(lines, 8, "selected_bias = bias_value[idx] if bias_value is not None else None")
        add(lines, 8, "weight_run = selected_weight.astype(x.dtype) if x.dtype != selected_weight.dtype and x.dtype in (jnp.float32, jnp.float16, jnp.bfloat16) else selected_weight")
        add(lines, 8, "bias_run = selected_bias.astype(x.dtype) if selected_bias is not None and x.dtype != selected_bias.dtype else selected_bias")
        add(lines, 8, "if transpose:")
        add(lines, 12, "y = (x[..., None, :] @ weight_run).squeeze(-2)")
        add(lines, 8, "else:")
        add(lines, 12, "y = (x[..., None, :] @ weight_run.swapaxes(-1, -2)).squeeze(-2)")
        add(lines, 8, "return y + bias_run if bias_run is not None else y")
        add(lines, 4, "")
        add(lines, 4, "def _swiglu_ffn(self, x, gate_weight_path, up_weight_path, down_weight_path, gate_bias_path='bias', up_bias_path='bias', down_bias_path='bias'):")
        add(lines, 8, "del gate_bias_path, up_bias_path, down_bias_path")
        add(lines, 8, "gate_weight = self._param(gate_weight_path)")
        add(lines, 8, "up_weight = self._param(up_weight_path)")
        add(lines, 8, "down_weight = self._param(down_weight_path)")
        add(lines, 8, "gate = x @ gate_weight.swapaxes(-1, -2)")
        add(lines, 8, "up = x @ up_weight.swapaxes(-1, -2)")
        add(lines, 8, "hidden = (nn.sigmoid(gate) * gate) * up")
        add(lines, 8, "return hidden @ down_weight.swapaxes(-1, -2)")
        add(lines, 4, "")
        add(lines, 4, "def _expert_swiglu_ffn(self, x, expert_idx, gate_weight_path, up_weight_path, down_weight_path):")
        add(lines, 8, "gate = self._expert_linear_weight(x, expert_idx, gate_weight_path)")
        add(lines, 8, "up = self._expert_linear_weight(x, expert_idx, up_weight_path)")
        add(lines, 8, "hidden = (nn.sigmoid(gate) * gate) * up")
        add(lines, 8, "return self._expert_linear_weight(hidden, expert_idx, down_weight_path)")
        add(lines, 4, "")
        add(lines, 4, "def _expert_packed_swiglu_ffn(self, x, expert_idx, gate_up_weight_path, down_weight_path, transpose=False):")
        add(lines, 8, "gate_up = self._expert_linear_weight(x, expert_idx, gate_up_weight_path, transpose=transpose)")
        add(lines, 8, "gate, up = jnp.split(gate_up, 2, axis=-1)")
        add(lines, 8, "hidden = (nn.sigmoid(gate) * gate) * up")
        add(lines, 8, "return self._expert_linear_weight(hidden, expert_idx, down_weight_path, transpose=transpose)")
        add(lines, 4, "")
        add(lines, 4, "def _selected_expert_packed_swiglu_ffn(self, x, topk_scores, topk_indices, gate_up_weight_path, down_weight_path, transpose=False):")
        add(lines, 8, "topk_indices = topk_indices.astype(jnp.int32)")
        add(lines, 8, "expanded = jnp.broadcast_to(jnp.expand_dims(x, 2), tuple(topk_indices.shape) + (x.shape[-1],))")
        add(lines, 8, "gate_up = self._expert_linear_weight(expanded, topk_indices, gate_up_weight_path, transpose=transpose)")
        add(lines, 8, "gate, up = jnp.split(gate_up, 2, axis=-1)")
        add(lines, 8, "hidden = (nn.sigmoid(gate) * gate) * up")
        add(lines, 8, "values = self._expert_linear_weight(hidden, topk_indices, down_weight_path, transpose=transpose)")
        add(lines, 8, "weights = jnp.expand_dims(topk_scores.astype(values.dtype), -1)")
        add(lines, 8, "return jnp.sum(values * weights, axis=2)")
        add(lines, 4, "")
        add(lines, 4, "def _selected_expert_swiglu_ffn(self, x, topk_scores, topk_indices, gate_weight_path, up_weight_path, down_weight_path, transpose=False):")
        add(lines, 8, "topk_indices = topk_indices.astype(jnp.int32)")
        add(lines, 8, "expanded = jnp.broadcast_to(jnp.expand_dims(x, 2), tuple(topk_indices.shape) + (x.shape[-1],))")
        add(lines, 8, "gate = self._expert_linear_weight(expanded, topk_indices, gate_weight_path, transpose=transpose)")
        add(lines, 8, "up = self._expert_linear_weight(expanded, topk_indices, up_weight_path, transpose=transpose)")
        add(lines, 8, "hidden = (nn.sigmoid(gate) * gate) * up")
        add(lines, 8, "values = self._expert_linear_weight(hidden, topk_indices, down_weight_path, transpose=transpose)")
        add(lines, 8, "weights = jnp.expand_dims(topk_scores.astype(values.dtype), -1)")
        add(lines, 8, "return jnp.sum(values * weights, axis=2)")
        add(lines, 4, "")
        add(lines, 4, "def _selected_expert_packed_gegelu_ffn(self, x, topk_scores, topk_indices, gate_up_weight_path, gate_up_bias_path, down_weight_path, down_bias_path, limit, alpha=1.702, bias=False, transpose=False):")
        add(lines, 8, "topk_indices = topk_indices.astype(jnp.int32)")
        add(lines, 8, "expanded = jnp.broadcast_to(jnp.expand_dims(x, 2), tuple(topk_indices.shape) + (x.shape[-1],))")
        add(lines, 8, "gate_up_bias = self._param(gate_up_bias_path) if bias else None")
        add(lines, 8, "gate_up = self._expert_linear_weight(expanded, topk_indices, gate_up_weight_path, bias_value=gate_up_bias, transpose=transpose)")
        add(lines, 8, "hidden = self._gegelu(gate_up, limit, alpha)")
        add(lines, 8, "down_bias = self._param(down_bias_path) if bias else None")
        add(lines, 8, "values = self._expert_linear_weight(hidden, topk_indices, down_weight_path, bias_value=down_bias, transpose=transpose)")
        add(lines, 8, "weights = jnp.expand_dims(topk_scores.astype(values.dtype), -1)")
        add(lines, 8, "return jnp.sum(values * weights, axis=2)")
        add(lines, 4, "")
        add(lines, 4, "def _selected_expert_clamped_packed_swiglu_ffn(self, x, topk_scores, topk_indices, gate_up_weight_path, down_weight_path, limit, transpose=False):")
        add(lines, 8, "topk_indices = topk_indices.astype(jnp.int32)")
        add(lines, 8, "expanded = jnp.broadcast_to(jnp.expand_dims(x, 2), tuple(topk_indices.shape) + (x.shape[-1],))")
        add(lines, 8, "gate_up = self._expert_linear_weight(expanded, topk_indices, gate_up_weight_path, transpose=transpose)")
        add(lines, 8, "gate, up = jnp.split(gate_up, 2, axis=-1)")
        add(lines, 8, "limit = float(limit)")
        add(lines, 8, "gate = jnp.where(jnp.isinf(gate), gate, jnp.minimum(gate, limit))")
        add(lines, 8, "up = jnp.where(jnp.isinf(up), up, jnp.clip(up, -limit, limit))")
        add(lines, 8, "hidden = (nn.sigmoid(gate) * gate) * up")
        add(lines, 8, "values = self._expert_linear_weight(hidden, topk_indices, down_weight_path, transpose=transpose)")
        add(lines, 8, "weights = jnp.expand_dims(topk_scores.astype(values.dtype), -1)")
        add(lines, 8, "return jnp.sum(values * weights, axis=2)")
        add(lines, 4, "")
        add(lines, 4, "def _selected_expert_relu2_ffn(self, x, topk_scores, topk_indices, up_weight_path, down_weight_path, transpose=False):")
        add(lines, 8, "topk_indices = topk_indices.astype(jnp.int32)")
        add(lines, 8, "expanded = jnp.broadcast_to(jnp.expand_dims(x, 2), tuple(topk_indices.shape) + (x.shape[-1],))")
        add(lines, 8, "up = self._expert_linear_weight(expanded, topk_indices, up_weight_path, transpose=transpose)")
        add(lines, 8, "relu = jnp.maximum(up, 0)")
        add(lines, 8, "hidden = relu * relu")
        add(lines, 8, "values = self._expert_linear_weight(hidden, topk_indices, down_weight_path, transpose=transpose)")
        add(lines, 8, "weights = jnp.expand_dims(topk_scores.astype(values.dtype), -1)")
        add(lines, 8, "return jnp.sum(values * weights, axis=2)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _weighted_topk_sum(values, topk_scores):")
        add(lines, 8, "weights = jnp.expand_dims(topk_scores.astype(values.dtype), -1)")
        add(lines, 8, "return jnp.sum(values * weights, axis=2)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _gegelu(x, limit=None, alpha=1.702):")
        add(lines, 8, "if x.shape[-1] % 2 != 0: raise ValueError('gegelu requires even last dimension')")
        add(lines, 8, "x_gelu = x[..., ::2]")
        add(lines, 8, "x_linear = x[..., 1::2]")
        add(lines, 8, "if limit is not None:")
        add(lines, 12, "limit_val = float(limit)")
        add(lines, 12, "x_gelu = jnp.clip(x_gelu, -limit_val, limit_val)")
        add(lines, 12, "x_linear = jnp.clip(x_linear, -limit_val, limit_val)")
        add(lines, 8, "return x_gelu * nn.sigmoid(float(alpha) * x_gelu) * (x_linear + 1.0)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _xielu(x, alpha_p_raw, alpha_n_raw, beta_raw, eps_raw):")
        add(lines, 8, "beta = jnp.full(x.shape, float(beta_raw)) if not isinstance(beta_raw, jax.Array) else beta_raw")
        add(lines, 8, "alpha_p = nn.softplus(alpha_p_raw) if isinstance(alpha_p_raw, jax.Array) else nn.softplus(jnp.full(x.shape, float(alpha_p_raw)))")
        add(lines, 8, "alpha_n_base = nn.softplus(alpha_n_raw) if isinstance(alpha_n_raw, jax.Array) else nn.softplus(jnp.full(x.shape, float(alpha_n_raw)))")
        add(lines, 8, "alpha_n = beta + alpha_n_base")
        add(lines, 8, "eps = eps_raw if isinstance(eps_raw, jax.Array) else float(eps_raw)")
        add(lines, 8, "return jnp.where(x > 0, alpha_p * x * x + beta * x, (jnp.exp(jnp.minimum(x, eps)) - 1.0 - x) * alpha_n + beta * x)")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _layer_norm(x, weight=None, bias=None, eps=1e-5):")
        add(lines, 8, "y = nn.standardize(x, axis=-1, epsilon=float(eps))")
        add(lines, 8, "if weight is not None: y = y * weight")
        add(lines, 8, "if bias is not None: y = y + bias")
        add(lines, 8, "return y")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _rms_norm(x, weight=None, eps=1e-6):")
        add(lines, 8, "y = x * jax.lax.rsqrt(jnp.mean(x.astype(jnp.float32) * x.astype(jnp.float32), axis=-1, keepdims=True) + float(eps))")
        add(lines, 8, "y = y.astype(x.dtype)")
        add(lines, 8, "return y if weight is None else y * weight")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _sdpa(q, k, v, mask=None, scale=None):")
        add(lines, 8, "if q.shape[1] != k.shape[1]:")
        add(lines, 12, "if q.shape[1] > k.shape[1] and k.shape[1] > 0 and q.shape[1] % k.shape[1] == 0:")
        add(lines, 16, "repeat = q.shape[1] // k.shape[1]")
        add(lines, 16, "k = jnp.repeat(k, repeat, axis=1)")
        add(lines, 16, "v = jnp.repeat(v, repeat, axis=1)")
        add(lines, 12, "elif k.shape[1] > q.shape[1] and q.shape[1] > 0 and k.shape[1] % q.shape[1] == 0:")
        add(lines, 16, "q = jnp.repeat(q, k.shape[1] // q.shape[1], axis=1)")
        add(lines, 8, "scale_value = (1.0 / (q.shape[-1] ** 0.5)) if scale is None else float(scale)")
        add(lines, 8, "scores = (q @ jnp.swapaxes(k, -1, -2)) * scale_value")
        add(lines, 8, "if mask is not None:")
        add(lines, 12, "if getattr(mask, 'dtype', None) == jnp.bool_:")
        add(lines, 16, "scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)")
        add(lines, 12, "else:")
        add(lines, 16, "scores = scores + mask")
        add(lines, 8, "return nn.softmax(scores, axis=-1) @ v")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _rope(x, traditional=False):")
        add(lines, 8, "del traditional")
        add(lines, 8, "raise NotImplementedError('__jax_rope lowering is not implemented yet')")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _split_sizes(x, sizes, axis):")
        add(lines, 8, "sizes = [int(s) for s in sizes]")
        add(lines, 8, "cuts = []")
        add(lines, 8, "total = 0")
        add(lines, 8, "for size in sizes[:-1]:")
        add(lines, 12, "total += int(size)")
        add(lines, 12, "cuts.append(total)")
        add(lines, 8, "return jnp.split(x, indices_or_sections=cuts, axis=int(axis))")
        add(lines, 4, "")
        add(lines, 4, "@staticmethod")
        add(lines, 4, "def _conv1d(x, weight, bias, stride, padding_left, padding_right, dilation, groups):")
        add(lines, 8, "if x.dtype != weight.dtype and jnp.issubdtype(x.dtype, jnp.floating) and jnp.issubdtype(weight.dtype, jnp.floating):")
        add(lines, 12, "weight = weight.astype(x.dtype)")
        add(lines, 8, "if bias is not None and bias.dtype != x.dtype and jnp.issubdtype(x.dtype, jnp.floating) and jnp.issubdtype(bias.dtype, jnp.floating):")
        add(lines, 12, "bias = bias.astype(x.dtype)")
        add(lines, 8, "y = jax.lax.conv_general_dilated(")
        add(lines, 12, "x,")
        add(lines, 12, "weight,")
        add(lines, 12, "window_strides=(int(stride),),")
        add(lines, 12, "padding=((int(padding_left), int(padding_right)),),")
        add(lines, 12, "rhs_dilation=(int(dilation),),")
        add(lines, 12, "dimension_numbers=('NCH', 'OIH', 'NCH'),")
        add(lines, 12, "feature_group_count=int(groups),")
        add(lines, 8, ")")
        add(lines, 8, "if bias is not None:")
        add(lines, 12, "y = y + bias.reshape((1, -1, 1))")
        add(lines, 8, "return y")
        add(lines, 4, "")
    def _emit_load_state_dict(self, lines: list[str]) -> None:
        add = self._add
        add(lines, 4, "def load_state_dict(self, state_dict, *, quantize=False, param_devices=None):")
        add(lines, 8, "del quantize")
        add(lines, 8, "state_dict = self._filter_state_dict(state_dict)")
        add(lines, 8, "state_dict = self._materialize_state_aliases(state_dict)")
        add(lines, 8, "state_dict = self._drop_collapsed_numeric_state_aliases(state_dict)")
        add(lines, 8, "devices = self._resolve_param_devices(param_devices)")
        add(lines, 8, "tensors = {}")
        add(lines, 8, "packed_source_keys = self._materialize_packed_parameters_from_state_dict(state_dict, tensors)")
        add(lines, 8, "layer_span = self._infer_layer_span(set(state_dict) | set(tensors))")
        add(lines, 8, "if devices:")
        add(lines, 12, "for key, value in list(tensors.items()):")
        add(lines, 16, "tensors[key] = self._place_param_array(key, value, devices, layer_span)")
        add(lines, 8, "for k, v in state_dict.items():")
        add(lines, 12, "if str(k) in packed_source_keys:")
        add(lines, 16, "continue")
        add(lines, 12, "if isinstance(v, jax.Array):")
        add(lines, 16, "array = v")
        add(lines, 12, "else:")
        add(lines, 16, "array = self._state_array_from_numpy(v)")
        add(lines, 12, "tensors[str(k)] = self._place_param_array(str(k), array, devices, layer_span)")
        add(lines, 8, "_jax_materialize_packed_parameters(tensors, self._PACKED_PARAMETER_SPECS)")
        add(lines, 8, "if devices:")
        add(lines, 12, "for key, value in list(tensors.items()):")
        add(lines, 16, "tensors[key] = self._place_param_array(key, value, devices, layer_span)")
        add(lines, 8, "self._flat_tensors = tensors")
        add(lines, 8, "self._materialize_expert_banks()")
        add(lines, 8, "")
        add(lines, 8, "object.__setattr__(self, '_symbols', self._eval_symbols())")

    def _param_expr_for_jax_attr(
        self,
        path_operand: Any,
        *,
        local: set[str],
        symbols_dict: str,
    ) -> str | None:
        del path_operand, local, symbols_dict
        return None

    def _param_expr_for_path(
        self,
        base: Any,
        leaf: Any,
        *,
        optional: bool = False,
        local: set[str],
        symbols_dict: str,
    ) -> str:
        key = self._static_param_key(base, leaf)
        if key is not None:
            return f"self._flat_tensors.get({key!r})" if optional else f"self._flat_tensors[{key!r}]"
        key_expr = self._param_key_expr(base, leaf, local=local, symbols_dict=symbols_dict)
        if key_expr is not None:
            getter = "_optional_param" if optional else "_param"
            return f"self.{getter}({key_expr})"
        base_expr = self._operand_expr(base, local=local, symbols_dict=symbols_dict)
        leaf_expr = repr(leaf) if isinstance(leaf, str) else self._operand_expr(leaf, local=local, symbols_dict=symbols_dict)
        getter = "_optional_param" if optional else "_param"
        return f"self.{getter}(self._compose_path({base_expr}, {leaf_expr}))"

    def _emit_forward(self, lines: list[str]) -> None:
        main = self.modules_by_name[self.program.main_module]
        add = self._add
        add(lines, 4, "def _forward(self, input_ids=None, **inputs):")
        args: list[str] = []
        first_input = main.inputs[0].name if main.inputs else None
        input_names = {value.name for value in main.inputs}
        static_attention_inputs = {
            value.name
            for value in main.inputs
            if value.name in {"attn_mask", "attention_mask", "decoder_attention_mask"}
            and _is_static_mask_type(value.type_expr)
        }
        static_attention_capacity_symbols = {
            value.name: capacity_dim
            for value in main.inputs
            if value.name in static_attention_inputs
            for capacity_dim in (_static_mask_capacity_dim(value.type_expr),)
            if isinstance(capacity_dim, str) and capacity_dim in self.global_symbol_names
        }
        for value in main.inputs:
            if value.name == "input_ids":
                add(lines, 8, "if input_ids is None:")
                add(lines, 12, "input_ids = inputs.get('input_ids')")
                add(lines, 8, "if input_ids is None:")
                add(lines, 12, "raise ValueError('Missing required input: input_ids')")
                add(lines, 8, "input_ids = self._to_jax(input_ids, jnp.int32)")
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
                add(lines, 8, f"{value.name} = self._to_jax({value.name})")
                args.append(value.name)
            else:
                if value.optional or isinstance(value.type_expr, TypeOptional):
                    add(lines, 8, f"{value.name} = inputs.get({value.name!r}, None)")
                else:
                    add(lines, 8, f"if {value.name!r} not in inputs:")
                    add(lines, 12, f"raise ValueError('Missing required input: {value.name}')")
                    add(lines, 8, f"{value.name} = inputs[{value.name!r}]")
                if value.name in static_attention_inputs:
                    capacity_expr = _static_mask_capacity_expr(value.type_expr, global_names=self.global_symbol_names)
                    add(lines, 8, f"{value.name} = self._to_jax({value.name})")
                    add(lines, 8, f"if {value.name} is not None and not (isinstance({value.name}, tuple) and len({value.name}) == 2):")
                    add(lines, 12, f"__static_len_{value.name} = int({value.name}.shape[1])")
                    if capacity_expr is None:
                        add(lines, 12, f"__static_declared_capacity_{value.name} = int(input_ids.shape[1]) if input_ids is not None else __static_len_{value.name}")
                    else:
                        add(lines, 12, f"__static_declared_capacity_{value.name} = int({capacity_expr})")
                    add(lines, 12, f"if __static_len_{value.name} > __static_declared_capacity_{value.name}:")
                    add(lines, 16, f"raise ValueError(f'requested JAX static input capacity {{__static_len_{value.name}}} exceeds declared capacity {{__static_declared_capacity_{value.name}}}')")
                    add(lines, 12, f"__static_capacity_{value.name} = __static_len_{value.name}")
                    add(lines, 12, f"__static_store_{value.name} = jnp.zeros(({value.name}.shape[0], __static_capacity_{value.name}), dtype={value.name}.dtype)")
                    add(lines, 12, f"__static_store_{value.name} = jax.lax.dynamic_update_slice(__static_store_{value.name}, {value.name}, (0, 0))")
                    add(lines, 12, f"{value.name} = (__static_store_{value.name}, __static_len_{value.name})")
                    add(lines, 8, f"elif {value.name} is not None:")
                    add(lines, 12, f"__static_capacity_{value.name} = int({value.name}[0].shape[1])")
                    add(lines, 8, "else:")
                    add(lines, 12, f"__static_capacity_{value.name} = int(input_ids.shape[1]) if input_ids is not None else 0")
                    add(lines, 12, f"__static_store_{value.name} = jnp.ones((input_ids.shape[0], __static_capacity_{value.name}), dtype=jnp.bool_)")
                    add(lines, 12, f"{value.name} = (__static_store_{value.name}, __static_capacity_{value.name})")
                else:
                    add(lines, 8, f"{value.name} = self._to_jax({value.name})")
                args.append(value.name)
        if "past_kv" in input_names and static_attention_inputs:
            capacity_source = sorted(static_attention_inputs)[0]
            add(lines, 8, "if past_kv is None:")
            add(lines, 12, f"past_kv = ([], [], 0, __static_capacity_{capacity_source})")
        static_capacity_overrides = [
            (name, symbol)
            for name, symbol in sorted(static_attention_capacity_symbols.items())
        ]
        if static_capacity_overrides:
            add(lines, 8, "__static_capacity_old = {}")
            add(lines, 8, "try:")
            for name, symbol in static_capacity_overrides:
                add(lines, 12, f"__static_capacity_old[{symbol!r}] = self._symbols.get({symbol!r})")
                add(lines, 12, f"self._symbols[{symbol!r}] = __static_capacity_{name}")
            call_indent = 12
        else:
            call_indent = 8
        add(lines, call_indent, f"result = self.{self.method_names[main.name]}({', '.join(args)})")
        if static_capacity_overrides:
            add(lines, 8, "finally:")
            for _, symbol in static_capacity_overrides:
                add(lines, 12, f"if __static_capacity_old[{symbol!r}] is None:")
                add(lines, 16, f"self._symbols.pop({symbol!r}, None)")
                add(lines, 12, "else:")
                add(lines, 16, f"self._symbols[{symbol!r}] = __static_capacity_old[{symbol!r}]")
        names = graph_main_output_names(self.program, main)
        if len(names) == 1:
            add(lines, 8, "return result[0]")
        else:
            add(lines, 8, f"return {{{', '.join(f'{name!r}: result[{idx}]' for idx, name in enumerate(names))}}}")
        add(lines, 4, "")
        add(lines, 4, "def forward(self, input_ids=None, **inputs):")
        add(lines, 8, "input_ids = self._to_jax(input_ids, jnp.int32) if input_ids is not None else None")
        add(lines, 8, "for _k, _v in list(inputs.items()):")
        skip_names = tuple(sorted({"use_cache", "past_kv"}))
        add(lines, 12, f"if _k in {skip_names!r}:")
        add(lines, 16, "continue")
        add(lines, 12, "inputs[_k] = self._to_jax(_v)")
        add(lines, 8, "__has_past_kv = inputs.get('past_kv', None) is not None")
        add(lines, 8, "if self._jit_enabled and not __has_past_kv and self._compiled_fn is None:")
        add(lines, 12, "self._compiled_fn = jax.jit(self._forward_with_state, static_argnames=('use_cache',))")
        add(lines, 8, "if self._jit_enabled and __has_past_kv and self._compiled_cached_fn is None:")
        add(lines, 12, "self._compiled_cached_fn = jax.jit(self._forward_with_state, static_argnames=('use_cache',))")
        add(lines, 8, "if not __has_past_kv and self._compiled_fn is not None:")
        add(lines, 12, "result = self._compiled_fn(self._flat_tensors, input_ids, **inputs)")
        add(lines, 12, "if isinstance(result, (list, tuple)):")
        add(lines, 16, f"return {{{', '.join(f'{name!r}: result[{idx}]' for idx, name in enumerate(names))}}}")
        add(lines, 12, "return result")
        add(lines, 8, "if __has_past_kv and self._compiled_cached_fn is not None:")
        add(lines, 12, "result = self._compiled_cached_fn(self._flat_tensors, input_ids, **inputs)")
        add(lines, 12, "if isinstance(result, (list, tuple)):")
        add(lines, 16, f"return {{{', '.join(f'{name!r}: result[{idx}]' for idx, name in enumerate(names))}}}")
        add(lines, 12, "return result")
        add(lines, 8, "return self._forward(input_ids, **inputs)")
        add(lines, 4, "")
        add(lines, 4, "def _forward_seq2seq_generate_step(self, input_ids, **inputs):")
        add(lines, 8, "decoder_ids = inputs.get('decoder_input_ids', None)")
        add(lines, 8, "decoder_len = None if decoder_ids is None else int(decoder_ids.shape[1])")
        add(lines, 8, "compiled = None if decoder_len is None else self._compiled_seq2seq_by_decoder_length.get(decoder_len)")
        add(lines, 8, "if self._jit_enabled and decoder_len is not None and compiled is None:")
        add(lines, 12, "compiled = jax.jit(self._forward_with_state)")
        add(lines, 12, "self._compiled_seq2seq_by_decoder_length[decoder_len] = compiled")
        add(lines, 8, "if compiled is not None:")
        add(lines, 12, "return compiled(self._flat_tensors, input_ids, **inputs)")
        add(lines, 8, "return self._forward(input_ids, **inputs)")
        add(lines, 4, "")
        add(lines, 4, "def compile(self, max_kv_length=2048):")
        add(lines, 8, "\"\"\"Compile _forward with jax.jit and warmup KV shapes 0..max_kv_length.\"\"\"")
        add(lines, 8, "if self._compiled_fn is not None:")
        add(lines, 12, "return self._compiled_fn")
        add(lines, 8, "self._compiled_fn = jax.jit(self._forward_with_state, static_argnames=('use_cache',))")
        add(lines, 8, "prompt_ids = jnp.zeros((1, 1), dtype=jnp.int32)")
        add(lines, 8, "kv = None")
        add(lines, 8, "for length in range(1, max_kv_length + 1):")
        add(lines, 12, "inp = jnp.array([[0]], dtype=jnp.int32)")
        add(lines, 12, "result = self._compiled_fn(self._flat_tensors, inp, past_kv=kv, use_cache=True)")
        add(lines, 12, "if isinstance(result, (list, tuple)):")
        add(lines, 16, "kv = result[1] if len(result) > 1 else None")
        add(lines, 12, "else:")
        add(lines, 16, "kv = result.get('new_kv') if isinstance(result, dict) else None")
        add(lines, 12, "if kv is None:")
        add(lines, 16, "break")
        add(lines, 8, "jnp.array(0).block_until_ready()")
        add(lines, 8, "return self._compiled_fn")

    def _emit_generate(self, lines: list[str]) -> None:
        add = self._add
        main = self.modules_by_name[self.program.main_module]
        input_names = {value.name for value in main.inputs}
        output_names = set(graph_main_output_names(self.program, main))
        attention_name = "attn_mask" if "attn_mask" in input_names else (
            "attention_mask" if "attention_mask" in input_names else None
        )
        attention_value = next(
            (value for value in main.inputs if value.name == attention_name),
            None,
        )
        uses_static_attention_mask = (
            attention_value is not None and _is_static_mask_type(attention_value.type_expr)
        )
        static_attention_capacity_expr = (
            _static_mask_capacity_expr(attention_value.type_expr, global_names=self.global_symbol_names)
            if attention_value is not None
            else None
        )
        static_attention_capacity_dim = (
            _static_mask_capacity_dim(attention_value.type_expr)
            if attention_value is not None
            else None
        )
        static_attention_capacity_symbol = (
            static_attention_capacity_dim
            if isinstance(static_attention_capacity_dim, str)
            and static_attention_capacity_dim in self.global_symbol_names
            else None
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
        is_decoder_only = not has_decoder_inputs
        is_cached_decoder = (
            is_decoder_only
            and cache_name is not None
            and cache_output_name is not None
            and use_cache_name is not None
        )
        add(lines, 4, "def generate(self, input_ids, max_new_tokens=20, **kwargs):")
        add(lines, 8, "input_ids = self._to_jax(input_ids, jnp.int32)")
        add(lines, 8, "def _logits(result):")
        add(lines, 12, "return result.get('logits') if isinstance(result, dict) else result")
        add(lines, 8, "def _next_id(logits):")
        add(lines, 12, "return logits[:, -1:, :].argmax(axis=-1).astype(jnp.int32)")
        add(lines, 8, "def _ones_like_ids(ids):")
        add(lines, 12, "return jnp.ones(ids.shape, dtype=jnp.int32)")
        add(lines, 8, "def _static_attention_mask(mask, prompt_ids, capacity):")
        add(lines, 12, "valid = _ones_like_ids(prompt_ids) if mask is None else self._to_jax(mask)")
        add(lines, 12, "length = int(valid.shape[1])")
        add(lines, 12, "store = jnp.zeros((valid.shape[0], int(capacity)), dtype=valid.dtype)")
        add(lines, 12, "store = jax.lax.dynamic_update_slice(store, valid, (0, 0))")
        add(lines, 12, "return (store, length)")
        add(lines, 8, "def _append_static_attention_mask(mask, next_id):")
        add(lines, 12, "store, length = mask")
        add(lines, 12, "src = _ones_like_ids(next_id).astype(store.dtype)")
        add(lines, 12, "store = jax.lax.dynamic_update_slice(store, src, (0, length))")
        add(lines, 12, "return (store, length + 1)")
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
        add(lines, 12, "eos = jnp.array(eos_values, dtype=jnp.int32).reshape((-1,))")
        add(lines, 12, "pad = int(eos_values[0] if pad_token_id is None else pad_token_id)")
        add(lines, 12, "finished = jnp.zeros((batch_size, 1), dtype=jnp.bool_)")
        add(lines, 12, "return eos, pad, finished")
        add(lines, 8, "def _apply_eos(next_id, eos, pad, finished):")
        add(lines, 12, "if eos is None: return next_id, finished")
        add(lines, 12, "raw_next = next_id")
        add(lines, 12, "next_id = jnp.where(finished, jnp.full(next_id.shape, pad, dtype=jnp.int32), next_id)")
        add(lines, 12, "hit = (raw_next == eos.reshape((1, -1))).max(axis=1, keepdims=True)")
        add(lines, 12, "finished = finished | hit")
        add(lines, 12, "return next_id, finished")
        add(lines, 8, "def _all_done(finished):")
        add(lines, 12, "return finished is not None and bool(jax.device_get(jnp.all(finished)))")
        if is_cached_decoder:
            add(lines, 8, "out = input_ids")
            add(lines, 8, f"cache = kwargs.pop({cache_name!r}, None)")
            add(lines, 8, "limit = _generation_limit(out)")
            add(lines, 8, "eos, pad, finished = _eos_state(out.shape[0])")
            if attention_name is not None:
                other = "attention_mask" if attention_name == "attn_mask" else "attn_mask"
                add(lines, 8, f"attention_mask = kwargs.pop({attention_name!r}, kwargs.pop({other!r}, None))")
                add(lines, 8, "if attention_mask is None: attention_mask = _ones_like_ids(out)")
                add(lines, 8, "else: attention_mask = self._to_jax(attention_mask)")
                if uses_static_attention_mask:
                    add(lines, 8, "requested_static_capacity = int(out.shape[1]) + int(limit)")
                    if static_attention_capacity_expr is None:
                        add(lines, 8, "declared_static_capacity = int(self.config.get('n_positions', self.config.get('max_position_embeddings', requested_static_capacity)))")
                    else:
                        add(lines, 8, f"declared_static_capacity = int({static_attention_capacity_expr})")
                    add(lines, 8, "if requested_static_capacity > declared_static_capacity:")
                    add(lines, 12, "raise ValueError(f'requested JAX static generation capacity {requested_static_capacity} exceeds declared capacity {declared_static_capacity}')")
                    add(lines, 8, "static_capacity = requested_static_capacity")
                    add(lines, 8, "attention_mask = _static_attention_mask(attention_mask, out, static_capacity)")
            if use_cache_name is not None:
                add(lines, 8, f"kwargs.pop({use_cache_name!r}, None)")
            if static_attention_capacity_symbol is not None:
                add(lines, 8, f"__static_capacity_symbol = {static_attention_capacity_symbol!r}")
                add(lines, 8, "__static_capacity_old = self._symbols.get(__static_capacity_symbol)")
                add(lines, 8, "self._symbols[__static_capacity_symbol] = static_capacity")
                add(lines, 8, "try:")
                loop_indent = 12
                body_indent = 16
            else:
                loop_indent = 8
                body_indent = 12
            add(lines, loop_indent, "for _ in range(limit):")
            add(lines, body_indent, "step_input = out[:, -1:] if cache is not None else out")
            add(lines, body_indent, "forward_kwargs = dict(kwargs)")
            add(lines, body_indent, f"forward_kwargs[{cache_name!r}] = cache")
            if use_cache_name is not None:
                add(lines, body_indent, f"forward_kwargs[{use_cache_name!r}] = True")
            if attention_name is not None:
                add(lines, body_indent, f"forward_kwargs[{attention_name!r}] = attention_mask")
            add(lines, body_indent, "result = self.forward(step_input, **forward_kwargs)")
            add(lines, body_indent, "if isinstance(result, dict): cache = result.get(" + repr(cache_output_name) + ", cache)")
            add(lines, body_indent, "next_id = _next_id(_logits(result))")
            add(lines, body_indent, "next_id, finished = _apply_eos(next_id, eos, pad, finished)")
            add(lines, body_indent, "out = jnp.concatenate([out, next_id], axis=1)")
            if attention_name is not None:
                if uses_static_attention_mask:
                    add(lines, body_indent, "attention_mask = _append_static_attention_mask(attention_mask, next_id)")
                else:
                    add(lines, body_indent, "attention_mask = jnp.concatenate([attention_mask, _ones_like_ids(next_id)], axis=1)")
            add(lines, body_indent, "if _all_done(finished): break")
            add(lines, loop_indent, "return out")
            if static_attention_capacity_symbol is not None:
                add(lines, 8, "finally:")
                add(lines, 12, "if __static_capacity_old is None:")
                add(lines, 16, "self._symbols.pop(__static_capacity_symbol, None)")
                add(lines, 12, "else:")
                add(lines, 16, "self._symbols[__static_capacity_symbol] = __static_capacity_old")
            return
        if is_decoder_only:
            add(lines, 8, "out = input_ids")
            add(lines, 8, "limit = _generation_limit(out)")
            add(lines, 8, "eos, pad, finished = _eos_state(out.shape[0])")
            if attention_name is not None:
                other = "attention_mask" if attention_name == "attn_mask" else "attn_mask"
                add(lines, 8, f"attention_mask = kwargs.pop({attention_name!r}, kwargs.pop({other!r}, None))")
                add(lines, 8, "if attention_mask is None: attention_mask = _ones_like_ids(out)")
                add(lines, 8, "else: attention_mask = self._to_jax(attention_mask)")
            add(lines, 8, "for _ in range(limit):")
            add(lines, 12, "forward_kwargs = dict(kwargs)")
            if attention_name is not None:
                add(lines, 12, f"forward_kwargs[{attention_name!r}] = attention_mask")
            add(lines, 12, "result = self.forward(out, **forward_kwargs)")
            add(lines, 12, "next_id = _next_id(_logits(result))")
            add(lines, 12, "next_id, finished = _apply_eos(next_id, eos, pad, finished)")
            add(lines, 12, "out = jnp.concatenate([out, next_id], axis=1)")
            if attention_name is not None:
                add(lines, 12, "attention_mask = jnp.concatenate([attention_mask, _ones_like_ids(next_id)], axis=1)")
            add(lines, 12, "if _all_done(finished): break")
            add(lines, 8, "return out")
            return
        add(lines, 8, "decoder_input_ids = kwargs.pop('decoder_input_ids', None)")
        add(lines, 8, "if decoder_input_ids is None:")
        add(lines, 12, "start_id = kwargs.pop('decoder_start_token_id', self.config.get('decoder_start_token_id', self.config.get('pad_token_id', 0)))")
        add(lines, 12, "decoder_input_ids = jnp.full((input_ids.shape[0], 1), int(start_id), dtype=jnp.int32)")
        add(lines, 8, "limit = _generation_limit(input_ids)")
        add(lines, 8, "eos, pad, finished = _eos_state(decoder_input_ids.shape[0])")
        if attention_name is not None:
            other = "attention_mask" if attention_name == "attn_mask" else "attn_mask"
            add(lines, 8, f"attention_mask = kwargs.pop({attention_name!r}, kwargs.pop({other!r}, None))")
            add(lines, 8, "if attention_mask is None: attention_mask = _ones_like_ids(input_ids)")
            add(lines, 8, "else: attention_mask = self._to_jax(attention_mask)")
        if decoder_attention_name is not None:
            add(lines, 8, f"decoder_attention_mask = kwargs.pop({decoder_attention_name!r}, None)")
            add(lines, 8, "if decoder_attention_mask is None: decoder_attention_mask = _ones_like_ids(decoder_input_ids)")
            add(lines, 8, "else: decoder_attention_mask = self._to_jax(decoder_attention_mask)")
        add(lines, 8, "initial_decoder_len = int(decoder_input_ids.shape[1])")
        add(lines, 8, "target_decoder_len = initial_decoder_len + int(limit)")
        add(lines, 8, "fill_id = 0 if pad is None else int(pad)")
        add(lines, 8, "decoder_buffer = jnp.full((int(decoder_input_ids.shape[0]), target_decoder_len), fill_id, dtype=jnp.int32)")
        add(lines, 8, "decoder_buffer = jax.lax.dynamic_update_slice(decoder_buffer, decoder_input_ids, (0, 0))")
        if decoder_attention_name is not None:
            add(lines, 8, "decoder_mask_buffer = jnp.zeros((int(decoder_input_ids.shape[0]), target_decoder_len), dtype=jnp.int32)")
            add(lines, 8, "decoder_mask_buffer = jax.lax.dynamic_update_slice(decoder_mask_buffer, decoder_attention_mask.astype(jnp.int32), (0, 0))")
        add(lines, 8, "decoder_input_ids = decoder_buffer")
        if decoder_attention_name is not None:
            add(lines, 8, "decoder_attention_mask = decoder_mask_buffer")
        else:
            add(lines, 8, "decoder_attention_mask = jnp.zeros((1, target_decoder_len), dtype=jnp.int32)")
        add(lines, 8, "def _seq2seq_cond(state):")
        add(lines, 12, "active_decoder_len, _decoder_input_ids, _decoder_attention_mask, finished = state")
        add(lines, 12, "within_limit = active_decoder_len < target_decoder_len")
        add(lines, 12, "if eos is None:")
        add(lines, 16, "return within_limit")
        add(lines, 12, "return within_limit & (~jnp.all(finished))")
        add(lines, 8, "def _seq2seq_body(state):")
        add(lines, 12, "active_decoder_len, decoder_input_ids, decoder_attention_mask, finished = state")
        add(lines, 12, "forward_kwargs = dict(kwargs)")
        add(lines, 12, "forward_kwargs['decoder_input_ids'] = decoder_input_ids")
        if attention_name is not None:
            add(lines, 12, f"forward_kwargs[{attention_name!r}] = attention_mask")
        if decoder_attention_name is not None:
            add(lines, 12, f"forward_kwargs[{decoder_attention_name!r}] = decoder_attention_mask")
        add(lines, 12, "result = self._forward(input_ids, **forward_kwargs)")
        add(lines, 12, "logits = _logits(result)")
        add(lines, 12, "step_logits = jax.lax.dynamic_slice(logits, (0, active_decoder_len - 1, 0), (int(logits.shape[0]), 1, int(logits.shape[-1])))")
        add(lines, 12, "next_id = _next_id(step_logits)")
        add(lines, 12, "next_id, finished = _apply_eos(next_id, eos, pad, finished)")
        add(lines, 12, "decoder_input_ids = jax.lax.dynamic_update_slice(decoder_input_ids, next_id, (0, active_decoder_len))")
        add(lines, 12, "decoder_attention_mask = jax.lax.dynamic_update_slice(decoder_attention_mask, _ones_like_ids(next_id), (0, active_decoder_len))")
        add(lines, 12, "return active_decoder_len + 1, decoder_input_ids, decoder_attention_mask, finished")
        add(lines, 8, "active_decoder_len, decoder_input_ids, decoder_attention_mask, finished = jax.lax.while_loop(")
        add(lines, 12, "_seq2seq_cond,")
        add(lines, 12, "_seq2seq_body,")
        add(lines, 12, "(jnp.array(initial_decoder_len, dtype=jnp.int32), decoder_input_ids, decoder_attention_mask, finished),")
        add(lines, 8, ")")
        add(lines, 8, "return decoder_input_ids[:, : int(active_decoder_len)]")

    def _primitive_expr(self, primitive: str, node: Any, *, local: set[str], symbols_dict: str) -> str:
        args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
        attrs = {k: self._operand_expr(v, local=local, symbols_dict=symbols_dict) for k, v in node.attrs.items()}
        if primitive == "embedding":
            path_operand = node.inputs[0]
            attr_expr = self._param_expr_for_jax_attr(path_operand, local=local, symbols_dict=symbols_dict)
            if attr_expr is not None:
                return f"self.{attr_expr}({args[1]})"
            return f"self._embedding({args[0]}, {args[1]})"
        if primitive == "linear":
            bias = args[3] if len(args) > 3 else "False"
            transpose = args[4] if len(args) > 4 else "False"
            expert = args[5] if len(args) > 5 else "None"
            weight_expr = self._param_expr_for_path(
                node.inputs[0],
                node.inputs[6] if len(node.inputs) > 6 else "weight",
                local=local,
                symbols_dict=symbols_dict,
            )
            bias_expr = self._param_expr_for_path(
                node.inputs[0],
                node.inputs[7] if len(node.inputs) > 7 else "bias",
                optional=True,
                local=local,
                symbols_dict=symbols_dict,
            )
            return (
                f"(lambda _w, _b: "
                f"(({args[1]} @ _w.swapaxes(-1, -2)) + _b if _b is not None else ({args[1]} @ _w.swapaxes(-1, -2)))"
                f"if not bool({transpose}) else "
                f"(({args[1]} @ _w) + _b if _b is not None else ({args[1]} @ _w)))"
                f"((lambda _w: (_w[int({expert})] if {expert} is not None else _w))({weight_expr}), "
                f"((lambda _b: (_b[int({expert})] if (_b is not None and {expert} is not None and len(_b.shape) >= 2) else _b))({bias_expr}) if bool({bias}) else None))"
            )
        if primitive == "expert_linear":
            bias = args[4] if len(args) > 4 else "False"
            transpose = args[5] if len(args) > 5 else "False"
            weight_leaf = args[6] if len(args) > 6 else "'weight'"
            bias_leaf = args[7] if len(args) > 7 else "'bias'"
            return f"self._expert_linear({args[0]}, {args[1]}, {args[2]}, bias=bool({bias}), transpose=bool({transpose}), weight_leaf={weight_leaf}, bias_leaf={bias_leaf})"
        if primitive == "_jax_swiglu_ffn":
            if len(args) < 7:
                raise ValueError("__jax_swiglu_ffn expects input, gate/up/down weight paths, and gate/up/down bias paths")
            return (
                f"self._swiglu_ffn({args[0]}, {args[1]}, {args[2]}, {args[3]}, "
                f"gate_bias_path={args[4]}, up_bias_path={args[5]}, down_bias_path={args[6]})"
            )
        if primitive == "_jax_expert_swiglu_ffn":
            if len(args) < 5:
                raise ValueError("__jax_expert_swiglu_ffn expects input, expert indices, and gate/up/down weight paths")
            return f"self._expert_swiglu_ffn({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]})"
        if primitive == "_jax_expert_packed_swiglu_ffn":
            if len(args) < 5:
                raise ValueError("__jax_expert_packed_swiglu_ffn expects input, expert indices, gate-up/down weight paths, and transpose")
            return f"self._expert_packed_swiglu_ffn({args[0]}, {args[1]}, {args[2]}, {args[3]}, transpose=bool({args[4]}))"
        if primitive == "_jax_selected_expert_packed_swiglu_ffn":
            if len(args) < 6:
                raise ValueError("__jax_selected_expert_packed_swiglu_ffn expects input, top-k scores/indices, gate-up/down weight paths, and transpose")
            return (
                "self._selected_expert_packed_swiglu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, transpose=bool({args[5]}))"
            )
        if primitive == "_jax_selected_expert_swiglu_ffn":
            if len(args) < 7:
                raise ValueError("__jax_selected_expert_swiglu_ffn expects input, top-k scores/indices, gate/up/down weight paths, and transpose")
            return (
                "self._selected_expert_swiglu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, {args[5]}, transpose=bool({args[6]}))"
            )
        if primitive == "_jax_selected_expert_packed_gegelu_ffn":
            if len(args) < 10:
                raise ValueError("__jax_selected_expert_packed_gegelu_ffn expects input, top-k scores/indices, gate-up/down weight/bias paths, limit, optional alpha, bias, and transpose")
            alpha_arg = args[8] if len(args) >= 11 else "1.702"
            bias_idx = 9 if len(args) >= 11 else 8
            transpose_idx = 10 if len(args) >= 11 else 9
            return (
                "self._selected_expert_packed_gegelu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, {args[5]}, {args[6]}, {args[7]}, "
                f"alpha={alpha_arg}, bias=bool({args[bias_idx]}), transpose=bool({args[transpose_idx]}))"
            )
        if primitive == "_jax_selected_expert_clamped_packed_swiglu_ffn":
            if len(args) < 7:
                raise ValueError("__jax_selected_expert_clamped_packed_swiglu_ffn expects input, top-k scores/indices, gate-up/down weight paths, limit, and transpose")
            return (
                "self._selected_expert_clamped_packed_swiglu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, {args[5]}, transpose=bool({args[6]}))"
            )
        if primitive == "_jax_selected_expert_relu2_ffn":
            if len(args) < 6:
                raise ValueError("__jax_selected_expert_relu2_ffn expects input, top-k scores/indices, up/down weight paths, and transpose")
            return (
                "self._selected_expert_relu2_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, transpose=bool({args[5]}))"
            )
        if primitive == "_jax_weighted_topk_sum":
            if len(args) < 2:
                raise ValueError("__jax_weighted_topk_sum expects expert values and top-k scores")
            return f"self._weighted_topk_sum({args[0]}, {args[1]})"
        if primitive == "_jax_sdpa":
            if len(args) < 6:
                raise ValueError("__jax_sdpa expects q, k, v, additive_mask, scale, enable_gqa")
            scale = f"float({args[4]})" if args[4] != "None" else "None"
            if scale == "None":
                return (
                    f"self._sdpa("
                    f"{args[0]}, {args[1]}, {args[2]}, "
                    f"mask={args[3]}, scale=None)"
                )
            return (
                f"self._sdpa("
                f"{args[0]}, {args[1]}, {args[2]}, "
                f"mask={args[3]}, scale={scale})"
            )
        if primitive == "layernorm":
            path_operand = node.inputs[0]
            attr_expr = self._param_expr_for_jax_attr(path_operand, local=local, symbols_dict=symbols_dict)
            if attr_expr is not None:
                return f"self.{attr_expr}({args[1]})"
            eps = args[2] if len(args) > 2 else "1e-5"
            bias = args[5] if len(args) > 5 else "True"
            weight_expr = self._param_expr_for_path(
                node.inputs[0],
                node.inputs[4] if len(node.inputs) > 4 else "weight",
                local=local,
                symbols_dict=symbols_dict,
            )
            bias_expr = self._param_expr_for_path(
                node.inputs[0],
                node.inputs[6] if len(node.inputs) > 6 else "bias",
                optional=True,
                local=local,
                symbols_dict=symbols_dict,
            )
            return (
                f"self._layer_norm({args[1]}, "
                f"{weight_expr}, "
                f"({bias_expr} if {bias} else None), "
                f"eps=float({eps}))"
            )
        if primitive == "rmsnorm":
            x = args[0]
            eps = args[1] if len(args) > 1 else "1e-6"
            cast_float = args[3] if len(args) > 3 else "False"
            return (
                f"(self._rms_norm({x}.astype(jnp.float32), None, float({eps})).astype({x}.dtype) "
                f"if {cast_float} else self._rms_norm({x}, None, float({eps})))"
            )
        if primitive == "tensor_like":
            dtype = args[2] if len(args) > 2 else "None"
            return f"({args[0]}.astype(self._dtype_from_name({dtype}) or {args[1]}.dtype) if isinstance({args[0]}, jax.Array) else jnp.array({args[0]}, dtype=(self._dtype_from_name({dtype}) or {args[1]}.dtype)))"
        if primitive == "softmax":
            dim = args[1] if len(args) > 1 else "-1"
            dtype = args[2] if len(args) > 2 else "None"
            if dtype == "None":
                return f"nn.softmax({args[0]}, axis=int({dim}))"
            return f"nn.softmax({args[0]}.astype(self._dtype_from_name({dtype})) if {dtype} != None else {args[0]}, axis=int({dim}))"
        if primitive == "topk":
            return f"self._topk({args[0]}, {args[1]}, dim={args[2]}, largest={args[3]}, sorted_={args[4]})"
        if primitive == "concat":
            if "dim" in attrs:
                return f"self._concat({', '.join(args)}, dim={attrs['dim']})"
            return f"self._concat({', '.join(args[:-1])}, dim={args[-1]})"
        if primitive == "slice":
            size_expr = "None"
            output_type = getattr(node.outputs[0], "type_expr", None) if getattr(node, "outputs", None) else None
            dim_operand = node.inputs[1] if len(node.inputs) > 1 else None
            if isinstance(output_type, TypeTensor) and isinstance(dim_operand, GraphLiteral) and type(dim_operand.value) is int:
                axis = dim_operand.value
                if axis < 0:
                    axis += len(output_type.dims)
                if 0 <= axis < len(output_type.dims):
                    dim_expr = self._dim_token_expr(output_type.dims[axis], local=local, symbols_dict=symbols_dict)
                    if dim_expr is not None:
                        size_expr = dim_expr
            if len(args) >= 4:
                start_operand = node.inputs[2] if len(node.inputs) > 2 else None
                end_operand = node.inputs[3] if len(node.inputs) > 3 else None
                start_expr = args[2].strip()
                end_expr = args[3].strip()
                match = re.fullmatch(r"\((?P<end>.+) - (?P<size>.+)\)", start_expr)
                if size_expr == "None" and match is not None and match.group("end").strip() == end_expr:
                    size_expr = match.group("size").strip()
                elif size_expr == "None" and isinstance(start_operand, GraphLiteral) and start_operand.value == 0:
                    size_expr = end_expr
                elif size_expr == "None":
                    start_type = getattr(start_operand, "type_expr", None)
                    end_type = getattr(end_operand, "type_expr", None)
                    if isinstance(start_type, TypeInt | TypeDim) and isinstance(end_type, TypeInt | TypeDim):
                        size_expr = f"({end_expr} - {start_expr})"
            return f"self._slice({args[0]}, {args[1]}, {args[2]}, {args[3]}, size={size_expr})"
        if primitive == "assign_slice":
            return f"self._assign_slice({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]})"
        if primitive == "conv1d":
            if len(args) != 8:
                raise ValueError("conv1d expects x, weight, bias, stride, padding_left, padding_right, dilation, groups")
            return f"self._conv1d({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, {args[5]}, {args[6]}, {args[7]})"
        if primitive == "_jax_rope":
            return f"self._rope({args[0]}, dims={args[0]}.shape[-1], traditional=bool({args[1]}))"
        if primitive == "params_has_root":
            return f"any(k == {args[0]} or k.startswith(str({args[0]}) + '.') for k in self._flat_tensors)"
        if primitive.startswith("config_") or primitive in {"params_param"}:
            return super()._primitive_expr(primitive, node, local=local, symbols_dict=symbols_dict)
        simple = {
            "reshape": lambda: f"{args[0]}.reshape(tuple(int(x) for x in {args[1]}))",
            "arange": lambda: f"self._arange({args[0]}, {args[1]}, {args[2]})",
            "chunk": lambda: f"jnp.split({args[0]}, indices_or_sections=int({args[2] if len(args) > 2 else attrs.get('parts', '1')}), axis=int({args[1] if len(args) > 1 else attrs.get('dim', '-1')}))",
            "split": lambda: f"self._split_sizes({args[0]}, {args[2] if len(args) > 2 else attrs.get('sizes', '[]')}, {args[1] if len(args) > 1 else attrs.get('dim', '-1')})",
            "sum": lambda: f"{args[0]}.sum(axis=int({args[1] if len(args) > 1 else '-1'}), keepdims=bool({args[2] if len(args) > 2 else 'False'}))",
            "expand": lambda: f"self._expand({args[0]}, {args[1]})",
            "permute": lambda: f"jnp.transpose({args[0]}, axes=tuple(int(x) for x in {args[1]}))",
            "transpose": lambda: f"jnp.swapaxes({args[0]}, int({args[1]}), int({args[2]}))",
            "unsqueeze": lambda: f"self._unsqueeze({args[0]}, {args[1]})",
            "repeat": lambda: f"jnp.repeat({args[0]}, int({args[1]}), axis=(int({args[2]}) if int({args[2]}) >= 0 else int({args[2]}) + len({args[0]}.shape)))",
            "matmul": lambda: f"self._matmul({args[0]}, {args[1]})",
            "where": lambda: f"self._where({args[0]}, {args[1]}, {args[2]})",
            "where_indices": lambda: f"self._where_indices({args[0]})",
            "require": lambda: f"self._require_value({args[0]})",
            "gather": lambda: f"self._gather({args[0]}, {args[1]}, dim={args[2] if len(args) > 2 else '-1'})",
            "scatter": lambda: f"self._scatter({args[0]}, {args[1]}, {args[2]}, dim={args[3] if len(args) > 3 else '-1'})",
            "index_add": lambda: f"self._index_add({args[0]}, {args[1]}, {args[2]}, {args[3] if len(args) > 3 else '0'})",
            "le": lambda: f"({args[0]} <= {args[1]})",
            "eq": lambda: f"self._eq({args[0]}, {args[1]})",
            "and": lambda: f"({args[0]} & {args[1]})",
            "add": lambda: f"self._binary_add({args[0]}, {args[1]})",
            "mul": lambda: f"({args[0]} * {args[1]})",
            "div": lambda: f"({args[0]} / {args[1]})",
            "pow": lambda: f"jnp.power({args[0]}, {args[1]})",
            "floor": lambda: f"jnp.floor({args[0]}) if isinstance({args[0]}, jax.Array) else int({args[0]} // 1)",
            "sqrt": lambda: f"jnp.sqrt({args[0]}) if isinstance({args[0]}, jax.Array) else ({args[0]} ** 0.5)",
            "sin": lambda: f"jnp.sin({args[0]}) if isinstance({args[0]}, jax.Array) else __import__('math').sin(float({args[0]}))",
            "cos": lambda: f"jnp.cos({args[0]}) if isinstance({args[0]}, jax.Array) else __import__('math').cos(float({args[0]}))",
            "exp": lambda: f"jnp.exp({args[0]}) if isinstance({args[0]}, jax.Array) else __import__('math').exp(float({args[0]}))",
            "log": lambda: f"jnp.log({args[0]}) if isinstance({args[0]}, jax.Array) else __import__('math').log(float({args[0]}))",
            "cast": lambda: f"{args[0]}.astype(self._dtype_from_name({args[1]}) or {args[0]}.dtype)",
            "cast_like": lambda: f"{args[0]}.astype({args[1]}.dtype)",
            "dtype_value": lambda: f"(lambda _x: self._dtype_value(_x.dtype, {args[1]}))(self._value({args[0]}))",
            "cumsum": lambda: f"{args[0]}.cumsum(axis=int({args[1] if len(args) > 1 else '-1'}))",
            "empty_like": lambda: f"(lambda _x: jnp.zeros(_x.shape, dtype=_x.dtype))(self._value({args[0]}))",
            "fill": lambda: f"(lambda _x: jnp.full(_x.shape, {args[1]}, dtype=(_x.dtype if self._dtype_from_name({args[2] if len(args) > 2 else 'None'}) is None else self._dtype_from_name({args[2] if len(args) > 2 else 'None'}))))(self._value({args[0]}))",
            "empty": lambda: f"jnp.zeros(tuple(int(x) for x in {args[1]}), dtype=(self._dtype_from_name({args[2] if len(args) > 2 else 'None'}) or {args[0]}.dtype))",
            "zeros": lambda: f"jnp.zeros(tuple(int(x) for x in {args[1]}), dtype=(self._dtype_from_name({args[2] if len(args) > 2 else 'None'}) or {args[0]}.dtype))",
            "full": lambda: f"jnp.full(tuple(int(x) for x in {args[1]}), {args[2]}, dtype=(self._dtype_from_name({args[3] if len(args) > 3 else 'None'}) or {args[0]}.dtype))",
            "zeros_like": lambda: f"(lambda _x: jnp.zeros(_x.shape, dtype=_x.dtype))(self._value({args[0]}))",
            "activations_tanh": lambda: f"jnp.tanh({args[0]})",
            "activations_silu": lambda: f"nn.sigmoid({args[0]}) * {args[0]}",
            "activations_sigmoid": lambda: f"nn.sigmoid({args[0]})",
            "activations_swiglu": lambda: f"(nn.sigmoid({args[0]}) * {args[0]} * {args[0]})",
            "l2norm": lambda: f"({args[0]}.astype(jnp.float32) * jax.lax.rsqrt(({args[0]}.astype(jnp.float32) * {args[0]}.astype(jnp.float32)).mean(axis=-1, keepdims=True) + float({args[1] if len(args) > 1 else '1e-6'}))).astype({args[0]}.dtype)",
            "activations_relu": lambda: f"jnp.maximum({args[0]}, 0)",
            "activations_relu2": lambda: f"(jnp.maximum({args[0]}, 0) * jnp.maximum({args[0]}, 0))",
            "activations_gelu": lambda: f"nn.gelu({args[0]}, approximate=False)",
            "activations_gelu_new": lambda: f"(0.5 * {args[0]} * (1.0 + jnp.tanh(0.7978845608028654 * ({args[0]} + 0.044715 * {args[0]} * {args[0]} * {args[0]}))))",
            "activations_gelu_pytorch_tanh": lambda: f"(0.5 * {args[0]} * (1.0 + jnp.tanh(0.7978845608028654 * ({args[0]} + 0.044715 * {args[0]} * {args[0]} * {args[0]}))))",
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
                return f"jnp.minimum({args[0]}, {max_value})"
            if max_value == "None":
                return f"jnp.maximum({args[0]}, {min_value})"
            return f"jnp.clip({args[0]}, {min_value}, {max_value})"
        if primitive in simple:
            return simple[primitive]()
        raise NotImplementedError(f"direct codegen2-jax unsupported graph op {primitive!r}")

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
        op_expr = f"self._layer_norm({args[1]}, {weight_name}, {bias_name}, eps={eps})"
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

    def _dim_token_expr(self, dim: Any, *, local: set[str], symbols_dict: str) -> str | None:
        if isinstance(dim, bool):
            return repr(int(dim))
        if isinstance(dim, int):
            return repr(dim)
        if isinstance(dim, str):
            if dim.startswith(".."):
                return None
            if dim in local:
                return _dim_ident(dim)
            if dim in self.global_symbol_names:
                return f"{symbols_dict}[{dim!r}]"
            return None
        if isinstance(dim, DimExprBinary):
            left = self._dim_token_expr(dim.left, local=local, symbols_dict=symbols_dict)
            right = self._dim_token_expr(dim.right, local=local, symbols_dict=symbols_dict)
            if left is None or right is None:
                return None
            op = "//" if dim.op == "/" else dim.op
            return f"({left} {op} {right})"
        return repr(dim)


def emit_model_code_from_graph_ir(
    graph: GraphProgram,
    *,
    class_name: str = "AxonJaxModel",
    model_config: dict[str, Any] | None = None,
    profile: bool = False,
) -> str:
    validate_graph_program(graph)
    unsupported = non_obvious_jax_ops(graph)
    if unsupported:
        table = jax_op_table_markdown(graph)
        raise NotImplementedError(
            "codegen2-jax cannot emit this Graph IR yet.\n"
            "Unsupported Graph IR ops:\n"
            f"{table}"
        )
    emitter = _DirectJaxEmitter(program=graph, class_name=class_name, profile=profile)
    body = emitter.emit()
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "import os",
            "import jax",
            "import jax.numpy as jnp",
            "from jax import nn",
            "from safetensors import safe_open",
            "from brainsurgery.synapse.axon.codegen2_jax.core import _jax_materialize_packed_parameters",
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
            "jax.config.update('jax_default_matmul_precision', 'highest')",
            "",
            f"_MODEL_CONFIG = {model_config!r}",
            "",
            body,
        ]
    )


def torch_state_dict_to_jax(state_dict: dict[str, Any]) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch

    return {k: jnp.asarray(v.numpy()) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}


OBVIOUS_JAX_PRIMITIVES = SUPPORTED_JAX_PRIMITIVES

__all__ = [
    "NON_OBVIOUS_JAX_OPS",
    "OBVIOUS_JAX_PRIMITIVES",
    "SHARED_COMMON_PRIMITIVES",
    "SUPPORTED_JAX_PRIMITIVES",
    "JaxUnsupportedOp",
    "emit_model_code_from_graph_ir",
    "non_obvious_jax_ops",
    "jax_op_table_markdown",
    "torch_state_dict_to_jax",
]
