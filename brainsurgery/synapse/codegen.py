from __future__ import annotations

import importlib.resources
import math
from collections.abc import Iterable
from typing import Any

from omegaconf import OmegaConf

from .axon.ast import TypeList, TypeOptional, TypeTensor, TypeTuple, parse_type_expr
from .ops import get_op_module
from .spec_normalize import normalize_synapse_spec_expressions


def load_synapse_torch_op_map() -> dict[str, Any]:
    data_text = (
        importlib.resources.files("brainsurgery.synapse")
        .joinpath("torch_op_map.yaml")
        .read_text(encoding="utf-8")
    )
    loaded = OmegaConf.create(data_text)
    data = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(data, dict):
        raise ValueError("synapse torch op map must be a mapping")
    return {str(key): value for key, value in data.items()}


def emit_model_code_from_synapse_spec(
    spec: dict[str, Any],
    *,
    class_name: str = "GeneratedSynapseModel",
    op_map: dict[str, Any] | None = None,
) -> str:
    if not class_name.isidentifier():
        raise ValueError(f"Invalid class name: {class_name!r}")
    normalized_spec = normalize_synapse_spec_expressions(spec)
    if normalized_spec.get("synapse") != 1:
        raise ValueError("Only synapse: 1 specs are supported")

    resolved_op_map = load_synapse_torch_op_map() if op_map is None else op_map
    _validate_spec_ops(normalized_spec, resolved_op_map)

    model = normalized_spec.get("model")
    if not isinstance(model, dict):
        raise ValueError("spec.model must be a mapping")

    symbols_raw = model.get("symbols", {})

    def _is_codegen_symbol_value(value: Any) -> bool:
        if isinstance(value, (int, float, bool, str)):
            return True
        if isinstance(value, dict):
            return True
        if isinstance(value, (list, tuple)):
            return all(_is_codegen_symbol_value(item) for item in value)
        return False

    symbols = {k: v for k, v in symbols_raw.items() if _is_codegen_symbol_value(v)}

    emitter = _Emitter(class_name=class_name, spec=normalized_spec, symbols=symbols)
    return emitter.render()


class _Emitter:
    def __init__(self, *, class_name: str, spec: dict[str, Any], symbols: dict[str, Any]) -> None:
        self.class_name = class_name
        self.spec = spec
        self.model = spec["model"]
        self.blocks = self.model.get("blocks", {})
        self.symbols = symbols
        types_raw = self.model.get("types", {})
        self.block_io_types = types_raw.get("block_io", {}) if isinstance(types_raw, dict) else {}
        self._counter = 0
        self._active_env: dict[str, str] = {}
        self._active_shape_aliases: dict[str, str] = {}
        self._active_shape_alias_penalties: dict[str, int] = {}
        self._node_hoists: dict[tuple[str, str], str] = {}

    def _shape_symbol_alias_candidates(
        self, *, env: dict[str, str], input_types: dict[str, Any]
    ) -> dict[str, tuple[int, str]]:
        aliases: dict[str, tuple[int, str]] = {}

        def _record(dim: str, penalty: int, expr: str) -> None:
            current = aliases.get(dim)
            if current is None or penalty < current[0]:
                aliases[dim] = (penalty, expr)

        def _collect_aliases(type_expr: Any, src_expr: str, *, penalty: int = 0) -> None:
            if isinstance(type_expr, TypeOptional):
                _collect_aliases(type_expr.inner, src_expr, penalty=penalty + 4)
                return
            if isinstance(type_expr, TypeTensor):
                for axis, dim in enumerate(type_expr.dims):
                    if isinstance(dim, str) and dim not in env:
                        _record(dim, penalty, f"int({src_expr}.shape[{axis}])")
                return
            if isinstance(type_expr, TypeList):
                _collect_aliases(type_expr.item, f"{src_expr}[0]", penalty=penalty + 3)
                return
            if isinstance(type_expr, TypeTuple):
                for idx, item in enumerate(type_expr.items):
                    _collect_aliases(item, f"{src_expr}[{idx}]", penalty=penalty + 2)
                return

        for input_name, type_expr in input_types.items():
            if input_name not in env or not isinstance(type_expr, str):
                continue
            try:
                parsed = parse_type_expr(type_expr)
            except Exception:
                continue
            src = env[input_name]
            _collect_aliases(parsed, src)
        return aliases

    def _shape_symbol_aliases(
        self, *, env: dict[str, str], input_types: dict[str, Any]
    ) -> dict[str, str]:
        return {
            dim: expr
            for dim, (_, expr) in self._shape_symbol_alias_candidates(
                env=env, input_types=input_types
            ).items()
        }

    def _shape_symbol_alias_penalties(
        self, *, env: dict[str, str], input_types: dict[str, Any]
    ) -> dict[str, int]:
        return {
            dim: penalty
            for dim, (penalty, _) in self._shape_symbol_alias_candidates(
                env=env, input_types=input_types
            ).items()
        }

    def render(self) -> str:
        lines: list[str] = []
        lines.extend(
            [
                "from __future__ import annotations",
                "",
                "from typing import Any",
                "",
                "from brainsurgery.synapse.axon.ast.path import (",
                "    resolve_path_expr_to_key,",
                "    resolve_static_path_expr_to_key,",
                "    runtime_value_to_path_expr,",
                ")",
                "",
                "from brainsurgery.synapse.mxfp4 import materialize_mxfp4_aliases",
                "",
                "import math",
                "import torch",
                "from torch import nn",
                "from torch.nn import functional as F",
                "",
                "inf = float('inf')",
                "nan = float('nan')",
                "",
                "",
                f"class {self.class_name}(nn.Module):",
                "    def __init__(self, state_dict: dict[str, torch.Tensor] | None = None, runtime_state_dict: Any | None = None) -> None:",
                "        super().__init__()",
                "        self._state: dict[str, torch.Tensor] = {}",
                "        self._runtime_state_dict = runtime_state_dict",
                f"        self._symbols: dict[str, Any] = {repr(self.symbols)}",
                "        self._trace_enabled = False",
                "        self.trace_ops: list[dict[str, Any]] = []",
                "        if state_dict is not None:",
                "            self.load_state_dict_tensors(state_dict)",
                "",
                "    @classmethod",
                '    def from_state_dict(cls, state_dict: dict[str, torch.Tensor], runtime_state_dict: Any | None = None) -> "'
                + self.class_name
                + '":',
                "        return cls(state_dict=state_dict, runtime_state_dict=runtime_state_dict)",
                "",
                "    def load_state_dict_tensors(self, state_dict: dict[str, torch.Tensor]) -> None:",
                "        loaded = dict(state_dict)",
                "        materialize_mxfp4_aliases(loaded, drop_packed=True)",
                "        self._state = loaded",
                "",
                "    def _param(self, path: str) -> torch.Tensor:",
                "        return self._state_tensor_from_resolved_path(path, field='_param')",
                "",
                "    def _join_scope(self, left: str, right: str) -> str:",
                "        if not left:",
                "            return right",
                "        if not right:",
                "            return left",
                '        return f"{left}.{right}"',
                "",
                "    def _scope_of(self, node_path: str) -> str:",
                "        if '.' not in node_path:",
                "            return ''",
                "        return node_path.rsplit('.', 1)[0]",
                "",
                "    def _is_runtime_expr_payload(self, value: Any) -> bool:",
                "        if isinstance(value, dict):",
                "            return isinstance(value.get('_expr'), str)",
                "        if isinstance(value, (list, tuple)):",
                "            return all(self._is_runtime_expr_payload(item) or isinstance(item, (str, int, float, bool, type(None))) for item in value)",
                "        return False",
                "",
                "    def _resolve_call_path_arg(self, raw: Any, env: dict[str, Any]) -> Any:",
                "        if not isinstance(raw, (str, dict)):",
                "            return raw",
                "        try:",
                "            expr = runtime_value_to_path_expr(raw, op_name='call path arg')",
                "            return resolve_path_expr_to_key(expr, self._path_template_env(env), op_name='call path arg')",
                "        except Exception:",
                "            return raw",
                "",
                "    def _path_template_env(self, env: dict[str, Any] | None, *, symbols: dict[str, Any] | None = None) -> dict[str, Any]:",
                "        if not isinstance(env, dict):",
                "            return {}",
                "        raw_env = dict(env)",
                "        resolved: dict[str, Any] = {}",
                "        symbol_values = self._symbols if symbols is None else dict(symbols)",
                "        for key, value in raw_env.items():",
                "            if self._is_runtime_expr_payload(value):",
                "                try:",
                "                    resolved[key] = self._eval_expr(value, raw_env, symbol_values)",
                "                    continue",
                "                except Exception:",
                "                    pass",
                "            resolved[key] = value",
                "        for key, value in list(resolved.items()):",
                "            if isinstance(value, dict) and value.get('_expr') != 'path':",
                "                continue",
                "            if not isinstance(value, (str, dict)):",
                "                continue",
                "            try:",
                "                expr = runtime_value_to_path_expr(value, op_name='path template env')",
                "                resolved[key] = resolve_path_expr_to_key(expr, resolved, op_name='path template env')",
                "            except Exception:",
                "                pass",
                "        return resolved",
                "",
                "    def _resolve_state_path(self, *, node_path: str, raw_path: Any, env: dict[str, Any] | None = None) -> str:",
                "        expr = runtime_value_to_path_expr(raw_path, op_name='state path')",
                "        token = resolve_path_expr_to_key(expr, self._path_template_env(env), op_name='state path')",
                "        if expr.absolute:",
                "            return token",
                "        scope = self._scope_of(node_path)",
                "        scope_parts = scope.split('.') if scope else []",
                "        synthetic_prefixes = ('n_for_', 'n_if_', 'n_else_', 'n_call_', 'n_op_')",
                "        while scope_parts:",
                "            if len(scope_parts) >= 2 and scope_parts[-1].isdigit() and any(scope_parts[-2].startswith(prefix) for prefix in synthetic_prefixes):",
                "                scope_parts.pop()",
                "                scope_parts.pop()",
                "                continue",
                "            if any(scope_parts[-1].startswith(prefix) for prefix in synthetic_prefixes):",
                "                scope_parts.pop()",
                "                continue",
                "            break",
                "        normalized_scope = '.'.join(scope_parts)",
                "        scoped = self._join_scope(normalized_scope, token)",
                "        if scoped in self._state:",
                "            return scoped",
                "        return scoped",
                "",
                "    def _state_tensor_from_resolved_path(self, path: str, *, field: str) -> torch.Tensor:",
                "        resolved = path[2:] if isinstance(path, str) and path.startswith('@@') else path",
                "        if resolved not in self._state:",
                "            alternatives = self._state_key_alternatives(resolved, limit=8)",
                "            alt_text = ', '.join(alternatives) if alternatives else '<none>'",
                "            raise ValueError(f'{field} tensor not found at path: {resolved}. Alternatives: {alt_text}')",
                "        return self._state[resolved]",
                "",
                "    def _state_tensor_from_path(self, *, node_path: str, raw_path: Any, field: str, env: dict[str, Any] | None = None) -> torch.Tensor:",
                "        path = self._resolve_state_path(node_path=node_path, raw_path=raw_path, env=env)",
                "        return self._state_tensor_from_resolved_path(path, field=field)",
                "",
                "    def _state_key_alternatives(self, key: str, *, limit: int = 8) -> list[str]:",
                "        if not isinstance(key, str) or not key:",
                "            return []",
                "        keys = [k for k in self._state.keys() if isinstance(k, str)]",
                "        if not keys:",
                "            return []",
                "        out: list[str] = []",
                "        seen: set[str] = set()",
                "",
                "        def _add(candidate: str) -> None:",
                "            if candidate in seen:",
                "                return",
                "            seen.add(candidate)",
                "            out.append(candidate)",
                "",
                "        segments = key.split('.')",
                "        leaf = segments[-1]",
                "        for existing in keys:",
                "            if existing.endswith('.' + leaf) or existing == leaf:",
                "                _add(existing)",
                "                if len(out) >= limit:",
                "                    return out",
                "        if len(segments) >= 2:",
                "            tail2 = '.'.join(segments[-2:])",
                "            for existing in keys:",
                "                if existing.endswith('.' + tail2) or existing == tail2:",
                "                    _add(existing)",
                "                    if len(out) >= limit:",
                "                        return out",
                "        for prefix in ('model.', 'transformer.'):",
                "            prefixed = prefix + key",
                "            if prefixed in self._state:",
                "                _add(prefixed)",
                "            if len(out) >= limit:",
                "                return out",
                "        if key.startswith('model.') and key[len('model.'):] in self._state:",
                "            _add(key[len('model.'):])",
                "        if key.startswith('transformer.') and key[len('transformer.'):] in self._state:",
                "            _add(key[len('transformer.'):])",
                "        if len(out) >= limit:",
                "            return out",
                "        for existing in keys:",
                "            if key in existing or existing in key:",
                "                _add(existing)",
                "                if len(out) >= limit:",
                "                    return out",
                "        return out",
                "",
                "    def _resolve_param_scope(self, scope: str, env: dict[str, Any]) -> str:",
                "        token = scope.strip()",
                "        if not token:",
                "            return token",
                "        raw = token if token.startswith('@') else '@@' + token",
                "        try:",
                "            return resolve_path_expr_to_key(raw, self._path_template_env(env), op_name='parameter scope')",
                "        except Exception:",
                "            return token",
                "",
                "    def _pick_param_path(self, scope: str, candidates: list[Any], env: dict[str, Any]) -> str:",
                "        scoped: list[str] = []",
                "        resolved_scope = self._resolve_param_scope(scope, env)",
                "        for candidate in candidates:",
                "            if not isinstance(candidate, (str, dict)):",
                "                continue",
                "            if isinstance(candidate, str) and candidate and not candidate.startswith('@'):",
                "                candidate = '@' + candidate",
                "            expr = runtime_value_to_path_expr(candidate, op_name='parameter path')",
                "            key = resolve_path_expr_to_key(expr, self._path_template_env(env), op_name='parameter path')",
                "            if expr.absolute:",
                "                if key:",
                "                    scoped.append(key)",
                "                continue",
                "            base = self._join_scope(resolved_scope, key)",
                "            scoped.append(base)",
                "        if not scoped:",
                "            raise ValueError('parameter candidate list is empty')",
                "        for candidate in scoped:",
                "            if candidate in self._state:",
                "                return candidate",
                "        return scoped[0]",
                "",
                "    def _pick_param_from_single(self, scope: str, candidate: Any, env: dict[str, Any]) -> str:",
                "        resolved_scope = self._resolve_param_scope(scope, env)",
                "        if isinstance(candidate, str) and candidate and not candidate.startswith('@'):",
                "            candidate = '@' + candidate",
                "        expr = runtime_value_to_path_expr(candidate, op_name='parameter path')",
                "        key = resolve_path_expr_to_key(expr, self._path_template_env(env), op_name='parameter path')",
                "        if expr.absolute:",
                "            return key",
                "        base = self._join_scope(resolved_scope, key)",
                "        if base in self._state:",
                "            return base",
                "        return base",
                "",
                "    def _infer_param_path(self, node_spec: dict[str, Any], *, node_path: str, param_name: str, env: dict[str, Any]) -> str:",
                "        abs_path = node_spec.get('_abs_path')",
                "        if isinstance(abs_path, (str, dict)) and abs_path:",
                "            scope = resolve_path_expr_to_key(abs_path, self._path_template_env(env), op_name='_abs_path')",
                "        else:",
                "            scope = self._scope_of(node_path)",
                "        explicit_params = node_spec.get('_params')",
                "        candidate = node_spec.get(param_name)",
                "        if isinstance(candidate, (str, dict)):",
                "            if not (isinstance(candidate, str) and candidate == param_name):",
                "                if not isinstance(candidate, str) or candidate.startswith('@') or '.' in candidate:",
                "                    return self._pick_param_from_single(scope, candidate, env)",
                "        if isinstance(explicit_params, dict):",
                "            explicit = explicit_params.get(param_name)",
                "            if isinstance(explicit, (str, dict)):",
                "                return self._pick_param_from_single(scope, explicit, env)",
                "            if isinstance(explicit, list) and all(isinstance(item, (str, dict)) for item in explicit):",
                "                return self._pick_param_path(scope, explicit, env)",
                "        return self._pick_param_from_single(scope, param_name, env)",
                "",
                "    def _expr_config_root(self) -> dict[str, Any]:",
                f"        model = {repr(self.model)}",
                "        cfg = model.get('config') if isinstance(model, dict) else None",
                "        return cfg if isinstance(cfg, dict) else {}",
                "",
                "    def _resolve_config_path_key(self, raw: Any, env: dict[str, Any], op_name: str = 'Config') -> str:",
                "        return resolve_path_expr_to_key(raw, self._path_template_env(env, symbols=self._symbols), op_name=op_name)",
                "",
                "    def _expr_config_lookup(self, key: str) -> tuple[bool, Any]:",
                "        value: Any = self._expr_config_root()",
                "        for part in key.split('.'):",
                "            if not isinstance(value, dict) or part not in value:",
                "                return False, None",
                "            value = value[part]",
                "        return True, value",
                "",
                "    def _expr_params_has_root(self, root: str) -> bool:",
                "        if root == '':",
                "            return True",
                "        prefix = root + '.'",
                "        for key in self._state.keys():",
                "            if not isinstance(key, str):",
                "                continue",
                "            if key == root or key.startswith(prefix):",
                "                return True",
                "        return False",
                "",
                "    def _eval_expr_call(self, callee: str, args: list[Any], kwargs: dict[str, Any], env: dict[str, Any], symbols: dict[str, Any]) -> Any:",
                "        inline_path_callexprs = {'Config.value', 'Config.has_key', 'Config.has_value', 'Config.int', 'Config.dim', 'Config.float', 'Config.str', 'Config.bool', 'Config.list'}",
                "        callee_base = callee.split('@', 1)[0] if '@' in callee else callee",
                "        if '@' in callee and callee_base in inline_path_callexprs:",
                "            suffix = callee[len(callee_base):]",
                "            if not suffix or not suffix.startswith('@'):",
                "                raise ValueError(f'Unsupported call expression: {callee!r}')",
                "            if args:",
                "                raise ValueError(f'{callee_base} expression call cannot mix inline @path and positional args')",
                "            args = [suffix]",
                "            callee = callee_base",
                "        if callee in {'sqrt', 'Prelude.sqrt', 'Math.sqrt', 'log', 'Prelude.log', 'Math.log', 'exp', 'Prelude.exp', 'Math.exp', 'sin', 'Prelude.sin', 'Math.sin', 'cos', 'Prelude.cos', 'Math.cos'}:",
                "            fn_name = callee.split('.', 1)[-1]",
                "            if kwargs:",
                "                raise ValueError(f'{fn_name} expression call does not support kwargs')",
                "            if len(args) != 1:",
                "                raise ValueError(f'{fn_name} expression call expects exactly one positional argument')",
                "            arg = args[0]",
                "            if isinstance(arg, bool) or not isinstance(arg, (int, float)):",
                "                raise ValueError(f'{fn_name} expression call expects numeric argument')",
                "            arg_f = float(arg)",
                "            if fn_name == 'sqrt':",
                "                return math.sqrt(arg_f)",
                "            if fn_name == 'log':",
                "                return math.log(arg_f)",
                "            if fn_name == 'exp':",
                "                return math.exp(arg_f)",
                "            if fn_name == 'sin':",
                "                return math.sin(arg_f)",
                "            if fn_name == 'cos':",
                "                return math.cos(arg_f)",
                "            raise ValueError(f'Unsupported unary expression call: {callee!r}')",
                "        if callee in {'pow', 'Prelude.pow', 'Math.pow'}:",
                "            if kwargs:",
                "                raise ValueError('pow expression call does not support kwargs')",
                "            if len(args) != 2:",
                "                raise ValueError('pow expression call expects exactly two positional arguments')",
                "            left, right = args",
                "            if isinstance(left, bool) or not isinstance(left, (int, float)):",
                "                raise ValueError('pow expression call expects numeric arguments')",
                "            if isinstance(right, bool) or not isinstance(right, (int, float)):",
                "                raise ValueError('pow expression call expects numeric arguments')",
                "            return math.pow(float(left), float(right))",
                "        if callee in {'abs', 'Prelude.abs'}:",
                "            if kwargs:",
                "                raise ValueError('abs expression call does not support kwargs')",
                "            if len(args) != 1:",
                "                raise ValueError('abs expression call expects exactly one positional argument')",
                "            arg = args[0]",
                "            if isinstance(arg, bool) or not isinstance(arg, (int, float)):",
                "                raise ValueError('abs expression call expects numeric argument')",
                "            return abs(arg)",
                "        if callee in {'min', 'Prelude.min'}:",
                "            if kwargs:",
                "                raise ValueError('min expression call does not support kwargs')",
                "            if len(args) < 1:",
                "                raise ValueError('min expression call expects at least one positional argument')",
                "            if any(isinstance(arg, bool) or not isinstance(arg, (int, float)) for arg in args):",
                "                raise ValueError('min expression call expects numeric arguments')",
                "            return min(args)",
                "        if callee in {'max', 'Prelude.max'}:",
                "            if kwargs:",
                "                raise ValueError('max expression call does not support kwargs')",
                "            if len(args) < 1:",
                "                raise ValueError('max expression call expects at least one positional argument')",
                "            if any(isinstance(arg, bool) or not isinstance(arg, (int, float)) for arg in args):",
                "                raise ValueError('max expression call expects numeric arguments')",
                "            return max(args)",
                "        if callee in {'Config.has_key', 'Config.has_value', 'Config.int', 'Config.dim', 'Config.float', 'Config.str', 'Config.bool', 'Config.list', 'Config.value'}:",
                "            if len(args) != 1:",
                "                raise ValueError(f'{callee} expression call expects one non-empty Path key')",
                "            if 'root' in kwargs:",
                "                raise ValueError(f'{callee} expression call does not support root kwarg')",
                "            key = self._resolve_config_path_key(args[0], {**symbols, **env}, callee)",
                "            found, value = self._expr_config_lookup(key)",
                "            if callee == 'Config.has_key':",
                "                if 'default' in kwargs:",
                "                    raise ValueError('Config.has_key expression call does not support default kwarg')",
                "                return bool(found)",
                "            if callee == 'Config.has_value':",
                "                if 'default' in kwargs:",
                "                    raise ValueError('Config.has_value expression call does not support default kwarg')",
                "                return bool(found) and value is not None",
                "            if not found:",
                "                if 'default' not in kwargs:",
                "                    raise KeyError(f'{callee} expression call missing required config key: {key}')",
                "                value = kwargs['default']",
                "            if callee in {'Config.int', 'Config.dim'}:",
                "                if isinstance(value, bool):",
                "                    raise ValueError(f'{callee} expression call expected int')",
                "                if isinstance(value, int):",
                "                    return int(value)",
                "                if isinstance(value, str):",
                "                    raw = value.strip()",
                "                    if raw and (raw.isdigit() or (raw[0] in ('+', '-') and raw[1:].isdigit())):",
                "                        return int(raw)",
                "                raise ValueError(f'{callee} expression call expected int')",
                "            if callee == 'Config.float':",
                "                if isinstance(value, bool):",
                "                    raise ValueError('Config.float expression call expected float')",
                "                if isinstance(value, (int, float)):",
                "                    return float(value)",
                "                if isinstance(value, str):",
                "                    raw = value.strip()",
                "                    if raw:",
                "                        return float(raw)",
                "                raise ValueError('Config.float expression call expected float')",
                "            if callee == 'Config.str':",
                "                if not isinstance(value, str):",
                "                    raise ValueError('Config.str expression call expected string')",
                "                return value",
                "            if callee == 'Config.bool':",
                "                if isinstance(value, bool):",
                "                    return value",
                "                if isinstance(value, str):",
                "                    raw = value.strip().lower()",
                "                    if raw == 'true':",
                "                        return True",
                "                    if raw == 'false':",
                "                        return False",
                "                raise ValueError('Config.bool expression call expected bool')",
                "            if callee == 'Config.list':",
                "                if isinstance(value, list):",
                "                    return value",
                "                if isinstance(value, tuple):",
                "                    return list(value)",
                "                raise ValueError('Config.list expression call expected list')",
                "            if callee == 'Config.value':",
                "                return value",
                "        if callee in {'Params.has_root', 'Params.root'}:",
                "            if len(args) != 1 or not isinstance(args[0], str):",
                "                raise ValueError(f'{callee} expression call expects one string root argument')",
                "            root = args[0]",
                "            if callee == 'Params.has_root':",
                "                if 'default' in kwargs:",
                "                    raise ValueError('Params.has_root expression call does not support default kwarg')",
                "                return bool(self._expr_params_has_root(root))",
                "            default_value = kwargs.get('default', '')",
                "            if not isinstance(default_value, str):",
                "                raise ValueError('Params.root expression call default must resolve to string')",
                "            return root if self._expr_params_has_root(root) else default_value",
                "        raise ValueError(f'Unsupported call expression: {callee!r}')",
                "",
                "    def _safe_get(self, env: dict[str, Any], name: str) -> Any:",
                "        if name not in env:",
                '            raise ValueError(f"Missing variable in graph env: {name}")',
                "        return env[name]",
                "",
                "    def _first_tensor(self, value: Any) -> torch.Tensor | None:",
                "        if torch.is_tensor(value):",
                "            return value",
                "        if isinstance(value, (list, tuple)):",
                "            for item in value:",
                "                tensor = self._first_tensor(item)",
                "                if tensor is not None:",
                "                    return tensor",
                "            return None",
                "        if isinstance(value, dict):",
                "            for item in value.values():",
                "                tensor = self._first_tensor(item)",
                "                if tensor is not None:",
                "                    return tensor",
                "            return None",
                "        return None",
                "",
                "    def _reset_trace(self) -> None:",
                "        self.trace_ops = []",
                "",
                "    def _trace_op(self, node_path: str, op: str, bind: str, value: Any, **meta: Any) -> None:",
                "        if not bool(getattr(self, '_trace_enabled', False)):",
                "            return",
                "        tensor = self._first_tensor(value)",
                "        if tensor is None:",
                "            return",
                "        payload = {",
                "            'node_path': str(node_path),",
                "            'op': str(op),",
                "            'bind': str(bind),",
                "            'dtype': str(tensor.dtype),",
                "            'tensor': tensor.detach().float().cpu(),",
                "        }",
                "        for key, item in meta.items():",
                "            payload[str(key)] = item",
                "        self.trace_ops.append(payload)",
                "",
                "    def _record_runtime_value(self, key: str, value: Any) -> None:",
                "        runtime_state_dict = self._runtime_state_dict",
                "        if runtime_state_dict is None:",
                "            return",
                "        if torch.is_tensor(value):",
                "            runtime_state_dict[key] = value.detach().clone()",
                "            return",
                "        if isinstance(value, (list, tuple)):",
                "            for idx, item in enumerate(value):",
                '                self._record_runtime_value(f"{key}[{idx}]", item)',
                "            return",
                "        if isinstance(value, dict):",
                "            for item_key, item_value in value.items():",
                '                self._record_runtime_value(f"{key}.{item_key}", item_value)',
                "",
                "    def _prepare_env(self, input_ids: torch.Tensor | None, inputs: dict[str, Any], input_specs: dict[str, Any]) -> dict[str, Any]:",
                "        env = {'input_ids': input_ids, **inputs} if input_ids is not None else dict(inputs)",
                "        for input_name, input_spec in input_specs.items():",
                "            optional = isinstance(input_spec, dict) and bool(input_spec.get('optional', False))",
                "            if input_name in env:",
                "                continue",
                "            if optional:",
                "                env[input_name] = None",
                "            else:",
                "                raise ValueError(f'Missing required input: {input_name}')",
                "        return env",
                "",
                "    def _for_values(self, *, from_value: int, to_value: int, step_value: int):",
                "        if not isinstance(from_value, int):",
                "            raise ValueError(f'for _from must resolve to int, got {from_value!r}')",
                "        if not isinstance(to_value, int):",
                "            raise ValueError(f'for _to must resolve to int, got {to_value!r}')",
                "        if not isinstance(step_value, int):",
                "            raise ValueError(f'for _step must resolve to int, got {step_value!r}')",
                "        if step_value == 0:",
                "            raise ValueError('for _step must be non-zero')",
                "        return range(from_value, to_value, step_value)",
                "",
            ]
        )

        blocks = self.model.get("blocks", {})
        if isinstance(blocks, dict):
            for block_name, block_spec in blocks.items():
                lines.extend(self._render_block_method(block_name, block_spec))

        lines.extend(self._render_forward())
        lines.extend(self._render_generate())
        return "\n".join(lines) + "\n"

    def _render_block_method(self, block_name: str, block_spec: Any) -> list[str]:
        if not isinstance(block_spec, dict):
            raise ValueError("block spec must be mapping")
        inputs = block_spec.get("inputs", {})
        if not isinstance(inputs, dict):
            raise ValueError("block inputs must be mapping")
        graph = block_spec.get("graph")
        if not isinstance(graph, list):
            raise ValueError("block graph must be list")
        outputs = block_spec.get("outputs", {})
        if not isinstance(outputs, dict):
            raise ValueError("block outputs must be mapping")

        input_py_names = self._unique_py_name_map(inputs.keys(), reserved={"self", "scope"})
        arg_names = [input_py_names[name] for name in inputs]
        env: dict[str, str] = {name: input_py_names[name] for name in inputs}
        block_types = self.block_io_types.get(block_name, {})
        input_types = block_types.get("inputs", {}) if isinstance(block_types, dict) else {}
        shape_aliases = (
            self._shape_symbol_aliases(env=env, input_types=input_types)
            if isinstance(input_types, dict)
            else {}
        )
        shape_alias_penalties = (
            self._shape_symbol_alias_penalties(env=env, input_types=input_types)
            if isinstance(input_types, dict)
            else {}
        )

        sig = ", ".join(["self", *arg_names, "scope: str"])
        lines = [f"    def _block_{self._py_name(block_name)}({sig}) -> tuple[Any, ...]:"]
        lines.append("        emitter = self")
        lines.append("        env: dict[str, Any] = {}")
        for syn_name, py_name in env.items():
            lines.append(f"        env[{syn_name!r}] = {py_name}")

        prev_shape_aliases = self._active_shape_aliases
        prev_shape_alias_penalties = self._active_shape_alias_penalties
        self._active_shape_aliases = shape_aliases
        self._active_shape_alias_penalties = shape_alias_penalties
        try:
            body = self._compile_graph(graph=graph, env=env, scope_var="scope", indent="        ")
        finally:
            self._active_shape_aliases = prev_shape_aliases
            self._active_shape_alias_penalties = prev_shape_alias_penalties
        lines.extend(body)

        return_values: list[str] = []
        for _, ref in outputs.items():
            if isinstance(ref, str):
                return_values.append(env[ref])
            else:
                raise ValueError("block outputs currently support string refs only")
        if len(return_values) == 1:
            lines.append(f"        return {return_values[0]}")
        else:
            tuple_expr = ", ".join(return_values)
            lines.append(f"        return ({tuple_expr})")
        lines.append("")
        return lines

    def _render_forward(self) -> list[str]:
        graph = self.model.get("graph")
        if not isinstance(graph, list):
            raise ValueError("model.graph must be list")
        inputs = self.model.get("inputs", {})
        if not isinstance(inputs, dict):
            raise ValueError("model.inputs must be mapping")
        outputs = self.model.get("outputs", {})
        if not isinstance(outputs, dict):
            raise ValueError("model.outputs must be mapping")

        lines = [
            "    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:",
            f"        input_specs = {repr(inputs)}",
            "        env = self._prepare_env(input_ids, inputs, input_specs)",
            "        scope = ''",
            "        emitter = self",
        ]

        input_py_names = self._unique_py_name_map(inputs.keys(), reserved={"self", "scope", "env"})
        env: dict[str, str] = {}
        for name, input_spec in inputs.items():
            is_optional = isinstance(input_spec, dict) and bool(input_spec.get("optional", False))
            py_name = input_py_names[name]
            if is_optional:
                lines.append(f"        {py_name} = env.get({name!r})")
            else:
                lines.append(f"        {py_name} = self._safe_get(env, {name!r})")
            env[name] = py_name
        main_types = self.block_io_types.get("main", {})
        main_input_types = main_types.get("inputs", {}) if isinstance(main_types, dict) else {}
        shape_aliases = (
            self._shape_symbol_aliases(env=env, input_types=main_input_types)
            if isinstance(main_input_types, dict)
            else {}
        )
        shape_alias_penalties = (
            self._shape_symbol_alias_penalties(env=env, input_types=main_input_types)
            if isinstance(main_input_types, dict)
            else {}
        )

        prev_shape_aliases = self._active_shape_aliases
        prev_shape_alias_penalties = self._active_shape_alias_penalties
        self._active_shape_aliases = shape_aliases
        self._active_shape_alias_penalties = shape_alias_penalties
        try:
            lines.extend(
                self._compile_graph(graph=graph, env=env, scope_var="scope", indent="        ")
            )
        finally:
            self._active_shape_aliases = prev_shape_aliases
            self._active_shape_alias_penalties = prev_shape_alias_penalties

        lines.append("        outputs: dict[str, Any] = {}")
        for out_name, ref in outputs.items():
            if isinstance(ref, str):
                lines.append(f"        outputs[{out_name!r}] = {env[ref]}")
            elif isinstance(ref, dict) and isinstance(ref.get("from"), str):
                lines.append(f"        outputs[{out_name!r}] = {env[ref['from']]}")
            else:
                raise ValueError(f"Unsupported output ref shape: {ref!r}")

        lines.append("        if 'logits' in outputs and len(outputs) == 1:")
        lines.append("            return outputs['logits']")
        lines.append("        return outputs")
        lines.append("")
        return lines

    def _render_generate(self) -> list[str]:
        inputs = self.model.get("inputs", {})
        if not isinstance(inputs, dict):
            raise ValueError("model.inputs must be mapping")
        outputs = self.model.get("outputs", {})
        if not isinstance(outputs, dict):
            raise ValueError("model.outputs must be mapping")

        state_input_name: str | None = None
        for candidate in (
            "past_key_values",
            "past_kv",
            "past",
            "cache_params",
            "cache_state",
            "state",
        ):
            if candidate in inputs:
                state_input_name = candidate
                break
        use_cache_name = "use_cache" if "use_cache" in inputs else None
        state_output_names = [
            name
            for name in (
                "past_key_values",
                "present_key_values",
                "new_kv",
                "past_kv",
                "cache_params",
                "cache_state",
                "state",
            )
            if name in outputs
        ]

        lines = [
            "    def generate(self, input_ids: torch.Tensor, *, eos_token_id: int, max_len: int, attention_mask: torch.Tensor | None = None, attn_mask: torch.Tensor | None = None) -> torch.Tensor:",
            "        if input_ids.ndim != 2:",
            "            raise ValueError('input_ids must be rank-2 [batch, seq]')",
            "        if max_len <= 0:",
            "            raise ValueError('max_len must be > 0')",
            "        if attention_mask is not None and attn_mask is not None:",
            "            raise ValueError('pass at most one of attention_mask or attn_mask')",
            "        mask = attention_mask if attention_mask is not None else attn_mask",
            "        if mask is not None:",
            "            if mask.ndim != 2:",
            "                raise ValueError('attention_mask must be rank-2 [batch, seq]')",
            "            if mask.shape != input_ids.shape:",
            "                raise ValueError('attention_mask must have same shape as input_ids')",
            "        if input_ids.size(1) >= max_len:",
            "            return input_ids[:, :max_len]",
            "",
            "        batch, start_len = input_ids.shape",
            "        generated = input_ids.new_empty((batch, max_len))",
            "        generated[:, :start_len] = input_ids",
            "        generated_mask = None",
            "        if mask is not None:",
            "            generated_mask = mask.new_zeros((batch, max_len))",
            "            generated_mask[:, :start_len] = mask",
            "        cache_state = None",
            "        finished = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)",
            "        cur_len = start_len",
            "        was_training = self.training",
            "        self.eval()",
            "        try:",
            "            with torch.inference_mode():",
            "                while cur_len < max_len and not torch.all(finished):",
            "                    step_input = generated[:, :cur_len] if cache_state is None else generated[:, cur_len - 1:cur_len]",
            "                    call_kwargs: dict[str, Any] = {}",
            "                    if generated_mask is not None:",
        ]
        has_mask_input = False
        if "attention_mask" in inputs:
            has_mask_input = True
            lines.append(
                "                        call_kwargs['attention_mask'] = generated_mask[:, :cur_len]"
            )
        if "attn_mask" in inputs:
            has_mask_input = True
            lines.append(
                "                        call_kwargs['attn_mask'] = generated_mask[:, :cur_len]"
            )
        if not has_mask_input:
            lines.append("                        pass")
        if state_input_name is not None:
            lines.append(f"                    call_kwargs[{state_input_name!r}] = cache_state")
        if use_cache_name is not None:
            lines.append(f"                    call_kwargs[{use_cache_name!r}] = True")
        lines.extend(
            [
                "                    model_out = self.forward(step_input, **call_kwargs)",
                "                    if isinstance(model_out, dict):",
                "                        if 'logits' in model_out:",
                "                            logits = model_out['logits']",
                "                        elif len(model_out) == 1:",
                "                            logits = next(iter(model_out.values()))",
                "                        else:",
                "                            raise KeyError(\"Expected 'logits' in model outputs or a single unnamed output\")",
            ]
        )
        for idx, out_name in enumerate(state_output_names):
            if idx == 0:
                lines.append(f"                        if {out_name!r} in model_out:")
            else:
                lines.append(f"                        elif {out_name!r} in model_out:")
            lines.append(f"                            cache_state = model_out[{out_name!r}]")
        lines.extend(
            [
                "                    else:",
                "                        logits = model_out",
                "                    next_token = torch.argmax(logits[:, -1, :], dim=-1)",
                "                    next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)",
                "                    generated[:, cur_len] = next_token",
                "                    finished = torch.logical_or(finished, next_token == eos_token_id)",
                "                    if generated_mask is not None:",
                "                        generated_mask[:, cur_len] = 1",
                "                    cur_len += 1",
                "        finally:",
                "            if was_training:",
                "                self.train()",
                "        return generated[:, :cur_len]",
                "",
            ]
        )
        return lines

    def _compile_graph(
        self, *, graph: list[Any], env: dict[str, str], scope_var: str, indent: str
    ) -> list[str]:
        lines: list[str] = []
        for item in graph:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f"Invalid graph item: {item!r}")
            node_name, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                raise ValueError(f"Invalid node spec: {node_spec!r}")

            inner_indent = indent

            op = node_spec.get("_op")
            if op == "for":
                raise ValueError("codegen no longer supports for-nodes; flatten/lower first")
                continue

            if op == "call":
                lines.extend(
                    self._compile_block_call(
                        node_spec=node_spec, env=env, scope_var=scope_var, indent=inner_indent
                    )
                )
                continue

            if "graph" in node_spec and op is None:
                nested = node_spec.get("graph")
                if not isinstance(nested, list):
                    raise ValueError("node graph must be list")
                child_scope = self._fresh("scope")
                lines.append(
                    f"{inner_indent}{child_scope} = self._join_scope({scope_var}, {node_name!r})"
                )
                lines.extend(
                    self._compile_graph(
                        graph=nested, env=env, scope_var=child_scope, indent=inner_indent
                    )
                )
                continue

            if not isinstance(op, str):
                raise ValueError(f"node {node_name!r} missing op")

            trace_node_path = self._fresh("trace_node_path")
            lines.append(
                f"{inner_indent}{trace_node_path} = self._join_scope({scope_var}, {node_name!r})"
            )
            node_path = scope_var
            if self._op_uses_node_path(op, node_spec):
                node_path = trace_node_path
            lines.extend(
                self._compile_op(
                    op=op,
                    node_spec=node_spec,
                    env=env,
                    node_path_var=node_path,
                    scope_var=scope_var,
                    indent=inner_indent,
                )
            )
            for out_name in self._node_output_names(node_spec):
                _out_var = env.get(out_name)
                if isinstance(_out_var, str):
                    lines.append(f"{inner_indent}env[{out_name!r}] = {_out_var}")
                    trace_extra = ""
                    if op == "linear":
                        weight_expr = self._infer_param_expr(
                            node_spec, node_path, "weight", env=env, scope_var=scope_var
                        )
                        trace_extra = f", weight_path={weight_expr}"
                    lines.append(
                        f"{inner_indent}self._trace_op({trace_node_path}, {op!r}, {out_name!r}, {_out_var}{trace_extra})"
                    )
                    lines.append(
                        f"{inner_indent}self._record_runtime_value(f'{{{trace_node_path}}}::{out_name}', {_out_var})"
                    )
        return lines

    def _op_uses_node_path(self, op: str, node_spec: dict[str, Any]) -> bool:
        op_module = get_op_module(op)
        if op_module is None:
            raise NotImplementedError(f"Unsupported op in codegen compiler: {op}")
        return bool(op_module.uses_node_path(self, node_spec))

    def _node_output_names(self, node_spec: dict[str, Any]) -> list[str]:
        if node_spec.get("_op") == "call":
            bind_value = node_spec.get("_bind")
            if isinstance(bind_value, str):
                return [bind_value]
            if isinstance(bind_value, list):
                return [str(v) for v in bind_value]
            return []
        out_value = node_spec.get("_bind")
        if isinstance(out_value, str):
            return [out_value]
        if isinstance(out_value, list):
            return [str(v) for v in out_value]
        return []

    def _compile_block_call(
        self, *, node_spec: dict[str, Any], env: dict[str, str], scope_var: str, indent: str
    ) -> list[str]:
        block_name = node_spec.get("_target")
        if not isinstance(block_name, str):
            raise ValueError("call must provide string _target block name")
        block_spec = self.blocks.get(block_name)
        if not isinstance(block_spec, dict):
            raise ValueError(f"Unknown block {block_name!r}")
        block_inputs = block_spec.get("inputs", {})
        if not isinstance(block_inputs, dict):
            raise ValueError("block must define mapping inputs")
        input_names = list(block_inputs.keys())
        input_py_names = self._unique_py_name_map(input_names, reserved={"self", "scope"})
        block_types = self.block_io_types.get(block_name, {})
        input_types = block_types.get("inputs", {}) if isinstance(block_types, dict) else {}

        def _path_typed_input(name: str) -> bool:
            raw_type = input_types.get(name) if isinstance(input_types, dict) else None
            return isinstance(raw_type, str) and raw_type.strip() in {"Path", "?Path"}

        raw_args = node_spec.get("_args")
        positional: list[Any]
        if raw_args is None:
            positional = []
        elif isinstance(raw_args, list):
            positional = list(raw_args)
        else:
            positional = [raw_args]
        arg_codes: list[str] = []
        for idx, src in enumerate(positional):
            if idx >= len(input_names):
                raise ValueError(f"too many positional args for call {block_name!r}")
            block_input_name = input_names[idx]
            py_input_name = input_py_names[block_input_name]
            if isinstance(src, str) and src in env:
                value_code = env[src]
            else:
                value_code = self._expr_code(src, env)
            if _path_typed_input(block_input_name):
                value_code = f"self._resolve_call_path_arg({value_code}, {{**env, **locals()}})"
            arg_codes.append(f"{py_input_name}={value_code}")
        for key, value in node_spec.items():
            if key.startswith("_") or key == "graph":
                continue
            if key not in block_inputs:
                continue
            py_input_name = input_py_names[key]
            if isinstance(value, str) and value in env:
                value_code = env[value]
            else:
                value_code = self._expr_code(value, env)
            if _path_typed_input(key):
                value_code = f"self._resolve_call_path_arg({value_code}, {{**env, **locals()}})"
            arg_codes.append(f"{py_input_name}={value_code}")

        block_outputs = block_spec.get("outputs", {})
        if not isinstance(block_outputs, dict):
            raise ValueError("block must define mapping outputs")
        output_names = list(block_outputs.keys())
        raw_bind = node_spec.get("_bind")
        binds = raw_bind if isinstance(raw_bind, list) else [raw_bind]
        if raw_bind is None or len(binds) != len(output_names):
            raise ValueError(
                f"call {block_name!r} bind arity mismatch: expected {len(output_names)}, got {len(binds)}"
            )

        tmp_vars: list[str] = []
        for block_out_name in output_names:
            var = self._fresh(self._py_name(block_out_name))
            tmp_vars.append(var)

        lines: list[str] = []
        call_args = ", ".join([*arg_codes, f"scope={scope_var}"])
        if len(tmp_vars) == 1:
            call_line = (
                f"{indent}{tmp_vars[0]} = self._block_{self._py_name(block_name)}({call_args})"
            )
        else:
            call_line = f"{indent}{', '.join(tmp_vars)} = self._block_{self._py_name(block_name)}({call_args})"

        lines.append(call_line)
        for dst_name, tmp in zip(binds, tmp_vars, strict=True):
            existing = env.get(dst_name)
            dst_var = (
                existing if isinstance(existing, str) else self._fresh(self._py_name(dst_name))
            )
            lines.append(f"{indent}{dst_var} = {tmp}")
            env[dst_name] = dst_var
        output_types = block_types.get("outputs", {}) if isinstance(block_types, dict) else {}
        exact_output_types = node_spec.get("_out_types")
        if isinstance(exact_output_types, dict):
            bound_output_types = {
                str(dst_name): exact_output_types[dst_name]
                for dst_name in binds
                if isinstance(dst_name, str) and isinstance(exact_output_types.get(dst_name), str)
            }
        elif isinstance(output_types, dict):
            bound_output_types = {
                str(dst_name): output_types[block_out_name]
                for dst_name, block_out_name in zip(binds, output_names, strict=True)
                if isinstance(dst_name, str) and isinstance(output_types.get(block_out_name), str)
            }
        else:
            bound_output_types = {}
        if bound_output_types:
            for dim_name, (penalty, expr) in self._shape_symbol_alias_candidates(
                env=env, input_types=bound_output_types
            ).items():
                current_penalty = self._active_shape_alias_penalties.get(dim_name)
                if current_penalty is None or penalty < current_penalty:
                    self._active_shape_aliases[dim_name] = expr
                    self._active_shape_alias_penalties[dim_name] = penalty
        return lines

    def _compile_op(
        self,
        *,
        op: str,
        node_spec: dict[str, Any],
        env: dict[str, str],
        node_path_var: str,
        scope_var: str,
        indent: str,
    ) -> list[str]:
        op_module = get_op_module(op)
        if op_module is None:
            raise NotImplementedError(f"Unsupported op in codegen compiler: {op}")
        prev_env = self._active_env
        prev_shape_aliases = self._active_shape_aliases
        prev_shape_alias_penalties = self._active_shape_alias_penalties
        prev_hoists = self._node_hoists
        self._active_env = env
        self._node_hoists = {}
        try:
            return op_module.compile(
                self,
                node_spec,
                env,
                node_path_var=node_path_var,
                scope_var=scope_var,
                indent=indent,
            )
        finally:
            self._active_env = prev_env
            self._active_shape_aliases = prev_shape_aliases
            self._active_shape_alias_penalties = prev_shape_alias_penalties
            self._node_hoists = prev_hoists

    def _assign_out_var(self, env: dict[str, str], out_name: str) -> str:
        existing = env.get(out_name)
        if isinstance(existing, str):
            return existing
        out_var = self._fresh(self._py_name(out_name))
        env[out_name] = out_var
        return out_var

    def _infer_param_expr(
        self,
        node_spec: dict[str, Any],
        node_path_var: str,
        param_name: str,
        *,
        env: dict[str, str] | None = None,
        scope_var: str | None = None,
    ) -> str:
        expr_env = self._active_env if env is None else env
        env_expr = "{**env, **locals()}"
        abs_path = node_spec.get("_abs_path")
        if isinstance(abs_path, (str, dict)):
            scope_expr = (
                "resolve_path_expr_to_key("
                f"{repr(abs_path)}, self._path_template_env({env_expr}), op_name='_abs_path')"
            )
        else:
            scope_expr = f"self._scope_of({node_path_var})"
        param_base = node_spec.get("param_base")
        if isinstance(param_base, str):
            if param_base in expr_env:
                base_expr = expr_env[param_base]
                return (
                    "self._pick_param_from_single("
                    f"{scope_expr}, self._join_scope({base_expr}, {param_name!r}), {env_expr})"
                )
            if isinstance(node_spec.get(param_base), str):
                base_expr = repr(node_spec[param_base])
                return (
                    "self._pick_param_from_single("
                    f"{scope_expr}, self._join_scope({base_expr}, {param_name!r}), {env_expr})"
                )
            base_expr = repr(param_base)
            return (
                "self._pick_param_from_single("
                f"{scope_expr}, self._join_scope({base_expr}, {param_name!r}), {env_expr})"
            )
        explicit_params = node_spec.get("_params")
        if isinstance(node_spec.get(param_name), str):
            candidate = node_spec[param_name]
            if candidate != param_name and (candidate.startswith("@") or "." in candidate):
                return f"self._pick_param_from_single({scope_expr}, {candidate!r}, {env_expr})"
        # Next precedence level: lowered path bindings from _params.
        if isinstance(explicit_params, dict):
            explicit = explicit_params.get(param_name)
            if isinstance(explicit, (str, dict)):
                expr = f"self._pick_param_from_single({scope_expr}, {explicit!r}, {env_expr})"
                return expr
            if isinstance(explicit, list) and all(
                isinstance(item, (str, dict)) for item in explicit
            ):
                expr = f"self._pick_param_path({scope_expr}, {explicit!r}, {env_expr})"
                return expr
        fallback_expr = f"self._pick_param_from_single({scope_expr}, {param_name!r}, {env_expr})"
        return fallback_expr

    def _hoist_expr(self, *, kind: str, key: str, expr: str, lines: list[str], indent: str) -> str:
        cache_key = (kind, key)
        existing = self._node_hoists.get(cache_key)
        if isinstance(existing, str):
            return existing
        var = self._fresh(kind)
        lines.append(f"{indent}{var} = {expr}")
        self._node_hoists[cache_key] = var
        return var

    def _hoisted_param_path(
        self,
        *,
        node_spec: dict[str, Any],
        node_path_var: str,
        param_name: str,
        lines: list[str],
        indent: str,
    ) -> str:
        expr = self._infer_param_expr(node_spec, node_path_var, param_name)
        return self._hoist_expr(
            kind="param_path",
            key=f"{param_name}:{expr}",
            expr=expr,
            lines=lines,
            indent=indent,
        )

    def _hoisted_param(
        self,
        *,
        node_spec: dict[str, Any],
        node_path_var: str,
        param_name: str,
        lines: list[str],
        indent: str,
    ) -> str:
        path_var = self._hoisted_param_path(
            node_spec=node_spec,
            node_path_var=node_path_var,
            param_name=param_name,
            lines=lines,
            indent=indent,
        )
        return self._hoist_expr(
            kind="param_tensor",
            key=f"required:{path_var}",
            expr=f"emitter._param({path_var})",
            lines=lines,
            indent=indent,
        )

    def _hoisted_optional_param(
        self,
        *,
        node_spec: dict[str, Any],
        node_path_var: str,
        param_name: str,
        lines: list[str],
        indent: str,
    ) -> str:
        path_var = self._hoisted_param_path(
            node_spec=node_spec,
            node_path_var=node_path_var,
            param_name=param_name,
            lines=lines,
            indent=indent,
        )
        return self._hoist_expr(
            kind="param_tensor_opt",
            key=f"optional:{path_var}",
            expr=f"self._state.get({path_var})",
            lines=lines,
            indent=indent,
        )

    def _read_env_var(self, env: dict[str, str], name: str) -> str:
        if name not in env:
            raise ValueError(f"Unknown input var {name!r}")
        return env[name]

    def _scalar_token_code(self, token: str) -> str | None:
        value = token.strip()
        lower = value.lower()
        if lower == "null":
            return "None"
        if lower == "true":
            return "True"
        if lower == "false":
            return "False"
        if value and (value.isdigit() or (value[0] in {"+", "-"} and value[1:].isdigit())):
            return repr(int(value))
        try:
            parsed = float(value)
        except ValueError:
            return None
        return repr(parsed)

    def _expr_code(self, expr: Any, env: dict[str, str]) -> str:
        if expr is None:
            return "None"
        if isinstance(expr, (int, float, bool)):
            if isinstance(expr, float):
                if math.isinf(expr):
                    return "float('inf')" if expr > 0 else "float('-inf')"
                if math.isnan(expr):
                    return "float('nan')"
            return repr(expr)
        if isinstance(expr, list):
            items = ", ".join(self._expr_code(item, env) for item in expr)
            return f"[{items}]"
        if isinstance(expr, tuple):
            items = ", ".join(self._expr_code(item, env) for item in expr)
            if len(expr) == 1:
                return f"({items},)"
            return f"({items})"
        if isinstance(expr, dict):
            kind = expr.get("_expr")
            if kind == "name":
                ident = expr.get("id")
                if ident is None:
                    ident = expr.get("value")
                if isinstance(ident, str):
                    if ident in env:
                        return env[ident]
                    if ident in self._active_shape_aliases:
                        return self._active_shape_aliases[ident]
                    if ident in self.symbols:
                        return self._expr_code(self.symbols[ident], env)
                    raise ValueError(f"Unknown symbol in expression: {ident}")
            if kind == "tuple":
                items_raw = expr.get("items")
                if isinstance(items_raw, list):
                    items = ", ".join(self._expr_code(item, env) for item in items_raw)
                    if len(items_raw) == 1:
                        return f"({items},)"
                    return f"({items})"
            if kind == "if":
                cond_code = self._expr_code(expr.get("cond"), env)
                then_code = self._expr_code(expr.get("then"), env)
                else_code = self._expr_code(expr.get("else"), env)
                return f"({then_code} if {cond_code} else {else_code})"
            if kind == "binary":
                op = expr.get("op")
                left_code = self._expr_code(expr.get("left"), env)
                right_code = self._expr_code(expr.get("right"), env)
                if isinstance(op, str) and op in {
                    "+",
                    "-",
                    "*",
                    "/",
                    "%",
                    "==",
                    "!=",
                    "<",
                    "<=",
                    ">",
                    ">=",
                    "and",
                    "or",
                }:
                    return f"({left_code} {op} {right_code})"
            if kind == "string":
                value = expr.get("value")
                if isinstance(value, str):
                    return repr(value)
                raise ValueError(f"Invalid string expression payload: {expr!r}")
            if kind == "path":
                absolute = expr.get("absolute")
                parts = expr.get("parts")
                if (
                    isinstance(absolute, bool)
                    and isinstance(parts, list)
                    and all(isinstance(part, str) for part in parts)
                ):
                    return repr({"_expr": "path", "absolute": absolute, "parts": list(parts)})
                raise ValueError(f"Invalid path expression payload: {expr!r}")
            if kind == "call":
                callee = expr.get("callee")
                args_raw = expr.get("args", [])
                kwargs_raw = expr.get("kwargs", {})
                if (
                    not isinstance(callee, str)
                    or not isinstance(args_raw, list)
                    or not isinstance(kwargs_raw, dict)
                ):
                    raise ValueError(f"Invalid call expression payload: {expr!r}")
                args_code = ", ".join(self._expr_code(item, env) for item in args_raw)
                kwargs_parts = [
                    f"{key!r}: {self._expr_code(value, env)}" for key, value in kwargs_raw.items()
                ]
                kwargs_code = ", ".join(kwargs_parts)
                return f"self._eval_expr_call({callee!r}, [{args_code}], {{{kwargs_code}}}, env, self._symbols)"
            return repr(expr)
        if isinstance(expr, str):
            token = expr.strip()
            if token in env:
                return env[token]
            if token in self._active_shape_aliases:
                return self._active_shape_aliases[token]
            if token in self.symbols:
                return self._expr_code(self.symbols[token], env)
            scalar_code = self._scalar_token_code(token)
            if scalar_code is not None:
                return scalar_code
            return repr(token)
        return repr(expr)

    def _py_name(self, value: str) -> str:
        out_chars: list[str] = []
        for ch in value:
            out_chars.append(ch if (ch.isalnum() or ch == "_") else "_")
        name = "".join(out_chars)
        if not name:
            name = "v"
        if name.startswith("__"):
            name = f"v{name}"
        if name[0].isdigit():
            name = f"v_{name}"
        return name

    def _unique_py_name_map(
        self, values: Iterable[str], *, reserved: set[str] | None = None
    ) -> dict[str, str]:
        used = set(reserved or ())
        mapping: dict[str, str] = {}
        for value in values:
            base = self._py_name(value)
            candidate = base
            suffix = 1
            while candidate in used:
                suffix += 1
                candidate = f"{base}_{suffix}"
            used.add(candidate)
            mapping[value] = candidate
        return mapping

    def _fresh(self, base: str) -> str:
        self._counter += 1
        return f"{base}_{self._counter}"


def _validate_spec_ops(spec: dict[str, Any], op_map: dict[str, Any]) -> None:
    ops = op_map.get("ops")
    if not isinstance(ops, dict):
        raise ValueError("op map must contain mapping key 'ops'")

    known_control_ops = {"call"}

    def _walk_graph(graph: list[Any]) -> None:
        for item in graph:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f"Invalid graph item: {item!r}")
            _, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                raise ValueError(f"Invalid node spec: {node_spec!r}")

            op = node_spec.get("_op")
            if isinstance(op, str):
                is_dynamic_activation = op.startswith("activations_")
                has_runtime_op = get_op_module(op) is not None
                if (
                    op not in known_control_ops
                    and not has_runtime_op
                    and not is_dynamic_activation
                    and op not in ops
                ):
                    raise ValueError(f"Unsupported op in spec: {op!r}")

            if "graph" in node_spec:
                nested = node_spec["graph"]
                if not isinstance(nested, list):
                    raise ValueError("node 'graph' must be a list")
                _walk_graph(nested)
            then_graph = node_spec.get("_then")
            if then_graph is not None:
                if not isinstance(then_graph, list):
                    raise ValueError("node '_then' must be a list")
                _walk_graph(then_graph)
            else_graph = node_spec.get("_else")
            if else_graph is not None:
                if not isinstance(else_graph, list):
                    raise ValueError("node '_else' must be a list")
                _walk_graph(else_graph)

    model = spec.get("model")
    if not isinstance(model, dict):
        raise ValueError("spec.model must be a mapping")

    graph = model.get("graph")
    if not isinstance(graph, list):
        raise ValueError("model.graph must be a list")
    _walk_graph(graph)

    blocks = model.get("blocks", {})
    if not isinstance(blocks, dict):
        raise ValueError("model.blocks must be a mapping when present")
    for block in blocks.values():
        if not isinstance(block, dict):
            raise ValueError("block spec must be mapping")
        block_graph = block.get("graph")
        if not isinstance(block_graph, list):
            raise ValueError("block.graph must be list")
        _walk_graph(block_graph)
