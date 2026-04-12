from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from ..core import StateDictLike
from .axon.type_system import TypeTensor, parse_type_expr
from .mxfp4 import materialize_mxfp4_aliases
from .ops import get_op_module
from .spec_normalize import normalize_synapse_spec_expressions

SymbolScalar = int | float | bool | str
SymbolValue = SymbolScalar | list[SymbolScalar] | tuple[SymbolScalar, ...]


def _is_int_token(token: str) -> bool:
    return bool(token) and (token.isdigit() or (token[0] in {"+", "-"} and token[1:].isdigit()))


def _is_symbol_value(value: Any) -> bool:
    if isinstance(value, (int, float, bool, str)):
        return True
    if isinstance(value, list | tuple):
        return all(isinstance(item, (int, float, bool, str)) for item in value)
    return False


class SynapseProgramModel(nn.Module):
    """Generic runtime for Synapse graph specs backed by checkpoint tensors."""

    SPEC: dict[str, Any] = {}
    OP_MAP: dict[str, Any] = {}

    def __init__(
        self,
        spec: dict[str, Any] | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
        runtime_state_dict: StateDictLike | None = None,
    ) -> None:
        super().__init__()
        self.spec: dict[str, Any] = self._resolve_spec(spec)
        self._state: Mapping[str, torch.Tensor] = {}
        self._runtime_state_dict = runtime_state_dict
        self._param_roots_stack: list[list[str]] = []
        if state_dict is not None:
            self.load_state_dict_tensors(state_dict)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        spec: dict[str, Any] | None = None,
        runtime_state_dict: StateDictLike | None = None,
    ) -> "SynapseProgramModel":
        return cls(spec=spec, state_dict=state_dict, runtime_state_dict=runtime_state_dict)

    @classmethod
    def from_spec(
        cls,
        spec: dict[str, Any],
        *,
        state_dict: dict[str, torch.Tensor] | None = None,
        runtime_state_dict: StateDictLike | None = None,
    ) -> "SynapseProgramModel":
        return cls(spec=spec, state_dict=state_dict, runtime_state_dict=runtime_state_dict)

    @classmethod
    def from_yaml(
        cls,
        spec_path: str | Path,
        *,
        state_dict: dict[str, torch.Tensor] | None = None,
        runtime_state_dict: StateDictLike | None = None,
    ) -> "SynapseProgramModel":
        loaded = OmegaConf.load(Path(spec_path))
        data = OmegaConf.to_container(loaded, resolve=True)
        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML mapping at {spec_path}, got {type(data).__name__}")
        return cls(
            spec={str(key): value for key, value in data.items()},
            state_dict=state_dict,
            runtime_state_dict=runtime_state_dict,
        )

    def load_state_dict_tensors(self, state_dict: dict[str, torch.Tensor]) -> None:
        loaded = dict(state_dict)
        materialize_mxfp4_aliases(loaded, drop_packed=True)
        self._state = loaded

    def _param(self, path: str) -> torch.Tensor:
        resolved = path[2:] if isinstance(path, str) and path.startswith("@@") else path
        return self._state[resolved]

    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:
        spec = self.spec
        model = spec.get("model", {})
        symbols_raw = model.get("symbols", {})
        symbols: dict[str, SymbolValue] = {
            k: v for k, v in symbols_raw.items() if _is_symbol_value(v)
        }
        blocks = model.get("blocks", {})
        input_specs = model.get("inputs", {})
        if not isinstance(input_specs, dict):
            raise ValueError("model.inputs must be a mapping when present")
        env = self._prepare_env(input_ids=input_ids, inputs=inputs, input_specs=input_specs)
        self._validate_input_shapes(env, input_specs, symbols)
        self._run_graph(model.get("graph", []), env, scope="", symbols=symbols, blocks=blocks)

        outputs = model.get("outputs", {})
        if not isinstance(outputs, dict):
            raise ValueError("model.outputs must be a mapping")
        resolved_outputs: dict[str, Any] = {}
        for key, ref in outputs.items():
            resolved_outputs[key] = self._resolve_output_ref(ref, env)

        if "logits" in resolved_outputs and len(resolved_outputs) == 1:
            return resolved_outputs["logits"]
        return resolved_outputs

    def _resolve_spec(self, spec: dict[str, Any] | None) -> dict[str, Any]:
        resolved = self.SPEC if spec is None else spec
        if isinstance(resolved, dict):
            resolved = normalize_synapse_spec_expressions(resolved)
        if not isinstance(resolved, dict):
            raise ValueError("Synapse spec must be a mapping")
        if resolved.get("synapse") != 1:
            raise ValueError("Only synapse: 1 specs are supported")
        model = resolved.get("model")
        if not isinstance(model, dict):
            raise ValueError("spec.model must be a mapping")
        graph = model.get("graph")
        if not isinstance(graph, list):
            raise ValueError("spec.model.graph must be a list")
        return resolved

    def _require_name(self, value: Any, *, field: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string")
        return value

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        eos_token_id: int,
        max_len: int,
        attention_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be rank-2 [batch, seq]")
        if max_len <= 0:
            raise ValueError("max_len must be > 0")
        if attention_mask is not None and attn_mask is not None:
            raise ValueError("pass at most one of attention_mask or attn_mask")
        mask = attention_mask if attention_mask is not None else attn_mask
        if mask is not None:
            if mask.ndim != 2:
                raise ValueError("attention_mask must be rank-2 [batch, seq]")
            if mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have same shape as input_ids")
        if input_ids.size(1) >= max_len:
            return input_ids[:, :max_len]

        batch, start_len = input_ids.shape
        generated = input_ids.new_empty((batch, max_len))
        generated[:, :start_len] = input_ids
        generated_mask = None
        if mask is not None:
            generated_mask = mask.new_zeros((batch, max_len))
            generated_mask[:, :start_len] = mask
        model = self.spec.get("model", {})
        input_specs = model.get("inputs", {})
        output_specs = model.get("outputs", {})
        if not isinstance(input_specs, dict):
            input_specs = {}
        if not isinstance(output_specs, dict):
            output_specs = {}

        state_input_name: str | None = None
        for candidate in (
            "past_key_values",
            "past_kv",
            "past",
            "cache_params",
            "cache_state",
            "state",
        ):
            if candidate in input_specs:
                state_input_name = candidate
                break

        use_cache_name = "use_cache" if "use_cache" in input_specs else None
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
            if name in output_specs
        ]

        cache_state = None
        finished = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
        cur_len = start_len
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                while cur_len < max_len and not torch.all(finished):
                    step_input = (
                        generated[:, :cur_len]
                        if cache_state is None
                        else generated[:, cur_len - 1 : cur_len]
                    )
                    call_kwargs: dict[str, Any] = {}
                    if generated_mask is not None:
                        if "attention_mask" in input_specs:
                            call_kwargs["attention_mask"] = generated_mask[:, :cur_len]
                        if "attn_mask" in input_specs:
                            call_kwargs["attn_mask"] = generated_mask[:, :cur_len]
                    if state_input_name is not None:
                        call_kwargs[state_input_name] = cache_state
                    if use_cache_name is not None:
                        call_kwargs[use_cache_name] = True
                    model_out = self.forward(step_input, **call_kwargs)
                    if isinstance(model_out, dict):
                        if "logits" in model_out:
                            logits = model_out["logits"]
                        elif len(model_out) == 1:
                            logits = next(iter(model_out.values()))
                        else:
                            raise KeyError(
                                "Expected 'logits' in model outputs or a single unnamed output"
                            )
                        for out_name in state_output_names:
                            if out_name in model_out:
                                cache_state = model_out[out_name]
                                break
                    else:
                        logits = model_out
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                    next_token = torch.where(
                        finished,
                        torch.full_like(next_token, eos_token_id),
                        next_token,
                    )
                    generated[:, cur_len] = next_token
                    finished = torch.logical_or(finished, next_token == eos_token_id)
                    if generated_mask is not None:
                        generated_mask[:, cur_len] = 1
                    cur_len += 1
        finally:
            if was_training:
                self.train()
        return generated[:, :cur_len]

    def _prepare_env(
        self,
        *,
        input_ids: torch.Tensor | None,
        inputs: dict[str, Any],
        input_specs: dict[str, Any],
    ) -> dict[str, Any]:
        env: dict[str, Any]
        if input_ids is not None:
            env = {"input_ids": input_ids, **inputs}
        else:
            env = dict(inputs)
        for input_name, input_spec in input_specs.items():
            optional = isinstance(input_spec, dict) and bool(input_spec.get("optional", False))
            if input_name in env:
                continue
            if optional:
                env[input_name] = None
            else:
                raise ValueError(f"Missing required input: {input_name}")
        return env

    def _for_values(self, *, from_value: int, to_value: int, step_value: int) -> range:
        if not isinstance(from_value, int):
            raise ValueError(f"for _from must resolve to int, got {from_value!r}")
        if not isinstance(to_value, int):
            raise ValueError(f"for _to must resolve to int, got {to_value!r}")
        if not isinstance(step_value, int):
            raise ValueError(f"for _step must resolve to int, got {step_value!r}")
        if step_value == 0:
            raise ValueError("for _step must be non-zero")
        return range(from_value, to_value, step_value)

    def _run_graph(
        self,
        graph: list[Any],
        env: dict[str, Any],
        *,
        scope: str,
        symbols: dict[str, SymbolValue],
        blocks: dict[str, Any],
    ) -> None:
        for item in graph:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f"Invalid graph item: {item!r}")
            node_name, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                raise ValueError(f"Node spec for {node_name!r} must be mapping")

            op = node_spec.get("_op")
            if op == "for":
                scope_name = node_spec.get("_scope")
                if not isinstance(scope_name, str):
                    raise ValueError("for requires string '_scope'")
                to_value = self._eval_expr(node_spec.get("_to"), env, symbols)
                from_value = self._eval_expr(node_spec.get("_from", 0), env, symbols)
                step_value = self._eval_expr(node_spec.get("_step", 1), env, symbols)
                var_name = node_spec.get("_var")
                if not isinstance(var_name, str):
                    raise ValueError("for requires string '_var'")
                body = node_spec.get("_body")
                if not isinstance(body, list):
                    raise ValueError("for requires list '_body'")
                for iter_value in self._for_values(
                    from_value=from_value, to_value=to_value, step_value=step_value
                ):
                    env[var_name] = iter_value
                    iter_segment = "" if not scope_name else f"{scope_name}.{iter_value}"
                    for_scope = self._join(scope, iter_segment)
                    self._run_graph(body, env, scope=for_scope, symbols=symbols, blocks=blocks)
                env.pop(var_name, None)
                continue

            if op == "call":
                self._run_block_call(node_spec, env, scope=scope, symbols=symbols, blocks=blocks)
                self._record_runtime_outputs(
                    node_spec=node_spec,
                    env=env,
                    node_path=self._join(scope, node_name),
                )
                continue

            if "graph" in node_spec and op is None:
                nested = node_spec.get("graph")
                if not isinstance(nested, list):
                    raise ValueError("graph node must contain list 'graph'")
                nested_scope = self._join(scope, node_name)
                self._run_graph(nested, env, scope=nested_scope, symbols=symbols, blocks=blocks)
                continue

            if not isinstance(op, str):
                raise ValueError(f"Node {node_name!r} missing string op")

            node_path = self._join(scope, node_name)
            exec_node_spec = node_spec
            param_base = node_spec.get("param_base")
            if isinstance(param_base, str) and isinstance(env.get(param_base), str):
                exec_node_spec = dict(node_spec)
                exec_node_spec[param_base] = env[param_base]
            param_root_value = node_spec.get("_param_root")
            if isinstance(param_root_value, dict):
                if exec_node_spec is node_spec:
                    exec_node_spec = dict(node_spec)
                resolved_param_root = self._eval_expr(param_root_value, env, symbols)
                exec_node_spec["_param_root"] = (
                    resolved_param_root if isinstance(resolved_param_root, str) else ""
                )
            self._execute_op(
                op, exec_node_spec, env, node_path=node_path, scope=scope, symbols=symbols
            )
            self._record_runtime_outputs(node_spec=node_spec, env=env, node_path=node_path)

    def _run_block_call(
        self,
        node_spec: dict[str, Any],
        env: dict[str, Any],
        *,
        scope: str,
        symbols: dict[str, SymbolValue],
        blocks: dict[str, Any],
    ) -> None:
        block_name = node_spec.get("_target")
        if not isinstance(block_name, str):
            raise ValueError("call requires string _target block name")
        block_spec = blocks.get(block_name)
        if not isinstance(block_spec, dict):
            raise ValueError(f"Unknown block {block_name!r}")

        block_env = dict(env)
        block_inputs = block_spec.get("inputs", {})
        if not isinstance(block_inputs, dict):
            raise ValueError("block spec must include mapping inputs")
        input_names = list(block_inputs.keys())
        raw_args = node_spec.get("_args")
        positional: list[Any]
        if raw_args is None:
            positional = []
        elif isinstance(raw_args, list):
            positional = list(raw_args)
        else:
            positional = [raw_args]
        for idx, src_name in enumerate(positional):
            if idx >= len(input_names):
                raise ValueError(f"too many positional args for call {block_name!r}")
            block_input_name = input_names[idx]
            if isinstance(src_name, str) and src_name in env:
                block_env[block_input_name] = env[src_name]
            else:
                block_env[block_input_name] = self._eval_expr(src_name, env, symbols)

        for key, value in node_spec.items():
            if key.startswith("_") or key == "graph":
                continue
            if key not in block_inputs:
                continue
            if isinstance(value, str) and value in env:
                block_env[key] = env[value]
            else:
                block_env[key] = self._eval_expr(value, env, symbols)

        block_graph = block_spec.get("graph")
        if not isinstance(block_graph, list):
            raise ValueError("block spec must include list graph")
        call_scope = scope
        raw_scope = node_spec.get("_scope")
        if isinstance(raw_scope, str) and raw_scope:
            call_scope = self._join(scope, raw_scope)
        raw_param_root = node_spec.get("_param_root")
        pushed_roots: list[str] | None = None
        if isinstance(raw_param_root, dict):
            resolved_root = self._eval_expr(raw_param_root, env, symbols)
            if isinstance(resolved_root, str):
                pushed_roots = [resolved_root]
                self._param_roots_stack.append(pushed_roots)
        elif isinstance(raw_param_root, str):
            pushed_roots = [raw_param_root]
            self._param_roots_stack.append(pushed_roots)
        elif (
            isinstance(raw_param_root, list)
            and bool(raw_param_root)
            and all(isinstance(item, str) for item in raw_param_root)
        ):
            pushed_roots = list(raw_param_root)
            self._param_roots_stack.append(pushed_roots)
        self._bind_shape_symbols_from_types(
            env=block_env,
            input_types=self._block_input_types(block_name),
            symbols=symbols,
        )
        self._validate_input_shapes(block_env, block_inputs, symbols)
        try:
            self._run_graph(
                block_graph, block_env, scope=call_scope, symbols=symbols, blocks=blocks
            )
        finally:
            if pushed_roots is not None:
                self._param_roots_stack.pop()

        block_outputs = block_spec.get("outputs", {})
        if not isinstance(block_outputs, dict):
            raise ValueError("block spec must include mapping outputs")
        output_names = list(block_outputs.keys())
        raw_bind = node_spec.get("_bind")
        if raw_bind is None:
            raise ValueError("call requires _bind")
        binds = raw_bind if isinstance(raw_bind, list) else [raw_bind]
        if len(binds) != len(output_names):
            raise ValueError(
                f"call {block_name!r} bind arity mismatch: expected {len(output_names)}, got {len(binds)}"
            )
        for out_name, dst_name in zip(output_names, binds, strict=True):
            env[str(dst_name)] = block_env.get(out_name)

    def _record_runtime_outputs(
        self,
        *,
        node_spec: dict[str, Any],
        env: dict[str, Any],
        node_path: str,
    ) -> None:
        runtime_state_dict = self._runtime_state_dict
        if runtime_state_dict is None:
            return
        for bind_name in self._node_output_names(node_spec):
            if bind_name not in env:
                continue
            value = env[bind_name]
            self._record_runtime_value(
                runtime_state_dict=runtime_state_dict,
                key=f"{node_path}::{bind_name}",
                value=value,
            )

    def _record_runtime_value(
        self,
        *,
        runtime_state_dict: StateDictLike,
        key: str,
        value: Any,
    ) -> None:
        if torch.is_tensor(value):
            runtime_state_dict[key] = value.detach().clone()
            return
        if isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                self._record_runtime_value(
                    runtime_state_dict=runtime_state_dict,
                    key=f"{key}[{idx}]",
                    value=item,
                )
            return
        if isinstance(value, dict):
            for item_key, item_value in value.items():
                self._record_runtime_value(
                    runtime_state_dict=runtime_state_dict,
                    key=f"{key}.{item_key}",
                    value=item_value,
                )

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

    def _execute_op(
        self,
        op: str,
        node_spec: dict[str, Any],
        env: dict[str, Any],
        *,
        node_path: str,
        scope: str,
        symbols: dict[str, SymbolValue],
    ) -> None:
        op_module = get_op_module(op)
        if op_module is None:
            raise NotImplementedError(f"Unsupported op: {op}")
        op_module.interpret(
            self,
            node_spec,
            env,
            node_path=node_path,
            scope=scope,
            symbols=symbols,
        )

    def _infer_param_path(
        self, node_spec: dict[str, Any], *, node_path: str, param_name: str
    ) -> str:
        def _effective_scope() -> str:
            abs_path = node_spec.get("_abs_path")
            if isinstance(abs_path, str) and abs_path:
                return abs_path
            ambient_scope = self._scope_of(node_path)
            explicit_scope = node_spec.get("_scope")
            if isinstance(explicit_scope, str) and explicit_scope:
                return self._join(ambient_scope, explicit_scope)
            return ambient_scope

        def _join_root(root: str, scoped: str) -> str:
            if not root:
                return scoped
            if not scoped:
                return root
            return self._join(root, scoped)

        def _current_roots() -> list[str]:
            roots = list(self._param_roots_stack[-1]) if self._param_roots_stack else [""]
            direct_root = node_spec.get("_param_root")
            if isinstance(direct_root, str):
                composed: list[str] = []
                for root in roots:
                    composed.append(_join_root(root, direct_root))
                roots = composed
            return roots

        def _explicit_candidates(raw_path: str) -> list[str]:
            if raw_path.startswith("@@"):
                absolute = raw_path[2:]
                if not absolute:
                    return []
                return [absolute]
            if raw_path.startswith("@"):
                raw_path = raw_path[1:]
                if not raw_path:
                    return []
            roots = _current_roots()
            scope_prefix = _effective_scope()
            scoped = self._join(scope_prefix, raw_path)
            scoped = scoped if scoped else raw_path
            scoped_candidates: list[str] = []
            for candidate in [_join_root(root, scoped) for root in roots]:
                if candidate not in scoped_candidates:
                    scoped_candidates.append(candidate)
            return scoped_candidates

        def _pick_explicit_candidate(raw: Any) -> str | None:
            if isinstance(raw, str):
                candidates = _explicit_candidates(raw)
                for candidate in candidates:
                    if candidate in self._state:
                        return candidate
                return candidates[0]
            if isinstance(raw, list | tuple):
                list_candidates: list[str] = []
                for item in raw:
                    if not isinstance(item, str):
                        continue
                    list_candidates.extend(_explicit_candidates(item))
                if not list_candidates:
                    return None
                for candidate in list_candidates:
                    if candidate in self._state:
                        return candidate
                return list_candidates[0]
            return None

        def _pick_scoped_candidate(scoped: str) -> str:
            candidates = [_join_root(root, scoped) for root in _current_roots()]
            for candidate in candidates:
                if candidate in self._state:
                    return candidate
            return candidates[0]

        param_base = node_spec.get("param_base")
        if isinstance(param_base, str):
            base_resolved = node_spec.get(param_base)
            base = base_resolved if isinstance(base_resolved, str) else param_base
            scoped_base = self._join(_effective_scope(), base)
            scoped_param = f"{scoped_base}.{param_name}" if scoped_base else param_name
            return _pick_scoped_candidate(scoped_param)
        # Explicit per-node path override wins over _params.
        explicit_params = node_spec.get("_params")
        if param_name in node_spec and isinstance(node_spec[param_name], str):
            candidate = node_spec[param_name]
            if candidate != param_name:
                explicit = _pick_explicit_candidate(candidate)
                if isinstance(explicit, str):
                    return explicit
        # Next precedence level: lowered path bindings.
        if isinstance(explicit_params, dict):
            explicit = _pick_explicit_candidate(explicit_params.get(param_name))
            if isinstance(explicit, str):
                return explicit
        scope_fallback = self._join(_effective_scope(), param_name)
        fallback = scope_fallback if scope_fallback else param_name
        return _pick_scoped_candidate(fallback)

    def _resolve_output_ref(self, ref: Any, env: dict[str, Any]) -> Any:
        if isinstance(ref, str):
            return env[ref]
        if isinstance(ref, dict):
            from_ref = ref.get("from")
            if isinstance(from_ref, str):
                return env[from_ref]
        raise ValueError(f"Unsupported output ref: {ref!r}")

    def _validate_input_shapes(
        self,
        env: dict[str, Any],
        input_specs: dict[str, Any],
        symbols: dict[str, SymbolValue],
    ) -> None:
        dim_bindings: dict[str, int] = {}
        for input_name, input_spec in input_specs.items():
            if not isinstance(input_spec, dict):
                continue
            shape_spec = input_spec.get("shape")
            if not isinstance(shape_spec, list):
                continue
            value = env.get(input_name)
            if value is None:
                continue
            if not torch.is_tensor(value):
                raise ValueError(f"Input {input_name!r} must be a tensor for declared shape checks")
            expected_rank = len(shape_spec)
            if value.ndim != expected_rank:
                raise ValueError(
                    f"Input {input_name!r} rank mismatch: expected {expected_rank}, got {value.ndim}"
                )
            for axis, dim_token in enumerate(shape_spec):
                actual = int(value.shape[axis])
                if isinstance(dim_token, bool):
                    raise ValueError(
                        f"Input {input_name!r} has invalid boolean dim token at axis {axis}"
                    )
                if isinstance(dim_token, int):
                    if actual != dim_token:
                        raise ValueError(
                            f"Input {input_name!r} shape mismatch at axis {axis}: "
                            f"expected {dim_token}, got {actual}"
                        )
                    continue
                if not isinstance(dim_token, str):
                    continue
                token = dim_token.strip()
                if not token:
                    continue
                if _is_int_token(token):
                    expected = int(token)
                    if actual != expected:
                        raise ValueError(
                            f"Input {input_name!r} shape mismatch at axis {axis}: "
                            f"expected {expected}, got {actual}"
                        )
                    continue
                symbol_value = symbols.get(token)
                if isinstance(symbol_value, bool):
                    raise ValueError(
                        f"Input {input_name!r} has invalid boolean symbol {token!r} at axis {axis}"
                    )
                if isinstance(symbol_value, int):
                    if actual != symbol_value:
                        raise ValueError(
                            f"Input {input_name!r} shape mismatch at axis {axis}: "
                            f"expected symbol {token}={symbol_value}, got {actual}"
                        )
                    continue
                bound = dim_bindings.get(token)
                if bound is None:
                    dim_bindings[token] = actual
                    symbols[token] = actual
                elif actual != bound:
                    raise ValueError(
                        f"Input {input_name!r} shape mismatch at axis {axis}: "
                        f"symbol {token} was previously bound to {bound}, got {actual}"
                    )

    def _block_input_types(self, block_name: str) -> dict[str, str]:
        model = self.spec.get("model", {})
        if not isinstance(model, dict):
            return {}
        types = model.get("types")
        if not isinstance(types, dict):
            return {}
        block_io = types.get("block_io")
        if not isinstance(block_io, dict):
            return {}
        block_type = block_io.get(block_name)
        if not isinstance(block_type, dict):
            return {}
        inputs = block_type.get("inputs")
        if not isinstance(inputs, dict):
            return {}
        out: dict[str, str] = {}
        for key, value in inputs.items():
            if isinstance(key, str) and isinstance(value, str):
                out[key] = value
        return out

    def _bind_shape_symbols_from_types(
        self,
        *,
        env: dict[str, Any],
        input_types: dict[str, str],
        symbols: dict[str, SymbolValue],
    ) -> None:
        for input_name, type_expr in input_types.items():
            value = env.get(input_name)
            if not torch.is_tensor(value):
                continue
            try:
                parsed = parse_type_expr(type_expr)
            except Exception:
                continue
            if not isinstance(parsed, TypeTensor):
                continue
            for axis, dim in enumerate(parsed.dims):
                if not isinstance(dim, str):
                    continue
                if axis >= value.ndim:
                    break
                actual = int(value.shape[axis])
                current = symbols.get(dim)
                if isinstance(current, bool):
                    raise ValueError(f"Invalid boolean symbol value for {dim!r}")
                if isinstance(current, int):
                    continue
                symbols[dim] = actual

    def _read_tensor_input(self, ref: Any, env: dict[str, Any]) -> torch.Tensor:
        if not isinstance(ref, str):
            raise ValueError(f"Expected string tensor reference, got {ref!r}")
        value = env.get(ref)
        if not torch.is_tensor(value):
            raise ValueError(f"Input reference {ref!r} does not resolve to tensor")
        return value

    def _eval_expr(self, expr: Any, env: dict[str, Any], symbols: dict[str, SymbolValue]) -> Any:
        if expr is None:
            return None
        if isinstance(expr, (int, float, bool)):
            return expr
        if isinstance(expr, list):
            return [self._eval_expr(item, env, symbols) for item in expr]
        if isinstance(expr, tuple):
            return tuple(self._eval_expr(item, env, symbols) for item in expr)
        if isinstance(expr, dict):
            kind = expr.get("_expr")
            if kind == "name":
                ident = expr.get("id")
                if not isinstance(ident, str) or not ident:
                    raise ValueError(f"Invalid name expression payload: {expr!r}")
                if ident in env:
                    return env[ident]
                if ident in symbols:
                    return symbols[ident]
                raise ValueError(f"Unknown symbol in expression: {ident}")
            if kind == "tuple":
                items = expr.get("items")
                if not isinstance(items, list):
                    raise ValueError(f"Invalid tuple expression payload: {expr!r}")
                return tuple(self._eval_expr(item, env, symbols) for item in items)
            if kind == "if":
                cond = bool(self._eval_expr(expr.get("cond"), env, symbols))
                branch = expr.get("then") if cond else expr.get("else")
                return self._eval_expr(branch, env, symbols)
            if kind == "binary":
                op = expr.get("op")
                left = self._eval_expr(expr.get("left"), env, symbols)
                right = self._eval_expr(expr.get("right"), env, symbols)
                if op == "+":
                    return left + right
                if op == "-":
                    return left - right
                if op == "*":
                    return left * right
                if op == "/":
                    return left / right
                if op == "%":
                    return left % right
                if op == "==":
                    return left == right
                if op == "!=":
                    return left != right
                if op == "<":
                    return left < right
                if op == "<=":
                    return left <= right
                if op == ">":
                    return left > right
                if op == ">=":
                    return left >= right
                if op == "and":
                    return bool(left) and bool(right)
                if op == "or":
                    return bool(left) or bool(right)
                raise ValueError(f"Unsupported binary operator in expression: {op!r}")
            if kind == "string":
                value = expr.get("value")
                if not isinstance(value, str):
                    raise ValueError(f"Invalid string expression payload: {expr!r}")
                return value
            if kind == "call":
                callee = expr.get("callee")
                args_raw = expr.get("args", [])
                kwargs_raw = expr.get("kwargs", {})
                if not isinstance(callee, str) or not callee:
                    raise ValueError(f"Invalid call expression payload: {expr!r}")
                if not isinstance(args_raw, list):
                    raise ValueError(f"Invalid call expression args payload: {expr!r}")
                if not isinstance(kwargs_raw, dict):
                    raise ValueError(f"Invalid call expression kwargs payload: {expr!r}")
                args_eval = [self._eval_expr(item, env, symbols) for item in args_raw]
                kwargs_eval = {
                    str(key): self._eval_expr(value, env, symbols)
                    for key, value in kwargs_raw.items()
                }
                return self._eval_expr_call(callee, args_eval, kwargs_eval)
            return {key: self._eval_expr(value, env, symbols) for key, value in expr.items()}
        if isinstance(expr, str):
            token = expr.strip()
            if token in env:
                return env[token]
            if token in symbols:
                return symbols[token]
            return token
        return expr

    def _expr_config_root(self) -> dict[str, Any]:
        model = self.spec.get("model", {})
        if isinstance(model, dict):
            cfg = model.get("config")
            if isinstance(cfg, dict):
                return cfg
        return {}

    def _expr_config_lookup(self, key: str, *, root: str = "") -> tuple[bool, Any]:
        value: Any = self._expr_config_root()
        if root:
            for part in root.split("."):
                if not isinstance(value, dict) or part not in value:
                    return False, None
                value = value[part]
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return False, None
            value = value[part]
        return True, value

    def _expr_params_has_root(self, root: str) -> bool:
        if root == "":
            return True
        prefix = f"{root}."
        for key in self._state.keys():
            if not isinstance(key, str):
                continue
            if key == root or key.startswith(prefix):
                return True
        return False

    def _eval_expr_call(self, callee: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if callee in {
            "sqrt",
            "Prelude.sqrt",
            "log",
            "Prelude.log",
            "exp",
            "Prelude.exp",
            "sin",
            "Prelude.sin",
            "cos",
            "Prelude.cos",
        }:
            fn_name = callee.split(".", 1)[-1]
            if kwargs:
                raise ValueError(f"{fn_name} expression call does not support kwargs")
            if len(args) != 1:
                raise ValueError(
                    f"{fn_name} expression call expects exactly one positional argument"
                )
            arg = args[0]
            if isinstance(arg, bool) or not isinstance(arg, (int, float)):
                raise ValueError(f"{fn_name} expression call expects numeric argument")
            arg_f = float(arg)
            if fn_name == "sqrt":
                return math.sqrt(arg_f)
            if fn_name == "log":
                return math.log(arg_f)
            if fn_name == "exp":
                return math.exp(arg_f)
            if fn_name == "sin":
                return math.sin(arg_f)
            if fn_name == "cos":
                return math.cos(arg_f)
            raise ValueError(f"Unsupported unary expression call: {callee!r}")
        if callee in {"pow", "Prelude.pow"}:
            if kwargs:
                raise ValueError("pow expression call does not support kwargs")
            if len(args) != 2:
                raise ValueError("pow expression call expects exactly two positional arguments")
            left, right = args
            if isinstance(left, bool) or not isinstance(left, (int, float)):
                raise ValueError("pow expression call expects numeric arguments")
            if isinstance(right, bool) or not isinstance(right, (int, float)):
                raise ValueError("pow expression call expects numeric arguments")
            return math.pow(float(left), float(right))
        if callee in {"abs", "Prelude.abs"}:
            if kwargs:
                raise ValueError("abs expression call does not support kwargs")
            if len(args) != 1:
                raise ValueError("abs expression call expects exactly one positional argument")
            arg = args[0]
            if isinstance(arg, bool) or not isinstance(arg, (int, float)):
                raise ValueError("abs expression call expects numeric argument")
            return abs(arg)
        if callee in {"min", "Prelude.min"}:
            if kwargs:
                raise ValueError("min expression call does not support kwargs")
            if len(args) < 1:
                raise ValueError("min expression call expects at least one positional argument")
            if any(isinstance(arg, bool) or not isinstance(arg, (int, float)) for arg in args):
                raise ValueError("min expression call expects numeric arguments")
            return min(args)
        if callee in {"max", "Prelude.max"}:
            if kwargs:
                raise ValueError("max expression call does not support kwargs")
            if len(args) < 1:
                raise ValueError("max expression call expects at least one positional argument")
            if any(isinstance(arg, bool) or not isinstance(arg, (int, float)) for arg in args):
                raise ValueError("max expression call expects numeric arguments")
            return max(args)
        if callee in {"Config.has", "Config.int", "Config.float", "Config.str", "Config.value"}:
            if len(args) != 1 or not isinstance(args[0], str) or not args[0]:
                raise ValueError(f"{callee} expression call expects one non-empty string key")
            key = args[0]
            root_raw = kwargs.get("root", "")
            root = root_raw if isinstance(root_raw, str) else ""
            found, value = self._expr_config_lookup(key, root=root)
            if callee == "Config.has":
                if "default" in kwargs:
                    raise ValueError("Config.has expression call does not support default kwarg")
                return bool(found)
            if not found:
                if "default" not in kwargs:
                    full_key = f"{root}.{key}" if root else key
                    raise KeyError(
                        f"{callee} expression call missing required config key: {full_key}"
                    )
                value = kwargs["default"]
            if callee == "Config.int":
                if isinstance(value, bool):
                    raise ValueError("Config.int expression call expected int")
                if isinstance(value, int):
                    return int(value)
                if isinstance(value, str):
                    raw = value.strip()
                    if raw and (raw.isdigit() or (raw[0] in {"+", "-"} and raw[1:].isdigit())):
                        return int(raw)
                raise ValueError("Config.int expression call expected int")
            if callee == "Config.float":
                if isinstance(value, bool):
                    raise ValueError("Config.float expression call expected float")
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    raw = value.strip()
                    if raw:
                        return float(raw)
                raise ValueError("Config.float expression call expected float")
            if callee == "Config.str":
                if not isinstance(value, str):
                    raise ValueError("Config.str expression call expected string")
                return value
            if callee == "Config.value":
                return value
        if callee in {"Params.has_root", "Params.root"}:
            if len(args) != 1 or not isinstance(args[0], str):
                raise ValueError(f"{callee} expression call expects one string root argument")
            root = args[0]
            if callee == "Params.has_root":
                if "default" in kwargs:
                    raise ValueError(
                        "Params.has_root expression call does not support default kwarg"
                    )
                return bool(self._expr_params_has_root(root))
            default_value = kwargs.get("default", "")
            if not isinstance(default_value, str):
                raise ValueError("Params.root expression call default must resolve to string")
            return root if self._expr_params_has_root(root) else default_value
        raise ValueError(f"Unsupported call expression: {callee!r}")

    def _join(self, left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        return f"{left}.{right}"

    def _join_scope(self, left: str, right: str) -> str:
        return self._join(left, right)

    def _scope_of(self, node_path: str) -> str:
        if "." not in node_path:
            return ""
        return node_path.rsplit(".", 1)[0]

    def _resolve_state_path(self, *, node_path: str, raw_path: str) -> str:
        if not isinstance(raw_path, str):
            raise ValueError(f"state path must resolve to string, got {raw_path!r}")
        token = raw_path.strip()
        if not token:
            raise ValueError("state path cannot be empty")
        if token.startswith("@@"):
            absolute = token[2:]
            if not absolute:
                raise ValueError("absolute state path cannot be empty")
            return absolute
        if token.startswith("@"):
            token = token[1:]
            if not token:
                raise ValueError("state path cannot be empty")
        scope = self._scope_of(node_path)
        scope_parts = scope.split(".") if scope else []
        synthetic_prefixes = ("n_for_", "n_if_", "n_else_", "n_call_", "n_op_")
        while scope_parts:
            if (
                len(scope_parts) >= 2
                and scope_parts[-1].isdigit()
                and any(scope_parts[-2].startswith(prefix) for prefix in synthetic_prefixes)
            ):
                scope_parts.pop()
                scope_parts.pop()
                continue
            if any(scope_parts[-1].startswith(prefix) for prefix in synthetic_prefixes):
                scope_parts.pop()
                continue
            break
        normalized_scope = ".".join(scope_parts)
        return self._join_scope(normalized_scope, token)

    def _state_tensor_from_resolved_path(self, path: str, *, field: str) -> torch.Tensor:
        resolved = path[2:] if isinstance(path, str) and path.startswith("@@") else path
        if resolved not in self._state:
            raise ValueError(f"{field} tensor not found at path: {resolved}")
        return self._state[resolved]

    def _state_tensor_from_path(
        self,
        *,
        node_path: str,
        raw_path: str,
        field: str,
    ) -> torch.Tensor:
        path = self._resolve_state_path(node_path=node_path, raw_path=raw_path)
        return self._state_tensor_from_resolved_path(path, field=field)
