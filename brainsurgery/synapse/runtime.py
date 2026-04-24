from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from ..core import StateDictLike
from .axon.ast import TypeTensor, parse_type_expr
from .axon.ast.path import (
    path_expr_to_runtime_value,
    resolve_path_expr_to_key,
    runtime_value_to_path_expr,
)
from .mxfp4 import materialize_mxfp4_aliases
from .ops import get_op_module
from .spec_normalize import normalize_synapse_spec_expressions

SymbolValue = Any


def _is_int_token(token: str) -> bool:
    return bool(token) and (token.isdigit() or (token[0] in {"+", "-"} and token[1:].isdigit()))


def _parse_scalar_token(token: str) -> Any:
    value = token.strip()
    lower = value.lower()
    if lower == "null":
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if value and (value.isdigit() or (value[0] in {"+", "-"} and value[1:].isdigit())):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _is_symbol_value(value: Any) -> bool:
    if isinstance(value, (int, float, bool, str)):
        return True
    if isinstance(value, dict):
        return True
    if isinstance(value, list | tuple):
        return all(_is_symbol_value(item) for item in value)
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
        return self._state_tensor_from_resolved_path(path, field="_param")

    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:
        spec = self.spec
        model = spec.get("model", {})
        symbols_raw = model.get("symbols", {})
        symbols: dict[str, SymbolValue] = {
            k: v for k, v in symbols_raw.items() if _is_symbol_value(v)
        }
        self._refresh_symbols_from_config(model, symbols)
        blocks = model.get("blocks", {})
        input_specs = model.get("inputs", {})
        if not isinstance(input_specs, dict):
            raise ValueError("model.inputs must be a mapping when present")
        env = self._prepare_env(input_ids=input_ids, inputs=inputs, input_specs=input_specs)
        # Mirror codegen behavior: seed shape symbols from declared main input
        # tensor types before graph execution so `_ir_alias`/expr name lookups
        # can resolve symbols like B/T even when not explicitly listed in
        # model.symbols.
        self._bind_shape_symbols_from_types(
            env=env,
            input_types=self._block_input_types("main"),
            symbols=symbols,
        )
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

    def _refresh_symbols_from_config(
        self,
        model: dict[str, Any],
        symbols: dict[str, SymbolValue],
    ) -> None:
        config = model.get("config")
        if not isinstance(config, dict):
            return
        graph = model.get("graph")
        if not isinstance(graph, list):
            return
        blocks = model.get("blocks")
        block_graphs: list[list[Any]] = []
        if isinstance(blocks, dict):
            for block in blocks.values():
                if not isinstance(block, dict):
                    continue
                block_graph = block.get("graph")
                if isinstance(block_graph, list):
                    block_graphs.append(block_graph)

        loop_symbol_names = self._collect_for_loop_bound_symbol_names([graph, *block_graphs])
        if not loop_symbol_names:
            return

        config_candidates: dict[str, tuple[str, ...]] = {
            "L": ("num_hidden_layers", "num_layers", "n_layer", "n_layers"),
            "H": ("num_attention_heads", "num_heads", "n_head"),
            "KVH": ("num_key_value_heads", "num_kv_heads", "n_kv_head"),
            "D": ("hidden_size", "d_model", "n_embd"),
            "V": ("vocab_size",),
            "FFN": ("intermediate_size", "ffn_dim"),
            "C": ("max_position_embeddings", "n_positions"),
        }
        for symbol_name in sorted(loop_symbol_names):
            if symbol_name not in symbols:
                continue
            key_candidates = config_candidates.get(symbol_name)
            if not key_candidates:
                continue
            config_value = self._lookup_first_int_config_value(config, key_candidates)
            if config_value is not None:
                symbols[symbol_name] = config_value

    def _collect_for_loop_bound_symbol_names(self, graphs: list[list[Any]]) -> set[str]:
        out: set[str] = set()

        def _visit(items: list[Any]) -> None:
            for item in items:
                if not isinstance(item, dict) or len(item) != 1:
                    continue
                _, node_spec = next(iter(item.items()))
                if not isinstance(node_spec, dict):
                    continue
                if node_spec.get("_op") == "for":
                    to_expr = node_spec.get("_to")
                    if isinstance(to_expr, str) and to_expr:
                        out.add(to_expr)
                    if (
                        isinstance(to_expr, dict)
                        and to_expr.get("_expr") == "name"
                        and isinstance(to_expr.get("id"), str)
                    ):
                        out.add(str(to_expr["id"]))
                    body = node_spec.get("_body")
                    if isinstance(body, list):
                        _visit(body)
                nested = node_spec.get("graph")
                if isinstance(nested, list):
                    _visit(nested)
                then_branch = node_spec.get("_then")
                if isinstance(then_branch, list):
                    _visit(then_branch)
                else_branch = node_spec.get("_else")
                if isinstance(else_branch, list):
                    _visit(else_branch)

        for graph in graphs:
            _visit(graph)
        return out

    def _lookup_first_int_config_value(
        self, config: dict[str, Any], key_candidates: tuple[str, ...]
    ) -> int | None:
        for key in key_candidates:
            value: Any = config
            ok = True
            for part in key.split("."):
                if not isinstance(value, dict) or part not in value:
                    ok = False
                    break
                value = value[part]
            if not ok:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return int(value)
            if isinstance(value, str):
                raw = value.strip()
                if raw and (raw.isdigit() or (raw[0] in {"+", "-"} and raw[1:].isdigit())):
                    return int(raw)
        return None

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
                raise ValueError("runtime no longer supports for-nodes; flatten/lower first")
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

        # Block calls should not implicitly capture caller locals by name.
        # Only declared inputs/kwargs are bound into the callee environment.
        block_env: dict[str, Any] = {}
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
        # Keep shape/type symbol inference local to the block call so that
        # helper signatures (e.g. Tensor[B,S,D]) do not leak/lock caller symbols.
        block_symbols = dict(symbols)
        self._bind_shape_symbols_from_types(
            env=block_env,
            input_types=self._block_input_types(block_name),
            symbols=block_symbols,
        )
        self._validate_input_shapes(block_env, block_inputs, block_symbols)
        self._run_graph(block_graph, block_env, scope=scope, symbols=block_symbols, blocks=blocks)

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
        self,
        node_spec: dict[str, Any],
        *,
        node_path: str,
        param_name: str,
        env: Mapping[str, Any],
    ) -> str:
        def _effective_scope() -> str:
            abs_path = node_spec.get("_abs_path")
            if isinstance(abs_path, (str, dict)) and abs_path:
                return resolve_path_expr_to_key(
                    abs_path,
                    self._path_template_env(env),
                    op_name="_abs_path",
                )
            return self._scope_of(node_path)

        def _explicit_candidates(raw_path: Any) -> list[str]:
            expr = runtime_value_to_path_expr(raw_path, op_name=f"{param_name} path")
            key = resolve_path_expr_to_key(
                expr,
                self._path_template_env(env),
                op_name=f"{param_name} path",
            )
            if expr.absolute:
                return [key] if key else []
            scope_prefix = _effective_scope()
            scoped = self._join(scope_prefix, key)
            return [scoped if scoped else key]

        def _pick_explicit_candidate(raw: Any) -> str | None:
            if isinstance(raw, (str, dict)):
                candidates = _explicit_candidates(raw)
                for candidate in candidates:
                    if candidate in self._state:
                        return candidate
                return candidates[0]
            if isinstance(raw, list | tuple):
                list_candidates: list[str] = []
                for item in raw:
                    if not isinstance(item, (str, dict)):
                        continue
                    list_candidates.extend(_explicit_candidates(item))
                if not list_candidates:
                    return None
                for candidate in list_candidates:
                    if candidate in self._state:
                        return candidate
                return list_candidates[0]
            return None

        param_base = node_spec.get("param_base")
        if isinstance(param_base, str):
            base_resolved = node_spec.get(param_base)
            base = base_resolved if isinstance(base_resolved, str) else param_base
            scoped_base = self._join(_effective_scope(), base)
            scoped_param = f"{scoped_base}.{param_name}" if scoped_base else param_name
            if scoped_param in self._state:
                return scoped_param
            return scoped_param
        # Explicit per-node path override wins over _params.
        explicit_params = node_spec.get("_params")
        if param_name in node_spec and isinstance(node_spec[param_name], (str, dict)):
            candidate = node_spec[param_name]
            if not (isinstance(candidate, str) and candidate == param_name):
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
        if fallback in self._state:
            return fallback
        return fallback

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
                    # Block-call symbol bindings are local to the callee frame.
                    # Allow type-driven dimensions to shadow caller-carried values
                    # when symbol names are reused across blocks.
                    if current != actual:
                        symbols[dim] = actual
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
                if ident is None:
                    ident = expr.get("value")
                if not isinstance(ident, str) or not ident:
                    raise ValueError(f"Invalid name expression payload: {expr!r}")
                if ident in env:
                    return env[ident]
                if ident in symbols:
                    return self._eval_expr(symbols[ident], env, symbols)
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
            if kind == "path":
                try:
                    return path_expr_to_runtime_value(
                        runtime_value_to_path_expr(expr, op_name="path expression")
                    )
                except ValueError as exc:
                    raise ValueError(f"Invalid path expression payload: {expr!r}") from exc
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
                return self._eval_expr_call(callee, args_eval, kwargs_eval, env, symbols)
            return {key: self._eval_expr(value, env, symbols) for key, value in expr.items()}
        if isinstance(expr, str):
            token = expr.strip()
            if token in env:
                return env[token]
            if token in symbols:
                return self._eval_expr(symbols[token], env, symbols)
            return _parse_scalar_token(token)
        return expr

    def _expr_config_root(self) -> dict[str, Any]:
        model = self.spec.get("model", {})
        if isinstance(model, dict):
            cfg = model.get("config")
            if isinstance(cfg, dict):
                return cfg
        return {}

    def _resolve_config_path_key(
        self, raw: Any, env: Mapping[str, Any], op_name: str = "Config"
    ) -> str:
        return resolve_path_expr_to_key(
            raw,
            self._path_template_env(env, symbols=self._symbols),
            op_name=op_name,
        )

    def _expr_config_lookup(self, key: str) -> tuple[bool, Any]:
        value: Any = self._expr_config_root()
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

    def _eval_expr_call(
        self,
        callee: str,
        args: list[Any],
        kwargs: dict[str, Any],
        env: Mapping[str, Any],
        symbols: Mapping[str, Any],
    ) -> Any:
        inline_path_callexprs = {
            "Config.value",
            "Config.has_key",
            "Config.has_value",
            "Config.int",
            "Config.dim",
            "Config.float",
            "Config.str",
            "Config.bool",
            "Config.list",
        }
        callee_base = callee.split("@", 1)[0] if "@" in callee else callee
        if "@" in callee and callee_base in inline_path_callexprs:
            suffix = callee[len(callee_base) :]
            if not suffix or not suffix.startswith("@"):
                raise ValueError(f"Unsupported call expression: {callee!r}")
            if args:
                raise ValueError(
                    f"{callee_base} expression call cannot mix inline @path and positional args"
                )
            args = [suffix]
            callee = callee_base
        if callee in {
            "sqrt",
            "Prelude.sqrt",
            "Math.sqrt",
            "log",
            "Prelude.log",
            "Math.log",
            "exp",
            "Prelude.exp",
            "Math.exp",
            "sin",
            "Prelude.sin",
            "Math.sin",
            "cos",
            "Prelude.cos",
            "Math.cos",
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
        if callee in {"pow", "Prelude.pow", "Math.pow"}:
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
        if callee == "Config.value":
            if len(args) != 1:
                raise ValueError("Config.value expression call expects one non-empty Path key")
            if "root" in kwargs:
                raise ValueError("Config.value expression call does not support root kwarg")
            key = self._resolve_config_path_key(args[0], {**symbols, **env}, "Config.value")
            found, value = self._expr_config_lookup(key)
            if not found:
                if "default" not in kwargs:
                    raise KeyError(f"{callee} expression call missing required config key: {key}")
                value = kwargs["default"]
            return value
        if callee in {
            "Config.has_key",
            "Config.has_value",
            "Config.int",
            "Config.dim",
            "Config.float",
            "Config.str",
            "Config.bool",
            "Config.list",
        }:
            if len(args) != 1:
                raise ValueError(f"{callee} expression call expects one non-empty Path key")
            if "root" in kwargs:
                raise ValueError(f"{callee} expression call does not support root kwarg")
            key = self._resolve_config_path_key(args[0], {**symbols, **env}, callee)
            found, value = self._expr_config_lookup(key)
            if callee == "Config.has_key":
                if "default" in kwargs:
                    raise ValueError(
                        "Config.has_key expression call does not support default kwarg"
                    )
                return bool(found)
            if callee == "Config.has_value":
                if "default" in kwargs:
                    raise ValueError(
                        "Config.has_value expression call does not support default kwarg"
                    )
                return bool(found) and value is not None
            if not found:
                if "default" not in kwargs:
                    raise KeyError(f"{callee} expression call missing required config key: {key}")
                value = kwargs["default"]
            if callee in {"Config.int", "Config.dim"}:
                if isinstance(value, bool):
                    raise ValueError(f"{callee} expression call expected int")
                if isinstance(value, int):
                    return int(value)
                if isinstance(value, str):
                    raw = value.strip()
                    if raw and (raw.isdigit() or (raw[0] in {"+", "-"} and raw[1:].isdigit())):
                        return int(raw)
                raise ValueError(f"{callee} expression call expected int")
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
            if callee == "Config.bool":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    raw = value.strip().lower()
                    if raw == "true":
                        return True
                    if raw == "false":
                        return False
                raise ValueError("Config.bool expression call expected bool")
            if callee == "Config.list":
                if isinstance(value, list):
                    return value
                if isinstance(value, tuple):
                    return list(value)
                raise ValueError("Config.list expression call expected list")
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

    def _path_template_env(
        self,
        env: Mapping[str, Any] | None,
        *,
        symbols: Mapping[str, SymbolValue] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(env, Mapping):
            return {}
        raw_env = dict(env)
        resolved: dict[str, Any] = {}
        symbol_values = self._symbols if symbols is None else dict(symbols)
        for key, value in raw_env.items():
            if isinstance(value, dict | list | tuple):
                try:
                    resolved[key] = self._eval_expr(value, raw_env, symbol_values)
                    continue
                except Exception:
                    pass
            resolved[key] = value
        for key, value in list(resolved.items()):
            if not isinstance(value, (str, dict)):
                continue
            try:
                expr = runtime_value_to_path_expr(value, op_name="path template env")
                resolved[key] = resolve_path_expr_to_key(
                    expr,
                    resolved,
                    op_name="path template env",
                )
            except Exception:
                pass
        return resolved

    def _resolve_state_path(
        self, *, node_path: str, raw_path: Any, env: Mapping[str, Any] | None = None
    ) -> str:
        expr = runtime_value_to_path_expr(raw_path, op_name="state path")
        token = resolve_path_expr_to_key(
            expr,
            self._path_template_env(env),
            op_name="state path",
        )
        if expr.absolute:
            return token
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
        scoped = self._join_scope(normalized_scope, token)
        if scoped in self._state:
            return scoped
        return scoped

    def _state_tensor_from_resolved_path(self, path: str, *, field: str) -> torch.Tensor:
        resolved = path[2:] if isinstance(path, str) and path.startswith("@@") else path
        if resolved not in self._state:
            alternatives = self._state_key_alternatives(resolved, limit=8)
            alt_text = ", ".join(alternatives) if alternatives else "<none>"
            raise ValueError(
                f"{field} tensor not found at path: {resolved}. Alternatives: {alt_text}"
            )
        return self._state[resolved]

    def _state_tensor_from_path(
        self,
        *,
        node_path: str,
        raw_path: Any,
        field: str,
        env: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        path = self._resolve_state_path(node_path=node_path, raw_path=raw_path, env=env)
        return self._state_tensor_from_resolved_path(path, field=field)

    def _state_key_alternatives(self, key: str, *, limit: int = 8) -> list[str]:
        if not isinstance(key, str) or not key:
            return []
        keys = [k for k in self._state.keys() if isinstance(k, str)]
        if not keys:
            return []
        out: list[str] = []
        seen: set[str] = set()

        def _add(candidate: str) -> None:
            if candidate in seen:
                return
            seen.add(candidate)
            out.append(candidate)

        segments = key.split(".")
        leaf = segments[-1]
        for existing in keys:
            if existing.endswith(f".{leaf}") or existing == leaf:
                _add(existing)
                if len(out) >= limit:
                    return out
        if len(segments) >= 2:
            tail2 = ".".join(segments[-2:])
            for existing in keys:
                if existing.endswith(f".{tail2}") or existing == tail2:
                    _add(existing)
                    if len(out) >= limit:
                        return out
        for prefix in ("model.", "transformer."):
            prefixed = f"{prefix}{key}"
            if prefixed in self._state:
                _add(prefixed)
            if len(out) >= limit:
                return out
        if key.startswith("model.") and key[len("model.") :] in self._state:
            _add(key[len("model.") :])
        if key.startswith("transformer.") and key[len("transformer.") :] in self._state:
            _add(key[len("transformer.") :])
        if len(out) >= limit:
            return out
        for existing in keys:
            if key in existing or existing in key:
                _add(existing)
                if len(out) >= limit:
                    return out
        return out
