from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from .mxfp4 import materialize_mxfp4_aliases
from .ops import get_op_module


class SynapseProgramModel(nn.Module):
    """Generic runtime for Synapse graph specs backed by checkpoint tensors."""

    SPEC: dict[str, Any] = {}
    OP_MAP: dict[str, Any] = {}

    def __init__(
        self,
        spec: dict[str, Any] | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.spec: dict[str, Any] = self._resolve_spec(spec)
        self._state: dict[str, torch.Tensor] = {}
        if state_dict is not None:
            self.load_state_dict_tensors(state_dict)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        spec: dict[str, Any] | None = None,
    ) -> "SynapseProgramModel":
        return cls(spec=spec, state_dict=state_dict)

    @classmethod
    def from_spec(
        cls,
        spec: dict[str, Any],
        *,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> "SynapseProgramModel":
        return cls(spec=spec, state_dict=state_dict)

    @classmethod
    def from_yaml(
        cls,
        spec_path: str | Path,
        *,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> "SynapseProgramModel":
        loaded = OmegaConf.load(Path(spec_path))
        data = OmegaConf.to_container(loaded, resolve=True)
        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML mapping at {spec_path}, got {type(data).__name__}")
        return cls(spec={str(key): value for key, value in data.items()}, state_dict=state_dict)

    def load_state_dict_tensors(self, state_dict: dict[str, torch.Tensor]) -> None:
        loaded = dict(state_dict)
        materialize_mxfp4_aliases(loaded, drop_packed=True)
        self._state = loaded

    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:
        spec = self.spec
        model = spec.get("model", {})
        symbols_raw = model.get("symbols", {})
        symbols = {k: v for k, v in symbols_raw.items() if isinstance(v, (int, float, bool))}
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
        symbols: dict[str, int | float | bool],
        blocks: dict[str, Any],
    ) -> None:
        for item in graph:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f"Invalid graph item: {item!r}")
            node_name, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                raise ValueError(f"Node spec for {node_name!r} must be mapping")

            when_expr = node_spec.get("when")
            if when_expr is not None:
                for produced_name in self._node_output_names(node_spec):
                    env.setdefault(produced_name, None)
                if not self._check_when(when_expr, env, symbols):
                    continue

            op = node_spec.get("_op")
            if op == "for":
                scope_name = node_spec.get("_scope")
                if not isinstance(scope_name, str) or not scope_name:
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
                    for_scope = self._join(scope, f"{scope_name}.{iter_value}")
                    self._run_graph(body, env, scope=for_scope, symbols=symbols, blocks=blocks)
                env.pop(var_name, None)
                continue

            if op == "call":
                self._run_block_call(node_spec, env, scope=scope, symbols=symbols, blocks=blocks)
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

    def _run_block_call(
        self,
        node_spec: dict[str, Any],
        env: dict[str, Any],
        *,
        scope: str,
        symbols: dict[str, int | float | bool],
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
            if key.startswith("_") or key in {"when", "graph"}:
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
            if (
                scope == raw_scope
                or scope.startswith(f"{raw_scope}.")
                or scope.endswith(f".{raw_scope}")
                or f".{raw_scope}." in scope
            ):
                call_scope = scope
            else:
                call_scope = self._join(scope, raw_scope)
        self._run_graph(block_graph, block_env, scope=call_scope, symbols=symbols, blocks=blocks)

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
        symbols: dict[str, int | float | bool],
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
        param_base = node_spec.get("param_base")
        if isinstance(param_base, str):
            base_resolved = node_spec.get(param_base)
            base = base_resolved if isinstance(base_resolved, str) else param_base
            scoped_base = self._join(self._scope_of(node_path), base)
            return f"{scoped_base}.{param_name}" if scoped_base else param_name
        explicit_params = node_spec.get("_params")
        if isinstance(explicit_params, dict):
            explicit = explicit_params.get(param_name)
            if isinstance(explicit, str):
                scoped_explicit = self._join(self._scope_of(node_path), explicit)
                return f"{scoped_explicit}" if scoped_explicit else explicit
        if param_name in node_spec and isinstance(node_spec[param_name], str):
            candidate = node_spec[param_name]
            scoped_candidate = self._join(self._scope_of(node_path), candidate)
            return f"{scoped_candidate}" if scoped_candidate else candidate
        return f"{node_path}.{param_name}" if node_path else param_name

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
        symbols: dict[str, int | float | bool],
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
                if re.fullmatch(r"-?[0-9]+", token):
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
                elif actual != bound:
                    raise ValueError(
                        f"Input {input_name!r} shape mismatch at axis {axis}: "
                        f"symbol {token} was previously bound to {bound}, got {actual}"
                    )

    def _read_tensor_input(self, ref: Any, env: dict[str, Any]) -> torch.Tensor:
        if not isinstance(ref, str):
            raise ValueError(f"Expected string tensor reference, got {ref!r}")
        value = env.get(ref)
        if not torch.is_tensor(value):
            raise ValueError(f"Input reference {ref!r} does not resolve to tensor")
        return value

    def _check_when(
        self, when_expr: Any, env: dict[str, Any], symbols: dict[str, int | float | bool]
    ) -> bool:
        if when_expr is None:
            return True
        value = self._eval_expr(when_expr, env, symbols)
        return bool(value)

    def _eval_expr(
        self, expr: Any, env: dict[str, Any], symbols: dict[str, int | float | bool]
    ) -> Any:
        if expr is None:
            return None
        if isinstance(expr, (int, float, bool)):
            return expr
        if isinstance(expr, str):
            token = expr.strip()
            if token in env:
                return env[token]
            if token in symbols:
                return symbols[token]
            if token.lower() == "true":
                return True
            if token.lower() == "false":
                return False
            if token.lower() == "null":
                return None
            return self._safe_eval_numeric(token, env, symbols)
        return expr

    def _safe_eval_numeric(
        self, text: str, env: dict[str, Any], symbols: dict[str, int | float | bool]
    ) -> Any:
        names: dict[str, Any] = {}
        for key, value in symbols.items():
            names[key] = value
        for key, value in env.items():
            if isinstance(value, (int, float, bool)) or value is None:
                names[key] = value

        parsed = ast.parse(text, mode="eval")
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Constant,
            ast.Name,
            ast.Load,
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.Not,
        )
        for node in ast.walk(parsed):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Unsupported expression: {text!r}")
            if isinstance(node, ast.Name) and node.id not in names:
                raise ValueError(f"Unknown symbol in expression: {node.id}")

        code = compile(parsed, "<synapse-expr>", "eval")
        return eval(code, {"__builtins__": {}, "math": math}, names)

    def _join(self, left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        return f"{left}.{right}"

    def _scope_of(self, node_path: str) -> str:
        if "." not in node_path:
            return ""
        return node_path.rsplit(".", 1)[0]
