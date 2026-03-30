from __future__ import annotations

import ast
import importlib.resources
import re
from typing import Any

from omegaconf import OmegaConf

from .ops import get_op_module


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
    if spec.get("synapse") != 1:
        raise ValueError("Only synapse: 1 specs are supported")

    resolved_op_map = load_synapse_torch_op_map() if op_map is None else op_map
    _validate_spec_ops(spec, resolved_op_map)

    model = spec.get("model")
    if not isinstance(model, dict):
        raise ValueError("spec.model must be a mapping")

    symbols_raw = model.get("symbols", {})
    symbols = {k: v for k, v in symbols_raw.items() if isinstance(v, (int, float, bool))}

    emitter = _Emitter(class_name=class_name, spec=spec, symbols=symbols)
    return emitter.render()


class _Emitter:
    def __init__(
        self, *, class_name: str, spec: dict[str, Any], symbols: dict[str, int | float | bool]
    ) -> None:
        self.class_name = class_name
        self.spec = spec
        self.model = spec["model"]
        self.blocks = self.model.get("blocks", {})
        self.symbols = symbols
        self._counter = 0
        self._active_env: dict[str, str] = {}

    def render(self) -> str:
        lines: list[str] = []
        lines.extend(
            [
                "from __future__ import annotations",
                "",
                "from typing import Any",
                "",
                "from brainsurgery.synapse.mxfp4 import materialize_mxfp4_aliases",
                "",
                "import math",
                "import torch",
                "from torch import nn",
                "from torch.nn import functional as F",
                "",
                "",
                f"class {self.class_name}(nn.Module):",
                "    def __init__(self, state_dict: dict[str, torch.Tensor] | None = None, runtime_state_dict: Any | None = None) -> None:",
                "        super().__init__()",
                "        self._state: dict[str, torch.Tensor] = {}",
                "        self._runtime_state_dict = runtime_state_dict",
                f"        self._symbols: dict[str, int | float | bool] = {repr(self.symbols)}",
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
                "        return self._state[path]",
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
                "    def _trace_op(self, node_path: str, op: str, bind: str, value: Any) -> None:",
                "        if not bool(getattr(self, '_trace_enabled', False)):",
                "            return",
                "        tensor = self._first_tensor(value)",
                "        if tensor is None:",
                "            return",
                "        self.trace_ops.append({",
                "            'node_path': str(node_path),",
                "            'op': str(op),",
                "            'bind': str(bind),",
                "            'dtype': str(tensor.dtype),",
                "            'tensor': tensor.detach().float().cpu(),",
                "        })",
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

        arg_names = [self._py_name(name) for name in inputs]
        env: dict[str, str] = {name: py for name, py in zip(inputs, arg_names, strict=True)}

        sig = ", ".join(["self", *arg_names, "scope: str"])
        lines = [f"    def _block_{self._py_name(block_name)}({sig}) -> tuple[Any, ...]:"]
        lines.append("        emitter = self")
        lines.append("        env: dict[str, Any] = {}")
        for syn_name, py_name in env.items():
            lines.append(f"        env[{syn_name!r}] = {py_name}")

        body = self._compile_graph(graph=graph, env=env, scope_var="scope", indent="        ")
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

        env: dict[str, str] = {}
        for name, input_spec in inputs.items():
            is_optional = isinstance(input_spec, dict) and bool(input_spec.get("optional", False))
            py_name = self._py_name(name)
            if is_optional:
                lines.append(f"        {py_name} = env.get({name!r})")
            else:
                lines.append(f"        {py_name} = self._safe_get(env, {name!r})")
            env[name] = py_name

        lines.extend(
            self._compile_graph(graph=graph, env=env, scope_var="scope", indent="        ")
        )

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

            when = node_spec.get("when")
            inner_indent = indent
            if when is not None:
                produced_names = self._node_output_names(node_spec)
                for produced_name in produced_names:
                    existing = env.get(produced_name)
                    if isinstance(existing, str):
                        # Preserve the previously bound value when the conditional does not execute.
                        continue
                    out_var = self._fresh(self._py_name(produced_name))
                    lines.append(f"{indent}{out_var} = None")
                    env[produced_name] = out_var
                cond = self._expr_code(when, env)
                lines.append(f"{indent}if {cond}:")
                inner_indent = indent + "    "

            op = node_spec.get("_op")
            if op == "for":
                scope_name = node_spec.get("_scope")
                if not isinstance(scope_name, str) or not scope_name:
                    raise ValueError("for requires string _scope")
                var_name = node_spec.get("_var")
                if not isinstance(var_name, str):
                    raise ValueError("for requires string _var")
                to_code = self._expr_code(node_spec.get("_to"), env)
                from_code = self._expr_code(node_spec.get("_from", 0), env)
                step_code = self._expr_code(node_spec.get("_step", 1), env)
                saved = env.get(var_name)
                iter_value = self._fresh(self._py_name(var_name))
                to_var = self._fresh("to")
                from_var = self._fresh("from")
                step_var = self._fresh("step")
                lines.append(f"{inner_indent}{to_var} = int({to_code})")
                lines.append(f"{inner_indent}{from_var} = int({from_code})")
                lines.append(f"{inner_indent}{step_var} = int({step_code})")
                lines.append(
                    f"{inner_indent}for {iter_value} in self._for_values(from_value={from_var}, to_value={to_var}, step_value={step_var}):"
                )
                env[var_name] = iter_value
                child_scope = self._fresh("scope")
                lines.append(
                    f"{inner_indent}    {child_scope} = self._join_scope({scope_var}, f'{scope_name}.{{{iter_value}}}')"
                )
                body = node_spec.get("_body")
                if not isinstance(body, list):
                    raise ValueError("for requires list _body")
                lines.extend(
                    self._compile_graph(
                        graph=body, env=env, scope_var=child_scope, indent=inner_indent + "    "
                    )
                )
                if saved is None:
                    env.pop(var_name, None)
                else:
                    env[var_name] = saved
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
                    lines.append(
                        f"{inner_indent}self._trace_op({trace_node_path}, {op!r}, {out_name!r}, {_out_var})"
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
            if isinstance(src, str) and src in env:
                arg_codes.append(f"{block_input_name}={env[src]}")
            else:
                arg_codes.append(f"{block_input_name}={self._expr_code(src, env)}")
        for key, value in node_spec.items():
            if key.startswith("_") or key in {"when", "graph"}:
                continue
            if key not in block_inputs:
                continue
            if isinstance(value, str) and value in env:
                arg_codes.append(f"{key}={env[value]}")
            else:
                arg_codes.append(f"{key}={self._expr_code(value, env)}")

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

        call_scope_var = scope_var
        raw_scope = node_spec.get("_scope")
        if isinstance(raw_scope, str) and raw_scope:
            scoped_var = self._fresh("scope")
            lines = [
                (
                    f"{indent}if ("
                    f"{scope_var} == {raw_scope!r} or "
                    f"{scope_var}.startswith({raw_scope!r} + '.') or "
                    f"{scope_var}.endswith('.' + {raw_scope!r}) or "
                    f"('.' + {raw_scope!r} + '.') in {scope_var}"
                    f"):"
                ),
                f"{indent}    {scoped_var} = {scope_var}",
                f"{indent}else:",
                f"{indent}    {scoped_var} = self._join_scope({scope_var}, {raw_scope!r})",
            ]
            call_scope_var = scoped_var
        else:
            lines = []

        call_args = ", ".join([*arg_codes, f"scope={call_scope_var}"])
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
        self._active_env = env
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

    def _assign_out_var(self, env: dict[str, str], out_name: str) -> str:
        existing = env.get(out_name)
        if isinstance(existing, str):
            return existing
        out_var = self._fresh(self._py_name(out_name))
        env[out_name] = out_var
        return out_var

    def _infer_param_expr(
        self, node_spec: dict[str, Any], node_path_var: str, param_name: str
    ) -> str:
        param_base = node_spec.get("param_base")
        if isinstance(param_base, str):
            scope_expr = f"self._scope_of({node_path_var})"
            if param_base in self._active_env:
                base_expr = self._active_env[param_base]
                return (
                    f"self._join_scope(self._join_scope({scope_expr}, {base_expr}), {param_name!r})"
                )
            if isinstance(node_spec.get(param_base), str):
                base_expr = repr(node_spec[param_base])
                return (
                    f"self._join_scope(self._join_scope({scope_expr}, {base_expr}), {param_name!r})"
                )
            base_expr = repr(param_base)
            return f"self._join_scope(self._join_scope({scope_expr}, {base_expr}), {param_name!r})"
        explicit_params = node_spec.get("_params")
        if isinstance(explicit_params, dict) and isinstance(explicit_params.get(param_name), str):
            scoped_explicit = f"self._join_scope(self._scope_of({node_path_var}), {explicit_params[param_name]!r})"
            return scoped_explicit
        if isinstance(node_spec.get(param_name), str):
            candidate = node_spec[param_name]
            return f"self._join_scope(self._scope_of({node_path_var}), {candidate!r})"
        return f"self._join_scope({node_path_var}, {param_name!r})"

    def _read_env_var(self, env: dict[str, str], name: str) -> str:
        if name not in env:
            raise ValueError(f"Unknown input var {name!r}")
        return env[name]

    def _expr_code(self, expr: Any, env: dict[str, str]) -> str:
        if expr is None:
            return "None"
        if isinstance(expr, (int, float, bool)):
            return repr(expr)
        if isinstance(expr, str):
            token = expr.strip()
            if token in env:
                return env[token]
            if token in self.symbols:
                return repr(self.symbols[token])
            if token.lower() in {"true", "false", "null"}:
                return {"true": "True", "false": "False", "null": "None"}[token.lower()]
            numeric = self._try_eval_numeric(token)
            if numeric is not None:
                return repr(numeric)
            return self._substitute_expr_names(token, env)
        return repr(expr)

    def _substitute_expr_names(self, text: str, env: dict[str, str]) -> str:
        rewritten = text
        for name, py_name in sorted(env.items(), key=lambda kv: len(kv[0]), reverse=True):
            rewritten = re.sub(rf"\b{re.escape(name)}\b", py_name, rewritten)
        for name, value in sorted(self.symbols.items(), key=lambda kv: len(kv[0]), reverse=True):
            rewritten = re.sub(rf"\b{re.escape(name)}\b", repr(value), rewritten)
        return rewritten

    def _try_eval_numeric(self, text: str) -> int | float | None:
        names = dict(self.symbols)
        try:
            parsed = ast.parse(text, mode="eval")
        except SyntaxError:
            return None

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
        )
        for node in ast.walk(parsed):
            if not isinstance(node, allowed_nodes):
                return None
            if isinstance(node, ast.Name) and node.id not in names:
                return None
        try:
            value = eval(compile(parsed, "<synapse-expr>", "eval"), {"__builtins__": {}}, names)
        except Exception:
            return None
        if isinstance(value, (int, float)):
            return value
        return None

    def _py_name(self, value: str) -> str:
        name = re.sub(r"[^0-9A-Za-z_]", "_", value)
        if not name:
            name = "v"
        if name[0].isdigit():
            name = f"v_{name}"
        return name

    def _fresh(self, base: str) -> str:
        self._counter += 1
        return f"{base}_{self._counter}"


def _validate_spec_ops(spec: dict[str, Any], op_map: dict[str, Any]) -> None:
    ops = op_map.get("ops")
    if not isinstance(ops, dict):
        raise ValueError("op map must contain mapping key 'ops'")

    known_control_ops = {"for", "call"}

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

            if op == "for":
                body = node_spec.get("_body")
                if not isinstance(body, list):
                    raise ValueError("for node requires list '_body'")
                _walk_graph(body)

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
