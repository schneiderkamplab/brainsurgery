from __future__ import annotations

from typing import Any

from ..base import BaseEmitter
from .ops import get_op_module as get_tinygrad_op


class TinyGradEmitter(BaseEmitter):
    """TinyGrad backend emitter that produces plain Python class source code."""

    BACKEND_NAME = "tinygrad"

    # ------------------------------------------------------------------
    # Op dispatch — uses TinyGrad-specific ops registry
    # ------------------------------------------------------------------

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
        """Compile a single op node using the TinyGrad ops registry."""
        op_module = get_tinygrad_op(op)
        if op_module is None:
            raise NotImplementedError(f"Unsupported op in TinyGrad codegen: {op!r}")
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

    def _op_uses_node_path(self, op: str, node_spec: dict[str, Any]) -> bool:
        """Check whether a TinyGrad op requires the full node path."""
        op_module = get_tinygrad_op(op)
        if op_module is None:
            raise NotImplementedError(f"Unsupported op in TinyGrad codegen: {op!r}")
        return bool(op_module.uses_node_path(self, node_spec))

    # ------------------------------------------------------------------
    # Main render — produces a plain Python class (NOT nn.Module)
    # ------------------------------------------------------------------

    def render(self) -> str:
        lines: list[str] = []
        lines.extend(
            [
                "from __future__ import annotations",
                "",
                "from typing import Any",
                "",
                "import math",
                "from tinygrad import Tensor, dtypes",
                "",
                "",
                f"class {self.class_name}:",
                "    def __init__(self, state_dict: dict[str, Any] | None = None) -> None:",
                "        self._state: dict[str, Any] = {}",
                f"        self._symbols: dict[str, int | float | bool] = {repr(self.symbols)}",
                "        self._trace_enabled = False",
                "        self.trace_ops: list[dict[str, Any]] = []",
                "        if state_dict is not None:",
                "            self.load_state_dict_tensors(state_dict)",
                "",
                "    def __call__(self, *args, **kwargs):",
                "        return self.forward(*args, **kwargs)",
                "",
                "    @classmethod",
                '    def from_state_dict(cls, state_dict):',
                "        return cls(state_dict=state_dict)",
                "",
                "    def load_state_dict_tensors(self, state_dict: dict[str, Any]) -> None:",
                "        self._state = dict(state_dict)",
                "",
                "    def _param(self, path: str) -> Any:",
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
                "    def _first_tensor(self, value: Any) -> Any | None:",
                "        if isinstance(value, Tensor):",
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
                "            'tensor': tensor.realize().numpy(),",
                "        })",
                "",
                "    def _prepare_env(self, input_ids: Any | None, inputs: dict[str, Any], input_specs: dict[str, Any]) -> dict[str, Any]:",
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

    # ------------------------------------------------------------------
    # Block methods
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Forward method
    # ------------------------------------------------------------------

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
            "    def forward(self, input_ids: Tensor | None = None, **inputs: Any) -> Any:",
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

    # ------------------------------------------------------------------
    # Generate method
    # ------------------------------------------------------------------

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
            "    def generate(self, input_ids: Tensor, *, eos_token_id: int, max_len: int, attention_mask: Tensor | None = None, attn_mask: Tensor | None = None) -> Tensor:",
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
            "        if input_ids.shape[1] >= max_len:",
            "            return input_ids[:, :max_len]",
            "",
            "        batch, start_len = input_ids.shape",
            "        generated = Tensor.zeros(batch, max_len, dtype=input_ids.dtype)",
            "        generated[:, :start_len] = input_ids",
            "        generated_mask = None",
            "        if mask is not None:",
            "            generated_mask = Tensor.zeros(batch, max_len, dtype=mask.dtype)",
            "            generated_mask[:, :start_len] = mask",
            "        cache_state = None",
            "        finished = Tensor.zeros(batch, dtype=dtypes.bool)",
            "        cur_len = start_len",
            "        while cur_len < max_len:",
            "            step_input = generated[:, :cur_len] if cache_state is None else generated[:, cur_len - 1:cur_len]",
            "            call_kwargs: dict[str, Any] = {}",
            "            if generated_mask is not None:",
        ]
        has_mask_input = False
        if "attention_mask" in inputs:
            has_mask_input = True
            lines.append(
                "                call_kwargs['attention_mask'] = generated_mask[:, :cur_len]"
            )
        if "attn_mask" in inputs:
            has_mask_input = True
            lines.append(
                "                call_kwargs['attn_mask'] = generated_mask[:, :cur_len]"
            )
        if not has_mask_input:
            lines.append("                pass")
        if state_input_name is not None:
            lines.append(f"            call_kwargs[{state_input_name!r}] = cache_state")
        if use_cache_name is not None:
            lines.append(f"            call_kwargs[{use_cache_name!r}] = True")
        lines.extend(
            [
                "            model_out = self.forward(step_input, **call_kwargs)",
                "            if isinstance(model_out, dict):",
                "                logits = model_out['logits']",
            ]
        )
        for idx, out_name in enumerate(state_output_names):
            if idx == 0:
                lines.append(f"                if {out_name!r} in model_out:")
            else:
                lines.append(f"                elif {out_name!r} in model_out:")
            lines.append(f"                    cache_state = model_out[{out_name!r}]")
        lines.extend(
            [
                "            else:",
                "                logits = model_out",
                "            logits = logits.realize()",
                "            next_token = logits[:, -1, :].argmax(axis=-1)",
                "            next_token = next_token.realize().numpy()",
                "            next_token = Tensor(where(finished.numpy(), eos_token_id, next_token), dtype=input_ids.dtype)",
                "            generated[:, cur_len] = next_token",
                "            finished = finished | (next_token == eos_token_id)",
                "            if generated_mask is not None:",
                "                generated_mask[:, cur_len] = 1",
                "            cur_len += 1",
                "        return generated[:, :cur_len]",
                "",
            ]
        )
        return lines
