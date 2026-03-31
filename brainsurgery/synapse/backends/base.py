from __future__ import annotations

import ast
import re
from typing import Any

from ..ops import OP_MODULES, get_op_module


class BaseEmitter:
    """Base emitter class with shared graph-walking, env management, and
    expression helpers.  Subclasses must implement the abstract methods to
    produce backend-specific code.
    """

    # Subclasses should override this to declare which backend they serve.
    BACKEND_NAME: str = ""

    def __init__(
        self,
        *,
        class_name: str,
        spec: dict[str, Any],
        symbols: dict[str, int | float | bool],
        op_map: dict[str, Any] | None = None,
    ) -> None:
        if not class_name.isidentifier():
            raise ValueError(f"Invalid class name: {class_name!r}")
        self.class_name = class_name
        self.spec = spec
        self.model = spec["model"]
        self.blocks = self.model.get("blocks", {})
        self.symbols = symbols
        self._counter = 0
        self._active_env: dict[str, str] = {}

        # op_map: optional backward-compatible PyTorch op map.
        # When None, callers should load the default PyTorch op map externally
        # and pass it in, or the backend-specific subclass may load its own.
        self._op_map = op_map

    @property
    def op_map(self) -> dict[str, Any] | None:
        """Return the backward-compatible op map (if any)."""
        return self._op_map

    # ------------------------------------------------------------------
    # Name mangling helpers
    # ------------------------------------------------------------------

    def _fresh(self, base: str) -> str:
        """Return a unique variable name based on *base*."""
        self._counter += 1
        return f"{base}_{self._counter}"

    def _py_name(self, value: str) -> str:
        """Sanitise a synapse identifier into a valid Python name."""
        name = re.sub(r"[^0-9A-Za-z_]", "_", value)
        if not name:
            name = "v"
        if name[0].isdigit():
            name = f"v_{name}"
        return name

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------

    def _assign_out_var(self, env: dict[str, str], out_name: str) -> str:
        """Return the Python variable for *out_name*, creating a fresh one if
        it does not already exist in *env*."""
        existing = env.get(out_name)
        if isinstance(existing, str):
            return existing
        out_var = self._fresh(self._py_name(out_name))
        env[out_name] = out_var
        return out_var

    def _read_env_var(self, env: dict[str, str], name: str) -> str:
        """Read a required variable from *env*."""
        if name not in env:
            raise ValueError(f"Unknown input var {name!r}")
        return env[name]

    def _infer_param_expr(
        self, node_spec: dict[str, Any], node_path_var: str, param_name: str
    ) -> str:
        """Infer the Python expression that resolves a parameter path at
        runtime, mirroring the logic in ``runtime._infer_param_path``."""
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
            if "." in candidate:
                return f"self._join_scope(self._scope_of({node_path_var}), {candidate!r})"
        return f"self._join_scope({node_path_var}, {param_name!r})"

    # ------------------------------------------------------------------
    # Expression helpers
    # ------------------------------------------------------------------

    def _expr_code(self, expr: Any, env: dict[str, str]) -> str:
        """Convert a synapse expression to a Python code string."""
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
        """Replace identifier tokens in *text* with their Python variable
        equivalents from *env* and *symbols*."""
        rewritten = text
        for name, py_name in sorted(env.items(), key=lambda kv: len(kv[0]), reverse=True):
            rewritten = re.sub(rf"\b{re.escape(name)}\b", py_name, rewritten)
        for name, value in sorted(self.symbols.items(), key=lambda kv: len(kv[0]), reverse=True):
            rewritten = re.sub(rf"\b{re.escape(name)}\b", repr(value), rewritten)
        return rewritten

    def _try_eval_numeric(self, text: str) -> int | float | None:
        """Attempt to evaluate *text* as a safe numeric expression using
        symbol values."""
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

    # ------------------------------------------------------------------
    # Node output helpers
    # ------------------------------------------------------------------

    def _node_output_names(self, node_spec: dict[str, Any]) -> list[str]:
        """Return the list of output names bound by a node spec."""
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

    # ------------------------------------------------------------------
    # Graph compilation
    # ------------------------------------------------------------------

    def _compile_graph(
        self, *, graph: list[Any], env: dict[str, str], scope_var: str, indent: str
    ) -> list[str]:
        """Walk a synapse graph and emit code lines.  Delegates to
        backend-specific ``_compile_op`` and ``_compile_block_call`` for op
        and block nodes."""
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
        return lines

    def _compile_block_call(
        self, *, node_spec: dict[str, Any], env: dict[str, str], scope_var: str, indent: str
    ) -> list[str]:
        """Emit code for a ``call`` control node."""
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

        call_args = ", ".join([*arg_codes, f"scope={scope_var}"])
        if len(tmp_vars) == 1:
            call_line = (
                f"{indent}{tmp_vars[0]} = self._block_{self._py_name(block_name)}({call_args})"
            )
        else:
            call_line = f"{indent}{', '.join(tmp_vars)} = self._block_{self._py_name(block_name)}({call_args})"

        lines = [call_line]
        for dst_name, tmp in zip(binds, tmp_vars, strict=True):
            existing = env.get(dst_name)
            dst_var = (
                existing if isinstance(existing, str) else self._fresh(self._py_name(dst_name))
            )
            lines.append(f"{indent}{dst_var} = {tmp}")
            env[dst_name] = dst_var
        return lines

    # ------------------------------------------------------------------
    # Op dispatch — delegates to backend-specific op_module
    # ------------------------------------------------------------------

    def _op_uses_node_path(self, op: str, node_spec: dict[str, Any]) -> bool:
        """Check whether an op requires the full node path.  Delegates to the
        op module's ``uses_node_path``."""
        op_module = get_op_module(op)
        if op_module is None:
            raise NotImplementedError(
                f"Unsupported op backend for: {op}"
            )
        return bool(op_module.uses_node_path(self, node_spec))

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
        """Compile a single op node.  Delegates to the backend-specific
        op module's ``compile`` function."""
        op_module = get_op_module(op)
        if op_module is None:
            raise NotImplementedError(
                f"Unsupported op backend for: {op}"
            )
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

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must override
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Render the full model class source code.

        Subclasses must implement this to produce backend-specific output
        including imports, class template, block methods, forward, and
        generate methods.
        """
        raise NotImplementedError(
            f"Backend {type(self).__name__!r} must implement render()"
        )

    def _render_block_method(self, block_name: str, block_spec: Any) -> list[str]:
        """Render a single block method."""
        raise NotImplementedError(
            f"Backend {type(self).__name__!r} must implement _render_block_method()"
        )

    def _render_forward(self) -> list[str]:
        """Render the forward method."""
        raise NotImplementedError(
            f"Backend {type(self).__name__!r} must implement _render_forward()"
        )

    def _render_generate(self) -> list[str]:
        """Render the generate method."""
        raise NotImplementedError(
            f"Backend {type(self).__name__!r} must implement _render_generate()"
        )
