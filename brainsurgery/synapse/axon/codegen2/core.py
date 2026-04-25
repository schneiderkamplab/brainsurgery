from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ...mxfp4 import materialize_mxfp4_aliases
from ...ops import get_op_module
from ...runtime import SynapseProgramModel
from ...codegen import emit_model_code_from_synapse_spec
from ..ast import (
    AxonExprPath,
    TypeList,
    TypeNamed,
    TypeOptional,
    TypeTensor,
    TypeTuple,
    render_type,
)
from ..ast.types import dim_token_names
from ..ast.path import path_expr_to_runtime_value, resolve_path_expr_to_key
from ..graph_ir import (
    GraphLiteral,
    GraphExpr,
    GraphModule,
    GraphOperand,
    GraphPath,
    GraphProgram,
    GraphValueRef,
    validate_graph_program,
)


def _graph_path_payload(path: GraphPath) -> dict[str, Any]:
    return path_expr_to_runtime_value(
        AxonExprPath(absolute=path.absolute, parts=path.parts)
    )


def _operand_payload(operand: GraphOperand) -> Any:
    if isinstance(operand, GraphValueRef):
        return operand.name
    if isinstance(operand, GraphLiteral):
        return operand.value
    if isinstance(operand, GraphPath):
        return _graph_path_payload(operand)
    if isinstance(operand, GraphExpr):
        return _graph_expr_payload(operand)
    if isinstance(operand, tuple):
        return [_operand_payload(item) for item in operand]
    raise TypeError(f"unsupported graph operand: {operand!r}")


def _normalize_primitive_op(name: str) -> str:
    if name.startswith("_activations_"):
        return name[1:]
    if name.startswith("_"):
        return name[1:]
    return name


def _module_inputs_spec(module: GraphModule) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for value in module.inputs:
        type_expr = value.type_expr
        optional = value.optional or isinstance(type_expr, TypeOptional)
        out[value.name] = {
            "type": render_type(type_expr),
            "optional": optional,
        }
    return out


def _module_outputs_spec(module: GraphModule) -> dict[str, Any]:
    out: dict[str, Any] = {}
    names = module.output_names or _fallback_output_names(module)
    if not names:
        names = tuple(f"out{idx}" for idx in range(len(module.outputs)))
    for name, operand in zip(names, module.outputs, strict=False):
        out[name] = _operand_payload(operand)
    return out


def _fallback_output_names(module: GraphModule) -> tuple[str, ...]:
    names: list[str] = []
    for idx, operand in enumerate(module.outputs):
        if isinstance(operand, GraphValueRef):
            names.append(operand.name)
        else:
            names.append("logits" if len(module.outputs) == 1 and idx == 0 else f"out{idx}")
    return tuple(names)


def _type_dim_names(type_expr: Any) -> set[str]:
    if isinstance(type_expr, TypeTensor):
        names: set[str] = set()
        for dim in type_expr.dims:
            names.update(dim_token_names(dim))
        return names
    if isinstance(type_expr, TypeOptional):
        return _type_dim_names(type_expr.inner)
    if isinstance(type_expr, TypeList):
        return _type_dim_names(type_expr.item)
    if isinstance(type_expr, TypeTuple):
        names: set[str] = set()
        for item in type_expr.items:
            names.update(_type_dim_names(item))
        return names
    if isinstance(type_expr, TypeNamed):
        names: set[str] = set()
        for dim in type_expr.args:
            names.update(dim_token_names(dim))
        return names
    return set()


class Codegen2GraphModel(SynapseProgramModel):
    """Execute the typed Axon graph IR without Synapse graph specs."""

    GRAPH: GraphProgram

    def __init__(
        self,
        graph: GraphProgram | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
        *,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        graph = self.GRAPH if graph is None else graph
        validate_graph_program(graph)
        self.graph = graph
        self.modules_by_name = {module.name: module for module in graph.modules}
        main = self.modules_by_name[graph.main_module]
        self.main_inputs_spec = _module_inputs_spec(main)
        self.main_outputs_spec = _module_outputs_spec(main)
        spec = {
            "synapse": 1,
            "model": {
                "inputs": self.main_inputs_spec,
                "outputs": self.main_outputs_spec,
                "graph": [],
                "blocks": {},
                "symbols": {},
                "config": model_config or {},
            },
        }
        super().__init__(spec=spec, state_dict=state_dict)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        graph: GraphProgram | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> "Codegen2GraphModel":
        return cls(graph=graph, state_dict=state_dict, model_config=model_config)

    def load_state_dict_tensors(self, state_dict: dict[str, torch.Tensor]) -> None:
        loaded = dict(state_dict)
        materialize_mxfp4_aliases(loaded, drop_packed=True)
        self._state = loaded

    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:
        env = self._prepare_env(
            input_ids=input_ids,
            inputs=inputs,
            input_specs=self.main_inputs_spec,
        )
        symbols = self._evaluate_constants(env)
        self._symbols = symbols
        self._bind_shape_symbols_from_types(
            env=env,
            input_types={
                value.name: render_type(value.type_expr)
                for value in self.modules_by_name[self.graph.main_module].inputs
            },
            symbols=symbols,
        )
        result = self._execute_module(self.graph.main_module, env, symbols)
        names = self.modules_by_name[self.graph.main_module].output_names
        if not names:
            names = _fallback_output_names(self.modules_by_name[self.graph.main_module])
        outputs = {name: value for name, value in zip(names, result, strict=False)}
        if "logits" in outputs and len(outputs) == 1:
            return outputs["logits"]
        return outputs

    def _evaluate_constants(self, env: dict[str, Any]) -> dict[str, Any]:
        symbols: dict[str, Any] = {}
        for node in self.graph.constant_nodes:
            self._execute_node(node, env=symbols, symbols=symbols)
        for name, operand in self.graph.constants.items():
            symbols[name] = self._eval_graph_operand(
                operand,
                env={**symbols, **env},
                symbols=symbols,
            )
        return symbols

    def _eval_graph_operand(
        self,
        operand: GraphOperand,
        *,
        env: Mapping[str, Any],
        symbols: Mapping[str, Any],
    ) -> Any:
        if isinstance(operand, GraphValueRef):
            if operand.name in env:
                value = env[operand.name]
                if (
                    operand.type_expr.__class__.__name__ == "TypeDim"
                    and isinstance(value, list | tuple)
                    and value
                ):
                    return value[-1]
                return value
            if operand.name in symbols:
                value = symbols[operand.name]
                if (
                    operand.type_expr.__class__.__name__ == "TypeDim"
                    and isinstance(value, list | tuple)
                    and value
                ):
                    return value[-1]
                return value
            raise ValueError(f"undefined graph value {operand.name!r}")
        if isinstance(operand, GraphLiteral):
            return operand.value
        if isinstance(operand, GraphPath):
            return _graph_path_payload(operand)
        if isinstance(operand, GraphExpr):
            return self._eval_graph_expr(operand, env=env, symbols=symbols)
        if isinstance(operand, tuple):
            return [self._eval_graph_operand(item, env=env, symbols=symbols) for item in operand]
        raise TypeError(f"unsupported graph operand: {operand!r}")

    def _eval_graph_expr(
        self,
        expr: GraphExpr,
        *,
        env: Mapping[str, Any],
        symbols: dict[str, Any],
    ) -> Any:
        scratch = dict(env)
        out_name = "__expr"
        pseudo_node = type(
            "_PseudoGraphNode",
            (),
            {
                "op": expr.op,
                "inputs": expr.inputs,
                "attrs": expr.attrs,
                "outputs": [type("_PseudoGraphValue", (), {"name": out_name})()],
            },
        )()
        self._execute_node(pseudo_node, env=scratch, symbols=symbols)
        return scratch[out_name]

    def _eval_expr(self, expr: Any, env: dict[str, Any], symbols: dict[str, Any]) -> Any:
        value = super()._eval_expr(expr, env, symbols)
        if isinstance(expr, str) and isinstance(value, list | tuple) and value:
            return value[-1]
        if (
            isinstance(expr, dict)
            and expr.get("_expr") == "name"
            and isinstance(value, list | tuple)
            and value
        ):
            return value[-1]
        return value

    def _operand_arg(self, operand: GraphOperand) -> Any:
        return _operand_payload(operand)

    def _execute_module(
        self,
        module_name: str,
        env: dict[str, Any],
        symbols: dict[str, Any],
    ) -> tuple[Any, ...]:
        module = self.modules_by_name[module_name]
        local_env = dict(env)
        self._bind_shape_symbols_from_types(
            env=local_env,
            input_types={value.name: render_type(value.type_expr) for value in module.inputs},
            symbols=symbols,
        )
        dim_names: set[str] = set()
        for value in module.inputs:
            dim_names.update(_type_dim_names(value.type_expr))
            for dim in value.dims or ():
                dim_names.update(dim_token_names(dim))
        for name in dim_names:
            value = local_env.get(name)
            if isinstance(value, list | tuple) and value:
                local_env[name] = value[-1]
            value = symbols.get(name)
            if isinstance(value, list | tuple) and value:
                symbols[name] = value[-1]
        for node in module.nodes:
            self._execute_node(node, env=local_env, symbols=symbols)
        return tuple(
            self._eval_graph_operand(operand, env=local_env, symbols=symbols)
            for operand in module.outputs
        )

    def _execute_node(
        self,
        node: Any,
        *,
        env: dict[str, Any],
        symbols: dict[str, Any],
    ) -> None:
        op = node.op.name
        out_names = tuple(value.name for value in node.outputs)
        if op == "core.alias":
            values = [
                self._eval_graph_operand(operand, env=env, symbols=symbols)
                for operand in node.inputs
            ]
            self._assign_outputs(out_names, values[0] if len(values) == 1 else tuple(values), env)
            return
        if op == "core.tuple":
            self._assign_outputs(
                out_names,
                tuple(
                    self._eval_graph_operand(operand, env=env, symbols=symbols)
                    for operand in node.inputs
                ),
                env,
            )
            return
        if op == "core.list":
            self._assign_outputs(
                out_names,
                [
                    self._eval_graph_operand(operand, env=env, symbols=symbols)
                    for operand in node.inputs
                ],
                env,
            )
            return
        if op == "core.select":
            cond, true_value, false_value = node.inputs
            selected = true_value if bool(
                self._eval_graph_operand(cond, env=env, symbols=symbols)
            ) else false_value
            self._assign_outputs(
                out_names,
                self._eval_graph_operand(selected, env=env, symbols=symbols),
                env,
            )
            return
        if op.startswith("core.binary."):
            operator = op.removeprefix("core.binary.")
            left = self._eval_graph_operand(node.inputs[0], env=env, symbols=symbols)
            right = self._eval_graph_operand(node.inputs[1], env=env, symbols=symbols)
            self._assign_outputs(out_names, self._eval_binary(operator, left, right), env)
            return
        if op == "Config.dim":
            self._execute_config_dim(node, env=env, symbols=symbols)
            return
        if op in self.modules_by_name:
            args = [
                self._eval_graph_operand(operand, env=env, symbols=symbols)
                for operand in node.inputs
            ]
            callee = self.modules_by_name[op]
            call_env = {
                value.name: arg for value, arg in zip(callee.inputs, args, strict=False)
            }
            result = self._execute_module(op, call_env, symbols)
            self._assign_outputs(out_names, result[0] if len(result) == 1 else result, env)
            return

        if op == "_sqrt":
            value = self._eval_graph_operand(node.inputs[0], env=env, symbols=symbols)
            if isinstance(value, list | tuple) and value:
                value = value[-1]
            self._assign_outputs(out_names, torch.sqrt(torch.tensor(float(value))).item(), env)
            return
        if op == "_reshape":
            src = self._eval_graph_operand(node.inputs[0], env=env, symbols=symbols)
            shape = self._eval_graph_operand(node.inputs[1], env=env, symbols=symbols)
            if not isinstance(shape, list | tuple) or not shape:
                raise ValueError("reshape.shape must be a non-empty list")
            self._assign_outputs(
                out_names,
                torch.reshape(src, tuple(int(item) for item in shape)),
                env,
            )
            return

        primitive = _normalize_primitive_op(op)
        op_module = get_op_module(primitive)
        if op_module is None:
            raise NotImplementedError(f"codegen2 unsupported graph op {op!r}")
        path_args = [operand for operand in node.inputs if isinstance(operand, GraphPath)]
        runtime_inputs = tuple(
            operand for operand in node.inputs if not isinstance(operand, GraphPath)
        )
        node_spec = {
            "_op": primitive,
            "_args": [self._operand_arg(operand) for operand in runtime_inputs],
            "_bind": list(out_names) if len(out_names) != 1 else out_names[0],
        }
        if primitive in {"zeros_like"} and len(node_spec["_args"]) == 1:
            node_spec["_args"] = node_spec["_args"][0]
        for key, value in node.attrs.items():
            node_spec[key] = self._operand_arg(value)
        if path_args:
            node_spec["_abs_path"] = _graph_path_payload(path_args[0])
        op_module.interpret(self, node_spec, env, node_path="", scope="", symbols=symbols)

    def _execute_config_dim(
        self,
        node: Any,
        *,
        env: dict[str, Any],
        symbols: dict[str, Any],
    ) -> None:
        del env
        if not node.inputs or not isinstance(node.inputs[0], GraphPath):
            raise ValueError("Config.dim expects a Path key")
        key = resolve_path_expr_to_key(
            _graph_path_payload(node.inputs[0]),
            {},
            op_name="Config.dim",
        )
        config = self.spec.get("model", {}).get("config", {})
        value = config.get(key) if isinstance(config, dict) else None
        if value is None:
            default = node.attrs.get("default")
            value = self._eval_graph_operand(default, env=symbols, symbols=symbols)
        self._assign_outputs(tuple(output.name for output in node.outputs), int(value), symbols)

    def _assign_outputs(
        self,
        names: tuple[str, ...],
        value: Any,
        env: dict[str, Any],
    ) -> None:
        if len(names) == 1:
            env[names[0]] = value
            return
        if not isinstance(value, (tuple, list)) or len(value) != len(names):
            raise ValueError(f"cannot assign {value!r} to outputs {names!r}")
        for name, item in zip(names, value, strict=True):
            env[name] = item

    @staticmethod
    def _eval_binary(op: str, left: Any, right: Any) -> Any:
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
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
        raise NotImplementedError(f"unsupported codegen2 binary op {op!r}")


def emit_model_code_from_graph_ir(
    program: GraphProgram,
    *,
    class_name: str = "GeneratedAxonModel",
    model_config: dict[str, Any] | None = None,
) -> str:
    """Emit Python model code from graph IR."""
    validate_graph_program(program)
    codegen_spec = graph_ir_to_codegen_spec(program, model_config=model_config)
    return emit_model_code_from_synapse_spec(codegen_spec, class_name=class_name)


def _type_spec(value: Any) -> dict[str, Any]:
    return {
        "type": render_type(value.type_expr),
        "optional": bool(value.optional or isinstance(value.type_expr, TypeOptional)),
    }


def _output_ref(operand: GraphOperand, *, name: str) -> Any:
    if isinstance(operand, GraphValueRef):
        return operand.name
    raise ValueError(f"codegen2 output {name!r} must be a value ref, got {operand!r}")


def _node_bind(node: Any) -> str | list[str]:
    names = [value.name for value in node.outputs]
    if len(names) == 1:
        return names[0]
    return names


def _graph_expr_payload(expr: GraphExpr) -> Any:
    op = expr.op.name
    if op == "core.list":
        return [_operand_payload(item) for item in expr.inputs]
    if op == "core.tuple":
        return {
            "_expr": "tuple",
            "items": [_operand_payload(item) for item in expr.inputs],
        }
    if op == "core.select":
        return {
            "_expr": "if",
            "cond": _operand_payload(expr.inputs[0]),
            "then": _operand_payload(expr.inputs[1]),
            "else": _operand_payload(expr.inputs[2]),
        }
    if op.startswith("core.binary."):
        return {
            "_expr": "binary",
            "op": op.removeprefix("core.binary."),
            "left": _operand_payload(expr.inputs[0]),
            "right": _operand_payload(expr.inputs[1]),
        }
    return {
        "_expr": "call",
        "callee": op,
        "args": [_operand_payload(item) for item in expr.inputs],
        "kwargs": {key: _operand_payload(value) for key, value in expr.attrs.items()},
    }


def _graph_node_to_codegen_node(
    node: Any,
    *,
    module_names: set[str],
) -> dict[str, Any]:
    op = node.op.name
    bind = _node_bind(node)
    if op == "core.alias":
        if len(node.inputs) != 1:
            raise ValueError("core.alias must have one input")
        return {"_op": "_ir_expr", "value": _operand_payload(node.inputs[0]), "_bind": bind}
    if op in {"core.list", "core.tuple"}:
        value: Any
        if op == "core.list":
            value = [_operand_payload(item) for item in node.inputs]
        else:
            value = {"_expr": "tuple", "items": [_operand_payload(item) for item in node.inputs]}
        return {"_op": "_ir_expr", "value": value, "_bind": bind}
    if op == "core.select":
        return {
            "_op": "_ir_expr",
            "value": _graph_expr_payload(
                GraphExpr(
                    op=node.op,
                    inputs=node.inputs,
                    attrs=node.attrs,
                    type_expr=node.outputs[0].type_expr,
                    dims=node.outputs[0].dims,
                )
            ),
            "_bind": bind,
        }
    if op.startswith("core.binary."):
        return {
            "_op": "_ir_expr",
            "value": {
                "_expr": "binary",
                "op": op.removeprefix("core.binary."),
                "left": _operand_payload(node.inputs[0]),
                "right": _operand_payload(node.inputs[1]),
            },
            "_bind": bind,
        }
    if op in module_names:
        return {
            "_op": "call",
            "_target": op,
            "_args": [_operand_payload(item) for item in node.inputs],
            "_bind": bind,
            "_out_types": {value.name: render_type(value.type_expr) for value in node.outputs},
            **{key: _operand_payload(value) for key, value in node.attrs.items()},
        }

    primitive = _normalize_primitive_op(op)
    path_args = [operand for operand in node.inputs if isinstance(operand, GraphPath)]
    runtime_inputs = [operand for operand in node.inputs if not isinstance(operand, GraphPath)]
    runtime_args = [_operand_payload(operand) for operand in runtime_inputs]
    node_spec: dict[str, Any] = {
        "_op": primitive,
        "_args": runtime_args[0] if len(runtime_args) == 1 else runtime_args,
        "_bind": bind,
        "_out_types": {value.name: render_type(value.type_expr) for value in node.outputs},
    }
    if path_args:
        node_spec["_abs_path"] = _graph_path_payload(path_args[0])
    for key, value in node.attrs.items():
        node_spec[key] = _operand_payload(value)
    return node_spec


def _module_graph_to_codegen_graph(
    module: GraphModule,
    *,
    module_names: set[str],
) -> list[dict[str, dict[str, Any]]]:
    graph: list[dict[str, dict[str, Any]]] = []
    for index, node in enumerate(module.nodes, start=1):
        name = f"g{index}"
        graph.append({name: _graph_node_to_codegen_node(node, module_names=module_names)})
    return graph


def graph_ir_to_codegen_spec(
    program: GraphProgram,
    *,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_graph_program(program)
    evaluated_symbols = _evaluate_static_symbols(program, model_config=model_config)
    modules_by_name = {module.name: module for module in program.modules}
    if program.main_module not in modules_by_name:
        raise ValueError(f"unknown graph IR main module {program.main_module!r}")
    module_names = set(modules_by_name)
    main = modules_by_name[program.main_module]
    main_output_names = main.output_names or _fallback_output_names(main)
    block_io: dict[str, dict[str, dict[str, str]]] = {
        "main": {
            "inputs": {value.name: render_type(value.type_expr) for value in main.inputs},
            "outputs": {
                name: render_type(operand.type_expr)
                for name, operand in zip(main_output_names, main.outputs, strict=False)
                if isinstance(operand, GraphValueRef)
            },
        }
    }
    spec: dict[str, Any] = {
        "synapse": 1,
        "model": {
            "inputs": {value.name: _type_spec(value) for value in main.inputs},
            "outputs": {
                name: _output_ref(operand, name=name)
                for name, operand in zip(main_output_names, main.outputs, strict=False)
            },
            "graph": _module_graph_to_codegen_graph(main, module_names=module_names),
            "blocks": {},
            "symbols": evaluated_symbols,
            "types": {"block_io": block_io},
            "config": model_config or {},
        },
    }
    blocks: dict[str, Any] = {}
    for module in program.modules:
        if module.name == program.main_module:
            continue
        output_names = module.output_names or _fallback_output_names(module)
        blocks[module.name] = {
            "inputs": {value.name: _type_spec(value) for value in module.inputs},
            "outputs": {
                name: _output_ref(operand, name=name)
                for name, operand in zip(output_names, module.outputs, strict=False)
            },
            "graph": _module_graph_to_codegen_graph(module, module_names=module_names),
        }
        block_io[module.name] = {
            "inputs": {value.name: render_type(value.type_expr) for value in module.inputs},
            "outputs": {
                name: render_type(operand.type_expr)
                for name, operand in zip(output_names, module.outputs, strict=False)
                if isinstance(operand, GraphValueRef)
            },
        }
    spec["model"]["blocks"] = blocks
    return spec


def _evaluate_static_symbols(
    program: GraphProgram,
    *,
    model_config: dict[str, Any] | None,
) -> dict[str, Any]:
    model = Codegen2GraphModel(program, model_config=model_config)
    try:
        return dict(model._evaluate_constants({}))
    except Exception:
        return {
            name: _operand_payload(operand)
            for name, operand in program.constants.items()
            if not (isinstance(operand, GraphValueRef) and operand.name == name)
        }


def make_graph_model_class(
    program: GraphProgram,
    *,
    model_config: dict[str, Any] | None = None,
) -> type[Codegen2GraphModel]:
    validate_graph_program(program)

    class GeneratedCodegen2GraphModel(Codegen2GraphModel):
        GRAPH = program

        @classmethod
        def from_state_dict(
            cls,
            state_dict: dict[str, torch.Tensor],
            *,
            graph: GraphProgram | None = None,
            model_config: dict[str, Any] | None = None,
        ) -> "GeneratedCodegen2GraphModel":
            return cls(
                graph=program if graph is None else graph,
                state_dict=state_dict,
                model_config=model_config if model_config is not None else captured_model_config,
            )

    GeneratedCodegen2GraphModel.__name__ = "GeneratedCodegen2GraphModel"
    captured_model_config = model_config
    GeneratedCodegen2GraphModel._codegen2_model_config = captured_model_config
    return GeneratedCodegen2GraphModel


Runtime2GraphModel = Codegen2GraphModel


def make_runtime2_model_class(
    program: GraphProgram,
    *,
    model_config: dict[str, Any] | None = None,
) -> type[Codegen2GraphModel]:
    return make_graph_model_class(program, model_config=model_config)


__all__ = [
    "Codegen2GraphModel",
    "Runtime2GraphModel",
    "emit_model_code_from_graph_ir",
    "graph_ir_to_codegen_spec",
    "make_graph_model_class",
    "make_runtime2_model_class",
]
