from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..ops import (
    get_op_lowering_infer_metadata,
    get_op_lowering_known_output_arity,
    get_op_lowering_normalizer,
    get_op_lowering_signature,
    get_op_lowering_type_signature,
    get_op_lowering_validator,
)
from ..type_inference import annotate_spec_with_block_io_types, infer_block_io_types_from_modules
from .ast import (
    AxonBind,
    AxonExpr,
    AxonExprAscribe,
    AxonExprBinary,
    AxonExprBind,
    AxonExprBool,
    AxonExprCall,
    AxonExprDo,
    AxonExprFloat,
    AxonExprIf,
    AxonExprInt,
    AxonExprLambda,
    AxonExprList,
    AxonExprName,
    AxonExprNull,
    AxonExprParen,
    AxonExprPath,
    AxonExprPipe,
    AxonExprString,
    AxonExprTernary,
    AxonExprTuple,
    AxonFile,
    AxonKwargValue,
    AxonModule,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
    AxonYield,
    TypeExpr,
    TypeList,
    TypeNamed,
    TypeOptional,
    TypeTensor,
    TypeTuple,
    dim_token_names,
    render_type,
)
from .expression_codec import axon_expr_to_runtime_value as _expr_to_runtime_value
from .ast.path import parse_path_token, path_expr_to_runtime_value
from .canonicalize import canonicalize_typed_axon_file
from .flatten import flatten_closed_axon_file
from .optimize import optimize_flat_typed_axon_file
from .resolve import resolve_axon_program_from_path
from .signatures import ModuleSignature, _build_module_signatures_for_closed_program
from .typecheck import typecheck_flat_axon_file
from .validate.closed import validate_closed_axon_file
from .validate.flat import validate_flat_axon_file
from .validate.typed import validate_typed_axon_file


def _is_identifier(token: str) -> bool:
    if not token:
        return False
    if not (token[0].isalpha() or token[0] == "_"):
        return False
    return all(ch.isalnum() or ch == "_" for ch in token[1:])


def _is_name_token(token: str) -> bool:
    return _is_identifier(token.strip())


def _path_expr_to_source(expr: AxonExprPath) -> str:
    return expr.to_source()


def _path_source_to_runtime_value(source: str) -> dict[str, Any]:
    return path_expr_to_runtime_value(parse_path_token(source, op_name="lowered path"))


def _expr_name(expr: AxonExpr) -> str | None:
    if isinstance(expr, AxonExprName):
        return expr.name
    return None


def _is_int_literal(token: str) -> bool:
    if not token:
        return False
    if token[0] == "-":
        return token[1:].isdigit()
    return token.isdigit()


def _sanitize_token(token: str, *, default: str) -> str:
    out_chars: list[str] = []
    for ch in token:
        if ch.isalnum() or ch == "_":
            out_chars.append(ch)
        else:
            out_chars.append("_")
    normalized = "".join(out_chars).strip("_")
    return normalized or default


def _parse_scalar_token(token: str) -> Any:
    value = token.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null":
        return None
    if value and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        return value[1:-1]
    if value and (value.isdigit() or (value[0] in {"+", "-"} and value[1:].isdigit())):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


_NO_CONST = object()

_CANONICAL_PATH_ARG_PRIMITIVES = frozenset(
    {"_embedding", "_linear", "_layernorm", "_activations_xielu"}
)


def _is_static_literal(value: Any) -> bool:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return True
    if isinstance(value, list):
        return all(_is_static_literal(item) for item in value)
    if isinstance(value, tuple):
        return all(_is_static_literal(item) for item in value)
    return False


def _surface_modules_from_file(ast: AxonFile) -> tuple[AxonModule, ...]:
    modules: list[AxonModule] = []
    for module in ast.modules:
        if isinstance(module.body_expr, AxonExprDo) and not module.body_expr.inline:
            statements = module.body_expr.body
        elif isinstance(module.body_expr, AxonExpr):
            statements = (AxonReturn(values=(module.body_expr,)),)
        else:
            statements = module.statements
        modules.append(
            AxonModule(
                name=module.name,
                path_param=module.path_param,
                path_params=module.path_params,
                params=module.params,
                returns=module.returns,
                statements=statements,
                body_expr=None,
                imports=ast.imports,
                imported_members=dict(ast.imported_members) or None,
                exports=ast.exports,
                symbols=module.symbols,
                pragmas=module.pragmas,
                type_aliases=dict(ast.type_aliases) or module.type_aliases,
                return_type_expr=module.return_type_expr,
                return_shape=module.return_shape,
                constraints=module.constraints,
            )
        )
    return tuple(modules)


def _wrap_modules_as_file(modules: tuple[AxonModule, ...]) -> AxonFile:
    return AxonFile(
        modules=modules,
        imports=(),
        imported_members={},
        exports=(),
        pragmas={},
        constants={},
        type_aliases={},
    )


def _lower_program_symbol_values(program: AxonFile) -> dict[str, Any]:
    lowered: dict[str, Any] = {}
    for key, value in program.constants.items():
        if isinstance(key, str) and isinstance(value, AxonExpr):
            lowered[key] = _expr_to_runtime_value(value)
    return lowered


def _prepare_program_for_lowering(
    modules: AxonFile | tuple[AxonModule, ...],
    *,
    main_module: str | None,
    optimize: bool = True,
) -> AxonFile:
    def finish_typed(program: AxonFile) -> AxonFile:
        if optimize:
            program = optimize_flat_typed_axon_file(program, main_module=main_module)
        return canonicalize_typed_axon_file(program, main_module=main_module)

    program = modules if isinstance(modules, AxonFile) else _wrap_modules_as_file(modules)
    try:
        validate_typed_axon_file(program, main_module=main_module)
        return finish_typed(program)
    except Exception:
        pass
    try:
        validate_flat_axon_file(program, main_module=main_module)
        typed = typecheck_flat_axon_file(program, main_module=main_module)
        return finish_typed(typed)
    except Exception:
        pass
    try:
        validate_closed_axon_file(program, main_module=main_module)
        flat = flatten_closed_axon_file(program, main_module=main_module)
        typed = typecheck_flat_axon_file(flat, main_module=main_module)
        return finish_typed(typed)
    except Exception:
        pass
    if program.origin_path is None:
        raise ValueError(
            "lowering requires a closed Axon program or an AxonFile with origin_path for resolve"
        )
    resolved = resolve_axon_program_from_path(program.origin_path).ast
    flat = flatten_closed_axon_file(resolved, main_module=main_module)
    typed = typecheck_flat_axon_file(flat, main_module=main_module)
    return finish_typed(typed)


def _const_fold_expr(expr: AxonExpr, ctx: "_LowerCtx") -> Any:
    if isinstance(expr, AxonExprParen):
        return _const_fold_expr(expr.inner, ctx)
    if isinstance(expr, AxonExprInt):
        return expr.value
    if isinstance(expr, AxonExprFloat):
        return expr.value
    if isinstance(expr, AxonExprBool):
        return expr.value
    if isinstance(expr, AxonExprNull):
        return None
    if isinstance(expr, AxonExprString):
        return expr.value
    if isinstance(expr, AxonExprPath):
        return _path_expr_to_source(expr)
    if isinstance(expr, AxonExprName):
        if expr.name in ctx.runtime_bound_symbol_names:
            return _NO_CONST
        value = ctx.symbol_values.get(expr.name, _NO_CONST)
        if value is None and expr.name in ctx.symbol_names:
            return _NO_CONST
        if value is _NO_CONST or not _is_static_literal(value):
            return _NO_CONST
        return value
    if isinstance(expr, AxonExprList):
        list_items: list[Any] = []
        for item in expr.items:
            folded = _const_fold_expr(item, ctx)
            if folded is _NO_CONST:
                return _NO_CONST
            list_items.append(folded)
        return list_items
    if isinstance(expr, AxonExprTuple):
        tuple_items: list[Any] = []
        for item in expr.items:
            folded = _const_fold_expr(item, ctx)
            if folded is _NO_CONST:
                return _NO_CONST
            tuple_items.append(folded)
        return tuple(tuple_items)
    if isinstance(expr, AxonExprIf):
        cond = _const_fold_expr(expr.cond, ctx)
        if not isinstance(cond, bool):
            return _NO_CONST
        return _const_fold_expr(expr.true_expr if cond else expr.false_expr, ctx)
    if isinstance(expr, AxonExprTernary):
        cond = _const_fold_expr(expr.cond, ctx)
        if not isinstance(cond, bool):
            return _NO_CONST
        return _const_fold_expr(expr.true_expr if cond else expr.false_expr, ctx)
    if isinstance(expr, AxonExprBinary):
        left = _const_fold_expr(expr.left, ctx)
        right = _const_fold_expr(expr.right, ctx)
        if left is _NO_CONST or right is _NO_CONST:
            return _NO_CONST
        try:
            if expr.op == "+":
                return left + right
            if expr.op == "-":
                return left - right
            if expr.op == "*":
                return left * right
            if expr.op == "/":
                return _NO_CONST
            if expr.op == "%":
                return left % right
            if expr.op == "==":
                return left == right
            if expr.op == "!=":
                return left != right
            if expr.op == "<":
                return left < right
            if expr.op == "<=":
                return left <= right
            if expr.op == ">":
                return left > right
            if expr.op == ">=":
                return left >= right
            if expr.op == "and":
                return bool(left) and bool(right)
            if expr.op == "or":
                return bool(left) or bool(right)
        except Exception:
            return _NO_CONST
        return _NO_CONST
    if isinstance(expr, AxonExprCall):
        if "." in expr.callee or "@" in expr.callee:
            return _NO_CONST
        if expr.kwargs:
            return _NO_CONST
        args: list[Any] = []
        for arg in expr.args:
            folded = _const_fold_expr(arg, ctx)
            if folded is _NO_CONST:
                return _NO_CONST
            args.append(folded)
        try:
            if expr.callee == "sqrt" and len(args) == 1:
                value = float(args[0])
                if value < 0:
                    return _NO_CONST
                return value**0.5
            if expr.callee == "int" and len(args) == 1:
                return int(args[0])
            if expr.callee == "float" and len(args) == 1:
                return float(args[0])
            if expr.callee == "bool" and len(args) == 1:
                return bool(args[0])
            if expr.callee == "str" and len(args) == 1:
                return str(args[0])
            if expr.callee == "abs" and len(args) == 1:
                return abs(args[0])
            if expr.callee == "min" and len(args) >= 1:
                return min(args)
            if expr.callee == "max" and len(args) >= 1:
                return max(args)
        except Exception:
            return _NO_CONST
    return _NO_CONST


def _const_fold_dim_expr(expr: AxonExpr, ctx: "_LowerCtx", *, allow_bare_name: bool = False) -> Any:
    if isinstance(expr, AxonExprParen):
        return _const_fold_dim_expr(expr.inner, ctx, allow_bare_name=allow_bare_name)
    if isinstance(expr, AxonExprInt):
        return int(expr.value)
    if isinstance(expr, AxonExprName):
        if expr.name in ctx.runtime_bound_symbol_names:
            return _NO_CONST
        value = ctx.symbol_values.get(expr.name, _NO_CONST)
        if value is _NO_CONST:
            return _NO_CONST
        if isinstance(value, bool):
            return _NO_CONST
        if isinstance(value, int):
            return int(value) if allow_bare_name else _NO_CONST
        if isinstance(value, float) and float(value).is_integer():
            return int(value) if allow_bare_name else _NO_CONST
        return _NO_CONST
    if isinstance(expr, AxonExprIf):
        cond = _const_fold_expr(expr.cond, ctx)
        if not isinstance(cond, bool):
            return _NO_CONST
        return _const_fold_dim_expr(
            expr.true_expr if cond else expr.false_expr,
            ctx,
            allow_bare_name=allow_bare_name,
        )
    if isinstance(expr, AxonExprTernary):
        cond = _const_fold_expr(expr.cond, ctx)
        if not isinstance(cond, bool):
            return _NO_CONST
        return _const_fold_dim_expr(
            expr.true_expr if cond else expr.false_expr,
            ctx,
            allow_bare_name=allow_bare_name,
        )
    if isinstance(expr, AxonExprBinary):
        left = _const_fold_dim_expr(expr.left, ctx, allow_bare_name=True)
        right = _const_fold_dim_expr(expr.right, ctx, allow_bare_name=True)
        if not isinstance(left, int) or not isinstance(right, int):
            return _NO_CONST
        try:
            if expr.op == "+":
                return left + right
            if expr.op == "-":
                return left - right
            if expr.op == "*":
                return left * right
            if expr.op == "/":
                if right == 0 or (left % right) != 0:
                    return _NO_CONST
                return left // right
            if expr.op == "%":
                if right == 0:
                    return _NO_CONST
                return left % right
        except Exception:
            return _NO_CONST
    return _NO_CONST


_IMPLICIT_ACTIVATION_ALIASES: dict[str, tuple[str, int]] = {
    "gelu": ("_activations_gelu", 0),
    "gelu_new": ("_activations_gelu_new", 0),
    "gelu_pytorch_tanh": ("_activations_gelu_pytorch_tanh", 0),
    "gegelu": ("_activations_gegelu", 0),
    "relu": ("_activations_relu", 0),
    "sigmoid": ("_activations_sigmoid", 0),
    "silu": ("_activations_silu", 0),
    "swiglu": ("_activations_swiglu", 0),
}
_PRIMITIVE_NAME_ALIASES: dict[str, str] = {
    "_list_init": "list_init",
    "_list_index": "list_index",
    "_list_append": "list_append",
}
_BUILTIN_MODULE_NAMESPACES: set[str] = {
    "SSM",
    "Activations",
    "Cache",
    "List",
    "MoE",
    "Config",
    "Params",
    "Positions",
    "Tensor",
    "Attention",
    "Math",
    "NN",
}


def _to_synapse_op(
    callee: str,
    args: list[str],
    kwargs: dict[str, Any],
    out: str | list[str],
) -> dict[str, Any]:
    canonical = _canonical_op_name(callee)
    if "@" in callee:
        at_op: dict[str, Any] = {"_op": canonical, "_bind": out}
        if args:
            at_op["_args"] = args[0] if len(args) == 1 else args
        for key, value in kwargs.items():
            at_op[key] = value
        return at_op

    default_op: dict[str, Any] = {"_op": canonical, "_bind": out}
    if args:
        default_op["_args"] = args[0] if len(args) == 1 else args
    for key, value in kwargs.items():
        default_op[key] = value
    return default_op


def _canonical_op_name(callee: str) -> str:
    base = callee.split("@", 1)[0] if "@" in callee else callee
    alias = _PRIMITIVE_NAME_ALIASES.get(base)
    if alias is not None:
        return alias
    if base.startswith("_") and len(base) > 1 and base[1].isalpha():
        return _canonical_op_name(base[1:])
    return base


def _normalize_dim_token(value: Any) -> Any:
    if isinstance(value, str):
        token = value.strip()
        if _is_int_literal(token):
            return int(token)
        return token
    return value


def _dims_compatible(left: Any, right: Any) -> bool:
    left_n = _normalize_dim_token(left)
    right_n = _normalize_dim_token(right)
    if left_n == right_n:
        return True
    # Lowering tracks many intermediate shape symbols (for example `pipe_4`)
    # that are unknown-but-consistent at compile time. Typecheck already
    # enforces stricter symbolic contracts; lowering should avoid false
    # negatives when both sides are unresolved symbolic dims.
    if isinstance(left_n, str) or isinstance(right_n, str):
        return True
    return False


def _is_symbolic_dim_token(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    token = value.strip()
    return not _is_int_literal(token)


def _is_variadic_shape_token(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith("..")


def _bind_shape_symbols(
    *,
    formal_shape: tuple[Any, ...],
    actual_shape: tuple[Any, ...],
    symbol_bindings: dict[str, Any],
) -> None:
    variadic_positions = [
        idx for idx, token in enumerate(formal_shape) if _is_variadic_shape_token(token)
    ]
    if not variadic_positions:
        if len(formal_shape) != len(actual_shape):
            return
        for sym, actual in zip(formal_shape, actual_shape, strict=True):
            if isinstance(sym, str) and sym not in symbol_bindings:
                symbol_bindings[sym] = actual
        return
    if len(variadic_positions) != 1:
        return
    idx = variadic_positions[0]
    prefix = formal_shape[:idx]
    suffix = formal_shape[idx + 1 :]
    if len(actual_shape) < len(prefix) + len(suffix):
        return
    for sym, actual in zip(prefix, actual_shape[: len(prefix)], strict=True):
        if isinstance(sym, str) and sym not in symbol_bindings:
            symbol_bindings[sym] = actual
    if suffix:
        suffix_actual = actual_shape[-len(suffix) :]
        for sym, actual in zip(suffix, suffix_actual, strict=True):
            if isinstance(sym, str) and sym not in symbol_bindings:
                symbol_bindings[sym] = actual
    captured = actual_shape[len(prefix) : len(actual_shape) - len(suffix) if suffix else len(actual_shape)]
    token = formal_shape[idx]
    if isinstance(token, str) and token not in symbol_bindings:
        symbol_bindings[token] = tuple(captured)


def _resolve_shape_tokens(shape_tokens: tuple[Any, ...], symbol_bindings: dict[str, Any]) -> tuple[Any, ...]:
    resolved: list[Any] = []
    for token in shape_tokens:
        value = symbol_bindings.get(token, token) if isinstance(token, str) else token
        if _is_variadic_shape_token(token) and isinstance(value, tuple):
            resolved.extend(value)
        else:
            resolved.append(value)
    return tuple(resolved)


def _is_kind(value: Any, kind: str) -> bool:
    is_expr_payload = isinstance(value, dict) and value.get("_expr") in {
        "name",
        "binary",
        "if",
        "tuple",
        "call",
        "string",
    }
    if kind == "bool":
        return (
            isinstance(value, bool)
            or (isinstance(value, str) and _is_identifier(value))
            or is_expr_payload
        )
    if kind == "int":
        return (
            (isinstance(value, int) and not isinstance(value, bool))
            or isinstance(value, str)
            or is_expr_payload
        )
    if kind == "number":
        if isinstance(value, bool):
            return False
        return isinstance(value, (int, float, str)) or is_expr_payload
    if kind == "str":
        return isinstance(value, str) or is_expr_payload
    if kind == "path":
        return isinstance(value, str) or (isinstance(value, dict) and value.get("_expr") == "path")
    if kind == "path_or_null":
        return (
            isinstance(value, str)
            or (isinstance(value, dict) and value.get("_expr") == "path")
            or value is None
        )
    if kind == "str_or_bool_or_null":
        return isinstance(value, (str, bool)) or value is None or is_expr_payload
    if kind == "dim":
        if isinstance(value, bool):
            return False
        return isinstance(value, (int, str)) or is_expr_payload
    if kind == "list_int":
        return (
            isinstance(value, list)
            and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
        ) or is_expr_payload
    if kind == "list_dim":
        return (
            isinstance(value, list)
            and all(
                (isinstance(v, int) and not isinstance(v, bool))
                or isinstance(v, str)
                or (
                    isinstance(v, dict)
                    and v.get("_expr") in {"name", "binary", "if", "tuple", "call", "string"}
                )
                for v in value
            )
        ) or is_expr_payload
    return True


def _validate_op_signature(op_name: str, args: list[str], kwargs: dict[str, Any]) -> None:
    signature = get_op_lowering_signature(op_name)
    if not isinstance(signature, dict):
        return
    arity = signature.get("arity")
    if arity is not None:
        min_args, max_args = arity
        if len(args) < min_args or len(args) > max_args:
            raise ValueError(
                f"{op_name} expects {min_args}"
                + (f"..{max_args}" if min_args != max_args else "")
                + f" positional args, got {len(args)}"
            )
    allowed = signature.get("allowed_kwargs")
    if allowed is not None:
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            allowed_text = ", ".join(sorted(str(name) for name in allowed))
            if allowed_text:
                raise ValueError(
                    f"{op_name} unsupported kwargs: {', '.join(unknown)}; allowed: {allowed_text}"
                )
            raise ValueError(f"{op_name} unsupported kwargs: {', '.join(unknown)}")
    required = signature.get("required_kwargs")
    if required:
        missing = sorted(required - set(kwargs))
        if missing:
            raise ValueError(f"{op_name} missing required kwargs: {', '.join(missing)}")
    kinds = signature.get("kwarg_kinds", {})
    for key, value in kwargs.items():
        expected = kinds.get(key)
        if expected is None:
            continue
        if not _is_kind(value, expected):
            raise ValueError(
                f"{op_name} kwarg {key!r} expects {expected}, got {type(value).__name__}"
            )


@dataclass
class _LowerCtx:
    counter: int = 0
    block_signatures: dict[str, tuple[list[str], list[str]]] | None = None
    block_path_params: dict[str, tuple[str, ...]] | None = None
    block_param_last_dims: dict[str, dict[str, Any]] | None = None
    block_output_last_dims: dict[str, dict[str, Any]] | None = None
    block_param_shapes: dict[str, dict[str, tuple[Any, ...]]] | None = None
    block_output_shapes: dict[str, dict[str, tuple[Any, ...]]] | None = None
    block_optional_inputs: dict[str, set[str]] | None = None
    block_default_inputs: dict[str, dict[str, AxonExpr]] | None = None
    typed_signatures: dict[str, ModuleSignature] | None = None
    tensor_last_dim: dict[str, Any] = field(default_factory=dict)
    tensor_heads: dict[str, Any] = field(default_factory=dict)
    tensor_shape: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    path_param_names: set[str] = field(default_factory=set)
    imported_namespaces: set[str] = field(default_factory=set)
    imported_member_namespaces: dict[str, set[str]] = field(default_factory=dict)
    prelude_aliases: dict[str, tuple[str, int]] = field(default_factory=dict)
    primitive_aliases: dict[str, tuple[str, int]] = field(default_factory=dict)
    current_module: str | None = None
    param_names: set[str] = field(default_factory=set)
    symbol_names: set[str] = field(default_factory=set)
    dim_symbol_names: set[str] = field(default_factory=set)
    symbol_values: dict[str, Any] = field(default_factory=dict)
    runtime_bound_symbol_names: set[str] = field(default_factory=set)
    enforce_source_primitive_syntax: bool = False

    def fresh(self, base: str = "t") -> str:
        self.counter += 1
        return f"{base}_{self.counter}"


def _with_guard(nodes: list[dict[str, Any]], guard: str | None) -> list[dict[str, Any]]:
    if guard is not None:
        raise ValueError("internal lowering error: guard-based lowering is no longer supported")
    return nodes


def _op_name_from_callee(callee: str) -> str:
    if "@" in callee:
        return callee.split("@", 1)[0]
    return callee


def _record_last_dim_for_call(
    *, callee: str, args: list[str], kwargs: dict[str, Any], out: str | list[str], ctx: _LowerCtx
) -> None:
    _validate_namespaced_block_call(callee, ctx)
    resolved_block = _resolve_block_call(callee, ctx)
    if resolved_block is not None and ctx.block_signatures is not None:
        block_name, _ = resolved_block
        input_names, output_names = ctx.block_signatures[block_name]
        provided: dict[str, str] = {}
        for idx, value in enumerate(args):
            if idx < len(input_names):
                provided[input_names[idx]] = value.strip()
        for key, value in kwargs.items():
            if key in input_names:
                provided[key] = str(value).strip()

        symbol_bindings: dict[str, Any] = {}
        for param_name, raw in provided.items():
            if _is_name_token(raw):
                if raw in ctx.tensor_last_dim:
                    symbol_bindings[param_name] = ctx.tensor_last_dim[raw]
                else:
                    symbol_bindings[param_name] = raw
                continue
            parsed = _parse_scalar_token(raw)
            symbol_bindings[param_name] = parsed

        param_last_dims = (
            ctx.block_param_last_dims.get(block_name, {})
            if isinstance(ctx.block_param_last_dims, dict)
            else {}
        )
        param_shapes = (
            ctx.block_param_shapes.get(block_name, {})
            if isinstance(ctx.block_param_shapes, dict)
            else {}
        )
        for param_name, shape_tokens in param_shapes.items():
            if param_name not in provided:
                continue
            raw = provided[param_name]
            if not _is_name_token(raw):
                continue
            actual_shape = ctx.tensor_shape.get(raw)
            if not isinstance(actual_shape, tuple):
                continue
            _bind_shape_symbols(
                formal_shape=tuple(shape_tokens),
                actual_shape=actual_shape,
                symbol_bindings=symbol_bindings,
            )
        for param_name, sym in param_last_dims.items():
            if param_name in symbol_bindings and isinstance(sym, str):
                if _is_variadic_shape_token(sym):
                    if sym in symbol_bindings:
                        continue
                    # A variadic shape capture is not a scalar last-dim alias.
                    # It is only usable when a real argument shape bound it above.
                    continue
                symbol_bindings[sym] = symbol_bindings[param_name]
        output_last_dims = (
            ctx.block_output_last_dims.get(block_name, {})
            if isinstance(ctx.block_output_last_dims, dict)
            else {}
        )
        output_shapes = (
            ctx.block_output_shapes.get(block_name, {})
            if isinstance(ctx.block_output_shapes, dict)
            else {}
        )

        def _is_token_resolved(token: Any) -> bool:
            if not isinstance(token, str):
                return True
            stripped = token.strip()
            if not stripped:
                return True
            if _is_variadic_shape_token(stripped):
                return isinstance(symbol_bindings.get(stripped), tuple)
            if stripped in symbol_bindings:
                return True
            if stripped in ctx.symbol_values:
                return True
            if _is_int_literal(stripped):
                return True
            return not _is_identifier(stripped)

        out_targets = [out] if isinstance(out, str) else list(out)
        for output_name, target in zip(output_names, out_targets, strict=False):
            dim_token = output_last_dims.get(output_name)
            if isinstance(dim_token, str):
                if not _is_token_resolved(dim_token):
                    continue
                resolved_dim = symbol_bindings.get(dim_token, dim_token)
                if _is_variadic_shape_token(dim_token) and isinstance(resolved_dim, tuple):
                    if not resolved_dim:
                        continue
                    resolved_dim = resolved_dim[-1]
                ctx.tensor_last_dim[target] = resolved_dim
            elif dim_token is not None:
                ctx.tensor_last_dim[target] = dim_token
            out_shape_tokens = output_shapes.get(output_name)
            if out_shape_tokens is not None:
                if not all(_is_token_resolved(tok) for tok in out_shape_tokens):
                    continue
                resolved_shape = _resolve_shape_tokens(tuple(out_shape_tokens), symbol_bindings)
                ctx.tensor_shape[target] = resolved_shape

    op_name = _canonical_op_name(callee)
    infer_metadata = get_op_lowering_infer_metadata(op_name)
    if callable(infer_metadata) and bool(
        infer_metadata(args=args, out=out, kwargs=kwargs, ctx=ctx)
    ):
        return


def _normalize_call_kwargs(
    *,
    op_name: str,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: _LowerCtx,
) -> dict[str, Any]:
    normalized = dict(kwargs)
    op_normalizer = get_op_lowering_normalizer(op_name)
    if callable(op_normalizer):
        op_normalizer(args=args, out=out, kwargs=normalized, ctx=ctx)
    return normalized


def _validate_normalized_kwargs(op_name: str, kwargs: dict[str, Any], args: list[str]) -> None:
    _validate_op_signature(op_name, args, kwargs)


def _is_path_type_expr(tp: TypeExpr) -> bool:
    if isinstance(tp, TypeNamed):
        return tp.name == "Path"
    if isinstance(tp, TypeOptional):
        return _is_path_type_expr(tp.inner)
    return False


def _callee_from_base_and_paths(base: str, paths: list[str], *, is_absolute: bool) -> str:
    if not paths:
        return base
    head = f"@@{paths[0]}" if is_absolute else f"@{paths[0]}"
    if len(paths) == 1:
        return f"{base}{head}"
    tail = "".join(f"@{part}" for part in paths[1:])
    return f"{base}{head}{tail}"


def _path_expr_from_suffix_token(token: str, *, is_absolute: bool, ctx: "_LowerCtx") -> AxonExpr:
    raw = token.strip()
    if not raw:
        raise ValueError("empty @path suffix is not allowed")
    if raw in ctx.path_param_names:
        return AxonExprName(name=raw)
    parts = tuple(part for part in raw.split(".") if part)
    if not parts:
        raise ValueError(f"invalid @path suffix {token!r}")
    return AxonExprPath(absolute=is_absolute, parts=parts)


def _strip_atomic_expr_wrappers(expr: AxonExpr) -> AxonExpr:
    current = expr
    while isinstance(current, AxonExprParen | AxonExprAscribe):
        if isinstance(current, AxonExprParen):
            current = current.inner
        else:
            current = current.expr
    return current


def _split_canonical_primitive_path_arg(
    callee: str, args: tuple[AxonExpr, ...]
) -> tuple[AxonExpr | None, tuple[AxonExpr, ...]]:
    if callee not in _CANONICAL_PATH_ARG_PRIMITIVES or not args:
        return None, args
    path_expr = _strip_atomic_expr_wrappers(args[0])
    if isinstance(path_expr, AxonExprPath | AxonExprName):
        return path_expr, args[1:]
    return None, args


def _bind_canonical_primitive_path(
    *,
    node_spec: dict[str, Any],
    path_expr: AxonExpr,
    ctx: _LowerCtx,
) -> None:
    raw = _strip_atomic_expr_wrappers(path_expr)
    if isinstance(raw, AxonExprPath):
        token = _path_expr_to_source(raw)
        if not token.startswith("@@"):
            raise ValueError(
                "typed flat lowering requires canonical primitive Path arguments to be absolute"
            )
        _normalize_bound_params_on_node(node_spec, base_path=token)
        return
    if isinstance(raw, AxonExprName):
        node_spec["param_base"] = raw.name
        return
    raise ValueError("canonical primitive Path argument must be an atomic Path expression")


def _apply_extra_path_suffix_bindings(
    callee: str,
    args: tuple[AxonExpr, ...],
    kwargs_expr: dict[str, AxonKwargValue],
    ctx: "_LowerCtx",
) -> tuple[str, dict[str, AxonKwargValue]]:
    if "@" not in callee:
        return callee, kwargs_expr
    is_absolute_path = "@@" in callee
    parse_callee = callee.replace("@@", "@", 1) if is_absolute_path else callee
    parts = parse_callee.split("@")
    base = parts[0]
    callee_paths = parts[1:]
    if not callee_paths:
        return callee, kwargs_expr

    resolved = _resolve_block_call(callee, ctx)
    if resolved is None:
        return callee, kwargs_expr
    block_name, path_bindings = resolved
    required_count = len(path_bindings)
    if len(callee_paths) <= required_count:
        return callee, kwargs_expr

    typed = ctx.typed_signatures or {}
    call_sig = typed.get(block_name)
    if call_sig is None:
        return callee, kwargs_expr

    extra_paths = callee_paths[required_count:]
    bound_param_names = set(kwargs_expr.keys())
    for idx, arg_expr in enumerate(args):
        del arg_expr
        if idx >= len(call_sig.param_names):
            break
        bound_param_names.add(call_sig.param_names[idx])
    available_path_params = [
        name
        for name, param_type in zip(call_sig.param_names, call_sig.params, strict=True)
        if param_type is not None
        and _is_path_type_expr(param_type)
        and name not in bound_param_names
    ]
    if len(extra_paths) > len(available_path_params):
        raise ValueError(
            f"call {callee!r} provides {len(extra_paths)} extra @path suffixes but only "
            f"{len(available_path_params)} unbound Path arguments are available"
        )

    updated_kwargs = dict(kwargs_expr)
    for path_token, param_name in zip(extra_paths, available_path_params, strict=True):
        if param_name in updated_kwargs:
            raise ValueError(
                f"call {callee!r} received multiple values for argument {param_name!r}"
            )
        updated_kwargs[param_name] = _path_expr_from_suffix_token(
            path_token, is_absolute=is_absolute_path, ctx=ctx
        )

    reduced_paths = callee_paths[:required_count]
    reduced_callee = _callee_from_base_and_paths(base, reduced_paths, is_absolute=is_absolute_path)
    return reduced_callee, updated_kwargs


def _resolve_block_call(callee: str, ctx: _LowerCtx) -> tuple[str, dict[str, str]] | None:
    parse_callee = callee.replace("@@", "@", 1) if "@@" in callee else callee
    if not ctx.block_signatures:
        return None
    if parse_callee in ctx.block_signatures:
        return parse_callee, {}
    if "@" not in parse_callee and "." not in parse_callee and "::" not in parse_callee:
        current_module = ctx.current_module if isinstance(ctx.current_module, str) else ""
        current_namespace = (
            current_module.rsplit(".", 1)[0] if "." in current_module else current_module
        )
        if current_namespace:
            namespaced_callee = f"{current_namespace}.{parse_callee}"
            if namespaced_callee in ctx.block_signatures:
                return namespaced_callee, {}
    if "." in parse_callee and "@" not in parse_callee:
        member = parse_callee.rsplit(".", 1)[1]
        namespace = parse_callee.rsplit(".", 1)[0]
        imported_for_member = ctx.imported_member_namespaces.get(member, set())
        if (
            member in ctx.block_signatures
            and namespace in ctx.imported_namespaces
            and namespace in imported_for_member
        ):
            return member, {}
    if "@" not in parse_callee and "." not in parse_callee and "::" not in parse_callee:
        imported_namespaces = ctx.imported_member_namespaces.get(parse_callee, set())
        if imported_namespaces:
            if len(imported_namespaces) > 1:
                choices = ", ".join(sorted(imported_namespaces))
                raise ValueError(
                    f"ambiguous imported member {parse_callee!r}; found in namespaces: {choices}"
                )
            namespace = next(iter(imported_namespaces))
            namespaced_callee = f"{namespace}.{parse_callee}"
            if namespaced_callee in ctx.block_signatures:
                return namespaced_callee, {}
            raise ValueError(
                f"imported member {parse_callee!r} from {namespace!r} not found as module "
                f"{namespaced_callee!r}"
            )
    if "@" not in parse_callee:
        return None
    parts = parse_callee.split("@")
    base = parts[0]
    concrete_paths = parts[1:]
    if base not in ctx.block_signatures and "." in base:
        namespace, member = base.rsplit(".", 1)
        imported_for_member = ctx.imported_member_namespaces.get(member, set())
        if (
            member in ctx.block_signatures
            and namespace in ctx.imported_namespaces
            and namespace in imported_for_member
        ):
            base = member
    if base not in ctx.block_signatures and "." not in base:
        current_module = ctx.current_module if isinstance(ctx.current_module, str) else ""
        current_namespace = (
            current_module.rsplit(".", 1)[0] if "." in current_module else current_module
        )
        if current_namespace:
            namespaced_base = f"{current_namespace}.{base}"
            if namespaced_base in ctx.block_signatures:
                base = namespaced_base
        if base in ctx.block_signatures:
            pass
        else:
            imported_namespaces = ctx.imported_member_namespaces.get(base, set())
            if imported_namespaces:
                if len(imported_namespaces) > 1:
                    choices = ", ".join(sorted(imported_namespaces))
                    raise ValueError(
                        f"ambiguous imported member {base!r}; found in namespaces: {choices}"
                    )
                namespace = next(iter(imported_namespaces))
                namespaced_base = f"{namespace}.{base}"
                if namespaced_base in ctx.block_signatures:
                    base = namespaced_base
    if base not in ctx.block_signatures:
        return None
    path_params = (
        ctx.block_path_params.get(base, ()) if isinstance(ctx.block_path_params, dict) else ()
    )
    if not path_params:
        return base, {}
    if len(concrete_paths) < len(path_params):
        raise ValueError(
            f"block call {callee!r} expects at least {len(path_params)} @path arguments, got {len(concrete_paths)}"
        )
    return base, {
        path_param: concrete
        for path_param, concrete in zip(
            path_params, concrete_paths[: len(path_params)], strict=True
        )
    }


def _validate_namespaced_block_call(callee: str, ctx: _LowerCtx) -> None:
    if "." not in callee or "@" in callee or "::" in callee:
        return
    if not ctx.block_signatures or callee not in ctx.block_signatures:
        return
    return


def _rewrite_prelude_alias_callee(callee: str, kwargs: dict[str, Any], ctx: _LowerCtx) -> str:
    if "::" in callee:
        return callee
    parts = callee.split("@")
    base = parts[0]
    path_parts = parts[1:]

    member_name: str | None = None
    if "." not in base:
        member_name = base
    elif base.startswith("Prelude."):
        member_name = base.split(".", 1)[1]
    if not member_name:
        return callee

    if isinstance(ctx.block_signatures, dict) and member_name in ctx.block_signatures:
        return callee
    imported_for_member = ctx.imported_member_namespaces.get(member_name, set())
    if imported_for_member and imported_for_member != {"Prelude"}:
        return callee
    alias = ctx.prelude_aliases.get(member_name)
    if alias is None:
        alias = _IMPLICIT_ACTIVATION_ALIASES.get(member_name)
    if alias is None:
        return callee
    target_base, expected_path_count = alias
    # If suffix @path arity does not match exactly, keep the original callee.
    # Extra @path suffixes are handled later by _apply_extra_path_suffix_bindings
    # and may map to positional Path-typed parameters on the wrapper module.
    if expected_path_count != len(path_parts):
        return callee
    if path_parts:
        return "@".join([target_base, *path_parts])
    return target_base


def _rewrite_primitive_alias_callee(callee: str, kwargs: dict[str, Any], ctx: _LowerCtx) -> str:
    if not ctx.primitive_aliases:
        return callee
    if "::" in callee:
        return callee
    parts = callee.split("@")
    base = parts[0]
    path_parts = parts[1:]

    full_name: str | None = None
    if "." in base:
        full_name = base
    else:
        imported_for_member = ctx.imported_member_namespaces.get(base, set())
        if len(imported_for_member) == 1:
            namespace = next(iter(imported_for_member))
            full_name = f"{namespace}.{base}"
    if not full_name:
        return callee

    # If a callable block exists for this symbol, keep normal module call
    # resolution and do not collapse to a primitive alias here.
    if isinstance(ctx.block_signatures, dict) and (
        full_name in ctx.block_signatures or base in ctx.block_signatures
    ):
        return callee

    alias = ctx.primitive_aliases.get(full_name)
    if alias is None:
        return callee
    target_base, expected_path_count = alias
    # Preserve wrapper calls when suffix @path arity does not match exactly.
    # This allows later path-suffix binding against wrapper Path parameters
    # before any primitive rewrite.
    if expected_path_count != len(path_parts):
        return callee
    if path_parts:
        return "@".join([target_base, *path_parts])
    return target_base


def _known_output_arity(callee: str, kwargs: dict[str, Any], ctx: _LowerCtx) -> int | None:
    resolved = _resolve_block_call(callee, ctx)
    if resolved is not None and ctx.block_signatures:
        block_name, _ = resolved
        _, output_names = ctx.block_signatures[block_name]
        return len(output_names)

    normalized = _canonical_op_name(callee)
    op_arity = get_op_lowering_known_output_arity(normalized)
    if callable(op_arity):
        resolved = op_arity(kwargs=kwargs)
        if isinstance(resolved, int):
            return resolved
    return None


def _pipeline_temp_out(stage: AxonExpr, ctx: _LowerCtx) -> str | list[str]:
    if isinstance(stage, AxonExprName):
        arity = _known_output_arity(stage.name, {}, ctx)
    elif isinstance(stage, AxonExprCall):
        kwargs = _render_kwargs_for_call(stage.kwargs)
        arity = _known_output_arity(stage.callee, kwargs, ctx)
    else:
        return ctx.fresh("pipe")
    if arity is None or arity <= 1:
        return ctx.fresh("pipe")
    return [ctx.fresh("pipe") for _ in range(arity)]


def _expand_call_outputs_for_ternary(
    call_expr: AxonExprCall | AxonExprName, out: str | list[str], ctx: _LowerCtx
) -> str | list[str]:
    if not isinstance(out, list):
        return out
    if isinstance(call_expr, AxonExprName):
        arity = _known_output_arity(call_expr.name, {}, ctx)
    else:
        kwargs = _render_kwargs_for_call(call_expr.kwargs)
        arity = _known_output_arity(call_expr.callee, kwargs, ctx)
    if not isinstance(arity, int) or arity <= len(out):
        return out
    return [*out, *[ctx.fresh("discard") for _ in range(arity - len(out))]]


def _render_kwargs_for_call(kwargs: dict[str, AxonKwargValue]) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, AxonExprInt):
            rendered[key] = value.value
        elif isinstance(value, AxonExprFloat):
            rendered[key] = value.lexeme if value.lexeme else value.value
        elif isinstance(value, AxonExprBool):
            rendered[key] = value.value
        elif isinstance(value, AxonExprNull):
            rendered[key] = None
        elif isinstance(value, AxonExprString):
            rendered[key] = value.value
        elif isinstance(value, AxonExprPath):
            rendered[key] = _path_expr_to_source(value)
        elif isinstance(value, AxonExprList):
            rendered[key] = _expr_to_runtime_value(value)
        elif isinstance(value, AxonExprName):
            rendered[key] = value.name
        elif isinstance(value, AxonExpr):
            try:
                rendered[key] = _expr_to_runtime_value(value)
            except ValueError:
                rendered[key] = None
        else:
            rendered[key] = value
    return rendered


def _substitute_expr(expr: AxonExpr, var_name: str, replacement: AxonExpr) -> AxonExpr:
    if isinstance(expr, AxonExprName):
        if expr.name == var_name:
            return replacement
        return expr
    if isinstance(
        expr,
        AxonExprInt | AxonExprFloat | AxonExprBool | AxonExprNull | AxonExprString | AxonExprPath,
    ):
        return expr
    if isinstance(expr, AxonExprTuple):
        return AxonExprTuple(
            items=tuple(_substitute_expr(item, var_name, replacement) for item in expr.items)
        )
    if isinstance(expr, AxonExprList):
        return AxonExprList(
            items=tuple(_substitute_expr(item, var_name, replacement) for item in expr.items)
        )
    if isinstance(expr, AxonExprCall):
        new_kwargs: dict[str, AxonKwargValue] = {}
        for key, value in expr.kwargs.items():
            if isinstance(value, AxonExpr):
                new_kwargs[key] = _substitute_expr(value, var_name, replacement)
            else:
                new_kwargs[key] = value
        return AxonExprCall(
            callee=expr.callee,
            args=tuple(_substitute_expr(arg, var_name, replacement) for arg in expr.args),
            kwargs=new_kwargs,
        )
    if isinstance(expr, AxonExprPipe):
        return AxonExprPipe(
            value=_substitute_expr(expr.value, var_name, replacement),
            stages=tuple(_substitute_expr(stage, var_name, replacement) for stage in expr.stages),
        )
    if isinstance(expr, AxonExprBind):
        if expr.var == var_name:
            return AxonExprBind(
                value=_substitute_expr(expr.value, var_name, replacement),
                var=expr.var,
                body=expr.body,
            )
        return AxonExprBind(
            value=_substitute_expr(expr.value, var_name, replacement),
            var=expr.var,
            body=_substitute_expr(expr.body, var_name, replacement),
        )
    if isinstance(expr, AxonExprIf):
        return AxonExprIf(
            cond=_substitute_expr(expr.cond, var_name, replacement),
            true_expr=_substitute_expr(expr.true_expr, var_name, replacement),
            false_expr=_substitute_expr(expr.false_expr, var_name, replacement),
        )
    if isinstance(expr, AxonExprTernary):
        return AxonExprTernary(
            cond=_substitute_expr(expr.cond, var_name, replacement),
            true_expr=_substitute_expr(expr.true_expr, var_name, replacement),
            false_expr=_substitute_expr(expr.false_expr, var_name, replacement),
        )
    if isinstance(expr, AxonExprBinary):
        return AxonExprBinary(
            op=expr.op,
            left=_substitute_expr(expr.left, var_name, replacement),
            right=_substitute_expr(expr.right, var_name, replacement),
        )
    if isinstance(expr, AxonExprLambda):
        if expr.var == var_name:
            return expr
        return AxonExprLambda(var=expr.var, body=_substitute_expr(expr.body, var_name, replacement))
    if isinstance(expr, AxonExprParen):
        return AxonExprParen(inner=_substitute_expr(expr.inner, var_name, replacement))
    if isinstance(expr, AxonExprDo):
        body_out: list[AxonStatement] = []
        for stmt in expr.body:
            if isinstance(stmt, AxonBind):
                body_out.append(
                    AxonBind(
                        targets=stmt.targets,
                        expr=_substitute_expr(stmt.expr, var_name, replacement),
                    )
                )
                continue
            if isinstance(stmt, AxonReturn):
                body_out.append(
                    AxonReturn(
                        values=tuple(
                            _substitute_expr(value, var_name, replacement) for value in stmt.values
                        )
                    )
                )
                continue
            if isinstance(stmt, AxonRepeat):
                body_out.append(
                    AxonRepeat(
                        name=stmt.name,
                        var=stmt.var,
                        to_expr=_substitute_expr(stmt.to_expr, var_name, replacement),
                        from_expr=_substitute_expr(stmt.from_expr, var_name, replacement),
                        step_expr=_substitute_expr(stmt.step_expr, var_name, replacement),
                        body=stmt.body,
                    )
                )
                continue
            if isinstance(stmt, AxonScopeBind):
                body_out.append(stmt)
                continue
        return AxonExprDo(body=tuple(body_out))
    return expr


def _runtime_value_to_expr(value: Any) -> AxonExpr:
    if isinstance(value, AxonExpr):
        return value
    if value is None:
        return AxonExprNull()
    if isinstance(value, bool):
        return AxonExprBool(value=value)
    if isinstance(value, int):
        return AxonExprInt(value=value)
    if isinstance(value, float):
        return AxonExprFloat(value=float(value))
    if isinstance(value, str):
        raw = value.strip()
        if raw and _is_name_token(raw):
            return AxonExprName(name=raw)
        parsed = _parse_scalar_token(raw)
        if parsed is None:
            return AxonExprNull()
        if isinstance(parsed, bool):
            return AxonExprBool(value=parsed)
        if isinstance(parsed, int):
            return AxonExprInt(value=parsed)
        if isinstance(parsed, float):
            return AxonExprFloat(value=float(parsed))
        if isinstance(parsed, str) and _is_name_token(raw):
            return AxonExprName(name=raw)
        return AxonExprString(value=str(parsed))
    if isinstance(value, list):
        return AxonExprList(items=tuple(_runtime_value_to_expr(item) for item in value))
    return AxonExprString(value=str(value))


def _expr_is_tensorish(expr: AxonExpr, ctx: _LowerCtx) -> bool:
    if isinstance(
        expr, AxonExprCall | AxonExprPipe | AxonExprBind | AxonExprIf | AxonExprTernary | AxonExprDo
    ):
        return True
    if isinstance(expr, AxonExprName):
        token = expr.name
        if token in ctx.symbol_names:
            return False
        return (
            token in ctx.param_names
            or token in ctx.tensor_shape
            or token in ctx.tensor_last_dim
            or token in ctx.tensor_heads
        )
    if isinstance(expr, AxonExprBinary):
        return _expr_is_tensorish(expr.left, ctx) or _expr_is_tensorish(expr.right, ctx)
    if isinstance(expr, AxonExprParen):
        return _expr_is_tensorish(expr.inner, ctx)
    if isinstance(expr, AxonExprTuple):
        return any(_expr_is_tensorish(item, ctx) for item in expr.items)
    if isinstance(expr, AxonExprList):
        return any(_expr_is_tensorish(item, ctx) for item in expr.items)
    return False


def _list_item_literal_value(expr: AxonExpr) -> Any:
    if isinstance(expr, AxonExprInt):
        return expr.value
    if isinstance(expr, AxonExprFloat):
        return expr.value
    if isinstance(expr, AxonExprBool):
        return expr.value
    if isinstance(expr, AxonExprNull):
        return None
    if isinstance(expr, AxonExprString):
        return expr.value
    if isinstance(expr, AxonExprPath):
        return _path_expr_to_source(expr)
    if isinstance(expr, AxonExprName):
        return expr.name
    if isinstance(expr, AxonExprParen):
        return _list_item_literal_value(expr.inner)
    if isinstance(expr, AxonExprList):
        return [_list_item_literal_value(item) for item in expr.items]
    raise ValueError(f"list literal item must be scalar/name/list, got {type(expr).__name__}")


def _kwarg_needs_temp_binding(expr: AxonExpr, ctx: _LowerCtx) -> bool:
    if _expr_is_tensorish(expr, ctx):
        return True
    if isinstance(expr, AxonExprCall | AxonExprPipe | AxonExprBind | AxonExprDo | AxonExprLambda):
        return True
    if isinstance(expr, AxonExprParen):
        return _kwarg_needs_temp_binding(expr.inner, ctx)
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return (
            _kwarg_needs_temp_binding(expr.cond, ctx)
            or _kwarg_needs_temp_binding(expr.true_expr, ctx)
            or _kwarg_needs_temp_binding(expr.false_expr, ctx)
        )
    if isinstance(expr, AxonExprBinary):
        return _kwarg_needs_temp_binding(expr.left, ctx) or _kwarg_needs_temp_binding(
            expr.right, ctx
        )
    if isinstance(expr, AxonExprTuple):
        return any(_kwarg_needs_temp_binding(item, ctx) for item in expr.items)
    if isinstance(expr, AxonExprList):
        return any(_kwarg_needs_temp_binding(item, ctx) for item in expr.items)
    return False


def _lower_simple_call(
    callee: str,
    args: tuple[AxonExpr, ...],
    kwargs_expr: dict[str, AxonKwargValue],
    out: str | list[str],
    ctx: _LowerCtx,
    *,
    guard: str | None = None,
    enforce_source_primitive_syntax: bool = True,
    call_expr: AxonExprCall | None = None,
) -> list[dict[str, Any]]:
    callee = callee.strip()
    raw_callee = callee
    canonical_path_arg, args = _split_canonical_primitive_path_arg(callee, args)
    _validate_namespaced_block_call(callee, ctx)
    # Generic sugar first: abc@p1@p2 binds to unbound Path-typed arguments.
    callee, kwargs_expr = _apply_extra_path_suffix_bindings(callee, args, kwargs_expr, ctx)
    callee = _rewrite_prelude_alias_callee(callee, kwargs_expr, ctx)
    callee = _rewrite_primitive_alias_callee(callee, kwargs_expr, ctx)
    raw_base = raw_callee.split("@", 1)[0].strip()
    op_name_raw = _canonical_op_name(raw_callee)
    raw_call_resolves_to_block = _resolve_block_call(raw_callee, ctx) is not None
    primitive_exists = (
        get_op_lowering_signature(op_name_raw) is not None
        or get_op_lowering_type_signature(op_name_raw) is not None
        or get_op_lowering_known_output_arity(op_name_raw) is not None
    )
    enforce_source_primitive_syntax = (
        enforce_source_primitive_syntax and ctx.enforce_source_primitive_syntax
    )
    if (
        enforce_source_primitive_syntax
        and primitive_exists
        and not raw_base.startswith("_")
        and not raw_call_resolves_to_block
    ):
        raise ValueError(f"direct primitive call must use _xyz syntax, got {raw_base!r}")
    if enforce_source_primitive_syntax and raw_base.startswith("_"):
        caller_namespace = (
            ctx.current_module.split(".", 1)[0]
            if isinstance(ctx.current_module, str) and ctx.current_module
            else ""
        )
        if (
            primitive_exists
            and "." in (ctx.current_module or "")
            and caller_namespace not in _BUILTIN_MODULE_NAMESPACES
        ):
            raise ValueError(
                f"direct primitive call {raw_base!r} is only allowed in builtins (*.axon)"
            )
    if raw_base == "_linear" and kwargs_expr:
        raise ValueError(
            "_linear only accepts positional arguments; use Prelude.linear for keyword/default syntax"
        )
    if raw_base == "_layernorm" and kwargs_expr:
        raise ValueError(
            "_layernorm only accepts positional arguments; use Prelude.layernorm for keyword/default syntax"
        )
    if raw_base == "_embedding" and kwargs_expr:
        raise ValueError(
            "_embedding only accepts positional arguments; use Prelude.embedding for keyword/default syntax"
        )
    if raw_base == "_split" and kwargs_expr:
        raise ValueError(
            "_split only accepts positional arguments; use Prelude.split for keyword/default syntax"
        )
    if raw_base == "_chunk" and kwargs_expr:
        raise ValueError(
            "_chunk only accepts positional arguments; use Prelude.chunk for keyword/default syntax"
        )
    if raw_base == "_cast" and kwargs_expr:
        raise ValueError(
            "_cast only accepts positional arguments; use Prelude.cast for keyword/default syntax"
        )
    if raw_base == "_cumsum" and kwargs_expr:
        raise ValueError(
            "_cumsum only accepts positional arguments; use Prelude.cumsum for keyword/default syntax"
        )
    if raw_base == "_arange" and kwargs_expr:
        raise ValueError(
            "_arange only accepts positional arguments; use Prelude.arange for keyword/default syntax"
        )
    if raw_base == "_expand" and kwargs_expr:
        raise ValueError(
            "_expand only accepts positional arguments; use Prelude.expand for keyword/default syntax"
        )
    if raw_base == "_slice" and kwargs_expr:
        raise ValueError(
            "_slice only accepts positional arguments; use Prelude.slice for keyword/default syntax"
        )
    resolved_block = _resolve_block_call(callee, ctx)
    pre_graph: list[dict[str, Any]] = []
    kwargs = _render_kwargs_for_call(kwargs_expr)
    effective_when = guard

    resolved_args: list[str] = []
    runtime_args: list[Any] = []
    for arg in args:
        if isinstance(arg, AxonExprName):
            resolved_args.append(arg.name)
            runtime_value: Any = arg.name
            is_dimlike_symbol = (
                arg.name.isidentifier()
                and arg.name.upper() == arg.name
                and any(ch.isalpha() for ch in arg.name)
            )
            if (
                resolved_block is not None
                and is_dimlike_symbol
                and arg.name not in ctx.param_names
                and arg.name not in ctx.tensor_last_dim
                and arg.name not in ctx.tensor_shape
            ):
                # Forward symbolic scalar/dim names through block wrappers as expression
                # payloads so codegen/runtime can resolve them via shape aliases/symbols.
                runtime_value = {"_expr": "name", "id": arg.name}
            runtime_args.append(runtime_value)
            continue
        if isinstance(arg, AxonExprInt):
            resolved_args.append(str(arg.value))
            runtime_args.append(arg.value)
            continue
        if isinstance(arg, AxonExprFloat):
            resolved_args.append(arg.lexeme if arg.lexeme else str(arg.value))
            runtime_args.append(arg.value)
            continue
        if isinstance(arg, AxonExprBool):
            resolved_args.append("true" if arg.value else "false")
            runtime_args.append(arg.value)
            continue
        if isinstance(arg, AxonExprNull):
            resolved_args.append("null")
            runtime_args.append(None)
            continue
        if isinstance(arg, AxonExprString):
            resolved_args.append(arg.value)
            runtime_args.append(arg.value)
            continue
        if isinstance(arg, AxonExprPath):
            token = _path_expr_to_source(arg)
            resolved_args.append(token)
            runtime_args.append(token)
            continue
        tmp = ctx.fresh("arg")
        pre_graph.extend(_lower_expr(arg, tmp, ctx, guard=guard))
        resolved_args.append(tmp)
        runtime_args.append(tmp)
    args_text = resolved_args

    op_name = _canonical_op_name(callee)
    signature = get_op_lowering_signature(op_name)
    kwarg_kinds = signature.get("kwarg_kinds") if isinstance(signature, dict) else {}
    resolved_kwargs: dict[str, Any] = {}
    for key, value_expr in kwargs_expr.items():
        expected_kind = kwarg_kinds.get(key) if isinstance(kwarg_kinds, dict) else None
        if isinstance(value_expr, AxonExpr):
            folded = _NO_CONST
            if expected_kind == "dim":
                folded = _const_fold_dim_expr(value_expr, ctx, allow_bare_name=False)
            elif expected_kind == "list_dim" and isinstance(value_expr, AxonExprList):
                folded_items: list[Any] = []
                for item in value_expr.items:
                    folded_item = _const_fold_dim_expr(item, ctx, allow_bare_name=False)
                    if folded_item is _NO_CONST:
                        folded_items = []
                        break
                    folded_items.append(folded_item)
                if folded_items:
                    folded = folded_items
            elif expected_kind not in {"dim", "list_dim", "str", "str_or_bool_or_null"}:
                folded = _const_fold_expr(value_expr, ctx)
            if folded is not _NO_CONST:
                resolved_kwargs[key] = folded
                continue
        if isinstance(value_expr, AxonExprName):
            if resolved_block is not None:
                is_dimlike_symbol = (
                    value_expr.name.isidentifier()
                    and value_expr.name.upper() == value_expr.name
                    and any(ch.isalpha() for ch in value_expr.name)
                )
                if (
                    is_dimlike_symbol
                    and value_expr.name not in ctx.param_names
                    and value_expr.name not in ctx.tensor_last_dim
                    and value_expr.name not in ctx.tensor_shape
                ):
                    resolved_kwargs[key] = {"_expr": "name", "id": value_expr.name}
                    continue
            if expected_kind in {"list_int", "list_dim"}:
                resolved_kwargs[key] = {"_expr": "name", "id": value_expr.name}
            else:
                resolved_kwargs[key] = value_expr.name
            continue
        if isinstance(value_expr, AxonExprInt):
            resolved_kwargs[key] = value_expr.value
            continue
        if isinstance(value_expr, AxonExprFloat):
            resolved_kwargs[key] = value_expr.value
            continue
        if isinstance(value_expr, AxonExprBool):
            resolved_kwargs[key] = value_expr.value
            continue
        if isinstance(value_expr, AxonExprNull):
            resolved_kwargs[key] = None
            continue
        if isinstance(value_expr, AxonExprString):
            resolved_kwargs[key] = value_expr.value
            continue
        if isinstance(value_expr, AxonExprPath):
            resolved_kwargs[key] = _path_expr_to_source(value_expr)
            continue
        if isinstance(value_expr, AxonExprList):
            if expected_kind in {"list_int", "list_dim"}:
                resolved_kwargs[key] = _expr_to_runtime_value(value_expr)
                continue
            if _kwarg_needs_temp_binding(value_expr, ctx):
                key_token = _sanitize_token(key, default="kwarg")
                tmp = ctx.fresh(f"kwarg_{key_token}")
                pre_graph.extend(_lower_expr(value_expr, tmp, ctx, guard=guard))
                resolved_kwargs[key] = tmp
            else:
                resolved_kwargs[key] = _expr_to_runtime_value(value_expr)
            continue
        if isinstance(value_expr, AxonExpr):
            if expected_kind in {
                "bool",
                "int",
                "number",
                "str",
                "str_or_bool_or_null",
                "dim",
                "any",
            }:
                if _kwarg_needs_temp_binding(value_expr, ctx):
                    key_token = _sanitize_token(key, default="kwarg")
                    tmp = ctx.fresh(f"kwarg_{key_token}")
                    pre_graph.extend(_lower_expr(value_expr, tmp, ctx, guard=guard))
                    resolved_kwargs[key] = tmp
                    continue
                resolved_kwargs[key] = _expr_to_runtime_value(value_expr)
                continue
            if _kwarg_needs_temp_binding(value_expr, ctx):
                key_token = _sanitize_token(key, default="kwarg")
                tmp = ctx.fresh(f"kwarg_{key_token}")
                pre_graph.extend(_lower_expr(value_expr, tmp, ctx, guard=guard))
                resolved_kwargs[key] = tmp
            else:
                resolved_kwargs[key] = _expr_to_runtime_value(value_expr)
            continue
        resolved_kwargs[key] = value_expr
    kwargs = resolved_kwargs

    is_absolute_path = "@@" in callee
    if is_absolute_path:
        callee = callee.replace("@@", "@", 1)
        if resolved_block is None:
            resolved_block = _resolve_block_call(callee, ctx)
    if resolved_block is None:
        kwargs = _normalize_call_kwargs(
            op_name=op_name,
            args=args_text,
            out=out,
            kwargs=kwargs,
            ctx=ctx,
        )
        _validate_normalized_kwargs(op_name, kwargs, args_text)
        op_validate = get_op_lowering_validator(op_name)
        if callable(op_validate):
            op_validate(args=args_text, out=out, kwargs=kwargs, ctx=ctx)
    _validate_namespaced_block_call(callee, ctx)
    if resolved_block is None and "." in callee and "@" not in callee and "::" not in callee:
        namespace = callee.split(".", 1)[0]
        raise ValueError(
            f"unknown namespaced module call {callee!r}; add `import {namespace}` and parse from file"
        )
    if resolved_block is not None and ctx.block_signatures:
        block_name, path_bindings = resolved_block
        input_names, output_names = ctx.block_signatures[block_name]
        provided: dict[str, Any] = {}
        for idx, value in enumerate(runtime_args):
            if idx >= len(input_names):
                raise ValueError(f"too many positional args for block call {callee!r}")
            provided[input_names[idx]] = value
        for key, value in kwargs.items():
            if key not in input_names:
                raise ValueError(f"unknown block input {key!r} for call {callee!r}")
            provided[key] = value
        for key, concrete_path in path_bindings.items():
            if key not in input_names:
                raise ValueError(f"unknown block path parameter {key!r} for call {callee!r}")
            if concrete_path in ctx.path_param_names:
                # Preserve symbolic path-param forwarding through wrappers (for example
                # Prelude wrappers called as @path inside another @path module).
                provided[key] = concrete_path
            else:
                concrete_value = f"@@{concrete_path}" if is_absolute_path else concrete_path
                provided[key] = repr(concrete_value)
        optional_inputs: set[str] = set()
        if isinstance(ctx.block_optional_inputs, dict):
            raw_optional = ctx.block_optional_inputs.get(block_name)
            if isinstance(raw_optional, set):
                optional_inputs = raw_optional
        default_inputs: dict[str, AxonExpr] = {}
        if isinstance(ctx.block_default_inputs, dict):
            raw_defaults = ctx.block_default_inputs.get(block_name)
            if isinstance(raw_defaults, dict):
                default_inputs = raw_defaults
        for input_name in input_names:
            if input_name in provided:
                continue
            default_expr = default_inputs.get(input_name)
            if isinstance(default_expr, AxonExpr):
                resolved_default = default_expr
                for param_name, param_value in provided.items():
                    resolved_default = _substitute_expr(
                        resolved_default,
                        param_name,
                        _runtime_value_to_expr(param_value),
                    )
                if _kwarg_needs_temp_binding(resolved_default, ctx):
                    tmp_name = ctx.fresh(f"default_{_sanitize_token(input_name, default='arg')}")
                    pre_graph.extend(_lower_expr(resolved_default, tmp_name, ctx, guard=guard))
                    provided[input_name] = tmp_name
                else:
                    provided[input_name] = _expr_to_runtime_value(resolved_default)
                continue
            if input_name in optional_inputs:
                provided[input_name] = None
        if isinstance(ctx.block_param_shapes, dict):
            param_shapes = ctx.block_param_shapes.get(block_name, {})
        else:
            param_shapes = {}
        symbol_bindings: dict[str, Any] = {}
        for param_name, raw in provided.items():
            if not isinstance(raw, str):
                continue
            token = raw.strip()
            if _is_name_token(token) and token in ctx.tensor_last_dim:
                symbol_bindings[param_name] = ctx.tensor_last_dim[token]
            elif not _is_name_token(token):
                symbol_bindings[param_name] = _parse_scalar_token(token)
        for param_name, param_shape in param_shapes.items():
            if param_name not in provided:
                continue
            raw_value = provided[param_name]
            if not isinstance(raw_value, str):
                continue
            token = raw_value.strip()
            if not _is_name_token(token):
                continue
            arg_shape = ctx.tensor_shape.get(token)
            if arg_shape is None:
                continue
            has_variadic_shape = any(_is_variadic_shape_token(tok) for tok in param_shape)
            if len(arg_shape) != len(param_shape) and not has_variadic_shape:
                raise ValueError(
                    f"shape mismatch in call {callee!r} for param {param_name!r}: "
                    f"expected rank {len(param_shape)} from signature {param_shape}, got rank {len(arg_shape)} from {arg_shape}"
                )
            _bind_shape_symbols(
                formal_shape=tuple(param_shape),
                actual_shape=arg_shape,
                symbol_bindings=symbol_bindings,
            )
            if has_variadic_shape:
                continue
            expected_shape = tuple(symbol_bindings.get(sym, sym) for sym in param_shape)
            if len(expected_shape) != len(arg_shape) or any(
                not _dims_compatible(exp, got)
                for exp, got in zip(expected_shape, arg_shape, strict=True)
            ):
                raise ValueError(
                    f"shape mismatch in call {callee!r} for param {param_name!r}: "
                    f"expected {expected_shape} from signature, got {arg_shape} from argument {token!r}"
                )
        tuple_result_out: str | None = None
        if isinstance(out, str) and len(output_names) > 1:
            tuple_result_out = out
            out_values = [ctx.fresh(f"{_sanitize_token(out, default='out')}_{idx}") for idx in range(len(output_names))]
        else:
            out_values = [out] if isinstance(out, str) else list(out)
        if len(out_values) != len(output_names):
            raise ValueError(
                f"block call {callee!r} expects {len(output_names)} outputs, got {len(out_values)}"
            )
        positional_args: list[str] = []
        extra_kwargs: dict[str, Any] = {}
        for input_name in input_names:
            if input_name not in provided:
                continue
            value = provided[input_name]
            if input_name in kwargs or input_name in path_bindings:
                extra_kwargs[input_name] = value
            elif len(positional_args) < len(args_text):
                positional_args.append(value)
            else:
                extra_kwargs[input_name] = value
        node_name = f"n_{ctx.fresh('call')}"
        node_spec: dict[str, Any] = {"_op": "call", "_target": block_name}
        if positional_args:
            node_spec["_args"] = (
                positional_args[0] if len(positional_args) == 1 else positional_args
            )
        node_spec["_bind"] = out_values[0] if len(out_values) == 1 else out_values
        if call_expr is not None and call_expr.inferred_type is not None:
            if len(out_values) == 1:
                node_spec["_out_types"] = {str(out_values[0]): render_type(call_expr.inferred_type)}
            elif isinstance(call_expr.inferred_type, TypeTuple) and len(
                call_expr.inferred_type.items
            ) == len(out_values):
                node_spec["_out_types"] = {
                    str(bind_name): render_type(item_type)
                    for bind_name, item_type in zip(
                        out_values, call_expr.inferred_type.items, strict=True
                    )
                }
        for key, value in extra_kwargs.items():
            node_spec[key] = value
        nodes = _with_guard([{node_name: node_spec}], effective_when)
        _record_last_dim_for_call(
            callee=block_name,
            args=args_text,
            kwargs=kwargs,
            out=out_values,
            ctx=ctx,
        )
        if tuple_result_out is not None:
            nodes.extend(
                _lower_expr(
                    AxonExprTuple(items=tuple(AxonExprName(name=name) for name in out_values)),
                    tuple_result_out,
                    ctx,
                    guard=guard,
                )
            )
        return [
            *pre_graph,
            *nodes,
        ]

    node_spec = _to_synapse_op(callee, args_text, kwargs, out)
    if canonical_path_arg is not None:
        _bind_canonical_primitive_path(node_spec=node_spec, path_expr=canonical_path_arg, ctx=ctx)
    if "@" in callee:
        op_name, param_path = callee.split("@", 1)
        if param_path in ctx.path_param_names:
            node_name = f"n_{ctx.fresh('op')}"
            templated_node = _to_synapse_op(op_name, args_text, kwargs, out)
            templated_node["param_base"] = param_path
            nodes = _with_guard([{node_name: templated_node}], effective_when)
            _record_last_dim_for_call(
                callee=op_name, args=args_text, kwargs=kwargs, out=out, ctx=ctx
            )
            return [
                *pre_graph,
                *nodes,
            ]
        concrete_node = _to_synapse_op(op_name, args_text, kwargs, out)
        try:
            bound_params = _path_bound_param_names(concrete_node)
        except ValueError:
            bound_params = []
        if bound_params:
            if not param_path.strip():
                raise ValueError(f"invalid @ path in Axon call: {callee!r}")
            node_name = f"n_{ctx.fresh('op')}"
            params: dict[str, str | list[str]] = {}
            for param_name in bound_params:
                explicit_name = concrete_node.get(param_name)
                if isinstance(explicit_name, str) and explicit_name.strip():
                    explicit_token = explicit_name.strip()
                    suffix = (
                        explicit_token
                        if "." in explicit_token
                        else f"{param_path}.{explicit_token}"
                    )
                else:
                    suffix = f"{param_path}.{param_name}"
                if is_absolute_path:
                    concrete_node["_abs_path"] = param_path
                    params[param_name] = (
                        explicit_token
                        if isinstance(explicit_name, str) and explicit_name.strip()
                        else param_name
                    )
                else:
                    params[param_name] = suffix
            concrete_node["_params"] = params
            nodes = _with_guard([{node_name: concrete_node}], effective_when)
            _record_last_dim_for_call(
                callee=callee, args=args_text, kwargs=kwargs, out=out, ctx=ctx
            )
            return [
                *pre_graph,
                *nodes,
            ]
        segments = [part.strip() for part in param_path.split(".") if part.strip()]
        if not segments:
            raise ValueError(f"invalid @ path in Axon call: {callee!r}")
        graph_item: dict[str, Any] = {segments[-1]: node_spec}
        for segment in reversed(segments[:-1]):
            graph_item = {segment: {"graph": [graph_item]}}
        nodes = _with_guard([graph_item], effective_when)
        _record_last_dim_for_call(callee=callee, args=args_text, kwargs=kwargs, out=out, ctx=ctx)
        return [
            *pre_graph,
            *nodes,
        ]
    node_name = f"n_{ctx.fresh('op')}"
    nodes = _with_guard([{node_name: node_spec}], effective_when)
    _record_last_dim_for_call(callee=callee, args=args_text, kwargs=kwargs, out=out, ctx=ctx)
    return [*pre_graph, *nodes]


def _lower_alias_or_const(
    expr: AxonExpr, out: str | list[str], ctx: _LowerCtx, *, guard: str | None = None
) -> list[dict[str, Any]]:
    if isinstance(expr, AxonExprAscribe | AxonExprParen):
        inner = expr.expr if isinstance(expr, AxonExprAscribe) else expr.inner
        return _lower_alias_or_const(inner, out, ctx, guard=guard)
    if isinstance(expr, AxonExprName):
        resolved = _resolve_block_call(expr.name, ctx)
        if resolved is not None and ctx.block_signatures is not None:
            block_name, _ = resolved
            input_names, _ = ctx.block_signatures.get(block_name, ([], []))
            if len(input_names) == 0:
                return _lower_simple_call(expr.name, (), {}, out, ctx, guard=guard)

    node_name = f"n_{ctx.fresh('op')}"
    node: dict[str, Any]
    if isinstance(expr, AxonExprName):
        if expr.name in ctx.symbol_names:
            node = {"_op": "_ir_expr", "value": expr.name, "_bind": out}
        else:
            node = {"_op": "_ir_alias", "_args": expr.name, "_bind": out}
            if isinstance(out, str) and expr.name in ctx.tensor_last_dim:
                ctx.tensor_last_dim[out] = ctx.tensor_last_dim[expr.name]
    elif isinstance(expr, AxonExprInt):
        node = {"_op": "_ir_expr", "value": expr.value, "_bind": out}
    elif isinstance(expr, AxonExprFloat):
        node = {
            "_op": "_ir_expr",
            "value": expr.value,
            "_bind": out,
        }
    elif isinstance(expr, AxonExprBool):
        node = {"_op": "_ir_expr", "value": expr.value, "_bind": out}
    elif isinstance(expr, AxonExprNull):
        node = {"_op": "_ir_expr", "value": None, "_bind": out}
    elif isinstance(expr, AxonExprString):
        node = {"_op": "_ir_expr", "value": _expr_to_runtime_value(expr), "_bind": out}
    elif isinstance(expr, AxonExprPath):
        node = {"_op": "_ir_expr", "value": _expr_to_runtime_value(expr), "_bind": out}
    elif isinstance(expr, AxonExprList):
        node = {
            "_op": "_ir_expr",
            "value": _expr_to_runtime_value(expr),
            "_bind": out,
        }
    else:
        node = {"_op": "_ir_expr", "value": _expr_to_runtime_value(expr), "_bind": out}
    return _with_guard([{node_name: node}], guard)


def _bind_names(out: str | list[str]) -> list[str]:
    return [out] if isinstance(out, str) else list(out)


def _bind_field(names: list[str]) -> str | list[str]:
    return names[0] if len(names) == 1 else names


def _is_null_expr(expr: AxonExpr | None) -> bool:
    return isinstance(expr, AxonExprNull)


def _infer_split_sizes_from_dim_for_bind(
    *,
    source_name: str,
    dim_value: int,
    bind_arity: int,
    ctx: _LowerCtx,
) -> list[Any] | None:
    if bind_arity <= 0:
        return None
    source_shape = ctx.tensor_shape.get(source_name)
    axis_dim: Any = None
    if isinstance(source_shape, tuple):
        rank = len(source_shape)
        axis = dim_value if dim_value >= 0 else rank + dim_value
        if 0 <= axis < rank:
            axis_dim = source_shape[axis]
    if axis_dim is None and dim_value == -1:
        axis_dim = ctx.tensor_last_dim.get(source_name)
    if axis_dim is None:
        return None
    if isinstance(axis_dim, int):
        if axis_dim % bind_arity != 0:
            return None
        piece = axis_dim // bind_arity
        return [piece for _ in range(bind_arity)]
    if isinstance(axis_dim, str):
        token = axis_dim.strip().replace(" ", "")
        if not token:
            return None
        if token.isdigit() or (token[0] in {"+", "-"} and token[1:].isdigit()):
            value = int(token)
            if value % bind_arity != 0:
                return None
            piece = value // bind_arity
            return [piece for _ in range(bind_arity)]
        parts = token.split("*")
        if len(parts) == 2:
            left, right = parts
            if left.isdigit() and int(left) % bind_arity == 0:
                factor = int(left) // bind_arity
                term = right
                piece_str = term if factor == 1 else f"{factor}*{term}"
                return [piece_str for _ in range(bind_arity)]
            if right.isdigit() and int(right) % bind_arity == 0:
                factor = int(right) // bind_arity
                term = left
                piece_str = term if factor == 1 else f"{factor}*{term}"
                return [piece_str for _ in range(bind_arity)]
    return None


def _with_multibind_inference(
    expr: AxonExprCall,
    *,
    bind_arity: int,
    ctx: _LowerCtx,
) -> AxonExprCall:
    resolved = _resolve_block_call(expr.callee, ctx)
    if resolved is None or ctx.block_signatures is None:
        return expr
    block_name, _ = resolved
    input_names, output_names = ctx.block_signatures.get(block_name, ([], []))
    if len(output_names) != 1:
        return expr
    kwargs = dict(expr.kwargs)
    bound_positional: dict[str, AxonExpr] = {}
    for idx, arg in enumerate(expr.args):
        if idx < len(input_names):
            bound_positional[input_names[idx]] = arg
    has_parts = False
    has_sizes = False
    if "parts" in kwargs and not _is_null_expr(_runtime_value_to_expr(kwargs["parts"])):
        has_parts = True
    if "sizes" in kwargs and not _is_null_expr(_runtime_value_to_expr(kwargs["sizes"])):
        has_sizes = True
    if "parts" in bound_positional and not _is_null_expr(bound_positional["parts"]):
        has_parts = True
    if "sizes" in bound_positional and not _is_null_expr(bound_positional["sizes"]):
        has_sizes = True
    if has_parts and has_sizes:
        return expr
    if "sizes" in input_names and not has_sizes:
        dim_raw = kwargs.get("dim", bound_positional.get("dim", AxonExprInt(value=-1)))
        dim_expr = _runtime_value_to_expr(dim_raw)
        dim_folded = _const_fold_dim_expr(dim_expr, ctx, allow_bare_name=True)
        source_expr = bound_positional.get(input_names[0]) if input_names else None
        if (
            isinstance(source_expr, AxonExprName)
            and isinstance(dim_folded, int)
            and not isinstance(dim_folded, bool)
        ):
            inferred_sizes = _infer_split_sizes_from_dim_for_bind(
                source_name=source_expr.name,
                dim_value=dim_folded,
                bind_arity=bind_arity,
                ctx=ctx,
            )
            if inferred_sizes is not None:
                kwargs["sizes"] = AxonExprList(
                    items=tuple(_runtime_value_to_expr(v) for v in inferred_sizes)
                )
                has_sizes = True
    if "parts" in input_names and not has_parts and not has_sizes:
        kwargs["parts"] = AxonExprInt(value=bind_arity)
    if kwargs == expr.kwargs:
        return expr
    return AxonExprCall(callee=expr.callee, args=expr.args, kwargs=kwargs)


def _lower_select_branch(
    *,
    branch_expr: AxonExpr,
    branch_binds: list[str],
    ctx: _LowerCtx,
) -> list[dict[str, Any]]:
    branch_graph: list[dict[str, Any]] = []
    if len(branch_binds) > 1 and isinstance(branch_expr, AxonExprCall | AxonExprName):
        expanded = _expand_call_outputs_for_ternary(branch_expr, list(branch_binds), ctx)
        branch_graph.extend(_lower_expr(branch_expr, expanded, ctx))
        return branch_graph
    branch_out: str | list[str] = _bind_field(branch_binds)
    branch_graph.extend(_lower_expr(branch_expr, branch_out, ctx))
    return branch_graph


def _lower_select_cond(
    cond_expr: AxonExpr,
    *,
    ctx: _LowerCtx,
) -> tuple[list[dict[str, Any]], Any]:
    if isinstance(cond_expr, AxonExprName):
        return [], cond_expr.name
    if isinstance(cond_expr, AxonExprBool):
        return [], cond_expr.value
    if isinstance(cond_expr, AxonExprInt):
        return [], cond_expr.value
    if isinstance(cond_expr, AxonExprFloat):
        return [], cond_expr.value
    if isinstance(cond_expr, AxonExprNull):
        return [], None
    if isinstance(cond_expr, AxonExprParen):
        return _lower_select_cond(cond_expr.inner, ctx=ctx)
    cond_ref = ctx.fresh("cond")
    return _lower_expr(cond_expr, cond_ref, ctx), cond_ref


def _lower_expr(
    expr: AxonExpr,
    out: str | list[str],
    ctx: _LowerCtx,
    *,
    guard: str | None = None,
) -> list[dict[str, Any]]:
    if isinstance(expr, AxonExprBind):
        bind_graph: list[dict[str, Any]] = []
        bind_ref: str
        if isinstance(expr.value, AxonExprName):
            bind_ref = expr.value.name
        else:
            bind_ref = ctx.fresh("bind")
            bind_graph.extend(_lower_expr(expr.value, bind_ref, ctx, guard=guard))
        body = _substitute_expr(expr.body, expr.var, AxonExprName(name=bind_ref))
        bind_graph.extend(_lower_expr(body, out, ctx, guard=guard))
        return bind_graph

    if isinstance(expr, AxonExprDo):
        do_graph: list[dict[str, Any]] = []
        do_outputs: dict[str, str] = {}
        _lower_statements(
            statements=expr.body,
            graph=do_graph,
            outputs=do_outputs,
            returns=(),
            ctx=ctx,
            guard=guard,
        )
        if isinstance(out, list):
            for idx, target in enumerate(out):
                output_name = f"out_{idx}"
                source = do_outputs.get(output_name)
                if source is None:
                    raise ValueError(f"do expression must return value {idx} via `return`")
                if target != source:
                    do_graph.extend(
                        _lower_expr(AxonExprName(name=source), target, ctx, guard=guard)
                    )
            return do_graph
        source0 = do_outputs.get("out_0")
        if source0 is None:
            raise ValueError("do expression must return value 0 via `return`")
        if out != source0:
            do_graph.extend(_lower_expr(AxonExprName(name=source0), out, ctx, guard=guard))
        return do_graph

    if isinstance(expr, AxonExprIf | AxonExprTernary):
        tuple_result_out: str | None = None
        if (
            not isinstance(out, list)
            and isinstance(expr.inferred_arity, int)
            and expr.inferred_arity > 1
        ):
            tuple_result_out = out
            out_names = [ctx.fresh(f"sel_{idx}") for idx in range(expr.inferred_arity)]
        else:
            out_names = _bind_names(out)
        cond_graph, cond_value = _lower_select_cond(expr.cond, ctx=ctx)

        then_binds = [
            ctx.fresh(f"then_{_sanitize_token(name, default='out')}") for name in out_names
        ]
        else_binds = [
            ctx.fresh(f"else_{_sanitize_token(name, default='out')}") for name in out_names
        ]
        then_graph = _lower_select_branch(
            branch_expr=expr.true_expr, branch_binds=then_binds, ctx=ctx
        )
        else_graph = _lower_select_branch(
            branch_expr=expr.false_expr, branch_binds=else_binds, ctx=ctx
        )

        node_name = f"n_{ctx.fresh('select')}"
        node_spec: dict[str, Any] = {
            "_op": "select",
            "_bind": _bind_field(out_names),
            "cond": cond_value,
            "_then": then_graph,
            "_else": else_graph,
            "_then_bind": _bind_field(then_binds),
            "_else_bind": _bind_field(else_binds),
        }
        select_graph = [
            *cond_graph,
            {node_name: node_spec},
        ]
        if tuple_result_out is not None:
            select_graph.extend(
                _lower_expr(
                    AxonExprTuple(items=tuple(AxonExprName(name=name) for name in out_names)),
                    tuple_result_out,
                    ctx,
                    guard=guard,
                )
            )
        return select_graph

    if isinstance(expr, AxonExprPipe):
        pipe_graph: list[dict[str, Any]] = []

        def _resolve_pipe_stage_signature(callee: str) -> ModuleSignature | None:
            base = callee.split("@", 1)[0]
            typed = ctx.typed_signatures or {}
            return (
                typed.get(callee)
                or typed.get(base)
                or (typed.get(base.rsplit(".", 1)[1]) if "." in base else None)
            )

        def _insert_piped_args(
            *,
            callee: str,
            stage_args: list[AxonExpr],
            stage_kwargs: dict[str, AxonKwargValue],
            piped_args: tuple[AxonExpr, ...],
        ) -> list[AxonExpr]:
            sig = _resolve_pipe_stage_signature(callee)
            if sig is None:
                return [*piped_args, *stage_args]
            param_index_by_name = {name: idx for idx, name in enumerate(sig.param_names)}
            bound_by_kwargs: set[int] = set()
            for kw_name in stage_kwargs:
                kw_idx = param_index_by_name.get(kw_name)
                if kw_idx is not None:
                    bound_by_kwargs.add(kw_idx)
            insert_at = 0
            for idx, param_type in enumerate(sig.params):
                if idx in bound_by_kwargs:
                    continue
                if param_type is None or not _is_path_type_expr(param_type):
                    insert_at = idx
                    break
            return [*stage_args[:insert_at], *piped_args, *stage_args[insert_at:]]

        if len(expr.stages) == 0:
            return _lower_expr(expr.value, out, ctx, guard=guard)
        if isinstance(expr.value, AxonExprName):
            pipe_ref: str | list[str] = expr.value.name
        else:
            first_out: str | list[str] = _pipeline_temp_out(expr.value, ctx)
            pipe_graph.extend(_lower_expr(expr.value, first_out, ctx, guard=guard))
            pipe_ref = first_out

        for idx, stage in enumerate(expr.stages, start=1):
            if isinstance(stage, AxonExprCall):
                stage_callee = stage.callee
                stage_args = list(stage.args)
                stage_kwargs = dict(stage.kwargs)
            elif isinstance(stage, AxonExprName):
                stage_callee = stage.name
                stage_args = []
                stage_kwargs = {}
            else:
                raise ValueError("pipeline stage must be a call or name")
            next_out: str | list[str] = (
                out if idx == len(expr.stages) else _pipeline_temp_out(stage, ctx)
            )
            piped_refs = [pipe_ref] if isinstance(pipe_ref, str) else list(pipe_ref)
            piped_args = tuple(AxonExprName(name=ref) for ref in piped_refs)
            if stage_args:
                same_prefix = len(stage_args) >= len(piped_refs)
                if same_prefix:
                    for idx_ref, ref in enumerate(piped_refs):
                        head = stage_args[idx_ref]
                        if not isinstance(head, AxonExprName) or head.name != ref:
                            same_prefix = False
                            break
                if same_prefix:
                    stage_args = stage_args[len(piped_refs) :]
            rewritten_args = _insert_piped_args(
                callee=stage_callee,
                stage_args=stage_args,
                stage_kwargs=stage_kwargs,
                piped_args=piped_args,
            )
            call_args = tuple(rewritten_args)
            pipe_graph.extend(
                _lower_simple_call(
                    stage_callee,
                    call_args,
                    stage_kwargs,
                    next_out,
                    ctx,
                    guard=guard,
                    call_expr=stage if isinstance(stage, AxonExprCall) else None,
                )
            )
            pipe_ref = next_out
        return pipe_graph

    if isinstance(expr, AxonExprCall):
        return _lower_simple_call(
            expr.callee, expr.args, expr.kwargs, out, ctx, guard=guard, call_expr=expr
        )

    if isinstance(expr, AxonExprBinary) and expr.op in {"+", "*"}:
        if not (_expr_is_tensorish(expr.left, ctx) or _expr_is_tensorish(expr.right, ctx)):
            return _lower_alias_or_const(expr, out, ctx, guard=guard)
        op_name = "add" if expr.op == "+" else "mul"
        binary_graph: list[dict[str, Any]] = []
        if isinstance(expr.left, AxonExprName) and expr.left.name not in ctx.symbol_names:
            left_ref = expr.left.name
        else:
            left_ref = ctx.fresh("bin")
            binary_graph.extend(_lower_expr(expr.left, left_ref, ctx, guard=guard))
        if isinstance(expr.right, AxonExprName) and expr.right.name not in ctx.symbol_names:
            right_ref = expr.right.name
        else:
            right_ref = ctx.fresh("bin")
            binary_graph.extend(_lower_expr(expr.right, right_ref, ctx, guard=guard))
        binary_graph.extend(
            _lower_simple_call(
                op_name,
                (AxonExprName(name=left_ref), AxonExprName(name=right_ref)),
                {},
                out,
                ctx,
                guard=guard,
                enforce_source_primitive_syntax=False,
            )
        )
        return binary_graph

    if isinstance(expr, AxonExprTuple):
        if not isinstance(out, list):
            # Preserve tuple-valued expressions (e.g. CacheLayer) as a single
            # runtime expression value when bound to one target.
            return _lower_alias_or_const(expr, out, ctx, guard=guard)
        if len(out) != len(expr.items):
            raise ValueError("tuple expression arity must match binding targets")
        tuple_graph: list[dict[str, Any]] = []
        for name, item in zip(out, expr.items, strict=True):
            tuple_graph.extend(_lower_expr(item, name, ctx, guard=guard))
        return tuple_graph

    if isinstance(expr, AxonExprParen):
        return _lower_expr(expr.inner, out, ctx, guard=guard)

    if isinstance(expr, AxonExprLambda):
        raise ValueError("lambda expression cannot be lowered directly")
    return _lower_alias_or_const(expr, out, ctx, guard=guard)


def _module_return_names(module: AxonModule) -> tuple[str, ...]:
    if module.returns:
        return module.returns
    for stmt in reversed(module.statements):
        if isinstance(stmt, AxonReturn):
            inferred: list[str] = []
            for idx, value in enumerate(stmt.values):
                maybe_name = _expr_name(value)
                inferred.append(maybe_name if maybe_name is not None else f"out_{idx}")
            if inferred:
                return tuple(inferred)
    return ()


def _module_output_names_from_declared_arity(
    module: AxonModule, *, expected_arity: int
) -> tuple[str, ...]:
    inferred = _module_return_names(module)
    if len(inferred) == expected_arity:
        return inferred
    if expected_arity <= 0:
        return ()
    return tuple(f"out_{idx}" for idx in range(expected_arity))


def _module_return_last_dims(module: AxonModule, returns: tuple[str, ...]) -> dict[str, Any]:
    return_shape = _shape_from_type_expr(module.return_type_expr) or module.return_shape
    if not returns or return_shape is None or len(return_shape) == 0:
        return {}
    if len(returns) != 1:
        return {}
    return {returns[0]: return_shape[-1]}


def _shape_from_type_expr(type_expr: TypeExpr | None) -> tuple[Any, ...] | None:
    if isinstance(type_expr, TypeTensor):
        return tuple(type_expr.dims)
    return None


def _module_param_last_dims(module: AxonModule) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for param in module.params:
        shape = _shape_from_type_expr(param.type_expr) or param.shape
        if shape is None or len(shape) == 0:
            continue
        out[param.name] = shape[-1]
    return out


def _module_param_shapes(module: AxonModule) -> dict[str, tuple[Any, ...]]:
    out: dict[str, tuple[Any, ...]] = {}
    for param in module.params:
        shape = _shape_from_type_expr(param.type_expr) or param.shape
        if shape is None:
            continue
        out[param.name] = tuple(shape)
    return out


def _module_return_shapes(
    module: AxonModule, returns: tuple[str, ...]
) -> dict[str, tuple[Any, ...]]:
    return_shape = _shape_from_type_expr(module.return_type_expr) or module.return_shape
    if not returns or return_shape is None:
        return {}
    if len(returns) != 1:
        return {}
    return {returns[0]: tuple(return_shape)}


def _module_return_heads(module: AxonModule, returns: tuple[str, ...]) -> dict[str, Any]:
    return_shape = _shape_from_type_expr(module.return_type_expr) or module.return_shape
    if not returns or return_shape is None or len(return_shape) < 2:
        return {}
    if len(returns) != 1:
        return {}
    return {returns[0]: return_shape[1]}


def _module_inputs(module: AxonModule) -> dict[str, dict[str, bool]]:
    inputs = {param.name: {"optional": param.optional} for param in module.params}
    for path_param in module.path_params:
        inputs[path_param] = {"optional": False}
    if not module.path_params and module.path_param is not None:
        inputs[module.path_param] = {"optional": False}
    return inputs


def _module_initial_dims(module: AxonModule, returns: tuple[str, ...]) -> dict[str, Any]:
    initial_dims = {
        param.name: shape[-1]
        for param in module.params
        for shape in (_shape_from_type_expr(param.type_expr) or param.shape,)
        if shape is not None and len(shape) > 0
    }
    initial_dims.update(_module_return_last_dims(module, returns))
    return initial_dims


def _module_initial_shapes(
    module: AxonModule, returns: tuple[str, ...]
) -> dict[str, tuple[Any, ...]]:
    initial_shapes = {
        param.name: tuple(shape)
        for param in module.params
        for shape in (_shape_from_type_expr(param.type_expr) or param.shape,)
        if shape is not None
    }
    initial_shapes.update(_module_return_shapes(module, returns))
    return initial_shapes


def _module_initial_heads(module: AxonModule, returns: tuple[str, ...]) -> dict[str, Any]:
    initial_heads = {
        param.name: shape[1]
        for param in module.params
        for shape in (_shape_from_type_expr(param.type_expr) or param.shape,)
        if shape is not None and len(shape) >= 2
    }
    initial_heads.update(_module_return_heads(module, returns))
    return initial_heads


def _module_path_param_names(module: AxonModule) -> set[str]:
    names = {p for p in module.path_params if isinstance(p, str)}
    if not names and isinstance(module.path_param, str):
        names.add(module.path_param)
    return names


def _type_dim_names(tp: TypeExpr) -> set[str]:
    if isinstance(tp, TypeOptional):
        return _type_dim_names(tp.inner)
    if isinstance(tp, TypeTensor):
        names: set[str] = set()
        for dim in tp.dims:
            names.update(dim_token_names(dim))
        return names
    if isinstance(tp, TypeList):
        return _type_dim_names(tp.item)
    if isinstance(tp, TypeTuple):
        tuple_names: set[str] = set()
        for item in tp.items:
            tuple_names.update(_type_dim_names(item))
        return tuple_names
    return set()


def _signature_dim_names(signature: ModuleSignature) -> set[str]:
    names: set[str] = set()
    for param in signature.params:
        if param is not None:
            names.update(_type_dim_names(param))
    for ret in signature.returns:
        names.update(_type_dim_names(ret))
    return names


def _ensure_outputs_from_returns(outputs: dict[str, str], returns: tuple[str, ...]) -> None:
    if outputs:
        return
    for name in returns:
        outputs[name] = name


def _extract_primitive_aliases(modules: tuple[AxonModule, ...]) -> dict[str, tuple[str, int]]:
    def _module_path_param_names(module: AxonModule) -> tuple[str, ...]:
        if module.path_params:
            return tuple(module.path_params)
        if module.path_param is not None:
            return (module.path_param,)
        return ()

    def _is_identity_alias_call(module: AxonModule, value: AxonExprCall) -> bool:
        # Defaulted/optional wrapper params change call-surface semantics
        # (especially kwargs), so do not collapse those wrappers to primitives.
        if any(param.optional for param in module.params):
            return False
        if any(param.default_expr is not None for param in module.params):
            return False
        callee_parts = value.callee.split("@")
        callee_path_params = tuple(callee_parts[1:])
        module_path_params = _module_path_param_names(module)
        if callee_path_params != module_path_params:
            return False
        if value.kwargs:
            return False
        if len(value.args) != len(module.params):
            return False
        for arg_expr, param in zip(value.args, module.params, strict=True):
            if not isinstance(arg_expr, AxonExprName) or arg_expr.name != param.name:
                return False
        return True

    def _module_signature_fingerprint(module: AxonModule) -> tuple[Any, ...]:
        path_params = (
            tuple(module.path_params)
            if module.path_params
            else tuple([module.path_param] if module.path_param is not None else [])
        )
        return (
            path_params,
            tuple(param.name for param in module.params),
            tuple(param.type_expr for param in module.params),
            tuple(bool(param.optional) for param in module.params),
            tuple(param.default_expr for param in module.params),
            tuple(
                tuple(param.shape) if param.shape is not None else None for param in module.params
            ),
            module.return_type_expr,
            tuple(module.return_shape) if module.return_shape is not None else None,
        )

    allowed_namespaces = set(_BUILTIN_MODULE_NAMESPACES)
    modules_by_name = {module.name: module for module in modules}
    called_with_kwargs: set[str] = set()
    for module in modules:
        for stmt in module.statements:
            values: tuple[AxonExpr, ...] = ()
            if isinstance(stmt, AxonBind):
                values = (stmt.expr,)
            elif isinstance(stmt, AxonReturn):
                values = tuple(stmt.values)
            if not values:
                continue
            stack: list[AxonExpr] = list(values)
            while stack:
                expr = stack.pop()
                if isinstance(expr, AxonExprCall):
                    if expr.kwargs:
                        called_with_kwargs.add(expr.callee.split("@", 1)[0])
                    stack.extend(expr.args)
                    for kw_value in expr.kwargs.values():
                        if isinstance(kw_value, AxonExpr):
                            stack.append(kw_value)
                elif isinstance(expr, AxonExprParen):
                    stack.append(expr.inner)
                elif isinstance(expr, AxonExprPipe):
                    stack.append(expr.value)
                    stack.extend(expr.stages)
                elif isinstance(expr, AxonExprBinary):
                    stack.append(expr.left)
                    stack.append(expr.right)
                elif isinstance(expr, AxonExprIf):
                    stack.append(expr.cond)
                    stack.append(expr.true_expr)
                    stack.append(expr.false_expr)
                elif isinstance(expr, AxonExprTernary):
                    stack.append(expr.cond)
                    stack.append(expr.true_expr)
                    stack.append(expr.false_expr)
                elif isinstance(expr, AxonExprList):
                    stack.extend(expr.items)
                elif isinstance(expr, AxonExprTuple):
                    stack.extend(expr.items)
                elif isinstance(expr, AxonExprDo):
                    for do_stmt in expr.body:
                        if isinstance(do_stmt, AxonBind):
                            stack.append(do_stmt.expr)
                        elif isinstance(do_stmt, AxonReturn):
                            stack.extend(do_stmt.values)
                elif isinstance(expr, AxonExprLambda):
                    stack.append(expr.body)
    direct_aliases: dict[str, tuple[str, int]] = {}
    for module in modules:
        if not isinstance(module.name, str) or "." not in module.name:
            continue
        namespace, _ = module.name.split(".", 1)
        if namespace not in allowed_namespaces:
            continue
        if len(module.statements) != 1:
            continue
        stmt = module.statements[0]
        if not isinstance(stmt, AxonReturn) or len(stmt.values) != 1:
            continue
        value = stmt.values[0]
        if not isinstance(value, AxonExprCall):
            continue
        if not _is_identity_alias_call(module, value):
            continue
        module_base_name = module.name.rsplit(".", 1)[-1]
        if module.name in called_with_kwargs or module_base_name in called_with_kwargs:
            continue
        target_base = value.callee.split("@", 1)[0]
        target_module = modules_by_name.get(target_base)
        if target_module is not None and (
            _module_signature_fingerprint(module) != _module_signature_fingerprint(target_module)
        ):
            continue
        direct_aliases[module.name] = (target_base, len(_module_path_param_names(module)))

    aliases: dict[str, tuple[str, int]] = {}
    for name, (target_base, expected_path_count) in direct_aliases.items():
        seen: set[str] = set()
        resolved = target_base
        while not resolved.startswith("_"):
            if resolved in seen:
                break
            seen.add(resolved)
            next_alias = direct_aliases.get(resolved)
            if next_alias is None:
                break
            next_base, next_path_count = next_alias
            if next_path_count != expected_path_count:
                break
            resolved = next_base
        if resolved.startswith("_"):
            aliases[name] = (resolved, expected_path_count)
    return aliases


def _extract_prelude_aliases(modules: tuple[AxonModule, ...]) -> dict[str, tuple[str, int]]:
    aliases: dict[str, tuple[str, int]] = {}
    primitive_aliases = _extract_primitive_aliases(modules)
    for full_name, alias in primitive_aliases.items():
        if not full_name.startswith("Prelude."):
            continue
        member_name = full_name.split(".", 1)[1]
        if not member_name:
            continue
        aliases[member_name] = alias
    return aliases


def _collect_imported_symbol_values(
    module: AxonModule,
    modules_by_name: dict[str, AxonModule] | None,
) -> dict[str, Any]:
    if not module.imported_members or not modules_by_name:
        return {}
    collected: dict[str, Any] = {}
    for namespace, members in module.imported_members.items():
        imported_module = modules_by_name.get(namespace)
        if imported_module is None or not isinstance(imported_module.symbols, dict):
            continue
        for member in members:
            if member in collected:
                continue
            if member in imported_module.symbols:
                collected[member] = imported_module.symbols[member]
    return collected


def _new_lower_ctx(
    *,
    module: AxonModule,
    returns: tuple[str, ...],
    signatures: dict[str, tuple[list[str], list[str]]] | None,
    block_path_params: dict[str, tuple[str, ...]] | None,
    block_param_last_dims: dict[str, dict[str, Any]] | None,
    block_output_last_dims: dict[str, dict[str, Any]] | None,
    block_param_shapes: dict[str, dict[str, tuple[Any, ...]]] | None = None,
    block_output_shapes: dict[str, dict[str, tuple[Any, ...]]] | None = None,
    block_optional_inputs: dict[str, set[str]] | None = None,
    block_default_inputs: dict[str, dict[str, AxonExpr]] | None = None,
    typed_signatures: dict[str, ModuleSignature] | None = None,
    implicit_prelude_members: set[str] | None = None,
    prelude_aliases: dict[str, tuple[str, int]] | None = None,
    primitive_aliases: dict[str, tuple[str, int]] | None = None,
    imported_symbol_values: dict[str, Any] | None = None,
    module_dim_symbol_names: set[str] | None = None,
) -> _LowerCtx:
    imported_member_namespaces: dict[str, set[str]] = {}
    if module.imported_members:
        for namespace, members in module.imported_members.items():
            for member in members:
                bucket = imported_member_namespaces.setdefault(member, set())
                bucket.add(namespace)
    if implicit_prelude_members:
        for member in implicit_prelude_members:
            if member in imported_member_namespaces:
                continue
            imported_member_namespaces.setdefault(member, set())
    module_namespace = module.name.split(".", 1)[0] if "." in module.name else module.name
    implicit_builtin_namespaces: set[str] = set(_BUILTIN_MODULE_NAMESPACES)
    if module_namespace in implicit_builtin_namespaces:
        implicit_builtin_namespaces.remove(module_namespace)
    imported_namespaces = set(module.imports) | implicit_builtin_namespaces
    if signatures:
        for member_name in imported_member_namespaces:
            if any(name.startswith(f"{member_name}.") for name in signatures):
                imported_namespaces.add(member_name)
    symbol_values = dict(imported_symbol_values or {})
    if isinstance(module.symbols, dict):
        symbol_values.update(module.symbols)
    dim_symbol_names = set(module_dim_symbol_names or set())
    runtime_bound_symbol_names: set[str] = set()
    for stmt in module.statements:
        if isinstance(stmt, AxonBind):
            for target in stmt.targets:
                if target != "_" and target in symbol_values:
                    runtime_bound_symbol_names.add(target)
    return _LowerCtx(
        block_signatures=signatures,
        block_path_params=block_path_params,
        block_param_last_dims=block_param_last_dims,
        block_output_last_dims=block_output_last_dims,
        block_param_shapes=block_param_shapes,
        block_output_shapes=block_output_shapes,
        block_optional_inputs=block_optional_inputs,
        block_default_inputs=block_default_inputs,
        typed_signatures=typed_signatures,
        tensor_last_dim=_module_initial_dims(module, returns),
        tensor_heads=_module_initial_heads(module, returns),
        tensor_shape=_module_initial_shapes(module, returns),
        path_param_names=_module_path_param_names(module),
        imported_namespaces=imported_namespaces,
        imported_member_namespaces=imported_member_namespaces,
        prelude_aliases=dict(prelude_aliases or {}),
        primitive_aliases=dict(primitive_aliases or {}),
        current_module=module.name,
        param_names={param.name for param in module.params},
        symbol_names=set(symbol_values.keys()) | dim_symbol_names,
        dim_symbol_names=dim_symbol_names,
        symbol_values=symbol_values,
        runtime_bound_symbol_names=runtime_bound_symbol_names,
    )


def lower_axon_module_to_synapse_block(module: AxonModule) -> dict[str, Any]:
    validate_typed_axon_file(_wrap_modules_as_file((module,)), main_module=module.name)
    typed_signatures = _build_module_signatures_for_closed_program(
        (module,), main_module=module.name
    )
    inputs = _module_inputs(module)
    graph: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    returns = _module_return_names(module)
    ctx = _new_lower_ctx(
        module=module,
        returns=returns,
        signatures={},
        block_path_params={
            module.name: module.path_params
            if module.path_params
            else tuple([module.path_param] if module.path_param is not None else [])
        },
        block_param_last_dims={module.name: _module_param_last_dims(module)},
        block_output_last_dims={module.name: _module_return_last_dims(module, returns)},
        block_param_shapes={module.name: _module_param_shapes(module)},
        block_output_shapes={module.name: _module_return_shapes(module, returns)},
        implicit_prelude_members=set(),
        prelude_aliases={},
        imported_symbol_values={},
        typed_signatures=typed_signatures,
        module_dim_symbol_names=_signature_dim_names(typed_signatures[module.name]),
    )

    _lower_statements(
        statements=module.statements,
        graph=graph,
        outputs=outputs,
        returns=returns,
        ctx=ctx,
    )

    _ensure_outputs_from_returns(outputs, returns)

    return {"inputs": inputs, "graph": graph, "outputs": outputs}


def lower_axon_module_to_synapse_spec(module: AxonModule) -> dict[str, Any]:
    block = lower_axon_module_to_synapse_block(module)
    model: dict[str, Any] = {
        "inputs": block["inputs"],
        "graph": block["graph"],
        "outputs": block["outputs"],
    }
    if module.symbols:
        model["symbols"] = dict(module.symbols)
    if module.pragmas:
        model["meta"] = dict(module.pragmas)
    spec = {
        "synapse": 1,
        "model": model,
    }
    inferred_block_io_types = infer_block_io_types_from_modules(
        spec=spec,
        modules=(module,),
        selected_main=module.name,
    )
    annotate_spec_with_block_io_types(spec, block_io_types=inferred_block_io_types)
    return spec


def _lower_statements(
    *,
    statements: tuple[AxonStatement, ...],
    graph: list[dict[str, Any]],
    outputs: dict[str, str],
    returns: tuple[str, ...],
    ctx: _LowerCtx,
    guard: str | None = None,
) -> None:
    def _record_bind_type(targets: tuple[str, ...], type_expr: TypeExpr | None) -> None:
        def record_shape(target: str, shape: tuple[Any, ...]) -> None:
            current = ctx.tensor_shape.get(target)
            if current is not None and len(current) == len(shape):
                return
            ctx.tensor_shape[target] = shape
            if shape:
                ctx.tensor_last_dim[target] = shape[-1]

        if type_expr is None:
            return
        if len(targets) == 1:
            shape = _shape_from_type_expr(type_expr)
            if shape is not None:
                record_shape(targets[0], shape)
            return
        if not isinstance(type_expr, TypeTuple):
            return
        for target, item_type in zip(targets, type_expr.items, strict=False):
            shape = _shape_from_type_expr(item_type)
            if shape is None:
                continue
            record_shape(target, shape)

    for stmt in statements:
        if isinstance(stmt, AxonRepeat):
            raise ValueError("lowering requires flat Axon; for/repeat must be flattened first")
            continue
        if isinstance(stmt, AxonYield):
            raise ValueError("yield statement must only appear as final statement in for-loop body")

        if isinstance(stmt, AxonScopeBind):
            raise ValueError("lowering requires flat Axon; scope blocks must be flattened first")
            continue

        if isinstance(stmt, AxonBind):
            if len(stmt.targets) == 1:
                graph.extend(_lower_expr(stmt.expr, stmt.targets[0], ctx, guard=guard))
                if not isinstance(stmt.expr, AxonExprCall):
                    _record_bind_type(stmt.targets, stmt.expr.inferred_type)
                continue
            if isinstance(stmt.expr, AxonExprCall) and ctx.block_signatures is not None:
                call_expr = _with_multibind_inference(
                    stmt.expr,
                    bind_arity=len(stmt.targets),
                    ctx=ctx,
                )
                resolved = _resolve_block_call(call_expr.callee, ctx)
                if resolved is not None:
                    block_name, _ = resolved
                    input_names, output_names = ctx.block_signatures.get(block_name, ([], []))
                    if len(output_names) == 1 and len(stmt.targets) > 1:
                        unpack_src = ctx.fresh("unpack")
                        graph.extend(_lower_expr(call_expr, unpack_src, ctx, guard=guard))
                        graph.extend(
                            _lower_expr(
                                AxonExprName(name=unpack_src), list(stmt.targets), ctx, guard=guard
                            )
                        )
                        continue
                    if len(output_names) == 1 and (
                        "parts" in input_names or "sizes" in input_names
                    ):
                        unpack_src = ctx.fresh("unpack")
                        graph.extend(_lower_expr(call_expr, unpack_src, ctx, guard=guard))
                        for idx, target in enumerate(stmt.targets):
                            if target == "_":
                                continue
                            graph.extend(
                                _lower_expr(
                                    AxonExprCall(
                                        callee="list_index",
                                        args=(
                                            AxonExprName(name=unpack_src),
                                            AxonExprInt(value=idx),
                                        ),
                                        kwargs={},
                                    ),
                                    target,
                                    ctx,
                                    guard=guard,
                                )
                            )
                        continue

            out: str | list[str] = list(stmt.targets)
            graph.extend(_lower_expr(stmt.expr, out, ctx, guard=guard))
            if not isinstance(stmt.expr, AxonExprCall):
                _record_bind_type(stmt.targets, stmt.expr.inferred_type)
            continue

        if isinstance(stmt, AxonReturn):
            if len(stmt.values) == 1 and len(returns) > 1:
                graph.extend(_lower_expr(stmt.values[0], list(returns), ctx, guard=guard))
                for name in returns:
                    outputs[name] = name
                continue
            for idx, value in enumerate(stmt.values):
                output_name = returns[idx] if idx < len(returns) else f"out_{idx}"
                maybe_name = _expr_name(value)
                if maybe_name is not None:
                    outputs[output_name] = maybe_name
                    continue
                graph.extend(_lower_expr(value, output_name, ctx, guard=guard))
                outputs[output_name] = output_name
            continue


def _as_concrete_path(value: Any) -> str | None:
    if isinstance(value, dict):
        if value.get("_expr") == "string":
            raw = value.get("value")
            if isinstance(raw, str):
                value = raw
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token:
        return None
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        inner = token[1:-1].strip()
        return inner or None
    return token


def _normalize_bound_params_on_node(
    node_spec: dict[str, Any],
    *,
    base_path: str,
    inherited_path_bindings: dict[str, str] | None = None,
) -> None:
    def _path_override_arg_token(param_name: str) -> str | None:
        op_name = node_spec.get("_op")
        raw_args = node_spec.get("_args")
        args: list[Any]
        if isinstance(raw_args, list):
            args = raw_args
        elif raw_args is None:
            args = []
        else:
            args = [raw_args]
        override_idx: int | None = None
        if op_name == "linear":
            if param_name == "weight":
                override_idx = 5
            elif param_name == "bias":
                override_idx = 6
        elif op_name == "layernorm":
            if param_name == "weight":
                override_idx = 3
            elif param_name == "bias":
                override_idx = 5
        if override_idx is None or override_idx >= len(args):
            return None
        token = args[override_idx]
        if not isinstance(token, str):
            return None
        stripped = token.strip()
        if not stripped or stripped.lower() == "null":
            return None
        return stripped

    explicit_params = node_spec.get("_params")
    params = dict(explicit_params) if isinstance(explicit_params, dict) else {}
    inherited = inherited_path_bindings if isinstance(inherited_path_bindings, dict) else {}
    is_absolute = base_path.startswith("@@")
    normalized_base = base_path[2:] if is_absolute else base_path
    if isinstance(normalized_base, str):
        mapped_base = inherited.get(normalized_base)
        if isinstance(mapped_base, str) and mapped_base.strip():
            normalized_base = mapped_base.strip()
    if is_absolute:
        node_spec["_abs_path"] = _path_source_to_runtime_value(base_path)
        node_spec.pop("_scope", None)
    for param_name in _path_bound_param_names(node_spec):
        explicit_value = params.get(param_name)
        if not (isinstance(explicit_value, str) and explicit_value.strip()):
            node_value = node_spec.get(param_name)
            if isinstance(node_value, str) and node_value.strip():
                explicit_value = node_value
        if not (isinstance(explicit_value, str) and explicit_value.strip()):
            override_token = _path_override_arg_token(param_name)
            if override_token is not None:
                explicit_value = override_token
        if isinstance(explicit_value, str) and explicit_value.strip():
            token = explicit_value.strip()
            original_token = token
            # If the explicit value refers to another bound path input name,
            # resolve through inherited/specialized `_params` first.
            mapped = params.get(token)
            if isinstance(mapped, str) and mapped.strip():
                token = mapped.strip()
                original_token = token
            else:
                # Token may refer to another node input/default (for example
                # `scale=scale_path` where `scale_path` defaulted to `@weight`).
                token_from_node = _as_concrete_path(node_spec.get(token))
                if isinstance(token_from_node, str) and token_from_node.strip():
                    token = token_from_node.strip()
                    original_token = token
            # Resolve tokens prefixed by inherited path-param names (e.g. `path.weight`).
            head, sep, tail = token.partition(".")
            mapped_head = inherited.get(head)
            if isinstance(mapped_head, str) and mapped_head.strip():
                token = f"{mapped_head.strip()}{sep}{tail}" if sep else mapped_head.strip()
                original_token = token
            explicit_absolute = token.startswith("@@")
            if explicit_absolute:
                params[param_name] = _path_source_to_runtime_value(original_token)
                node_spec[param_name] = param_name
                continue
            elif token.startswith("@"):
                token = token[1:]
            if is_absolute or explicit_absolute:
                params[param_name] = token or param_name
            else:
                params[param_name] = token if "." in token else f"{normalized_base}.{token}"
            # Canonicalize the direct kwarg field to its param-name sentinel so
            # runtime path inference takes the normalized `_params` mapping.
            node_spec[param_name] = param_name
            continue
        params[param_name] = param_name if is_absolute else f"{normalized_base}.{param_name}"
        node_spec[param_name] = param_name
    node_spec["_params"] = params
    node_spec.pop("param_base", None)


def _path_bound_param_names(node_spec: dict[str, Any]) -> list[str]:
    def _raw_arg(index: int) -> Any:
        raw_args = node_spec.get("_args")
        if isinstance(raw_args, list):
            return raw_args[index] if index < len(raw_args) else None
        if index == 0:
            return raw_args
        return None

    def _parse_bool(value: Any, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            token = value.strip().lower()
            if token == "true":
                return True
            if token in {"false", "null"}:
                return False
            # Symbolic/non-literal bool expression: keep bias path binding enabled.
            return True
        if isinstance(value, dict):
            kind = value.get("_expr")
            if kind == "bool":
                parsed = value.get("value")
                if isinstance(parsed, bool):
                    return parsed
            if kind == "name":
                return True
        return bool(value)

    def _has_explicit_path_arg(name: str) -> bool:
        raw = node_spec.get(name)
        if raw is None:
            return False
        if isinstance(raw, str):
            token = raw.strip()
            return bool(token) and token.lower() != "null"
        return True

    op = node_spec.get("_op")
    if op == "embedding":
        return ["weight"]
    if op == "linear":
        names = ["weight_path" if _has_explicit_path_arg("weight_path") else "weight"]
        raw_bias = node_spec.get("bias", _raw_arg(2))
        if _parse_bool(raw_bias, default=False):
            names.append("bias_path" if _has_explicit_path_arg("bias_path") else "bias")
        return names
    if op == "rmsnorm":
        return ["weight"]
    if op == "layernorm":
        names = ["weight_path" if _has_explicit_path_arg("weight_path") else "weight"]
        raw_bias = node_spec.get("bias", _raw_arg(4))
        has_bias = _parse_bool(raw_bias, default=True)
        if has_bias:
            names.append("bias_path" if _has_explicit_path_arg("bias_path") else "bias")
        return names
    if op == "activations_xielu":
        return ["alpha_p", "alpha_n", "beta", "eps"]
    raise ValueError(f"unsupported param_base resolution for op {op!r}")


def _canonical_type_source_block_name(block_name: str) -> str:
    if "__" in block_name:
        return block_name.split("__", 1)[0]
    return block_name


def _finalize_block_io_types_for_model(
    *,
    model: dict[str, Any],
    inferred: dict[str, dict[str, dict[str, str]]],
) -> dict[str, dict[str, dict[str, str]]]:
    final_types: dict[str, dict[str, dict[str, str]]] = {}
    main_types = inferred.get("main")
    if isinstance(main_types, dict):
        final_types["main"] = main_types

    blocks = model.get("blocks")
    if not isinstance(blocks, dict):
        return final_types

    for block_name in blocks:
        if not isinstance(block_name, str):
            continue
        direct = inferred.get(block_name)
        if isinstance(direct, dict):
            final_types[block_name] = direct
            continue
        source_name = _canonical_type_source_block_name(block_name)
        source = inferred.get(source_name)
        if isinstance(source, dict):
            final_types[block_name] = source
    return final_types


def lower_axon_program_to_synapse_spec(
    modules: AxonFile | tuple[AxonModule, ...],
    *,
    main_module: str | None = None,
    optimize: bool = True,
) -> dict[str, Any]:
    program = _prepare_program_for_lowering(
        modules, main_module=main_module, optimize=optimize
    )
    validate_typed_axon_file(program, main_module=main_module)
    program_symbol_values = _lower_program_symbol_values(program)
    modules = program.modules
    typed_signatures = _build_module_signatures_for_closed_program(modules, main_module=main_module)
    if not modules:
        raise ValueError("Axon program must contain at least one module")

    by_name = {module.name: module for module in modules}
    if len(by_name) != len(modules):
        raise ValueError("Axon program contains duplicate module names")

    main_name = modules[-1].name if main_module is None else main_module
    if main_name not in by_name:
        raise ValueError(f"Unknown main module: {main_name!r}")

    signatures: dict[str, tuple[list[str], list[str]]] = {}
    block_path_params: dict[str, tuple[str, ...]] = {}
    block_param_last_dims: dict[str, dict[str, Any]] = {}
    block_output_last_dims: dict[str, dict[str, Any]] = {}
    block_param_shapes: dict[str, dict[str, tuple[Any, ...]]] = {}
    block_output_shapes: dict[str, dict[str, tuple[Any, ...]]] = {}
    block_optional_inputs: dict[str, set[str]] = {}
    block_default_inputs: dict[str, dict[str, AxonExpr]] = {}
    module_output_names: dict[str, tuple[str, ...]] = {}
    module_dim_symbol_names: dict[str, set[str]] = {}
    for module in modules:
        input_names = [param.name for param in module.params]
        path_params = module.path_params
        if not path_params and module.path_param is not None:
            path_params = (module.path_param,)
        input_names.extend(path_params)
        declared_arity = len(typed_signatures[module.name].returns)
        output_names = list(
            _module_output_names_from_declared_arity(module, expected_arity=declared_arity)
        )
        module_output_names[module.name] = tuple(output_names)
        signatures[module.name] = (input_names, output_names)
        block_path_params[module.name] = path_params
        block_param_last_dims[module.name] = _module_param_last_dims(module)
        block_output_last_dims[module.name] = _module_return_last_dims(module, tuple(output_names))
        block_param_shapes[module.name] = _module_param_shapes(module)
        block_output_shapes[module.name] = _module_return_shapes(module, tuple(output_names))
        block_optional_inputs[module.name] = {
            param.name for param in module.params if bool(param.optional)
        }
        block_default_inputs[module.name] = {
            param.name: param.default_expr
            for param in module.params
            if isinstance(param.default_expr, AxonExpr)
        }
        module_dim_symbol_names[module.name] = _signature_dim_names(typed_signatures[module.name])
    primitive_aliases = _extract_primitive_aliases(modules)
    prelude_aliases = _extract_prelude_aliases(modules)
    implicit_prelude_members = {
        name.split(".", 1)[1]
        for name in signatures
        if isinstance(name, str) and name.startswith("Prelude.") and "." in name
    }

    main = by_name[main_name]
    main_returns = module_output_names[main.name]
    main_inputs = _module_inputs(main)
    main_graph: list[dict[str, Any]] = []
    main_outputs: dict[str, str] = {}
    _lower_statements(
        statements=main.statements,
        graph=main_graph,
        outputs=main_outputs,
        returns=main_returns,
        ctx=_new_lower_ctx(
            module=main,
            returns=main_returns,
            signatures=signatures,
            block_path_params=block_path_params,
            block_param_last_dims=block_param_last_dims,
            block_output_last_dims=block_output_last_dims,
            block_param_shapes=block_param_shapes,
            block_output_shapes=block_output_shapes,
            block_optional_inputs=block_optional_inputs,
            block_default_inputs=block_default_inputs,
            typed_signatures=typed_signatures,
            implicit_prelude_members=implicit_prelude_members,
            prelude_aliases=prelude_aliases,
            primitive_aliases=primitive_aliases,
            imported_symbol_values=_collect_imported_symbol_values(main, by_name),
            module_dim_symbol_names=module_dim_symbol_names.get(main.name, set()),
        ),
    )
    _ensure_outputs_from_returns(main_outputs, main_returns)
    model: dict[str, Any] = {"inputs": main_inputs, "graph": main_graph, "outputs": main_outputs}
    combined_symbols: dict[str, Any] = {}
    for key, value in program_symbol_values.items():
        combined_symbols[key] = value
    for module in modules:
        if not isinstance(module.symbols, dict):
            continue
        for key, value in module.symbols.items():
            if key not in combined_symbols:
                combined_symbols[key] = value
                continue
            existing = combined_symbols[key]
            if existing is None and value is not None:
                combined_symbols[key] = value
    if combined_symbols:
        model["symbols"] = combined_symbols
    if main.pragmas:
        model["meta"] = dict(main.pragmas)
    spec: dict[str, Any] = {"synapse": 1, "model": model}

    blocks: dict[str, Any] = {}
    for module in modules:
        if module.name == main_name:
            continue
        block_inputs = _module_inputs(module)
        block_returns = module_output_names[module.name]
        block_graph: list[dict[str, Any]] = []
        block_outputs: dict[str, str] = {}
        _lower_statements(
            statements=module.statements,
            graph=block_graph,
            outputs=block_outputs,
            returns=block_returns,
            ctx=_new_lower_ctx(
                module=module,
                returns=block_returns,
                signatures=signatures,
                block_path_params=block_path_params,
                block_param_last_dims=block_param_last_dims,
                block_output_last_dims=block_output_last_dims,
                block_param_shapes=block_param_shapes,
                block_output_shapes=block_output_shapes,
                block_optional_inputs=block_optional_inputs,
                block_default_inputs=block_default_inputs,
                typed_signatures=typed_signatures,
                implicit_prelude_members=implicit_prelude_members,
                prelude_aliases=prelude_aliases,
                primitive_aliases=primitive_aliases,
                imported_symbol_values=_collect_imported_symbol_values(module, by_name),
                module_dim_symbol_names=module_dim_symbol_names.get(module.name, set()),
            ),
        )
        _ensure_outputs_from_returns(block_outputs, block_returns)
        blocks[module.name] = {
            "inputs": block_inputs,
            "graph": block_graph,
            "outputs": block_outputs,
        }
    if blocks:
        spec["model"]["blocks"] = blocks

    inferred_block_io_types = infer_block_io_types_from_modules(
        spec=spec,
        modules=modules,
        selected_main=main_name,
    )
    final_block_io_types = _finalize_block_io_types_for_model(
        model=spec["model"],
        inferred=inferred_block_io_types,
    )
    annotate_spec_with_block_io_types(spec, block_io_types=final_block_io_types)

    return spec


__all__ = [
    "lower_axon_module_to_synapse_block",
    "lower_axon_module_to_synapse_spec",
    "lower_axon_program_to_synapse_spec",
]
