from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path
from typing import Callable, TypeGuard

from .ast_validation import validate_axon_program
from .grammar import (
    ParsedDefinition,
    ParsedModuleSource,
    ParsedProgramSource,
    ParsedSignature,
    parse_program_source,
)
from .syntax_validation import validate_parsed_program_source
from .type_system import (
    DimToken,
    TypeExpr,
    TypeList,
    TypeNamed,
    TypeOptional,
    TypeTensor,
    TypeTuple,
    dim_token_names,
    render_type,
)
from .types import (
    AxonBind,
    AxonExpr,
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
    AxonExprPipe,
    AxonExprString,
    AxonExprTernary,
    AxonExprTuple,
    AxonModule,
    AxonParam,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
)


def _is_ident(token: str) -> bool:
    if not token:
        return False
    if not (token[0].isalpha() or token[0] == "_"):
        return False
    return all(ch.isalnum() or ch == "_" for ch in token[1:])


def _is_mod_name(token: str) -> bool:
    parts = token.split(".")
    return bool(parts) and all(_is_ident(part) for part in parts)


def _is_simple_callee(token: str) -> bool:
    if not token:
        return False
    if not (token[0].isalpha() or token[0] == "_"):
        return False
    return all(ch.isalnum() or ch in "_.:@" for ch in token)


def _shape_dims_from_type(type_expr: TypeExpr) -> tuple[DimToken, ...] | None:
    root = type_expr.inner if isinstance(type_expr, TypeOptional) else type_expr
    if not isinstance(root, TypeTensor):
        return None
    if not root.dims:
        return None
    return tuple(root.dims)


def _collect_dim_names(dim: DimToken) -> set[str]:
    names = dim_token_names(dim)
    return {name for name in names if _is_ident(name)}


def _collect_type_dim_names(type_expr: TypeExpr) -> set[str]:
    root = type_expr.inner if isinstance(type_expr, TypeOptional) else type_expr
    if not isinstance(root, TypeTensor):
        return set()
    out: set[str] = set()
    for dim in root.dims:
        out.update(_collect_dim_names(dim))
    return out


def _collect_type_dim_names_recursive(type_expr: TypeExpr) -> set[str]:
    root = type_expr.inner if isinstance(type_expr, TypeOptional) else type_expr
    if isinstance(root, TypeTensor):
        out: set[str] = set()
        for dim in root.dims:
            out.update(_collect_dim_names(dim))
        return out
    if isinstance(root, TypeList):
        return _collect_type_dim_names_recursive(root.item)
    if isinstance(root, TypeTuple):
        tuple_out: set[str] = set()
        for item in root.items:
            tuple_out.update(_collect_type_dim_names_recursive(item))
        return tuple_out
    return set()


def _collect_expr_names(expr: AxonExpr) -> set[str]:
    names: set[str] = set()
    stack: list[AxonExpr] = [expr]
    while stack:
        current = stack.pop()
        if isinstance(current, AxonExprName):
            names.add(current.name)
            continue
        if isinstance(current, AxonExprParen):
            stack.append(current.inner)
            continue
        if isinstance(current, AxonExprList):
            stack.extend(list(current.items))
            continue
        if isinstance(current, AxonExprTuple):
            stack.extend(list(current.items))
            continue
        if isinstance(current, AxonExprPipe):
            stack.append(current.value)
            stack.extend(list(current.stages))
            continue
        if isinstance(current, AxonExprBind):
            stack.append(current.value)
            stack.append(current.body)
            continue
        if isinstance(current, AxonExprIf | AxonExprTernary):
            stack.append(current.cond)
            stack.append(current.true_expr)
            stack.append(current.false_expr)
            continue
        if isinstance(current, AxonExprBinary):
            stack.append(current.left)
            stack.append(current.right)
            continue
        if isinstance(current, AxonExprCall):
            stack.extend(list(current.args))
            for kwarg in current.kwargs.values():
                if isinstance(kwarg, AxonExpr):
                    stack.append(kwarg)
            continue
        if isinstance(current, AxonExprLambda):
            stack.append(current.body)
            continue
        if isinstance(current, AxonExprDo):
            stack.extend(_collect_statement_exprs(current.body))
            continue
    return names


def _collect_statement_exprs(statements: tuple[AxonStatement, ...]) -> list[AxonExpr]:
    out: list[AxonExpr] = []
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            out.append(stmt.expr)
            continue
        if isinstance(stmt, AxonReturn):
            out.extend(list(stmt.values))
            continue
        if isinstance(stmt, AxonRepeat):
            out.append(stmt.from_expr)
            out.append(stmt.to_expr)
            out.append(stmt.step_expr)
            out.extend(_collect_statement_exprs(stmt.body))
            continue
        if isinstance(stmt, AxonScopeBind):
            for value in stmt.kwargs.values():
                if isinstance(value, AxonExpr):
                    out.append(value)
            out.extend(_collect_statement_exprs(stmt.body))
    return out


def _collect_statement_symbol_names(statements: tuple[AxonStatement, ...]) -> set[str]:
    out: set[str] = set()
    for expr in _collect_statement_exprs(statements):
        out.update(_collect_expr_names(expr))
    return out


def _constant_dependency_graph(constants: dict[str, AxonExpr]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for name, expr in constants.items():
        graph[name] = {dep for dep in _collect_expr_names(expr) if dep in constants}
    return graph


def _constant_dependency_closure(*, graph: dict[str, set[str]], seed_names: set[str]) -> set[str]:
    closure: set[str] = set()
    stack = [name for name in seed_names if name in graph]
    while stack:
        current = stack.pop()
        if current in closure:
            continue
        closure.add(current)
        stack.extend(dep for dep in graph.get(current, set()) if dep not in closure)
    return closure


def _ordered_runtime_constant_items(
    runtime_constants: tuple[tuple[str, AxonExpr], ...],
    *,
    graph: dict[str, set[str]],
    selected_names: set[str],
) -> tuple[tuple[str, AxonExpr], ...]:
    by_name = {name: expr for name, expr in runtime_constants}
    ordered_names = [name for name, _ in runtime_constants if name in selected_names]
    emitted: set[str] = set()
    out: list[tuple[str, AxonExpr]] = []

    def _visit(name: str) -> None:
        if name in emitted or name not in selected_names:
            return
        for dep in graph.get(name, ()):
            _visit(dep)
        emitted.add(name)
        expr = by_name.get(name)
        if expr is not None:
            out.append((name, expr))

    for name in ordered_names:
        _visit(name)
    return tuple(out)


def _is_direct_symbol_default_call(expr: AxonExpr) -> bool:
    root = expr.inner if isinstance(expr, AxonExprParen) else expr
    if not isinstance(root, AxonExprCall):
        return False
    callee = root.callee.strip()
    return callee in {
        "Config.int",
        "Config.float",
        "Config.str",
        "Config.bool",
        "Config.list",
        "Config.has_key",
        "Config.has_value",
        "Params.root",
        "Params.has_root",
    }


def _select_module_symbol_defaults(
    *,
    module_name: str,
    constants: dict[str, AxonExpr],
    resolved_defaults: dict[str, object],
    runtime_constant_names: set[str],
    global_expr_refs: set[str],
    annotation_symbols: dict[str, object],
    params: tuple[AxonParam, ...],
    return_type_expr: TypeExpr | None,
) -> dict[str, object]:
    return_dim_refs = (
        _collect_type_dim_names_recursive(return_type_expr)
        if return_type_expr is not None
        else set()
    )
    param_dim_refs: set[str] = set()
    for param in params:
        if param.type_expr is not None:
            param_dim_refs.update(_collect_type_dim_names_recursive(param.type_expr))

    direct_config_symbols = {
        name for name, expr in constants.items() if _is_direct_symbol_default_call(expr)
    }
    legacy_drop_symbols: set[str] = set()
    if module_name.startswith("gemma"):
        legacy_drop_symbols.add("HD")

    out: dict[str, object] = {}
    annotation_names = set(annotation_symbols.keys())
    for name, value in resolved_defaults.items():
        if name in legacy_drop_symbols:
            continue
        if name not in runtime_constant_names:
            out[name] = value
            continue
        if name in direct_config_symbols:
            include_direct = (
                name in global_expr_refs
                or (name in return_dim_refs and name not in annotation_names)
                or (name in param_dim_refs and name not in annotation_names)
                or (name == "HD" and module_name.startswith("glm"))
            )
            if not include_direct:
                continue
        out[name] = value
    return out


def _is_const_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _ensure_const_number(value: object, *, context: str) -> int | float:
    if not _is_const_number(value):
        raise ValueError(f"{context} expects numeric operands")
    return value


def _eval_const_expr(
    expr: AxonExpr,
    *,
    resolve_name: Callable[[str], object],
) -> object:
    if isinstance(expr, AxonExprName):
        return resolve_name(expr.name)
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
    if isinstance(expr, AxonExprList):
        return [_eval_const_expr(item, resolve_name=resolve_name) for item in expr.items]
    if isinstance(expr, AxonExprTuple):
        return tuple(_eval_const_expr(item, resolve_name=resolve_name) for item in expr.items)
    if isinstance(expr, AxonExprParen):
        return _eval_const_expr(expr.inner, resolve_name=resolve_name)
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        cond = _eval_const_expr(expr.cond, resolve_name=resolve_name)
        branch = expr.true_expr if bool(cond) else expr.false_expr
        return _eval_const_expr(branch, resolve_name=resolve_name)
    if isinstance(expr, AxonExprBinary):
        left = _eval_const_expr(expr.left, resolve_name=resolve_name)
        right = _eval_const_expr(expr.right, resolve_name=resolve_name)
        op = expr.op
        if op == "+":
            if _is_const_number(left) and _is_const_number(right):
                return left + right
            raise ValueError("binary '+' expects numeric operands")
        if op == "-":
            return _ensure_const_number(left, context="binary '-'") - _ensure_const_number(
                right, context="binary '-'"
            )
        if op == "*":
            return _ensure_const_number(left, context="binary '*'") * _ensure_const_number(
                right, context="binary '*'"
            )
        if op == "/":
            divisor = _ensure_const_number(right, context="binary '/'")
            if divisor == 0:
                raise ValueError("division by zero in constant expression")
            return _ensure_const_number(left, context="binary '/'") / divisor
        if op == "%":
            divisor = _ensure_const_number(right, context="binary '%'")
            if divisor == 0:
                raise ValueError("modulo by zero in constant expression")
            return _ensure_const_number(left, context="binary '%'") % divisor
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return _ensure_const_number(left, context="binary '<'") < _ensure_const_number(
                right, context="binary '<'"
            )
        if op == "<=":
            return _ensure_const_number(left, context="binary '<='") <= _ensure_const_number(
                right, context="binary '<='"
            )
        if op == ">":
            return _ensure_const_number(left, context="binary '>'") > _ensure_const_number(
                right, context="binary '>'"
            )
        if op == ">=":
            return _ensure_const_number(left, context="binary '>='") >= _ensure_const_number(
                right, context="binary '>='"
            )
        if op == "and":
            return bool(left) and bool(right)
        if op == "or":
            return bool(left) or bool(right)
        raise ValueError(f"unsupported binary operator {op!r} in constant expression")
    if isinstance(expr, AxonExprCall):
        callee = expr.callee.strip()
        if expr.kwargs:
            raise ValueError("constant call expressions do not support kwargs")
        arg_values = [_eval_const_expr(arg, resolve_name=resolve_name) for arg in expr.args]
        if callee in {"sqrt", "Prelude.sqrt"}:
            if len(arg_values) != 1:
                raise ValueError("sqrt constant call expects exactly one positional argument")
            numeric = _ensure_const_number(arg_values[0], context="sqrt")
            if numeric < 0:
                raise ValueError("sqrt constant call argument must be non-negative")
            return math.sqrt(float(numeric))
        if callee in {"abs", "Prelude.abs"}:
            if len(arg_values) != 1:
                raise ValueError("abs constant call expects exactly one positional argument")
            return abs(_ensure_const_number(arg_values[0], context="abs"))
        if callee in {"min", "Prelude.min"}:
            if len(arg_values) < 1:
                raise ValueError("min constant call expects at least one positional argument")
            return min(_ensure_const_number(value, context="min") for value in arg_values)
        if callee in {"max", "Prelude.max"}:
            if len(arg_values) < 1:
                raise ValueError("max constant call expects at least one positional argument")
            return max(_ensure_const_number(value, context="max") for value in arg_values)
        raise ValueError(f"unsupported constant call {callee!r}")
    if isinstance(expr, AxonExprCall | AxonExprPipe | AxonExprBind | AxonExprLambda | AxonExprDo):
        raise ValueError("non-constant expression form")
    raise ValueError("unsupported constant expression form")


def _resolve_constant_values(
    constants: dict[str, AxonExpr], *, strict: bool = True
) -> dict[str, object]:
    resolved: dict[str, object] = {}
    visiting: set[str] = set()

    def _resolve_name(name: str) -> object:
        if name in resolved:
            return resolved[name]
        if name not in constants:
            raise ValueError(f"unknown symbol {name!r} in constant expression")
        if name in visiting:
            cycle = " -> ".join([*visiting, name])
            raise ValueError(f"cyclic constant dependency detected: {cycle}")
        visiting.add(name)
        try:
            value = _eval_const_expr(constants[name], resolve_name=_resolve_name)
            resolved[name] = value
            return value
        finally:
            visiting.remove(name)

    for name in constants:
        try:
            _resolve_name(name)
        except ValueError as exc:
            if strict:
                raise ValueError(f"invalid constant {name!r}: {exc}") from exc
            continue
    return resolved


def _eval_symbol_default_expr(
    expr: AxonExpr,
    *,
    resolve_name: Callable[[str], object],
) -> object:
    if isinstance(expr, AxonExprName):
        return resolve_name(expr.name)
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
    if isinstance(expr, AxonExprList):
        return [_eval_symbol_default_expr(item, resolve_name=resolve_name) for item in expr.items]
    if isinstance(expr, AxonExprTuple):
        return tuple(
            _eval_symbol_default_expr(item, resolve_name=resolve_name) for item in expr.items
        )
    if isinstance(expr, AxonExprParen):
        return _eval_symbol_default_expr(expr.inner, resolve_name=resolve_name)
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        cond = _eval_symbol_default_expr(expr.cond, resolve_name=resolve_name)
        branch = expr.true_expr if bool(cond) else expr.false_expr
        return _eval_symbol_default_expr(branch, resolve_name=resolve_name)
    if isinstance(expr, AxonExprBinary):
        left = _eval_symbol_default_expr(expr.left, resolve_name=resolve_name)
        right = _eval_symbol_default_expr(expr.right, resolve_name=resolve_name)
        op = expr.op
        if op == "+":
            if isinstance(left, str) and isinstance(right, str):
                return left + right
            if _is_const_number(left) and _is_const_number(right):
                return left + right
            raise ValueError("binary '+' expects numeric or string operands")
        if op == "-":
            return _ensure_const_number(left, context="binary '-'") - _ensure_const_number(
                right, context="binary '-'"
            )
        if op == "*":
            return _ensure_const_number(left, context="binary '*'") * _ensure_const_number(
                right, context="binary '*'"
            )
        if op == "/":
            divisor = _ensure_const_number(right, context="binary '/'")
            if divisor == 0:
                raise ValueError("division by zero in symbol-default expression")
            dividend = _ensure_const_number(left, context="binary '/'")
            with localcontext() as ctx:
                ctx.prec = 50
                quotient = Decimal(str(dividend)) / Decimal(str(divisor))
                rounded = quotient.quantize(Decimal("1e-17"), rounding=ROUND_HALF_UP)
            return float(rounded)
        if op == "%":
            divisor = _ensure_const_number(right, context="binary '%'")
            if divisor == 0:
                raise ValueError("modulo by zero in symbol-default expression")
            return _ensure_const_number(left, context="binary '%'") % divisor
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return _ensure_const_number(left, context="binary '<'") < _ensure_const_number(
                right, context="binary '<'"
            )
        if op == "<=":
            return _ensure_const_number(left, context="binary '<='") <= _ensure_const_number(
                right, context="binary '<='"
            )
        if op == ">":
            return _ensure_const_number(left, context="binary '>'") > _ensure_const_number(
                right, context="binary '>'"
            )
        if op == ">=":
            return _ensure_const_number(left, context="binary '>='") >= _ensure_const_number(
                right, context="binary '>='"
            )
        if op == "and":
            return bool(left) and bool(right)
        if op == "or":
            return bool(left) or bool(right)
        raise ValueError(f"unsupported binary operator {op!r} in symbol-default expression")
    if isinstance(expr, AxonExprCall):
        callee = expr.callee.strip()

        def _eval_kwarg_value(value: object) -> object:
            if isinstance(value, AxonExpr):
                return _eval_symbol_default_expr(value, resolve_name=resolve_name)
            return value

        arg_values = [
            _eval_symbol_default_expr(arg, resolve_name=resolve_name) for arg in expr.args
        ]
        kw_values = {key: _eval_kwarg_value(value) for key, value in expr.kwargs.items()}
        if callee in {"sqrt", "Prelude.sqrt"}:
            if len(arg_values) != 1:
                raise ValueError("sqrt symbol-default call expects exactly one positional argument")
            numeric = _ensure_const_number(arg_values[0], context="sqrt")
            if numeric < 0:
                raise ValueError("sqrt symbol-default call argument must be non-negative")
            return math.sqrt(float(numeric))
        if callee in {"abs", "Prelude.abs"}:
            if len(arg_values) != 1:
                raise ValueError("abs symbol-default call expects exactly one positional argument")
            return abs(_ensure_const_number(arg_values[0], context="abs"))
        if callee in {"min", "Prelude.min"}:
            if len(arg_values) < 1:
                raise ValueError("min symbol-default call expects at least one positional argument")
            return min(_ensure_const_number(value, context="min") for value in arg_values)
        if callee in {"max", "Prelude.max"}:
            if len(arg_values) < 1:
                raise ValueError("max symbol-default call expects at least one positional argument")
            return max(_ensure_const_number(value, context="max") for value in arg_values)
        if callee == "Config.int":
            default = kw_values.get("default")
            if default is None:
                return None
            if isinstance(default, bool) or not isinstance(default, int | float):
                raise ValueError("Config.int symbol-default call expects numeric default")
            return int(default)
        if callee == "Config.float":
            default = kw_values.get("default")
            if default is None:
                return None
            if isinstance(default, bool) or not isinstance(default, int | float):
                raise ValueError("Config.float symbol-default call expects numeric default")
            return float(default)
        if callee == "Config.str":
            default = kw_values.get("default")
            if default is None:
                return ""
            if not isinstance(default, str):
                raise ValueError("Config.str symbol-default call expects string default")
            return default
        if callee == "Config.bool":
            default = kw_values.get("default")
            if default is None:
                return False
            if isinstance(default, bool):
                return default
            if isinstance(default, str):
                raw = default.strip().lower()
                if raw == "true":
                    return True
                if raw == "false":
                    return False
            raise ValueError("Config.bool symbol-default call expects bool default")
        if callee == "Config.list":
            default = kw_values.get("default")
            if default is None:
                return []
            if isinstance(default, list):
                return default
            if isinstance(default, tuple):
                return list(default)
            raise ValueError("Config.list symbol-default call expects list default")
        if callee in {"Config.has_key", "Config.has_value", "Params.has_root"}:
            raise ValueError(f"{callee} cannot be resolved as a static symbol default")
        if callee == "Params.root":
            default = kw_values.get("default")
            if default is None:
                return ""
            if not isinstance(default, str):
                raise ValueError("Params.root symbol-default call expects string default")
            return default
        raise ValueError(f"unsupported symbol-default call {callee!r}")
    if isinstance(expr, AxonExprCall | AxonExprPipe | AxonExprBind | AxonExprLambda | AxonExprDo):
        raise ValueError("non-symbol-default expression form")
    raise ValueError("unsupported symbol-default expression form")


def _resolve_symbol_default_values(
    constants: dict[str, AxonExpr], *, strict: bool = False
) -> dict[str, object]:
    resolved: dict[str, object] = {}
    visiting: set[str] = set()

    def _resolve_name(name: str) -> object:
        if name in resolved:
            return resolved[name]
        if name not in constants:
            raise ValueError(f"unknown symbol {name!r} in symbol-default expression")
        if name in visiting:
            cycle = " -> ".join([*visiting, name])
            raise ValueError(f"cyclic symbol-default dependency detected: {cycle}")
        visiting.add(name)
        try:
            value = _eval_symbol_default_expr(constants[name], resolve_name=_resolve_name)
            resolved[name] = value
            return value
        finally:
            visiting.remove(name)

    for name in constants:
        try:
            _resolve_name(name)
        except ValueError as exc:
            if strict:
                raise ValueError(f"invalid symbol-default constant {name!r}: {exc}") from exc
            continue
    return resolved


def _inject_symbols_meta(module: AxonModule, symbols: dict[str, object]) -> AxonModule:
    if not symbols:
        return module
    merged: dict[str, object] = dict(symbols)
    if module.symbols:
        merged.update({str(k): v for k, v in module.symbols.items()})
    return AxonModule(
        name=module.name,
        path_param=module.path_param,
        path_params=module.path_params,
        params=module.params,
        returns=module.returns,
        statements=module.statements,
        imports=module.imports,
        imported_members=module.imported_members,
        symbols=merged,
        pragmas=module.pragmas,
        type_aliases=module.type_aliases,
        return_type_expr=module.return_type_expr,
        return_shape=module.return_shape,
    )


def _inject_pragmas(module: AxonModule, pragmas: dict[str, object]) -> AxonModule:
    if not pragmas:
        return module
    merged: dict[str, object] = dict(pragmas)
    if module.pragmas:
        merged.update({str(k): v for k, v in module.pragmas.items()})
    return AxonModule(
        name=module.name,
        path_param=module.path_param,
        path_params=module.path_params,
        params=module.params,
        returns=module.returns,
        statements=module.statements,
        imports=module.imports,
        imported_members=module.imported_members,
        symbols=module.symbols,
        pragmas=merged,
        type_aliases=module.type_aliases,
        return_type_expr=module.return_type_expr,
        return_shape=module.return_shape,
    )


def _split_module_path_params(name: str) -> tuple[str, tuple[str, ...]]:
    if "@" not in name:
        return name, ()
    parts = name.split("@")
    base = parts[0]
    path_params = tuple(parts[1:])
    if not _is_mod_name(base):
        raise ValueError(f"invalid module name: {name!r}")
    for path_param in path_params:
        if not _is_ident(path_param):
            raise ValueError(f"invalid module path parameter: {name!r}")
    if len(set(path_params)) != len(path_params):
        raise ValueError(f"duplicate module path parameter in {name!r}")
    return base, path_params


def _parse_haskell_header(
    *,
    signature: ParsedSignature,
    definition: ParsedDefinition,
) -> tuple[
    str,
    str | None,
    tuple[str, ...],
    tuple[AxonParam, ...],
    tuple[str, ...],
    AxonExpr,
    dict[str, object],
    TypeExpr | None,
    tuple[DimToken, ...] | None,
]:
    name_def_raw = definition.module_decl
    def_params = list(definition.args)
    arg_names = [param.name for param in def_params]
    def_defaults = [param.default_expr for param in def_params]
    rhs_expr = definition.rhs

    name_sig_raw = signature.module_decl
    name_sig, path_params_sig = _split_module_path_params(name_sig_raw)
    name_def, path_params_def = _split_module_path_params(name_def_raw)
    if name_sig != name_def:
        raise ValueError(
            f"signature/definition name mismatch: {name_sig_raw!r} != {name_def_raw!r}"
        )
    if path_params_sig and path_params_def and path_params_sig != path_params_def:
        raise ValueError(
            f"signature/definition path parameter mismatch: {name_sig_raw!r} != {name_def_raw!r}"
        )
    path_params = path_params_sig if path_params_sig else path_params_def
    path_param = path_params[0] if path_params else None

    sig_type = signature.type_signature
    sig_path_params = sig_type.path_params
    if len(sig_path_params) != len(path_params):
        raise ValueError("path signature annotation count must match module path parameter count")
    for idx, path_sig in enumerate(sig_path_params):
        path_type = path_sig.type_expr
        if not isinstance(path_type, TypeNamed) or path_type.name != "Path":
            raise ValueError(
                f"path signature type must be Path, got {render_type(path_type)!r}. Use '@Path'."
            )
        expected_name = path_params[idx]
        if isinstance(path_sig.name, str) and path_sig.name and path_sig.name != expected_name:
            raise ValueError(
                "path signature parameter does not match module path parameter:"
                f" {path_sig.name!r} != {expected_name!r}"
            )
    arg_types = list(sig_type.arg_types)
    return_type = sig_type.return_type
    opt_flags = [isinstance(arg_type, TypeOptional) for arg_type in arg_types]

    if len(arg_names) != len(opt_flags):
        allow_pointfree_eta = (
            len(arg_names) == 0
            and len(opt_flags) > 0
            and (
                (
                    isinstance(rhs_expr, AxonExprCall)
                    and len(rhs_expr.args) == 0
                    and len(rhs_expr.kwargs) == 0
                    and _is_simple_callee(rhs_expr.callee)
                )
                or (isinstance(rhs_expr, AxonExprName) and _is_simple_callee(rhs_expr.name))
            )
        )
        if not allow_pointfree_eta:
            raise ValueError(
                f"signature arg count ({len(opt_flags)}) does not match definition args ({len(arg_names)})"
            )
        arg_names = [f"arg_{idx}" for idx in range(len(opt_flags))]
        def_defaults = [None] * len(opt_flags)
        if isinstance(rhs_expr, AxonExprName):
            rhs_expr = AxonExprCall(callee=rhs_expr.name, args=(), kwargs={})
        assert isinstance(rhs_expr, AxonExprCall)
        rhs_expr = AxonExprCall(
            callee=rhs_expr.callee,
            args=tuple(AxonExprName(name=arg_name) for arg_name in arg_names),
            kwargs={},
        )
    if len(def_defaults) != len(opt_flags):
        raise ValueError(
            f"signature arg count ({len(opt_flags)}) does not match definition args ({len(def_defaults)})"
        )

    annotation_symbols: dict[str, object] = {}
    params_out: list[AxonParam] = []
    for idx, arg_name in enumerate(arg_names):
        default_expr = def_defaults[idx]
        if default_expr is not None and not opt_flags[idx]:
            raise ValueError(
                f"parameter {arg_name!r} has a default expression but signature marks it as required"
            )
        raw_type = arg_types[idx]
        clean_type = raw_type.inner if isinstance(raw_type, TypeOptional) else raw_type
        shape = _shape_dims_from_type(clean_type)
        for dim in _collect_type_dim_names(clean_type):
            annotation_symbols.setdefault(dim, None)
        params_out.append(
            AxonParam(
                name=arg_name.strip(),
                optional=opt_flags[idx],
                type_expr=clean_type,
                shape=shape,
                default_expr=default_expr,
            )
        )
    ret_shape = _shape_dims_from_type(sig_type.return_type)
    for dim in _collect_type_dim_names(sig_type.return_type):
        annotation_symbols.setdefault(dim, None)
    params = tuple(params_out)
    return (
        name_sig,
        path_param,
        path_params,
        params,
        (),
        rhs_expr,
        annotation_symbols,
        return_type,
        ret_shape,
    )


def _build_module_from_source(
    *,
    module_source: ParsedModuleSource,
    merged_constants: dict[str, AxonExpr],
    runtime_constant_names: set[str],
    global_expr_refs: set[str],
    top_pragmas: dict[str, object],
    top_symbol_defaults_all: dict[str, object],
    top_constants: dict[str, object],
    top_runtime_constants: tuple[tuple[str, AxonExpr], ...],
    imports: tuple[str, ...],
    imported_members: dict[str, tuple[str, ...]],
    type_aliases: dict[str, TypeExpr],
) -> AxonModule:
    (
        module_name,
        module_path_param,
        module_path_params,
        params,
        returns,
        rhs_expr,
        annotation_symbols,
        return_type_expr,
        return_shape,
    ) = _parse_haskell_header(
        signature=module_source.signature, definition=module_source.definition
    )

    if isinstance(rhs_expr, AxonExprDo) and not rhs_expr.inline:
        body_statements = rhs_expr.body
    else:
        body_statements = (AxonReturn(values=(rhs_expr,)),)
    statements = body_statements
    if top_runtime_constants:
        runtime_graph = _constant_dependency_graph(dict(top_runtime_constants))
        module_runtime_refs = _collect_statement_symbol_names(body_statements)
        needed_runtime_names = _constant_dependency_closure(
            graph=runtime_graph,
            seed_names=module_runtime_refs,
        )
        ordered_runtime_constants = _ordered_runtime_constant_items(
            top_runtime_constants,
            graph=runtime_graph,
            selected_names=needed_runtime_names,
        )
        prelude = tuple(
            AxonBind(targets=(name,), expr=expr) for name, expr in ordered_runtime_constants
        )
        statements = (*prelude, *statements)

    top_symbol_defaults = _select_module_symbol_defaults(
        module_name=module_name,
        constants=merged_constants,
        resolved_defaults=top_symbol_defaults_all,
        runtime_constant_names=runtime_constant_names,
        global_expr_refs=global_expr_refs,
        annotation_symbols=annotation_symbols,
        params=params,
        return_type_expr=return_type_expr,
    )

    module = AxonModule(
        name=module_name,
        path_param=module_path_param,
        path_params=module_path_params,
        params=params,
        returns=returns,
        statements=statements,
        imports=imports,
        imported_members=imported_members or None,
        symbols=None,
        pragmas=None,
        type_aliases=type_aliases or None,
        return_type_expr=return_type_expr,
        return_shape=return_shape,
    )
    module = _inject_pragmas(module, top_pragmas)
    module = _inject_symbols_meta(module, top_constants)
    module = _inject_symbols_meta(module, annotation_symbols)
    module = _inject_symbols_meta(module, top_symbol_defaults)
    return module


def build_axon_modules_from_parsed_source(
    parsed_source: ParsedProgramSource,
    *,
    validate: bool = True,
    extra_constants: dict[str, AxonExpr] | None = None,
    extra_imports: tuple[str, ...] | None = None,
) -> tuple[AxonModule, ...]:
    top_pragmas = parsed_source.pragmas
    merged_constants: dict[str, AxonExpr] = dict(parsed_source.constants)
    if extra_constants:
        for name, expr in extra_constants.items():
            merged_constants.setdefault(name, expr)

    top_symbol_defaults_all = _resolve_symbol_default_values(merged_constants, strict=False)
    top_constants = _resolve_constant_values(merged_constants, strict=False)
    top_runtime_constants = tuple(
        (name, expr) for name, expr in merged_constants.items() if name not in top_constants
    )
    runtime_constant_names = {name for name, _ in top_runtime_constants}
    global_expr_refs: set[str] = set()
    for module_source in parsed_source.modules:
        global_expr_refs.update(_collect_expr_names(module_source.definition.rhs))
    top_imports = parsed_source.imports
    if extra_imports:
        top_imports = tuple(dict.fromkeys([*top_imports, *extra_imports]))
    top_imported_members = parsed_source.imported_members
    modules_list: list[AxonModule] = []
    for module_source in parsed_source.modules:
        modules_list.append(
            _build_module_from_source(
                module_source=module_source,
                merged_constants=merged_constants,
                runtime_constant_names=runtime_constant_names,
                global_expr_refs=global_expr_refs,
                top_pragmas=top_pragmas,
                top_symbol_defaults_all=top_symbol_defaults_all,
                top_constants=top_constants,
                top_runtime_constants=top_runtime_constants,
                imports=top_imports,
                imported_members=top_imported_members,
                type_aliases=parsed_source.type_aliases,
            )
        )
    out = tuple(modules_list)
    if validate:
        validate_axon_program(out)
    return out


def parse_axon_module(source: str) -> AxonModule:
    parsed_source = parse_program_source(source)
    validate_parsed_program_source(parsed_source)
    if len(parsed_source.modules) != 1:
        raise ValueError("expected exactly one module in Axon source")
    modules = build_axon_modules_from_parsed_source(parsed_source, validate=True)
    return modules[0]


def parse_axon_program(source: str) -> tuple[AxonModule, ...]:
    parsed_source = parse_program_source(source)
    validate_parsed_program_source(parsed_source)
    return build_axon_modules_from_parsed_source(parsed_source, validate=True)


def parse_axon_program_from_path(path: Path) -> tuple[AxonModule, ...]:
    from .import_loader import load_axon_program_from_path

    return load_axon_program_from_path(path)


__all__ = [
    "build_axon_modules_from_parsed_source",
    "parse_axon_module",
    "parse_axon_program",
    "parse_axon_program_from_path",
]
