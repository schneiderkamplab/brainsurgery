from __future__ import annotations

import math
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
    TypeNamed,
    TypeOptional,
    TypeTensor,
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
    AxonReturn,
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
    arg_names = list(definition.args)
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
        if isinstance(rhs_expr, AxonExprName):
            rhs_expr = AxonExprCall(callee=rhs_expr.name, args=(), kwargs={})
        assert isinstance(rhs_expr, AxonExprCall)
        rhs_expr = AxonExprCall(
            callee=rhs_expr.callee,
            args=tuple(AxonExprName(name=arg_name) for arg_name in arg_names),
            kwargs={},
        )

    annotation_symbols: dict[str, object] = {}
    params_out: list[AxonParam] = []
    for idx, arg_name in enumerate(arg_names):
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
    top_pragmas: dict[str, object],
    top_constants: dict[str, object],
    top_runtime_constants: tuple[tuple[str, AxonExpr], ...],
    imports: tuple[str, ...],
    imported_members: dict[str, tuple[str, ...]],
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
        statements = rhs_expr.body
    else:
        statements = (AxonReturn(values=(rhs_expr,)),)
    if top_runtime_constants:
        prelude = tuple(
            AxonBind(targets=(name,), expr=expr) for name, expr in top_runtime_constants
        )
        statements = (*prelude, *statements)

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
        return_type_expr=return_type_expr,
        return_shape=return_shape,
    )
    module = _inject_pragmas(module, top_pragmas)
    module = _inject_symbols_meta(module, annotation_symbols)
    module = _inject_symbols_meta(module, top_constants)
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

    top_constants = _resolve_constant_values(merged_constants, strict=False)
    top_runtime_constants = tuple(
        (name, expr) for name, expr in merged_constants.items() if name not in top_constants
    )
    top_imports = parsed_source.imports
    if extra_imports:
        top_imports = tuple(dict.fromkeys([*top_imports, *extra_imports]))
    top_imported_members = parsed_source.imported_members
    modules_list: list[AxonModule] = []
    for module_source in parsed_source.modules:
        modules_list.append(
            _build_module_from_source(
                module_source=module_source,
                top_pragmas=top_pragmas,
                top_constants=top_constants,
                top_runtime_constants=top_runtime_constants,
                imports=top_imports,
                imported_members=top_imported_members,
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
