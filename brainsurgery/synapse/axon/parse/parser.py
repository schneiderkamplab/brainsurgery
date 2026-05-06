from __future__ import annotations

from pathlib import Path

from ..ast.nodes import (
    AxonExpr,
    AxonExprDo,
    AxonDefinition,
    AxonParam,
    AxonReturn,
)
from ..ast.source import AxonFile
from ..ast.types import (
    TypeExpr,
    TypeOptional,
    TypePath,
    render_type,
)
from ..validate.ast import validate_axon_program
from ..validate.surface import validate_parsed_program_source
from ._cst import (
    CstDefinition,
    CstDefinitionSource,
    CstPathTypeParam,
    CstSignature,
)
from .grammar import parse_surface_program_source


def _is_ident(token: str) -> bool:
    if not token:
        return False
    if not (token[0].isalpha() or token[0] == "_"):
        return False
    return all(ch.isalnum() or ch == "_" for ch in token[1:])


def _is_mod_name(token: str) -> bool:
    parts = token.split(".")
    return bool(parts) and all(_is_ident(part) for part in parts)


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
    signature: CstSignature | None,
    definition: CstDefinition,
) -> tuple[
    str,
    str | None,
    tuple[str, ...],
    tuple[AxonParam, ...],
    tuple[str, ...],
    AxonExpr,
    TypeExpr | None,
]:
    name_def_raw = definition.definition_decl
    def_params = list(definition.args)
    arg_names = [param.name for param in def_params]
    def_defaults = [param.default_expr for param in def_params]
    rhs_expr = definition.rhs

    name_def, path_params_def = _split_module_path_params(name_def_raw)
    if signature is None:
        params = tuple(
            AxonParam(
                name=param.name.strip(),
                optional=param.default_expr is not None,
                type_expr=None,
                default_expr=param.default_expr,
            )
            for param in def_params
        )
        return (
            name_def,
            path_params_def[0] if path_params_def else None,
            path_params_def,
            params,
            (),
            rhs_expr,
            None,
        )

    name_sig_raw = signature.definition_decl
    name_sig, path_params_sig = _split_module_path_params(name_sig_raw)
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
    if not sig_path_params and path_params:
        arg_types_raw = list(sig_type.arg_types)
        if len(arg_types_raw) < len(path_params):
            raise ValueError("signature must provide one Path argument per module path parameter")
        inferred_path_params: list[CstPathTypeParam] = []
        for idx, path_name in enumerate(path_params):
            path_type = arg_types_raw[idx]
            if not isinstance(path_type, TypePath):
                raise ValueError(
                    "path-bound module parameters require leading Path arguments in signature"
                )
            inferred_path_params.append(CstPathTypeParam(name=path_name, type_expr=path_type))
        sig_path_params = tuple(inferred_path_params)
        arg_types = arg_types_raw[len(path_params) :]
    else:
        arg_types = list(sig_type.arg_types)
    if len(sig_path_params) != len(path_params):
        raise ValueError("path signature annotation count must match module path parameter count")
    for idx, path_sig in enumerate(sig_path_params):
        path_type = path_sig.type_expr
        if not isinstance(path_type, TypePath):
            raise ValueError(f"path signature type must be Path, got {render_type(path_type)!r}.")
        expected_name = path_params[idx]
        if isinstance(path_sig.name, str) and path_sig.name and path_sig.name != expected_name:
            raise ValueError(
                "path signature parameter does not match module path parameter:"
                f" {path_sig.name!r} != {expected_name!r}"
            )
    return_type = sig_type.return_type
    opt_flags = [isinstance(arg_type, TypeOptional) for arg_type in arg_types]

    if len(arg_names) != len(opt_flags):
        raise ValueError(
            f"signature arg count ({len(opt_flags)}) does not match definition args ({len(arg_names)})"
        )
    if len(def_defaults) != len(opt_flags):
        raise ValueError(
            f"signature arg count ({len(opt_flags)}) does not match definition args ({len(def_defaults)})"
        )

    params_out: list[AxonParam] = []
    for idx, arg_name in enumerate(arg_names):
        default_expr = def_defaults[idx]
        if default_expr is not None and not opt_flags[idx]:
            raise ValueError(
                f"parameter {arg_name!r} has a default expression but signature marks it as required"
            )
        raw_type = arg_types[idx]
        clean_type = raw_type.inner if isinstance(raw_type, TypeOptional) else raw_type
        params_out.append(
            AxonParam(
                name=arg_name.strip(),
                optional=opt_flags[idx],
                type_expr=clean_type,
                default_expr=default_expr,
            )
        )
    params = tuple(params_out)
    return (
        name_sig,
        path_param,
        path_params,
        params,
        (),
        rhs_expr,
        return_type,
    )


def _build_file_module_from_surface_source(
    *,
    definition_source: CstDefinitionSource,
) -> AxonDefinition:
    (
        module_name,
        module_path_param,
        module_path_params,
        params,
        returns,
        rhs_expr,
        return_type_expr,
    ) = _parse_haskell_header(
        signature=definition_source.signature, definition=definition_source.definition
    )
    return AxonDefinition(
        name=module_name,
        path_param=module_path_param,
        path_params=module_path_params,
        params=params,
        returns=returns,
        statements=rhs_expr.body if isinstance(rhs_expr, AxonExprDo) and not rhs_expr.inline else (),
        body_expr=None if isinstance(rhs_expr, AxonExprDo) and not rhs_expr.inline else rhs_expr,
        imports=(),
        imported_members=None,
        exports=(),
        symbols=None,
        pragmas=None,
        type_aliases=None,
        return_type_expr=return_type_expr,
    )


def _ast_pragmas_with_explicit_main(
    pragmas: dict[str, object], modules: tuple[AxonDefinition, ...]
) -> dict[str, object]:
    ast_pragmas = dict(pragmas)
    if "main" not in ast_pragmas and modules:
        ast_pragmas["main"] = modules[-1].name
    return ast_pragmas


def parse_axon_module(source: str) -> AxonDefinition:
    ast = parse_axon_program(source)
    if len(ast.modules) != 1:
        raise ValueError("expected exactly one module in Axon source")
    return ast.modules[0]


def parse_axon_program(source: str) -> AxonFile:
    parsed_source = parse_surface_program_source(source)
    validate_parsed_program_source(parsed_source)
    modules = tuple(
        _build_file_module_from_surface_source(definition_source=definition_source)
        for definition_source in parsed_source.modules
    )
    validate_axon_program(modules)
    return AxonFile(
        modules=modules,
        imports=parsed_source.imports,
        imported_members=dict(parsed_source.imported_members),
        exports=parsed_source.exports,
        pragmas=_ast_pragmas_with_explicit_main(dict(parsed_source.pragmas), modules),
        type_aliases=dict(parsed_source.type_aliases),
    )


def parse_axon_program_from_path(path: Path) -> AxonFile:
    ast = parse_axon_program(path.read_text(encoding="utf-8"))
    return AxonFile(
        modules=ast.modules,
        imports=ast.imports,
        imported_members=dict(ast.imported_members),
        exports=ast.exports,
        pragmas=dict(ast.pragmas),
        type_aliases=dict(ast.type_aliases),
        origin_path=path.resolve(),
    )


__all__ = [
    "parse_axon_module",
    "parse_axon_program",
    "parse_axon_program_from_path",
]
