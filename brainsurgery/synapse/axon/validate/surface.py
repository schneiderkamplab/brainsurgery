from __future__ import annotations

from typing import Any

from ..ast.nodes import (
    AxonBind,
    AxonCond,
    AxonExpr,
    AxonExprDo,
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


def _is_mod_decl(token: str) -> bool:
    parts = token.split("@")
    if not parts:
        return False
    if not _is_mod_name(parts[0]):
        return False
    return all(_is_ident(part) for part in parts[1:])


def _split_module_path_params(decl: str) -> tuple[str, tuple[str, ...]]:
    parts = decl.split("@")
    base = parts[0].strip()
    path_params = tuple(part.strip() for part in parts[1:])
    if not _is_mod_name(base):
        raise ValueError(f"invalid module name {decl!r}")
    for path_param in path_params:
        if not _is_ident(path_param):
            raise ValueError(f"invalid module path parameter in declaration {decl!r}")
    if len(set(path_params)) != len(path_params):
        raise ValueError(f"duplicate module path parameter in declaration {decl!r}")
    return base, path_params


def _has_return_statement(stmts: tuple[AxonStatement, ...]) -> bool:
    for stmt in stmts:
        if isinstance(stmt, AxonReturn):
            return True
        nested: tuple[AxonStatement, ...] = ()
        if isinstance(stmt, AxonRepeat | AxonScopeBind):
            nested = stmt.body
        elif isinstance(stmt, AxonCond):
            nested = (*stmt.true_body, *stmt.false_body)
        elif isinstance(stmt, AxonBind) and isinstance(stmt.expr, AxonExprDo):
            nested = stmt.expr.body
        if nested and _has_return_statement(nested):
            return True
    return False


def _do_expr_requires_return(expr: AxonExpr, *, module_index: int, module_name: str) -> None:
    if not isinstance(expr, AxonExprDo):
        return
    if not expr.body:
        raise ValueError(
            f"Axon syntax validation failed at module[{module_index}] ({module_name}): "
            "'do' expression requires at least one statement"
        )
    if not _has_return_statement(expr.body):
        raise ValueError(
            f"Axon syntax validation failed at module[{module_index}] ({module_name}): "
            "'do' expression requires at least one return statement"
        )


def validate_parsed_program_source(parsed_source: Any) -> None:
    modules = parsed_source.modules
    seen_module_decls: set[str] = set()
    for idx, module in enumerate(modules):
        sig_decl = module.signature.module_decl.strip()
        def_decl = module.definition.module_decl.strip()
        if not sig_decl:
            raise ValueError(
                f"Axon syntax validation failed at module[{idx}]: empty module declaration in signature"
            )
        if not _is_mod_decl(sig_decl):
            raise ValueError(
                f"Axon syntax validation failed at module[{idx}]: invalid module declaration {sig_decl!r}"
            )
        if sig_decl in seen_module_decls:
            raise ValueError(
                f"Axon syntax validation failed: duplicate module declaration {sig_decl!r}"
            )
        seen_module_decls.add(sig_decl)
        if not _is_mod_decl(def_decl):
            raise ValueError(
                f"Axon syntax validation failed at module[{idx}]: invalid definition declaration {def_decl!r}"
            )

        sig_base, sig_path_params = _split_module_path_params(sig_decl)
        def_base, def_path_params = _split_module_path_params(def_decl)
        if sig_base != def_base:
            raise ValueError(
                "Axon syntax validation failed at module"
                f"[{idx}]: signature/definition name mismatch: {sig_decl!r} != {def_decl!r}"
            )
        if sig_path_params and def_path_params and sig_path_params != def_path_params:
            raise ValueError(
                "Axon syntax validation failed at module"
                f"[{idx}]: signature/definition path-parameter mismatch: "
                f"{sig_decl!r} != {def_decl!r}"
            )

        _do_expr_requires_return(module.definition.rhs, module_index=idx, module_name=sig_decl)

    imported_namespaces = set(parsed_source.imports)
    for namespace in parsed_source.imported_members:
        if namespace not in imported_namespaces:
            raise ValueError(
                "Axon syntax validation failed: imported-members entry without matching "
                f"'import {namespace}'"
            )
    for exported in parsed_source.exports:
        if not _is_ident(exported):
            raise ValueError(f"Axon syntax validation failed: invalid exported symbol {exported!r}")


__all__ = ["validate_parsed_program_source"]
