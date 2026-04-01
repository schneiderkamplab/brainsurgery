from __future__ import annotations

from .grammar import ParsedProgramSource, ParsedReturn, parse_definition_line, parse_statement_head


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


def _contains_return_statement(body_lines: tuple[str, ...]) -> bool:
    for raw_line in body_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            parsed = parse_statement_head(stripped)
        except ValueError:
            continue
        if isinstance(parsed, ParsedReturn):
            return True
    return False


def validate_parsed_program_source(parsed_source: ParsedProgramSource) -> None:
    modules = parsed_source.modules
    if not modules:
        raise ValueError("Axon syntax validation failed: program must contain at least one module")

    seen_module_decls: set[str] = set()
    for idx, module in enumerate(modules):
        sig_decl = module.signature.module_decl.strip()
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

        parsed_def = parse_definition_line(module.definition_line.strip())
        if parsed_def is None:
            raise ValueError(
                f"Axon syntax validation failed at module[{idx}]: invalid definition line {module.definition_line!r}"
            )
        def_decl = parsed_def.module_decl.strip()
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

        is_do_form = parsed_def.rhs.strip() == "do"
        has_body = bool(module.body_lines)
        if is_do_form and not has_body:
            raise ValueError(
                f"Axon syntax validation failed at module[{idx}] ({sig_decl}): 'do' definition requires an indented body"
            )
        if is_do_form and not _contains_return_statement(module.body_lines):
            raise ValueError(
                f"Axon syntax validation failed at module[{idx}] ({sig_decl}): 'do' definition requires at least one return statement"
            )
        if not is_do_form and has_body:
            raise ValueError(
                f"Axon syntax validation failed at module[{idx}] ({sig_decl}): expression definition cannot have an indented body"
            )

    imported_namespaces = set(parsed_source.imports)
    for namespace in parsed_source.imported_members:
        if namespace not in imported_namespaces:
            raise ValueError(
                "Axon syntax validation failed: imported-members entry without matching "
                f"'import {namespace}'"
            )


__all__ = ["validate_parsed_program_source"]
