from __future__ import annotations

from pathlib import Path

from .ast_validation import validate_axon_program
from .call_parser import split_top_level as _split_top_level_shared
from .expression_parser import parse_expression
from .grammar import (
    ParsedBind,
    ParsedFor,
    ParsedModuleSource,
    ParsedProgramSource,
    ParsedReturn,
    ParsedScope,
    ParsedScopeBind,
    ParsedSignature,
    parse_program_source,
    parse_statement_head,
)
from .syntax_validation import validate_parsed_program_source
from .types import (
    AxonBind,
    AxonExpr,
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


def _is_mod_decl(token: str) -> bool:
    parts = token.split("@")
    if not parts:
        return False
    if not _is_mod_name(parts[0]):
        return False
    return all(_is_ident(part) for part in parts[1:])


def _is_simple_callee(token: str) -> bool:
    if not token:
        return False
    if not _is_ident(token[0]):
        if not (token[0].isalpha() or token[0] == "_"):
            return False
    for ch in token:
        if not (ch.isalnum() or ch in "_.:@"):
            return False
    return True


def _parse_def_line(line: str) -> tuple[str, list[str], str | None]:
    if "=" not in line:
        raise ValueError(f"invalid Axon definition line: {line!r}")
    left, right = line.split("=", 1)
    rhs = right.strip()
    left_parts = [part for part in left.strip().split() if part]
    if not left_parts:
        raise ValueError(f"invalid Axon definition line: {line!r}")
    decl = left_parts[0]
    if not _is_mod_decl(decl):
        raise ValueError(f"invalid Axon definition name: {decl!r}")
    args = left_parts[1:]
    if rhs == "do":
        return decl, args, None
    if not rhs:
        raise ValueError(f"invalid Axon definition line: {line!r}")
    return decl, args, rhs


def _parse_path_sig_annotation(token: str) -> tuple[str, str] | None:
    stripped = token.strip()
    if not stripped.startswith("@"):
        return None
    body = stripped[1:].strip()
    if not body:
        return None
    if ":" in body:
        left, right = body.split(":", 1)
        name = left.strip()
        type_name = right.strip()
        if not _is_ident(name) or not _is_ident(type_name):
            return None
        return name, type_name
    if not _is_ident(body):
        return None
    return body, body


def _split_top_level_csv(text: str) -> list[str]:
    return _split_top_level_shared(text, ",")


def _split_top_level(text: str, sep: str) -> list[str]:
    return _split_top_level_shared(text, sep)


def _parse_params(raw: str) -> tuple[AxonParam, ...]:
    if not raw.strip():
        return ()
    out: list[AxonParam] = []
    for token in _split_top_level_csv(raw):
        if token.endswith("?"):
            out.append(AxonParam(name=token[:-1].strip(), optional=True))
        else:
            out.append(AxonParam(name=token.strip(), optional=False))
    return tuple(out)


def _shape_dims_from_type(type_expr: str) -> tuple[str, ...] | None:
    text = type_expr.strip()
    if "[" not in text or not text.endswith("]"):
        return None
    base, inner = text.split("[", 1)
    if not _is_ident(base.strip()):
        return None
    inner = inner[:-1].strip()
    dims = tuple(part.strip() for part in _split_top_level_csv(inner) if part.strip())
    if not dims:
        return None
    return dims


def _parse_const_scalar(token: str) -> object:
    value = token.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null":
        return None
    if value and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        if "." in value or "e" in value.lower():
            return float(value)
    except ValueError:
        pass
    return value


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
    definition_line: str,
) -> (
    tuple[
        str,
        str | None,
        tuple[str, ...],
        tuple[AxonParam, ...],
        tuple[str, ...],
        int,
        AxonExpr | None,
        dict[str, object],
        str | None,
        tuple[str, ...] | None,
    ]
    | None
):
    try:
        name_def_raw, arg_names, inline_expr = _parse_def_line(definition_line)
    except ValueError:
        return None

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

    sig_expr = signature.type_expr.strip()
    parts = _split_top_level(sig_expr, "->")
    if len(parts) < 1:
        raise ValueError("invalid Axon type signature")
    arg_types = parts[:-1]
    consumed_path_types = 0
    while consumed_path_types < len(arg_types):
        current = arg_types[consumed_path_types].strip()
        parsed_path_sig = _parse_path_sig_annotation(current)
        if parsed_path_sig is None:
            break
        path_sig_name, path_sig_type = parsed_path_sig
        if path_sig_type == path_sig_name:
            if consumed_path_types >= len(path_params):
                raise ValueError(
                    "path signature annotation count exceeds module path parameters in definition"
                )
            path_sig_name = path_params[consumed_path_types]
        if path_sig_type != "Path":
            raise ValueError(
                f"path signature type must be Path, got {path_sig_type!r}. Use '@Path'."
            )
        if not path_params:
            raise ValueError(
                "path signature annotation requires a module path parameter in the definition"
            )
        expected_name = (
            path_params[consumed_path_types] if consumed_path_types < len(path_params) else None
        )
        if path_sig_name != expected_name:
            raise ValueError(
                "path signature parameter does not match module path parameter:"
                f" {path_sig_name!r} != {expected_name!r}"
            )
        consumed_path_types += 1
    if consumed_path_types != len(path_params):
        raise ValueError("path signature annotation count must match module path parameter count")
    arg_types = arg_types[consumed_path_types:]
    raw_return_type = parts[-1].strip()
    opt_flags = [arg.strip().startswith("?") for arg in arg_types]

    if len(arg_names) != len(opt_flags):
        allow_pointfree_eta = (
            len(arg_names) == 0
            and len(opt_flags) > 0
            and inline_expr is not None
            and _is_simple_callee(inline_expr)
        )
        if not allow_pointfree_eta:
            raise ValueError(
                f"signature arg count ({len(opt_flags)}) does not match definition args ({len(arg_names)})"
            )
        arg_names = [f"arg_{idx}" for idx in range(len(opt_flags))]
        inline_expr = f"{inline_expr} {' '.join(arg_names)}"
    annotation_symbols: dict[str, object] = {}
    params_out: list[AxonParam] = []
    for idx, arg_name in enumerate(arg_names):
        raw_type = arg_types[idx].strip()
        clean_type = raw_type[1:].strip() if raw_type.startswith("?") else raw_type
        shape = _shape_dims_from_type(clean_type)
        if shape is not None:
            for dim in shape:
                annotation_symbols.setdefault(dim, None)
        params_out.append(
            AxonParam(
                name=arg_name.strip(),
                optional=opt_flags[idx],
                type_expr=clean_type,
                shape=shape,
            )
        )
    ret_shape = _shape_dims_from_type(raw_return_type)
    if ret_shape is not None:
        for dim in ret_shape:
            annotation_symbols.setdefault(dim, None)
    params = tuple(params_out)
    # Haskell-style signatures carry output types, not names. Return names will be inferred from `return`.
    parsed_inline_expr = parse_expression(inline_expr) if inline_expr is not None else None
    return (
        name_sig,
        path_param,
        path_params,
        params,
        (),
        0,
        parsed_inline_expr,
        annotation_symbols,
        raw_return_type,
        ret_shape,
    )


def _build_module_from_lines(
    *,
    module_source: ParsedModuleSource,
    top_pragmas: dict[str, object],
    top_constants: dict[str, object],
    imports: tuple[str, ...],
    imported_members: dict[str, tuple[str, ...]],
) -> AxonModule:
    lines = list(module_source.body_lines)
    if not module_source.signature.module_decl:
        raise ValueError("empty Axon source")

    parsed = _parse_haskell_header(
        signature=module_source.signature,
        definition_line=module_source.definition_line,
    )
    if parsed is None:
        raise ValueError("expected haskell-style pair: '<name> :: ...' + '<name> ... = do|<expr>'")
    (
        module_name,
        module_path_param,
        module_path_params,
        params,
        returns,
        body_start,
        inline_expr,
        annotation_symbols,
        return_type_expr,
        return_shape,
    ) = parsed

    if inline_expr is not None:
        module = AxonModule(
            name=module_name,
            path_param=module_path_param,
            path_params=module_path_params,
            params=params,
            returns=returns,
            statements=(AxonReturn(values=(inline_expr,)),),
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

    entries = _line_entries(lines[body_start:])
    if not entries:
        module = AxonModule(
            name=module_name,
            path_param=module_path_param,
            path_params=module_path_params,
            params=params,
            returns=returns,
            statements=(),
            imports=imports,
            imported_members=imported_members or None,
            symbols=top_constants if top_constants else None,
            pragmas=top_pragmas if top_pragmas else None,
            return_type_expr=return_type_expr,
            return_shape=return_shape,
        )
        return module
    base_indent = min(indent for indent, _ in entries)
    statements, index = _parse_statements(entries, 0, base_indent)
    if index != len(entries):
        raise ValueError("unexpected trailing lines in module body")

    module = AxonModule(
        name=module_name,
        path_param=module_path_param,
        path_params=module_path_params,
        params=params,
        returns=returns,
        statements=tuple(statements),
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
    parsed_source: "ParsedProgramSource", *, validate: bool = True
) -> tuple[AxonModule, ...]:
    top_pragmas = parsed_source.pragmas
    top_constants = {k: _parse_const_scalar(v) for k, v in parsed_source.constants.items()}
    top_imports = parsed_source.imports
    top_imported_members = parsed_source.imported_members
    modules_list: list[AxonModule] = []
    for chunk_source in parsed_source.modules:
        modules_list.append(
            _build_module_from_lines(
                module_source=chunk_source,
                top_pragmas=top_pragmas,
                top_constants=top_constants,
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


def _parse_simple_line(line: str) -> AxonBind | AxonReturn:
    parsed = parse_statement_head(line)
    if isinstance(parsed, ParsedReturn):
        values = tuple(parse_expression(part) for part in _split_top_level_csv(parsed.raw_values))
        return AxonReturn(values=values)
    if isinstance(parsed, ParsedBind):
        targets = tuple(part.strip() for part in _split_top_level_csv(parsed.raw_targets))
        return AxonBind(targets=targets, expr=parse_expression(parsed.expr))
    raise ValueError(f"unsupported Axon statement: {line!r}")


def _line_entries(lines: list[str]) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for raw in lines:
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        entries.append((indent, raw.strip()))
    return entries


def _parse_statements(
    lines: list[tuple[int, str]],
    start: int,
    current_indent: int,
) -> tuple[list[AxonStatement], int]:
    out: list[AxonStatement] = []
    i = start
    while i < len(lines):
        indent, line = lines[i]
        if indent < current_indent:
            return out, i
        if indent > current_indent:
            raise ValueError(f"unexpected indentation at line: {line!r}")

        try:
            parsed_head: (
                ParsedFor | ParsedScopeBind | ParsedScope | ParsedReturn | ParsedBind | None
            )
            parsed_head = parse_statement_head(line)
        except ValueError:
            parsed_head = None

        if isinstance(parsed_head, ParsedFor):
            repeat_name = parsed_head.name
            var = parsed_head.var
            start_expr = (
                parsed_head.start_expr
                if parsed_head.start_delim == "["
                else f"({parsed_head.start_expr}) + 1"
            )
            end_exclusive = (
                f"({parsed_head.end_expr}) + 1"
                if parsed_head.end_delim == "]"
                else parsed_head.end_expr
            )
            step_expr = parsed_head.step_expr if parsed_head.step_expr else "1"
            if i + 1 >= len(lines):
                raise ValueError("for@ requires indented body")
            next_indent, _ = lines[i + 1]
            if next_indent <= indent:
                raise ValueError("for@ requires indented body")
            body, new_i = _parse_statements(lines, i + 1, next_indent)
            out.append(
                AxonRepeat(
                    name=repeat_name,
                    var=var,
                    to_expr=parse_expression(end_exclusive),
                    from_expr=parse_expression(start_expr),
                    step_expr=parse_expression(step_expr),
                    body=tuple(body),
                )
            )
            i = new_i
            continue
        if isinstance(parsed_head, ParsedScopeBind):
            raw_targets = parsed_head.raw_targets
            targets = tuple(part.strip() for part in _split_top_level_csv(raw_targets))
            if not targets:
                raise ValueError("scope bind requires one or more targets")
            prefix = parsed_head.prefix
            if i + 1 >= len(lines):
                raise ValueError("scope bind requires indented body")
            next_indent, _ = lines[i + 1]
            if next_indent <= indent:
                raise ValueError("scope bind requires indented body")
            body, new_i = _parse_statements(lines, i + 1, next_indent)
            out.append(AxonScopeBind(targets=targets, prefix=prefix, body=tuple(body)))
            i = new_i
            continue
        if isinstance(parsed_head, ParsedScope):
            raise ValueError(
                "scope statement form is not supported; use '<target> <- scope@name do ... return ...'"
            )
        if isinstance(parsed_head, ParsedBind) and parsed_head.expr.strip() == "do":
            targets = tuple(part.strip() for part in _split_top_level_csv(parsed_head.raw_targets))
            if not targets:
                raise ValueError("do-expression bind requires one or more targets")
            if i + 1 >= len(lines):
                raise ValueError("do-expression bind requires indented body")
            next_indent, _ = lines[i + 1]
            if next_indent <= indent:
                raise ValueError("do-expression bind requires indented body")
            body, new_i = _parse_statements(lines, i + 1, next_indent)
            out.append(AxonScopeBind(targets=targets, prefix=f"__do_expr_{i}", body=tuple(body)))
            i = new_i
            continue

        while i + 1 < len(lines):
            nxt_indent, nxt = lines[i + 1]
            current = line.rstrip()
            nxt_line = nxt.strip()
            current_continues = current.endswith("|>") or current.endswith(">>=")
            next_is_continuation = nxt_line.startswith("|>") or nxt_line.startswith(">>=")
            if nxt_indent > indent and (current_continues or next_is_continuation):
                line = f"{current} {nxt_line}"
                i += 1
                continue
            break

        out.append(_parse_simple_line(line))
        i += 1

    return out, i


def parse_axon_program(source: str) -> tuple[AxonModule, ...]:
    parsed_source = parse_program_source(source)
    validate_parsed_program_source(parsed_source)
    return build_axon_modules_from_parsed_source(parsed_source, validate=True)


def parse_axon_program_from_path(path: "Path") -> tuple[AxonModule, ...]:
    from .import_loader import load_axon_program_from_path

    return load_axon_program_from_path(path)


__all__ = [
    "build_axon_modules_from_parsed_source",
    "parse_axon_module",
    "parse_axon_program",
    "parse_axon_program_from_path",
]
