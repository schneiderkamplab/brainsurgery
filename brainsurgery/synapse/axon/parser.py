from __future__ import annotations

from pathlib import Path

from .ast_validation import validate_axon_program
from .call_parser import split_top_level as _split_top_level_shared
from .lark_statements import (
    ParsedBind,
    ParsedFor,
    ParsedReturn,
    ParsedScope,
    ParsedScopeBind,
    parse_statement_head,
)
from .lark_toplevel import parse_import_line, parse_padding_side_pragma, parse_signature_line
from .types import (
    AxonBind,
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


def _parse_top_const_line(line: str) -> tuple[str, str] | None:
    if "=" not in line:
        return None
    left, right = line.split("=", 1)
    name = left.strip()
    value = right.strip()
    if not _is_ident(name) or not value:
        return None
    return name, value


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


def _strip_haskell_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(line):
        prev = line[idx - 1] if idx > 0 else ""
        if ch == "'" and not in_double and prev != "\\":
            in_single = not in_single
            continue
        if ch == '"' and not in_single and prev != "\\":
            in_double = not in_double
            continue
        if (
            ch == "-"
            and not in_single
            and not in_double
            and idx + 1 < len(line)
            and line[idx + 1] == "-"
        ):
            return line[:idx]
    return line


def _normalized_source_lines(source: str) -> list[str]:
    out: list[str] = []
    for raw in source.splitlines():
        line = _strip_haskell_comment(raw).rstrip()
        if line.strip():
            out.append(line)
    return out


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


def _extract_top_level_constants(lines: list[str]) -> tuple[list[str], dict[str, object]]:
    body: list[str] = []
    constants: dict[str, object] = {}
    prev_was_sig = False
    for line in lines:
        if len(line) != len(line.lstrip(" ")):
            body.append(line)
            prev_was_sig = False
            continue
        stripped = line.strip()
        if parse_signature_line(stripped) is not None:
            body.append(line)
            prev_was_sig = True
            continue
        parsed = _parse_top_const_line(stripped)
        if parsed is not None and not prev_was_sig:
            key, raw_value = parsed
            constants[key] = _parse_const_scalar(raw_value)
            prev_was_sig = False
            continue
        body.append(line)
        prev_was_sig = False
    return body, constants


def _extract_top_level_pragmas(lines: list[str]) -> tuple[list[str], dict[str, object]]:
    body: list[str] = []
    pragmas: dict[str, object] = {}
    for line in lines:
        if len(line) != len(line.lstrip(" ")):
            body.append(line)
            continue
        stripped = line.strip()
        value = parse_padding_side_pragma(stripped)
        if value is not None:
            prev = pragmas.get("padding_side")
            if prev is not None and prev != value:
                raise ValueError(
                    "conflicting PADDING_SIDE pragmas; expected a single consistent value"
                )
            pragmas["padding_side"] = value
            continue
        body.append(line)
    return body, pragmas


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


def _parse_import_members(raw: str) -> tuple[str, ...]:
    text = raw.strip()
    if not text:
        return ()
    if text.startswith("("):
        if not text.endswith(")"):
            raise ValueError(f"invalid import member list: {raw!r}")
        inner = text[1:-1].strip()
        if not inner:
            return ()
        tokens = _split_top_level_csv(inner)
    else:
        normalized = text.replace(",", " ")
        tokens = [part.strip() for part in normalized.split() if part.strip()]
    if not tokens:
        return ()
    for token in tokens:
        if not _is_ident(token):
            raise ValueError(f"invalid imported member name: {token!r}")
    deduped = tuple(dict.fromkeys(tokens))
    return deduped


def _extract_top_level_imports(
    lines: list[str],
) -> tuple[list[str], tuple[str, ...], dict[str, tuple[str, ...]]]:
    body: list[str] = []
    imports: list[str] = []
    imported_members: dict[str, tuple[str, ...]] = {}
    for line in lines:
        if len(line) != len(line.lstrip(" ")):
            body.append(line)
            continue
        parsed = parse_import_line(line.strip())
        if parsed is not None:
            namespace = parsed.namespace
            imports.append(namespace)
            raw_members = parsed.members_tail
            members = _parse_import_members(raw_members)
            if members:
                prev = imported_members.get(namespace, ())
                imported_members[namespace] = tuple(dict.fromkeys([*prev, *members]))
            continue
        body.append(line)
    deduped = tuple(dict.fromkeys(imports))
    return body, deduped, imported_members


def _parse_haskell_header(
    lines: list[str],
) -> (
    tuple[
        str,
        str | None,
        tuple[str, ...],
        tuple[AxonParam, ...],
        tuple[str, ...],
        int,
        str | None,
        dict[str, object],
        str | None,
        tuple[str, ...] | None,
    ]
    | None
):
    if len(lines) < 2:
        return None
    parsed_sig = parse_signature_line(lines[0])
    if parsed_sig is None:
        return None
    try:
        name_def_raw, arg_names, inline_expr = _parse_def_line(lines[1])
    except ValueError:
        return None

    name_sig_raw = parsed_sig.module_decl
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

    sig_expr = parsed_sig.type_expr.strip()
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
    return (
        name_sig,
        path_param,
        path_params,
        params,
        (),
        2,
        inline_expr,
        annotation_symbols,
        raw_return_type,
        ret_shape,
    )


def parse_axon_module(source: str) -> AxonModule:
    lines, top_pragmas = _extract_top_level_pragmas(_normalized_source_lines(source))
    lines, top_constants = _extract_top_level_constants(lines)
    lines, imports, imported_members = _extract_top_level_imports(lines)
    if not lines:
        raise ValueError("empty Axon source")

    parsed = _parse_haskell_header(lines)
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
        validate_axon_program((module,), main_module=module.name)
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
        validate_axon_program((module,), main_module=module.name)
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
    validate_axon_program((module,), main_module=module.name)
    return module


def _parse_simple_line(line: str) -> AxonBind | AxonReturn:
    parsed = parse_statement_head(line)
    if isinstance(parsed, ParsedReturn):
        values = tuple(_split_top_level_csv(parsed.raw_values))
        return AxonReturn(values=values)
    if isinstance(parsed, ParsedBind):
        targets = tuple(part.strip() for part in _split_top_level_csv(parsed.raw_targets))
        return AxonBind(targets=targets, expr=parsed.expr)
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

        parsed_head: ParsedFor | ParsedScopeBind | ParsedScope | ParsedReturn | ParsedBind | None
        try:
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
                    to_expr=end_exclusive,
                    from_expr=start_expr,
                    step_expr=step_expr,
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
    raw_lines, top_pragmas = _extract_top_level_pragmas(_normalized_source_lines(source))
    raw_lines, top_constants = _extract_top_level_constants(raw_lines)
    raw_lines, top_imports, top_imported_members = _extract_top_level_imports(raw_lines)
    module_starts: list[int] = []
    for idx, line in enumerate(raw_lines):
        if len(line) != len(line.lstrip(" ")):
            continue
        stripped = line.strip()
        if parse_signature_line(stripped) is not None:
            module_starts.append(idx)
    if not module_starts:
        parsed_modules = (parse_axon_module(source),)
        validate_axon_program(parsed_modules)
        return parsed_modules

    modules_list: list[AxonModule] = []
    for i, start in enumerate(module_starts):
        end = module_starts[i + 1] if i + 1 < len(module_starts) else len(raw_lines)
        chunk = "\n".join(raw_lines[start:end]).strip()
        if not chunk:
            continue
        module = parse_axon_module(chunk)
        merged_imports = tuple(dict.fromkeys([*top_imports, *module.imports]))
        merged_imported_members: dict[str, tuple[str, ...]] = dict(top_imported_members)
        if module.imported_members:
            for namespace, members in module.imported_members.items():
                prev = merged_imported_members.get(namespace, ())
                merged_imported_members[namespace] = tuple(dict.fromkeys([*prev, *members]))
        modules_list.append(
            AxonModule(
                name=module.name,
                path_param=module.path_param,
                path_params=module.path_params,
                params=module.params,
                returns=module.returns,
                statements=module.statements,
                imports=merged_imports,
                imported_members=merged_imported_members or None,
                symbols=module.symbols,
                pragmas=module.pragmas,
                return_type_expr=module.return_type_expr,
                return_shape=module.return_shape,
            )
        )
        modules_list[-1] = _inject_pragmas(modules_list[-1], top_pragmas)
    if modules_list:
        modules_list[-1] = _inject_symbols_meta(modules_list[-1], top_constants)
    out = tuple(modules_list)
    validate_axon_program(out)
    return out


def parse_axon_program_from_path(path: Path) -> tuple[AxonModule, ...]:
    root = path.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Axon file not found: {root}")
    if not root.is_file():
        raise ValueError(f"Axon import root must be a file: {root}")

    seen_paths: set[Path] = set()
    visiting: list[Path] = []
    ordered_modules: list[AxonModule] = []

    builtins_dir = (Path(__file__).resolve().parents[1] / "builtins").resolve()
    prelude_file = (builtins_dir / "Prelude.axon").resolve()

    def _apply_namespace(
        modules: tuple[AxonModule, ...], namespace: str | None
    ) -> tuple[AxonModule, ...]:
        if not namespace:
            return modules
        namespaced: list[AxonModule] = []
        for module in modules:
            if "." in module.name:
                namespaced.append(module)
                continue
            namespaced.append(
                AxonModule(
                    name=f"{namespace}.{module.name}",
                    path_param=module.path_param,
                    path_params=module.path_params,
                    params=module.params,
                    returns=module.returns,
                    statements=module.statements,
                    imports=module.imports,
                    imported_members=module.imported_members,
                    symbols=module.symbols,
                    pragmas=module.pragmas,
                    return_type_expr=module.return_type_expr,
                    return_shape=module.return_shape,
                )
            )
        return tuple(namespaced)

    def _resolve_import_path(base_file: Path, import_name: str) -> Path:
        rel = Path(*import_name.split(".")).with_suffix(".axon")
        local_candidate = (base_file.parent / rel).resolve()
        if local_candidate.exists():
            return local_candidate
        builtin_candidate = (builtins_dir / rel).resolve()
        if builtin_candidate.exists():
            return builtin_candidate
        raise FileNotFoundError(
            f"Axon import {import_name!r} not found from {base_file}: "
            f"tried {local_candidate} and {builtin_candidate}"
        )

    def _load_file(file_path: Path, *, namespace: str | None = None) -> None:
        resolved = file_path.resolve()
        if resolved in seen_paths:
            return
        if resolved in visiting:
            cycle = " -> ".join(str(p) for p in [*visiting, resolved])
            raise ValueError(f"Cyclic Axon imports detected: {cycle}")
        visiting.append(resolved)
        source = resolved.read_text(encoding="utf-8")
        modules = _apply_namespace(parse_axon_program(source), namespace)
        import_names: set[str] = set()
        for module in modules:
            import_names.update(module.imports)
        for import_name in sorted(import_names):
            _load_file(_resolve_import_path(resolved, import_name), namespace=import_name)
        ordered_modules.extend(modules)
        seen_paths.add(resolved)
        visiting.pop()

    if prelude_file.exists() and prelude_file != root:
        _load_file(prelude_file, namespace="Prelude")
    _load_file(root)
    out = tuple(ordered_modules)
    validate_axon_program(out)
    return out


__all__ = ["parse_axon_module", "parse_axon_program", "parse_axon_program_from_path"]
