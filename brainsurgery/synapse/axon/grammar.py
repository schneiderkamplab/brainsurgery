from __future__ import annotations

import re
from dataclasses import dataclass

from lark import Lark
from lark.exceptions import LarkError
from lark.tree import Tree

_GRAMMAR = r"""
line: sig_line
    | def_line
    | const_line
    | import_line
    | padding_pragma
    | for_stmt
    | scope_bind_stmt
    | scope_stmt
    | return_stmt
    | bind_stmt

sig_line: MOD_DECL "::" type_expr
def_line: MOD_DECL IDENT* "=" DEF_RHS
const_line: IDENT "=" CONST_EXPR
import_line: "import" MOD_NAME [import_members]
padding_pragma: "{-#" "PADDING_SIDE" QUOTED_SIDE "#-}"

for_stmt: "for" ["@" SCOPE] IDENT "<-" RANGE ["step" "=" STEP_EXPR] "do"
scope_bind_stmt: TARGETS "<-" "scope" ["@"] SCOPE "do"
scope_stmt: "scope" ["@"] SCOPE "do"
return_stmt: "return" EXPR
bind_stmt: TARGETS "<-" EXPR

MOD_NAME: /[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*/
MOD_DECL: /[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:@[A-Za-z_][A-Za-z0-9_]*)*/
TYPE_NAME: /[@?]?[A-Za-z_][A-Za-z0-9_.]*/
INT: /[0-9]+/
QUOTED_SIDE: /"(?:left|right)"|'(?:left|right)'/

SCOPE: /[A-Za-z_][A-Za-z0-9_.]*/
IDENT: /[A-Za-z_][A-Za-z0-9_]*/
RANGE: /[\[\(]\s*.+?\s*\.\.\s*.+?\s*[\]\)\[]/
STEP_EXPR: /.+?(?=\s+do$)/
TARGETS: /.+?(?=<-)/
EXPR: /.+?(?=\s*(--|$))/
DEF_RHS: /.+?(?=\s*(--|$))/
CONST_EXPR: /.+?(?=\s*(--|$))/
COMMENT: /--[^\n]*/

?type_expr: type_term ("->" type_expr)?
?type_term: tuple_type | type_atom
tuple_type: "(" type_expr ("," type_expr)+ ")"
type_atom: TYPE_NAME [type_params]
type_params: "[" [type_arg ("," type_arg)*] "]"
?type_arg: TYPE_NAME | INT

import_members: import_members_paren | import_members_bare
import_members_paren: "(" [IDENT ("," IDENT)*] ")"
import_members_bare: IDENT+

%import common.WS
%ignore WS
%ignore COMMENT
"""

_RANGE_RE = re.compile(r"^([\[\(])\s*(.+?)\s*\.\.\s*(.+?)\s*([\]\)\[])$")

_PARSER = Lark(
    _GRAMMAR,
    parser="lalr",
    start=[
        "sig_line",
        "def_line",
        "const_line",
        "import_line",
        "padding_pragma",
        "for_stmt",
        "scope_bind_stmt",
        "scope_stmt",
        "return_stmt",
        "bind_stmt",
    ],
)


@dataclass(frozen=True)
class ParsedSignature:
    module_decl: str
    type_expr: str


@dataclass(frozen=True)
class ParsedImport:
    namespace: str
    members_tail: str


@dataclass(frozen=True)
class ParsedDefinition:
    module_decl: str
    args: tuple[str, ...]
    rhs: str


@dataclass(frozen=True)
class ParsedConstant:
    name: str
    value: str


@dataclass(frozen=True)
class ParsedFor:
    name: str | None
    var: str
    start_delim: str
    start_expr: str
    end_expr: str
    end_delim: str
    step_expr: str | None


@dataclass(frozen=True)
class ParsedScopeBind:
    raw_targets: str
    prefix: str


@dataclass(frozen=True)
class ParsedScope:
    prefix: str


@dataclass(frozen=True)
class ParsedReturn:
    raw_values: str


@dataclass(frozen=True)
class ParsedBind:
    raw_targets: str
    expr: str


@dataclass(frozen=True)
class ParsedModuleSource:
    signature: ParsedSignature
    definition_line: str
    body_lines: tuple[str, ...]


@dataclass(frozen=True)
class ParsedProgramSource:
    modules: tuple[ParsedModuleSource, ...]
    imports: tuple[str, ...]
    imported_members: dict[str, tuple[str, ...]]
    pragmas: dict[str, object]
    constants: dict[str, str]


def _parse_lark_start(text: str, *, start: str) -> Tree | None:
    try:
        return _PARSER.parse(text, start=start)
    except LarkError:
        return None


def _first_token_value(tree: Tree, token_type: str) -> str | None:
    for child in tree.children:
        if getattr(child, "type", None) == token_type:
            return str(child)
    return None


def _parse_range(raw: str) -> tuple[str, str, str, str]:
    match = _RANGE_RE.match(raw.strip())
    if match is None:
        raise ValueError(f"invalid for-range expression: {raw!r}")
    return match.group(1), match.group(2).strip(), match.group(3).strip(), match.group(4)


def parse_signature_line(line: str) -> ParsedSignature | None:
    text = line.strip()
    if not text:
        return None
    tree = _parse_lark_start(text, start="sig_line")
    if tree is None:
        return None
    mod_decl = _first_token_value(tree, "MOD_DECL")
    if mod_decl is None:
        return None
    _, sep, rhs = text.partition("::")
    if not sep:
        return None
    type_expr = rhs.strip()
    if not type_expr:
        return None
    return ParsedSignature(module_decl=mod_decl.strip(), type_expr=type_expr)


def parse_definition_line(line: str) -> ParsedDefinition | None:
    text = line.strip()
    if not text:
        return None
    tree = _parse_lark_start(text, start="def_line")
    if tree is None:
        return None
    mod_decl = _first_token_value(tree, "MOD_DECL")
    rhs = _first_token_value(tree, "DEF_RHS")
    if mod_decl is None or rhs is None:
        return None
    args: list[str] = []
    for child in tree.children:
        if getattr(child, "type", None) == "IDENT":
            args.append(str(child).strip())
    rhs_text = rhs.strip()
    if not rhs_text:
        return None
    return ParsedDefinition(module_decl=mod_decl.strip(), args=tuple(args), rhs=rhs_text)


def parse_constant_line(line: str) -> ParsedConstant | None:
    text = line.strip()
    if not text:
        return None
    tree = _parse_lark_start(text, start="const_line")
    if tree is None:
        return None
    name = _first_token_value(tree, "IDENT")
    value = _first_token_value(tree, "CONST_EXPR")
    if name is None or value is None:
        return None
    return ParsedConstant(name=name.strip(), value=value.strip())


def parse_import_line(line: str) -> ParsedImport | None:
    text = line.strip()
    if not text:
        return None
    tree = _parse_lark_start(text, start="import_line")
    if tree is None:
        return None
    namespace = _first_token_value(tree, "MOD_NAME")
    if namespace is None:
        return None
    parts = text.split(None, 2)
    members_tail = parts[2].strip() if len(parts) > 2 else ""
    return ParsedImport(namespace=namespace.strip(), members_tail=members_tail)


def parse_padding_side_pragma(line: str) -> str | None:
    text = line.strip()
    if not text:
        return None
    tree = _parse_lark_start(text, start="padding_pragma")
    if tree is None:
        return None
    quoted_side = _first_token_value(tree, "QUOTED_SIDE")
    if quoted_side is None or len(quoted_side) < 2:
        return None
    return quoted_side[1:-1].lower()


ParsedStatementHead = ParsedFor | ParsedScopeBind | ParsedScope | ParsedReturn | ParsedBind


def parse_statement_head(line: str) -> ParsedStatementHead:
    text = line.strip()
    if not text:
        raise ValueError("empty Axon statement")
    for start in ("for_stmt", "scope_bind_stmt", "scope_stmt", "return_stmt", "bind_stmt"):
        tree = _parse_lark_start(text, start=start)
        if tree is None:
            continue
        if start == "for_stmt":
            raw_scope = _first_token_value(tree, "SCOPE")
            raw_var = _first_token_value(tree, "IDENT")
            raw_range = _first_token_value(tree, "RANGE")
            raw_step = _first_token_value(tree, "STEP_EXPR")
            if raw_var is None or raw_range is None:
                raise ValueError(f"invalid for statement: {line!r}")
            start_delim, start_expr, end_expr, end_delim = _parse_range(raw_range)
            return ParsedFor(
                name=raw_scope.strip() if raw_scope else None,
                var=raw_var.strip(),
                start_delim=start_delim,
                start_expr=start_expr,
                end_expr=end_expr,
                end_delim=end_delim,
                step_expr=raw_step.strip() if raw_step else None,
            )
        if start == "scope_bind_stmt":
            raw_targets = _first_token_value(tree, "TARGETS")
            raw_scope = _first_token_value(tree, "SCOPE")
            if raw_targets is None or raw_scope is None:
                raise ValueError(f"invalid scope bind statement: {line!r}")
            return ParsedScopeBind(raw_targets=raw_targets.strip(), prefix=raw_scope.strip())
        if start == "scope_stmt":
            raw_scope = _first_token_value(tree, "SCOPE")
            if raw_scope is None:
                raise ValueError(f"invalid scope statement: {line!r}")
            return ParsedScope(prefix=raw_scope.strip())
        if start == "return_stmt":
            raw_values = _first_token_value(tree, "EXPR")
            if raw_values is None:
                raise ValueError(f"invalid return statement: {line!r}")
            return ParsedReturn(raw_values=raw_values.strip())
        if start == "bind_stmt":
            raw_targets = _first_token_value(tree, "TARGETS")
            expr = _first_token_value(tree, "EXPR")
            if raw_targets is None or expr is None:
                raise ValueError(f"invalid bind statement: {line!r}")
            return ParsedBind(raw_targets=raw_targets.strip(), expr=expr.strip())
    raise ValueError(f"unsupported Axon statement: {line!r}")


def _is_ident(token: str) -> bool:
    if not token:
        return False
    if not (token[0].isalpha() or token[0] == "_"):
        return False
    return all(ch.isalnum() or ch == "_" for ch in token[1:])


def _normalized_source_lines(source: str) -> list[str]:
    out: list[str] = []
    for raw in source.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("--"):
            continue
        out.append(line)
    return out


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
        parts = [part.strip() for part in inner.split(",") if part.strip()]
    else:
        normalized = text.replace(",", " ")
        parts = [part.strip() for part in normalized.split() if part.strip()]
    for token in parts:
        if not _is_ident(token):
            raise ValueError(f"invalid imported member name: {token!r}")
    return tuple(dict.fromkeys(parts))


def parse_program_source(source: str) -> ParsedProgramSource:
    raw_lines = _normalized_source_lines(source)
    kept_lines: list[str] = []
    pragmas: dict[str, object] = {}
    constants: dict[str, str] = {}
    imports: list[str] = []
    imported_members: dict[str, tuple[str, ...]] = {}
    prev_was_sig = False
    for line in raw_lines:
        if len(line) != len(line.lstrip(" ")):
            kept_lines.append(line)
            prev_was_sig = False
            continue
        stripped = line.strip()
        pad = parse_padding_side_pragma(stripped)
        if pad is not None:
            prev = pragmas.get("padding_side")
            if prev is not None and prev != pad:
                raise ValueError(
                    "conflicting PADDING_SIDE pragmas; expected a single consistent value"
                )
            pragmas["padding_side"] = pad
            prev_was_sig = False
            continue
        parsed_import = parse_import_line(stripped)
        if parsed_import is not None:
            namespace = parsed_import.namespace
            imports.append(namespace)
            members = _parse_import_members(parsed_import.members_tail)
            if members:
                prev_members = imported_members.get(namespace, ())
                imported_members[namespace] = tuple(dict.fromkeys([*prev_members, *members]))
            prev_was_sig = False
            continue
        parsed_sig = parse_signature_line(stripped)
        if parsed_sig is not None:
            kept_lines.append(line)
            prev_was_sig = True
            continue
        parsed_const = parse_constant_line(stripped)
        if parsed_const is not None and not prev_was_sig:
            constants[parsed_const.name] = parsed_const.value
            prev_was_sig = False
            continue
        kept_lines.append(line)
        prev_was_sig = False

    module_starts: list[int] = []
    for idx, line in enumerate(kept_lines):
        if len(line) != len(line.lstrip(" ")):
            continue
        if parse_signature_line(line.strip()) is not None:
            module_starts.append(idx)
    modules: tuple[ParsedModuleSource, ...]
    if not module_starts:
        modules = ()
    else:
        module_list: list[ParsedModuleSource] = []
        for i, start in enumerate(module_starts):
            end = module_starts[i + 1] if i + 1 < len(module_starts) else len(kept_lines)
            block = kept_lines[start:end]
            if len(block) < 2:
                continue
            parsed_sig = parse_signature_line(block[0].strip())
            if parsed_sig is None:
                continue
            parsed_def = parse_definition_line(block[1].strip())
            if parsed_def is None:
                continue
            def_line = (
                f"{parsed_def.module_decl} {' '.join(parsed_def.args)} = {parsed_def.rhs}".rstrip()
            )
            body_lines = tuple(block[2:])
            module_list.append(
                ParsedModuleSource(
                    signature=parsed_sig,
                    definition_line=def_line,
                    body_lines=body_lines,
                )
            )
        modules = tuple(module_list)
    return ParsedProgramSource(
        modules=modules,
        imports=tuple(dict.fromkeys(imports)),
        imported_members=imported_members,
        pragmas=pragmas,
        constants=constants,
    )
