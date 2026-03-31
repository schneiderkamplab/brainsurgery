from __future__ import annotations

from dataclasses import dataclass

from lark import Lark
from lark.exceptions import LarkError
from lark.tree import Tree

_GRAMMAR = r"""
sig_line: MOD_DECL "::" TYPE_EXPR
import_line: "import" MOD_NAME [IMPORT_TAIL]
padding_pragma: "{-#" "PADDING_SIDE" QUOTED_SIDE "#-}"

MOD_NAME: /[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*/
MOD_DECL: /[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:@[A-Za-z_][A-Za-z0-9_]*)*/
TYPE_EXPR: /.+/
IMPORT_TAIL: /.+/
QUOTED_SIDE: /"(?:left|right)"|'(?:left|right)'/

%import common.WS
%ignore WS
"""


@dataclass(frozen=True)
class ParsedSignature:
    module_decl: str
    type_expr: str


@dataclass(frozen=True)
class ParsedImport:
    namespace: str
    members_tail: str


_PARSER = Lark(_GRAMMAR, parser="lalr", start=["sig_line", "import_line", "padding_pragma"])


def _first_token_value(tree: Tree, token_type: str) -> str | None:
    for child in tree.children:
        if getattr(child, "type", None) == token_type:
            return str(child)
    return None


def parse_signature_line(line: str) -> ParsedSignature | None:
    text = line.strip()
    if not text:
        return None
    try:
        tree = _PARSER.parse(text, start="sig_line")
    except LarkError:
        return None
    mod_decl = _first_token_value(tree, "MOD_DECL")
    type_expr = _first_token_value(tree, "TYPE_EXPR")
    if mod_decl is None or type_expr is None:
        return None
    return ParsedSignature(module_decl=mod_decl.strip(), type_expr=type_expr.strip())


def parse_import_line(line: str) -> ParsedImport | None:
    text = line.strip()
    if not text:
        return None
    try:
        tree = _PARSER.parse(text, start="import_line")
    except LarkError:
        return None
    namespace = _first_token_value(tree, "MOD_NAME")
    if namespace is None:
        return None
    tail = _first_token_value(tree, "IMPORT_TAIL")
    return ParsedImport(namespace=namespace.strip(), members_tail=tail.strip() if tail else "")


def parse_padding_side_pragma(line: str) -> str | None:
    text = line.strip()
    if not text:
        return None
    try:
        tree = _PARSER.parse(text, start="padding_pragma")
    except LarkError:
        return None
    quoted_side = _first_token_value(tree, "QUOTED_SIDE")
    if quoted_side is None or len(quoted_side) < 2:
        return None
    return quoted_side[1:-1].lower()


__all__ = [
    "ParsedImport",
    "ParsedSignature",
    "parse_import_line",
    "parse_padding_side_pragma",
    "parse_signature_line",
]
