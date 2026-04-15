from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, cast

from lark import Lark, Token, Transformer
from lark.exceptions import LarkError, VisitError
from lark.indenter import DedentError, Indenter

from .type_system import (
    DimExprBinary,
    DimToken,
    TypeAny,
    TypeBool,
    TypeExpr,
    TypeFloat,
    TypeInt,
    TypeList,
    TypeNamed,
    TypeNull,
    TypeOptional,
    TypeString,
    TypeTensor,
    TypeTuple,
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
    AxonKwargValue,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
)


@dataclass(frozen=True)
class ParsedSignature:
    module_decl: str
    type_signature: ParsedFunctionType


@dataclass(frozen=True)
class ParsedPathTypeParam:
    name: str | None
    type_expr: TypeExpr


@dataclass(frozen=True)
class ParsedFunctionType:
    path_params: tuple[ParsedPathTypeParam, ...]
    arg_types: tuple[TypeExpr, ...]
    return_type: TypeExpr


@dataclass(frozen=True)
class ParsedDefParam:
    name: str
    default_expr: AxonExpr | None = None


@dataclass(frozen=True)
class ParsedDefinition:
    module_decl: str
    args: tuple[ParsedDefParam, ...]
    rhs: AxonExpr


@dataclass(frozen=True)
class ParsedModuleSource:
    signature: ParsedSignature
    definition: ParsedDefinition


@dataclass(frozen=True)
class ParsedProgramSource:
    modules: tuple[ParsedModuleSource, ...]
    imports: tuple[str, ...]
    imported_members: dict[str, tuple[str, ...]]
    exports: tuple[str, ...]
    pragmas: dict[str, object]
    constants: dict[str, AxonExpr]
    type_aliases: dict[str, TypeExpr]


@dataclass(frozen=True)
class _SuiteBody:
    body: tuple[AxonStatement, ...]
    inline: bool


class _AxonIndenter(Indenter):
    NL_type = "_NL"
    OPEN_PAREN_types = ["LPAR", "LSQB"]
    CLOSE_PAREN_types = ["RPAR", "RSQB"]
    INDENT_type = "INDENT"
    DEDENT_type = "DEDENT"
    tab_len = 8

    def __init__(self) -> None:
        super().__init__()
        self._continuation_indents: list[int] = []
        self._prev_type: str | None = None
        self._pending_paren_do_block = False
        self._paren_do_base_indents: list[int] = []

    def handle_NL(self, token: Token) -> Iterator[Token]:
        in_paren_do_block = bool(self._paren_do_base_indents)
        if self.paren_level > 0 and not (self._pending_paren_do_block or in_paren_do_block):
            return

        indent_str = token.rsplit("\n", 1)[1]
        indent = indent_str.count(" ") + indent_str.count("\t") * self.tab_len

        while self._continuation_indents and indent < self._continuation_indents[-1]:
            self._continuation_indents.pop()

        current = self.indent_level[-1]
        if (
            self._continuation_indents
            and indent == self._continuation_indents[-1]
            and indent >= current
        ):
            yield token
            return

        if self._pending_paren_do_block:
            self._pending_paren_do_block = False
            if indent > current:
                self._paren_do_base_indents.append(current)
                self.indent_level.append(indent)
                yield Token.new_borrow_pos(self.INDENT_type, indent_str, token)
                return
            if indent == current:
                yield token
                return

        if self._prev_type in {"PIPE_OP", "MONAD_BIND"} and indent > current:
            self._continuation_indents.append(indent)
            yield token
            return

        if indent == current:
            yield token
            return

        if indent > current:
            self.indent_level.append(indent)
            yield Token.new_borrow_pos(self.INDENT_type, indent_str, token)
            return

        while indent < self.indent_level[-1]:
            self.indent_level.pop()
            yield Token.new_borrow_pos(self.DEDENT_type, indent_str, token)

        if indent > self.indent_level[-1]:
            yield token
            return

        yield token

        while (
            self._paren_do_base_indents and self.indent_level[-1] == self._paren_do_base_indents[-1]
        ):
            self._paren_do_base_indents.pop()

        if indent != self.indent_level[-1]:
            raise DedentError(
                f"Unexpected dedent to column {indent}. Expected dedent to {self.indent_level[-1]}"
            )

    def process(self, stream: Iterator[Token]) -> Iterator[Token]:
        self.paren_level = 0
        self.indent_level = [0]
        self._continuation_indents = []
        self._prev_type = None
        self._pending_paren_do_block = False
        self._paren_do_base_indents = []
        return self._process(stream)

    def _process(self, stream: Iterator[Token]) -> Iterator[Token]:
        token: Token | None = None
        for token in stream:
            if token.type == self.NL_type:
                yield from self.handle_NL(token)
            else:
                yield token

            if token.type in self.OPEN_PAREN_types:
                self.paren_level += 1
            elif token.type in self.CLOSE_PAREN_types:
                self.paren_level -= 1
                assert self.paren_level >= 0

            if token.type == "DO" and self.paren_level > 0:
                self._pending_paren_do_block = True
            elif self._pending_paren_do_block and token.type != self.NL_type and token.type != "DO":
                self._pending_paren_do_block = False

            if token.type not in {self.NL_type, self.INDENT_type, self.DEDENT_type}:
                self._prev_type = token.type

        while len(self.indent_level) > 1:
            self.indent_level.pop()
            yield (
                Token.new_borrow_pos(self.DEDENT_type, "", token)
                if token
                else Token(self.DEDENT_type, "", 0, 0, 0, 0, 0, 0)
            )


_GRAMMAR = r"""
?start: program

program: _NL* top_item (_NL+ top_item)* _NL*

top_item: module_decl
    | import_decl
    | export_decl
    | pragma
    | constant
    | type_alias_decl

module_decl: signature _NL definition
signature: mod_decl TYPE_SEP signature_type
signature_type: signature_segment (LAMBDA_ARROW signature_segment)*
?signature_segment: path_type_annotation
    | type_expr
path_type_annotation: "@" NAME ":" type_expr -> path_type_named
    | "@" type_expr -> path_type_implicit

?type_expr: type_optional
?type_optional: "?" type_optional -> type_optional
    | type_tuple
    | type_list
    | type_tensor
    | type_name -> type_named
type_tuple: LPAR type_expr "," type_expr ("," type_expr)* ","? RPAR
type_list: "List" LSQB type_expr RSQB
type_tensor: type_name LSQB type_dim_expr ("," type_dim_expr)* RSQB
type_name: NAME

?type_dim_expr: type_dim_term
    | type_dim_expr ADD_OP type_dim_term -> type_dim_binary
?type_dim_term: type_dim_factor
    | type_dim_term MUL_OP type_dim_factor -> type_dim_binary
?type_dim_factor: INT -> type_dim_int
    | RANGE_DOTS type_name -> type_dim_rest
    | type_name -> type_dim_name
    | LPAR type_dim_expr RPAR -> type_dim_paren

definition: mod_decl def_param* "=" expr
def_param: NAME -> def_param_positional
    | "?" NAME "=" def_param_simple -> def_param_default
    | "?" NAME "=" LPAR expr RPAR -> def_param_default_paren
mod_decl: module_name mod_decl_param*
mod_decl_param: "@" NAME
module_name: NAME ("." NAME)*

import_decl: IMPORT import_item ("," import_item)*
import_item: module_name [import_members]
import_members: import_members_paren | import_members_bare
import_members_paren: LPAR [NAME ("," NAME)* ","?] RPAR
import_members_bare: NAME+
export_decl: EXPORT export_members
export_members: export_members_paren | export_members_bare
export_members_paren: LPAR [NAME ("," NAME)* ","?] RPAR
export_members_bare: NAME+

pragma: "{-#" NAME pragma_value "#-}"
?pragma_value: tuple_expr
    | list_expr
    | literal
constant: NAME "=" expr
type_alias_decl: TYPE_KW NAME "=" type_expr

?statement: for_statement
    | scope_bind_statement
    | return_statement
    | bind_statement

for_statement: FOR for_scope? NAME BIND_ARROW range_expr for_step? DO suite
for_scope: "@" scoped_name
for_step: STEP "=" expr

scope_bind_statement: target_list BIND_ARROW SCOPE_KW scope_ref scope_bind_kwarg* DO suite
scope_bind_kwarg: NAME "=" kwarg_value
scope_ref: "@"? scoped_name
scoped_name: NAME ("." NAME)*
return_statement: RETURN expr_list
bind_statement: target_list BIND_ARROW expr

target_list: NAME ("," NAME)*
expr_list: expr ("," expr)*

suite: inline_suite block_suite? -> suite_inline_maybe_block
    | block_suite -> suite_block

inline_suite: inline_statement (";" inline_statement)* ";"?
inline_statement: return_statement | bind_statement

block_suite: INDENT _NL* statement_line (_NL+ statement_line)* _NL* DEDENT
statement_line: statement (";" statement)* ";"?

range_expr: range_start expr RANGE_DOTS expr range_end
range_start: LSQB | LPAR
range_end: RSQB | RPAR

?expr: do_expr | bind_expr
do_expr: DO suite

?bind_expr: pipe_expr
    | pipe_expr MONAD_BIND nl_gap? lambda_expr -> bind_once

?pipe_expr: ternary_expr
    | pipe_expr PIPE_OP nl_gap? ternary_expr -> pipe

?ternary_expr: if_expr
    | or_expr "?" arg_ws? tuple_value arg_ws? ":" arg_ws? tuple_value -> ternary
    | or_expr INDENT "?" arg_ws? tuple_value arg_ws? ":" arg_ws? tuple_value DEDENT -> ternary

?if_expr: "if" arg_ws? expr arg_ws? "then" arg_ws? tuple_value arg_ws? "else" arg_ws? tuple_value -> if_expr
    | "if" arg_ws? expr arg_ws? "then" INDENT tuple_value DEDENT arg_ws? "else" INDENT tuple_value DEDENT -> if_expr
    | or_expr

?or_expr: and_expr
    | or_expr "or" and_expr -> or_expr

?and_expr: cmp_expr
    | and_expr "and" cmp_expr -> and_expr

?cmp_expr: add_expr
    | cmp_expr CMP_OP add_expr -> cmp_expr

?add_expr: mul_expr
    | add_expr ADD_OP mul_expr -> add_expr

?mul_expr: app_expr
    | mul_expr MUL_OP app_expr -> mul_expr

?app_expr: callable bare_arg+ -> bare_call
    | atom

bare_arg: kwarg_bare | arg_expr
kwarg_bare: NAME "=" kwarg_value

kwarg_value: path_lit | arg_expr
path_lit: PATH_LIT

?arg_expr: do_expr
    | arg_ternary
?arg_ternary: arg_if
    | arg_or "?" arg_ws? arg_expr arg_ws? ":" arg_ws? arg_expr -> ternary
    | arg_or INDENT "?" arg_ws? arg_expr arg_ws? ":" arg_ws? arg_expr DEDENT -> ternary
?def_param_simple: literal
    | path_lit
    | name_ref
?arg_if: "if" arg_ws? arg_expr arg_ws? "then" arg_ws? arg_expr arg_ws? "else" arg_ws? arg_expr -> if_expr
    | "if" arg_ws? arg_expr arg_ws? "then" INDENT arg_expr DEDENT arg_ws? "else" INDENT arg_expr DEDENT -> if_expr
    | arg_or
?arg_or: arg_and
    | arg_or "or" arg_and -> or_expr
?arg_and: arg_cmp
    | arg_and "and" arg_cmp -> and_expr
?arg_cmp: arg_add
    | arg_cmp CMP_OP arg_add -> cmp_expr
?arg_add: arg_mul
    | arg_add ADD_OP arg_mul -> add_expr
?arg_mul: atom_no_tuple
    | arg_mul MUL_OP atom_no_tuple -> mul_expr

?tuple_value: expr "," expr ("," expr)* ","? -> tuple_value
    | expr

lambda_expr: "\\" NAME LAMBDA_ARROW expr -> lambda_expr

?atom: tuple_expr
    | LPAR _NL* expr _NL* RPAR -> paren
    | list_expr
    | literal
    | name_ref

?atom_no_tuple: LPAR _NL* expr _NL* RPAR -> paren
    | list_expr
    | literal
    | name_ref

list_expr: LSQB [expr ("," expr)* ","?] RSQB

tuple_expr: LPAR expr "," expr ("," expr)* ","? RPAR
name_ref: callable -> name
callable: NAME

?literal: INT -> lit_int
    | FLOAT -> lit_float
    | TRUE -> lit_true
    | FALSE -> lit_false
    | NULL -> lit_null
    | STRING -> lit_string

arg_ws: (_NL | INDENT | DEDENT)*
nl_gap: (_NL | INDENT | DEDENT)+

NAME: /[A-Za-z_](?:[A-Za-z0-9_:@]|\.(?!\.))*(?=[ \t\r\n]|$|[),;?:|+\-*\/%<>=\[\]]|\.\.)/
INT: /-?[0-9]+/
FLOAT: /-?(?:[0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?|[0-9]+(?:[eE][+-]?[0-9]+))/
STRING: /"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'/
PATH_LIT: /@@?(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+)(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+))*/

CMP_OP: "==" | "!=" | "<=" | ">=" | "<" | ">"
ADD_OP: "+" | "-"
MUL_OP: "*" | "/" | "%"
TYPE_SEP.3: "::"
BIND_ARROW.3: "<-"
LAMBDA_ARROW.3: "->"
PIPE_OP.3: "|>"
MONAD_BIND.3: ">>="
RANGE_DOTS.3: ".."
IMPORT.3: "import"
EXPORT.3: "export"
TYPE_KW.3: "type"
FOR.3: "for"
STEP.3: "step"
DO.3: "do"
SCOPE_KW.3: "scope"
RETURN.3: "return"
TRUE.3: "true"
FALSE.3: "false"
NULL.3: "null"

LPAR: "("
RPAR: ")"
LSQB: "["
RSQB: "]"

COMMENT: /--[^\n]*/
_NL: /(\r?\n[ \t]*)+/

%declare INDENT DEDENT
%import common.WS_INLINE
%ignore WS_INLINE
%ignore COMMENT
"""


_PARSER = Lark(
    _GRAMMAR,
    parser="lalr",
    postlex=_AxonIndenter(),
    start=["program", "expr"],
)
_ZERO_ARG_CALLS = {"list_init", "init", "Cache.init", "List.init", "_list_init"}


def _plus_one(expr: AxonExpr) -> AxonExpr:
    return AxonExprBinary(op="+", left=AxonExprParen(inner=expr), right=AxonExprInt(value=1))


class _ProgramTransformer(Transformer[Token, object]):
    @staticmethod
    def _is_expr(value: object) -> bool:
        return isinstance(
            value,
            (
                AxonExprName,
                AxonExprInt,
                AxonExprFloat,
                AxonExprBool,
                AxonExprNull,
                AxonExprString,
                AxonExprList,
                AxonExprTuple,
                AxonExprCall,
                AxonExprPipe,
                AxonExprBind,
                AxonExprIf,
                AxonExprTernary,
                AxonExprBinary,
                AxonExprLambda,
                AxonExprParen,
                AxonExprDo,
            ),
        )

    @classmethod
    def _as_expr(cls, value: object) -> AxonExpr:
        if not cls._is_expr(value):
            raise ValueError("expected expression node")
        return cast(AxonExpr, value)

    @classmethod
    def _pragma_literal_to_python(cls, value: object) -> object:
        expr = cls._as_expr(value)
        if isinstance(expr, AxonExprString):
            return expr.value
        if isinstance(expr, AxonExprInt):
            return expr.value
        if isinstance(expr, AxonExprFloat):
            return expr.value
        if isinstance(expr, AxonExprBool):
            return expr.value
        if isinstance(expr, AxonExprNull):
            return None
        if isinstance(expr, AxonExprList):
            return [cls._pragma_literal_to_python(item) for item in expr.items]
        if isinstance(expr, AxonExprTuple):
            return tuple(cls._pragma_literal_to_python(item) for item in expr.items)
        raise ValueError("pragma values must be literals, lists, or tuples of literals")

    @staticmethod
    def _normalize_pragma_value(name: str, value: object) -> object:
        if name == "padding_side":
            side = str(value).strip().lower()
            if side not in {"left", "right"}:
                raise ValueError("PADDING_SIDE must be 'left' or 'right'")
            return side
        if name == "tokenizer":
            if isinstance(value, str) and value:
                return value
            if isinstance(value, list | tuple) and len(value) == 2:
                checkpoint, tokenizer = value
                if (
                    isinstance(checkpoint, str)
                    and checkpoint
                    and isinstance(tokenizer, str)
                    and tokenizer
                ):
                    return (checkpoint, tokenizer)
            raise ValueError(
                "TOKENIZER must be a non-empty string or a [checkpoint, tokenizer] pair"
            )
        if name == "checkpoints":
            if isinstance(value, str):
                return (value,)
            if isinstance(value, list | tuple):
                items = tuple(str(item) for item in value)
                if not all(isinstance(item, str) and item for item in items):
                    raise ValueError("CHECKPOINTS entries must be strings")
                return items
            raise ValueError("CHECKPOINTS must be a string or a list/tuple of strings")
        return value

    @staticmethod
    def _merge_tokenizer_pragma(prev_value: object | None, pragma_value: object) -> object:
        def _entries(value: object | None) -> list[object]:
            if value is None:
                return []
            if isinstance(value, str):
                return [value]
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and all(isinstance(item, str) for item in value)
            ):
                return [value]
            if isinstance(value, list | tuple):
                return list(value)
            raise ValueError("invalid TOKENIZER pragma state")

        entries = _entries(prev_value)
        new_entry = pragma_value
        if isinstance(new_entry, str):
            for entry in entries:
                if isinstance(entry, str):
                    if entry != new_entry:
                        raise ValueError(
                            "conflicting TOKENIZER pragmas; expected a single consistent global tokenizer"
                        )
                    return prev_value if prev_value is not None else new_entry
            entries.insert(0, new_entry)
        else:
            checkpoint, tokenizer = cast(tuple[str, str], new_entry)
            for idx, entry in enumerate(entries):
                if (
                    isinstance(entry, tuple)
                    and len(entry) == 2
                    and all(isinstance(item, str) for item in entry)
                    and entry[0] == checkpoint
                ):
                    if entry[1] != tokenizer:
                        raise ValueError(
                            "conflicting TOKENIZER pragmas; expected a single tokenizer per checkpoint"
                        )
                    return prev_value if prev_value is not None else new_entry
            entries.append(new_entry)

        if len(entries) == 1:
            return entries[0]
        return tuple(entries)

    @staticmethod
    def _is_stmt(value: object) -> bool:
        return isinstance(value, AxonBind | AxonReturn | AxonRepeat | AxonScopeBind)

    @classmethod
    def _as_stmt(cls, value: object) -> AxonStatement:
        if not cls._is_stmt(value):
            raise ValueError("expected statement node")
        return cast(AxonStatement, value)

    @staticmethod
    def _is_type(value: object) -> bool:
        return isinstance(
            value,
            (
                TypeAny,
                TypeInt,
                TypeFloat,
                TypeBool,
                TypeNull,
                TypeString,
                TypeNamed,
                TypeOptional,
                TypeTensor,
                TypeList,
                TypeTuple,
            ),
        )

    @classmethod
    def _as_type(cls, value: object) -> TypeExpr:
        if not cls._is_type(value):
            raise ValueError("expected type expression node")
        return cast(TypeExpr, value)

    def program(self, children: list[object]) -> ParsedProgramSource:
        modules: list[ParsedModuleSource] = []
        imports: list[str] = []
        imported_members: dict[str, tuple[str, ...]] = {}
        exports: list[str] = []
        pragmas: dict[str, object] = {}
        constants: dict[str, AxonExpr] = {}
        type_aliases: dict[str, TypeExpr] = {}
        for child in children:
            if isinstance(child, ParsedModuleSource):
                modules.append(child)
                continue
            if isinstance(child, tuple) and len(child) == 2 and child[0] == "import":
                namespace = cast(str, child[1])
                imports.append(namespace)
                continue
            if isinstance(child, tuple) and len(child) == 3 and child[0] == "import_members":
                namespace = cast(str, child[1])
                members = cast(tuple[str, ...], child[2])
                imports.append(namespace)
                prev_members = imported_members.get(namespace, ())
                imported_members[namespace] = tuple(dict.fromkeys([*prev_members, *members]))
                continue
            if isinstance(child, tuple) and len(child) == 2 and child[0] == "imports":
                import_items = cast(
                    tuple[tuple[str, str] | tuple[str, str, tuple[str, ...]], ...],
                    child[1],
                )
                for item in import_items:
                    if len(item) == 2 and item[0] == "import":
                        namespace = item[1]
                        imports.append(namespace)
                        continue
                    if len(item) == 3 and item[0] == "import_members":
                        namespace = item[1]
                        members = item[2]
                        imports.append(namespace)
                        prev_members = imported_members.get(namespace, ())
                        imported_members[namespace] = tuple(
                            dict.fromkeys([*prev_members, *members])
                        )
                continue
            if isinstance(child, tuple) and len(child) == 2 and child[0] == "export":
                exported = cast(tuple[str, ...], child[1])
                exports.extend(exported)
                continue
            if isinstance(child, tuple) and len(child) == 3 and child[0] == "pragma":
                pragma_name = cast(str, child[1])
                pragma_value = child[2]
                prev_value = pragmas.get(pragma_name)
                if pragma_name == "tokenizer":
                    pragmas[pragma_name] = self._merge_tokenizer_pragma(prev_value, pragma_value)
                    continue
                if (
                    pragma_name in {"padding_side", "tokenizer"}
                    and isinstance(prev_value, str)
                    and prev_value != pragma_value
                ):
                    raise ValueError(
                        f"conflicting {pragma_name.upper()} pragmas; expected a single consistent value"
                    )
                pragmas[pragma_name] = pragma_value
                continue
            if isinstance(child, tuple) and len(child) == 3 and child[0] == "constant":
                name = cast(str, child[1])
                constant_value = cast(AxonExpr, child[2])
                constants[name] = constant_value
                continue
            if isinstance(child, tuple) and len(child) == 3 and child[0] == "type_alias":
                name = cast(str, child[1])
                type_value = cast(TypeExpr, child[2])
                type_aliases[name] = type_value
        return ParsedProgramSource(
            modules=tuple(modules),
            imports=tuple(dict.fromkeys(imports)),
            imported_members=imported_members,
            exports=tuple(dict.fromkeys(exports)),
            pragmas=pragmas,
            constants=constants,
            type_aliases=type_aliases,
        )

    def top_item(self, children: list[object]) -> object:
        return children[0]

    def module_decl(self, children: list[object]) -> ParsedModuleSource:
        signature: ParsedSignature | None = None
        definition: ParsedDefinition | None = None
        for child in children:
            if isinstance(child, ParsedSignature):
                signature = child
            elif isinstance(child, ParsedDefinition):
                definition = child
        if signature is None or definition is None:
            raise ValueError("invalid module declaration syntax")
        return ParsedModuleSource(
            signature=signature,
            definition=definition,
        )

    def mod_decl(self, children: list[object]) -> str:
        if not children:
            raise ValueError("module declaration cannot be empty")
        base = cast(str, children[0])
        if len(children) == 1:
            return base
        suffix = "".join(cast(str, child) for child in children[1:])
        return f"{base}{suffix}"

    def mod_decl_param(self, children: list[object]) -> str:
        token = cast(Token, children[0])
        return f"@{token}"

    def module_name(self, children: list[object]) -> str:
        parts = [str(child) for child in children if isinstance(child, Token)]
        if not parts:
            raise ValueError("module name cannot be empty")
        return ".".join(parts)

    def scoped_name(self, children: list[object]) -> str:
        parts = [str(child) for child in children if isinstance(child, Token)]
        if not parts:
            raise ValueError("scope name cannot be empty")
        return ".".join(parts)

    def type_name(self, children: list[object]) -> str:
        token = children[0]
        assert isinstance(token, Token)
        return str(token)

    def type_named(self, children: list[object]) -> TypeExpr:
        raw = cast(str, children[0]).strip()
        if raw == "Any":
            return TypeAny()
        if raw == "Int":
            return TypeInt()
        if raw in {"F", "Float"}:
            return TypeFloat()
        if raw == "Bool":
            return TypeBool()
        if raw == "Null":
            return TypeNull()
        if raw in {"Str", "String"}:
            return TypeString()
        return TypeNamed(name=raw)

    def type_optional(self, children: list[object]) -> TypeExpr:
        inner = self._as_type(children[-1])
        return TypeOptional(inner=inner)

    def type_tuple(self, children: list[object]) -> TypeExpr:
        items = tuple(self._as_type(child) for child in children if self._is_type(child))
        return TypeTuple(items=items)

    def type_list(self, children: list[object]) -> TypeExpr:
        items = [self._as_type(child) for child in children if self._is_type(child)]
        if len(items) != 1:
            raise ValueError("list type requires exactly one item type")
        return TypeList(item=items[0])

    def type_dim_int(self, children: list[object]) -> DimToken:
        token = children[0]
        assert isinstance(token, Token)
        return int(str(token))

    def type_dim_name(self, children: list[object]) -> DimToken:
        return cast(str, children[0])

    def type_dim_rest(self, children: list[object]) -> DimToken:
        if len(children) != 2:
            raise ValueError("invalid variadic tensor dimension token")
        name = cast(str, children[1]).strip()
        if not name:
            raise ValueError("variadic tensor dimension requires a name")
        return f"..{name}"

    def type_dim_paren(self, children: list[object]) -> DimToken:
        token = children[0]
        assert isinstance(token, int | str | DimExprBinary)
        return token

    def type_dim_binary(self, children: list[object]) -> DimToken:
        left = children[0]
        op_token = children[1]
        right = children[2]
        assert isinstance(left, int | str | DimExprBinary)
        assert isinstance(right, int | str | DimExprBinary)
        assert isinstance(op_token, Token)
        return DimExprBinary(op=str(op_token), left=left, right=right)

    def type_tensor(self, children: list[object]) -> TypeExpr:
        if not children:
            raise ValueError("tensor type requires a base name")
        base = cast(str, children[0]).strip()
        dims: list[DimToken] = []
        for child in children[1:]:
            if isinstance(child, Token):
                continue
            if isinstance(child, int | str | DimExprBinary):
                dims.append(child)
        return TypeTensor(base=base, dims=tuple(dims))

    def path_type_named(self, children: list[object]) -> ParsedPathTypeParam:
        if len(children) != 2:
            raise ValueError("invalid named path type annotation")
        name = cast(Token, children[0])
        type_expr = self._as_type(children[1])
        return ParsedPathTypeParam(name=str(name), type_expr=type_expr)

    def path_type_implicit(self, children: list[object]) -> ParsedPathTypeParam:
        if len(children) != 1:
            raise ValueError("invalid implicit path type annotation")
        type_expr = self._as_type(children[0])
        return ParsedPathTypeParam(name=None, type_expr=type_expr)

    def signature_type(self, children: list[object]) -> ParsedFunctionType:
        segments: list[ParsedPathTypeParam | TypeExpr] = []
        for child in children:
            if isinstance(child, ParsedPathTypeParam):
                segments.append(child)
                continue
            if self._is_type(child):
                segments.append(self._as_type(child))
        if not segments:
            raise ValueError("empty signature type")
        if isinstance(segments[-1], ParsedPathTypeParam):
            raise ValueError("signature return type cannot be a path annotation")
        path_params: list[ParsedPathTypeParam] = []
        args_and_return: list[TypeExpr] = []
        in_args = False
        for segment in segments:
            if isinstance(segment, ParsedPathTypeParam):
                if in_args:
                    raise ValueError("path annotations must precede value arguments in signature")
                path_params.append(segment)
                continue
            in_args = True
            args_and_return.append(segment)
        if not args_and_return:
            raise ValueError("signature requires a return type")
        return ParsedFunctionType(
            path_params=tuple(path_params),
            arg_types=tuple(args_and_return[:-1]),
            return_type=args_and_return[-1],
        )

    def signature(self, children: list[object]) -> ParsedSignature:
        mod = cast(str, children[0])
        signature_type = next(
            (child for child in children[1:] if isinstance(child, ParsedFunctionType)),
            None,
        )
        if not isinstance(signature_type, ParsedFunctionType):
            raise ValueError("signature type expression is required")
        return ParsedSignature(module_decl=mod, type_signature=signature_type)

    def definition(self, children: list[object]) -> ParsedDefinition:
        mod = cast(str, children[0])
        args: list[ParsedDefParam] = []
        rhs: AxonExpr | None = None
        for child in children[1:]:
            if isinstance(child, ParsedDefParam):
                args.append(child)
                continue
            if self._is_expr(child):
                rhs = self._as_expr(child)
        if rhs is None:
            raise ValueError("definition rhs expression is required")
        return ParsedDefinition(module_decl=mod, args=tuple(args), rhs=rhs)

    def def_param_positional(self, children: list[object]) -> ParsedDefParam:
        if len(children) != 1:
            raise ValueError("invalid positional definition parameter")
        token = cast(Token, children[0])
        return ParsedDefParam(name=str(token), default_expr=None)

    def def_param_default(self, children: list[object]) -> ParsedDefParam:
        if len(children) != 2:
            raise ValueError("invalid defaulted definition parameter")
        token = cast(Token, children[0])
        default_expr = self._as_expr(children[1])
        return ParsedDefParam(name=str(token), default_expr=default_expr)

    def def_param_default_paren(self, children: list[object]) -> ParsedDefParam:
        token = next(
            (child for child in children if isinstance(child, Token) and child.type == "NAME"),
            None,
        )
        default_expr = next(
            (self._as_expr(child) for child in children if self._is_expr(child)), None
        )
        if not isinstance(token, Token) or default_expr is None:
            raise ValueError("invalid parenthesized defaulted definition parameter")
        return ParsedDefParam(name=str(token), default_expr=default_expr)

    def import_members_paren(self, children: list[object]) -> tuple[str, ...]:
        return tuple(
            str(child) for child in children if isinstance(child, Token) and child.type == "NAME"
        )

    def import_members_bare(self, children: list[object]) -> tuple[str, ...]:
        return tuple(
            str(child) for child in children if isinstance(child, Token) and child.type == "NAME"
        )

    def import_decl(self, children: list[object]) -> tuple[str, tuple[object, ...]]:
        values = [child for child in children if not isinstance(child, Token)]
        specs = cast(tuple[tuple[str, tuple[str, ...] | None], ...], tuple(values))
        out: list[tuple[str, str] | tuple[str, str, tuple[str, ...]]] = []
        for namespace, members in specs:
            if members is None:
                out.append(("import", namespace))
            else:
                out.append(("import_members", namespace, members))
        return ("imports", tuple(out))

    def import_item(self, children: list[object]) -> tuple[str, tuple[str, ...] | None]:
        values = [child for child in children if not isinstance(child, Token)]
        namespace = cast(str, values[0])
        members: tuple[str, ...] | None = None
        if len(values) > 1 and isinstance(values[1], tuple):
            members = cast(tuple[str, ...], values[1])
        return (namespace, members)

    def export_members_paren(self, children: list[object]) -> tuple[str, ...]:
        return tuple(
            str(child) for child in children if isinstance(child, Token) and child.type == "NAME"
        )

    def export_members_bare(self, children: list[object]) -> tuple[str, ...]:
        return tuple(
            str(child) for child in children if isinstance(child, Token) and child.type == "NAME"
        )

    def export_decl(self, children: list[object]) -> tuple[str, tuple[str, ...]]:
        values = [child for child in children if not isinstance(child, Token)]
        members = cast(tuple[str, ...], values[0]) if values else ()
        return ("export", members)

    def pragma(self, children: list[object]) -> tuple[str, str, object]:
        if len(children) != 2:
            raise ValueError("invalid pragma syntax")
        name_token = children[0]
        assert isinstance(name_token, Token)
        pragma_name = str(name_token).strip().lower()
        pragma_value = self._normalize_pragma_value(
            pragma_name,
            self._pragma_literal_to_python(children[1]),
        )
        return ("pragma", pragma_name, pragma_value)

    def constant(self, children: list[object]) -> tuple[str, str, AxonExpr]:
        name = str(cast(Token, children[0]))
        value = self._as_expr(children[1])
        return ("constant", name, value)

    def type_alias_decl(self, children: list[object]) -> tuple[str, str, TypeExpr]:
        name_token = next(
            (child for child in children if isinstance(child, Token) and child.type == "NAME"),
            None,
        )
        alias_type = next(
            (self._as_type(child) for child in children if self._is_type(child)), None
        )
        if not isinstance(name_token, Token) or alias_type is None:
            raise ValueError("invalid type alias syntax")
        name = str(name_token)
        return ("type_alias", name, alias_type)

    def target_list(self, children: list[object]) -> tuple[str, ...]:
        out = [str(child) for child in children if isinstance(child, Token)]
        return tuple(out)

    def expr_list(self, children: list[object]) -> tuple[AxonExpr, ...]:
        return tuple(self._as_expr(child) for child in children if self._is_expr(child))

    def statement_line(self, children: list[object]) -> tuple[AxonStatement, ...]:
        return tuple(self._as_stmt(child) for child in children if self._is_stmt(child))

    def block_suite(self, children: list[object]) -> tuple[AxonStatement, ...]:
        out: list[AxonStatement] = []
        for child in children:
            if isinstance(child, tuple):
                out.extend(cast(tuple[AxonStatement, ...], child))
        return tuple(out)

    def inline_statement(self, children: list[object]) -> AxonStatement:
        return self._as_stmt(children[0])

    def inline_suite(self, children: list[object]) -> tuple[AxonStatement, ...]:
        return tuple(self._as_stmt(child) for child in children if self._is_stmt(child))

    def suite_inline_maybe_block(self, children: list[object]) -> _SuiteBody:
        inline = cast(tuple[AxonStatement, ...], children[0])
        if len(children) == 1:
            return _SuiteBody(body=inline, inline=True)
        block = cast(tuple[AxonStatement, ...], children[1])
        return _SuiteBody(body=(*inline, *block), inline=True)

    def suite_block(self, children: list[object]) -> _SuiteBody:
        return _SuiteBody(body=cast(tuple[AxonStatement, ...], children[0]), inline=False)

    def do_expr(self, children: list[object]) -> AxonExprDo:
        suite = next(
            (child for child in children if isinstance(child, _SuiteBody)),
            None,
        )
        if isinstance(suite, _SuiteBody):
            return AxonExprDo(body=suite.body, inline=suite.inline)
        body = next(
            (
                cast(tuple[AxonStatement, ...], child)
                for child in children
                if isinstance(child, tuple)
            ),
            (),
        )
        return AxonExprDo(body=body, inline=False)

    def return_statement(self, children: list[object]) -> AxonReturn:
        values = next(
            (
                cast(tuple[AxonExpr, ...], child)
                for child in children
                if isinstance(child, tuple) and all(self._is_expr(item) for item in child)
            ),
            (),
        )
        return AxonReturn(values=values)

    def bind_statement(self, children: list[object]) -> AxonBind:
        targets = cast(tuple[str, ...], children[0])
        expr = next((self._as_expr(child) for child in children[1:] if self._is_expr(child)), None)
        if expr is None:
            raise ValueError("binding expression is required")
        return AxonBind(targets=targets, expr=expr)

    def for_scope(self, children: list[object]) -> str:
        return cast(str, children[0])

    def scope_ref(self, children: list[object]) -> str:
        if not children:
            raise ValueError("scope reference cannot be empty")
        scoped = cast(str, children[-1])
        if len(children) > 1:
            return f"@{scoped}"
        return scoped

    def for_step(self, children: list[object]) -> AxonExpr:
        for child in reversed(children):
            if self._is_expr(child):
                return self._as_expr(child)
        raise ValueError("for-step expression is required")

    def range_expr(self, children: list[object]) -> tuple[str, AxonExpr, AxonExpr, str]:
        start_token: Token | None = None
        end_token: Token | None = None
        exprs: list[AxonExpr] = []
        for child in children:
            if isinstance(child, Token):
                if child.type in {"LSQB", "LPAR"} and start_token is None:
                    start_token = child
                elif child.type in {"RSQB", "RPAR"}:
                    end_token = child
                continue
            if self._is_expr(child):
                exprs.append(self._as_expr(child))
        if start_token is None or end_token is None or len(exprs) != 2:
            raise ValueError("invalid range expression")
        start = str(start_token)
        start_expr = exprs[0]
        end_expr = exprs[1]
        end = str(end_token)
        return (start, start_expr, end_expr, end)

    def for_statement(self, children: list[object]) -> AxonRepeat:
        scope_name: str | None = None
        var: str | None = None
        range_value: tuple[str, AxonExpr, AxonExpr, str] | None = None
        step_expr: AxonExpr = AxonExprInt(value=1)
        body: tuple[AxonStatement, ...] = ()

        for child in children:
            if isinstance(child, Token):
                if child.type == "NAME" and var is None:
                    var = str(child)
                continue
            if isinstance(child, str):
                if scope_name is None:
                    scope_name = child
                continue
            if isinstance(child, _SuiteBody):
                body = child.body
                continue
            if isinstance(child, tuple):
                if (
                    len(child) == 4
                    and isinstance(child[0], str)
                    and self._is_expr(child[1])
                    and self._is_expr(child[2])
                    and isinstance(child[3], str)
                ):
                    range_value = cast(tuple[str, AxonExpr, AxonExpr, str], child)
                    continue
                if all(self._is_stmt(item) for item in child):
                    body = cast(tuple[AxonStatement, ...], child)
                    continue
            if self._is_expr(child):
                step_expr = self._as_expr(child)

        if var is None or range_value is None:
            raise ValueError("invalid for-statement syntax")
        start_delim, start_expr, end_expr, end_delim = range_value

        from_expr = start_expr if start_delim == "[" else _plus_one(start_expr)
        to_expr = _plus_one(end_expr) if end_delim == "]" else end_expr
        return AxonRepeat(
            name=scope_name,
            var=var,
            to_expr=to_expr,
            from_expr=from_expr,
            step_expr=step_expr,
            body=body,
        )

    def scope_bind_statement(self, children: list[object]) -> AxonScopeBind:
        targets = cast(tuple[str, ...], children[0])
        prefix: str | None = None
        kwargs: dict[str, AxonKwargValue] = {}
        body: tuple[AxonStatement, ...] = ()
        for child in children[1:]:
            if isinstance(child, Token):
                continue
            if isinstance(child, str) and prefix is None:
                prefix = child
                continue
            if isinstance(child, tuple) and len(child) == 2 and isinstance(child[0], str):
                kwargs[child[0]] = child[1]
                continue
            if isinstance(child, _SuiteBody):
                body = child.body
                continue
            if isinstance(child, tuple) and all(self._is_stmt(item) for item in child):
                body = cast(tuple[AxonStatement, ...], child)
        if prefix is None:
            raise ValueError("scope bind prefix is required")
        return AxonScopeBind(targets=targets, prefix=prefix, body=body, kwargs=kwargs)

    def scope_bind_kwarg(self, children: list[object]) -> tuple[str, AxonKwargValue]:
        return self.kwarg_item(children)

    def range_start(self, children: list[object]) -> Token:
        return cast(Token, children[0])

    def range_end(self, children: list[object]) -> Token:
        return cast(Token, children[0])

    def nl_gap(self, _: list[object]) -> None:
        return None

    def __default__(self, data: str, children: list[object], meta: object) -> object:
        if len(children) == 1:
            return children[0]
        return children

    def name(self, children: list[object]) -> AxonExpr:
        token = children[0]
        assert isinstance(token, Token)
        text = str(token)
        if "@" in text or "." in text or "::" in text or text in _ZERO_ARG_CALLS:
            return AxonExprCall(callee=text, args=(), kwargs={})
        return AxonExprName(name=text)

    def callable(self, children: list[object]) -> object:
        return children[0]

    def lit_int(self, children: list[object]) -> AxonExpr:
        token = children[0]
        assert isinstance(token, Token)
        return AxonExprInt(value=int(str(token)))

    def lit_float(self, children: list[object]) -> AxonExpr:
        token = children[0]
        assert isinstance(token, Token)
        text = str(token)
        return AxonExprFloat(value=float(text), lexeme=text)

    def lit_true(self, _: list[object]) -> AxonExpr:
        return AxonExprBool(value=True)

    def lit_false(self, _: list[object]) -> AxonExpr:
        return AxonExprBool(value=False)

    def lit_null(self, _: list[object]) -> AxonExpr:
        return AxonExprNull()

    def lit_string(self, children: list[object]) -> AxonExpr:
        token = children[0]
        assert isinstance(token, Token)
        text = str(token)
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            return AxonExprString(value=text[1:-1])
        return AxonExprString(value=text)

    def path_lit(self, children: list[object]) -> AxonExpr:
        token = children[0]
        assert isinstance(token, Token)
        return AxonExprString(value=str(token))

    def tuple_expr(self, children: list[object]) -> AxonExpr:
        items = tuple(self._as_expr(child) for child in children if self._is_expr(child))
        return AxonExprTuple(items=items)

    def paren(self, children: list[object]) -> AxonExpr:
        inner = next((self._as_expr(child) for child in children if self._is_expr(child)), None)
        if inner is None:
            raise ValueError("parenthesized expression requires an inner expression")
        return AxonExprParen(inner=inner)

    def bare_call(self, children: list[object]) -> AxonExpr:
        callee_token = children[0]
        assert isinstance(callee_token, Token)
        args: list[AxonExpr] = []
        kwargs: dict[str, AxonKwargValue] = {}
        for child in children[1:]:
            if isinstance(child, tuple) and len(child) == 2 and isinstance(child[0], str):
                kwargs[child[0]] = child[1]
            elif self._is_expr(child):
                args.append(self._as_expr(child))
        return AxonExprCall(callee=str(callee_token), args=tuple(args), kwargs=kwargs)

    def kwarg_item(self, children: list[object]) -> tuple[str, AxonKwargValue]:
        key_token = children[0]
        value = children[1]
        assert isinstance(key_token, Token)
        if self._is_expr(value):
            return str(key_token), self._as_expr(value)
        if isinstance(value, list):
            return str(key_token), value
        raise ValueError("unsupported kwarg expression")

    def kwarg_bare(self, children: list[object]) -> tuple[str, AxonKwargValue]:
        return self.kwarg_item(children)

    def bare_arg(self, children: list[object]) -> object:
        return children[0]

    def bind_once(self, children: list[object]) -> AxonExpr:
        value = self._as_expr(children[0])
        lam = next((child for child in children[1:] if isinstance(child, AxonExprLambda)), None)
        assert isinstance(lam, AxonExprLambda)
        return AxonExprBind(value=value, var=lam.var, body=lam.body)

    def lambda_expr(self, children: list[object]) -> AxonExpr:
        var_token = next(
            (child for child in children if isinstance(child, Token) and child.type == "NAME"),
            None,
        )
        body = next((self._as_expr(child) for child in children if self._is_expr(child)), None)
        if var_token is None or body is None:
            raise ValueError("invalid lambda expression")
        assert isinstance(var_token, Token)
        return AxonExprLambda(var=str(var_token), body=body)

    def list_expr(self, children: list[object]) -> AxonExpr:
        items = tuple(self._as_expr(child) for child in children if self._is_expr(child))
        return AxonExprList(items=items)

    def pipe(self, children: list[object]) -> AxonExpr:
        left = self._as_expr(children[0])
        right = next((self._as_expr(child) for child in children[1:] if self._is_expr(child)), None)
        if right is None:
            raise ValueError("pipe stage expression is required")
        if isinstance(left, AxonExprPipe):
            return AxonExprPipe(value=left.value, stages=(*left.stages, right))
        return AxonExprPipe(value=left, stages=(right,))

    def ternary(self, children: list[object]) -> AxonExpr:
        exprs = [self._as_expr(child) for child in children if self._is_expr(child)]
        if len(exprs) != 3:
            raise ValueError("ternary expression requires condition and two branches")
        cond, true_expr, false_expr = exprs
        return AxonExprTernary(cond=cond, true_expr=true_expr, false_expr=false_expr)

    def if_expr(self, children: list[object]) -> AxonExpr:
        exprs = [self._as_expr(child) for child in children if self._is_expr(child)]
        if len(exprs) != 3:
            raise ValueError("if-expression requires condition and two branches")
        cond, true_expr, false_expr = exprs
        return AxonExprIf(cond=cond, true_expr=true_expr, false_expr=false_expr)

    def or_expr(self, children: list[object]) -> AxonExpr:
        left = self._as_expr(children[0])
        right = self._as_expr(children[1])
        return AxonExprBinary(op="or", left=left, right=right)

    def and_expr(self, children: list[object]) -> AxonExpr:
        left = self._as_expr(children[0])
        right = self._as_expr(children[1])
        return AxonExprBinary(op="and", left=left, right=right)

    def cmp_expr(self, children: list[object]) -> AxonExpr:
        left = self._as_expr(children[0])
        op_token = children[1]
        right = self._as_expr(children[2])
        assert isinstance(op_token, Token)
        return AxonExprBinary(op=str(op_token), left=left, right=right)

    def add_expr(self, children: list[object]) -> AxonExpr:
        left = self._as_expr(children[0])
        op_token = children[1]
        right = self._as_expr(children[2])
        assert isinstance(op_token, Token)
        return AxonExprBinary(op=str(op_token), left=left, right=right)

    def mul_expr(self, children: list[object]) -> AxonExpr:
        left = self._as_expr(children[0])
        op_token = children[1]
        right = self._as_expr(children[2])
        assert isinstance(op_token, Token)
        return AxonExprBinary(op=str(op_token), left=left, right=right)

    def tuple_value(self, children: list[object]) -> AxonExpr:
        exprs = [self._as_expr(child) for child in children if self._is_expr(child)]
        if len(exprs) == 1:
            return exprs[0]
        return AxonExprTuple(items=tuple(exprs))


def parse_program_source(source: str) -> ParsedProgramSource:
    try:
        tree = _PARSER.parse(source, start="program")
    except LarkError as exc:
        raise ValueError("invalid Axon source syntax") from exc
    try:
        transformed = _ProgramTransformer().transform(tree)
    except VisitError as exc:
        if isinstance(exc.orig_exc, ValueError):
            raise exc.orig_exc
        raise
    if not isinstance(transformed, ParsedProgramSource):
        raise ValueError("invalid Axon source syntax")
    return transformed


def parse_expression_source(source: str) -> AxonExpr:
    try:
        tree = _PARSER.parse(source, start="expr")
    except LarkError as exc:
        raise ValueError("invalid Axon expression syntax") from exc
    try:
        transformed = _ProgramTransformer().transform(tree)
    except VisitError as exc:
        if isinstance(exc.orig_exc, ValueError):
            raise exc.orig_exc
        raise
    if not _ProgramTransformer._is_expr(transformed):
        raise ValueError("invalid Axon expression syntax")
    return cast(AxonExpr, transformed)


__all__ = [
    "ParsedDefinition",
    "ParsedFunctionType",
    "ParsedModuleSource",
    "ParsedPathTypeParam",
    "ParsedProgramSource",
    "ParsedSignature",
    "parse_expression_source",
    "parse_program_source",
]
