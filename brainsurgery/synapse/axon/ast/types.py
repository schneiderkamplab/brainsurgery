from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class TypingRule:
    name: str
    judgment: str
    premises: tuple[str, ...]
    conclusion: str
    notes: str


@dataclass(frozen=True)
class TypeAny:
    pass


@dataclass(frozen=True)
class TypeInt:
    pass


@dataclass(frozen=True)
class TypeFloat:
    pass


@dataclass(frozen=True)
class TypeBool:
    pass


@dataclass(frozen=True)
class TypeNull:
    pass


@dataclass(frozen=True)
class TypeString:
    pass


@dataclass(frozen=True)
class TypePath:
    pass


@dataclass(frozen=True)
class TypeDim:
    pass


@dataclass(frozen=True)
class TypeVar:
    name: str


@dataclass(frozen=True)
class TypeNamed:
    name: str
    args: tuple["DimToken", ...] = ()


@dataclass(frozen=True)
class TypeOptional:
    inner: "TypeExpr"


@dataclass(frozen=True)
class DimExprBinary:
    op: str
    left: "DimToken"
    right: "DimToken"


DimToken: TypeAlias = int | str | DimExprBinary


@dataclass(frozen=True)
class TypeTensor:
    base: str
    dims: tuple[DimToken, ...]


@dataclass(frozen=True)
class TypeList:
    item: "TypeExpr"


@dataclass(frozen=True)
class TypeTuple:
    items: tuple["TypeExpr", ...]


@dataclass(frozen=True)
class TypeAliasDef:
    params: tuple[str, ...]
    value: "TypeExpr"


ConstraintAtom: TypeAlias = int | str | bool | None | DimExprBinary
ConstraintOperand: TypeAlias = ConstraintAtom | tuple[ConstraintAtom, ...]


@dataclass(frozen=True)
class Constraint:
    relation: str
    left: ConstraintOperand
    right: ConstraintOperand | None = None
    guards: tuple["Constraint", ...] = ()


TypeExpr: TypeAlias = (
    TypeAny
    | TypeInt
    | TypeFloat
    | TypeBool
    | TypeNull
    | TypeString
    | TypePath
    | TypeDim
    | TypeVar
    | TypeNamed
    | TypeOptional
    | TypeTensor
    | TypeList
    | TypeTuple
)


def _tokenize_dim_expr(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit():
            j = i + 1
            while j < n and text[j].isdigit():
                j += 1
            out.append(("INT", text[i:j]))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            out.append(("NAME", text[i:j]))
            i = j
            continue
        if ch == "." and i + 1 < n and text[i + 1] == ".":
            j = i + 2
            if j >= n or not _is_name_start(text[j]):
                raise ValueError(f"invalid variadic dimension token in {text!r}")
            j += 1
            while j < n and _is_name_char(text[j]):
                j += 1
            out.append(("NAME", text[i:j]))
            i = j
            continue
        if ch in "+-*/":
            out.append(("OP", ch))
            i += 1
            continue
        if ch == "(":
            out.append(("LPAR", ch))
            i += 1
            continue
        if ch == ")":
            out.append(("RPAR", ch))
            i += 1
            continue
        raise ValueError(f"invalid dimension token {ch!r} in {text!r}")
    return out


def parse_dim_expr(text: str) -> DimToken:
    tokens = _tokenize_dim_expr(text.strip())
    if not tokens:
        raise ValueError("empty dimension expression")
    pos = 0

    def _peek() -> tuple[str, str] | None:
        nonlocal pos
        if pos >= len(tokens):
            return None
        return tokens[pos]

    def _take(expected: str | None = None) -> tuple[str, str]:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("unexpected end of dimension expression")
        tok = tokens[pos]
        pos += 1
        if expected is not None and tok[0] != expected:
            raise ValueError(f"expected {expected}, got {tok[0]}")
        return tok

    def _parse_primary() -> DimToken:
        tok = _peek()
        if tok is None:
            raise ValueError("missing dimension operand")
        if tok[0] == "OP" and tok[1] == "-":
            _take("OP")
            inner = _parse_primary()
            return DimExprBinary(op="*", left=-1, right=inner)
        if tok[0] == "INT":
            _, raw = _take("INT")
            return int(raw)
        if tok[0] == "NAME":
            _, raw = _take("NAME")
            return raw
        if tok[0] == "LPAR":
            _take("LPAR")
            inner = _parse_add()
            close = _take("RPAR")
            assert close[0] == "RPAR"
            return inner
        raise ValueError(f"unexpected dimension token {tok!r}")

    def _parse_mul() -> DimToken:
        left = _parse_primary()
        while True:
            tok = _peek()
            if tok is None or tok[0] != "OP" or tok[1] not in {"*", "/"}:
                return left
            _, op = _take("OP")
            right = _parse_primary()
            left = DimExprBinary(op=op, left=left, right=right)

    def _parse_add() -> DimToken:
        left = _parse_mul()
        while True:
            tok = _peek()
            if tok is None or tok[0] != "OP" or tok[1] not in {"+", "-"}:
                return left
            _, op = _take("OP")
            right = _parse_mul()
            left = DimExprBinary(op=op, left=left, right=right)

    parsed = _parse_add()
    if pos != len(tokens):
        raise ValueError(f"trailing dimension tokens in {text!r}")
    return parsed


def _dim_precedence(dim: DimToken) -> int:
    if isinstance(dim, DimExprBinary):
        if dim.op in {"*", "/"}:
            return 2
        if dim.op in {"+", "-"}:
            return 1
    return 3


def render_dim_token(dim: DimToken, *, parent_prec: int = 0) -> str:
    if isinstance(dim, int):
        return str(dim)
    if isinstance(dim, str):
        return dim
    assert isinstance(dim, DimExprBinary)
    prec = _dim_precedence(dim)
    left = render_dim_token(dim.left, parent_prec=prec)
    right = render_dim_token(dim.right, parent_prec=prec)
    text = f"{left} {dim.op} {right}"
    if prec < parent_prec:
        return f"({text})"
    return text


def dim_token_names(dim: DimToken) -> set[str]:
    if isinstance(dim, int):
        return set()
    if isinstance(dim, str):
        return {dim}
    assert isinstance(dim, DimExprBinary)
    return dim_token_names(dim.left) | dim_token_names(dim.right)


TYPING_RULES: tuple[TypingRule, ...] = (
    TypingRule(
        name="T-Var",
        judgment="Gamma |- x : T",
        premises=("Gamma(x) = T",),
        conclusion="Gamma |- x : T",
        notes="Variable lookup in typing environment.",
    ),
    TypingRule(
        name="T-Lit",
        judgment="Gamma |- lit : Tlit",
        premises=("lit is one of int/float/bool/null/string literal",),
        conclusion="Gamma |- lit : Tlit",
        notes="Primitive literal typing.",
    ),
    TypingRule(
        name="T-Optional",
        judgment="Gamma |- e : ?T",
        premises=("Gamma |- e : T  or  Gamma |- e : Null",),
        conclusion="Gamma |- e : Optional(T)",
        notes="Optional type denotes the disjoint union T | Null.",
    ),
    TypingRule(
        name="T-Tuple",
        judgment="Gamma |- (e1, ..., en) : (T1, ..., Tn)",
        premises=("Gamma |- ei : Ti  for all i",),
        conclusion="Gamma |- tuple(e1..en) : Tuple(T1..Tn)",
        notes="Tuple introduction.",
    ),
    TypingRule(
        name="T-Call-User",
        judgment="Gamma |- f(e1..en) : Tr",
        premises=(
            "f : T1 -> ... -> Tn -> Tr in module signature table",
            "Gamma |- ei : Si and Si compatible with Ti for all i",
        ),
        conclusion="Gamma |- f(e1..en) : Tr",
        notes="User-module call typing against declared signatures.",
    ),
    TypingRule(
        name="T-Call-Prim",
        judgment="Gamma |- op(args, kwargs) : Tout",
        premises=(
            "op arity/kwarg constraints hold",
            "kwarg kinds hold",
        ),
        conclusion="Gamma |- op(args, kwargs) : Tout",
        notes="Primitive op typing via lowering signature metadata.",
    ),
    TypingRule(
        name="T-Bind",
        judgment="Gamma |- (x <- e) ; s",
        premises=(
            "Gamma |- e : Te",
            "Gamma, x:Te |- s",
        ),
        conclusion="Gamma |- bind",
        notes="Statement binding extends environment.",
    ),
    TypingRule(
        name="T-Return",
        judgment="Gamma |- return e1..en",
        premises=(
            "Gamma |- ei : Ti for all i",
            "Ti compatible with declared module return types",
        ),
        conclusion="Gamma |- return",
        notes="Return typing against module result type.",
    ),
    TypingRule(
        name="T-If",
        judgment="Gamma |- if c then a else b : T",
        premises=(
            "Gamma |- c : Bool-compatible",
            "Gamma |- a : Ta",
            "Gamma |- b : Tb",
            "Ta and Tb are unifiable to T",
        ),
        conclusion="Gamma |- if ... : T",
        notes="Conditional expression typing.",
    ),
    TypingRule(
        name="T-BinOp",
        judgment="Gamma |- a op b : T",
        premises=("Gamma |- a : Ta", "Gamma |- b : Tb", "op-specific compatibility holds"),
        conclusion="Gamma |- a op b : T",
        notes="Arithmetic/comparison/logical typing.",
    ),
    TypingRule(
        name="T-Shape",
        judgment="Gamma |- Tensor[..., D] ~ Tensor[..., E]",
        premises=("ranks equal", "each dim pair unifies with dim substitution sigma"),
        conclusion="Gamma |- dims compatible",
        notes="Shape constraint solving over dimension tokens.",
    ),
)


def _is_name_start(ch: str) -> bool:
    return ch.isalpha() or ch == "_"


def _is_name_char(ch: str) -> bool:
    return ch.isalnum() or ch in "_."


class _TypeExprParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.n = len(text)
        self.i = 0

    def _skip_ws(self) -> None:
        while self.i < self.n and self.text[self.i].isspace():
            self.i += 1

    def _peek(self) -> str | None:
        self._skip_ws()
        if self.i >= self.n:
            return None
        return self.text[self.i]

    def _consume(self, token: str) -> bool:
        self._skip_ws()
        if self.i < self.n and self.text[self.i] == token:
            self.i += 1
            return True
        return False

    def _expect(self, token: str) -> None:
        if not self._consume(token):
            raise ValueError(f"expected {token!r} in type expression {self.text!r}")

    def _parse_name(self) -> str:
        self._skip_ws()
        if self.i >= self.n or not _is_name_start(self.text[self.i]):
            raise ValueError(f"expected type name in {self.text!r}")
        start = self.i
        self.i += 1
        while self.i < self.n and _is_name_char(self.text[self.i]):
            self.i += 1
        return self.text[start : self.i]

    def _parse_dim_list(self) -> tuple[DimToken, ...]:
        dims: list[DimToken] = []
        while True:
            self._skip_ws()
            start = self.i
            paren_depth = 0
            while self.i < self.n:
                ch = self.text[self.i]
                if ch == "(":
                    paren_depth += 1
                elif ch == ")":
                    paren_depth -= 1
                    if paren_depth < 0:
                        raise ValueError(f"invalid dimension expression in {self.text!r}")
                elif ch == "]" and paren_depth == 0:
                    break
                elif ch == "," and paren_depth == 0:
                    break
                self.i += 1
            raw_dim = self.text[start : self.i].strip()
            if not raw_dim:
                raise ValueError(f"empty tensor dimension in {self.text!r}")
            if raw_dim.startswith(".."):
                if len(raw_dim) <= 2 or not _is_name_start(raw_dim[2]):
                    raise ValueError(f"invalid variadic tensor dimension {raw_dim!r}")
                if not all(_is_name_char(ch) for ch in raw_dim[3:]):
                    raise ValueError(f"invalid variadic tensor dimension {raw_dim!r}")
                dims.append(raw_dim)
            else:
                dims.append(parse_dim_expr(raw_dim))
            self._skip_ws()
            if self._consume(","):
                continue
            break
        return tuple(dims)

    def parse(self) -> TypeExpr:
        self._skip_ws()
        if self.i >= self.n:
            return TypeAny()
        parsed = self._parse_type_expr()
        self._skip_ws()
        if self.i != self.n:
            raise ValueError(f"trailing type tokens in {self.text!r}")
        return parsed

    def _parse_type_expr(self) -> TypeExpr:
        if self._consume("?"):
            return TypeOptional(inner=self._parse_type_expr())

        if self._consume("("):
            first = self._parse_type_expr()
            self._skip_ws()
            if not self._consume(","):
                self._expect(")")
                return first
            items = [first, self._parse_type_expr()]
            while True:
                self._skip_ws()
                if self._consume(","):
                    items.append(self._parse_type_expr())
                    continue
                break
            self._expect(")")
            return TypeTuple(items=tuple(items))

        name = self._parse_name()
        self._skip_ws()
        if self._consume("["):
            if name == "List":
                item = self._parse_type_expr()
                self._expect("]")
                return TypeList(item=item)
            dims: tuple[DimToken, ...]
            self._skip_ws()
            if self._peek() == "]":
                dims = ()
            else:
                dims = self._parse_dim_list()
            self._expect("]")
            if name == "Tensor":
                return TypeTensor(base=name, dims=dims)
            return TypeNamed(name=name, args=dims)

        if name == "Int":
            return TypeInt()
        if name in {"F", "Float"}:
            return TypeFloat()
        if name == "Bool":
            return TypeBool()
        if name == "Null":
            return TypeNull()
        if name in {"Str", "String"}:
            return TypeString()
        if name == "Path":
            return TypePath()
        if name == "Dim":
            return TypeDim()
        if re.fullmatch(r"_T[0-9]+", name):
            return TypeVar(name=name)
        if name == "Any":
            return TypeAny()
        return TypeNamed(name=name)


def parse_type_expr(type_expr: str) -> TypeExpr:
    return _TypeExprParser(type_expr).parse()


def is_numeric_type(tp: TypeExpr) -> bool:
    return isinstance(tp, TypeInt | TypeFloat)


def is_bool_like(tp: TypeExpr) -> bool:
    if isinstance(tp, TypeOptional):
        return is_bool_like(tp.inner)
    return isinstance(tp, TypeBool | TypeAny | TypeNamed)


def render_type(tp: TypeExpr) -> str:
    if isinstance(tp, TypeAny):
        return "Any"
    if isinstance(tp, TypeInt):
        return "Int"
    if isinstance(tp, TypeFloat):
        return "Float"
    if isinstance(tp, TypeBool):
        return "Bool"
    if isinstance(tp, TypeNull):
        return "Null"
    if isinstance(tp, TypeString):
        return "String"
    if isinstance(tp, TypePath):
        return "Path"
    if isinstance(tp, TypeDim):
        return "Dim"
    if isinstance(tp, TypeVar):
        return tp.name
    if isinstance(tp, TypeNamed):
        if tp.args:
            dims = ",".join(render_dim_token(dim) for dim in tp.args)
            return f"{tp.name}[{dims}]"
        return tp.name
    if isinstance(tp, TypeOptional):
        return f"?{render_type(tp.inner)}"
    if isinstance(tp, TypeTensor):
        if tp.dims:
            dims = ",".join(render_dim_token(dim) for dim in tp.dims)
            return f"{tp.base}[{dims}]"
        return tp.base
    if isinstance(tp, TypeList):
        return f"List[{render_type(tp.item)}]"
    if isinstance(tp, TypeTuple):
        return "(" + ", ".join(render_type(item) for item in tp.items) + ")"
    return "Any"


__all__ = [
    "DimToken",
    "DimExprBinary",
    "TypeAny",
    "TypeInt",
    "TypeFloat",
    "TypeBool",
    "TypeNull",
    "TypeString",
    "TypePath",
    "TypeDim",
    "TypeVar",
    "TypeNamed",
    "TypeOptional",
    "TypeTensor",
    "TypeList",
    "TypeTuple",
    "TypeAliasDef",
    "ConstraintAtom",
    "ConstraintOperand",
    "Constraint",
    "TypeExpr",
    "TypingRule",
    "TYPING_RULES",
    "parse_dim_expr",
    "parse_type_expr",
    "is_numeric_type",
    "is_bool_like",
    "render_type",
    "render_dim_token",
    "dim_token_names",
]
