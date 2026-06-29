from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TypeAlias

from .types import Constraint, DimToken, TypeAliasDef, TypeExpr


@dataclass(frozen=True, kw_only=True)
class AxonExprTyping:
    inferred_type: TypeExpr | None = None
    inferred_arity: int | None = None
    inferred_dims: tuple[DimToken, ...] | None = None


@dataclass(frozen=True)
class AxonParam:
    name: str
    optional: bool = False
    type_expr: TypeExpr | None = None
    default_expr: "AxonExpr | None" = None


@dataclass(frozen=True)
class AxonExprName(AxonExprTyping):
    name: str


@dataclass(frozen=True)
class AxonExprInt(AxonExprTyping):
    value: int


@dataclass(frozen=True)
class AxonExprFloat(AxonExprTyping):
    value: float
    lexeme: str | None = None


@dataclass(frozen=True)
class AxonExprBool(AxonExprTyping):
    value: bool


@dataclass(frozen=True)
class AxonExprNull(AxonExprTyping):
    pass


@dataclass(frozen=True)
class AxonExprString(AxonExprTyping):
    value: str


@dataclass(frozen=True)
class AxonExprPath(AxonExprTyping):
    absolute: bool
    parts: tuple[str, ...]

    def to_source(self) -> str:
        head = "@@" if self.absolute else "@"
        dotted = ".".join(self.parts)
        simple_part = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+)$")
        if all(simple_part.match(part) for part in self.parts):
            return head + dotted
        escaped = dotted.replace("\\", "\\\\").replace("'", "\\'")
        return f"{head}'{escaped}'"


@dataclass(frozen=True)
class AxonExprList(AxonExprTyping):
    items: tuple["AxonExpr", ...]


@dataclass(frozen=True)
class AxonExprTuple(AxonExprTyping):
    items: tuple["AxonExpr", ...]


@dataclass(frozen=True)
class AxonExprCall(AxonExprTyping):
    callee: str
    args: tuple["AxonExpr", ...]
    kwargs: dict[str, "AxonKwargValue"]


@dataclass(frozen=True)
class AxonExprPipe(AxonExprTyping):
    value: "AxonExpr"
    stages: tuple["AxonExpr", ...]


@dataclass(frozen=True)
class AxonExprBind(AxonExprTyping):
    value: "AxonExpr"
    var: str
    body: "AxonExpr"


@dataclass(frozen=True)
class AxonExprIf(AxonExprTyping):
    cond: "AxonExpr"
    true_expr: "AxonExpr"
    false_expr: "AxonExpr"


@dataclass(frozen=True)
class AxonExprTernary(AxonExprTyping):
    cond: "AxonExpr"
    true_expr: "AxonExpr"
    false_expr: "AxonExpr"


@dataclass(frozen=True)
class AxonExprBinary(AxonExprTyping):
    op: str
    left: "AxonExpr"
    right: "AxonExpr"


@dataclass(frozen=True)
class AxonExprLambda(AxonExprTyping):
    var: str
    body: "AxonExpr"


@dataclass(frozen=True)
class AxonExprParen(AxonExprTyping):
    inner: "AxonExpr"


@dataclass(frozen=True)
class AxonExprAscribe(AxonExprTyping):
    expr: "AxonExpr"
    type_expr: TypeExpr


@dataclass(frozen=True)
class AxonExprDo(AxonExprTyping):
    body: tuple["AxonStatement", ...]
    inline: bool = False


AxonExpr = (
    AxonExprName
    | AxonExprInt
    | AxonExprFloat
    | AxonExprBool
    | AxonExprNull
    | AxonExprString
    | AxonExprPath
    | AxonExprList
    | AxonExprTuple
    | AxonExprCall
    | AxonExprPipe
    | AxonExprBind
    | AxonExprIf
    | AxonExprTernary
    | AxonExprBinary
    | AxonExprLambda
    | AxonExprParen
    | AxonExprAscribe
    | AxonExprDo
)

AxonScalarValue: TypeAlias = bool | int | float | str | None
AxonListScalarValue: TypeAlias = list[int | str]
AxonKwargValue: TypeAlias = AxonExpr | AxonScalarValue | AxonListScalarValue


@dataclass(frozen=True)
class AxonBind:
    targets: tuple[str, ...]
    expr: AxonExpr


@dataclass(frozen=True)
class AxonReturn:
    values: tuple[AxonExpr, ...]


@dataclass(frozen=True)
class AxonRepeat:
    name: str | None
    var: str
    to_expr: AxonExpr
    from_expr: AxonExpr
    step_expr: AxonExpr
    body: tuple["AxonStatement", ...]
    targets: tuple[str, ...] | None = None
    carry: tuple[str, ...] | None = None


@dataclass(frozen=True)
class AxonYield:
    values: tuple[AxonExpr, ...]


@dataclass(frozen=True)
class AxonCond:
    cond: AxonExpr
    true_body: tuple["AxonStatement", ...]
    false_body: tuple["AxonStatement", ...]


@dataclass(frozen=True)
class AxonScopeBind:
    targets: tuple[str, ...]
    prefix: AxonExprPath
    body: tuple["AxonStatement", ...]
    kwargs: dict[str, "AxonKwargValue"] = field(default_factory=dict)


AxonStatement = AxonBind | AxonReturn | AxonRepeat | AxonYield | AxonCond | AxonScopeBind


@dataclass(frozen=True)
class AxonDefinition:
    name: str
    path_param: str | None
    params: tuple[AxonParam, ...]
    returns: tuple[str, ...]
    statements: tuple[AxonStatement, ...]
    body_expr: AxonExpr | None = None
    path_params: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    imported_members: dict[str, tuple[str, ...]] | None = None
    exports: tuple[str, ...] = ()
    symbols: dict[str, object] | None = None
    pragmas: dict[str, object] | None = None
    type_aliases: dict[str, TypeAliasDef] | None = None
    return_type_expr: TypeExpr | None = None
    constraints: tuple[Constraint, ...] | None = None
    is_global_binding: bool = False


__all__ = [
    "AxonParam",
    "AxonExpr",
    "AxonExprTyping",
    "AxonExprName",
    "AxonExprInt",
    "AxonExprFloat",
    "AxonExprBool",
    "AxonExprNull",
    "AxonExprString",
    "AxonExprPath",
    "AxonExprList",
    "AxonScalarValue",
    "AxonListScalarValue",
    "AxonKwargValue",
    "AxonExprTuple",
    "AxonExprCall",
    "AxonExprPipe",
    "AxonExprBind",
    "AxonExprIf",
    "AxonExprTernary",
    "AxonExprBinary",
    "AxonExprLambda",
    "AxonExprParen",
    "AxonExprAscribe",
    "AxonExprDo",
    "AxonBind",
    "AxonReturn",
    "AxonRepeat",
    "AxonYield",
    "AxonCond",
    "AxonScopeBind",
    "AxonStatement",
    "AxonDefinition",
]
