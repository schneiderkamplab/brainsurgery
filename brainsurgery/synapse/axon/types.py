from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from .type_system import DimToken, TypeExpr


@dataclass(frozen=True)
class AxonParam:
    name: str
    optional: bool = False
    type_expr: TypeExpr | None = None
    shape: tuple[DimToken, ...] | None = None
    default_expr: "AxonExpr | None" = None


@dataclass(frozen=True)
class AxonExprName:
    name: str


@dataclass(frozen=True)
class AxonExprInt:
    value: int


@dataclass(frozen=True)
class AxonExprFloat:
    value: float
    lexeme: str | None = None


@dataclass(frozen=True)
class AxonExprBool:
    value: bool


@dataclass(frozen=True)
class AxonExprNull:
    pass


@dataclass(frozen=True)
class AxonExprString:
    value: str


@dataclass(frozen=True)
class AxonExprList:
    items: tuple["AxonExpr", ...]


@dataclass(frozen=True)
class AxonExprTuple:
    items: tuple["AxonExpr", ...]


@dataclass(frozen=True)
class AxonExprCall:
    callee: str
    args: tuple["AxonExpr", ...]
    kwargs: dict[str, "AxonKwargValue"]


@dataclass(frozen=True)
class AxonExprPipe:
    value: "AxonExpr"
    stages: tuple["AxonExpr", ...]


@dataclass(frozen=True)
class AxonExprBind:
    value: "AxonExpr"
    var: str
    body: "AxonExpr"


@dataclass(frozen=True)
class AxonExprIf:
    cond: "AxonExpr"
    true_expr: "AxonExpr"
    false_expr: "AxonExpr"


@dataclass(frozen=True)
class AxonExprTernary:
    cond: "AxonExpr"
    true_expr: "AxonExpr"
    false_expr: "AxonExpr"


@dataclass(frozen=True)
class AxonExprBinary:
    op: str
    left: "AxonExpr"
    right: "AxonExpr"


@dataclass(frozen=True)
class AxonExprLambda:
    var: str
    body: "AxonExpr"


@dataclass(frozen=True)
class AxonExprParen:
    inner: "AxonExpr"


@dataclass(frozen=True)
class AxonExprDo:
    body: tuple["AxonStatement", ...]
    inline: bool = False


AxonExpr = (
    AxonExprName
    | AxonExprInt
    | AxonExprFloat
    | AxonExprBool
    | AxonExprNull
    | AxonExprString
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


@dataclass(frozen=True)
class AxonScopeBind:
    targets: tuple[str, ...]
    prefix: str
    body: tuple["AxonStatement", ...]
    kwargs: dict[str, "AxonKwargValue"] = field(default_factory=dict)


AxonStatement = AxonBind | AxonReturn | AxonRepeat | AxonScopeBind


@dataclass(frozen=True)
class AxonModule:
    name: str
    path_param: str | None
    params: tuple[AxonParam, ...]
    returns: tuple[str, ...]
    statements: tuple[AxonStatement, ...]
    path_params: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    imported_members: dict[str, tuple[str, ...]] | None = None
    symbols: dict[str, object] | None = None
    pragmas: dict[str, object] | None = None
    type_aliases: dict[str, TypeExpr] | None = None
    return_type_expr: TypeExpr | None = None
    return_shape: tuple[DimToken, ...] | None = None


__all__ = [
    "AxonParam",
    "AxonExpr",
    "AxonExprName",
    "AxonExprInt",
    "AxonExprFloat",
    "AxonExprBool",
    "AxonExprNull",
    "AxonExprString",
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
    "AxonExprDo",
    "AxonBind",
    "AxonReturn",
    "AxonRepeat",
    "AxonScopeBind",
    "AxonStatement",
    "AxonModule",
]
