from __future__ import annotations

import re
from dataclasses import dataclass

_NUMERIC_TOKEN_RE = re.compile(r"-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")


@dataclass(frozen=True)
class AxonParam:
    name: str
    optional: bool = False
    type_expr: str | None = None
    shape: tuple[str, ...] | None = None


@dataclass(frozen=True)
class AxonExprName:
    name: str


@dataclass(frozen=True)
class AxonExprLiteral:
    value: object


@dataclass(frozen=True)
class AxonExprTuple:
    items: tuple["AxonExpr", ...]


@dataclass(frozen=True)
class AxonExprCall:
    callee: str
    args: tuple["AxonExpr", ...]
    kwargs: dict[str, "AxonExpr | object"]


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


AxonExpr = (
    AxonExprName
    | AxonExprLiteral
    | AxonExprTuple
    | AxonExprCall
    | AxonExprPipe
    | AxonExprBind
    | AxonExprIf
    | AxonExprTernary
    | AxonExprBinary
    | AxonExprLambda
    | AxonExprParen
)


def render_axon_expr(expr: AxonExpr) -> str:
    if isinstance(expr, AxonExprName):
        return expr.name
    if isinstance(expr, AxonExprLiteral):
        value = expr.value
        if isinstance(value, str):
            if _NUMERIC_TOKEN_RE.fullmatch(value):
                return value
            return repr(value)
        if value is True:
            return "true"
        if value is False:
            return "false"
        if value is None:
            return "null"
        return str(value)
    if isinstance(expr, AxonExprTuple):
        return ", ".join(render_axon_expr(item) for item in expr.items)
    if isinstance(expr, AxonExprCall):
        args = [render_axon_expr(arg) for arg in expr.args]
        kwargs = []
        for key, value in expr.kwargs.items():
            if isinstance(
                value,
                (
                    AxonExprName,
                    AxonExprLiteral,
                    AxonExprTuple,
                    AxonExprCall,
                    AxonExprPipe,
                    AxonExprBind,
                    AxonExprIf,
                    AxonExprTernary,
                    AxonExprBinary,
                    AxonExprLambda,
                    AxonExprParen,
                ),
            ):
                kwargs.append(f"{key}={render_axon_expr(value)}")
            else:
                kwargs.append(f"{key}={value!r}" if isinstance(value, str) else f"{key}={value}")
        all_args = [*args, *kwargs]
        return f"{expr.callee}({', '.join(all_args)})"
    if isinstance(expr, AxonExprPipe):
        return " |> ".join(
            [render_axon_expr(expr.value), *[render_axon_expr(s) for s in expr.stages]]
        )
    if isinstance(expr, AxonExprBind):
        return f"{render_axon_expr(expr.value)} >>= \\{expr.var} -> {render_axon_expr(expr.body)}"
    if isinstance(expr, AxonExprIf):
        return f"if {render_axon_expr(expr.cond)} then {render_axon_expr(expr.true_expr)} else {render_axon_expr(expr.false_expr)}"
    if isinstance(expr, AxonExprTernary):
        return f"{render_axon_expr(expr.cond)} ? {render_axon_expr(expr.true_expr)} : {render_axon_expr(expr.false_expr)}"
    if isinstance(expr, AxonExprBinary):
        if expr.op in {"and", "or"}:
            return f"{render_axon_expr(expr.left)} {expr.op} {render_axon_expr(expr.right)}"
        return f"{render_axon_expr(expr.left)}{expr.op}{render_axon_expr(expr.right)}"
    if isinstance(expr, AxonExprLambda):
        return f"\\{expr.var} -> {render_axon_expr(expr.body)}"
    return f"({render_axon_expr(expr.inner)})"


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
    return_type_expr: str | None = None
    return_shape: tuple[str, ...] | None = None


__all__ = [
    "AxonParam",
    "AxonExpr",
    "AxonExprName",
    "AxonExprLiteral",
    "AxonExprTuple",
    "AxonExprCall",
    "AxonExprPipe",
    "AxonExprBind",
    "AxonExprIf",
    "AxonExprTernary",
    "AxonExprBinary",
    "AxonExprLambda",
    "AxonExprParen",
    "render_axon_expr",
    "AxonBind",
    "AxonReturn",
    "AxonRepeat",
    "AxonScopeBind",
    "AxonStatement",
    "AxonModule",
]
