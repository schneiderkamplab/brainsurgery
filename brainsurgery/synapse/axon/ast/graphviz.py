from __future__ import annotations

from .nodes import (
    AxonBind,
    AxonCond,
    AxonExpr,
    AxonExprAscribe,
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
    AxonExprPath,
    AxonExprPipe,
    AxonExprString,
    AxonExprTernary,
    AxonExprTuple,
    AxonModule,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
    AxonYield,
)
from .render import render_axon_expr
from .source import AxonFile
from .types import render_type


def _quote(text: str) -> str:
    return '"' + text.replace('"', '\\"') + '"'


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


class _DotBuilder:
    def __init__(self) -> None:
        self._next = 0
        self.lines = [
            "digraph AxonAst {",
            "  rankdir=LR;",
            "  node [shape=box, fontname=Helvetica];",
        ]

    def node(self, label: str) -> str:
        name = f"n{self._next}"
        self._next += 1
        if "\n" in label:
            html = "<BR/>".join(_html_escape(part) for part in label.split("\n"))
            self.lines.append(f"  {name} [label=<{html}>];")
        else:
            self.lines.append(f"  {name} [label={_quote(label)}];")
        return name

    def edge(self, src: str, dst: str, label: str | None = None) -> None:
        if label is None:
            self.lines.append(f"  {src} -> {dst};")
        else:
            self.lines.append(f"  {src} -> {dst} [label={_quote(label)}];")

    def finish(self) -> str:
        return "\n".join([*self.lines, "}"]) + "\n"


def _typed_suffix(expr: AxonExpr, *, show_types: bool) -> str:
    if not show_types or expr.inferred_type is None:
        return ""
    return f"\n:: {render_type(expr.inferred_type)}"


def _emit_expr(dot: _DotBuilder, expr: AxonExpr, *, show_types: bool = False) -> str:
    if isinstance(expr, AxonExprName):
        return dot.node(f"name\n{expr.name}{_typed_suffix(expr, show_types=show_types)}")
    if isinstance(expr, AxonExprInt):
        return dot.node(f"int\n{expr.value}{_typed_suffix(expr, show_types=show_types)}")
    if isinstance(expr, AxonExprFloat):
        return dot.node(
            f"float\n{expr.lexeme if expr.lexeme is not None else expr.value}{_typed_suffix(expr, show_types=show_types)}"
        )
    if isinstance(expr, AxonExprBool):
        return dot.node(f"bool\n{expr.value}{_typed_suffix(expr, show_types=show_types)}")
    if isinstance(expr, AxonExprNull):
        return dot.node(f"null{_typed_suffix(expr, show_types=show_types)}")
    if isinstance(expr, AxonExprString):
        return dot.node(f"string\n{expr.value}{_typed_suffix(expr, show_types=show_types)}")
    if isinstance(expr, AxonExprPath):
        return dot.node(f"path\n{expr.to_source()}{_typed_suffix(expr, show_types=show_types)}")
    if isinstance(expr, AxonExprList):
        root = dot.node(f"list{_typed_suffix(expr, show_types=show_types)}")
        for idx, item in enumerate(expr.items):
            child = _emit_expr(dot, item, show_types=show_types)
            dot.edge(root, child, str(idx))
        return root
    if isinstance(expr, AxonExprTuple):
        root = dot.node(f"tuple{_typed_suffix(expr, show_types=show_types)}")
        for idx, item in enumerate(expr.items):
            child = _emit_expr(dot, item, show_types=show_types)
            dot.edge(root, child, str(idx))
        return root
    if isinstance(expr, AxonExprCall):
        root = dot.node(f"call\n{expr.callee}{_typed_suffix(expr, show_types=show_types)}")
        for idx, arg in enumerate(expr.args):
            child = _emit_expr(dot, arg, show_types=show_types)
            dot.edge(root, child, f"arg{idx}")
        for key, value in expr.kwargs.items():
            child = (
                _emit_expr(dot, value, show_types=show_types)
                if isinstance(value, AxonExpr)
                else dot.node(f"kw\n{key}={value}")
            )
            dot.edge(root, child, key)
        return root
    if isinstance(expr, AxonExprPipe):
        root = dot.node(f"pipe{_typed_suffix(expr, show_types=show_types)}")
        dot.edge(root, _emit_expr(dot, expr.value, show_types=show_types), "value")
        for idx, stage in enumerate(expr.stages):
            dot.edge(root, _emit_expr(dot, stage, show_types=show_types), f"stage{idx}")
        return root
    if isinstance(expr, AxonExprBind):
        root = dot.node(f"bind-expr\n{expr.var}{_typed_suffix(expr, show_types=show_types)}")
        dot.edge(root, _emit_expr(dot, expr.value, show_types=show_types), "value")
        dot.edge(root, _emit_expr(dot, expr.body, show_types=show_types), "body")
        return root
    if isinstance(expr, AxonExprIf):
        root = dot.node(f"if{_typed_suffix(expr, show_types=show_types)}")
        dot.edge(root, _emit_expr(dot, expr.cond, show_types=show_types), "cond")
        dot.edge(root, _emit_expr(dot, expr.true_expr, show_types=show_types), "then")
        dot.edge(root, _emit_expr(dot, expr.false_expr, show_types=show_types), "else")
        return root
    if isinstance(expr, AxonExprTernary):
        root = dot.node(f"?:{_typed_suffix(expr, show_types=show_types)}")
        dot.edge(root, _emit_expr(dot, expr.cond, show_types=show_types), "cond")
        dot.edge(root, _emit_expr(dot, expr.true_expr, show_types=show_types), "true")
        dot.edge(root, _emit_expr(dot, expr.false_expr, show_types=show_types), "false")
        return root
    if isinstance(expr, AxonExprBinary):
        root = dot.node(f"binary\n{expr.op}{_typed_suffix(expr, show_types=show_types)}")
        dot.edge(root, _emit_expr(dot, expr.left, show_types=show_types), "left")
        dot.edge(root, _emit_expr(dot, expr.right, show_types=show_types), "right")
        return root
    if isinstance(expr, AxonExprLambda):
        root = dot.node(f"lambda\n{expr.var}{_typed_suffix(expr, show_types=show_types)}")
        dot.edge(root, _emit_expr(dot, expr.body, show_types=show_types), "body")
        return root
    if isinstance(expr, AxonExprAscribe):
        root = dot.node(
            f"ascribe\n{render_type(expr.type_expr)}{_typed_suffix(expr, show_types=show_types)}"
        )
        dot.edge(root, _emit_expr(dot, expr.expr, show_types=show_types), "expr")
        return root
    if isinstance(expr, AxonExprParen):
        root = dot.node("paren")
        dot.edge(root, _emit_expr(dot, expr.inner, show_types=show_types))
        return root
    if isinstance(expr, AxonExprDo):
        root = dot.node(f"do{_typed_suffix(expr, show_types=show_types)}")
        for idx, stmt in enumerate(expr.body):
            dot.edge(root, _emit_stmt(dot, stmt, show_types=show_types), str(idx))
        return root
    return dot.node(f"expr\n{render_axon_expr(expr)}")


def _emit_stmt(dot: _DotBuilder, stmt: AxonStatement, *, show_types: bool = False) -> str:
    if isinstance(stmt, AxonBind):
        root = dot.node("bind\n" + ", ".join(stmt.targets))
        dot.edge(root, _emit_expr(dot, stmt.expr, show_types=show_types), "expr")
        return root
    if isinstance(stmt, AxonReturn):
        root = dot.node("return")
        for idx, value in enumerate(stmt.values):
            dot.edge(root, _emit_expr(dot, value, show_types=show_types), str(idx))
        return root
    if isinstance(stmt, AxonYield):
        root = dot.node("yield")
        for idx, value in enumerate(stmt.values):
            dot.edge(root, _emit_expr(dot, value, show_types=show_types), str(idx))
        return root
    if isinstance(stmt, AxonCond):
        root = dot.node("if")
        dot.edge(root, _emit_expr(dot, stmt.cond, show_types=show_types), "cond")
        for idx, inner in enumerate(stmt.true_body):
            dot.edge(root, _emit_stmt(dot, inner, show_types=show_types), f"then{idx}")
        for idx, inner in enumerate(stmt.false_body):
            dot.edge(root, _emit_stmt(dot, inner, show_types=show_types), f"else{idx}")
        return root
    if isinstance(stmt, AxonRepeat):
        root = dot.node(f"for\n{stmt.name or ''}\n{stmt.var}")
        dot.edge(root, _emit_expr(dot, stmt.from_expr, show_types=show_types), "from")
        dot.edge(root, _emit_expr(dot, stmt.to_expr, show_types=show_types), "to")
        dot.edge(root, _emit_expr(dot, stmt.step_expr, show_types=show_types), "step")
        for idx, inner in enumerate(stmt.body):
            dot.edge(root, _emit_stmt(dot, inner, show_types=show_types), f"body{idx}")
        return root
    if isinstance(stmt, AxonScopeBind):
        root = dot.node(f"scope\n{stmt.prefix.to_source()}\n{', '.join(stmt.targets)}")
        for idx, inner in enumerate(stmt.body):
            dot.edge(root, _emit_stmt(dot, inner, show_types=show_types), f"body{idx}")
        return root
    return dot.node("stmt")


def render_axon_file_to_dot(ast: AxonFile, *, show_types: bool = False) -> str:
    dot = _DotBuilder()
    root = dot.node(f"file\n{ast.origin_path if ast.origin_path is not None else '<memory>'}")
    if ast.imports:
        imports = dot.node("imports")
        dot.edge(root, imports)
        for namespace in ast.imports:
            members = ast.imported_members.get(namespace, ())
            label = namespace if not members else f"{namespace}\n({', '.join(members)})"
            dot.edge(imports, dot.node(label))
    if ast.exports:
        exports = dot.node("exports")
        dot.edge(root, exports)
        for symbol in ast.exports:
            dot.edge(exports, dot.node(symbol))
    if ast.pragmas:
        pragmas = dot.node("pragmas")
        dot.edge(root, pragmas)
        for key, value in ast.pragmas.items():
            dot.edge(pragmas, dot.node(f"{key}={value}"))
    if ast.type_aliases:
        aliases = dot.node("type_aliases")
        dot.edge(root, aliases)
        for name, alias_def in ast.type_aliases.items():
            params = f"[{', '.join(alias_def.params)}]" if alias_def.params else ""
            dot.edge(aliases, dot.node(f"{name}{params} = {render_type(alias_def.value)}"))
    if ast.constants:
        consts = dot.node("constants")
        dot.edge(root, consts)
        for name, expr in ast.constants.items():
            item = dot.node(f"const\n{name}")
            dot.edge(consts, item)
            dot.edge(item, _emit_expr(dot, expr, show_types=show_types), "expr")
    modules = dot.node("modules")
    dot.edge(root, modules)
    for module in ast.modules:
        dot.edge(modules, _emit_module(dot, module, show_types=show_types))
    return dot.finish()


def _emit_module(dot: _DotBuilder, module: AxonModule, *, show_types: bool = False) -> str:
    root = dot.node(f"module\n{module.name}")
    if module.return_type_expr is not None:
        dot.edge(root, dot.node(f"returns\n{render_type(module.return_type_expr)}"), "type")
    if module.params:
        params = dot.node("params")
        dot.edge(root, params)
        for param in module.params:
            label = param.name
            if param.type_expr is not None:
                label += f"\n{render_type(param.type_expr)}"
            if param.default_expr is not None:
                label += f"\n= {render_axon_expr(param.default_expr)}"
            dot.edge(params, dot.node(label))
    body = dot.node("body")
    dot.edge(root, body)
    if isinstance(module.body_expr, AxonExpr):
        dot.edge(body, _emit_expr(dot, module.body_expr, show_types=show_types), "expr")
    else:
        for idx, stmt in enumerate(module.statements):
            dot.edge(body, _emit_stmt(dot, stmt, show_types=show_types), str(idx))
    return root


__all__ = ["render_axon_file_to_dot"]
