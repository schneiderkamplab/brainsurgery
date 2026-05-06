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
    AxonKwargValue,
    AxonDefinition,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
    AxonYield,
)
from .source import AxonFile
from .types import TypeAliasDef, render_type


def _collect_file_type_aliases(ast: AxonFile) -> dict[str, TypeAliasDef]:
    merged: dict[str, TypeAliasDef] = dict(ast.type_aliases)
    for module in ast.modules:
        if not module.type_aliases:
            continue
        for name, alias_def in module.type_aliases.items():
            prev = merged.get(name)
            if prev is not None and prev != alias_def:
                raise ValueError(f"conflicting type alias {name!r} in Axon AST render")
            merged[name] = alias_def
    return merged


def _format_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _format_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return _format_string(value)
    return repr(value)


def _format_pragma_value(value: object) -> str:
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    return _format_scalar(value)


def _emit_inferred_expr_types(show_types: bool, show_inferred_expr_types: bool | None) -> bool:
    return show_types if show_inferred_expr_types is None else show_inferred_expr_types


def _render_kwarg_value(
    value: AxonKwargValue, *, bound_names: set[str], show_types: bool = False, show_inferred_expr_types: bool | None = None
) -> str:
    if isinstance(value, AxonExpr):
        return _render_call_arg(
            value,
            bound_names=bound_names,
            show_types=show_types,
            show_inferred_expr_types=show_inferred_expr_types,
        )
    if isinstance(value, list):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    return _format_scalar(value)


def _render_call_arg(expr: AxonExpr, *, bound_names: set[str], show_types: bool = False, show_inferred_expr_types: bool | None = None) -> str:
    rendered = render_axon_expr(expr, bound_names=bound_names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)
    if _emit_inferred_expr_types(show_types, show_inferred_expr_types) and expr.inferred_type is not None:
        return f"({rendered})"
    if isinstance(
        expr,
        AxonExprCall
        | AxonExprPipe
        | AxonExprBind
        | AxonExprIf
        | AxonExprTernary
        | AxonExprBinary
        | AxonExprLambda
        | AxonExprAscribe,
    ):
        return f"({rendered})"
    return rendered


def _render_axon_expr_base(
    expr: AxonExpr, *, bound_names: set[str] | None = None, show_types: bool = False, show_inferred_expr_types: bool | None = None
) -> str:
    names = set() if bound_names is None else set(bound_names)
    if isinstance(expr, AxonExprName):
        return expr.name
    if isinstance(expr, AxonExprInt):
        return str(expr.value)
    if isinstance(expr, AxonExprFloat):
        return expr.lexeme if expr.lexeme is not None else repr(expr.value)
    if isinstance(expr, AxonExprBool):
        return "true" if expr.value else "false"
    if isinstance(expr, AxonExprNull):
        return "null"
    if isinstance(expr, AxonExprString):
        return _format_string(expr.value)
    if isinstance(expr, AxonExprPath):
        return expr.to_source()
    if isinstance(expr, AxonExprList):
        return (
            "["
            + ", ".join(
                render_axon_expr(item, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)
                for item in expr.items
            )
            + "]"
        )
    if isinstance(expr, AxonExprTuple):
        return (
            "("
            + ", ".join(
                render_axon_expr(item, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)
                for item in expr.items
            )
            + ")"
        )
    if isinstance(expr, AxonExprCall):
        parts = [_render_call_arg(arg, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types) for arg in expr.args]
        parts.extend(
            f"{key}={_render_kwarg_value(value, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)}"
            for key, value in expr.kwargs.items()
        )
        return expr.callee if not parts else f"{expr.callee} {' '.join(parts)}"
    if isinstance(expr, AxonExprPipe):
        return (
            f"{render_axon_expr(expr.value, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)} |> "
            + " |> ".join(
                render_axon_expr(stage, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)
                for stage in expr.stages
            )
        )
    if isinstance(expr, AxonExprBind):
        nested = {*names, expr.var}
        return (
            f"{expr.var} <- {render_axon_expr(expr.value, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)}; "
            f"{render_axon_expr(expr.body, bound_names=nested, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)}"
        )
    if isinstance(expr, AxonExprIf):
        return (
            f"if {render_axon_expr(expr.cond, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)} then "
            f"{render_axon_expr(expr.true_expr, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)} else "
            f"{render_axon_expr(expr.false_expr, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)}"
        )
    if isinstance(expr, AxonExprTernary):
        return (
            f"{render_axon_expr(expr.cond, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)} ? "
            f"{render_axon_expr(expr.true_expr, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)} : "
            f"{render_axon_expr(expr.false_expr, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)}"
        )
    if isinstance(expr, AxonExprBinary):
        return (
            f"{render_axon_expr(expr.left, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)} {expr.op} "
            f"{render_axon_expr(expr.right, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)}"
        )
    if isinstance(expr, AxonExprLambda):
        return f"\\{expr.var} -> {render_axon_expr(expr.body, bound_names={*names, expr.var}, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)}"
    if isinstance(expr, AxonExprParen):
        if isinstance(expr.inner, AxonExprAscribe):
            return render_axon_expr(expr.inner, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)
        return f"({render_axon_expr(expr.inner, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)})"
    if isinstance(expr, AxonExprAscribe):
        inner = render_axon_expr(expr.expr, bound_names=names, show_types=False)
        if isinstance(
            expr.expr,
            AxonExprCall
            | AxonExprPipe
            | AxonExprBind
            | AxonExprIf
            | AxonExprTernary
            | AxonExprBinary
            | AxonExprLambda
            | AxonExprDo,
        ):
            inner = f"({inner})"
        return f"({inner} :: {render_type(expr.type_expr)})"
    if isinstance(expr, AxonExprDo):
        return (
            "(do "
            + _render_inline_statements(expr.body, bound_names=names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)
            + ")"
        )
    raise TypeError(f"unsupported Axon expression: {type(expr).__name__}")


def render_axon_expr(
    expr: AxonExpr, *, bound_names: set[str] | None = None, show_types: bool = False, show_inferred_expr_types: bool | None = None
) -> str:
    rendered = _render_axon_expr_base(expr, bound_names=bound_names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)
    if (
        _emit_inferred_expr_types(show_types, show_inferred_expr_types)
        and not isinstance(expr, AxonExprAscribe)
        and expr.inferred_type is not None
    ):
        if isinstance(
            expr,
            AxonExprCall
            | AxonExprPipe
            | AxonExprBind
            | AxonExprIf
            | AxonExprTernary
            | AxonExprBinary
            | AxonExprLambda
            | AxonExprDo,
        ):
            rendered = f"({rendered})"
        return f"({rendered} :: {render_type(expr.inferred_type)})"
    return rendered


def render_axon_statement(
    stmt: AxonStatement, *, indent: str, bound_names: set[str], show_types: bool = False, show_inferred_expr_types: bool | None = None
) -> list[str]:
    local = set(bound_names)
    if isinstance(stmt, AxonBind):
        lhs = ", ".join(stmt.targets)
        expr_lines = _render_expr_lines(
            stmt.expr, indent=indent + "  ", bound_names=local, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
        )
        first, *rest = expr_lines
        return [f"{indent}{lhs} <- {first}", *(f"{indent}  {line}" for line in rest)]
    if isinstance(stmt, AxonReturn):
        if len(stmt.values) == 1 and _expr_needs_multiline(stmt.values[0]):
            expr_lines = _render_expr_lines(
                stmt.values[0], indent=indent + "  ", bound_names=local, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
            )
            first, *rest = expr_lines
            return [f"{indent}return {first}", *(f"{indent}  {line}" for line in rest)]
        return [
            f"{indent}return "
            + ", ".join(
                render_axon_expr(v, bound_names=local, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types) for v in stmt.values
            )
        ]
    if isinstance(stmt, AxonYield):
        return [
            f"{indent}yield "
            + ", ".join(
                render_axon_expr(v, bound_names=local, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types) for v in stmt.values
            )
        ]
    if isinstance(stmt, AxonCond):
        header = f"{indent}if {render_axon_expr(stmt.cond, bound_names=local, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)} then do"
        return [
            header,
            *_render_statements(
                stmt.true_body, indent=indent + "  ", bound_names=local, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
            ),
            f"{indent}else do",
            *_render_statements(
                stmt.false_body, indent=indent + "  ", bound_names=local, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
            ),
        ]
    if isinstance(stmt, AxonRepeat):
        header = ""
        if stmt.targets:
            header = ", ".join(stmt.targets) + " <- "
        loop = (
            f"{indent}{header}for"
            f"{'@' + stmt.name if stmt.name else ''} {stmt.var} <- "
            f"[{render_axon_expr(stmt.from_expr, bound_names=local, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)}..{render_axon_expr(stmt.to_expr, bound_names=local, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)})"
        )
        step = render_axon_expr(stmt.step_expr, bound_names=local, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)
        if step != "1":
            loop += f" step={step}"
        if stmt.carry:
            loop += " carry (" + ", ".join(stmt.carry) + ")"
        loop_bound = {*local, stmt.var}
        if stmt.carry:
            loop_bound.update(name for name in stmt.carry if name != "_")
        if len(stmt.body) == 1 and isinstance(stmt.body[0], AxonYield):
            if len(stmt.body[0].values) == 1:
                inline_expr = render_axon_expr(
                    stmt.body[0].values[0], bound_names=loop_bound, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
                )
            else:
                inline_expr = render_axon_expr(
                    AxonExprTuple(items=stmt.body[0].values),
                    bound_names=loop_bound,
                    show_types=show_types, show_inferred_expr_types=show_inferred_expr_types,
                )
            return [f"{loop} {inline_expr}"]
        loop += " do"
        out = [loop]
        out.extend(
            _render_statements(
                stmt.body, indent=indent + "  ", bound_names=loop_bound, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
            )
        )
        return out
    if isinstance(stmt, AxonScopeBind):
        lhs = ", ".join(stmt.targets)
        kwargs = " ".join(
            f"{key}={_render_kwarg_value(value, bound_names=local, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)}"
            for key, value in stmt.kwargs.items()
        )
        header = f"{indent}{lhs} <- scope{stmt.prefix.to_source()}"
        if kwargs:
            header += f" {kwargs}"
        header += " do"
        return [
            header,
            *_render_statements(
                stmt.body, indent=indent + "  ", bound_names=local, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
            ),
        ]
    raise TypeError(f"unsupported Axon statement: {type(stmt).__name__}")


def _render_statements(
    statements: tuple[AxonStatement, ...],
    *,
    indent: str,
    bound_names: set[str],
    show_types: bool = False, show_inferred_expr_types: bool | None = None,
) -> list[str]:
    lines: list[str] = []
    local = set(bound_names)
    for stmt in statements:
        rendered = render_axon_statement(
            stmt, indent=indent, bound_names=local, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
        )
        lines.extend(rendered)
        if isinstance(stmt, AxonBind):
            local.update(name for name in stmt.targets if name != "_")
        elif isinstance(stmt, AxonRepeat) and stmt.targets:
            local.update(name for name in stmt.targets if name != "_")
        elif isinstance(stmt, AxonScopeBind):
            local.update(name for name in stmt.targets if name != "_")
    return lines


def _render_inline_statements(
    statements: tuple[AxonStatement, ...], *, bound_names: set[str], show_types: bool = False, show_inferred_expr_types: bool | None = None
) -> str:
    return "; ".join(
        line.strip()
        for line in _render_statements(
            statements, indent="", bound_names=bound_names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
        )
    )


def _expr_needs_multiline(expr: AxonExpr) -> bool:
    if isinstance(expr, AxonExprParen):
        return _expr_needs_multiline(expr.inner)
    if isinstance(expr, AxonExprAscribe):
        return _expr_needs_multiline(expr.expr)
    if isinstance(expr, AxonExprDo) and not expr.inline:
        return True
    if isinstance(expr, AxonExprTernary):
        return _expr_needs_multiline(expr.true_expr) or _expr_needs_multiline(expr.false_expr)
    if isinstance(expr, AxonExprIf):
        return _expr_needs_multiline(expr.true_expr) or _expr_needs_multiline(expr.false_expr)
    return False


def _render_expr_lines(
    expr: AxonExpr, *, indent: str, bound_names: set[str], show_types: bool = False, show_inferred_expr_types: bool | None = None
) -> list[str]:
    if isinstance(expr, AxonExprParen) and _expr_needs_multiline(expr.inner):
        return _render_expr_lines(
            expr.inner, indent=indent, bound_names=bound_names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
        )
    if isinstance(expr, AxonExprAscribe) and _expr_needs_multiline(expr.expr):
        return [render_axon_expr(expr, bound_names=bound_names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)]
    if isinstance(expr, AxonExprDo) and not expr.inline:
        return [
            "do",
            *_render_statements(
                expr.body, indent=indent + "  ", bound_names=bound_names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
            ),
        ]
    if isinstance(expr, AxonExprTernary) and _expr_needs_multiline(expr):
        cond = render_axon_expr(expr.cond, bound_names=bound_names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)
        true_lines = _render_expr_lines(
            expr.true_expr, indent=indent + "  ", bound_names=bound_names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
        )
        false_lines = _render_expr_lines(
            expr.false_expr, indent=indent + "  ", bound_names=bound_names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
        )
        out = [cond]
        first_true, *rest_true = true_lines
        out.append(f"{indent}? {first_true}")
        out.extend(rest_true)
        first_false, *rest_false = false_lines
        out.append(f"{indent}: {first_false}")
        out.extend(rest_false)
        return out
    if isinstance(expr, AxonExprIf) and _expr_needs_multiline(expr):
        cond = render_axon_expr(expr.cond, bound_names=bound_names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)
        true_lines = _render_expr_lines(
            expr.true_expr, indent=indent + "  ", bound_names=bound_names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
        )
        false_lines = _render_expr_lines(
            expr.false_expr, indent=indent + "  ", bound_names=bound_names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
        )
        out = [f"if {cond} then"]
        out.extend(f"{indent}  {line}" for line in true_lines)
        out.append(f"{indent}else")
        out.extend(f"{indent}  {line}" for line in false_lines)
        return out
    return [render_axon_expr(expr, bound_names=bound_names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)]


def render_axon_module(module: AxonDefinition, *, show_types: bool = False, show_inferred_expr_types: bool | None = None) -> str:
    path_params = list(module.path_params)
    if module.path_param and module.path_param not in path_params:
        path_params.append(module.path_param)
    sig_parts = ["Path" for _ in path_params]
    for param in module.params:
        type_text = render_type(param.type_expr) if param.type_expr is not None else "Any"
        sig_parts.append(f"?{type_text}" if param.optional else type_text)
    sig_parts.append(
        render_type(module.return_type_expr) if module.return_type_expr is not None else "Any"
    )
    signature = " -> ".join(sig_parts)
    head = module.name + "".join(f"@{name}" for name in path_params)
    bound_names = set(path_params)
    rendered_params: list[str] = []
    for param in module.params:
        text = param.name
        if param.default_expr is not None:
            text = f"?{text}={render_axon_expr(param.default_expr, bound_names={*bound_names, *(p.name for p in module.params)}, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types)}"
        rendered_params.append(text)
        bound_names.add(param.name)
    header = head + (" " + " ".join(rendered_params) if rendered_params else "")
    if isinstance(module.body_expr, AxonExpr):
        expr_lines = _render_expr_lines(
            module.body_expr, indent="  ", bound_names=bound_names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
        )
        first, *rest = expr_lines
        body_lines = [header + " = " + first, *(f"  {line}" for line in rest)]
    else:
        body_lines = [
            header + " = do",
            *_render_statements(
                module.statements, indent="  ", bound_names=bound_names, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types
            ),
        ]
    has_signature = (
        module.return_type_expr is not None
        or any(param.type_expr is not None for param in module.params)
        or bool(path_params)
    )
    if not has_signature:
        return "\n".join(body_lines)
    return "\n".join([f"{module.name} :: {signature}", *body_lines])


def render_axon_file(ast: AxonFile, *, show_types: bool = False, show_inferred_expr_types: bool | None = None) -> str:
    blocks: list[str] = []
    for key, value in ast.pragmas.items():
        name = key.upper()
        blocks.append(f"{{-# {name} {_format_pragma_value(value)} #-}}")
    for namespace in ast.imports:
        members = ast.imported_members.get(namespace, ())
        if members:
            blocks.append(f"import {namespace} ({', '.join(members)})")
        else:
            blocks.append(f"import {namespace}")
    if ast.exports:
        blocks.append(f"export ({', '.join(ast.exports)})")
    for name, alias_def in _collect_file_type_aliases(ast).items():
        params = f"[{', '.join(alias_def.params)}]" if alias_def.params else ""
        blocks.append(f"type {name}{params} = {render_type(alias_def.value)}")
    blocks.extend(render_axon_module(module, show_types=show_types, show_inferred_expr_types=show_inferred_expr_types) for module in ast.modules)
    return "\n\n".join(blocks) + "\n"


__all__ = ["render_axon_expr", "render_axon_file", "render_axon_module", "render_axon_statement"]
