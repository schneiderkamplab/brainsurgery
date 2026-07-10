from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from ..ast import (
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
    AxonFile,
    AxonKwargValue,
    AxonDefinition,
    AxonParam,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
    AxonYield,
    DimExprBinary,
    DimToken,
    TypeAliasDef,
    TypeAny,
    TypeExpr,
    TypeInt,
    TypeList,
    TypeNamed,
    TypeNull,
    TypeOptional,
    TypePath,
    TypeTensor,
    TypeTuple,
    TypeVar,
    absolutize_path_expr,
)
from ..validate import validate_elaborated_axon_file, validate_flat_axon_file

_ATOMIC_EXPR_TYPES = (
    AxonExprName,
    AxonExprInt,
    AxonExprFloat,
    AxonExprBool,
    AxonExprNull,
    AxonExprString,
    AxonExprPath,
)


@dataclass
class _FlattenCtx:
    used_names: set[str]
    module_path_params: tuple[str, ...] = ()
    temp_counter: int = 0
    helper_counter: int = 0
    type_var_counter: int = 0

    def fresh(self, *, prefix: str = "__flat") -> str:
        while True:
            self.temp_counter += 1
            candidate = f"{prefix}_{self.temp_counter}"
            if candidate not in self.used_names:
                self.used_names.add(candidate)
                return candidate

    def fresh_helper(self, *, prefix: str) -> str:
        while True:
            self.helper_counter += 1
            candidate = f"{prefix}_{self.helper_counter}"
            if candidate not in self.used_names:
                self.used_names.add(candidate)
                return candidate

    def fresh_type_var(self) -> TypeVar:
        self.type_var_counter += 1
        return TypeVar(name=f"_T{self.type_var_counter}")


@dataclass
class _FlattenProgramCtx:
    modules_by_name: dict[str, AxonDefinition]
    type_aliases: dict[str, TypeAliasDef]
    scoped_modules: frozenset[str]
    root_called_modules: set[str]
    nonempty_called_modules: set[str]
    scope_param_name: str = "__scope"


def _type_aliases_for_module(
    program_ctx: _FlattenProgramCtx,
    module: AxonDefinition,
) -> dict[str, TypeAliasDef]:
    aliases = dict(program_ctx.type_aliases)
    if module.type_aliases:
        aliases.update(module.type_aliases)
    return aliases


def _expr_has_relative_path(expr: AxonExpr) -> bool:
    if isinstance(expr, AxonExprPath):
        return not expr.absolute
    if isinstance(expr, AxonExprCall):
        return any(_expr_has_relative_path(arg) for arg in expr.args) or any(
            _expr_has_relative_path(value)
            for value in expr.kwargs.values()
            if isinstance(value, AxonExpr)
        )
    if isinstance(expr, AxonExprPipe):
        return _expr_has_relative_path(expr.value) or any(
            _expr_has_relative_path(stage) for stage in expr.stages
        )
    if isinstance(expr, AxonExprBind):
        return _expr_has_relative_path(expr.value) or _expr_has_relative_path(expr.body)
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return (
            _expr_has_relative_path(expr.cond)
            or _expr_has_relative_path(expr.true_expr)
            or _expr_has_relative_path(expr.false_expr)
        )
    if isinstance(expr, AxonExprBinary):
        return _expr_has_relative_path(expr.left) or _expr_has_relative_path(expr.right)
    if isinstance(expr, AxonExprLambda):
        return _expr_has_relative_path(expr.body)
    if isinstance(expr, AxonExprParen):
        return _expr_has_relative_path(expr.inner)
    if isinstance(expr, AxonExprAscribe):
        return _expr_has_relative_path(expr.expr)
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return any(_expr_has_relative_path(item) for item in expr.items)
    if isinstance(expr, AxonExprDo):
        return any(_stmt_has_lexical_path_dependency(stmt) for stmt in expr.body)
    return False


def _stmt_has_lexical_path_dependency(stmt: AxonStatement) -> bool:
    if isinstance(stmt, AxonScopeBind):
        return True
    if isinstance(stmt, AxonBind):
        return _expr_has_relative_path(stmt.expr)
    if isinstance(stmt, AxonReturn | AxonYield):
        return any(_expr_has_relative_path(value) for value in stmt.values)
    if isinstance(stmt, AxonCond):
        return (
            _expr_has_relative_path(stmt.cond)
            or any(_stmt_has_lexical_path_dependency(item) for item in stmt.true_body)
            or any(_stmt_has_lexical_path_dependency(item) for item in stmt.false_body)
        )
    if isinstance(stmt, AxonRepeat):
        return (
            _expr_has_relative_path(stmt.from_expr)
            or _expr_has_relative_path(stmt.to_expr)
            or _expr_has_relative_path(stmt.step_expr)
            or any(_stmt_has_lexical_path_dependency(item) for item in stmt.body)
        )
    return False


def _expr_called_definitions(expr: AxonExpr, module_names: set[str]) -> set[str]:
    out: set[str] = set()
    if isinstance(expr, AxonExprCall):
        if expr.callee in module_names:
            out.add(expr.callee)
        for arg in expr.args:
            out.update(_expr_called_definitions(arg, module_names))
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                out.update(_expr_called_definitions(value, module_names))
    elif isinstance(expr, AxonExprPipe):
        out.update(_expr_called_definitions(expr.value, module_names))
        for stage in expr.stages:
            out.update(_expr_called_definitions(stage, module_names))
    elif isinstance(expr, AxonExprBind):
        out.update(_expr_called_definitions(expr.value, module_names))
        out.update(_expr_called_definitions(expr.body, module_names))
    elif isinstance(expr, AxonExprIf | AxonExprTernary):
        out.update(_expr_called_definitions(expr.cond, module_names))
        out.update(_expr_called_definitions(expr.true_expr, module_names))
        out.update(_expr_called_definitions(expr.false_expr, module_names))
    elif isinstance(expr, AxonExprBinary):
        out.update(_expr_called_definitions(expr.left, module_names))
        out.update(_expr_called_definitions(expr.right, module_names))
    elif isinstance(expr, AxonExprLambda):
        out.update(_expr_called_definitions(expr.body, module_names))
    elif isinstance(expr, AxonExprParen):
        out.update(_expr_called_definitions(expr.inner, module_names))
    elif isinstance(expr, AxonExprAscribe):
        out.update(_expr_called_definitions(expr.expr, module_names))
    elif isinstance(expr, AxonExprList | AxonExprTuple):
        for item in expr.items:
            out.update(_expr_called_definitions(item, module_names))
    elif isinstance(expr, AxonExprDo):
        for stmt in expr.body:
            out.update(_stmt_called_definitions(stmt, module_names))
    return out


def _stmt_called_definitions(stmt: AxonStatement, module_names: set[str]) -> set[str]:
    out: set[str] = set()
    if isinstance(stmt, AxonBind):
        out.update(_expr_called_definitions(stmt.expr, module_names))
    elif isinstance(stmt, AxonReturn | AxonYield):
        for value in stmt.values:
            out.update(_expr_called_definitions(value, module_names))
    elif isinstance(stmt, AxonCond):
        out.update(_expr_called_definitions(stmt.cond, module_names))
        for item in stmt.true_body:
            out.update(_stmt_called_definitions(item, module_names))
        for item in stmt.false_body:
            out.update(_stmt_called_definitions(item, module_names))
    elif isinstance(stmt, AxonRepeat):
        out.update(_expr_called_definitions(stmt.from_expr, module_names))
        out.update(_expr_called_definitions(stmt.to_expr, module_names))
        out.update(_expr_called_definitions(stmt.step_expr, module_names))
        for item in stmt.body:
            out.update(_stmt_called_definitions(item, module_names))
    elif isinstance(stmt, AxonScopeBind):
        for raw_value in stmt.kwargs.values():
            if isinstance(raw_value, AxonExpr):
                out.update(_expr_called_definitions(raw_value, module_names))
        for item in stmt.body:
            out.update(_stmt_called_definitions(item, module_names))
    return out


def _lexically_path_dependent_modules(modules: tuple[AxonDefinition, ...]) -> frozenset[str]:
    module_names = {module.name for module in modules}
    direct: set[str] = set()
    calls: dict[str, set[str]] = {}
    for module in modules:
        body = _module_body_statements(module)
        if any(_stmt_has_lexical_path_dependency(stmt) for stmt in body):
            direct.add(module.name)
        calls[module.name] = set()
        for stmt in body:
            calls[module.name].update(_stmt_called_definitions(stmt, module_names))

    dependent = set(direct)
    changed = True
    while changed:
        changed = False
        for name, callees in calls.items():
            if name in dependent:
                continue
            if callees & dependent:
                dependent.add(name)
                changed = True
    return frozenset(dependent)


def _module_used_names(module: AxonDefinition) -> set[str]:
    names = {module.name}
    if module.path_param is not None:
        names.add(module.path_param)
    names.update(module.path_params)
    names.update(param.name for param in module.params)

    def visit_expr(expr: AxonExpr) -> None:
        if isinstance(expr, AxonExprName):
            names.add(expr.name)
        elif isinstance(expr, AxonExprCall):
            names.add(expr.callee.split("@", 1)[0])
            for arg in expr.args:
                visit_expr(arg)
            for value in expr.kwargs.values():
                if isinstance(value, AxonExpr):
                    visit_expr(value)
        elif isinstance(expr, AxonExprPipe):
            visit_expr(expr.value)
            for stage in expr.stages:
                visit_expr(stage)
        elif isinstance(expr, AxonExprBind):
            names.add(expr.var)
            visit_expr(expr.value)
            visit_expr(expr.body)
        elif isinstance(expr, AxonExprIf | AxonExprTernary):
            visit_expr(expr.cond)
            visit_expr(expr.true_expr)
            visit_expr(expr.false_expr)
        elif isinstance(expr, AxonExprBinary):
            visit_expr(expr.left)
            visit_expr(expr.right)
        elif isinstance(expr, AxonExprLambda):
            names.add(expr.var)
            visit_expr(expr.body)
        elif isinstance(expr, AxonExprParen):
            visit_expr(expr.inner)
        elif isinstance(expr, AxonExprAscribe):
            visit_expr(expr.expr)
        elif isinstance(expr, AxonExprList | AxonExprTuple):
            for item in expr.items:
                visit_expr(item)
        elif isinstance(expr, AxonExprDo):
            for stmt in expr.body:
                visit_stmt(stmt)

    def visit_stmt(stmt: AxonStatement) -> None:
        if isinstance(stmt, AxonBind):
            names.update(name for name in stmt.targets if name != "_")
            visit_expr(stmt.expr)
        elif isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                visit_expr(value)
        elif isinstance(stmt, AxonCond):
            visit_expr(stmt.cond)
            for inner in stmt.true_body:
                visit_stmt(inner)
            for inner in stmt.false_body:
                visit_stmt(inner)
        elif isinstance(stmt, AxonRepeat):
            names.add(stmt.var)
            names.update(name for name in (stmt.targets or ()) if name != "_")
            names.update(name for name in (stmt.carry or ()) if name != "_")
            visit_expr(stmt.from_expr)
            visit_expr(stmt.to_expr)
            visit_expr(stmt.step_expr)
            for inner in stmt.body:
                visit_stmt(inner)
        elif isinstance(stmt, AxonScopeBind):
            names.update(name for name in stmt.targets if name != "_")
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    visit_expr(raw_value)
            for inner in stmt.body:
                visit_stmt(inner)

    if module.body_expr is not None:
        visit_expr(module.body_expr)
    for stmt in module.statements:
        visit_stmt(stmt)
    return names


def _is_atomic_expr(expr: AxonExpr) -> bool:
    if isinstance(expr, _ATOMIC_EXPR_TYPES):
        return True
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return all(_is_atomic_expr(item) for item in expr.items)
    if isinstance(expr, AxonExprAscribe):
        return _is_atomic_expr(expr.expr)
    return False


def _bind_if_non_atomic(
    prelude: list[AxonStatement],
    expr: AxonExpr,
    ctx: _FlattenCtx,
) -> AxonExpr:
    if _is_atomic_expr(expr):
        return expr
    temp = ctx.fresh()
    prelude.append(AxonBind(targets=(temp,), expr=expr))
    return AxonExprName(name=temp)


def _wrap_inline_do(prelude: list[AxonStatement], expr: AxonExpr) -> AxonExpr:
    if not prelude:
        return expr
    return AxonExprDo(body=tuple([*prelude, AxonReturn(values=(expr,))]), inline=True)


def _substitute_name_expr(expr: AxonExpr, *, name: str, replacement: AxonExpr) -> AxonExpr:
    if isinstance(expr, AxonExprName):
        return replacement if expr.name == name else expr
    if isinstance(expr, AxonExprCall):
        return replace(
            expr,
            args=tuple(_substitute_name_expr(arg, name=name, replacement=replacement) for arg in expr.args),
            kwargs={
                key: (
                    _substitute_name_expr(raw_value, name=name, replacement=replacement)
                    if isinstance(raw_value, AxonExpr)
                    else raw_value
                )
                for key, raw_value in expr.kwargs.items()
            },
        )
    if isinstance(expr, AxonExprBinary):
        return replace(
            expr,
            left=_substitute_name_expr(expr.left, name=name, replacement=replacement),
            right=_substitute_name_expr(expr.right, name=name, replacement=replacement),
        )
    if isinstance(expr, AxonExprBind):
        value = _substitute_name_expr(expr.value, name=name, replacement=replacement)
        if expr.var == name:
            return replace(expr, value=value)
        return replace(
            expr,
            value=value,
            body=_substitute_name_expr(expr.body, name=name, replacement=replacement),
        )
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return replace(
            expr,
            cond=_substitute_name_expr(expr.cond, name=name, replacement=replacement),
            true_expr=_substitute_name_expr(expr.true_expr, name=name, replacement=replacement),
            false_expr=_substitute_name_expr(expr.false_expr, name=name, replacement=replacement),
        )
    if isinstance(expr, AxonExprLambda):
        if expr.var == name:
            return expr
        return replace(
            expr,
            body=_substitute_name_expr(expr.body, name=name, replacement=replacement),
        )
    if isinstance(expr, AxonExprParen):
        return replace(
            expr,
            inner=_substitute_name_expr(expr.inner, name=name, replacement=replacement),
        )
    if isinstance(expr, AxonExprAscribe):
        return replace(
            expr,
            expr=_substitute_name_expr(expr.expr, name=name, replacement=replacement),
        )
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return replace(
            expr,
            items=tuple(
                _substitute_name_expr(item, name=name, replacement=replacement)
                for item in expr.items
            ),
        )
    if isinstance(expr, AxonExprPipe):
        return replace(
            expr,
            value=_substitute_name_expr(expr.value, name=name, replacement=replacement),
            stages=tuple(
                _substitute_name_expr(stage, name=name, replacement=replacement)
                for stage in expr.stages
            ),
        )
    if isinstance(expr, AxonExprDo):
        return replace(
            expr,
            body=tuple(_substitute_name_stmt(stmt, name=name, replacement=replacement) for stmt in expr.body),
        )
    return expr


def _substitute_name_stmt(stmt: AxonStatement, *, name: str, replacement: AxonExpr) -> AxonStatement:
    if isinstance(stmt, AxonBind):
        expr = _substitute_name_expr(stmt.expr, name=name, replacement=replacement)
        if name in stmt.targets:
            return replace(stmt, expr=expr)
        return replace(stmt, expr=expr)
    if isinstance(stmt, AxonReturn | AxonYield):
        return replace(
            stmt,
            values=tuple(
                _substitute_name_expr(value, name=name, replacement=replacement) for value in stmt.values
            ),
        )
    if isinstance(stmt, AxonCond):
        return replace(
            stmt,
            cond=_substitute_name_expr(stmt.cond, name=name, replacement=replacement),
            true_body=tuple(
                _substitute_name_stmt(inner, name=name, replacement=replacement)
                for inner in stmt.true_body
            ),
            false_body=tuple(
                _substitute_name_stmt(inner, name=name, replacement=replacement)
                for inner in stmt.false_body
            ),
        )
    if isinstance(stmt, AxonRepeat):
        if stmt.var == name or name in (stmt.targets or ()) or name in (stmt.carry or ()):
            return stmt
        return replace(
            stmt,
            from_expr=_substitute_name_expr(stmt.from_expr, name=name, replacement=replacement),
            to_expr=_substitute_name_expr(stmt.to_expr, name=name, replacement=replacement),
            step_expr=_substitute_name_expr(stmt.step_expr, name=name, replacement=replacement),
            body=tuple(_substitute_name_stmt(inner, name=name, replacement=replacement) for inner in stmt.body),
        )
    if isinstance(stmt, AxonScopeBind):
        if name in stmt.targets:
            return stmt
        return replace(
            stmt,
            kwargs={
                key: (
                    _substitute_name_expr(raw_value, name=name, replacement=replacement)
                    if isinstance(raw_value, AxonExpr)
                    else raw_value
                )
                for key, raw_value in stmt.kwargs.items()
            },
            body=tuple(_substitute_name_stmt(inner, name=name, replacement=replacement) for inner in stmt.body),
        )
    return stmt


def _reduce_do_expr(
    expr: AxonExprDo,
    ctx: _FlattenCtx,
    *,
    path_prefix: tuple[str, ...],
    program_ctx: _FlattenProgramCtx,
) -> tuple[list[AxonStatement], AxonExpr]:
    body = _flatten_statements(expr.body, ctx, path_prefix=path_prefix, program_ctx=program_ctx)
    if not body:
        return [], AxonExprDo(body=body, inline=expr.inline)
    *prelude, last = body
    if isinstance(last, AxonReturn) and len(last.values) == 1:
        return list(prelude), last.values[0]
    return [], AxonExprDo(body=body, inline=expr.inline)


def _flatten_terminal_expr(
    expr: AxonExpr,
    ctx: _FlattenCtx,
    *,
    path_prefix: tuple[str, ...],
    final_ctor: type[AxonReturn] | type[AxonYield],
    program_ctx: _FlattenProgramCtx,
) -> list[AxonStatement]:
    prelude, atom = _ensure_atom(
        expr,
        ctx,
        path_prefix=path_prefix,
        program_ctx=program_ctx,
    )
    return [*prelude, final_ctor(values=(atom,))]


def _pipe_stage_to_call(value: AxonExpr, stage: AxonExpr) -> AxonExpr:
    if isinstance(stage, AxonExprName):
        return AxonExprCall(callee=stage.name, args=(value,), kwargs={})
    if isinstance(stage, AxonExprCall):
        return AxonExprCall(
            callee=stage.callee, args=(value, *stage.args), kwargs=dict(stage.kwargs)
        )
    raise ValueError("flatten failed: pipeline stage must be a name or call")


def _split_callee_path_sugar(callee: str) -> tuple[str, tuple[AxonExprPath, ...]]:
    if "@" not in callee:
        return callee, ()
    base, rest = callee.split("@", 1)
    path_args: list[AxonExprPath] = []
    while rest:
        absolute = rest.startswith("@")
        suffix_start = 1 if rest.startswith("@") else 0
        next_sep = rest.find("@", suffix_start)
        suffix = rest[suffix_start:] if next_sep < 0 else rest[suffix_start:next_sep]
        if not suffix:
            raise ValueError(f"flatten failed: invalid callee path sugar {callee!r}")
        path_args.append(AxonExprPath(absolute=absolute, parts=tuple(suffix.split("."))))
        if next_sep < 0:
            break
        rest = rest[next_sep + 1 :]
    return base, tuple(path_args)


def _leading_path_param_count(module: AxonDefinition) -> int:
    count = 0
    for param in module.params:
        if isinstance(param.type_expr, TypePath):
            count += 1
            continue
        break
    return count


def _module_declares_path_inputs(module: AxonDefinition) -> bool:
    return (
        bool(module.path_params)
        or module.path_param is not None
        or _leading_path_param_count(module) > 0
    )


def _ensure_elaborated_input(program: AxonFile) -> None:
    for module in program.modules:
        for param in module.params:
            if param.default_expr is not None:
                raise ValueError(
                    "flatten requires elaborated Axon input; "
                    f"parameter {module.name}.{param.name} still has a default"
                )


def _make_path_param(name: str) -> AxonParam:
    return AxonParam(name=name, type_expr=TypePath())


def _fresh_scope_param_name(used_names: set[str], base: str) -> str:
    if base not in used_names:
        return base
    idx = 1
    while True:
        candidate = f"{base}_{idx}"
        if candidate not in used_names:
            return candidate
        idx += 1


def _expand_call_surface(
    expr: AxonExprCall,
    program_ctx: _FlattenProgramCtx,
) -> AxonExprCall:
    base_callee, sugared_path_args = _split_callee_path_sugar(expr.callee)
    if sugared_path_args:
        raise ValueError(
            f"flatten failed: callee path sugar remains after normalize: {expr.callee!r}"
        )
    module = program_ctx.modules_by_name.get(base_callee)
    if module is None:
        return expr

    path_slot_count = len(module.path_params) + _leading_path_param_count(module)
    if len(expr.args) < path_slot_count:
        raise ValueError(
            f"flatten failed: explicit path args are missing after normalize in call {expr.callee!r}"
        )
    return expr


def _loop_scope_parts(stmt: AxonRepeat) -> tuple[str, ...]:
    if not stmt.name:
        return ()
    return (*tuple(part for part in stmt.name.split(".") if part), f"{{{stmt.var}}}")


def _normalize_path_expr(
    expr: AxonExprPath,
    *,
    path_prefix: tuple[str, ...],
    path_names: tuple[str, ...] = (),
) -> AxonExprPath:
    if expr.absolute:
        return expr
    if len(expr.parts) == 1 and expr.parts[0] in path_names:
        return AxonExprPath(absolute=True, parts=(f"{{{expr.parts[0]}}}",))
    return absolutize_path_expr(expr, prefix=path_prefix)


def _substitute_alias_dim(
    dim: DimToken, *, subst: dict[str, DimToken | tuple[DimToken, ...]]
) -> tuple[DimToken, ...]:
    if isinstance(dim, str):
        mapped = subst.get(dim)
        if mapped is None:
            return (dim,)
        if isinstance(mapped, tuple):
            return mapped
        return (mapped,)
    if isinstance(dim, int):
        return (dim,)
    left = _substitute_alias_dim(dim.left, subst=subst)
    right = _substitute_alias_dim(dim.right, subst=subst)
    if len(left) == 1 and len(right) == 1:
        return (
            DimExprBinary(
                op=dim.op,
                left=left[0],
                right=right[0],
            ),
        )
    raise ValueError("variadic type alias dimension cannot appear inside dimension arithmetic")


def _match_type_alias_dims(
    params: tuple[str, ...], args: tuple[DimToken, ...]
) -> dict[str, DimToken | tuple[DimToken, ...]] | None:
    variadic_idx = next((idx for idx, param in enumerate(params) if param.startswith("..")), None)
    if variadic_idx is None:
        if len(args) != len(params):
            return None
        return {name: dim for name, dim in zip(params, args, strict=True)}
    fixed_after = len(params) - variadic_idx - 1
    if len(args) < variadic_idx + fixed_after:
        return None
    subst: dict[str, DimToken | tuple[DimToken, ...]] = {}
    for name, dim in zip(params[:variadic_idx], args[:variadic_idx], strict=True):
        subst[name] = dim
    variadic_end = len(args) - fixed_after
    subst[params[variadic_idx]] = tuple(args[variadic_idx:variadic_end])
    if fixed_after:
        for name, dim in zip(params[variadic_idx + 1 :], args[variadic_end:], strict=True):
            subst[name] = dim
    return subst


def _expand_type_aliases(
    tp: TypeExpr | None, *, type_aliases: dict[str, TypeAliasDef]
) -> TypeExpr | None:
    if tp is None:
        return None
    if isinstance(tp, TypeOptional):
        inner = _expand_type_aliases(tp.inner, type_aliases=type_aliases)
        assert inner is not None
        return TypeOptional(inner=inner)
    if isinstance(tp, TypeList):
        item = _expand_type_aliases(tp.item, type_aliases=type_aliases)
        assert item is not None
        return TypeList(item=item)
    if isinstance(tp, TypeTuple):
        return TypeTuple(
            items=tuple(
                expanded
                for expanded in (
                    _expand_type_aliases(item, type_aliases=type_aliases) for item in tp.items
                )
                if expanded is not None
            )
        )
    if isinstance(tp, TypeTensor):
        return TypeTensor(base=tp.base, dims=tp.dims)
    if isinstance(tp, TypeNamed):
        alias = type_aliases.get(tp.name)
        if alias is None and "." in tp.name:
            alias = type_aliases.get(tp.name.rsplit(".", 1)[1])
        if alias is None:
            return TypeNamed(name=tp.name, args=tp.args)
        subst = _match_type_alias_dims(alias.params, tp.args)
        if subst is None:
            raise ValueError(
                f"flatten failed: type alias {tp.name!r} expects {len(alias.params)} args, got {len(tp.args)}"
            )
        expanded = _expand_type_aliases(alias.value, type_aliases=type_aliases)
        assert expanded is not None
        return _substitute_type_alias_dims(expanded, subst=subst)
    return tp


def _substitute_type_alias_dims(
    tp: TypeExpr, *, subst: dict[str, DimToken | tuple[DimToken, ...]]
) -> TypeExpr:
    if isinstance(tp, TypeOptional):
        return TypeOptional(inner=_substitute_type_alias_dims(tp.inner, subst=subst))
    if isinstance(tp, TypeList):
        return TypeList(item=_substitute_type_alias_dims(tp.item, subst=subst))
    if isinstance(tp, TypeTuple):
        return TypeTuple(
            items=tuple(_substitute_type_alias_dims(item, subst=subst) for item in tp.items)
        )
    if isinstance(tp, TypeTensor):
        return TypeTensor(
            base=tp.base,
            dims=tuple(
                item
                for dim in tp.dims
                for item in _substitute_alias_dim(dim, subst=subst)
            ),
        )
    if isinstance(tp, TypeNamed):
        return TypeNamed(
            name=tp.name,
            args=tuple(
                item
                for dim in tp.args
                for item in _substitute_alias_dim(dim, subst=subst)
            ),
        )
    return tp


def _expand_expr_aliases(expr: AxonExpr, *, type_aliases: dict[str, TypeAliasDef]) -> AxonExpr:
    if isinstance(expr, AxonExprAscribe):
        expanded_type = _expand_type_aliases(expr.type_expr, type_aliases=type_aliases)
        assert expanded_type is not None
        return AxonExprAscribe(
            expr=_expand_expr_aliases(expr.expr, type_aliases=type_aliases),
            type_expr=expanded_type,
            inferred_type=expr.inferred_type,
            inferred_arity=expr.inferred_arity,
            inferred_dims=expr.inferred_dims,
        )
    if isinstance(expr, AxonExprCall):
        return AxonExprCall(
            callee=expr.callee,
            args=tuple(_expand_expr_aliases(arg, type_aliases=type_aliases) for arg in expr.args),
            kwargs={
                key: _expand_expr_aliases(value, type_aliases=type_aliases)
                if isinstance(value, AxonExpr)
                else value
                for key, value in expr.kwargs.items()
            },
            inferred_type=expr.inferred_type,
            inferred_arity=expr.inferred_arity,
            inferred_dims=expr.inferred_dims,
        )
    if isinstance(expr, AxonExprBinary):
        return AxonExprBinary(
            op=expr.op,
            left=_expand_expr_aliases(expr.left, type_aliases=type_aliases),
            right=_expand_expr_aliases(expr.right, type_aliases=type_aliases),
            inferred_type=expr.inferred_type,
            inferred_arity=expr.inferred_arity,
            inferred_dims=expr.inferred_dims,
        )
    if isinstance(expr, AxonExprTernary):
        return AxonExprTernary(
            cond=_expand_expr_aliases(expr.cond, type_aliases=type_aliases),
            true_expr=_expand_expr_aliases(expr.true_expr, type_aliases=type_aliases),
            false_expr=_expand_expr_aliases(expr.false_expr, type_aliases=type_aliases),
            inferred_type=expr.inferred_type,
            inferred_arity=expr.inferred_arity,
            inferred_dims=expr.inferred_dims,
        )
    if isinstance(expr, AxonExprList):
        return AxonExprList(
            items=tuple(
                _expand_expr_aliases(item, type_aliases=type_aliases) for item in expr.items
            ),
            inferred_type=expr.inferred_type,
            inferred_arity=expr.inferred_arity,
            inferred_dims=expr.inferred_dims,
        )
    if isinstance(expr, AxonExprTuple):
        return AxonExprTuple(
            items=tuple(
                _expand_expr_aliases(item, type_aliases=type_aliases) for item in expr.items
            ),
            inferred_type=expr.inferred_type,
            inferred_arity=expr.inferred_arity,
            inferred_dims=expr.inferred_dims,
        )
    if isinstance(expr, AxonExprParen):
        return AxonExprParen(
            inner=_expand_expr_aliases(expr.inner, type_aliases=type_aliases),
            inferred_type=expr.inferred_type,
            inferred_arity=expr.inferred_arity,
            inferred_dims=expr.inferred_dims,
        )
    if isinstance(expr, AxonExprDo):
        return AxonExprDo(
            body=tuple(_expand_stmt_aliases(stmt, type_aliases=type_aliases) for stmt in expr.body),
            inline=expr.inline,
            inferred_type=expr.inferred_type,
            inferred_arity=expr.inferred_arity,
            inferred_dims=expr.inferred_dims,
        )
    if isinstance(expr, AxonExprBind):
        return AxonExprBind(
            value=_expand_expr_aliases(expr.value, type_aliases=type_aliases),
            var=expr.var,
            body=_expand_expr_aliases(expr.body, type_aliases=type_aliases),
            inferred_type=expr.inferred_type,
            inferred_arity=expr.inferred_arity,
            inferred_dims=expr.inferred_dims,
        )
    if isinstance(expr, AxonExprIf):
        return AxonExprIf(
            cond=_expand_expr_aliases(expr.cond, type_aliases=type_aliases),
            true_expr=_expand_expr_aliases(expr.true_expr, type_aliases=type_aliases),
            false_expr=_expand_expr_aliases(expr.false_expr, type_aliases=type_aliases),
            inferred_type=expr.inferred_type,
            inferred_arity=expr.inferred_arity,
            inferred_dims=expr.inferred_dims,
        )
    if isinstance(expr, AxonExprLambda):
        return AxonExprLambda(
            var=expr.var,
            body=_expand_expr_aliases(expr.body, type_aliases=type_aliases),
            inferred_type=expr.inferred_type,
            inferred_arity=expr.inferred_arity,
            inferred_dims=expr.inferred_dims,
        )
    if isinstance(expr, AxonExprPipe):
        return AxonExprPipe(
            value=_expand_expr_aliases(expr.value, type_aliases=type_aliases),
            stages=tuple(
                _expand_expr_aliases(stage, type_aliases=type_aliases) for stage in expr.stages
            ),
            inferred_type=expr.inferred_type,
            inferred_arity=expr.inferred_arity,
            inferred_dims=expr.inferred_dims,
        )
    return expr


def _expand_stmt_aliases(
    stmt: AxonStatement, *, type_aliases: dict[str, TypeAliasDef]
) -> AxonStatement:
    if isinstance(stmt, AxonBind):
        return AxonBind(
            targets=stmt.targets, expr=_expand_expr_aliases(stmt.expr, type_aliases=type_aliases)
        )
    if isinstance(stmt, AxonReturn):
        return AxonReturn(
            values=tuple(
                _expand_expr_aliases(value, type_aliases=type_aliases) for value in stmt.values
            )
        )
    if isinstance(stmt, AxonYield):
        return AxonYield(
            values=tuple(
                _expand_expr_aliases(value, type_aliases=type_aliases) for value in stmt.values
            )
        )
    if isinstance(stmt, AxonCond):
        return AxonCond(
            cond=_expand_expr_aliases(stmt.cond, type_aliases=type_aliases),
            true_body=tuple(
                _expand_stmt_aliases(item, type_aliases=type_aliases) for item in stmt.true_body
            ),
            false_body=tuple(
                _expand_stmt_aliases(item, type_aliases=type_aliases) for item in stmt.false_body
            ),
        )
    if isinstance(stmt, AxonRepeat):
        return AxonRepeat(
            name=stmt.name,
            var=stmt.var,
            to_expr=_expand_expr_aliases(stmt.to_expr, type_aliases=type_aliases),
            from_expr=_expand_expr_aliases(stmt.from_expr, type_aliases=type_aliases),
            step_expr=_expand_expr_aliases(stmt.step_expr, type_aliases=type_aliases),
            body=tuple(_expand_stmt_aliases(item, type_aliases=type_aliases) for item in stmt.body),
            targets=stmt.targets,
            carry=stmt.carry,
        )
    if isinstance(stmt, AxonScopeBind):
        return AxonScopeBind(
            targets=stmt.targets,
            prefix=stmt.prefix,
            body=tuple(_expand_stmt_aliases(item, type_aliases=type_aliases) for item in stmt.body),
            kwargs={
                key: _expand_expr_aliases(value, type_aliases=type_aliases)
                if isinstance(value, AxonExpr)
                else value
                for key, value in stmt.kwargs.items()
            },
        )
    return stmt


def _expand_definition_aliases(
    module: AxonDefinition,
    *,
    type_aliases: dict[str, TypeAliasDef],
) -> AxonDefinition:
    return replace(
        module,
        params=tuple(
            replace(param, type_expr=_expand_type_aliases(param.type_expr, type_aliases=type_aliases))
            for param in module.params
        ),
        statements=tuple(_expand_stmt_aliases(stmt, type_aliases=type_aliases) for stmt in module.statements),
        return_type_expr=_expand_type_aliases(module.return_type_expr, type_aliases=type_aliases),
        type_aliases=None,
    )


def _absolutize_call_relative_paths(
    *,
    callee: str,
    args: list[AxonExpr],
    kwargs: dict[str, AxonKwargValue],
    path_prefix: tuple[str, ...],
    program_ctx: _FlattenProgramCtx,
) -> None:
    module = program_ctx.modules_by_name.get(callee)
    if module is None:
        return
    path_slot_count = len(module.path_params) + _leading_path_param_count(module)
    if path_slot_count <= 0 and callee in program_ctx.scoped_modules and args:
        path_slot_count = 1
    if path_slot_count <= 0 or not args:
        base_path = None
    else:
        base_path = next(
            (arg for arg in args[:path_slot_count] if isinstance(arg, AxonExprPath) and arg.absolute),
            None,
        )
    if base_path is None and not path_prefix:
        return
    for key, raw_value in list(kwargs.items()):
        if isinstance(raw_value, AxonExprPath):
            if raw_value.absolute:
                continue
            kwargs[key] = absolutize_path_expr(raw_value, prefix=path_prefix)
            continue


def _expr_name_uses(expr: AxonExpr) -> list[str]:
    names: list[str] = []

    def visit(current: AxonExpr) -> None:
        if isinstance(current, AxonExprName):
            names.append(current.name)
        elif isinstance(current, AxonExprCall):
            for arg in current.args:
                visit(arg)
            for value in current.kwargs.values():
                if isinstance(value, AxonExpr):
                    visit(value)
        elif isinstance(current, AxonExprBinary):
            visit(current.left)
            visit(current.right)
        elif isinstance(current, AxonExprIf | AxonExprTernary):
            visit(current.cond)
            visit(current.true_expr)
            visit(current.false_expr)
        elif isinstance(current, AxonExprBind):
            visit(current.value)
            visit(current.body)
        elif isinstance(current, AxonExprLambda):
            visit(current.body)
        elif isinstance(current, AxonExprParen):
            visit(current.inner)
        elif isinstance(current, AxonExprAscribe):
            visit(current.expr)
        elif isinstance(current, AxonExprDo):
            for stmt in current.body:
                visit_stmt(stmt)
        elif isinstance(current, AxonExprList | AxonExprTuple):
            for item in current.items:
                visit(item)

    def visit_stmt(stmt: AxonStatement) -> None:
        if isinstance(stmt, AxonBind):
            visit(stmt.expr)
        elif isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                visit(value)
        elif isinstance(stmt, AxonCond):
            visit(stmt.cond)
            for inner in stmt.true_body:
                visit_stmt(inner)
            for inner in stmt.false_body:
                visit_stmt(inner)
        elif isinstance(stmt, AxonRepeat):
            visit(stmt.from_expr)
            visit(stmt.to_expr)
            visit(stmt.step_expr)
            for inner in stmt.body:
                visit_stmt(inner)
        elif isinstance(stmt, AxonScopeBind):
            for raw_value in stmt.kwargs.values():
                kw_value: AxonKwargValue = raw_value
                if isinstance(kw_value, AxonExpr):
                    visit(kw_value)
            for inner in stmt.body:
                visit_stmt(inner)

    visit(expr)
    return names


def _statement_name_uses(stmt: AxonStatement) -> list[str]:
    names: list[str] = []
    if isinstance(stmt, AxonBind):
        names.extend(_expr_name_uses(stmt.expr))
    elif isinstance(stmt, AxonReturn | AxonYield):
        for value in stmt.values:
            names.extend(_expr_name_uses(value))
    elif isinstance(stmt, AxonCond):
        names.extend(_expr_name_uses(stmt.cond))
        for inner in stmt.true_body:
            names.extend(_statement_name_uses(inner))
        for inner in stmt.false_body:
            names.extend(_statement_name_uses(inner))
    elif isinstance(stmt, AxonRepeat):
        names.extend(_expr_name_uses(stmt.from_expr))
        names.extend(_expr_name_uses(stmt.to_expr))
        names.extend(_expr_name_uses(stmt.step_expr))
        for inner in stmt.body:
            names.extend(_statement_name_uses(inner))
    elif isinstance(stmt, AxonScopeBind):
        for raw_value in stmt.kwargs.values():
            kw_value: AxonKwargValue = raw_value
            if isinstance(kw_value, AxonExpr):
                names.extend(_expr_name_uses(kw_value))
        for inner in stmt.body:
            names.extend(_statement_name_uses(inner))
    return names


def _normalize_repeat_yield(stmt: AxonRepeat) -> AxonRepeat:
    if stmt.body and isinstance(stmt.body[-1], AxonYield):
        return stmt
    raise ValueError("flatten failed: repeat/for loop was not yield-normalized before flatten")


def _repeat_yield_expr(stmt: AxonRepeat) -> AxonExpr:
    if not stmt.body or not isinstance(stmt.body[-1], AxonYield):
        raise ValueError("flatten failed: normalized repeat body must end in yield")
    values = stmt.body[-1].values
    if len(values) == 1:
        return values[0]
    return AxonExprTuple(items=values)


def _extract_repeat_recursive_helper(
    stmt: AxonRepeat,
    *,
    module_name: str,
    ctx: _FlattenCtx,
    globals_by_name: set[str],
) -> tuple[list[AxonStatement], tuple[AxonDefinition, ...]]:
    normalized = _normalize_repeat_yield(stmt)
    last_stmt = normalized.body[-1]
    assert isinstance(last_stmt, AxonYield)
    helper_prefix = f"{module_name}__loop_{normalized.name or normalized.var}_recur"
    helper_name = ctx.fresh_helper(prefix=helper_prefix)
    continue_helper_name = ctx.fresh_helper(prefix=f"{helper_prefix}_continue")

    carry_names = tuple(normalized.carry or ())
    local_bound = {normalized.var, *carry_names, *ctx.module_path_params}
    free_names: list[str] = []
    seen_free: set[str] = set()
    for inner in normalized.body:
        for name in _statement_name_uses(inner):
            if name in local_bound or name in globals_by_name or name in seen_free:
                continue
            seen_free.add(name)
            free_names.append(name)
        if isinstance(inner, AxonBind):
            local_bound.update(name for name in inner.targets if name != "_")
        elif isinstance(inner, AxonRepeat):
            local_bound.add(inner.var)
            local_bound.update(name for name in (inner.targets or ()) if name != "_")
            local_bound.update(name for name in (inner.carry or ()) if name != "_")

    to_name = ctx.fresh(prefix="__loop_to")
    step_name = ctx.fresh(prefix="__loop_step")
    carry_param_types = tuple(ctx.fresh_type_var() for _ in carry_names)
    free_param_types = tuple(ctx.fresh_type_var() for _ in free_names)
    recur_params = (
        AxonParam(name=normalized.var, type_expr=TypeInt()),
        AxonParam(name=to_name, type_expr=TypeInt()),
        AxonParam(name=step_name, type_expr=TypeInt()),
        *(
            AxonParam(name=name, type_expr=tp)
            for name, tp in zip(carry_names, carry_param_types, strict=True)
        ),
        *(
            AxonParam(name=name, type_expr=tp)
            for name, tp in zip(free_names, free_param_types, strict=True)
        ),
    )

    step_positive = AxonExprBinary(
        op=">", left=AxonExprName(name=step_name), right=AxonExprInt(value=0)
    )
    done_pos = AxonExprBinary(
        op=">=", left=AxonExprName(name=normalized.var), right=AxonExprName(name=to_name)
    )
    done_neg = AxonExprBinary(
        op="<=", left=AxonExprName(name=normalized.var), right=AxonExprName(name=to_name)
    )
    step_positive_name = ctx.fresh(prefix="__loop_step_pos")
    done_pos_name = ctx.fresh(prefix="__loop_done_pos")
    done_neg_name = ctx.fresh(prefix="__loop_done_neg")
    done_name = ctx.fresh(prefix="__loop_done")

    true_values: tuple[AxonExpr, ...]
    if carry_names:
        true_values = tuple(AxonExprName(name=name) for name in carry_names)
    else:
        true_values = (AxonExprNull(),)

    false_body: list[AxonStatement] = list(normalized.body[:-1])
    yielded_expr = _repeat_yield_expr(normalized)
    if carry_names:
        false_body.append(AxonBind(targets=carry_names, expr=yielded_expr))
    else:
        ignore_targets = (
            ("_",) if len(last_stmt.values) == 1 else tuple("_" for _ in last_stmt.values)
        )
        false_body.append(AxonBind(targets=ignore_targets, expr=yielded_expr))
    next_i = ctx.fresh(prefix="__loop_i")
    false_body.append(
        AxonBind(
            targets=(next_i,),
            expr=AxonExprBinary(
                op="+",
                left=AxonExprName(name=normalized.var),
                right=AxonExprName(name=step_name),
            ),
        )
    )
    recur_args = (
        AxonExprName(name=next_i),
        AxonExprName(name=to_name),
        AxonExprName(name=step_name),
        *(AxonExprName(name=name) for name in carry_names),
        *(AxonExprName(name=name) for name in free_names),
    )
    recur_call_expr = AxonExprCall(callee=helper_name, args=recur_args, kwargs={})
    if ctx.module_path_params:
        recur_call_expr = replace(
            recur_call_expr,
            args=tuple(AxonExprName(name=name) for name in ctx.module_path_params)
            + recur_call_expr.args,
        )
    if carry_names:
        false_body.append(AxonBind(targets=carry_names, expr=recur_call_expr))
        false_body.append(AxonReturn(values=tuple(AxonExprName(name=name) for name in carry_names)))
    else:
        recur_temp = ctx.fresh(prefix="__loop_result")
        false_body.append(AxonBind(targets=(recur_temp,), expr=recur_call_expr))
        false_body.append(AxonReturn(values=(AxonExprName(name=recur_temp),)))

    continue_helper_module = AxonDefinition(
        name=continue_helper_name,
        path_param=None,
        params=tuple(_make_path_param(name) for name in ctx.module_path_params)
        + tuple(recur_params),
        returns=(),
        statements=tuple(false_body),
        body_expr=None,
        path_params=(),
        imports=(),
        imported_members=None,
        exports=(),
        symbols=None,
        pragmas=None,
        type_aliases=None,
        return_type_expr=(
            carry_param_types[0]
            if len(carry_param_types) == 1
            else TypeTuple(items=carry_param_types)
        )
        if carry_param_types
        else TypeNull(),
        constraints=None,
    )

    true_expr: AxonExpr
    if len(true_values) == 1:
        true_expr = true_values[0]
    else:
        true_expr = AxonExprTuple(items=true_values)
    false_expr = AxonExprCall(
        callee=continue_helper_name,
        args=(
            AxonExprName(name=normalized.var),
            AxonExprName(name=to_name),
            AxonExprName(name=step_name),
            *(AxonExprName(name=name) for name in carry_names),
            *(AxonExprName(name=name) for name in free_names),
        ),
        kwargs={},
    )
    if ctx.module_path_params:
        false_expr = replace(
            false_expr,
            args=tuple(AxonExprName(name=name) for name in ctx.module_path_params)
            + false_expr.args,
        )

    helper_result_expr = AxonExprTernary(
        cond=AxonExprName(name=done_name),
        true_expr=true_expr,
        false_expr=false_expr,
    )

    helper_body: list[AxonStatement] = [
        AxonBind(targets=(step_positive_name,), expr=step_positive),
        AxonBind(targets=(done_pos_name,), expr=done_pos),
        AxonBind(targets=(done_neg_name,), expr=done_neg),
        AxonBind(
            targets=(done_name,),
            expr=AxonExprTernary(
                cond=AxonExprName(name=step_positive_name),
                true_expr=AxonExprName(name=done_pos_name),
                false_expr=AxonExprName(name=done_neg_name),
            ),
        ),
    ]
    if carry_names:
        helper_body.append(AxonBind(targets=carry_names, expr=helper_result_expr))
        helper_body.append(
            AxonReturn(values=tuple(AxonExprName(name=name) for name in carry_names))
        )
    else:
        helper_result_name = ctx.fresh(prefix="__loop_result")
        helper_body.append(AxonBind(targets=(helper_result_name,), expr=helper_result_expr))
        helper_body.append(AxonReturn(values=(AxonExprName(name=helper_result_name),)))

    helper_module = AxonDefinition(
        name=helper_name,
        path_param=None,
        params=tuple(_make_path_param(name) for name in ctx.module_path_params)
        + tuple(recur_params),
        returns=(),
        statements=tuple(helper_body),
        body_expr=None,
        path_params=(),
        imports=(),
        imported_members=None,
        exports=(),
        symbols=None,
        pragmas=None,
        type_aliases=None,
        return_type_expr=(
            carry_param_types[0]
            if len(carry_param_types) == 1
            else TypeTuple(items=carry_param_types)
        )
        if carry_param_types
        else TypeNull(),
        constraints=None,
    )

    outer_targets = (
        stmt.targets
        if stmt.targets is not None
        else (stmt.carry if stmt.carry is not None else ("_",))
    )
    initial_call = AxonExprCall(
        callee=helper_name,
        args=(
            normalized.from_expr,
            normalized.to_expr,
            normalized.step_expr,
            *(AxonExprName(name=name) for name in carry_names),
            *(AxonExprName(name=name) for name in free_names),
        ),
        kwargs={},
    )
    if ctx.module_path_params:
        initial_call = replace(
            initial_call,
            args=tuple(AxonExprName(name=name) for name in ctx.module_path_params)
            + initial_call.args,
        )
    replacement: list[AxonStatement] = [AxonBind(targets=tuple(outer_targets), expr=initial_call)]
    return replacement, (continue_helper_module, helper_module)


def _extract_repeat_helper(
    stmt: AxonRepeat,
    *,
    module_name: str,
    ctx: _FlattenCtx,
    globals_by_name: set[str],
) -> tuple[AxonRepeat, AxonDefinition] | None:
    normalized = _normalize_repeat_yield(stmt)
    if not normalized.body or not isinstance(normalized.body[-1], AxonYield):
        return None
    if (
        normalized.carry is not None
        and len(normalized.body) == 1
        and len(normalized.body[-1].values) == 1
        and isinstance(normalized.body[-1].values[0], AxonExprCall)
    ):
        return None

    helper_prefix = f"{module_name}__loop_{normalized.name or normalized.var}_step"
    helper_name = ctx.fresh_helper(prefix=helper_prefix)

    carry_names = tuple(normalized.carry or ())
    local_bound = {normalized.var, *carry_names, *ctx.module_path_params}
    free_names: list[str] = []
    seen_free: set[str] = set()

    for inner in normalized.body:
        for name in _statement_name_uses(inner):
            if name in local_bound or name in globals_by_name or name in seen_free:
                continue
            seen_free.add(name)
            free_names.append(name)
        if isinstance(inner, AxonBind):
            local_bound.update(name for name in inner.targets if name != "_")
        elif isinstance(inner, AxonRepeat):
            local_bound.add(inner.var)
            local_bound.update(name for name in (inner.targets or ()) if name != "_")
            local_bound.update(name for name in (inner.carry or ()) if name != "_")
        elif isinstance(inner, AxonScopeBind):
            local_bound.update(name for name in inner.targets if name != "_")

    loop_var_type = TypeInt()
    carry_param_types = tuple(ctx.fresh_type_var() for _ in carry_names)
    free_param_types = tuple(ctx.fresh_type_var() for _ in free_names)
    helper_params = (
        AxonParam(name=normalized.var, type_expr=loop_var_type),
        *(
            AxonParam(name=name, type_expr=tp)
            for name, tp in zip(carry_names, carry_param_types, strict=True)
        ),
        *(
            AxonParam(name=name, type_expr=tp)
            for name, tp in zip(free_names, free_param_types, strict=True)
        ),
    )

    helper_body: list[AxonStatement] = []
    for inner in normalized.body[:-1]:
        helper_body.append(inner)
    helper_body.append(AxonReturn(values=normalized.body[-1].values))

    helper_module = AxonDefinition(
        name=helper_name,
        path_param=None,
        params=tuple(_make_path_param(name) for name in ctx.module_path_params)
        + tuple(helper_params),
        returns=(),
        statements=tuple(helper_body),
        body_expr=None,
        path_params=(),
        imports=(),
        imported_members=None,
        exports=(),
        symbols=None,
        pragmas=None,
        type_aliases=None,
        return_type_expr=None,
        constraints=None,
    )

    helper_args = (
        *(AxonExprName(name=name) for name in ctx.module_path_params),
        AxonExprName(name=normalized.var),
        *(AxonExprName(name=name) for name in carry_names),
        *(AxonExprName(name=name) for name in free_names),
    )
    rewritten_repeat = AxonRepeat(
        # The loop scope has already been threaded into path expressions while
        # flattening the body. Flat Axon keeps structured loops but not scope
        # sugar.
        name=None,
        var=normalized.var,
        to_expr=normalized.to_expr,
        from_expr=normalized.from_expr,
        step_expr=normalized.step_expr,
        body=(AxonYield(values=(AxonExprCall(callee=helper_name, args=helper_args, kwargs={}),)),),
        targets=normalized.targets,
        carry=normalized.carry,
    )
    return rewritten_repeat, helper_module


def _extract_repeat_helpers_from_statements(
    statements: tuple[AxonStatement, ...],
    *,
    module_name: str,
    ctx: _FlattenCtx,
    globals_by_name: set[str],
) -> tuple[tuple[AxonStatement, ...], tuple[AxonDefinition, ...]]:
    out: list[AxonStatement] = []
    helpers: list[AxonDefinition] = []
    for stmt in statements:
        if isinstance(stmt, AxonCond):
            true_body, true_helpers = _extract_repeat_helpers_from_statements(
                stmt.true_body,
                module_name=module_name,
                ctx=ctx,
                globals_by_name=globals_by_name,
            )
            false_body, false_helpers = _extract_repeat_helpers_from_statements(
                stmt.false_body,
                module_name=module_name,
                ctx=ctx,
                globals_by_name=globals_by_name,
            )
            out.append(AxonCond(cond=stmt.cond, true_body=true_body, false_body=false_body))
            helpers.extend(true_helpers)
            helpers.extend(false_helpers)
            continue
        if isinstance(stmt, AxonScopeBind):
            body, nested_helpers = _extract_repeat_helpers_from_statements(
                stmt.body,
                module_name=module_name,
                ctx=ctx,
                globals_by_name=globals_by_name,
            )
            out.append(
                AxonScopeBind(
                    targets=stmt.targets, prefix=stmt.prefix, body=body, kwargs=stmt.kwargs
                )
            )
            helpers.extend(nested_helpers)
            continue
        if isinstance(stmt, AxonRepeat):
            body, nested_helpers = _extract_repeat_helpers_from_statements(
                stmt.body,
                module_name=module_name,
                ctx=ctx,
                globals_by_name=globals_by_name,
            )
            normalized_repeat = AxonRepeat(
                name=stmt.name,
                var=stmt.var,
                to_expr=stmt.to_expr,
                from_expr=stmt.from_expr,
                step_expr=stmt.step_expr,
                body=body,
                targets=stmt.targets,
                carry=stmt.carry,
            )
            extracted = _extract_repeat_helper(
                normalized_repeat,
                module_name=module_name,
                ctx=ctx,
                globals_by_name=globals_by_name,
            )
            if extracted is None:
                out.append(replace(normalized_repeat, name=None))
            else:
                rewritten_repeat, step_helper = extracted
                out.append(rewritten_repeat)
                helpers.append(step_helper)
            helpers.extend(nested_helpers)
            continue
        out.append(stmt)
    return tuple(out), tuple(helpers)


def _extract_cond_target_binding(
    body: tuple[AxonStatement, ...],
) -> tuple[str, tuple[AxonStatement, ...], AxonExpr] | None:
    if not body:
        return None
    last = body[-1]
    if not isinstance(last, AxonBind) or len(last.targets) != 1:
        return None
    target = last.targets[0]
    if target == "_":
        return None
    return target, body[:-1], last.expr


def _extract_cond_terminal_return(
    body: tuple[AxonStatement, ...],
) -> tuple[tuple[AxonStatement, ...], tuple[AxonExpr, ...]] | None:
    if not body:
        return None
    last = body[-1]
    if not isinstance(last, AxonReturn):
        return None
    return body[:-1], last.values


def _extract_cond_branch_expr_payload(
    expr: AxonExpr,
) -> tuple[tuple[AxonStatement, ...], AxonExpr] | None:
    if isinstance(expr, AxonExprDo):
        terminal = _extract_cond_terminal_return(expr.body)
        if terminal is None:
            return None
        prelude, values = terminal
        if len(values) == 1:
            return prelude, values[0]
        return prelude, AxonExprTuple(items=values)
    return (), expr


def _dim_token_from_expr(expr: AxonExpr) -> DimToken | None:
    if isinstance(expr, AxonExprAscribe | AxonExprParen):
        return _dim_token_from_expr(expr.expr if isinstance(expr, AxonExprAscribe) else expr.inner)
    if isinstance(expr, AxonExprInt):
        return expr.value
    if isinstance(expr, AxonExprName):
        return expr.name
    if isinstance(expr, AxonExprBinary):
        left = _dim_token_from_expr(expr.left)
        right = _dim_token_from_expr(expr.right)
        if left is not None and right is not None:
            return DimExprBinary(op=expr.op, left=left, right=right)
    return None


def _collect_type_dim_substitutions(
    formal: TypeExpr,
    actual: TypeExpr,
    out: dict[str, DimToken | tuple[DimToken, ...]],
) -> None:
    if isinstance(formal, TypeOptional):
        if isinstance(actual, TypeOptional):
            _collect_type_dim_substitutions(formal.inner, actual.inner, out)
        else:
            _collect_type_dim_substitutions(formal.inner, actual, out)
        return
    if isinstance(actual, TypeOptional):
        _collect_type_dim_substitutions(formal, actual.inner, out)
        return
    if isinstance(formal, TypeList) and isinstance(actual, TypeList):
        _collect_type_dim_substitutions(formal.item, actual.item, out)
        return
    if isinstance(formal, TypeTuple) and isinstance(actual, TypeTuple):
        for left, right in zip(formal.items, actual.items, strict=False):
            _collect_type_dim_substitutions(left, right, out)
        return
    if isinstance(formal, TypeTensor) and isinstance(actual, TypeTensor):
        variadic_indexes = [
            index
            for index, dim in enumerate(formal.dims)
            if isinstance(dim, str) and dim.startswith("..")
        ]
        if len(variadic_indexes) > 1:
            return
        if variadic_indexes:
            variadic_index = variadic_indexes[0]
            prefix = formal.dims[:variadic_index]
            suffix = formal.dims[variadic_index + 1 :]
            if len(actual.dims) < len(prefix) + len(suffix):
                return
            middle_end = len(actual.dims) - len(suffix) if suffix else len(actual.dims)
            pairs = [
                *zip(prefix, actual.dims[: len(prefix)], strict=False),
                *zip(suffix, actual.dims[-len(suffix) :] if suffix else (), strict=False),
            ]
            out.setdefault(formal.dims[variadic_index], actual.dims[len(prefix) : middle_end])
        else:
            if len(formal.dims) != len(actual.dims):
                return
            pairs = list(zip(formal.dims, actual.dims, strict=True))
        for formal_dim, actual_dim in pairs:
            if isinstance(formal_dim, str) and not formal_dim.startswith(".."):
                out.setdefault(formal_dim, actual_dim)
        return
    if isinstance(formal, TypeNamed) and isinstance(actual, TypeNamed) and formal.name == actual.name:
        variadic_indexes = [
            index
            for index, dim in enumerate(formal.args)
            if isinstance(dim, str) and dim.startswith("..")
        ]
        if len(variadic_indexes) > 1:
            return
        if variadic_indexes:
            variadic_index = variadic_indexes[0]
            prefix = formal.args[:variadic_index]
            suffix = formal.args[variadic_index + 1 :]
            if len(actual.args) < len(prefix) + len(suffix):
                return
            middle_end = len(actual.args) - len(suffix) if suffix else len(actual.args)
            pairs = [
                *zip(prefix, actual.args[: len(prefix)], strict=False),
                *zip(suffix, actual.args[-len(suffix) :] if suffix else (), strict=False),
            ]
            out.setdefault(formal.args[variadic_index], actual.args[len(prefix) : middle_end])
        else:
            if len(formal.args) != len(actual.args):
                return
            pairs = list(zip(formal.args, actual.args, strict=True))
        for formal_dim, actual_dim in pairs:
            if isinstance(formal_dim, str) and not formal_dim.startswith(".."):
                out.setdefault(formal_dim, actual_dim)


def _is_dim_like_type(type_expr: TypeExpr | None) -> bool:
    return isinstance(type_expr, TypeInt) or (
        isinstance(type_expr, TypeNamed) and type_expr.name in {"Dim", "Int"}
    )


def _call_signature_return_type(
    expr: AxonExprCall,
    *,
    known_types: Mapping[str, TypeExpr],
    program_ctx: _FlattenProgramCtx,
) -> TypeExpr | None:
    module = program_ctx.modules_by_name.get(expr.callee)
    if module is None or module.return_type_expr is None:
        return None
    type_aliases = _type_aliases_for_module(program_ctx, module)
    return_type = _expand_type_aliases(
        module.return_type_expr,
        type_aliases=type_aliases,
    )
    if return_type is None:
        return None
    dim_subst: dict[str, DimToken | tuple[DimToken, ...]] = {}
    for param, arg in zip(module.params, expr.args, strict=False):
        if param.type_expr is not None:
            formal_type = _expand_type_aliases(param.type_expr, type_aliases=type_aliases)
            actual_type = _expr_known_type(arg, known_types=known_types, program_ctx=program_ctx)
            if formal_type is not None and actual_type is not None:
                _collect_type_dim_substitutions(formal_type, actual_type, dim_subst)
        if _is_dim_like_type(param.type_expr):
            dim_token = _dim_token_from_expr(arg)
            if dim_token is not None:
                dim_subst.setdefault(param.name, dim_token)
    if dim_subst:
        return_type = _substitute_type_alias_dims(return_type, subst=dim_subst)
    return return_type


def _expr_known_type(
    expr: AxonExpr,
    *,
    known_types: Mapping[str, TypeExpr],
    program_ctx: _FlattenProgramCtx,
) -> TypeExpr | None:
    inferred = getattr(expr, "inferred_type", None)
    if inferred is not None:
        return inferred
    if isinstance(expr, AxonExprAscribe):
        return expr.type_expr
    if isinstance(expr, AxonExprParen):
        return _expr_known_type(expr.inner, known_types=known_types, program_ctx=program_ctx)
    if isinstance(expr, AxonExprName):
        return known_types.get(expr.name)
    if isinstance(expr, AxonExprTuple):
        items = tuple(
            _expr_known_type(item, known_types=known_types, program_ctx=program_ctx)
            for item in expr.items
        )
        if all(item is not None for item in items):
            return TypeTuple(items=items)  # type: ignore[arg-type]
    if isinstance(expr, AxonExprTernary):
        true_type = _expr_known_type(expr.true_expr, known_types=known_types, program_ctx=program_ctx)
        false_type = _expr_known_type(expr.false_expr, known_types=known_types, program_ctx=program_ctx)
        if true_type == false_type:
            return true_type
    if isinstance(expr, AxonExprCall):
        return _call_signature_return_type(expr, known_types=known_types, program_ctx=program_ctx)
    return None


def _bind_known_target_types(
    known_types: Mapping[str, TypeExpr],
    targets: tuple[str, ...],
    expr: AxonExpr,
    *,
    program_ctx: _FlattenProgramCtx,
) -> None:
    if not isinstance(known_types, dict):
        return
    if len(targets) == 1:
        target = targets[0]
        if target == "_":
            return
        expr_type = _expr_known_type(expr, known_types=known_types, program_ctx=program_ctx)
        if expr_type is not None:
            known_types[target] = expr_type
        return
    expr_type = _expr_known_type(expr, known_types=known_types, program_ctx=program_ctx)
    if isinstance(expr_type, TypeTuple) and len(expr_type.items) == len(targets):
        for target, item_type in zip(targets, expr_type.items, strict=True):
            if target != "_":
                known_types[target] = item_type
        return
    if isinstance(expr, AxonExprTuple) and len(expr.items) == len(targets):
        for target, item in zip(targets, expr.items, strict=True):
            if target == "_":
                continue
            item_type = _expr_known_type(item, known_types=known_types, program_ctx=program_ctx)
            if item_type is not None:
                known_types[target] = item_type


def _extract_cond_branch_helper_expr(
    *,
    module_name: str,
    ctx: _FlattenCtx,
    globals_by_name: set[str],
    known_types: Mapping[str, TypeExpr],
    branch_non_null_names: set[str],
    branch_stmts: tuple[AxonStatement, ...],
    branch_expr: AxonExpr,
    helper_suffix: str,
) -> tuple[AxonExpr, AxonDefinition | None]:
    if not branch_stmts:
        return branch_expr, None
    helper_prefix = f"{module_name}__cond_{helper_suffix}"
    helper_name = ctx.fresh_helper(prefix=helper_prefix)

    free_names: list[str] = []
    seen_free: set[str] = set()
    local_bound: set[str] = set(ctx.module_path_params)
    all_stmts = [*branch_stmts, AxonReturn(values=(branch_expr,))]
    for stmt in all_stmts:
        for name in _statement_name_uses(stmt):
            if name in local_bound or name in globals_by_name or name in seen_free:
                continue
            seen_free.add(name)
            free_names.append(name)
        if isinstance(stmt, AxonBind):
            local_bound.update(name for name in stmt.targets if name != "_")
        elif isinstance(stmt, AxonRepeat):
            local_bound.add(stmt.var)
            local_bound.update(name for name in (stmt.targets or ()) if name != "_")
            local_bound.update(name for name in (stmt.carry or ()) if name != "_")
        elif isinstance(stmt, AxonScopeBind):
            local_bound.update(name for name in stmt.targets if name != "_")

    helper_free_names: list[str] = []
    helper_branch_stmts = branch_stmts
    helper_branch_expr = branch_expr
    for name in free_names:
        # Captured names can share spelling with type dimension variables.
        # Give the helper's value parameters separate lexical names so later
        # type-dimension substitutions cannot capture them.
        helper_name_for_free = ctx.fresh(prefix=f"__arg_{name}")
        replacement = AxonExprName(name=helper_name_for_free)
        helper_branch_stmts = tuple(
            _substitute_name_stmt(stmt, name=name, replacement=replacement)
            for stmt in helper_branch_stmts
        )
        helper_branch_expr = _substitute_name_expr(
            helper_branch_expr,
            name=name,
            replacement=replacement,
        )
        helper_free_names.append(helper_name_for_free)

    helper_params = tuple(
        _helper_param_for_free_name(
            helper_name,
            known_types.get(source_name, TypeAny()),
            force_non_null=source_name in branch_non_null_names,
        )
        for source_name, helper_name in zip(free_names, helper_free_names, strict=True)
    )
    helper_return_expr: AxonExpr
    helper_return_type: TypeExpr
    if len(branch_stmts) == 0:
        helper_return_expr = helper_branch_expr
        helper_return_type = TypeAny()
    elif isinstance(helper_branch_expr, AxonExprTuple):
        temp_names = tuple(ctx.fresh(prefix="__cond_result") for _ in helper_branch_expr.items)
        helper_statements = tuple(
            [
                *helper_branch_stmts,
                AxonBind(targets=temp_names, expr=helper_branch_expr),
                AxonReturn(values=tuple(AxonExprName(name=name) for name in temp_names)),
            ]
        )
        helper_return_type = TypeAny()
        helper_module = AxonDefinition(
            name=helper_name,
            path_param=None,
            params=tuple(_make_path_param(name) for name in ctx.module_path_params) + helper_params,
            returns=(),
            statements=helper_statements,
            body_expr=None,
            path_params=(),
            imports=(),
            imported_members=None,
            exports=(),
            symbols=None,
            pragmas=None,
            type_aliases=None,
            return_type_expr=None,
            constraints=None,
        )
        helper_call = AxonExprCall(
            callee=helper_name,
            args=tuple(AxonExprName(name=name) for name in (*ctx.module_path_params, *free_names)),
            kwargs={},
        )
        return helper_call, helper_module
    else:
        temp_name = ctx.fresh(prefix="__cond_result")
        helper_return_expr = AxonExprName(name=temp_name)
        helper_return_type = TypeAny()
        helper_module = AxonDefinition(
            name=helper_name,
            path_param=None,
            params=tuple(_make_path_param(name) for name in ctx.module_path_params) + helper_params,
            returns=(),
            statements=tuple(
                [
                    *helper_branch_stmts,
                    AxonBind(targets=(temp_name,), expr=helper_branch_expr),
                    AxonReturn(values=(helper_return_expr,)),
                ]
            ),
            body_expr=None,
            path_params=(),
            imports=(),
            imported_members=None,
            exports=(),
            symbols=None,
            pragmas=None,
            type_aliases=None,
            return_type_expr=None,
            constraints=None,
        )
        helper_call = AxonExprCall(
            callee=helper_name,
            args=tuple(AxonExprName(name=name) for name in (*ctx.module_path_params, *free_names)),
            kwargs={},
        )
        return helper_call, helper_module

    helper_module = AxonDefinition(
        name=helper_name,
        path_param=None,
        params=tuple(_make_path_param(name) for name in ctx.module_path_params) + helper_params,
        returns=(),
        statements=tuple([*branch_stmts, AxonReturn(values=(helper_return_expr,))]),
        body_expr=None,
        path_params=(),
        imports=(),
        imported_members=None,
        exports=(),
        symbols=None,
        pragmas=None,
        type_aliases=None,
        return_type_expr=helper_return_type,
        constraints=None,
    )
    helper_call = AxonExprCall(
        callee=helper_name,
        args=tuple(AxonExprName(name=name) for name in (*ctx.module_path_params, *free_names)),
        kwargs={},
    )
    return helper_call, helper_module


def _helper_param_for_free_name(name: str, type_expr: TypeExpr, *, force_non_null: bool = False) -> AxonParam:
    if isinstance(type_expr, TypeAny):
        return AxonParam(name=name, type_expr=None)
    if isinstance(type_expr, TypeOptional) and force_non_null:
        return AxonParam(name=name, type_expr=type_expr.inner)
    if isinstance(type_expr, TypeOptional):
        return AxonParam(name=name, optional=True, type_expr=type_expr.inner)
    return AxonParam(name=name, type_expr=type_expr)


def _condition_non_null_names_for_branch(
    condition: AxonExpr,
    *,
    branch_value: bool,
    conditions: Mapping[str, AxonExpr],
) -> set[str]:
    if isinstance(condition, AxonExprName):
        condition = conditions.get(condition.name, condition)
    if isinstance(condition, AxonExprAscribe | AxonExprParen):
        return _condition_non_null_names_for_branch(
            condition.inner if isinstance(condition, AxonExprParen) else condition.expr,
            branch_value=branch_value,
            conditions=conditions,
        )
    if not isinstance(condition, AxonExprBinary) or condition.op not in {"==", "!="}:
        return set()
    equality_branch = branch_value if condition.op == "==" else not branch_value
    if equality_branch:
        return set()
    names: set[str] = set()
    if isinstance(condition.left, AxonExprName) and isinstance(condition.right, AxonExprNull):
        names.add(condition.left.name)
    if isinstance(condition.right, AxonExprName) and isinstance(condition.left, AxonExprNull):
        names.add(condition.right.name)
    return names


def _extract_cond_helpers_from_statements(
    statements: tuple[AxonStatement, ...],
    *,
    module_name: str,
    ctx: _FlattenCtx,
    globals_by_name: set[str],
    local_bound_names: set[str],
    known_types: Mapping[str, TypeExpr],
    program_ctx: _FlattenProgramCtx,
    conditions: Mapping[str, AxonExpr] | None = None,
) -> tuple[tuple[AxonStatement, ...], tuple[AxonDefinition, ...]]:
    out: list[AxonStatement] = []
    helpers: list[AxonDefinition] = []
    local = set(local_bound_names)
    local_conditions = dict(conditions or {})
    for stmt in statements:
        if isinstance(stmt, AxonBind) and isinstance(stmt.expr, AxonExprTernary):
            true_non_null = _condition_non_null_names_for_branch(
                stmt.expr.cond,
                branch_value=True,
                conditions=local_conditions,
            )
            false_non_null = _condition_non_null_names_for_branch(
                stmt.expr.cond,
                branch_value=False,
                conditions=local_conditions,
            )
            true_payload = _extract_cond_branch_expr_payload(stmt.expr.true_expr)
            false_payload = _extract_cond_branch_expr_payload(stmt.expr.false_expr)
            if true_payload is not None and false_payload is not None:
                true_pre, true_expr_payload = true_payload
                false_pre, false_expr_payload = false_payload
                true_expr, true_helper = _extract_cond_branch_helper_expr(
                    module_name=module_name,
                    ctx=ctx,
                    globals_by_name=globals_by_name,
                    known_types=known_types,
                    branch_non_null_names=true_non_null,
                    branch_stmts=true_pre,
                    branch_expr=true_expr_payload,
                    helper_suffix="true",
                )
                false_expr, false_helper = _extract_cond_branch_helper_expr(
                    module_name=module_name,
                    ctx=ctx,
                    globals_by_name=globals_by_name,
                    known_types=known_types,
                    branch_non_null_names=false_non_null,
                    branch_stmts=false_pre,
                    branch_expr=false_expr_payload,
                    helper_suffix="else",
                )
                if true_helper is not None:
                    helpers.append(true_helper)
                if false_helper is not None:
                    helpers.append(false_helper)
                _bind_known_target_types(
                    known_types,
                    stmt.targets,
                    stmt.expr,
                    program_ctx=program_ctx,
                )
                out.append(
                    AxonBind(
                        targets=stmt.targets,
                        expr=AxonExprTernary(
                            cond=stmt.expr.cond,
                            true_expr=true_expr,
                            false_expr=false_expr,
                        ),
                    )
                )
                local.update(name for name in stmt.targets if name != "_")
                continue
        if isinstance(stmt, AxonCond):
            true_non_null = _condition_non_null_names_for_branch(
                stmt.cond,
                branch_value=True,
                conditions=local_conditions,
            )
            false_non_null = _condition_non_null_names_for_branch(
                stmt.cond,
                branch_value=False,
                conditions=local_conditions,
            )
            true_body, true_helpers = _extract_cond_helpers_from_statements(
                stmt.true_body,
                module_name=module_name,
                ctx=ctx,
                globals_by_name=globals_by_name,
                local_bound_names=local,
                known_types=known_types,
                program_ctx=program_ctx,
                conditions=local_conditions,
            )
            false_body, false_helpers = _extract_cond_helpers_from_statements(
                stmt.false_body,
                module_name=module_name,
                ctx=ctx,
                globals_by_name=globals_by_name,
                local_bound_names=local,
                known_types=known_types,
                program_ctx=program_ctx,
                conditions=local_conditions,
            )
            helpers.extend(true_helpers)
            helpers.extend(false_helpers)

            true_binding = _extract_cond_target_binding(true_body)
            false_binding = _extract_cond_target_binding(false_body)
            if (
                true_binding is not None
                and false_binding is not None
                and true_binding[0] == false_binding[0]
            ):
                target = true_binding[0]
                true_expr, true_helper = _extract_cond_branch_helper_expr(
                    module_name=module_name,
                    ctx=ctx,
                    globals_by_name=globals_by_name,
                    known_types=known_types,
                    branch_non_null_names=true_non_null,
                    branch_stmts=true_binding[1],
                    branch_expr=true_binding[2],
                    helper_suffix="true",
                )
                false_expr, false_helper = _extract_cond_branch_helper_expr(
                    module_name=module_name,
                    ctx=ctx,
                    globals_by_name=globals_by_name,
                    known_types=known_types,
                    branch_non_null_names=false_non_null,
                    branch_stmts=false_binding[1],
                    branch_expr=false_binding[2],
                    helper_suffix="else",
                )
                if true_helper is not None:
                    helpers.append(true_helper)
                if false_helper is not None:
                    helpers.append(false_helper)
                _bind_known_target_types(
                    known_types,
                    (target,),
                    true_binding[2],
                    program_ctx=program_ctx,
                )
                out.append(
                    AxonBind(
                        targets=(target,),
                        expr=AxonExprTernary(
                            cond=stmt.cond,
                            true_expr=true_expr,
                            false_expr=false_expr,
                        ),
                    )
                )
                local.add(target)
                continue

            true_return = _extract_cond_terminal_return(true_body)
            false_return = _extract_cond_terminal_return(false_body)
            if (
                true_return is not None
                and false_return is not None
                and len(true_return[1]) == len(false_return[1])
                and len(true_return[1]) > 0
            ):
                true_values_expr: AxonExpr
                false_values_expr: AxonExpr
                true_pre, true_values = true_return
                false_pre, false_values = false_return
                if len(true_values) == 1:
                    true_values_expr = true_values[0]
                    false_values_expr = false_values[0]
                else:
                    true_values_expr = AxonExprTuple(items=true_values)
                    false_values_expr = AxonExprTuple(items=false_values)
                true_expr, true_helper = _extract_cond_branch_helper_expr(
                    module_name=module_name,
                    ctx=ctx,
                    globals_by_name=globals_by_name,
                    known_types=known_types,
                    branch_non_null_names=true_non_null,
                    branch_stmts=true_pre,
                    branch_expr=true_values_expr,
                    helper_suffix="true",
                )
                false_expr, false_helper = _extract_cond_branch_helper_expr(
                    module_name=module_name,
                    ctx=ctx,
                    globals_by_name=globals_by_name,
                    known_types=known_types,
                    branch_non_null_names=false_non_null,
                    branch_stmts=false_pre,
                    branch_expr=false_values_expr,
                    helper_suffix="else",
                )
                if true_helper is not None:
                    helpers.append(true_helper)
                if false_helper is not None:
                    helpers.append(false_helper)
                temp_targets = (
                    (ctx.fresh(),)
                    if len(true_values) == 1
                    else tuple(ctx.fresh() for _ in range(len(true_values)))
                )
                out.append(
                    AxonBind(
                        targets=temp_targets,
                        expr=AxonExprTernary(
                            cond=stmt.cond,
                            true_expr=true_expr,
                            false_expr=false_expr,
                        ),
                    )
                )
                out.append(
                    AxonReturn(values=tuple(AxonExprName(name=name) for name in temp_targets))
                )
                for target, value in zip(temp_targets, true_values, strict=True):
                    _bind_known_target_types(
                        known_types,
                        (target,),
                        value,
                        program_ctx=program_ctx,
                    )
                local.update(temp_targets)
                continue

            out.append(AxonCond(cond=stmt.cond, true_body=true_body, false_body=false_body))
            continue
        if isinstance(stmt, AxonScopeBind):
            body, nested_helpers = _extract_cond_helpers_from_statements(
                stmt.body,
                module_name=module_name,
                ctx=ctx,
                globals_by_name=globals_by_name,
                local_bound_names=set(local) | {name for name in stmt.targets if name != "_"},
                known_types=known_types,
                program_ctx=program_ctx,
            )
            out.append(
                AxonScopeBind(
                    targets=stmt.targets, prefix=stmt.prefix, body=body, kwargs=stmt.kwargs
                )
            )
            helpers.extend(nested_helpers)
            local.update(name for name in stmt.targets if name != "_")
            continue
        if isinstance(stmt, AxonRepeat):
            loop_bound = set(local)
            loop_bound.add(stmt.var)
            loop_bound.update(name for name in (stmt.targets or ()) if name != "_")
            loop_bound.update(name for name in (stmt.carry or ()) if name != "_")
            body, nested_helpers = _extract_cond_helpers_from_statements(
                stmt.body,
                module_name=module_name,
                ctx=ctx,
                globals_by_name=globals_by_name,
                local_bound_names=loop_bound,
                known_types=known_types,
                program_ctx=program_ctx,
            )
            out.append(
                AxonRepeat(
                    name=stmt.name,
                    var=stmt.var,
                    to_expr=stmt.to_expr,
                    from_expr=stmt.from_expr,
                    step_expr=stmt.step_expr,
                    body=body,
                    targets=stmt.targets,
                    carry=stmt.carry,
                )
            )
            helpers.extend(nested_helpers)
            local.update(name for name in (stmt.targets or ()) if name != "_")
            local.update(name for name in (stmt.carry or ()) if name != "_")
            continue
        if (
            isinstance(stmt, AxonBind)
            and len(stmt.targets) == 1
            and stmt.targets[0] != "_"
            and isinstance(stmt.expr, AxonExprBinary)
            and stmt.expr.op in {"==", "!="}
        ):
            local_conditions[stmt.targets[0]] = stmt.expr
        out.append(stmt)
        if isinstance(stmt, AxonBind):
            _bind_known_target_types(
                known_types,
                stmt.targets,
                stmt.expr,
                program_ctx=program_ctx,
            )
            local.update(name for name in stmt.targets if name != "_")
    return tuple(out), tuple(helpers)


def _flatten_expr(
    expr: AxonExpr,
    ctx: _FlattenCtx,
    *,
    path_prefix: tuple[str, ...],
    program_ctx: _FlattenProgramCtx,
) -> tuple[list[AxonStatement], AxonExpr]:
    if isinstance(expr, _ATOMIC_EXPR_TYPES):
        if isinstance(expr, AxonExprPath):
            return [], _normalize_path_expr(
                expr, path_prefix=path_prefix, path_names=ctx.module_path_params
            )
        return [], expr
    if isinstance(expr, AxonExprParen):
        return _flatten_expr(expr.inner, ctx, path_prefix=path_prefix, program_ctx=program_ctx)
    if isinstance(expr, AxonExprPipe):
        current: AxonExpr = expr.value
        for stage in expr.stages:
            current = _pipe_stage_to_call(current, stage)
        return _flatten_expr(current, ctx, path_prefix=path_prefix, program_ctx=program_ctx)
    if isinstance(expr, AxonExprAscribe):
        ascribe_prelude, inner = _flatten_expr(
            expr.expr,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        return ascribe_prelude, AxonExprAscribe(expr=inner, type_expr=expr.type_expr)
    if isinstance(expr, AxonExprList):
        list_prelude: list[AxonStatement] = []
        list_items: list[AxonExpr] = []
        for item in expr.items:
            item_pre, item_flat = _flatten_expr(
                item,
                ctx,
                path_prefix=path_prefix,
                program_ctx=program_ctx,
            )
            list_prelude.extend(item_pre)
            list_items.append(_bind_if_non_atomic(list_prelude, item_flat, ctx))
        return list_prelude, AxonExprList(items=tuple(list_items))
    if isinstance(expr, AxonExprTuple):
        tuple_prelude: list[AxonStatement] = []
        tuple_items: list[AxonExpr] = []
        for item in expr.items:
            item_pre, item_flat = _flatten_expr(
                item,
                ctx,
                path_prefix=path_prefix,
                program_ctx=program_ctx,
            )
            tuple_prelude.extend(item_pre)
            tuple_items.append(_bind_if_non_atomic(tuple_prelude, item_flat, ctx))
        return tuple_prelude, AxonExprTuple(items=tuple(tuple_items))
    if isinstance(expr, AxonExprCall):
        call_expr = _expand_call_surface(expr, program_ctx)
        callee_module = program_ctx.modules_by_name.get(call_expr.callee)
        if callee_module is not None and not _module_declares_path_inputs(callee_module):
            if path_prefix:
                program_ctx.nonempty_called_modules.add(call_expr.callee)
                if call_expr.callee in program_ctx.scoped_modules:
                    call_expr = replace(
                        call_expr,
                        args=(AxonExprPath(absolute=True, parts=path_prefix), *call_expr.args),
                    )
            else:
                program_ctx.root_called_modules.add(call_expr.callee)
        call_prelude: list[AxonStatement] = []
        args: list[AxonExpr] = []
        for arg in call_expr.args:
            arg_pre, arg_expr = _flatten_expr(
                arg,
                ctx,
                path_prefix=path_prefix,
                program_ctx=program_ctx,
            )
            call_prelude.extend(arg_pre)
            args.append(_bind_if_non_atomic(call_prelude, arg_expr, ctx))
        kwargs: dict[str, AxonKwargValue] = {}
        for key, raw_value in call_expr.kwargs.items():
            if isinstance(raw_value, AxonExprPath):
                kwargs[key] = raw_value
                continue
            if isinstance(raw_value, AxonExpr):
                kw_pre, kw_expr = _flatten_expr(
                    raw_value,
                    ctx,
                    path_prefix=path_prefix,
                    program_ctx=program_ctx,
                )
                call_prelude.extend(kw_pre)
                kwargs[key] = _bind_if_non_atomic(call_prelude, kw_expr, ctx)
            else:
                kwargs[key] = raw_value
        _absolutize_call_relative_paths(
            callee=call_expr.callee,
            args=args,
            kwargs=kwargs,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        return call_prelude, AxonExprCall(callee=call_expr.callee, args=tuple(args), kwargs=kwargs)
    if isinstance(expr, AxonExprBinary):
        left_pre, left_expr = _flatten_expr(
            expr.left,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        right_pre, right_expr = _flatten_expr(
            expr.right,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        prelude = [*left_pre, *right_pre]
        if not _is_atomic_expr(left_expr):
            temp = ctx.fresh()
            prelude.append(AxonBind(targets=(temp,), expr=left_expr))
            left_expr = AxonExprName(name=temp)
        if not _is_atomic_expr(right_expr):
            temp = ctx.fresh()
            prelude.append(AxonBind(targets=(temp,), expr=right_expr))
            right_expr = AxonExprName(name=temp)
        return prelude, AxonExprBinary(op=expr.op, left=left_expr, right=right_expr)
    if isinstance(expr, AxonExprBind):
        value_pre, value_expr = _flatten_expr(
            expr.value,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        body_pre, body_expr = _flatten_expr(
            expr.body,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        value_stmt = AxonBind(targets=(expr.var,), expr=value_expr)
        return [*value_pre, value_stmt, *body_pre], body_expr
    if isinstance(expr, AxonExprIf):
        cond_pre, cond_expr = _flatten_expr(
            expr.cond,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        cond_expr = _bind_if_non_atomic(cond_pre, cond_expr, ctx)
        true_pre, true_expr = _flatten_expr(
            expr.true_expr,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        false_pre, false_expr = _flatten_expr(
            expr.false_expr,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        if not true_pre and not false_pre:
            return [*cond_pre], AxonExprTernary(
                cond=cond_expr, true_expr=true_expr, false_expr=false_expr
            )
        temp = ctx.fresh()
        return [
            *cond_pre,
            AxonCond(
                cond=cond_expr,
                true_body=tuple([*true_pre, AxonBind(targets=(temp,), expr=true_expr)]),
                false_body=tuple([*false_pre, AxonBind(targets=(temp,), expr=false_expr)]),
            ),
        ], AxonExprName(name=temp)
    if isinstance(expr, AxonExprTernary):
        cond_pre, cond_expr = _flatten_expr(
            expr.cond,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        cond_expr = _bind_if_non_atomic(cond_pre, cond_expr, ctx)
        true_pre, true_expr = _flatten_expr(
            expr.true_expr,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        false_pre, false_expr = _flatten_expr(
            expr.false_expr,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        if not true_pre and not false_pre:
            return [*cond_pre], AxonExprTernary(
                cond=cond_expr, true_expr=true_expr, false_expr=false_expr
            )
        temp = ctx.fresh()
        return [
            *cond_pre,
            AxonCond(
                cond=cond_expr,
                true_body=tuple([*true_pre, AxonBind(targets=(temp,), expr=true_expr)]),
                false_body=tuple([*false_pre, AxonBind(targets=(temp,), expr=false_expr)]),
            ),
        ], AxonExprName(name=temp)
    if isinstance(expr, AxonExprLambda):
        body_pre, body_expr = _flatten_expr(
            expr.body,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        return [], AxonExprLambda(var=expr.var, body=_wrap_inline_do(body_pre, body_expr))
    if isinstance(expr, AxonExprDo):
        return _reduce_do_expr(expr, ctx, path_prefix=path_prefix, program_ctx=program_ctx)
    return [], expr


def _ensure_atom(
    expr: AxonExpr,
    ctx: _FlattenCtx,
    *,
    path_prefix: tuple[str, ...],
    program_ctx: _FlattenProgramCtx,
) -> tuple[list[AxonStatement], AxonExpr]:
    prelude, flat = _flatten_expr(expr, ctx, path_prefix=path_prefix, program_ctx=program_ctx)
    if _is_atomic_expr(flat):
        return prelude, flat
    temp = ctx.fresh()
    return [*prelude, AxonBind(targets=(temp,), expr=flat)], AxonExprName(name=temp)


def _flatten_kwarg_value(
    value: AxonKwargValue,
    ctx: _FlattenCtx,
    *,
    path_prefix: tuple[str, ...],
    program_ctx: _FlattenProgramCtx,
) -> tuple[list[AxonStatement], AxonKwargValue]:
    if isinstance(value, AxonExpr):
        prelude, flat = _flatten_expr(value, ctx, path_prefix=path_prefix, program_ctx=program_ctx)
        return prelude, flat
    return [], value


def _flatten_statement(
    stmt: AxonStatement,
    ctx: _FlattenCtx,
    *,
    path_prefix: tuple[str, ...],
    program_ctx: _FlattenProgramCtx,
) -> list[AxonStatement]:
    if isinstance(stmt, AxonBind):
        bind_prelude, flat_expr = _flatten_expr(
            stmt.expr,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        return [*bind_prelude, AxonBind(targets=stmt.targets, expr=flat_expr)]
    if isinstance(stmt, AxonReturn):
        return_prelude: list[AxonStatement] = []
        return_values: list[AxonExpr] = []
        for raw_value in stmt.values:
            value_pre, value_atom = _ensure_atom(
                raw_value,
                ctx,
                path_prefix=path_prefix,
                program_ctx=program_ctx,
            )
            return_prelude.extend(value_pre)
            return_values.append(value_atom)
        return [*return_prelude, AxonReturn(values=tuple(return_values))]
    if isinstance(stmt, AxonYield):
        yield_prelude: list[AxonStatement] = []
        yield_values: list[AxonExpr] = []
        for raw_value in stmt.values:
            value_pre, value_atom = _ensure_atom(
                raw_value,
                ctx,
                path_prefix=path_prefix,
                program_ctx=program_ctx,
            )
            yield_prelude.extend(value_pre)
            yield_values.append(value_atom)
        return [*yield_prelude, AxonYield(values=tuple(yield_values))]
    if isinstance(stmt, AxonCond):
        cond_pre, cond_atom = _ensure_atom(
            stmt.cond,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        cond_true_body: tuple[AxonStatement, ...] = _flatten_statements(
            stmt.true_body,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        cond_false_body: tuple[AxonStatement, ...] = _flatten_statements(
            stmt.false_body,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        return [
            *cond_pre,
            AxonCond(
                cond=cond_atom,
                true_body=cond_true_body,
                false_body=cond_false_body,
            ),
        ]
    if isinstance(stmt, AxonRepeat):
        from_pre, from_expr = _ensure_atom(
            stmt.from_expr,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        to_pre, to_expr = _ensure_atom(
            stmt.to_expr,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        step_pre, step_expr = _ensure_atom(
            stmt.step_expr,
            ctx,
            path_prefix=path_prefix,
            program_ctx=program_ctx,
        )
        body = _flatten_statements(
            stmt.body,
            ctx,
            path_prefix=(*path_prefix, *_loop_scope_parts(stmt)),
            program_ctx=program_ctx,
        )
        return [
            *from_pre,
            *to_pre,
            *step_pre,
            AxonRepeat(
                name=stmt.name,
                var=stmt.var,
                to_expr=to_expr,
                from_expr=from_expr,
                step_expr=step_expr,
                body=body,
                targets=stmt.targets,
                carry=stmt.carry,
            ),
        ]
    if isinstance(stmt, AxonScopeBind):
        scope_prelude: list[AxonStatement] = []
        kwargs: dict[str, AxonKwargValue] = {}
        normalized_prefix = _normalize_path_expr(
            stmt.prefix, path_prefix=path_prefix, path_names=ctx.module_path_params
        )
        for key in stmt.kwargs:
            scope_value: AxonKwargValue = stmt.kwargs[key]
            kw_pre, kw_value = _flatten_kwarg_value(
                scope_value,
                ctx,
                path_prefix=path_prefix,
                program_ctx=program_ctx,
            )
            scope_prelude.extend(kw_pre)
            kwargs[key] = kw_value
        body_path_prefix = (
            normalized_prefix.parts
            if normalized_prefix.absolute
            else (*path_prefix, *normalized_prefix.parts)
        )
        body = _flatten_statements(
            stmt.body,
            ctx,
            path_prefix=body_path_prefix,
            program_ctx=program_ctx,
        )
        if kwargs:
            raise ValueError("flatten failed: scope kwargs are not yet supported in flat core")
        if not body:
            raise ValueError("flatten failed: empty scope body")
        *body_prelude, last = body
        if not isinstance(last, AxonReturn):
            raise ValueError("flatten failed: scope body must end in return")
        if len(last.values) != len(stmt.targets):
            raise ValueError("flatten failed: scope return arity mismatch")
        tail: list[AxonStatement] = []
        if stmt.targets:
            tail.append(
                AxonBind(
                    targets=stmt.targets,
                    expr=AxonExprTuple(items=last.values)
                    if len(last.values) > 1
                    else last.values[0],
                )
            )
        return [*scope_prelude, *body_prelude, *tail]
    return [stmt]


def _flatten_statements(
    stmts: tuple[AxonStatement, ...],
    ctx: _FlattenCtx,
    *,
    path_prefix: tuple[str, ...],
    program_ctx: _FlattenProgramCtx,
) -> tuple[AxonStatement, ...]:
    out: list[AxonStatement] = []
    for stmt in stmts:
        out.extend(_flatten_statement(stmt, ctx, path_prefix=path_prefix, program_ctx=program_ctx))
    return tuple(out)


def _is_inlineable_flat_temp_expr(expr: AxonExpr) -> bool:
    return False


def _inline_trivial_temp_binds(stmts: tuple[AxonStatement, ...]) -> tuple[AxonStatement, ...]:
    out: list[AxonStatement] = []
    idx = 0
    while idx < len(stmts):
        stmt = stmts[idx]
        if (
            isinstance(stmt, AxonBind)
            and len(stmt.targets) == 1
            and stmt.targets[0].startswith("__flat_")
            and _is_inlineable_flat_temp_expr(stmt.expr)
        ):
            name = stmt.targets[0]
            rest = stmts[idx + 1 :]
            if sum(_statement_name_uses(item).count(name) for item in rest) == 1:
                replacement = stmt.expr
                out.extend(
                    _substitute_name_stmt(item, name=name, replacement=replacement)
                    for item in rest
                )
                return _inline_trivial_temp_binds(tuple(out))
        out.append(stmt)
        idx += 1
    return tuple(out)


def _module_body_statements(module: AxonDefinition) -> tuple[AxonStatement, ...]:
    if module.body_expr is None:
        return module.statements
    if isinstance(module.body_expr, AxonExprDo) and not module.body_expr.inline:
        return module.body_expr.body
    return (AxonReturn(values=(module.body_expr,)),)


def _flatten_module(
    module: AxonDefinition,
    program_ctx: _FlattenProgramCtx,
    *,
    globals_by_name: set[str],
) -> tuple[AxonDefinition, tuple[AxonDefinition, ...]]:
    module_path_params = tuple(
        dict.fromkeys(
            (
                *((module.path_param,) if module.path_param is not None else ()),
                *module.path_params,
            )
        )
    )
    initial_path_prefix: tuple[str, ...] = ()
    used_names = _module_used_names(module)
    if module.path_param is not None:
        initial_path_prefix = (f"{{{module.path_param}}}",)
    if (
        module.name in program_ctx.scoped_modules
        and program_ctx.scope_param_name not in module_path_params
        and module.path_param != program_ctx.scope_param_name
    ):
        scope_param_name = _fresh_scope_param_name(used_names, program_ctx.scope_param_name)
        module_path_params = (scope_param_name, *module_path_params)
        initial_path_prefix = (f"{{{scope_param_name}}}",)
    used_names.update(module_path_params)
    ctx = _FlattenCtx(used_names=used_names, module_path_params=module_path_params)
    type_aliases = _type_aliases_for_module(program_ctx, module)
    expanded_params = tuple(
        AxonParam(
            name=param.name,
            optional=param.optional,
            type_expr=_expand_type_aliases(param.type_expr, type_aliases=type_aliases),
            default_expr=None,
        )
        for param in module.params
    )
    known_types: dict[str, TypeExpr] = {name: TypePath() for name in module_path_params}
    if module.path_param is not None:
        known_types[module.path_param] = TypePath()
    for param in expanded_params:
        if param.type_expr is not None:
            known_types[param.name] = (
                TypeOptional(param.type_expr)
                if param.optional and not isinstance(param.type_expr, TypeOptional)
                else param.type_expr
            )
    statements = _flatten_statements(
        _module_body_statements(module),
        ctx,
        path_prefix=initial_path_prefix,
        program_ctx=program_ctx,
    )
    statements, cond_helper_modules = _extract_cond_helpers_from_statements(
        statements,
        module_name=module.name,
        ctx=ctx,
        globals_by_name=globals_by_name,
        local_bound_names=set(module_path_params)
        | ({module.path_param} if module.path_param is not None else set())
        | {param.name for param in module.params},
        known_types=known_types,
        program_ctx=program_ctx,
    )
    statements, helper_modules = _extract_repeat_helpers_from_statements(
        statements,
        module_name=module.name,
        ctx=ctx,
        globals_by_name=globals_by_name,
    )
    seen_path_param_names: set[str] = set()
    ordered_path_param_names: list[str] = []
    for name in (
        *module_path_params,
        *((module.path_param,) if module.path_param is not None else ()),
    ):
        if name in seen_path_param_names:
            continue
        seen_path_param_names.add(name)
        ordered_path_param_names.append(name)
    desugared_path_params = tuple(_make_path_param(name) for name in ordered_path_param_names)
    flattened = AxonDefinition(
        name=module.name,
        path_param=None,
        params=desugared_path_params + expanded_params,
        returns=module.returns,
        statements=tuple(
            _expand_stmt_aliases(stmt, type_aliases=type_aliases) for stmt in statements
        ),
        body_expr=None,
        path_params=(),
        imports=module.imports,
        imported_members=module.imported_members,
        exports=module.exports,
        symbols=module.symbols,
        pragmas=module.pragmas,
        type_aliases=None,
        return_type_expr=_expand_type_aliases(
            module.return_type_expr, type_aliases=type_aliases
        ),
        constraints=module.constraints,
        is_global_binding=module.is_global_binding,
    )
    expanded_helpers = tuple(
        _expand_definition_aliases(helper, type_aliases=type_aliases)
        for helper in (*cond_helper_modules, *helper_modules)
    )
    return flattened, expanded_helpers


def flatten_closed_axon_file(program: AxonFile, *, main_module: str | None = None) -> AxonFile:
    validate_elaborated_axon_file(program, main_module=main_module)
    _ensure_elaborated_input(program)
    scoped_modules: frozenset[str] = frozenset()
    globals_by_name = {module.name for module in program.modules}
    source_path_dependent_modules = _lexically_path_dependent_modules(program.modules)
    while True:
        program_ctx = _FlattenProgramCtx(
            modules_by_name={module.name: module for module in program.modules},
            type_aliases=dict(program.type_aliases),
            scoped_modules=scoped_modules,
            root_called_modules=set(),
            nonempty_called_modules=set(),
        )
        flattened_modules: list[AxonDefinition] = []
        for module in program.modules:
            flattened, helper_modules = _flatten_module(
                module,
                program_ctx,
                globals_by_name=globals_by_name,
            )
            flattened_modules.extend(helper_modules)
            flattened_modules.append(flattened)
        path_dependent_modules = source_path_dependent_modules | _lexically_path_dependent_modules(
            tuple(flattened_modules)
        )
        discovered = frozenset(
            (program_ctx.nonempty_called_modules - program_ctx.root_called_modules)
            & path_dependent_modules
        )
        if discovered == scoped_modules:
            flat_program = AxonFile(
                modules=tuple(flattened_modules),
                imports=program.imports,
                imported_members=dict(program.imported_members),
                exports=program.exports,
                pragmas=dict(program.pragmas),
                type_aliases={},
                origin_path=program.origin_path,
            )
            validate_flat_axon_file(flat_program, main_module=main_module)
            return flat_program
        scoped_modules = discovered


__all__ = ["flatten_closed_axon_file"]
