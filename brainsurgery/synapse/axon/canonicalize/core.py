from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

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
    AxonDefinition,
    AxonParam,
    AxonRepeat,
    AxonReturn,
    AxonScopeBind,
    AxonStatement,
    AxonYield,
    Constraint,
    DimExprBinary,
    TypeAny,
    TypeBool,
    TypeDim,
    TypeExpr,
    TypeFloat,
    TypeInt,
    TypeList,
    TypeNamed,
    TypeNull,
    TypeOptional,
    TypePath,
    TypeString,
    TypeTensor,
    TypeTuple,
    TypeVar,
    ast_equal,
    dim_token_names,
)
from ..validate import validate_typed_axon_file

_GENERATED_HELPER_RE = re.compile(
    r"^(?P<base>.+?)(?:__cond_(?:true|else)_\d+|__loop_[A-Za-z0-9_]+_\d+)$"
)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class _DimProvenance:
    signature: frozenset[str]
    body_terms: frozenset[str]
    inferred: frozenset[str]
    synthetic_signature: frozenset[str] = frozenset()
    allow_external_signature_names: bool = False


def _is_generated_helper(name: str) -> bool:
    return bool(_GENERATED_HELPER_RE.match(name))


def _split_variadic_dim(name: str) -> tuple[bool, str]:
    if name.startswith(".."):
        return True, name[2:]
    return False, name


def _dim_score(name: str) -> tuple[int, int, int, str]:
    generic = 1 if len(name) == 1 and name.isupper() else 0
    return (generic, len(name), name)


def _expr_names(expr: AxonExpr) -> set[str]:
    if isinstance(expr, AxonExprName):
        return {expr.name}
    if isinstance(expr, AxonExprBinary):
        return _expr_names(expr.left) | _expr_names(expr.right)
    if isinstance(expr, AxonExprBind):
        return _expr_names(expr.value) | (_expr_names(expr.body) - {expr.var})
    if isinstance(expr, AxonExprCall):
        call_names: set[str] = set()
        for arg in expr.args:
            call_names.update(_expr_names(arg))
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                call_names.update(_expr_names(value))
        return call_names
    if isinstance(expr, AxonExprDo):
        return _stmt_names(expr.body)
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return _expr_names(expr.cond) | _expr_names(expr.true_expr) | _expr_names(expr.false_expr)
    if isinstance(expr, AxonExprLambda):
        return _expr_names(expr.body) - {expr.var}
    if isinstance(expr, AxonExprAscribe):
        return _expr_names(expr.expr)
    if isinstance(expr, AxonExprList | AxonExprTuple):
        names: set[str] = set()
        for item in expr.items:
            names.update(_expr_names(item))
        return names
    if isinstance(expr, AxonExprParen):
        return _expr_names(expr.inner)
    if isinstance(expr, AxonExprPipe):
        pipe_names = _expr_names(expr.value)
        for item in expr.stages:
            pipe_names.update(_expr_names(item))
        return pipe_names
    return set()


def _expr_param_like_names(expr: AxonExpr) -> set[str]:
    names = _expr_names(expr)
    if isinstance(expr, AxonExprPath):
        for part in expr.parts:
            if part.startswith("{") and part.endswith("}") and len(part) > 2:
                names.add(part[1:-1])
            elif _IDENT_RE.match(part):
                names.add(part)
        return names
    if isinstance(expr, AxonExprBinary):
        return names | _expr_param_like_names(expr.left) | _expr_param_like_names(expr.right)
    if isinstance(expr, AxonExprBind):
        return names | _expr_param_like_names(expr.value) | _expr_param_like_names(expr.body)
    if isinstance(expr, AxonExprCall):
        out = set(names)
        for arg in expr.args:
            out.update(_expr_param_like_names(arg))
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                out.update(_expr_param_like_names(value))
        return out
    if isinstance(expr, AxonExprDo):
        return names | _stmt_param_like_names(expr.body)
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return (
            names
            | _expr_param_like_names(expr.cond)
            | _expr_param_like_names(expr.true_expr)
            | _expr_param_like_names(expr.false_expr)
        )
    if isinstance(expr, AxonExprLambda):
        return names | _expr_param_like_names(expr.body)
    if isinstance(expr, AxonExprAscribe):
        return names | _expr_param_like_names(expr.expr)
    if isinstance(expr, AxonExprList | AxonExprTuple):
        out = set(names)
        for item in expr.items:
            out.update(_expr_param_like_names(item))
        return out
    if isinstance(expr, AxonExprParen):
        return names | _expr_param_like_names(expr.inner)
    if isinstance(expr, AxonExprPipe):
        out = names | _expr_param_like_names(expr.value)
        for item in expr.stages:
            out.update(_expr_param_like_names(item))
        return out
    return names


def _stmt_param_like_names(statements: tuple[AxonStatement, ...]) -> set[str]:
    names: set[str] = set()
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            names.update(_expr_param_like_names(stmt.expr))
        elif isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                names.update(_expr_param_like_names(value))
        elif isinstance(stmt, AxonCond):
            names.update(_expr_param_like_names(stmt.cond))
            names.update(_stmt_param_like_names(stmt.true_body))
            names.update(_stmt_param_like_names(stmt.false_body))
        elif isinstance(stmt, AxonRepeat):
            names.update(_expr_param_like_names(stmt.from_expr))
            names.update(_expr_param_like_names(stmt.to_expr))
            names.update(_expr_param_like_names(stmt.step_expr))
            names.update(_stmt_param_like_names(stmt.body))
        elif isinstance(stmt, AxonScopeBind):
            names.update(_expr_param_like_names(stmt.prefix))
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    names.update(_expr_param_like_names(raw_value))
            names.update(_stmt_param_like_names(stmt.body))
    return names


def _stmt_names(statements: tuple[AxonStatement, ...]) -> set[str]:
    names: set[str] = set()
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            names.update(_expr_names(stmt.expr))
        elif isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                names.update(_expr_names(value))
        elif isinstance(stmt, AxonCond):
            names.update(_expr_names(stmt.cond))
            names.update(_stmt_names(stmt.true_body))
            names.update(_stmt_names(stmt.false_body))
        elif isinstance(stmt, AxonRepeat):
            names.update(_expr_names(stmt.from_expr))
            names.update(_expr_names(stmt.to_expr))
            names.update(_expr_names(stmt.step_expr))
            names.update(_stmt_names(stmt.body))
        elif isinstance(stmt, AxonScopeBind):
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    names.update(_expr_names(raw_value))
            names.update(_stmt_names(stmt.body))
    return names


def _bound_names_in_order(statements: tuple[AxonStatement, ...]) -> list[str]:
    names: list[str] = []
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            for target in stmt.targets:
                if target != "_":
                    names.append(target)
    return names


def _deduped_path_names(module: AxonDefinition) -> tuple[str, ...]:
    names = list(module.path_params)
    if module.path_param is not None and module.path_param not in names:
        names.append(module.path_param)
    return tuple(names)


def _param_names(module: AxonDefinition) -> tuple[str, ...]:
    names: list[str] = list(_deduped_path_names(module))
    names.extend(param.name for param in module.params)
    return tuple(names)


def _canonicalize_generated_local_names(module: AxonDefinition) -> AxonDefinition:
    used = set(_param_names(module)) | _bound_names_statements(module.statements)
    renames: dict[str, str] = {}
    next_idx = 1
    for name in _bound_names_in_order(module.statements):
        if not name.startswith("__") or name in renames:
            continue
        while True:
            candidate = f"_v{next_idx}"
            next_idx += 1
            if candidate not in used and candidate not in renames.values():
                renames[name] = candidate
                break
    if not renames:
        return module
    return replace(module, statements=_rename_stmts(module.statements, renames))


def _canonicalize_generated_helper_names(program: AxonFile) -> AxonFile:
    used_names = {module.name for module in program.modules}
    renames: dict[str, str] = {}
    counters: dict[str, int] = {}
    for module in program.modules:
        match = _GENERATED_HELPER_RE.match(module.name)
        if match is None:
            continue
        base = match.group("base")
        idx = counters.get(base, 1)
        while True:
            candidate = f"{base}_h{idx}"
            idx += 1
            if candidate not in used_names and candidate not in renames.values():
                counters[base] = idx
                renames[module.name] = candidate
                break
    if not renames:
        return program
    return replace(
        program,
        modules=tuple(
            replace(
                module,
                name=renames.get(module.name, module.name),
                statements=_rename_callee_stmts(module.statements, renames),
            )
            for module in program.modules
        ),
    )


def _bound_names_statements(statements: tuple[AxonStatement, ...]) -> set[str]:
    names: set[str] = set()
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            for target in stmt.targets:
                if target != "_":
                    names.add(target)
        elif isinstance(stmt, AxonCond):
            names.update(_bound_names_statements(stmt.true_body))
            names.update(_bound_names_statements(stmt.false_body))
        elif isinstance(stmt, AxonRepeat):
            names.update(_bound_names_statements(stmt.body))
        elif isinstance(stmt, AxonScopeBind):
            names.update(_bound_names_statements(stmt.body))
    return names


def _rename_path(expr: AxonExprPath, renames: Mapping[str, str]) -> AxonExprPath:
    rewritten_parts: list[str] = []
    for part in expr.parts:
        if part.startswith("{") and part.endswith("}") and len(part) > 2:
            inner = part[1:-1]
            rewritten_parts.append("{" + renames.get(inner, inner) + "}")
        else:
            rewritten_parts.append(part)
    return replace(expr, parts=tuple(rewritten_parts))


def _rename_expr(expr: AxonExpr, renames: Mapping[str, str]) -> AxonExpr:
    if isinstance(expr, AxonExprName):
        return replace(expr, name=renames.get(expr.name, expr.name))
    if isinstance(expr, AxonExprPath):
        return _rename_path(expr, renames)
    if isinstance(expr, AxonExprBinary):
        return replace(expr, left=_rename_expr(expr.left, renames), right=_rename_expr(expr.right, renames))
    if isinstance(expr, AxonExprBind):
        next_renames = dict(renames)
        next_renames.pop(expr.var, None)
        return replace(expr, value=_rename_expr(expr.value, renames), body=_rename_expr(expr.body, next_renames))
    if isinstance(expr, AxonExprCall):
        return replace(
            expr,
            args=tuple(_rename_expr(arg, renames) for arg in expr.args),
            kwargs={k: _rename_expr(v, renames) if isinstance(v, AxonExpr) else v for k, v in expr.kwargs.items()},
        )
    if isinstance(expr, AxonExprDo):
        return replace(expr, body=_rename_stmts(expr.body, renames))
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return replace(
            expr,
            cond=_rename_expr(expr.cond, renames),
            true_expr=_rename_expr(expr.true_expr, renames),
            false_expr=_rename_expr(expr.false_expr, renames),
        )
    if isinstance(expr, AxonExprLambda):
        next_renames = dict(renames)
        next_renames.pop(expr.var, None)
        return replace(expr, body=_rename_expr(expr.body, next_renames))
    if isinstance(expr, AxonExprAscribe):
        return replace(expr, expr=_rename_expr(expr.expr, renames))
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return replace(expr, items=tuple(_rename_expr(item, renames) for item in expr.items))
    if isinstance(expr, AxonExprParen):
        return replace(expr, inner=_rename_expr(expr.inner, renames))
    if isinstance(expr, AxonExprPipe):
        return replace(
            expr,
            value=_rename_expr(expr.value, renames),
            stages=tuple(_rename_expr(item, renames) for item in expr.stages),
        )
    return expr


def _rename_stmts(
    statements: tuple[AxonStatement, ...], renames: Mapping[str, str]
) -> tuple[AxonStatement, ...]:
    rewritten: list[AxonStatement] = []
    active = dict(renames)
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            rewritten_expr = _rename_expr(stmt.expr, active)
            new_targets = tuple(active.get(target, target) for target in stmt.targets)
            rewritten.append(replace(stmt, targets=new_targets, expr=rewritten_expr))
            continue
        if isinstance(stmt, AxonReturn | AxonYield):
            rewritten.append(replace(stmt, values=tuple(_rename_expr(value, active) for value in stmt.values)))
            continue
        if isinstance(stmt, AxonCond):
            rewritten.append(
                replace(
                    stmt,
                    cond=_rename_expr(stmt.cond, active),
                    true_body=_rename_stmts(stmt.true_body, active),
                    false_body=_rename_stmts(stmt.false_body, active),
                )
            )
            continue
        if isinstance(stmt, AxonRepeat):
            next_renames = dict(active)
            next_renames.pop(stmt.var, None)
            rewritten.append(
                replace(
                    stmt,
                    name=active.get(stmt.name, stmt.name) if stmt.name is not None else None,
                    var=stmt.var,
                    from_expr=_rename_expr(stmt.from_expr, active),
                    to_expr=_rename_expr(stmt.to_expr, active),
                    step_expr=_rename_expr(stmt.step_expr, active),
                    body=_rename_stmts(stmt.body, next_renames),
                    targets=tuple(active.get(target, target) for target in (stmt.targets or ())),
                    carry=tuple(active.get(name, name) for name in (stmt.carry or ())),
                )
            )
            continue
        if isinstance(stmt, AxonScopeBind):
            rewritten.append(
                replace(
                    stmt,
                    targets=tuple(active.get(target, target) for target in stmt.targets),
                    prefix=_rename_path(stmt.prefix, active),
                    body=_rename_stmts(stmt.body, active),
                    kwargs={
                        key: _rename_expr(value, active) if isinstance(value, AxonExpr) else value
                        for key, value in stmt.kwargs.items()
                    },
                )
            )
            continue
        rewritten.append(stmt)
    return tuple(rewritten)


def _rename_callee_expr(expr: AxonExpr, renames: Mapping[str, str]) -> AxonExpr:
    if isinstance(expr, AxonExprCall):
        return replace(
            expr,
            callee=renames.get(expr.callee, expr.callee),
            args=tuple(_rename_callee_expr(arg, renames) for arg in expr.args),
            kwargs={k: _rename_callee_expr(v, renames) if isinstance(v, AxonExpr) else v for k, v in expr.kwargs.items()},
        )
    if isinstance(expr, AxonExprBinary):
        return replace(expr, left=_rename_callee_expr(expr.left, renames), right=_rename_callee_expr(expr.right, renames))
    if isinstance(expr, AxonExprAscribe):
        return replace(expr, expr=_rename_callee_expr(expr.expr, renames))
    if isinstance(expr, AxonExprTernary):
        return replace(
            expr,
            cond=_rename_callee_expr(expr.cond, renames),
            true_expr=_rename_callee_expr(expr.true_expr, renames),
            false_expr=_rename_callee_expr(expr.false_expr, renames),
        )
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return replace(expr, items=tuple(_rename_callee_expr(item, renames) for item in expr.items))
    return expr


def _rename_callee_stmts(
    statements: tuple[AxonStatement, ...], renames: Mapping[str, str]
) -> tuple[AxonStatement, ...]:
    rewritten: list[AxonStatement] = []
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            rewritten.append(replace(stmt, expr=_rename_callee_expr(stmt.expr, renames)))
        elif isinstance(stmt, AxonReturn | AxonYield):
            rewritten.append(replace(stmt, values=tuple(_rename_callee_expr(value, renames) for value in stmt.values)))
        else:
            rewritten.append(stmt)
    return tuple(rewritten)


def _canonicalize_path_params(program: AxonFile) -> AxonFile:
    modules: list[AxonDefinition] = []
    for module in program.modules:
        path_names = _deduped_path_names(module)
        if not path_names:
            modules.append(module)
            continue
        path_params = tuple(AxonParam(name=name, type_expr=TypePath()) for name in path_names)
        modules.append(replace(module, path_param=None, path_params=(), params=path_params + module.params))
    return replace(program, modules=tuple(modules))


def _walk_expr_calls(expr: AxonExpr):
    if isinstance(expr, AxonExprCall):
        yield expr
        for arg in expr.args:
            yield from _walk_expr_calls(arg)
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                yield from _walk_expr_calls(value)
        return
    if isinstance(expr, AxonExprBinary):
        yield from _walk_expr_calls(expr.left)
        yield from _walk_expr_calls(expr.right)
        return
    if isinstance(expr, AxonExprBind):
        yield from _walk_expr_calls(expr.value)
        yield from _walk_expr_calls(expr.body)
        return
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        yield from _walk_expr_calls(expr.cond)
        yield from _walk_expr_calls(expr.true_expr)
        yield from _walk_expr_calls(expr.false_expr)
        return
    if isinstance(expr, AxonExprLambda):
        yield from _walk_expr_calls(expr.body)
        return
    if isinstance(expr, AxonExprAscribe):
        yield from _walk_expr_calls(expr.expr)
        return
    if isinstance(expr, AxonExprList | AxonExprTuple):
        for item in expr.items:
            yield from _walk_expr_calls(item)
        return
    if isinstance(expr, AxonExprParen):
        yield from _walk_expr_calls(expr.inner)
        return
    if isinstance(expr, AxonExprDo):
        yield from _walk_stmt_calls(expr.body)


def _walk_stmt_calls(statements: tuple[AxonStatement, ...]):
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            yield from _walk_expr_calls(stmt.expr)
        elif isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                yield from _walk_expr_calls(value)
        elif isinstance(stmt, AxonCond):
            yield from _walk_expr_calls(stmt.cond)
            yield from _walk_stmt_calls(stmt.true_body)
            yield from _walk_stmt_calls(stmt.false_body)
        elif isinstance(stmt, AxonRepeat):
            yield from _walk_expr_calls(stmt.from_expr)
            yield from _walk_expr_calls(stmt.to_expr)
            yield from _walk_expr_calls(stmt.step_expr)
            yield from _walk_stmt_calls(stmt.body)
        elif isinstance(stmt, AxonScopeBind):
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    yield from _walk_expr_calls(raw_value)
            yield from _walk_stmt_calls(stmt.body)


def _walk_exprs(expr: AxonExpr):
    yield expr
    if isinstance(expr, AxonExprBinary):
        yield from _walk_exprs(expr.left)
        yield from _walk_exprs(expr.right)
        return
    if isinstance(expr, AxonExprBind):
        yield from _walk_exprs(expr.value)
        yield from _walk_exprs(expr.body)
        return
    if isinstance(expr, AxonExprCall):
        for arg in expr.args:
            yield from _walk_exprs(arg)
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                yield from _walk_exprs(value)
        return
    if isinstance(expr, AxonExprDo):
        yield from _walk_stmt_exprs(expr.body)
        return
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        yield from _walk_exprs(expr.cond)
        yield from _walk_exprs(expr.true_expr)
        yield from _walk_exprs(expr.false_expr)
        return
    if isinstance(expr, AxonExprLambda):
        yield from _walk_exprs(expr.body)
        return
    if isinstance(expr, AxonExprAscribe):
        yield from _walk_exprs(expr.expr)
        return
    if isinstance(expr, AxonExprList | AxonExprTuple):
        for item in expr.items:
            yield from _walk_exprs(item)
        return
    if isinstance(expr, AxonExprParen):
        yield from _walk_exprs(expr.inner)
        return
    if isinstance(expr, AxonExprPipe):
        yield from _walk_exprs(expr.value)
        for item in expr.stages:
            yield from _walk_exprs(item)


def _walk_stmt_exprs(statements: tuple[AxonStatement, ...]):
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            yield from _walk_exprs(stmt.expr)
        elif isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                yield from _walk_exprs(value)
        elif isinstance(stmt, AxonCond):
            yield from _walk_exprs(stmt.cond)
            yield from _walk_stmt_exprs(stmt.true_body)
            yield from _walk_stmt_exprs(stmt.false_body)
        elif isinstance(stmt, AxonRepeat):
            yield from _walk_exprs(stmt.from_expr)
            yield from _walk_exprs(stmt.to_expr)
            yield from _walk_exprs(stmt.step_expr)
            yield from _walk_stmt_exprs(stmt.body)
        elif isinstance(stmt, AxonScopeBind):
            yield stmt.prefix
            for raw_value in stmt.kwargs.values():
                if isinstance(raw_value, AxonExpr):
                    yield from _walk_exprs(raw_value)
            yield from _walk_stmt_exprs(stmt.body)


def _type_dim_names(tp: TypeExpr | None) -> set[str]:
    if tp is None:
        return set()
    if isinstance(tp, TypeOptional):
        return _type_dim_names(tp.inner)
    if isinstance(tp, TypeList):
        return _type_dim_names(tp.item)
    if isinstance(tp, TypeTuple):
        names: set[str] = set()
        for item in tp.items:
            names.update(_type_dim_names(item))
        return names
    if isinstance(tp, TypeTensor):
        tensor_names: set[str] = set()
        for dim in tp.dims:
            tensor_names.update(dim_token_names(dim))
        return tensor_names
    if isinstance(tp, TypeNamed):
        named_names: set[str] = set()
        for dim in tp.args:
            named_names.update(dim_token_names(dim))
        return named_names
    return set()


def _module_dim_names(module: AxonDefinition) -> set[str]:
    names: set[str] = set()
    for param in module.params:
        names.update(_type_dim_names(param.type_expr))
    names.update(_type_dim_names(module.return_type_expr))
    for stmt in module.statements:
        names.update(_stmt_dim_names(stmt))
    for constraint in module.constraints or ():
        names.update(_constraint_dim_names(constraint))
    return names


def _module_signature_dim_names(module: AxonDefinition) -> set[str]:
    names: set[str] = set()
    for param in module.params:
        names.update(_type_dim_names(param.type_expr))
    names.update(_type_dim_names(module.return_type_expr))
    for constraint in module.constraints or ():
        names.update(_constraint_dim_names(constraint))
    return names


def _module_body_term_dim_names(module: AxonDefinition) -> set[str]:
    local_names = set(_param_names(module))
    local_names.update(_bound_names_statements(module.statements))
    names = {
        name
        for name in _stmt_names(module.statements)
        if name not in local_names
    }
    for expr in _walk_stmt_exprs(module.statements):
        token = _dim_token_from_expr(expr)
        if isinstance(token, str) and token not in local_names:
            names.add(token)
    return names


def _module_dim_provenance(
    module: AxonDefinition, *, source_dim_names: frozenset[str] = frozenset()
) -> _DimProvenance:
    signature = _module_signature_dim_names(module)
    body_terms = _module_body_term_dim_names(module)
    all_dims = _module_dim_names(module)
    synthetic_signature = (
        frozenset(signature - body_terms - set(source_dim_names))
        if _is_generated_helper(module.name)
        else frozenset()
    )
    return _DimProvenance(
        signature=frozenset(signature),
        body_terms=frozenset(body_terms),
        inferred=frozenset(all_dims - signature - body_terms),
        synthetic_signature=synthetic_signature,
        allow_external_signature_names=_is_generated_helper(module.name),
    )


def _stmt_dim_names(stmt: AxonStatement) -> set[str]:
    names: set[str] = set()
    if isinstance(stmt, AxonBind):
        names.update(_expr_dim_names(stmt.expr))
    elif isinstance(stmt, AxonReturn | AxonYield):
        for value in stmt.values:
            names.update(_expr_dim_names(value))
    elif isinstance(stmt, AxonCond):
        names.update(_expr_dim_names(stmt.cond))
        for inner in stmt.true_body:
            names.update(_stmt_dim_names(inner))
        for inner in stmt.false_body:
            names.update(_stmt_dim_names(inner))
    elif isinstance(stmt, AxonRepeat):
        names.update(_expr_dim_names(stmt.from_expr))
        names.update(_expr_dim_names(stmt.to_expr))
        names.update(_expr_dim_names(stmt.step_expr))
        for inner in stmt.body:
            names.update(_stmt_dim_names(inner))
    elif isinstance(stmt, AxonScopeBind):
        names.update(_expr_dim_names(stmt.prefix))
        for raw in stmt.kwargs.values():
            if isinstance(raw, AxonExpr):
                names.update(_expr_dim_names(raw))
        for inner in stmt.body:
            names.update(_stmt_dim_names(inner))
    return names


def _expr_dim_names(expr: AxonExpr) -> set[str]:
    names: set[str] = set()
    if expr.inferred_type is not None:
        names.update(_type_dim_names(expr.inferred_type))
    if expr.inferred_dims is not None:
        for dim in expr.inferred_dims:
            names.update(dim_token_names(dim))
    if isinstance(expr, AxonExprBinary):
        names.update(_expr_dim_names(expr.left))
        names.update(_expr_dim_names(expr.right))
    elif isinstance(expr, AxonExprBind):
        names.update(_expr_dim_names(expr.value))
        names.update(_expr_dim_names(expr.body))
    elif isinstance(expr, AxonExprCall):
        for arg in expr.args:
            names.update(_expr_dim_names(arg))
        for raw in expr.kwargs.values():
            if isinstance(raw, AxonExpr):
                names.update(_expr_dim_names(raw))
    elif isinstance(expr, AxonExprDo):
        for stmt in expr.body:
            names.update(_stmt_dim_names(stmt))
    elif isinstance(expr, AxonExprIf | AxonExprTernary):
        names.update(_expr_dim_names(expr.cond))
        names.update(_expr_dim_names(expr.true_expr))
        names.update(_expr_dim_names(expr.false_expr))
    elif isinstance(expr, AxonExprLambda):
        names.update(_expr_dim_names(expr.body))
    elif isinstance(expr, AxonExprAscribe):
        names.update(_type_dim_names(expr.type_expr))
        names.update(_expr_dim_names(expr.expr))
    elif isinstance(expr, AxonExprList | AxonExprTuple):
        for item in expr.items:
            names.update(_expr_dim_names(item))
    elif isinstance(expr, AxonExprParen):
        names.update(_expr_dim_names(expr.inner))
    elif isinstance(expr, AxonExprPipe):
        names.update(_expr_dim_names(expr.value))
        for item in expr.stages:
            names.update(_expr_dim_names(item))
    return names


def _expr_inferred_dim_term_names(expr: AxonExpr, inferred_names: frozenset[str]) -> set[str]:
    expr = _unwrap_dim_expr(expr)
    if isinstance(expr, AxonExprName):
        return {expr.name} if expr.name in inferred_names else set()
    if isinstance(expr, AxonExprBinary):
        return _expr_inferred_dim_term_names(
            expr.left, inferred_names
        ) | _expr_inferred_dim_term_names(expr.right, inferred_names)
    if isinstance(expr, AxonExprBind):
        return _expr_inferred_dim_term_names(
            expr.value, inferred_names
        ) | _expr_inferred_dim_term_names(expr.body, inferred_names)
    if isinstance(expr, AxonExprCall):
        names: set[str] = set()
        for arg in expr.args:
            names.update(_expr_inferred_dim_term_names(arg, inferred_names))
        for value in expr.kwargs.values():
            if isinstance(value, AxonExpr):
                names.update(_expr_inferred_dim_term_names(value, inferred_names))
        return names
    if isinstance(expr, AxonExprDo):
        return _stmt_inferred_dim_term_names(expr.body, inferred_names)
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return (
            _expr_inferred_dim_term_names(expr.cond, inferred_names)
            | _expr_inferred_dim_term_names(expr.true_expr, inferred_names)
            | _expr_inferred_dim_term_names(expr.false_expr, inferred_names)
        )
    if isinstance(expr, AxonExprLambda):
        return _expr_inferred_dim_term_names(expr.body, inferred_names)
    if isinstance(expr, AxonExprList | AxonExprTuple):
        item_names: set[str] = set()
        for item in expr.items:
            item_names.update(_expr_inferred_dim_term_names(item, inferred_names))
        return item_names
    if isinstance(expr, AxonExprPipe):
        names = _expr_inferred_dim_term_names(expr.value, inferred_names)
        for stage in expr.stages:
            names.update(_expr_inferred_dim_term_names(stage, inferred_names))
        return names
    return set()


def _stmt_inferred_dim_term_names(
    statements: tuple[AxonStatement, ...], inferred_names: frozenset[str]
) -> set[str]:
    names: set[str] = set()
    for stmt in statements:
        if isinstance(stmt, AxonBind):
            names.update(_expr_inferred_dim_term_names(stmt.expr, inferred_names))
        elif isinstance(stmt, AxonReturn | AxonYield):
            for value in stmt.values:
                names.update(_expr_inferred_dim_term_names(value, inferred_names))
        elif isinstance(stmt, AxonCond):
            names.update(_expr_inferred_dim_term_names(stmt.cond, inferred_names))
            names.update(_stmt_inferred_dim_term_names(stmt.true_body, inferred_names))
            names.update(_stmt_inferred_dim_term_names(stmt.false_body, inferred_names))
        elif isinstance(stmt, AxonRepeat):
            names.update(_expr_inferred_dim_term_names(stmt.from_expr, inferred_names))
            names.update(_expr_inferred_dim_term_names(stmt.to_expr, inferred_names))
            names.update(_expr_inferred_dim_term_names(stmt.step_expr, inferred_names))
            names.update(_stmt_inferred_dim_term_names(stmt.body, inferred_names))
        elif isinstance(stmt, AxonScopeBind):
            names.update(_expr_inferred_dim_term_names(stmt.prefix, inferred_names))
            for raw in stmt.kwargs.values():
                if isinstance(raw, AxonExpr):
                    names.update(_expr_inferred_dim_term_names(raw, inferred_names))
            names.update(_stmt_inferred_dim_term_names(stmt.body, inferred_names))
    return names


def _assert_no_free_generated_dim_terms(program: AxonFile) -> None:
    for module in program.modules:
        provenance = _module_dim_provenance(module)
        allowed = set(provenance.signature)
        for constraint in module.constraints or ():
            allowed.update(_constraint_dim_names(constraint))
        free = sorted(
            _stmt_inferred_dim_term_names(
                module.statements,
                provenance.inferred | provenance.synthetic_signature,
            )
            - allowed,
            key=_dim_score,
        )
        if free:
            joined = ", ".join(free)
            raise ValueError(f"module {module.name}: inferred dim term is not bound by signature or constraints: {joined}")


def _constraint_dim_names(constraint: Constraint) -> set[str]:
    names: set[str] = set()
    names.update(_constraint_operand_dim_names(constraint.left))
    if constraint.right is not None:
        names.update(_constraint_operand_dim_names(constraint.right))
    for guard in constraint.guards:
        names.update(_constraint_dim_names(guard))
    return names


def _constraint_operand_dim_names(operand: object) -> set[str]:
    if isinstance(operand, str):
        return {operand}
    if isinstance(operand, DimExprBinary):
        return dim_token_names(operand)
    if isinstance(operand, tuple):
        names: set[str] = set()
        for item in operand:
            names.update(_constraint_operand_dim_names(item))
        return names
    return set()


def _collect_module_callsites(program: AxonFile) -> dict[str, list[AxonExprCall]]:
    module_names = {module.name for module in program.modules}
    out: dict[str, list[AxonExprCall]] = {module.name: [] for module in program.modules}
    for module in program.modules:
        for call in _walk_stmt_calls(module.statements):
            if call.callee in module_names:
                out[call.callee].append(call)
    return out


def _dim_is_inferred(name: str, provenance: _DimProvenance) -> bool:
    return name in provenance.inferred or name in provenance.synthetic_signature


def _dim_is_source(name: str, provenance: _DimProvenance) -> bool:
    return name in provenance.signature or name in provenance.body_terms


def _record_dim_candidate(
    left: object,
    right: object,
    out: dict[str, set[str]],
    provenance: _DimProvenance,
) -> None:
    if not isinstance(left, str) or not isinstance(right, str) or left == right:
        return
    left_variadic, _left_base = _split_variadic_dim(left)
    right_variadic, _right_base = _split_variadic_dim(right)
    if left_variadic != right_variadic:
        return
    if _dim_is_inferred(left, provenance) and _dim_is_source(right, provenance):
        out.setdefault(left, set()).add(right)
        return
    if _dim_is_inferred(right, provenance) and _dim_is_source(left, provenance):
        out.setdefault(right, set()).add(left)
        return
    if left in provenance.signature and right in provenance.body_terms:
        out.setdefault(left, set()).add(right)
        return
    if right in provenance.signature and left in provenance.body_terms:
        out.setdefault(right, set()).add(left)
        return
    if (
        provenance.allow_external_signature_names
        and
        left in provenance.signature
        and right not in provenance.signature
        and not _dim_is_inferred(right, provenance)
    ):
        out.setdefault(left, set()).add(right)
        return
    if (
        provenance.allow_external_signature_names
        and
        right in provenance.signature
        and left not in provenance.signature
        and not _dim_is_inferred(left, provenance)
    ):
        out.setdefault(right, set()).add(left)


def _match_type_dims(
    formal: TypeExpr | None,
    actual: TypeExpr | None,
    out: dict[str, set[str]],
    provenance: _DimProvenance,
) -> None:
    if formal is None or actual is None:
        return
    if isinstance(formal, TypeOptional) and isinstance(actual, TypeOptional):
        _match_type_dims(formal.inner, actual.inner, out, provenance)
        return
    if isinstance(formal, TypeOptional):
        _match_type_dims(formal.inner, actual, out, provenance)
        return
    if isinstance(actual, TypeOptional):
        _match_type_dims(formal, actual.inner, out, provenance)
        return
    if isinstance(formal, TypeList) and isinstance(actual, TypeList):
        _match_type_dims(formal.item, actual.item, out, provenance)
        return
    if isinstance(formal, TypeTuple) and isinstance(actual, TypeTuple):
        for left, right in zip(formal.items, actual.items, strict=False):
            _match_type_dims(left, right, out, provenance)
        return
    if isinstance(formal, TypeTensor) and isinstance(actual, TypeTensor):
        for formal_dim, actual_dim in zip(formal.dims, actual.dims, strict=False):
            _match_dim_tokens(formal_dim, actual_dim, out, provenance)
        return
    if isinstance(formal, TypeNamed) and isinstance(actual, TypeNamed) and formal.name == actual.name:
        for formal_dim, actual_dim in zip(formal.args, actual.args, strict=False):
            _match_dim_tokens(formal_dim, actual_dim, out, provenance)


def _match_dim_tokens(
    formal: object,
    actual: object,
    out: dict[str, set[str]],
    provenance: _DimProvenance,
) -> None:
    if isinstance(formal, str):
        _record_dim_candidate(formal, actual, out, provenance)
        return
    if isinstance(formal, DimExprBinary) and isinstance(actual, DimExprBinary) and formal.op == actual.op:
        _match_dim_tokens(formal.left, actual.left, out, provenance)
        _match_dim_tokens(formal.right, actual.right, out, provenance)


def _unwrap_dim_expr(expr: AxonExpr) -> AxonExpr:
    while isinstance(expr, AxonExprAscribe | AxonExprParen):
        expr = expr.expr if isinstance(expr, AxonExprAscribe) else expr.inner
    return expr


def _match_dim_expr_to_token(
    expr: AxonExpr,
    actual: object,
    out: dict[str, set[str]],
    provenance: _DimProvenance,
) -> None:
    expr = _unwrap_dim_expr(expr)
    if isinstance(expr, AxonExprName):
        _match_dim_tokens(expr.name, actual, out, provenance)
        return
    if isinstance(expr, AxonExprInt):
        _match_dim_tokens(expr.value, actual, out, provenance)
        return
    if isinstance(expr, AxonExprBinary) and isinstance(actual, DimExprBinary) and expr.op == actual.op:
        _match_dim_expr_to_token(expr.left, actual.left, out, provenance)
        _match_dim_expr_to_token(expr.right, actual.right, out, provenance)


def _is_dim_typed_expr(expr: AxonExpr) -> bool:
    return isinstance(expr.inferred_type, TypeDim | TypeInt)


def _dim_token_from_expr(expr: AxonExpr) -> object | None:
    expr = _unwrap_dim_expr(expr)
    if isinstance(expr, AxonExprName):
        return expr.name
    if isinstance(expr, AxonExprInt):
        return expr.value
    if isinstance(expr, AxonExprCall) and not expr.args and not expr.kwargs and _is_dim_typed_expr(expr):
        return expr.callee
    if isinstance(expr, AxonExprBinary):
        left = _dim_token_from_expr(expr.left)
        right = _dim_token_from_expr(expr.right)
        if left is not None and right is not None:
            return DimExprBinary(op=expr.op, left=left, right=right)
    return None


def _substitute_dim_token(dim: object, subst: Mapping[str, object]) -> object:
    if isinstance(dim, str):
        return subst.get(dim, dim)
    if isinstance(dim, DimExprBinary):
        return DimExprBinary(
            op=dim.op,
            left=_substitute_dim_token(dim.left, subst),
            right=_substitute_dim_token(dim.right, subst),
        )
    return dim


def _single_bind_exprs(statements: tuple[AxonStatement, ...]) -> dict[str, AxonExpr]:
    out: dict[str, AxonExpr] = {}
    for stmt in statements:
        if isinstance(stmt, AxonBind) and len(stmt.targets) == 1:
            out[stmt.targets[0]] = stmt.expr
        elif isinstance(stmt, AxonCond):
            out.update(_single_bind_exprs(stmt.true_body))
            out.update(_single_bind_exprs(stmt.false_body))
        elif isinstance(stmt, AxonRepeat | AxonScopeBind):
            out.update(_single_bind_exprs(stmt.body))
    return out


def _resolve_bound_expr(expr: AxonExpr, binds: Mapping[str, AxonExpr]) -> AxonExpr:
    expr = _unwrap_dim_expr(expr)
    if isinstance(expr, AxonExprName):
        return binds.get(expr.name, expr)
    return expr


def _shape_expr_from_call(call: AxonExprCall, binds: Mapping[str, AxonExpr]) -> AxonExpr | None:
    if call.callee not in {"_reshape", "Tensor.reshape"}:
        return None
    raw: AxonExpr | None = None
    if len(call.args) >= 2:
        raw = call.args[1]
    shape_kw = call.kwargs.get("shape")
    if isinstance(shape_kw, AxonExpr):
        raw = shape_kw
    if raw is None:
        return None
    return _resolve_bound_expr(raw, binds)


def _collect_shape_dim_candidates(
    module: AxonDefinition,
    out: dict[str, set[str]],
    provenance: _DimProvenance,
) -> None:
    binds = _single_bind_exprs(module.statements)
    for expr in _walk_stmt_exprs(module.statements):
        if not isinstance(expr, AxonExprCall) or not isinstance(expr.inferred_type, TypeTensor):
            continue
        shape_expr = _shape_expr_from_call(expr, binds)
        if shape_expr is None:
            continue
        shape_expr = _unwrap_dim_expr(shape_expr)
        if not isinstance(shape_expr, AxonExprList | AxonExprTuple):
            continue
        local: dict[str, set[str]] = {}
        for item, actual in zip(shape_expr.items, expr.inferred_type.dims, strict=False):
            _match_dim_expr_to_token(item, actual, local, provenance)
        for inferred, actuals in local.items():
            out.setdefault(inferred, set()).update(actuals)


def _param_accepts_dim_value(param: AxonParam) -> bool:
    tp = param.type_expr
    return isinstance(tp, TypeDim) or (
        isinstance(tp, TypeOptional) and isinstance(tp.inner, TypeDim)
    )


def _call_actual_by_param(callee: AxonDefinition, call: AxonExprCall) -> dict[str, AxonExpr]:
    actuals: dict[str, AxonExpr] = {}
    for param, arg in zip(callee.params, call.args, strict=False):
        actuals[param.name] = arg
    for key, value in call.kwargs.items():
        if isinstance(value, AxonExpr):
            actuals[key] = value
    return actuals


def _collect_call_dim_param_candidates(
    *,
    caller: AxonDefinition,
    call: AxonExprCall,
    callee: AxonDefinition,
    out: dict[str, set[str]],
    provenance: _DimProvenance,
) -> None:
    binds = _single_bind_exprs(caller.statements)
    actuals = _call_actual_by_param(callee, call)
    dim_actuals: dict[str, object] = {}
    for param in callee.params:
        if not _param_accepts_dim_value(param):
            continue
        raw_actual = actuals.get(param.name)
        if raw_actual is None:
            continue
        token = _dim_token_from_expr(_resolve_bound_expr(raw_actual, binds))
        if token is not None:
            dim_actuals[param.name] = token
    if not dim_actuals:
        return
    _match_call_return_dims(
        callee.return_type_expr,
        call.inferred_type,
        dim_actuals=dim_actuals,
        out=out,
        provenance=provenance,
    )


def _match_call_return_dims(
    formal: TypeExpr | None,
    actual: TypeExpr | None,
    *,
    dim_actuals: Mapping[str, object],
    out: dict[str, set[str]],
    provenance: _DimProvenance,
) -> None:
    if formal is None or actual is None:
        return
    if isinstance(formal, TypeOptional) and isinstance(actual, TypeOptional):
        _match_call_return_dims(
            formal.inner,
            actual.inner,
            dim_actuals=dim_actuals,
            out=out,
            provenance=provenance,
        )
        return
    if isinstance(formal, TypeOptional):
        _match_call_return_dims(
            formal.inner,
            actual,
            dim_actuals=dim_actuals,
            out=out,
            provenance=provenance,
        )
        return
    if isinstance(actual, TypeOptional):
        _match_call_return_dims(
            formal,
            actual.inner,
            dim_actuals=dim_actuals,
            out=out,
            provenance=provenance,
        )
        return
    if isinstance(formal, TypeList) and isinstance(actual, TypeList):
        _match_call_return_dims(
            formal.item,
            actual.item,
            dim_actuals=dim_actuals,
            out=out,
            provenance=provenance,
        )
        return
    if isinstance(formal, TypeTuple) and isinstance(actual, TypeTuple):
        for formal_item, actual_item in zip(formal.items, actual.items, strict=False):
            _match_call_return_dims(
                formal_item,
                actual_item,
                dim_actuals=dim_actuals,
                out=out,
                provenance=provenance,
            )
        return
    if isinstance(formal, TypeTensor) and isinstance(actual, TypeTensor):
        for formal_dim, actual_dim in _aligned_dim_pairs(formal.dims, actual.dims):
            substituted = _substitute_dim_token(formal_dim, dim_actuals)
            _match_dim_tokens(actual_dim, substituted, out, provenance)
        return
    if isinstance(formal, TypeNamed) and isinstance(actual, TypeNamed) and formal.name == actual.name:
        for formal_dim, actual_dim in _aligned_dim_pairs(formal.args, actual.args):
            substituted = _substitute_dim_token(formal_dim, dim_actuals)
            _match_dim_tokens(actual_dim, substituted, out, provenance)


def _aligned_dim_pairs(
    formal: tuple[object, ...],
    actual: tuple[object, ...],
) -> tuple[tuple[object, object], ...]:
    variadic_idx = next(
        (idx for idx, dim in enumerate(formal) if isinstance(dim, str) and dim.startswith("..")),
        None,
    )
    if variadic_idx is None:
        return tuple(zip(formal, actual, strict=False))
    prefix = formal[:variadic_idx]
    suffix = formal[variadic_idx + 1 :]
    if len(actual) < len(prefix) + len(suffix):
        return ()
    pairs: list[tuple[object, object]] = list(zip(prefix, actual[: len(prefix)], strict=False))
    if suffix:
        pairs.extend(zip(suffix, actual[-len(suffix) :], strict=False))
    return tuple(pairs)


def _preferred_module_dim_renames(program: AxonFile) -> dict[str, dict[str, str]]:
    callsites = _collect_module_callsites(program)
    modules_by_name = {module.name: module for module in program.modules}
    source_dim_names: set[str] = set()
    for module in program.modules:
        if _is_generated_helper(module.name):
            continue
        source_dim_names.update(_module_signature_dim_names(module))
        source_dim_names.update(_module_body_term_dim_names(module))
    out: dict[str, dict[str, str]] = {}
    for module in program.modules:
        provenance = _module_dim_provenance(
            module, source_dim_names=frozenset(source_dim_names)
        )
        candidates: dict[str, set[str]] = {}
        exprs = list(_walk_stmt_exprs(module.statements))
        module_calls = callsites.get(module.name, ())
        for expr in exprs:
            if isinstance(expr, AxonExprCall):
                callee = modules_by_name.get(expr.callee)
                if callee is not None:
                    _collect_call_dim_param_candidates(
                        caller=module,
                        call=expr,
                        callee=callee,
                        out=candidates,
                        provenance=provenance,
                    )
            if isinstance(expr, AxonExprName):
                param = next((item for item in module.params if item.name == expr.name), None)
                if param is not None:
                    name_match: dict[str, set[str]] = {}
                    _match_type_dims(param.type_expr, expr.inferred_type, name_match, provenance)
                    for inferred, actuals in name_match.items():
                        candidates.setdefault(inferred, set()).update(actuals)
        for stmt in module.statements:
            if not isinstance(stmt, AxonReturn):
                continue
            if module.return_type_expr is None:
                continue
            if len(stmt.values) == 1:
                return_local: dict[str, set[str]] = {}
                _match_type_dims(
                    module.return_type_expr,
                    stmt.values[0].inferred_type,
                    return_local,
                    provenance,
                )
                for inferred, actuals in return_local.items():
                    candidates.setdefault(inferred, set()).update(actuals)
        for call in module_calls:
            local: dict[str, set[str]] = {}
            for param, arg in zip(module.params, call.args, strict=False):
                _match_type_dims(param.type_expr, arg.inferred_type, local, provenance)
            for key, raw in call.kwargs.items():
                if not isinstance(raw, AxonExpr):
                    continue
                kw_param = next((item for item in module.params if item.name == key), None)
                if kw_param is None:
                    continue
                _match_type_dims(kw_param.type_expr, raw.inferred_type, local, provenance)
            _match_type_dims(module.return_type_expr, call.inferred_type, local, provenance)
            for inferred, actuals in local.items():
                candidates.setdefault(inferred, set()).update(actuals)
        _collect_shape_dim_candidates(module, candidates, provenance)
        if not candidates:
            continue
        used_in_module = _module_dim_names(module)
        used_targets: set[str] = set(name for name in used_in_module if not _dim_is_inferred(name, provenance))
        renames: dict[str, str] = {}
        for generated in sorted(candidates, key=_dim_score):
            options = sorted(candidates[generated], key=_dim_score)
            chosen: str | None = None
            for candidate in options:
                if _dim_is_inferred(candidate, provenance):
                    continue
                if candidate not in source_dim_names and candidate not in provenance.body_terms:
                    continue
                if not _dim_is_inferred(generated, provenance) and _dim_score(candidate) >= _dim_score(generated):
                    continue
                existing = next((old for old, new in renames.items() if new == candidate), None)
                if existing is not None and existing != generated:
                    continue
                if candidate in used_targets and candidate not in used_in_module:
                    continue
                chosen = candidate
                break
            if chosen is None:
                continue
            renames[generated] = chosen
            used_targets.add(chosen)
        if renames:
            out[module.name] = renames
    return out


def _rename_dim(dim: object, renames: Mapping[str, str]):
    if isinstance(dim, str):
        return renames.get(dim, dim)
    if isinstance(dim, DimExprBinary):
        return DimExprBinary(
            op=dim.op,
            left=_rename_dim(dim.left, renames),
            right=_rename_dim(dim.right, renames),
        )
    return dim


def _rename_type_dims(tp: TypeExpr | None, renames: Mapping[str, str]) -> TypeExpr | None:
    if tp is None:
        return None
    if isinstance(tp, (TypeAny, TypeInt, TypeFloat, TypeBool, TypeNull, TypeString, TypePath, TypeDim)):
        return tp
    if isinstance(tp, TypeVar):
        return tp
    if isinstance(tp, TypeOptional):
        inner = _rename_type_dims(tp.inner, renames)
        assert inner is not None
        return TypeOptional(inner=inner)
    if isinstance(tp, TypeList):
        item = _rename_type_dims(tp.item, renames)
        assert item is not None
        return TypeList(item=item)
    if isinstance(tp, TypeTuple):
        items = tuple(_rename_type_dims(item, renames) for item in tp.items)
        assert all(item is not None for item in items)
        return TypeTuple(items=items)  # type: ignore[arg-type]
    if isinstance(tp, TypeTensor):
        return TypeTensor(base=tp.base, dims=tuple(_rename_dim(dim, renames) for dim in tp.dims))
    if isinstance(tp, TypeNamed):
        return TypeNamed(name=tp.name, args=tuple(_rename_dim(dim, renames) for dim in tp.args))
    return tp


def _rename_constraint_dims(constraint: Constraint, renames: Mapping[str, str]) -> Constraint:
    return Constraint(
        relation=constraint.relation,
        left=_rename_constraint_operand_dims(constraint.left, renames),
        right=_rename_constraint_operand_dims(constraint.right, renames) if constraint.right is not None else None,
        guards=tuple(_rename_constraint_dims(guard, renames) for guard in constraint.guards),
    )


def _rename_constraint_operand_dims(operand: object, renames: Mapping[str, str]):
    if isinstance(operand, str):
        return renames.get(operand, operand)
    if isinstance(operand, DimExprBinary):
        return _rename_dim(operand, renames)
    if isinstance(operand, tuple):
        return tuple(_rename_constraint_operand_dims(item, renames) for item in operand)
    return operand


def _rename_expr_dims(expr: AxonExpr, renames: Mapping[str, str]) -> AxonExpr:
    inferred_type = _rename_type_dims(expr.inferred_type, renames)
    inferred_dims = (
        tuple(_rename_dim(dim, renames) for dim in expr.inferred_dims)
        if expr.inferred_dims is not None
        else None
    )
    expr = replace(expr, inferred_type=inferred_type, inferred_dims=inferred_dims)
    if isinstance(expr, AxonExprName) and expr.name in renames:
        return replace(expr, name=renames.get(expr.name, expr.name))
    if isinstance(expr, AxonExprBinary):
        return replace(expr, left=_rename_expr_dims(expr.left, renames), right=_rename_expr_dims(expr.right, renames))
    if isinstance(expr, AxonExprBind):
        return replace(expr, value=_rename_expr_dims(expr.value, renames), body=_rename_expr_dims(expr.body, renames))
    if isinstance(expr, AxonExprCall):
        return replace(
            expr,
            args=tuple(_rename_expr_dims(arg, renames) for arg in expr.args),
            kwargs={k: _rename_expr_dims(v, renames) if isinstance(v, AxonExpr) else v for k, v in expr.kwargs.items()},
        )
    if isinstance(expr, AxonExprDo):
        return replace(expr, body=tuple(_rename_stmt_dims(stmt, renames) for stmt in expr.body))
    if isinstance(expr, AxonExprIf | AxonExprTernary):
        return replace(
            expr,
            cond=_rename_expr_dims(expr.cond, renames),
            true_expr=_rename_expr_dims(expr.true_expr, renames),
            false_expr=_rename_expr_dims(expr.false_expr, renames),
        )
    if isinstance(expr, AxonExprLambda):
        return replace(expr, body=_rename_expr_dims(expr.body, renames))
    if isinstance(expr, AxonExprAscribe):
        return replace(expr, expr=_rename_expr_dims(expr.expr, renames), type_expr=_rename_type_dims(expr.type_expr, renames))  # type: ignore[arg-type]
    if isinstance(expr, AxonExprList | AxonExprTuple):
        return replace(expr, items=tuple(_rename_expr_dims(item, renames) for item in expr.items))
    if isinstance(expr, AxonExprParen):
        return replace(expr, inner=_rename_expr_dims(expr.inner, renames))
    if isinstance(expr, AxonExprPipe):
        return replace(
            expr,
            value=_rename_expr_dims(expr.value, renames),
            stages=tuple(_rename_expr_dims(item, renames) for item in expr.stages),
        )
    return expr


def _rename_stmt_dims(stmt: AxonStatement, renames: Mapping[str, str]) -> AxonStatement:
    if isinstance(stmt, AxonBind):
        return replace(stmt, expr=_rename_expr_dims(stmt.expr, renames))
    if isinstance(stmt, AxonReturn | AxonYield):
        return replace(stmt, values=tuple(_rename_expr_dims(value, renames) for value in stmt.values))
    if isinstance(stmt, AxonCond):
        return replace(
            stmt,
            cond=_rename_expr_dims(stmt.cond, renames),
            true_body=tuple(_rename_stmt_dims(item, renames) for item in stmt.true_body),
            false_body=tuple(_rename_stmt_dims(item, renames) for item in stmt.false_body),
        )
    if isinstance(stmt, AxonRepeat):
        return replace(
            stmt,
            from_expr=_rename_expr_dims(stmt.from_expr, renames),
            to_expr=_rename_expr_dims(stmt.to_expr, renames),
            step_expr=_rename_expr_dims(stmt.step_expr, renames),
            body=tuple(_rename_stmt_dims(item, renames) for item in stmt.body),
        )
    if isinstance(stmt, AxonScopeBind):
        return replace(
            stmt,
            body=tuple(_rename_stmt_dims(item, renames) for item in stmt.body),
            kwargs={k: _rename_expr_dims(v, renames) if isinstance(v, AxonExpr) else v for k, v in stmt.kwargs.items()},
        )
    return stmt


def _canonicalize_module_dims(module: AxonDefinition, renames: Mapping[str, str]) -> AxonDefinition:
    if not renames:
        return module
    return replace(
        module,
        params=tuple(replace(param, type_expr=_rename_type_dims(param.type_expr, renames)) for param in module.params),
        statements=tuple(_rename_stmt_dims(stmt, renames) for stmt in module.statements),
        return_type_expr=_rename_type_dims(module.return_type_expr, renames),
        constraints=tuple(_rename_constraint_dims(item, renames) for item in (module.constraints or ())),
    )


def canonicalize_typed_axon_file(program: AxonFile, *, main_module: str | None = None) -> AxonFile:
    validate_typed_axon_file(program, main_module=main_module)
    current = _canonicalize_path_params(program)
    current = replace(
        current,
        modules=tuple(_canonicalize_generated_local_names(module) for module in current.modules),
    )
    while True:
        dim_renames = _preferred_module_dim_renames(current)
        if not dim_renames:
            break
        rewritten = replace(
            current,
            modules=tuple(
                _canonicalize_module_dims(module, dim_renames.get(module.name, {}))
                for module in current.modules
            ),
        )
        if ast_equal(current, rewritten):
            break
        current = rewritten
    current = _canonicalize_generated_helper_names(current)
    _assert_no_free_generated_dim_terms(current)
    validate_typed_axon_file(current, main_module=main_module)
    return current


__all__ = ["canonicalize_typed_axon_file"]
